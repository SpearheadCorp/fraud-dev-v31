"""
Metrics collection (v4): telemetry (K8s pod logs), system (node-exporter), GPU (Prometheus/DCGM),
FlashBlade storage latency (REST API), queue depth, fraud scores.
"""
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

import urllib3
import pandas as pd
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from kubernetes import client, config
from kubernetes.client.rest import ApiException

log = logging.getLogger(__name__)

PROMETHEUS_URL    = os.environ.get("PROMETHEUS_URL",          "http://prometheus:9090")
MODEL_REPO        = Path(os.environ.get("MODEL_REPO_PATH",    "/data/models"))
RAW_PATH          = Path(os.environ.get("OUTPUT_PATH",         "/data/raw"))
FEATURES_PATH     = Path(os.environ.get("FEATURES_PATH",      "/data/features"))
SCORES_PATH       = Path(os.environ.get("SCORES_PATH",        "/data/scores"))
NAMESPACE         = os.environ.get("K8S_NAMESPACE",           "fraud-det-v31")
FLASHBLADE_FS_NAME = os.environ.get("FLASHBLADE_FS_NAME",    "financial-fraud-detection-demo")
FLASHBLADE_MGMT_IP = os.environ.get("FLASHBLADE_MGMT_IP",    "10.23.181.60")
FLASHBLADE_API_TOKEN = os.environ.get("FLASHBLADE_API_TOKEN", "")  # Set via K8s Secret or env var
NFS_NODE_INSTANCE  = os.environ.get("NFS_NODE_INSTANCE",      "10.23.181.44:9100")

_fb_session_token: Optional[str] = None


def _fb_login() -> Optional[str]:
    """Authenticate to FlashBlade REST API and return session token."""
    try:
        resp = requests.post(
            f"https://{FLASHBLADE_MGMT_IP}/api/login",
            headers={"api-token": FLASHBLADE_API_TOKEN},
            verify=False, timeout=5,
        )
        if resp.status_code == 200:
            token = resp.headers.get("X-Auth-Token")
            log.info("FlashBlade API login OK")
            return token
    except Exception as exc:
        log.debug("_fb_login: %s", exc)
    return None


def _core_v1() -> client.CoreV1Api:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CoreV1Api()


class PipelineState:
    """Mutable state shared between HTTP endpoints and the metrics collector."""

    def __init__(self) -> None:
        self.is_running: bool = False
        self.start_time: Optional[float] = None
        self.last_telemetry: dict = {}
        self.total_rows_processed: int = 0
        self._last_prep_chunk_id: int = -1

    def reset(self) -> None:
        self.is_running = False
        self.start_time = None
        self.last_telemetry = {}
        self.total_rows_processed = 0
        self._last_prep_chunk_id = -1

    @property
    def elapsed_sec(self) -> int:
        if self.start_time is None:
            return 0
        return int(time.time() - self.start_time)


_TELEMETRY_CACHE = MODEL_REPO / "last_telemetry.json"


class MetricsCollector:
    def __init__(self, state: PipelineState) -> None:
        self.state = state
        self._load_telemetry_cache()

    def _load_telemetry_cache(self) -> None:
        try:
            if _TELEMETRY_CACHE.exists():
                self.state.last_telemetry = json.loads(_TELEMETRY_CACHE.read_text())
                log.info("Loaded telemetry cache (%d stages)", len(self.state.last_telemetry))
        except Exception as exc:
            log.debug("load_telemetry_cache: %s", exc)

    def _save_telemetry_cache(self) -> None:
        try:
            _TELEMETRY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _TELEMETRY_CACHE.write_text(json.dumps(self.state.last_telemetry))
        except Exception as exc:
            log.debug("save_telemetry_cache: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self) -> dict:
        # Skip pod-log telemetry when pipeline is stopped — stale log lines
        # would otherwise repopulate KPIs from the last run's data.
        telemetry  = self._parse_telemetry() if self.state.is_running else {}
        system     = self._collect_system()
        gpu        = self._collect_gpu()
        business   = self._compute_kpis(telemetry)
        storage    = self._collect_storage()
        flashblade = self._collect_flashblade()
        queue      = self._collect_queue_depth()
        fraud      = self._collect_fraud_metrics()

        if telemetry:
            self.state.last_telemetry = telemetry
            self._save_telemetry_cache()

        return {
            "is_running":  self.state.is_running,
            "elapsed_sec": self.state.elapsed_sec,
            "timestamp":   time.time(),
            "pipeline": {
                "gather":  telemetry.get("gather",  {}),
                "prep":    telemetry.get("prep",    {}),
                "scoring": telemetry.get("scoring", {}),
                "train":   telemetry.get("train",   {}),
            },
            "queue":     queue,
            "fraud":     fraud,
            "business":  business,
            "storage":   storage,
            "flashblade": flashblade,
            "system":    system,
            "gpu":       gpu,
        }

    def collect_fast(self) -> dict:
        """Light-weight collection: GPU, CPU, FlashBlade latency only.
        Called at 200ms intervals between full collect() cycles."""
        system     = self._collect_system()
        gpu        = self._collect_gpu()
        flashblade = self._collect_flashblade()
        return {
            "is_running":  self.state.is_running,
            "elapsed_sec": self.state.elapsed_sec,
            "timestamp":   time.time(),
            "system":    system,
            "gpu":       gpu,
            "flashblade": flashblade,
        }

    # ------------------------------------------------------------------
    # Telemetry from K8s pod logs
    # ------------------------------------------------------------------

    def _get_deployment_pod_logs(self, dep_name: str, tail: int = 200) -> str:
        """Get logs from the most-recent running pod of a Deployment."""
        try:
            core_v1 = _core_v1()
            pods = core_v1.list_namespaced_pod(
                namespace=NAMESPACE,
                label_selector=f"app={dep_name}",
            )
            # Prefer Running pods; fall back to any latest pod
            running = [p for p in pods.items
                       if p.status and p.status.phase == "Running"]
            pod_list = running if running else pods.items
            if not pod_list:
                return ""
            pod_name = pod_list[-1].metadata.name
            return core_v1.read_namespaced_pod_log(
                name=pod_name, namespace=NAMESPACE, tail_lines=tail,
            )
        except ApiException:
            return ""
        except Exception as exc:
            log.debug("_get_deployment_pod_logs %s: %s", dep_name, exc)
            return ""

    def _get_job_pod_logs(self, job_name: str, tail: int = 200) -> str:
        """Get logs from the pod created by a Job (e.g. model-build)."""
        try:
            core_v1 = _core_v1()
            pods = core_v1.list_namespaced_pod(
                namespace=NAMESPACE,
                label_selector=f"job-name={job_name}",
            )
            if not pods.items:
                return ""
            pod_name = pods.items[-1].metadata.name
            return core_v1.read_namespaced_pod_log(
                name=pod_name, namespace=NAMESPACE, tail_lines=tail,
            )
        except ApiException:
            return ""
        except Exception as exc:
            log.debug("_get_job_pod_logs %s: %s", job_name, exc)
            return ""

    @staticmethod
    def _parse_lines(logs: str) -> dict:
        """Parse last TELEMETRY line per stage from log text."""
        result: dict = {}
        for line in logs.splitlines():
            if "[TELEMETRY]" not in line:
                continue
            try:
                parts = line.split("[TELEMETRY]")[1].strip().split()
                kv: dict = {}
                for part in parts:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        v_clean = v.rstrip("x")
                        try:
                            kv[k] = float(v_clean)
                        except ValueError:
                            kv[k] = v
                stage = kv.pop("stage", "unknown")
                result[stage] = kv
            except Exception as exc:
                log.debug("telemetry parse: %s", exc)
        return result

    def _parse_telemetry(self) -> dict:
        result: dict = {}

        # Continuous Deployment pods
        for dep in ("data-gather", "data-prep", "scoring", "model-train"):
            logs = self._get_deployment_pod_logs(dep)
            result.update(self._parse_lines(logs))

        # Fill missing stages from cache so pod restarts don't zero out live gauges
        for stage, cached in self.state.last_telemetry.items():
            if stage not in result:
                result[stage] = cached

        return result

    # ------------------------------------------------------------------
    # Queue depth (file-queue on NFS)
    # ------------------------------------------------------------------

    def _collect_queue_depth(self) -> dict:
        depths: dict = {}
        for name, path in [
            ("raw",      RAW_PATH),
            ("features", FEATURES_PATH),
        ]:
            try:
                depths[name] = {
                    "pending":    len(list(path.glob("*.parquet")))      if path.exists() else 0,
                    "processing": len(list(path.glob("*.parquet.processing"))) if path.exists() else 0,
                    "done":       len(list(path.glob("*.parquet.done"))) if path.exists() else 0,
                }
            except Exception as exc:
                log.debug("queue_depth %s: %s", name, exc)
                depths[name] = {"pending": 0, "processing": 0, "done": 0}
        return depths

    # ------------------------------------------------------------------
    # Fraud scores from GPU scoring output
    # ------------------------------------------------------------------

    def _collect_fraud_metrics(self) -> dict:
        try:
            score_files = sorted(SCORES_PATH.glob("*.parquet"))[-10:] \
                          if SCORES_PATH.exists() else []
            if not score_files:
                return {}
            df = pd.concat([pd.read_parquet(str(f)) for f in score_files], ignore_index=True)
            if "fraud_score" not in df.columns:
                return {}
            alerts = (df[df["fraud_score"] > 0.8]
                      .sort_values("scored_at", ascending=False)
                      .head(20))
            alert_cols = [c for c in ("trans_num", "merchant", "amt", "category", "fraud_score")
                          if c in alerts.columns]
            return {
                "fraud_rate_pct":  float((df["fraud_score"] > 0.5).mean() * 100),
                "total_scored":    len(df),
                "recent_alerts":   alerts[alert_cols].to_dict("records"),
            }
        except Exception as exc:
            log.debug("_collect_fraud_metrics: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # System metrics (psutil)
    # ------------------------------------------------------------------

    def _collect_system(self) -> dict:
        N29_INSTANCE = NFS_NODE_INSTANCE           # 10.23.181.44:9100 (worker .44 / n29)
        N25_INSTANCE = os.environ.get("N25_NODE_INSTANCE", "10.23.181.40:9100")  # worker .40 / n25
        try:
            # CPU % for both worker nodes via node-exporter
            cpu_n29 = 0.0
            cpu_n25 = 0.0
            for inst, label in [(N29_INSTANCE, "n29"), (N25_INSTANCE, "n25")]:
                query = ('100 - (avg(rate(node_cpu_seconds_total'
                         f'{{instance="{inst}",mode="idle"}}[1m]))*100)')
                resp = requests.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": query}, timeout=3,
                )
                resp.raise_for_status()
                result = resp.json().get("data", {}).get("result", [])
                val = round(float(result[0]["value"][1]), 1) if result else 0.0
                if label == "n29":
                    cpu_n29 = val
                else:
                    cpu_n25 = val

            # RAM from node-exporter on .44
            mem_total_resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": f'node_memory_MemTotal_bytes{{instance="{N29_INSTANCE}"}}'},
                timeout=3,
            )
            mem_avail_resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": f'node_memory_MemAvailable_bytes{{instance="{N29_INSTANCE}"}}'},
                timeout=3,
            )
            mem_total_res = mem_total_resp.json().get("data", {}).get("result", [])
            mem_avail_res = mem_avail_resp.json().get("data", {}).get("result", [])
            mem_total = float(mem_total_res[0]["value"][1]) if mem_total_res else 0.0
            mem_avail = float(mem_avail_res[0]["value"][1]) if mem_avail_res else 0.0
            mem_used  = mem_total - mem_avail
            ram_pct   = round(mem_used / mem_total * 100, 1) if mem_total else 0.0

            return {
                "cpu_percent":      cpu_n29,
                "cpu_percent_n29":  cpu_n29,
                "cpu_percent_n25":  cpu_n25,
                "ram_percent":  ram_pct,
                "ram_used_gb":  round(mem_used / 1e9, 2),
                "ram_total_gb": round(mem_total / 1e9, 2),
            }
        except Exception as exc:
            log.debug("_collect_system: %s", exc)
            return {"cpu_percent": 0.0, "cpu_percent_n29": 0.0, "cpu_percent_n25": 0.0,
                    "ram_percent": 0.0, "ram_used_gb": 0.0, "ram_total_gb": 0.0}

    # ------------------------------------------------------------------
    # GPU metrics via Prometheus / DCGM
    # ------------------------------------------------------------------

    def _collect_gpu(self) -> dict:
        """Query DCGM GPU metrics for all GPU nodes (both .44 and .40).

        Uses DCGM_FI_PROF_GR_ENGINE_ACTIVE (1s profiling metric, 0-1 range)
        instead of DCGM_FI_DEV_GPU_UTIL (30s averaged %).  This shows actual
        GPU engine activity spikes (e.g. 99% during mega-batch kernels).
        """
        try:
            metrics: dict = {}
            for metric_name, key_prefix, scale in [
                ("DCGM_FI_PROF_GR_ENGINE_ACTIVE", "gpu_{host}_{gpu}_util_pct", 100.0),
                ("DCGM_FI_DEV_MEM_COPY_UTIL",     "gpu_{host}_{gpu}_mem_pct",  1.0),
            ]:
                # No hostname filter — get all GPUs across all nodes
                resp = requests.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": metric_name}, timeout=3,
                )
                resp.raise_for_status()
                for result in resp.json().get("data", {}).get("result", []):
                    gpu_id = result.get("metric", {}).get("gpu", "0")
                    hostname = result.get("metric", {}).get("Hostname", "unknown")
                    # Shorten hostname: slc6-lg-n3-b30-29 -> n29, slc6-lg-n3-b30-25 -> n25
                    short = "n" + hostname.split("-")[-1] if "-" in hostname else hostname
                    key = key_prefix.format(host=short, gpu=gpu_id)
                    metrics[key] = float(result["value"][1]) * scale
            return metrics if metrics else self._gpu_zeros()
        except Exception as exc:
            log.debug("_collect_gpu: %s", exc)
            return self._gpu_zeros()

    @staticmethod
    def _gpu_zeros() -> dict:
        return {
            "gpu_n29_0_util_pct": 0.0, "gpu_n29_0_mem_pct": 0.0,
            "gpu_n29_1_util_pct": 0.0, "gpu_n29_1_mem_pct": 0.0,
            "gpu_n25_0_util_pct": 0.0, "gpu_n25_0_mem_pct": 0.0,
            "gpu_n25_1_util_pct": 0.0, "gpu_n25_1_mem_pct": 0.0,
        }

    # ------------------------------------------------------------------
    # FlashBlade latency via REST API (file-systems/performance)
    # ------------------------------------------------------------------

    def _collect_flashblade(self) -> dict:
        global _fb_session_token
        try:
            if not _fb_session_token:
                _fb_session_token = _fb_login()
            if not _fb_session_token:
                return {"read_latency_ms": 0.0, "write_latency_ms": 0.0, "avg_latency_ms": 0.0}

            resp = requests.get(
                f"https://{FLASHBLADE_MGMT_IP}/api/2.24/file-systems/performance",
                headers={"X-Auth-Token": _fb_session_token},
                params={"names": FLASHBLADE_FS_NAME},
                verify=False, timeout=5,
            )
            if resp.status_code == 401 or resp.status_code == 403:
                # Token expired — re-login once
                _fb_session_token = _fb_login()
                if not _fb_session_token:
                    return {"read_latency_ms": 0.0, "write_latency_ms": 0.0, "avg_latency_ms": 0.0}
                resp = requests.get(
                    f"https://{FLASHBLADE_MGMT_IP}/api/2.24/file-systems/performance",
                    headers={"X-Auth-Token": _fb_session_token},
                    params={"names": FLASHBLADE_FS_NAME},
                    verify=False, timeout=5,
                )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                return {"read_latency_ms": 0.0, "write_latency_ms": 0.0, "avg_latency_ms": 0.0}
            item = items[0]
            read_ms  = item.get("usec_per_read_op",  0.0) / 1000
            write_ms = item.get("usec_per_write_op", 0.0) / 1000
            return {
                "read_latency_ms":  round(read_ms,  3),
                "write_latency_ms": round(write_ms, 3),
                "avg_latency_ms":   round((read_ms + write_ms) / 2, 3),
            }
        except Exception as exc:
            log.debug("_collect_flashblade: %s", exc)
            return {"read_latency_ms": 0.0, "write_latency_ms": 0.0, "avg_latency_ms": 0.0}

    # ------------------------------------------------------------------
    # Business KPIs
    # ------------------------------------------------------------------

    def _compute_kpis(self, telemetry: dict) -> dict:
        prep       = telemetry.get("prep",    self.state.last_telemetry.get("prep", {}))
        scoring    = telemetry.get("scoring", self.state.last_telemetry.get("scoring", {}))
        # Accumulate rows processed across batches (each telemetry has a chunk_id)
        chunk_id = int(prep.get("chunk_id", -1))
        batch_rows = int(prep.get("rows", 0))
        if chunk_id >= 0 and chunk_id != self.state._last_prep_chunk_id:
            self.state.total_rows_processed += batch_rows
            self.state._last_prep_chunk_id = chunk_id
        total_txns = self.state.total_rows_processed
        # Compute rows/sec from last batch
        gpu_time = float(prep.get("gpu_time_s", 0))
        prep_rps = int(batch_rows / gpu_time) if gpu_time > 0 else 0
        # Prefer real fraud rate from scorer; fall back to synthetic rate
        fraud_rate = float(scoring.get("fraud_rate", 0.005))
        fraud_flagged = int(total_txns * fraud_rate)
        avg_fraud_amt = 250.0
        fraud_exposure = fraud_flagged * avg_fraud_amt
        return {
            "total_transactions": total_txns,
            "prep_rows_per_sec": prep_rps,
            "fraud_flagged": fraud_flagged,
            "fraud_rate_pct": round(fraud_rate * 100, 2),
            "fraud_exposure_usd": round(fraud_exposure, 0),
        }

    # ------------------------------------------------------------------
    # Storage stats
    # ------------------------------------------------------------------

    def _collect_storage(self) -> dict:
        try:
            def _dir_stats(p: Path):
                files = list(p.glob("*.parquet")) if p.exists() else []
                size = sum(f.stat().st_size for f in files) / 1e9
                return len(files), round(size, 2)

            raw_n, raw_gb       = _dir_stats(RAW_PATH)
            feat_n, feat_gb     = _dir_stats(FEATURES_PATH)
            score_n, score_gb   = _dir_stats(SCORES_PATH)
            models_ready = (MODEL_REPO / "fraud_gnn_gpu" / "1" / "state_dict_gnn.pth").exists()
            total_gb = round(raw_gb + feat_gb + score_gb, 2)
            return {
                "raw_files":      raw_n,
                "raw_size_gb":    raw_gb,
                "features_files": feat_n,
                "feat_size_gb":   feat_gb,
                "score_files":    score_n,
                "score_size_gb":  score_gb,
                "total_size_gb":  total_gb,
                "models_ready":   models_ready,
            }
        except Exception as exc:
            log.debug("_collect_storage: %s", exc)
            return {"raw_files": 0, "raw_size_gb": 0.0, "features_files": 0, "feat_size_gb": 0.0,
                    "score_files": 0, "score_size_gb": 0.0, "total_size_gb": 0.0, "models_ready": False}


def load_shap_summary() -> Optional[dict]:
    shap_path = MODEL_REPO / "shap_summary.json"
    if not shap_path.exists():
        return None
    try:
        return json.loads(shap_path.read_text())
    except Exception as exc:
        log.warning("load_shap_summary: %s", exc)
        return None


def load_training_metrics() -> Optional[dict]:
    metrics_path = MODEL_REPO / "training_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except Exception as exc:
        log.warning("load_training_metrics: %s", exc)
        return None
