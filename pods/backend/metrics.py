"""
Metrics collection: telemetry (K8s pod logs), system (psutil), GPU (Prometheus/DCGM),
FlashBlade storage latency (Prometheus/purefb exporter), queue depth, fraud scores.
"""
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import psutil
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

log = logging.getLogger(__name__)

PROMETHEUS_URL    = os.environ.get("PROMETHEUS_URL",          "http://prometheus:9090")
MODEL_REPO        = Path(os.environ.get("MODEL_REPO_PATH",    "/data/models"))
RAW_PATH_GPU      = Path(os.environ.get("OUTPUT_PATH_GPU",    "/data/raw/gpu"))
RAW_PATH_CPU      = Path(os.environ.get("OUTPUT_PATH_CPU",    "/data/raw/cpu"))
FEATURES_GPU_PATH = Path(os.environ.get("FEATURES_GPU_PATH",  "/data/features/gpu"))
FEATURES_CPU_PATH = Path(os.environ.get("FEATURES_CPU_DATA_PATH", "/data/features-cpu"))
SCORES_GPU_PATH   = Path(os.environ.get("SCORES_GPU_PATH",    "/data/features/scores"))
SCORES_CPU_PATH   = Path(os.environ.get("SCORES_CPU_PATH",    "/data/features-cpu/scores"))
NAMESPACE         = os.environ.get("K8S_NAMESPACE",           "fraud-det-v31")
FLASHBLADE_FS_NAME = os.environ.get("FLASHBLADE_FS_NAME",    "financial-fraud-detection-demo")
GPU_NODE_HOSTNAME  = os.environ.get("GPU_NODE_HOSTNAME",      "slc6-lg-n3-b30-29")
NFS_NODE_INSTANCE  = os.environ.get("NFS_NODE_INSTANCE",      "10.23.181.44:9100")


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
        self.stress_mode: bool = False
        self.start_time: Optional[float] = None
        self.last_telemetry: dict = {}

    def reset(self) -> None:
        self.is_running = False
        self.stress_mode = False
        self.start_time = None
        self.last_telemetry = {}

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
                log.info("[INFO] Loaded telemetry cache (%d stages)", len(self.state.last_telemetry))
        except Exception as exc:
            log.debug("[DEBUG] load_telemetry_cache: %s", exc)

    def _save_telemetry_cache(self) -> None:
        try:
            _TELEMETRY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _TELEMETRY_CACHE.write_text(json.dumps(self.state.last_telemetry))
        except Exception as exc:
            log.debug("[DEBUG] save_telemetry_cache: %s", exc)

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
            "stress_mode": self.state.stress_mode,
            "elapsed_sec": self.state.elapsed_sec,
            "timestamp":   time.time(),
            "pipeline": {
                "gather":      telemetry.get("gather",      {}),
                "prep_gpu":    telemetry.get("prep-gpu",    {}),
                "prep_cpu":    telemetry.get("prep-cpu",    {}),
                "scoring_gpu": telemetry.get("scoring-gpu", {}),
                "scoring_cpu": telemetry.get("scoring-cpu", {}),
                "train":       telemetry.get("train",       {}),
            },
            "queue":     queue,
            "fraud":     fraud,
            "business":  business,
            "storage":   storage,
            "flashblade": flashblade,
            "system":    system,
            "gpu":       gpu,
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
            log.debug("[DEBUG] _get_deployment_pod_logs %s: %s", dep_name, exc)
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
            log.debug("[DEBUG] _get_job_pod_logs %s: %s", job_name, exc)
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
                log.debug("[DEBUG] telemetry parse: %s", exc)
        return result

    def _parse_telemetry(self) -> dict:
        result: dict = {}

        # Continuous Deployment pods
        for dep in ("data-gather", "data-prep-gpu", "data-prep-cpu", "scoring-gpu", "scoring-cpu"):
            logs = self._get_deployment_pod_logs(dep)
            result.update(self._parse_lines(logs))

        # model-build is still a Job (offline, run once)
        logs = self._get_job_pod_logs("model-build")
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
            ("raw_gpu",      RAW_PATH_GPU),
            ("raw_cpu",      RAW_PATH_CPU),
            ("features_gpu", FEATURES_GPU_PATH),
            ("features_cpu", FEATURES_CPU_PATH),
        ]:
            try:
                depths[name] = {
                    "pending":    len(list(path.glob("*.parquet")))      if path.exists() else 0,
                    "processing": len(list(path.glob("*.parquet.processing"))) if path.exists() else 0,
                    "done":       len(list(path.glob("*.parquet.done"))) if path.exists() else 0,
                }
            except Exception as exc:
                log.debug("[DEBUG] queue_depth %s: %s", name, exc)
                depths[name] = {"pending": 0, "processing": 0, "done": 0}
        return depths

    # ------------------------------------------------------------------
    # Fraud scores from GPU scoring output
    # ------------------------------------------------------------------

    def _collect_fraud_metrics(self) -> dict:
        try:
            score_files = sorted(SCORES_GPU_PATH.glob("*.parquet"))[-10:] \
                          if SCORES_GPU_PATH.exists() else []
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
            log.debug("[DEBUG] _collect_fraud_metrics: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # System metrics (psutil)
    # ------------------------------------------------------------------

    def _collect_system(self) -> dict:
        try:
            cpu = psutil.cpu_percent(interval=None)
            vm  = psutil.virtual_memory()
            return {
                "cpu_percent":  cpu,
                "ram_percent":  vm.percent,
                "ram_used_gb":  round(vm.used / 1e9, 2),
                "ram_total_gb": round(vm.total / 1e9, 2),
            }
        except Exception as exc:
            log.debug("[DEBUG] _collect_system: %s", exc)
            return {"cpu_percent": 0.0, "ram_percent": 0.0, "ram_used_gb": 0.0, "ram_total_gb": 0.0}

    # ------------------------------------------------------------------
    # GPU metrics via Prometheus / DCGM
    # ------------------------------------------------------------------

    def _collect_gpu(self) -> dict:
        try:
            metrics: dict = {}
            for metric_name, key_prefix in [
                ("DCGM_FI_DEV_GPU_UTIL",     "gpu_{}_util_pct"),
                ("DCGM_FI_DEV_MEM_COPY_UTIL", "gpu_{}_mem_pct"),
            ]:
                query = (f'{metric_name}{{Hostname="{GPU_NODE_HOSTNAME}"}}'
                         if GPU_NODE_HOSTNAME else metric_name)
                resp = requests.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": query}, timeout=3,
                )
                resp.raise_for_status()
                for result in resp.json().get("data", {}).get("result", []):
                    gpu_id = result.get("metric", {}).get("gpu", "0")
                    metrics[key_prefix.format(gpu_id)] = float(result["value"][1])
            return metrics if metrics else self._gpu_zeros()
        except Exception as exc:
            log.debug("[DEBUG] _collect_gpu: %s", exc)
            return self._gpu_zeros()

    @staticmethod
    def _gpu_zeros() -> dict:
        return {"gpu_0_util_pct": 0.0, "gpu_0_mem_pct": 0.0}

    # ------------------------------------------------------------------
    # FlashBlade NFS ops/s via node-exporter (node .44 = slc6-lg-n3-b30-29)
    # purefb exporter not deployed; node_nfs_requests_total is the available proxy.
    # ------------------------------------------------------------------

    def _collect_flashblade(self) -> dict:
        try:
            out: dict = {}
            for method, key in [("Read", "nfs_read_ops"), ("Write", "nfs_write_ops")]:
                query = (f'sum(rate(node_nfs_requests_total'
                         f'{{instance="{NFS_NODE_INSTANCE}",method="{method}"}}[30s]))')
                resp = requests.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": query}, timeout=3,
                )
                resp.raise_for_status()
                result = resp.json().get("data", {}).get("result", [])
                out[key] = round(float(result[0]["value"][1]), 1) if result else 0.0
            return out
        except Exception as exc:
            log.debug("[DEBUG] _collect_flashblade: %s", exc)
            return {"nfs_read_ops": 0.0, "nfs_write_ops": 0.0}

    # ------------------------------------------------------------------
    # Business KPIs
    # ------------------------------------------------------------------

    def _compute_kpis(self, telemetry: dict) -> dict:
        gather     = telemetry.get("gather",      self.state.last_telemetry.get("gather", {}))
        scoring    = telemetry.get("scoring-gpu", self.state.last_telemetry.get("scoring-gpu", {}))
        total_txns = int(gather.get("rows_generated", 0))
        # Prefer real fraud rate from scorer; fall back to gather synthetic rate
        fraud_rate = float(scoring.get("fraud_rate",
                           gather.get("fraud_rate", 0.005)))
        fraud_flagged = int(total_txns * fraud_rate)
        avg_fraud_amt = 250.0
        fraud_exposure = fraud_flagged * avg_fraud_amt
        annual_savings = fraud_exposure * 365 / max(self.state.elapsed_sec / 86400, 1 / 365)
        return {
            "total_transactions": total_txns,
            "fraud_flagged": fraud_flagged,
            "fraud_rate_pct": round(fraud_rate * 100, 3),
            "fraud_exposure_usd": round(fraud_exposure, 0),
            "projected_annual_savings_usd": round(min(annual_savings, 50_000_000), 0),
        }

    # ------------------------------------------------------------------
    # Storage stats
    # ------------------------------------------------------------------

    def _collect_storage(self) -> dict:
        try:
            raw_gpu_files  = list(RAW_PATH_GPU.glob("*.parquet"))  if RAW_PATH_GPU.exists()      else []
            raw_cpu_files  = list(RAW_PATH_CPU.glob("*.parquet"))  if RAW_PATH_CPU.exists()      else []
            feat_gpu_files = list(FEATURES_GPU_PATH.glob("*.parquet")) if FEATURES_GPU_PATH.exists() else []
            feat_cpu_files = list(FEATURES_CPU_PATH.glob("*.parquet")) if FEATURES_CPU_PATH.exists() else []
            raw_size = sum(f.stat().st_size for f in raw_gpu_files + raw_cpu_files) / 1e9
            models_ready = (MODEL_REPO / "fraud_gnn_gpu" / "1" / "state_dict_gnn.pth").exists()
            return {
                "raw_gpu_files":      len(raw_gpu_files),
                "raw_cpu_files":      len(raw_cpu_files),
                "raw_size_gb":        round(raw_size, 2),
                "features_gpu_files": len(feat_gpu_files),
                "features_cpu_files": len(feat_cpu_files),
                "models_ready":       models_ready,
            }
        except Exception as exc:
            log.debug("[DEBUG] _collect_storage: %s", exc)
            return {"raw_gpu_files": 0, "raw_cpu_files": 0, "raw_size_gb": 0.0,
                    "features_gpu_files": 0, "features_cpu_files": 0, "models_ready": False}


def load_shap_summary() -> Optional[dict]:
    shap_path = MODEL_REPO / "shap_summary.json"
    if not shap_path.exists():
        return None
    try:
        return json.loads(shap_path.read_text())
    except Exception as exc:
        log.warning("[WARN] load_shap_summary: %s", exc)
        return None


def load_training_metrics() -> Optional[dict]:
    metrics_path = MODEL_REPO / "training_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except Exception as exc:
        log.warning("[WARN] load_training_metrics: %s", exc)
        return None
