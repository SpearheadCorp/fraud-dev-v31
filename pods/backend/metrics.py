"""
Metrics collection: telemetry (K8s pod logs), system (psutil), GPU (Prometheus/DCGM).
"""
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

import psutil
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

log = logging.getLogger(__name__)

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
MODEL_REPO = Path(os.environ.get("MODEL_REPO_PATH", "/data/models"))
RAW_PATH = Path(os.environ.get("RAW_DATA_PATH", "/data/raw"))
FEATURES_PATH = Path(os.environ.get("FEATURES_DATA_PATH", "/data/features"))
FEATURES_CPU_PATH = Path(os.environ.get("FEATURES_CPU_DATA_PATH", "/data/features-cpu"))
NAMESPACE = os.environ.get("K8S_NAMESPACE", "fraud-det-v31")


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


class MetricsCollector:
    def __init__(self, state: PipelineState) -> None:
        self.state = state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self) -> dict:
        telemetry = self._parse_telemetry()
        system = self._collect_system()
        gpu = self._collect_gpu()
        business = self._compute_kpis(telemetry)
        storage = self._collect_storage()

        # Cache latest telemetry for KPI continuity
        if telemetry:
            self.state.last_telemetry = telemetry

        return {
            "is_running": self.state.is_running,
            "stress_mode": self.state.stress_mode,
            "elapsed_sec": self.state.elapsed_sec,
            "timestamp": time.time(),
            "system": system,
            "gpu": gpu,
            "pipeline": {
                "gather": telemetry.get("gather", {}),
                "prep": telemetry.get("prep", {}),
                "prep-cpu": telemetry.get("prep-cpu", {}),
                "train": telemetry.get("train", {}),
            },
            "business": business,
            "storage": storage,
        }

    # ------------------------------------------------------------------
    # Telemetry from K8s pod logs
    # ------------------------------------------------------------------

    def _get_job_pod_logs(self, job_name: str, tail: int = 300) -> str:
        """Get logs from the pod created by a Job. Returns empty string on failure."""
        try:
            core_v1 = _core_v1()
            pods = core_v1.list_namespaced_pod(
                namespace=NAMESPACE,
                label_selector=f"job-name={job_name}",
            )
            if not pods.items:
                return ""
            pod_name = pods.items[-1].metadata.name  # latest pod
            return core_v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=NAMESPACE,
                tail_lines=tail,
            )
        except ApiException:
            return ""
        except Exception as exc:
            log.debug("[DEBUG] _get_job_pod_logs %s: %s", job_name, exc)
            return ""

    def _parse_telemetry(self) -> dict:
        result: dict = {}
        for job_name in ("data-gather", "data-prep", "data-prep-cpu", "model-build"):
            logs = self._get_job_pod_logs(job_name, tail=300)
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
                    result[stage] = kv   # latest line wins per stage
                except Exception as exc:
                    log.debug("[DEBUG] telemetry parse error on line %r: %s", line, exc)
        return result if result else self.state.last_telemetry

    # ------------------------------------------------------------------
    # System metrics (psutil)
    # ------------------------------------------------------------------

    def _collect_system(self) -> dict:
        try:
            cpu = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            return {
                "cpu_percent": cpu,
                "ram_percent": vm.percent,
                "ram_used_gb": round(vm.used / 1e9, 2),
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
            metrics = {}
            for metric_name, key_prefix in [
                ("DCGM_FI_DEV_GPU_UTIL", "gpu_{}_util_pct"),
                ("DCGM_FI_DEV_MEM_COPY_UTIL", "gpu_{}_mem_pct"),
            ]:
                url = f"{PROMETHEUS_URL}/api/v1/query"
                resp = requests.get(url, params={"query": metric_name}, timeout=3)
                resp.raise_for_status()
                data = resp.json()
                for result in data.get("data", {}).get("result", []):
                    gpu_id = result.get("metric", {}).get("gpu", "0")
                    value = float(result["value"][1])
                    metrics[key_prefix.format(gpu_id)] = value
            return metrics if metrics else self._gpu_zeros()
        except Exception as exc:
            log.debug("[DEBUG] _collect_gpu: %s", exc)
            return self._gpu_zeros()

    @staticmethod
    def _gpu_zeros() -> dict:
        return {"gpu_0_util_pct": 0.0, "gpu_0_mem_pct": 0.0}

    # ------------------------------------------------------------------
    # Business KPIs derived from telemetry
    # ------------------------------------------------------------------

    def _compute_kpis(self, telemetry: dict) -> dict:
        gather = telemetry.get("gather", self.state.last_telemetry.get("gather", {}))
        total_txns = int(gather.get("rows_generated", 0))
        fraud_rate = float(gather.get("fraud_rate", 0.005))
        fraud_flagged = int(total_txns * fraud_rate)

        # Estimate avg fraud amount from training metrics if available
        avg_fraud_amt = 250.0  # fallback
        train = telemetry.get("train", self.state.last_telemetry.get("train", {}))
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
            raw_files = list(RAW_PATH.glob("*.parquet")) if RAW_PATH.exists() else []
            feat_files = list(FEATURES_PATH.glob("*.parquet")) if FEATURES_PATH.exists() else []
            feat_cpu_files = list(FEATURES_CPU_PATH.glob("*.parquet")) if FEATURES_CPU_PATH.exists() else []
            raw_size = sum(f.stat().st_size for f in raw_files) / 1e9
            models_ready = (MODEL_REPO / "fraud_xgboost_gpu" / "1" / "xgboost.json").exists()
            return {
                "raw_files": len(raw_files),
                "raw_size_gb": round(raw_size, 2),
                "features_files": len(feat_files),
                "features_cpu_files": len(feat_cpu_files),
                "models_ready": models_ready,
            }
        except Exception as exc:
            log.debug("[DEBUG] _collect_storage: %s", exc)
            return {"raw_files": 0, "raw_size_gb": 0.0, "features_files": 0, "features_cpu_files": 0, "models_ready": False}


def load_shap_summary() -> Optional[dict]:
    """Load SHAP summary from model repository."""
    shap_path = MODEL_REPO / "shap_summary.json"
    if not shap_path.exists():
        return None
    try:
        return json.loads(shap_path.read_text())
    except Exception as exc:
        log.warning("[WARN] load_shap_summary: %s", exc)
        return None


def load_training_metrics() -> Optional[dict]:
    """Load training metrics from model repository."""
    metrics_path = MODEL_REPO / "training_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except Exception as exc:
        log.warning("[WARN] load_training_metrics: %s", exc)
        return None
