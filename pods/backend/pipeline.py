"""
Pipeline control: K8s Jobs (gather/prep/train) + Deployment scaling (inference).
Uses the kubernetes Python client via in-cluster ServiceAccount credentials.
"""
import logging
import os
import shutil
import time
from pathlib import Path

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

log = logging.getLogger(__name__)

NAMESPACE = os.environ.get("K8S_NAMESPACE", "fraud-det-v31")
JOB_YAML_DIR = Path(__file__).parent / "k8s" / "jobs"


def _k8s():
    """Return (BatchV1Api, AppsV1Api, CoreV1Api) — load in-cluster or kubeconfig."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.BatchV1Api(), client.AppsV1Api(), client.CoreV1Api()


# ---------------------------------------------------------------------------
# Job helpers
# ---------------------------------------------------------------------------

def _load_job_spec(yaml_name: str) -> dict:
    path = JOB_YAML_DIR / yaml_name
    with open(str(path)) as f:
        return yaml.safe_load(f)


def _apply_env_overrides(job_body: dict, overrides: dict) -> dict:
    containers = job_body["spec"]["template"]["spec"]["containers"]
    existing = {e["name"]: e for e in containers[0].get("env", [])}
    for k, v in overrides.items():
        existing[k] = {"name": k, "value": str(v)}
    containers[0]["env"] = list(existing.values())
    return job_body


def _delete_job_if_exists(batch_v1: client.BatchV1Api, job_name: str) -> None:
    try:
        batch_v1.delete_namespaced_job(
            name=job_name,
            namespace=NAMESPACE,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )
        log.info("[INFO] Deleted existing Job: %s", job_name)
        time.sleep(3)
    except ApiException as e:
        if e.status != 404:
            log.warning("[WARN] delete Job %s: %s", job_name, e.reason)


def _create_job(batch_v1: client.BatchV1Api, yaml_name: str, env_overrides: dict = None) -> None:
    job_body = _load_job_spec(yaml_name)
    job_name = job_body["metadata"]["name"]
    _delete_job_if_exists(batch_v1, job_name)
    if env_overrides:
        job_body = _apply_env_overrides(job_body, env_overrides)
    batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job_body)
    log.info("[INFO] Created Job: %s", job_name)


def _wait_for_job(batch_v1: client.BatchV1Api, job_name: str, timeout_s: int = 3600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace=NAMESPACE)
            if job.status.succeeded and job.status.succeeded > 0:
                log.info("[INFO] Job %s completed successfully", job_name)
                return True
            if job.status.failed and job.status.failed > 0:
                log.error("[ERROR] Job %s failed", job_name)
                return False
        except ApiException as e:
            log.warning("[WARN] wait_for_job %s: %s", job_name, e.reason)
        time.sleep(5)
    log.error("[ERROR] Job %s timed out after %ds", job_name, timeout_s)
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_pipeline(env_overrides: dict = None) -> dict:
    """
    Run gather → prep → train Jobs sequentially.
    NOTE: This blocks — call from a background thread/task.
    """
    batch_v1, _, _ = _k8s()
    overrides = env_overrides or {}

    stages = [
        ("data-gather.yaml", "data-gather", overrides),
        ("data-prep.yaml",   "data-prep",   {}),
        ("model-build.yaml", "model-build", {}),
    ]
    for yaml_name, job_name, stage_overrides in stages:
        log.info("[INFO] Starting stage: %s", job_name)
        try:
            _create_job(batch_v1, yaml_name, stage_overrides)
        except ApiException as e:
            msg = f"Failed to create Job {job_name}: {e.reason}"
            log.error("[ERROR] %s", msg)
            return {"status": "error", "stage": job_name, "message": msg}

        if not _wait_for_job(batch_v1, job_name):
            return {"status": "error", "stage": job_name, "message": f"{job_name} failed or timed out"}

    return {"status": "completed", "message": "Pipeline finished successfully"}


def stop_pipeline() -> dict:
    """Delete all running pipeline Jobs."""
    batch_v1, _, _ = _k8s()
    for job_name in ("data-gather", "data-prep", "model-build"):
        _delete_job_if_exists(batch_v1, job_name)
    return {"status": "stopped"}


def reset_pipeline(raw_path: Path, features_path: Path) -> dict:
    """Stop Jobs and clear raw + features data (models preserved)."""
    stop_pipeline()
    deleted = []
    for p in (raw_path, features_path):
        if p.exists():
            shutil.rmtree(str(p), ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)
            deleted.append(str(p))
            log.info("[INFO] Cleared %s", p)
    return {"status": "reset", "cleared": deleted}


def get_service_states() -> dict:
    """Get status of pipeline Jobs + inference Deployment."""
    batch_v1, apps_v1, _ = _k8s()
    states: dict = {}

    for job_name in ("data-gather", "data-prep", "model-build"):
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace=NAMESPACE)
            if job.status.active:
                states[job_name] = "Running"
            elif job.status.succeeded:
                states[job_name] = "Completed"
            elif job.status.failed:
                states[job_name] = "Failed"
            else:
                states[job_name] = "Pending"
        except ApiException:
            states[job_name] = "NotFound"

    try:
        dep = apps_v1.read_namespaced_deployment(name="inference", namespace=NAMESPACE)
        ready = dep.status.ready_replicas or 0
        states["inference"] = "Ready" if ready > 0 else "NotReady"
    except ApiException:
        states["inference"] = "NotFound"

    return states


def write_stress_config(stress_on: bool, num_workers: int = 32, chunk_size: int = 200000) -> None:
    """Re-submit data-gather Job with stress env vars."""
    batch_v1, _, _ = _k8s()
    overrides = {
        "STRESS_MODE": "true" if stress_on else "false",
        "NUM_WORKERS": str(num_workers if stress_on else 8),
        "CHUNK_SIZE": str(chunk_size if stress_on else 100000),
        "TARGET_ROWS": str(5000000 if stress_on else 1000000),
    }
    try:
        _create_job(batch_v1, "data-gather.yaml", overrides)
        log.info("[INFO] Stress data-gather Job submitted: stress=%s", stress_on)
    except ApiException as e:
        log.error("[ERROR] write_stress_config: %s", e.reason)
