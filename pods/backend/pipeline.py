"""
Pipeline control (v4): Deployment scaling for continuous pipeline stages.
Normal mode: 32 NFS writers (4 gather × 8 workers), 2 prep, 2 scoring, 1 triton, 1 model-train.
Stress mode: 128 NFS writers (8 gather × 16 workers), model-train paused.
"""
import logging
import os
import shutil
from pathlib import Path

from kubernetes import client, config
from kubernetes.client.rest import ApiException

log = logging.getLogger(__name__)

NAMESPACE = os.environ.get("K8S_NAMESPACE", "fraud-det-v31")

NORMAL_REPLICAS = {
    "data-gather":   4,   # 4 pods × 8 workers = 32 NFS writers
    "data-prep":     1,   # 1 GPU on .44 (other GPU = triton)
    "triton":        1,
    "scoring":       2,
    "model-train":   1,   # 1 GPU on .40
}

STRESS_REPLICAS = {
    "data-gather":   8,   # 8 pods × 16 workers = 128 NFS writers
    "data-prep":     2,
    "triton":        1,
    "scoring":       2,
    "model-train":   0,   # paused during stress — frees GPU for prep replica
}


def _k8s():
    """Return (BatchV1Api, AppsV1Api, CoreV1Api) — load in-cluster or kubeconfig."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.BatchV1Api(), client.AppsV1Api(), client.CoreV1Api()


def _scale(apps_v1: client.AppsV1Api, name: str, replicas: int) -> None:
    try:
        apps_v1.patch_namespaced_deployment_scale(
            name=name,
            namespace=NAMESPACE,
            body={"spec": {"replicas": replicas}},
        )
        log.info("[INFO] Scaled deployment/%s to %d", name, replicas)
    except ApiException as e:
        log.warning("[WARN] scale %s: %s", name, e.reason)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_pipeline(overrides: dict = None) -> dict:
    """Scale all pipeline Deployments to NORMAL_REPLICAS. Returns immediately."""
    _, apps_v1, _ = _k8s()
    for dep, n in NORMAL_REPLICAS.items():
        _scale(apps_v1, dep, n)
    return {"status": "started"}


def stop_pipeline() -> dict:
    """Scale all pipeline Deployments to 0."""
    _, apps_v1, _ = _k8s()
    for dep in NORMAL_REPLICAS:
        _scale(apps_v1, dep, 0)
    return {"status": "stopped"}


def reset_pipeline(*paths: Path) -> dict:
    """Stop all Deployments and clear all data directories (raw, features, scores)."""
    stop_pipeline()
    return clear_data_files(*paths)


def clear_data_files(*paths: Path) -> dict:
    """Delete all data directories without stopping pods (pods must be stopped first)."""
    cleared = []
    for p in paths:
        if p is None:
            continue
        if p.exists():
            shutil.rmtree(str(p), ignore_errors=True)
            log.info("[INFO] Cleared %s", p)
        p.mkdir(parents=True, exist_ok=True)
        p.chmod(0o777)  # NFS provisioner creates dirs world-writable; match it post-rmtree
        cleared.append(str(p))
    return {"status": "cleared", "cleared": cleared}


def get_service_states() -> dict:
    """Return status of all pipeline Deployments."""
    _, apps_v1, _ = _k8s()
    states: dict = {}
    for dep in NORMAL_REPLICAS:
        try:
            d = apps_v1.read_namespaced_deployment(name=dep, namespace=NAMESPACE)
            ready   = d.status.ready_replicas or 0
            desired = d.spec.replicas or 0
            if desired == 0:
                states[dep] = "Stopped"
            elif ready >= desired:
                states[dep] = "Ready"
            else:
                states[dep] = "Scaling"
        except ApiException:
            states[dep] = "NotFound"
    return states


def get_replica_counts() -> dict:
    """Return {name: {desired, ready}} for all pipeline Deployments."""
    _, apps_v1, _ = _k8s()
    counts: dict = {}
    for dep in NORMAL_REPLICAS:
        try:
            d = apps_v1.read_namespaced_deployment(name=dep, namespace=NAMESPACE)
            counts[dep] = {
                "desired": d.spec.replicas or 0,
                "ready":   d.status.ready_replicas or 0,
            }
        except ApiException:
            counts[dep] = {"desired": 0, "ready": 0}
    return counts


def write_stress_config(stress_on: bool) -> None:
    """Scale all pipeline Deployments to STRESS_REPLICAS or NORMAL_REPLICAS."""
    _, apps_v1, _ = _k8s()
    replicas = STRESS_REPLICAS if stress_on else NORMAL_REPLICAS
    for dep, n in replicas.items():
        _scale(apps_v1, dep, n)
    log.info("[INFO] Stress mode %s: replica counts updated", "on" if stress_on else "off")
