"""
Pipeline control (v4): Deployment scaling for continuous pipeline stages.
GPU time-slicing: 5 pods per GPU, 20 GPU slots total (4 physical L40S).
Normal: 16 gather + 1 prep + 1 triton + 1 scoring + 1 train = 20 GPU pods.
Stress: scoring scales to 4 (uses freed gather slots).
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
    "data-gather":  12,   # 3 per GPU × 4 GPUs (time-sliced, 10M rows each)
    "data-prep":     1,   # 1 GPU (time-sliced)
    "triton":        1,   # 1 GPU (time-sliced)
    "scoring":       1,   # 1 GPU slot (calls Triton via HTTP)
    "model-train":   1,   # 1 GPU (time-sliced)
}

STRESS_REPLICAS = {
    "data-gather":  12,   # already maxed (VRAM limited at 10M chunks)
    "data-prep":     1,
    "triton":        1,
    "scoring":       4,   # scale scoring throughput
    "model-train":   1,
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
