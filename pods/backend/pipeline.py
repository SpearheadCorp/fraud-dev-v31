"""
Pipeline control: Deployment scaling for continuous pipeline stages.
data-gather, data-prep-gpu, data-prep-cpu, scoring-gpu, scoring-cpu are
scaled by start/stop/stress. triton is always-on (not scaled by pipeline
control). model-build remains a Job but is run manually (offline, pre-demo).
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
    "data-gather":   1,
    "data-prep-gpu": 1,   # 1 GPU for prep; triton holds the other L40S
    "data-prep-cpu": 1,
    "scoring-gpu":   1,
    "scoring-cpu":   1,
    # triton is always-on (replicas: 1 in deployments.yaml) — not touched by start/stop
}

# Deployments that are always-on (not scaled by pipeline start/stop/stress).
# get_service_states() reports these separately so dashboard still shows their status.
ALWAYS_ON = ("triton",)

STRESS_REPLICAS = {
    "data-gather":   1,   # rate governed by TARGET_ROWS_PER_SEC env var
    "data-prep-gpu": 1,   # can't exceed 1 GPU prep pod (triton holds the other GPU)
    "data-prep-cpu": 2,   # scale CPU prep workers for stress throughput
    "scoring-gpu":   2,
    "scoring-cpu":   2,
    # triton is always-on — not touched by stress
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
    return {"status": "reset", "cleared": cleared}


def get_service_states() -> dict:
    """Return status of all pipeline Deployments (scaled + always-on)."""
    _, apps_v1, _ = _k8s()
    states: dict = {}
    for dep in list(NORMAL_REPLICAS) + list(ALWAYS_ON):
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


def write_stress_config(stress_on: bool) -> None:
    """Scale all pipeline Deployments to STRESS_REPLICAS or NORMAL_REPLICAS."""
    _, apps_v1, _ = _k8s()
    replicas = STRESS_REPLICAS if stress_on else NORMAL_REPLICAS
    for dep, n in replicas.items():
        _scale(apps_v1, dep, n)
    log.info("[INFO] Stress mode %s: replica counts updated", "on" if stress_on else "off")
