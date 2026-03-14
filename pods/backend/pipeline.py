"""
Pipeline control (v4): Deployment scaling for demo pipeline.
Each pod gets a dedicated L40S GPU.

Demo flow:
  1. Pre-demo (offline): kubectl scale data-gather to fill /data/raw
  2. Demo: start_pipeline() → data-prep + model-train + triton + scoring (4 GPU pods)
"""
import logging
import os
import shutil
from pathlib import Path

from kubernetes import client, config
from kubernetes.client.rest import ApiException

log = logging.getLogger(__name__)

NAMESPACE = os.environ.get("K8S_NAMESPACE", "fraud-det-v31")

# All deployments tracked for status/replica queries.
ALL_DEPLOYMENTS = ["data-gather", "data-prep", "triton", "scoring", "model-train"]

# Pipeline = everything except gather. 4 pods on 4 GPUs.
PIPELINE_REPLICAS = {
    "data-prep":     1,   # 1 dedicated GPU — mega-batch (100M+ rows/batch)
    "triton":        1,   # 1 dedicated GPU
    "scoring":       1,   # 1 dedicated GPU
    "model-train":   1,   # 1 dedicated GPU
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
        log.info("Scaled deployment/%s to %d", name, replicas)
    except ApiException as e:
        log.warning("scale %s: %s", name, e.reason)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_pipeline() -> dict:
    """Scale pipeline Deployments (prep + train + triton + scoring).
    Data-gather must be stopped first (offline, pre-demo only)."""
    _, apps_v1, _ = _k8s()
    _scale(apps_v1, "data-gather", 0)  # safety: ensure gather is off
    for dep, n in PIPELINE_REPLICAS.items():
        _scale(apps_v1, dep, n)
    return {"status": "started"}


def stop_pipeline() -> dict:
    """Scale all Deployments to 0 (gather + pipeline)."""
    _, apps_v1, _ = _k8s()
    for dep in ALL_DEPLOYMENTS:
        _scale(apps_v1, dep, 0)
    return {"status": "stopped"}


def reset_pipeline(raw_path: Path, *output_paths: Path) -> dict:
    """Stop pipeline, re-queue raw data, clear downstream output.

    Raw files (.done / .processing) are renamed back to .parquet so the
    pre-generated data can be reprocessed without re-running data-gather.
    Features and scores directories are wiped (they get regenerated).
    """
    stop_pipeline()
    # Re-queue raw data: rename .done/.processing back to .parquet
    requeued = 0
    if raw_path and raw_path.exists():
        for suffix in (".done", ".processing"):
            for f in raw_path.glob(f"*{suffix}"):
                orig = f.with_name(f.name[: -len(suffix)])
                try:
                    f.rename(orig)
                    requeued += 1
                except OSError:
                    pass
        log.info("Re-queued %d raw files in %s", requeued, raw_path)
    # Clear downstream output dirs (features, scores)
    cleared = []
    for p in output_paths:
        if p is None:
            continue
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            log.info("Cleared %s", p)
        p.mkdir(parents=True, exist_ok=True)
        p.chmod(0o777)
        cleared.append(str(p))
    return {"status": "reset", "requeued_raw": requeued, "cleared": cleared}


def get_service_states() -> dict:
    """Return status of all pipeline Deployments."""
    _, apps_v1, _ = _k8s()
    states: dict = {}
    for dep in ALL_DEPLOYMENTS:
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
    for dep in ALL_DEPLOYMENTS:
        try:
            d = apps_v1.read_namespaced_deployment(name=dep, namespace=NAMESPACE)
            counts[dep] = {
                "desired": d.spec.replicas or 0,
                "ready":   d.status.ready_replicas or 0,
            }
        except ApiException:
            counts[dep] = {"desired": 0, "ready": 0}
    return counts
