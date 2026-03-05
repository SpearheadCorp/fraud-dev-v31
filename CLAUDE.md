# fraud-det-v31 — Claude Code Instructions

## Project
Financial Fraud Detection demo v3.1. Kubernetes-based ML pipeline on a bare-metal lab cluster.
Generates synthetic credit card transactions → GPU/CPU feature engineering → XGBoost training →
Triton inference, with a real-time web dashboard. Purpose: demo Pure FlashBlade NFS performance
alongside GPU-accelerated ML.

---

## Lab Access

| Machine | IP | Role | SSH |
|---|---|---|---|
| Build VM | 10.23.181.247 | Docker builds + private registry | `ssh -i ~/.ssh/id_rsa tduser@10.23.181.247` |
| K8s Worker | 10.23.181.44 | All workload runs here | `ssh -i ~/.ssh/id_rsa tduser@10.23.181.44` |
| K8s Master | 10.23.181.153 | Control plane | **No SSH access** |
| GPU Worker | 10.23.181.40 | GPU-dedicated, tainted | **DO NOT SSH** |

**kubectl runs locally on Windows VDI only** — never from inside a node.

---

## Build Workflow (CRITICAL — follow exactly)

```bash
# 1. Edit code locally, commit and push
git add <files> && git commit -m "message" && git push

# 2. On build VM: pull + build + push to private registry
ssh -i ~/.ssh/id_rsa tduser@10.23.181.247 \
  "cd /home/tduser/fraud-det-v31 && git pull && \
   docker build -t 10.23.181.247:5000/fraud-det-v31/backend:latest -f pods/backend/Dockerfile . && \
   docker push 10.23.181.247:5000/fraud-det-v31/backend:latest"

# 3. Restart deployment (run locally)
kubectl -n fraud-det-v31 rollout restart deployment/backend
```

**NEVER use `scp`** — always git push then git pull on the build VM.
**All Dockerfiles use repo root as build context** (e.g., `docker build -f pods/backend/Dockerfile .`).

### Per-image build commands (on build VM after git pull)

```bash
REGISTRY=10.23.181.247:5000/fraud-det-v31
docker build -t $REGISTRY/backend:latest       -f pods/backend/Dockerfile .
docker build -t $REGISTRY/data-gather:latest   -f pods/data-gather/Dockerfile .
docker build -t $REGISTRY/data-prep:latest     -f pods/data-prep/Dockerfile .
docker build -t $REGISTRY/data-prep-cpu:latest -f pods/data-prep-cpu/Dockerfile .
docker build -t $REGISTRY/model-build:latest   -f pods/model-build/Dockerfile .
docker build -t $REGISTRY/inference:latest     -f pods/inference/Dockerfile .
docker build -t $REGISTRY/inference-cpu:latest -f pods/inference-cpu/Dockerfile .
```

---

## K8s Namespace: `fraud-det-v31`

### Dashboard
**http://10.23.181.44:30880** (NodePort via backend service)

### Deployments (always-on)
| Name | Image | Normal replicas |
|---|---|---|
| backend | `fraud-det-v31/backend:latest` | 1 |
| inference | `fraud-det-v31/inference:latest` | 0 (scaled up on pipeline Start) |
| inference-cpu | `fraud-det-v31/inference-cpu:latest` | 0 (scaled up on pipeline Start) |

### Pipeline Jobs (created/deleted by backend)
| Job | Image | GPU |
|---|---|---|
| data-gather | `fraud-det-v31/data-gather:latest` | No |
| data-prep | `fraud-det-v31/data-prep:latest` | Yes (RAPIDS) |
| data-prep-cpu | `fraud-det-v31/data-prep-cpu:latest` | No |
| model-build | `fraud-det-v31/model-build:latest` | Yes (XGBoost) |

### PVCs (FlashBlade NFS at 10.23.181.65)
| PVC | Capacity | Mount | Purpose |
|---|---|---|---|
| raw-data-pvc | 500Gi | /data/raw | Raw parquet from data-gather |
| features-data-pvc | 100Gi | /data/features | GPU-path features |
| features-cpu-data-pvc | 100Gi | /data/features-cpu | CPU-path features |
| model-repo-pvc | 50Gi | /data/models | Triton model repo + metrics |

### Services
- `backend` NodePort 8080→**30880**
- `inference` ClusterIP 8000/8001/8002 (Triton HTTP/gRPC/metrics)
- `inference-cpu` ClusterIP 8000/8001/8002

---

## Private Registry

`10.23.181.247:5000` — HTTP only, no auth, DELETE enabled.

```bash
# List repos
curl -s http://10.23.181.247:5000/v2/_catalog

# Check disk (critical — 128GB free as of 2026-03-05, was full earlier)
ssh -i ~/.ssh/id_rsa tduser@10.23.181.247 "df -h /"

# Clean Docker build cache if low
ssh -i ~/.ssh/id_rsa tduser@10.23.181.247 "docker builder prune -af"

# Clean unused images if low
ssh -i ~/.ssh/id_rsa tduser@10.23.181.247 "docker image prune -af"
```

---

## Monitoring

- **Prometheus:** `http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090`
- **GPU metrics:** `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_MEM_COPY_UTIL`
  - Scraped via `k8s/dcgm-servicemonitor.yaml` (added 2026-03-05)
  - Source: `nvidia-dcgm-exporter.gpu-operator.svc.cluster.local:9400`
- **FlashBlade latency:** `purefb_file_systems_performance_latency_usec{name="financial-fraud-detection-demo",dimension="read"}`
- **FLASHBLADE_FS_NAME** env var on backend (default: `financial-fraud-detection-demo`)

---

## Key Coding Conventions

- **Pipeline runs async** in backend: `_run_pipeline_task()` uses `asyncio.create_task` + `run_in_executor`. HTTP endpoints remain responsive during pipeline.
- **Telemetry persistence**: `last_telemetry` is cached to `/data/models/last_telemetry.json` on every collect cycle and reloaded on backend startup.
- **Inference lifecycle**: Scaled to 0 at rest. `start_pipeline()` scales inference to 1, `stop_pipeline()` scales back to 0.
- **Stress mode**: Sets `RUN_MODE=continuous` on data-gather job so TX/s stays sustained.
- **GPU probe**: `prepare.py` uses subprocess to probe `cudf` before importing it — prevents SIGSEGV from killing main process.
- **Write ordering**: `config.pbtxt` written before `xgboost.json` in train.py. Inference polls for `xgboost.json` as the trigger — config must already exist.
- **Inference start.sh**: polls with `-s` (non-empty) not `-f` (exists) to guard against NFS flush lag.

---

## Node / GPU Details
- Worker .44 has 2× NVIDIA L40S (CC 8.9), Driver 580.105.08 (CUDA 13.0 capable)
- RAPIDS base image: `nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.11` (only tag that exists for py3.11)
- `nodeSelector: kubernetes.io/hostname: slc6-lg-n3-b30-29` on GPU jobs — this is worker .44
