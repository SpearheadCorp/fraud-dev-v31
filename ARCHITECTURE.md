# fraud-det-v31 Architecture

## Overview

K8s-native fraud detection ML pipeline running on a bare-metal cluster. The demo shows:
1. GPU-accelerated data generation + feature engineering vs CPU baseline
2. XGBoost GPU training speedup vs CPU
3. Pure FlashBlade NFS performance (read latency, throughput) under ML workload

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  K8s Job: data-gather                                           │
│  gather.py — generates synthetic credit card transactions        │
│  Kaggle-seeded distributions, multiprocessing pool              │
│  Normal: 1M rows (once) │ Stress: 5M rows (continuous loop)     │
│  Writes: /data/raw/*.parquet                                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ NFS read
               ┌───────────────┴───────────────┐
               │                               │
┌──────────────▼──────────────┐ ┌──────────────▼──────────────┐
│  K8s Job: data-prep          │ │  K8s Job: data-prep-cpu      │
│  prepare.py (RAPIDS/cudf)    │ │  prepare_cpu.py (pandas)     │
│  GPU feature engineering     │ │  CPU feature engineering     │
│  → /data/features/*.parquet  │ │  → /data/features-cpu/*.pqt  │
│  TELEMETRY: stage=prep       │ │  TELEMETRY: stage=prep-cpu   │
└──────────────┬──────────────┘ └──────────────┬──────────────┘
               └───────────────┬───────────────┘
                               │ (both run in parallel, both must succeed)
┌──────────────────────────────▼──────────────────────────────────┐
│  K8s Job: model-build                                           │
│  train.py — XGBoost GPU training (falls back to CPU)            │
│  Temporal 70/15/15 split, SHAP analysis                         │
│  Writes model repo: /data/models/fraud_xgboost_gpu/1/xgboost.json │
│                      /data/models/fraud_xgboost_cpu/1/xgboost.json │
│                      /data/models/shap_summary.json              │
│                      /data/models/training_metrics.json          │
│  TELEMETRY: stage=train                                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
┌──────────────▼──────────────┐ ┌──────────────▼──────────────┐
│  K8s Deployment: inference   │ │  K8s Deployment: inference-cpu│
│  Triton Server (FIL) KIND_GPU│ │  Triton Server (FIL) KIND_CPU│
│  start.sh polls for          │ │  same pattern, CPU path      │
│  xgboost.json (non-empty)    │ │                              │
│  + config.pbtxt exists       │ │                              │
│  replicas: 0 → 1 on Start    │ │  replicas: 0 → 1 on Start   │
└──────────────────────────────┘ └──────────────────────────────┘
```

---

## Backend Pod

`pods/backend/` — FastAPI + WebSocket, always running (1 replica).

| File | Purpose |
|---|---|
| `backend.py` | FastAPI app, all HTTP/WS endpoints, pipeline state |
| `pipeline.py` | K8s client — creates/deletes Jobs, scales Deployments |
| `metrics.py` | MetricsCollector — telemetry, psutil, DCGM/Prometheus, FlashBlade |
| `static/dashboard.html` | Single-file dashboard (Chart.js, 3 tabs) |
| `k8s/jobs/*.yaml` | Job manifests loaded at runtime by pipeline.py |

### API Endpoints
| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Redirect to dashboard |
| `/ws/dashboard` | WS | Metrics push every 1s |
| `/api/metrics` | GET | Full metrics snapshot |
| `/api/metrics/shap` | GET | SHAP summary JSON |
| `/api/metrics/training` | GET | Training metrics JSON |
| `/api/control/start` | POST | Start pipeline (async) |
| `/api/control/stop` | POST | Stop all jobs + scale inference to 0 |
| `/api/control/reset` | POST | Stop + clear raw/features data |
| `/api/control/stress` | POST | Toggle stress mode (re-submits data-gather) |
| `/api/control/status` | GET | K8s job + deployment states |

### Pipeline State (in-memory)
```python
state.is_running: bool
state.stress_mode: bool
state.start_time: float
state.last_telemetry: dict  # persisted to /data/models/last_telemetry.json
```

---

## Dashboard (3 Tabs)

### Tab 1 — Business
- KPI cards: Transactions, Fraud Flagged, Fraud Rate, Exposure USD, Projected Savings
- TX/s line chart (full-width) — shows data generation rate and stress step-up
- Category fraud bar chart (14 Kaggle categories)
- SHAP feature importance (loaded from shap_summary.json)
- Training metrics (confusion matrix, F1, AUC-PR, GPU speedup)

### Tab 2 — Infrastructure
- Pipeline Funnel: Generated → Prep CPU → Prep GPU → Trained on → Evaluated (with row counts)
- Stage status pills: Gather / Prep / Train / Inference
- **Combined 4-line chart** (System Overview):
  - Y-left `%`: CPU % (psutil, node-level), GPU % (DCGM, device-level)
  - Y-right `TX/s`: transaction rate (delta rows_generated per second)
  - Y-right-far `ms`: FlashBlade read latency (purefb Prometheus, scale 2–3ms)
  - Checkboxes to toggle each line individually
- CPU vs GPU Speedup table (prep speedup, train speedup, inference speedup)

### Tab 3 — Pipeline Status
- Job/Deployment status cards with colour coding

---

## Telemetry Format

Each pod emits `[TELEMETRY]` lines to stdout, parsed by `metrics.py` from K8s pod logs:

```
[TELEMETRY] stage=gather rows_generated=850000 throughput_mbps=312.4 files_written=9 workers=8 fraud_rate=0.00498
[TELEMETRY] stage=prep   files_read=9 rows_processed=850000 gpu_used=1 elapsed_sec=42
[TELEMETRY] stage=prep-cpu files_read=9 rows_processed=850000 elapsed_sec=67
[TELEMETRY] stage=train  rows_trained=595000 rows_evaluated=127500 f1_cpu=0.912 f1_gpu=0.918 auc_pr_gpu=0.871 speedup_prep=1.59x speedup_train=4.2x
```

Dashboard maps:
- `gather.rows_generated` → funnel-generated, TX/s delta
- `prep.rows_processed` → funnel-prep-gpu, stage-prep
- `prep-cpu.rows_processed` → funnel-prep-cpu
- `train.rows_trained` → funnel-trained; `train.rows_evaluated` → funnel-evaluated
- `train.f1_gpu` → stage-train display

---

## Feature Engineering (22 columns)

Input raw Sparkov schema → 21 engineered features + `is_fraud`:

`amt_log, amt_scaled, hour_of_day, day_of_week, is_weekend, is_night,
distance_km, category_encoded, state_encoded, gender_encoded,
city_pop_log, zip_region, amt, lat, long, city_pop, unix_time,
merch_lat, merch_long, merch_zipcode, zip, is_fraud`

Temporal split: 70% train / 15% validation / 15% test (NOT random — prevents data leakage).

---

## Triton Model Repository

```
/data/models/
├── fraud_xgboost_gpu/
│   └── 1/
│       ├── xgboost.json      ← trigger file (inference start.sh polls for this)
│       └── config.pbtxt      ← written BEFORE xgboost.json (write-order race fix)
├── fraud_xgboost_cpu/
│   └── 1/
│       ├── xgboost.json
│       └── config.pbtxt
├── shap_summary.json         ← top 10 SHAP features + 100 sample values
├── training_metrics.json     ← F1, AUC-PR, confusion matrix, speedup ratios
└── last_telemetry.json       ← persisted stage telemetry (survives backend restart)
```

---

## Monitoring Stack

| Component | URL / Location |
|---|---|
| Prometheus | `http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090` |
| DCGM Exporter | `nvidia-dcgm-exporter.gpu-operator.svc.cluster.local:9400` |
| DCGM ServiceMonitor | `k8s/dcgm-servicemonitor.yaml` (namespace: monitoring, label: release=kube-prometheus-stack) |
| FlashBlade metrics | via purefb Prometheus exporter already scraped by kube-prometheus-stack |

Key Prometheus queries used by backend:
```
DCGM_FI_DEV_GPU_UTIL                                                           ← GPU util %
DCGM_FI_DEV_MEM_COPY_UTIL                                                      ← GPU mem %
purefb_file_systems_performance_latency_usec{name="...",dimension="read"}      ← NFS read latency
```

---

## K8s RBAC

The backend ServiceAccount (`backend-sa`) has these permissions in `k8s/rbac.yaml`:
- `batch`: `get, list, watch, create, delete, patch, update` on `jobs`
- `core`: `get, list, watch` on `pods`; `get` on `pods/log`
- `apps`: `get, patch, update` on `deployments` (for scaling inference)

---

## Seed Data

- Path: `seed-data/credit_card_transactions.csv.zip` (in repo, gitignored from pushes)
- Schema: Sparkov simulation — 1.3M rows, 0.58% fraud rate, 14 categories, 51 states
- `gather.py` uses it to seed statistical distributions; falls back to hardcoded defaults if missing
- Set env var `KAGGLE_SEED_PATH=seed-data/credit_card_transactions.csv.zip` in the Job YAML or .env
