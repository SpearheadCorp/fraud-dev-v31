# Implementation Plan: Continuous Pipeline Redesign (v3.2)

## Context
The current pipeline is sequential batch (gather → wait → prep → wait → train → done). The intended design is a continuous pipeline where all stages run simultaneously with NFS as the buffer between them, matching the Pure Storage/NVIDIA NIMs architecture. GNN (GraphSAGE via PyTorch Geometric) is added in front of XGBoost for fraud detection, making the solution NIM-compliant.

**Key decisions confirmed:**
- Real GraphSAGE (PyTorch Geometric) in model-build
- Sub-folders on existing PVCs (no new PVCs needed)
- model-build: manual `kubectl create job model-build` before demo (offline, pre-trained)
- GNN + XGBoost served as combined Triton Python backend
- Single triton pod hosts both GPU and CPU models

---

## New Architecture

| Pod | K8s Type | Replicas (normal/stress) | Change |
|---|---|---|---|
| data-gather | Deployment | 1 / 1 | Writes to /data/raw/gpu/ AND /data/raw/cpu/ |
| data-prep-gpu | Deployment (was Job) | 2 / 4 | Continuous file-queue loop |
| data-prep-cpu | Deployment (was Job) | 2 / 4 | Same, CPU path |
| scoring-gpu | Deployment (NEW) | 2 / 2 | File-queue, sliding-window graph, Triton calls |
| scoring-cpu | Deployment (NEW) | 2 / 2 | Same for CPU model |
| triton | Deployment (replaces inference + inference-cpu) | 1 / 1 | Python backend, both models |
| model-build | Job (unchanged trigger, new code) | — | GraphSAGE + XGBoost, offline |
| backend | Deployment | 1 / 1 | Scales Deployments instead of creating Jobs |

## Storage Layout (sub-folders on existing PVCs)

```
raw-data-pvc      → /data/raw/
                        gpu/          ← data-gather writes here (GPU lane raw chunks)
                        cpu/          ← data-gather writes here (CPU lane raw chunks)

features-data-pvc → /data/features/
                        gpu/          ← data-prep-gpu writes prepped chunks
                        scores/       ← scoring-gpu writes fraud score results

features-cpu-pvc  → /data/features-cpu/
                        (root)        ← data-prep-cpu writes prepped chunks
                        scores/       ← scoring-cpu writes fraud score results

model-repo-pvc    → /data/models/
                        fraud_gnn_gpu/    ← Triton Python backend (GPU)
                        fraud_gnn_cpu/    ← Triton Python backend (CPU)
                        shap_summary.json
                        training_metrics.json
                        last_telemetry.json
```

## File Queue Protocol (all prep and scoring pods)

Atomic NFS rename used as distributed queue lock:
1. Scan directory for `*.parquet` (skip `*.parquet.processing`, `*.parquet.done`)
2. `rename(chunk.parquet → chunk.parquet.processing)` — success = this worker owns it; ENOENT = another worker claimed it first
3. Process claimed file
4. `rename(chunk.parquet.processing → chunk.parquet.done)`
5. Loop. Sleep 0.5s if no files available.

Reset deletes all files including `.processing` and `.done`.

---

## Phase 1: data-gather — Dual Output

**File:** `pods/data-gather/gather.py`

**Changes:**
- Add env vars `OUTPUT_PATH_GPU` (default `/data/raw/gpu`) and `OUTPUT_PATH_CPU` (default `/data/raw/cpu`)
- Remove single `OUTPUT_PATH` global (or keep as fallback for backwards compat)
- Both dirs created at startup
- In both `once` and `continuous` loops: write each chunk to **both** paths with identical filename `raw_chunk_{idx:06d}.parquet`
- Telemetry: no change needed (rows_per_sec etc. unchanged)

**File:** `k8s/jobs/data-gather.yaml` → **DELETE** (replaced by Deployment in deployments.yaml)

---

## Phase 2: data-prep (Continuous File-Queue Loop)

### pods/data-prep/prepare.py — Rewrite main()

**Input:** `INPUT_PATH=/data/raw/gpu` | **Output:** `OUTPUT_PATH=/data/features/gpu`

New `main()`:
```
Create INPUT_PATH/OUTPUT_PATH dirs
chunk_id = 0
while not _SHUTDOWN:
    files = sorted(INPUT_PATH.glob("*.parquet"))  # excludes .processing/.done
    claimed = False
    for f in files:
        try:
            f.rename(str(f) + ".processing")   # atomic on NFS POSIX
            claimed = True; break
        except (FileNotFoundError, OSError):
            continue                            # another worker got it
    if not claimed:
        time.sleep(0.5); continue

    proc_file = Path(str(f) + ".processing")
    df = pd.read_parquet(proc_file)
    if len(df) == 0:
        proc_file.rename(str(f) + ".done"); continue

    features_cpu, cpu_timing = engineer_features_cpu(df)
    # Try GPU
    try:
        features_gpu, gpu_timing = engineer_features_gpu(df)
        output = features_gpu; gpu_used = 1
    except Exception as exc:
        log.warning("[WARN] GPU failed (%s: %s) — using CPU", type(exc).__name__, exc)
        output = features_cpu; gpu_used = 0; gpu_timing = {k:0.0 for k in cpu_timing}

    # Keep cc_num, merchant, trans_num, is_fraud alongside features for scorer
    for col in ["cc_num", "merchant", "trans_num", "is_fraud", "amt", "category"]:
        if col in df.columns:
            output[col] = df[col].values

    out_file = OUTPUT_PATH / f"features_{chunk_id:06d}.parquet"
    pq.write_table(pa.Table.from_pandas(output), str(out_file))
    proc_file.rename(str(f) + ".done")

    speedup = cpu_timing["total"] / max(gpu_timing.get("total", 0), 1e-6)
    emit_telemetry(stage="prep-gpu", chunk_id=chunk_id, rows=len(df),
                   cpu_time=cpu_timing["total"], gpu_time=gpu_timing.get("total",0),
                   speedup=speedup, gpu_used=gpu_used)
    chunk_id += 1
```

**Remove:** temporal split function (not applicable to streaming chunks)
**Remove:** batch `pd.concat` of all input files
**Keep:** engineer_features_cpu(), engineer_features_gpu(), GPU probe logic (unchanged)
**Keep:** signal handlers (SIGTERM/SIGINT)

### pods/data-prep-cpu/prepare_cpu.py — Same pattern

- `INPUT_PATH=/data/raw/cpu`, `OUTPUT_PATH=/data/features-cpu`
- CPU only (no GPU probe, no cuDF)
- Telemetry: `stage=prep-cpu`

---

## Phase 3: model-build — GraphSAGE + XGBoost

**File:** `pods/model-build/train.py` — significant additions to existing file

### New: Graph Construction

```python
def build_transaction_graph(df: pd.DataFrame):
    """
    Build tri-partite graph: User ↔ Transaction ↔ Merchant
    Returns: (node_features, edge_index, tx_mask, n_users, n_merchants)
    """
    users = {cc: i for i, cc in enumerate(df["cc_num"].unique())}
    n_users = len(users)
    merchants = {m: n_users + i for i, m in enumerate(df["merchant"].unique())}
    n_merchants = len(merchants)
    n_tx = len(df)

    # Node feature matrix: user/merchant rows are zeros, tx rows are tabular features
    feature_cols = [c for c in FEATURE_COLS if c != "is_fraud"]
    tx_features = df[feature_cols].values.astype(np.float32)
    zeros = np.zeros((n_users + n_merchants, len(feature_cols)), dtype=np.float32)
    node_features = np.vstack([zeros, tx_features])  # shape: (n_users+n_merchants+n_tx, n_features)

    # Edge list (bidirectional)
    tx_offset = n_users + n_merchants
    src, dst = [], []
    for i, row in enumerate(df.itertuples()):
        u = users[row.cc_num]
        m = merchants[row.merchant]
        t = tx_offset + i
        src += [u, t, m, t]
        dst += [t, u, t, m]
    edge_index = np.array([src, dst], dtype=np.int64)

    # Mask: which nodes are transactions (for loss computation)
    tx_mask = np.zeros(len(node_features), dtype=bool)
    tx_mask[tx_offset:] = True

    return node_features, edge_index, tx_mask, n_users, n_merchants
```

### New: GraphSAGE Model (PyG)

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEFraud(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int = 16, out_channels: int = 8):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.1, training=self.training)
        return self.conv2(x, edge_index)
```

Training: 16 epochs, Adam lr=0.01, BCEWithLogitsLoss on transaction nodes only (tx_mask).

### Updated XGBoost Training

After GNN training:
1. Extract 8-dim embeddings for all transaction nodes
2. Concatenate with original 21 tabular features → 29-dim input
3. Train XGBoost (keep existing params: max_depth=8, 100 estimators, scale_pos_weight capped at 100)

### New Model Output Format

Write ordering: `config.pbtxt` → `model.py` → `embedding_xgboost.json` → **`state_dict_gnn.pth` last** (Triton polls for this as readiness trigger)

```
/data/models/fraud_gnn_gpu/
    config.pbtxt                    ← written first
    1/
        model.py                    ← Triton Python backend (copied from template)
        state_dict_gnn.pth          ← written last (trigger)
        embedding_xgboost.json      ← XGBoost booster

/data/models/fraud_gnn_cpu/
    config.pbtxt                    ← KIND_CPU variant
    1/
        model.py                    ← identical to GPU version
        state_dict_gnn.pth          ← same weights (CPU inference)
        embedding_xgboost.json      ← same booster
```

`config.pbtxt` (Python backend, GPU):
```
name: "fraud_gnn_gpu"
backend: "python"
input  [{ name:"NODE_FEATURES" data_type:TYPE_FP32 dims:[-1,21] },
        { name:"EDGE_INDEX"    data_type:TYPE_INT64 dims:[2,-1] },
        { name:"FEATURE_MASK"  data_type:TYPE_INT32 dims:[-1] },
        { name:"COMPUTE_SHAP"  data_type:TYPE_BOOL  dims:[1] }]
output [{ name:"PREDICTION"   data_type:TYPE_FP32 dims:[-1,1] },
        { name:"SHAP_VALUES"  data_type:TYPE_FP32 dims:[-1,21] }]
instance_group [{ kind:KIND_GPU }]
```

`model.py` is a file written by train.py to the model repo path (not baked into Docker image):
```python
# Triton Python backend — GNN + XGBoost inference
import triton_python_backend_utils as pb_utils
import torch, xgboost as xgb, numpy as np
from pathlib import Path

class GraphSAGEFraud(torch.nn.Module): ...  # same as training

class TritonPythonModel:
    def initialize(self, args):
        d = Path(args["model_repository"]) / args["model_version"]
        self.gnn = GraphSAGEFraud(21, 16, 8)
        self.gnn.load_state_dict(torch.load(d/"state_dict_gnn.pth", map_location="cpu"))
        self.gnn.eval()
        self.xgb = xgb.Booster(); self.xgb.load_model(str(d/"embedding_xgboost.json"))

    def execute(self, requests):
        for request in requests:
            x = torch.tensor(pb_utils.get_input_tensor_by_name(request,"NODE_FEATURES").as_numpy())
            ei = torch.tensor(pb_utils.get_input_tensor_by_name(request,"EDGE_INDEX").as_numpy())
            compute_shap = pb_utils.get_input_tensor_by_name(request,"COMPUTE_SHAP").as_numpy()[0]
            n_tx = int((ei < ei.max()).sum() / 2)  # infer n_tx from edge structure

            with torch.no_grad():
                embeddings = self.gnn(x, ei)[-n_tx:].numpy()
            tabular = x[-n_tx:].numpy()
            combined = np.concatenate([tabular, embeddings], axis=1)
            dm = xgb.DMatrix(combined)
            probs = self.xgb.predict(dm).reshape(-1,1).astype(np.float32)
            shap = self.xgb.predict(dm, pred_contribs=True)[:,:21].astype(np.float32) \
                   if compute_shap else np.zeros((n_tx,21), dtype=np.float32)
            yield pb_utils.InferenceResponse([
                pb_utils.Tensor("PREDICTION",  probs),
                pb_utils.Tensor("SHAP_VALUES", shap),
            ])
```

**Decision note (review with user):** `n_tx` inference from edge structure is fragile. Alternative: pass n_tx as a 4th input tensor. For now using edge-based inference since it avoids changing Triton input schema.

---

## Phase 4: Triton Unified Pod

**New directory:** `pods/triton/` (replaces `pods/inference/` and `pods/inference-cpu/`)

### pods/triton/Dockerfile
Base: `nvcr.io/nvidia/tritonserver:25.04-py3` (matches NVIDIA blueprint)

Additional (via conda, mirrors `/tmp/nvidia-fraud-det/triton/Dockerfile`):
- python=3.12
- pytorch=2.7.0=*cuda126*
- py-xgboost=3.0.2=*cuda128*
- cupy=13.4.1
- torch-geometric==2.6.1
- captum==0.7.0

Note: `model.py` lives on NFS model-repo-pvc, NOT in the Docker image. Triton loads it via Python backend from the model repository path.

### pods/triton/start.sh
```bash
#!/bin/bash
set -e
MODEL_REPO="${MODEL_REPO:-/data/models}"
echo "[INFO] Waiting for GNN models (fraud_gnn_gpu + fraud_gnn_cpu)..."
until [ -s "${MODEL_REPO}/fraud_gnn_gpu/1/state_dict_gnn.pth" ]  && \
      [ -f "${MODEL_REPO}/fraud_gnn_gpu/config.pbtxt" ]           && \
      [ -s "${MODEL_REPO}/fraud_gnn_cpu/1/state_dict_gnn.pth" ]   && \
      [ -f "${MODEL_REPO}/fraud_gnn_cpu/config.pbtxt" ]; do
    echo "[INFO] Models not ready, waiting 10s..."
    sleep 10
done
echo "[INFO] Both models ready. Starting Triton..."
exec tritonserver \
    --model-repository="${MODEL_REPO}" \
    --model-control-mode=poll \
    --repository-poll-secs=30 \
    --strict-model-config=false \
    --log-verbose=0 \
    --http-port=8000 --grpc-port=8001 --metrics-port=8002
```

---

## Phase 5: New Scoring Pods

### pods/scoring-gpu/scorer.py (NEW)

```python
FEATURES_PATH = Path(os.environ.get("FEATURES_PATH", "/data/features/gpu"))
SCORES_PATH   = Path(os.environ.get("SCORES_PATH",   "/data/features/scores"))
TRITON_URL    = os.environ.get("TRITON_URL",    "triton:8000")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "fraud_gnn_gpu")
WINDOW_CHUNKS = int(os.environ.get("WINDOW_CHUNKS", "50"))
FEATURE_COLS  = [21 tabular cols, same as prepare.py]

class WindowedGraph:
    """Maintains sliding-window tri-partite graph across last WINDOW_CHUNKS chunks."""
    def __init__(self, max_chunks):
        self.max_chunks = max_chunks
        self.chunks = deque()          # deque of DataFrames
        self.user_ids = {}             # cc_num → persistent node_id
        self.merch_ids = {}            # merchant → persistent node_id
        self._next_user_id = 0
        self._next_merch_id = 0

    def add_chunk(self, df):
        self.chunks.append(df)
        # Register new users/merchants
        for cc in df["cc_num"].unique():
            if cc not in self.user_ids:
                self.user_ids[cc] = self._next_user_id; self._next_user_id += 1
        for m in df["merchant"].unique():
            if m not in self.merch_ids:
                self.merch_ids[m] = self._next_merch_id; self._next_merch_id += 1
        # Prune oldest chunk if over limit
        if len(self.chunks) > self.max_chunks:
            self.chunks.popleft()
            # Rebuild user/merch maps from remaining chunks
            self._rebuild_maps()

    def build_inference_graph(self, new_df):
        """
        Returns (node_features, edge_index) for full window + new_df.
        Transaction nodes for new_df are at the END of the node list.
        """
        all_df = pd.concat(list(self.chunks) + [new_df], ignore_index=True)
        # Build graph same as train.py build_transaction_graph()
        # but using persistent user_ids / merch_ids
        ...
        return node_features, edge_index, n_tx_new

def score_chunk(df, graph, triton_client, model_name):
    graph.add_chunk(df)
    node_features, edge_index, n_tx_new = graph.build_inference_graph(df)
    # Call Triton
    inputs = [
        httpclient.InferInput("NODE_FEATURES", node_features.shape, "FP32"),
        httpclient.InferInput("EDGE_INDEX",    edge_index.shape,    "INT64"),
        httpclient.InferInput("FEATURE_MASK",  np.zeros(node_features.shape[1], dtype=np.int32).shape, "INT32"),
        httpclient.InferInput("COMPUTE_SHAP",  (1,), "BOOL"),
    ]
    inputs[0].set_data_from_numpy(node_features)
    inputs[1].set_data_from_numpy(edge_index)
    inputs[2].set_data_from_numpy(np.zeros(node_features.shape[1], dtype=np.int32))
    inputs[3].set_data_from_numpy(np.array([False]))
    response = triton_client.infer(model_name, inputs=inputs,
                                   outputs=[httpclient.InferRequestedOutput("PREDICTION")])
    probs = response.as_numpy("PREDICTION")[-n_tx_new:].flatten()
    return probs

def main():
    FEATURES_PATH.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.mkdir(parents=True, exist_ok=True)
    graph = WindowedGraph(WINDOW_CHUNKS)
    chunk_id = 0
    with httpclient.InferenceServerClient(TRITON_URL) as client:
        while not _SHUTDOWN:
            # File-queue claim
            files = sorted(FEATURES_PATH.glob("*.parquet"))
            claimed_file = None
            for f in files:
                try: f.rename(str(f)+".processing"); claimed_file=f; break
                except (FileNotFoundError, OSError): continue
            if not claimed_file: time.sleep(0.5); continue

            proc = Path(str(claimed_file)+".processing")
            df = pd.read_parquet(proc)
            t0 = time.perf_counter()
            probs = score_chunk(df, graph, client, MODEL_NAME)
            latency_ms = (time.perf_counter()-t0)*1000

            result = df[["trans_num","cc_num","merchant","amt","category","is_fraud"]].copy()
            result["fraud_score"] = probs
            result["scored_at"] = time.time()
            out_file = SCORES_PATH / f"scores_{chunk_id:06d}.parquet"
            pq.write_table(pa.Table.from_pandas(result), str(out_file))
            proc.rename(str(claimed_file)+".done")

            fraud_rate = float((probs > 0.5).mean())
            emit_telemetry(stage="scoring-gpu", chunk_id=chunk_id,
                           rows=len(df), latency_ms=latency_ms, fraud_rate=fraud_rate)
            chunk_id += 1
```

### pods/scoring-cpu/scorer.py
Identical to scoring-gpu except:
- `FEATURES_PATH=/data/features-cpu`, `SCORES_PATH=/data/features-cpu/scores`
- `MODEL_NAME=fraud_gnn_cpu`
- Telemetry: `stage=scoring-cpu`

### Dockerfiles

`pods/scoring-gpu/Dockerfile`:
```dockerfile
FROM python:3.11-slim
RUN pip install pandas pyarrow tritonclient[http] torch numpy torch-geometric
COPY pods/scoring-gpu/scorer.py /app/scorer.py
WORKDIR /app
CMD ["python", "scorer.py"]
```

`pods/scoring-cpu/Dockerfile`: identical (no GPU libs — Triton handles inference device)

---

## Phase 6: Backend

### pods/backend/pipeline.py — Full Rewrite

Remove all Job creation/deletion/wait logic. Replace with Deployment scaling.

```python
NORMAL_REPLICAS = {
    "data-gather":   1,
    "data-prep-gpu": 2,
    "data-prep-cpu": 2,
    "scoring-gpu":   2,
    "scoring-cpu":   2,
    "triton":        1,
}
STRESS_REPLICAS = {
    "data-gather":   1,  # same; rate governed by env vars
    "data-prep-gpu": 4,
    "data-prep-cpu": 4,
    "scoring-gpu":   2,  # graph size does the work
    "scoring-cpu":   2,
    "triton":        1,
}

def _scale(apps_v1, name, replicas):
    apps_v1.patch_namespaced_deployment_scale(
        name=name, namespace=NAMESPACE,
        body={"spec": {"replicas": replicas}})

def start_pipeline(overrides=None) -> dict:
    _, apps_v1, _ = _k8s()
    for dep, n in NORMAL_REPLICAS.items():
        _scale(apps_v1, dep, n)
    return {"status": "started"}

def stop_pipeline() -> dict:
    _, apps_v1, _ = _k8s()
    for dep in NORMAL_REPLICAS:
        _scale(apps_v1, dep, 0)
    return {"status": "stopped"}

def write_stress_config(stress_on: bool) -> None:
    _, apps_v1, _ = _k8s()
    replicas = STRESS_REPLICAS if stress_on else NORMAL_REPLICAS
    for dep, n in replicas.items():
        _scale(apps_v1, dep, n)

def get_service_states() -> dict:
    _, apps_v1, _ = _k8s()
    states = {}
    for dep in list(NORMAL_REPLICAS.keys()):
        try:
            d = apps_v1.read_namespaced_deployment(name=dep, namespace=NAMESPACE)
            ready = d.status.ready_replicas or 0
            desired = d.spec.replicas or 0
            states[dep] = "Ready" if ready == desired > 0 else ("Scaling" if desired > 0 else "Stopped")
        except ApiException:
            states[dep] = "NotFound"
    return states
```

### pods/backend/backend.py — Updates

- `start_pipeline`: remove `asyncio.create_task(_run_pipeline_task({}))` — `pl.start_pipeline()` is now instant (just API calls)
- `reset_pipeline`: pass all new paths including `/data/raw/gpu`, `/data/raw/cpu`, `/data/features/gpu`, `/data/features-cpu`, `/data/features/scores`, `/data/features-cpu/scores`
- **Clear telemetry on reset**: add `(MODEL_REPO_PATH / "last_telemetry.json").unlink(missing_ok=True)` + `state.reset()`
- Keep stress endpoints as-is (already call `pl.write_stress_config` via executor)
- Remove `_run_pipeline_task()` function
- Add env var `MODEL_REPO_PATH = Path(os.environ.get("MODEL_REPO_PATH", "/data/models"))`

### pods/backend/metrics.py — Updates

**Telemetry collection:**
- Replace Job pod log scraping with Deployment pod log scraping for: data-gather, data-prep-gpu, data-prep-cpu, scoring-gpu, scoring-cpu
- Helper `_get_deployment_pod_logs(dep_name)` → finds pods for deployment via label selector `app={dep_name}`

**Queue depth (new):**
```python
def _collect_queue_depth(self) -> dict:
    depths = {}
    for name, path in [
        ("raw_gpu",      RAW_PATH / "gpu"),
        ("raw_cpu",      RAW_PATH / "cpu"),
        ("features_gpu", FEATURES_PATH / "gpu"),
        ("features_cpu", FEATURES_CPU_PATH),
    ]:
        if path.exists():
            depths[name] = {
                "pending":    len(list(path.glob("*.parquet"))),
                "processing": len(list(path.glob("*.parquet.processing"))),
                "done":       len(list(path.glob("*.parquet.done"))),
            }
    return depths
```

**Fraud metrics from scores (new):**
```python
def _collect_fraud_metrics(self) -> dict:
    # Read last N score files, compute rolling fraud rate and recent alerts
    score_files = sorted((FEATURES_PATH/"scores").glob("*.parquet"))[-10:]
    if not score_files: return {}
    df = pd.concat([pd.read_parquet(f) for f in score_files])
    alerts = df[df["fraud_score"] > 0.8].sort_values("scored_at", ascending=False).head(20)
    return {
        "fraud_rate_pct": float((df["fraud_score"] > 0.5).mean() * 100),
        "total_scored": len(df),
        "recent_alerts": alerts[["trans_num","merchant","amt","category","fraud_score"]].to_dict("records"),
    }
```

**Updated WebSocket payload:**
```json
{
  "pipeline": {
    "gather":      { "rows_per_sec": N, "workers": N, "fraud_rate": N },
    "prep_gpu":    { "chunks_per_sec": N, "rows_per_sec": N, "speedup": N, "gpu_used": 0/1 },
    "prep_cpu":    { "chunks_per_sec": N, "rows_per_sec": N },
    "scoring_gpu": { "chunks_scored": N, "fraud_rate": N, "latency_ms": N },
    "scoring_cpu": { "chunks_scored": N, "fraud_rate": N, "latency_ms": N }
  },
  "queue": {
    "raw_gpu":      { "pending": N, "processing": N, "done": N },
    "raw_cpu":      { "pending": N, "processing": N, "done": N },
    "features_gpu": { "pending": N, "processing": N, "done": N },
    "features_cpu": { "pending": N, "processing": N, "done": N }
  },
  "fraud": {
    "fraud_rate_pct": N,
    "total_scored": N,
    "recent_alerts": [...]
  },
  "system":     { "cpu_percent": N, "ram_percent": N },
  "gpu":        { "gpu_0_util_pct": N, "gpu_0_mem_pct": N },
  "flashblade": { "read_latency_ms": N, "write_latency_ms": N }
}
```

---

## Phase 7: K8s Manifests

### k8s/deployments.yaml — Full Update

**Remove:** `inference`, `inference-cpu` deployments

**Add/Update:** (all start at replicas=0, scaled by backend on Start)

```yaml
# data-gather: NEW Deployment (was Job)
name: data-gather, replicas: 0
env: OUTPUT_PATH_GPU=/data/raw/gpu, OUTPUT_PATH_CPU=/data/raw/cpu, RUN_MODE=continuous
    TARGET_ROWS_PER_SEC=10000, NUM_WORKERS=2, CHUNK_SIZE=10000, KAGGLE_SEED_PATH=...
volumes: raw-data-pvc → /data/raw (rw)

# data-prep-gpu: NEW Deployment (was Job)
name: data-prep-gpu, replicas: 0
env: INPUT_PATH=/data/raw/gpu, OUTPUT_PATH=/data/features/gpu
volumes: raw-data-pvc → /data/raw (ro), features-data-pvc → /data/features (rw)
resources: nvidia.com/gpu: "1"

# data-prep-cpu: NEW Deployment (was Job)
name: data-prep-cpu, replicas: 0
env: INPUT_PATH=/data/raw/cpu, OUTPUT_PATH=/data/features-cpu
volumes: raw-data-pvc → /data/raw (ro), features-cpu-data-pvc → /data/features-cpu (rw)

# triton: NEW (replaces inference + inference-cpu)
name: triton, replicas: 0
image: fraud-det-v31/triton:latest
env: MODEL_REPO=/data/models
volumes: model-repo-pvc → /data/models (ro)
resources: nvidia.com/gpu: "1"
ports: 8000, 8001, 8002

# scoring-gpu: NEW
name: scoring-gpu, replicas: 0
env: FEATURES_PATH=/data/features/gpu, SCORES_PATH=/data/features/scores,
     TRITON_URL=triton:8000, MODEL_NAME=fraud_gnn_gpu, WINDOW_CHUNKS=50
volumes: features-data-pvc → /data/features (rw)

# scoring-cpu: NEW
name: scoring-cpu, replicas: 0
env: FEATURES_PATH=/data/features-cpu, SCORES_PATH=/data/features-cpu/scores,
     TRITON_URL=triton:8000, MODEL_NAME=fraud_gnn_cpu, WINDOW_CHUNKS=50
volumes: features-cpu-data-pvc → /data/features-cpu (rw)

# backend: update env + add MODEL_REPO_PATH
env: add OUTPUT_PATH_GPU=/data/raw/gpu, OUTPUT_PATH_CPU=/data/raw/cpu,
         FEATURES_GPU_PATH=/data/features/gpu,
         SCORES_GPU_PATH=/data/features/scores,
         SCORES_CPU_PATH=/data/features-cpu/scores,
         MODEL_REPO_PATH=/data/models
```

**Add:** ClusterIP Service for `triton` (ports 8000/8001/8002) so scoring pods can reach it.

### k8s/jobs/ — Changes
- **Delete:** `data-gather.yaml`, `data-prep.yaml`, `data-prep-cpu.yaml`
- **Keep + Update:** `model-build.yaml`:
  - Add GPU resource (nvidia.com/gpu: "1")
  - Update env: `INPUT_PATH=/data/features` (reads GPU features for graph training)
  - Add volume: features-data-pvc (ro) → /data/features
  - Update image to include PyG + PyTorch

---

## Phase 8: Dashboard

### Tab 1 (Business) — Updates
- **Pipeline funnel** → replace 5-cell batch display with **2-lane live throughput cards**:
  - GPU Lane: `↓ Gen: {rows/s}` → `↓ Prep: {rows/s} ({speedup}x)` → `↓ Score: {scores/s}`
  - CPU Lane: `↓ Gen: {rows/s}` → `↓ Prep: {rows/s}` → `↓ Score: {scores/s}`
  - Queue depth per step shown as small badge `[{pending} pending]`
- **Alerts table**: replace synthetic random data with `data.fraud.recent_alerts` (real scored transactions)
- **KPI cards**: fraud exposure + transactions sourced from `data.fraud.*`

### Tab 2 (Infrastructure) — Updates
- **Combined 4-line chart**: unchanged
- **Pipeline stages row** → replace batch stage text with **queue depth progress bars**:
  - `raw/gpu [████░░░░] 42 pending` vs `raw/cpu [███░░░░░] 38 pending`
- **Speedup table**: keep — sourced from `data.pipeline.prep_gpu.speedup`

### Tab 3 (Model/SHAP) — Additions
- **Pipeline flow diagram** (new SVG/HTML): `[GEN] → [NFS] → [PREP GPU] → [NFS] → [GNN] → [XGB] → [TRITON] → [SCORES]` with live throughput on each arrow
- **Fraud score histogram** (new Chart.js bar): distribution of last 1000 fraud_score values in 10 buckets (0.0-0.1, 0.1-0.2 ... 0.9-1.0)
- **SHAP bar chart**: keep (already working, sourced from /api/metrics/shap)
- **Model metrics KPIs**: keep F1/AUC-PR

---

## Decisions Made During Implementation (for review)

1. **n_tx inference in model.py**: Transaction count passed implicitly via graph structure rather than as explicit Triton input. May want to add explicit `N_NEW_TX: INT32` input for robustness — flag for review.
2. **data-gather still writes to single raw-data-pvc** with gpu/ and cpu/ sub-folders. If GPU prep consistently backfills slower, raw/gpu will have larger queue. Could separate PVCs in future if needed.
3. **scoring pods connect to `triton:8000`** via K8s Service DNS. Triton startup takes ~60s after model weights appear. Scoring pods retry connection with backoff (5s, up to 10 retries).
4. **model.py written to NFS** by train.py rather than baked into the Docker image. This means the Python backend code lives on FlashBlade, not in the container — intentional (avoids image rebuild when tuning inference logic).
5. **WindowedGraph._rebuild_maps()** rebuilds user/merchant ID maps after oldest chunk is pruned. IDs are NOT stable across window slides (node index changes). This is fine for inference but means embeddings from different time points are not directly comparable.
6. **data-prepare no longer does temporal split** — each chunk produces a flat feature parquet. model-build reads the accumulated features from /data/features/gpu/ (most recent N files) for graph construction and training.
7. **Stress mode only scales prep workers** (2→4), not scoring workers. If scoring becomes a bottleneck under stress, increase STRESS_REPLICAS["scoring-gpu/cpu"] to 3.

---

## Build Order

```bash
REGISTRY=10.23.181.247:5000/fraud-det-v31

# 1. Data pipeline pods
docker build -t $REGISTRY/data-gather:latest   -f pods/data-gather/Dockerfile .
docker build -t $REGISTRY/data-prep:latest     -f pods/data-prep/Dockerfile .
docker build -t $REGISTRY/data-prep-cpu:latest -f pods/data-prep-cpu/Dockerfile .

# 2. Model build (new GNN training)
docker build -t $REGISTRY/model-build:latest   -f pods/model-build/Dockerfile .

# 3. Triton unified pod (largest image ~8GB, build last)
docker build -t $REGISTRY/triton:latest        -f pods/triton/Dockerfile .

# 4. Scoring pods
docker build -t $REGISTRY/scoring-gpu:latest   -f pods/scoring-gpu/Dockerfile .
docker build -t $REGISTRY/scoring-cpu:latest   -f pods/scoring-cpu/Dockerfile .

# 5. Backend
docker build -t $REGISTRY/backend:latest       -f pods/backend/Dockerfile .

# After all builds:
# Apply updated K8s manifests
kubectl apply -f k8s/

# Run model-build once to produce GNN artifacts
kubectl -n fraud-det-v31 create job model-build-init --from=cronjob/model-build  # or apply job YAML directly
```

---

## Critical Files

| File | Change |
|---|---|
| `pods/data-gather/gather.py` | Add dual output (OUTPUT_PATH_GPU + OUTPUT_PATH_CPU) |
| `pods/data-prep/prepare.py` | Rewrite main(): continuous file-queue loop, no temporal split |
| `pods/data-prep-cpu/prepare_cpu.py` | Same |
| `pods/model-build/train.py` | Add GraphSAGE, new model format, write model.py to NFS |
| `pods/triton/` (NEW dir) | Dockerfile + start.sh |
| `pods/triton/Dockerfile` | Based on nvcr.io/nvidia/tritonserver:25.04-py3 + PyG + XGBoost |
| `pods/scoring-gpu/scorer.py` (NEW) | WindowedGraph + file-queue + Triton calls |
| `pods/scoring-cpu/scorer.py` (NEW) | Same for CPU model |
| `pods/scoring-gpu/Dockerfile` (NEW) | python:3.11-slim + tritonclient |
| `pods/scoring-cpu/Dockerfile` (NEW) | Same |
| `pods/backend/pipeline.py` | Full rewrite: Deployment scaling |
| `pods/backend/backend.py` | Remove run_pipeline_task, update reset paths |
| `pods/backend/metrics.py` | New stages, queue depth, fraud metrics |
| `pods/backend/static/dashboard.html` | Tab 1 funnel, Tab 3 additions |
| `k8s/deployments.yaml` | New deployments, remove inference/inference-cpu |
| `k8s/jobs/model-build.yaml` | Update env + volumes |
| `k8s/` (delete) | `data-gather.yaml`, `data-prep.yaml`, `data-prep-cpu.yaml` |

---

## Verification

1. **model-build Job** → verify `/data/models/fraud_gnn_gpu/1/state_dict_gnn.pth` written non-empty
2. **Start pipeline** → verify 6 Deployments scale up, dashboard shows non-zero TX/s within 30s
3. **Queue depth** → raw/gpu and raw/cpu show pending chunks growing then stabilizing as prep keeps up
4. **Prep speedup** → Tab 2 speedup table shows GPU prep Nx faster than CPU
5. **Triton ready** → scoring pods connect after ~60s, fraud_rate_pct appears in dashboard
6. **Fraud alert feed** → real transactions with fraud_score > 0.8 appear in Tab 1 + Tab 3
7. **Stress mode** → data-prep-gpu/cpu scale to 4 replicas, GPU util spikes >60%, FlashBlade latency stays 2-3ms
8. **Stop** → all Deployments scale to 0, TX/s drops to 0 on dashboard
9. **Reset** → queue dirs cleared, last_telemetry.json deleted, all dashboard metrics zero
