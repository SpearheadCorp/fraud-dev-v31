"""
Pod: model-build (offline, run once before demo via kubectl create job)
Loads accumulated feature chunks from INPUT_PATH, builds tri-partite graph,
trains GraphSAGE (PyTorch Geometric), extracts 8-dim embeddings, trains
XGBoost on 21 tabular + 8 GNN = 29 features. Writes Triton Python backend
model repository for both GPU and CPU Triton model instances.
"""
import os
import sys
import json
import time
import logging
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

_SHUTDOWN = False


def _handle_signal(signum, frame):
    global _SHUTDOWN
    log.info("[INFO] Signal %s received — shutting down", signum)
    _SHUTDOWN = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH  = Path(os.environ.get("INPUT_PATH",  "/data/features/gpu"))
MODEL_REPO  = Path(os.environ.get("MODEL_REPO",  "/data/models"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES",  "500000"))
MAX_FILES   = int(os.environ.get("MAX_FILES",    "200"))    # most-recent N chunk files
GNN_EPOCHS  = int(os.environ.get("GNN_EPOCHS",  "16"))
GNN_HIDDEN  = int(os.environ.get("GNN_HIDDEN",  "16"))
GNN_OUT     = int(os.environ.get("GNN_OUT",     "8"))
GNN_LR      = float(os.environ.get("GNN_LR",    "0.01"))
GNN_MAX_TX  = int(os.environ.get("GNN_MAX_TX",  "100000"))  # cap graph size for training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COL = "is_fraud"
FEATURE_COLS = [
    "amt_log", "amt_scaled", "hour_of_day", "day_of_week", "is_weekend",
    "is_night", "distance_km", "category_encoded", "state_encoded",
    "gender_encoded", "city_pop_log", "zip_region", "amt", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long", "merch_zipcode", "zip",
]
N_TABULAR = len(FEATURE_COLS)  # 21

XGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 10,
    "verbosity": 0,
}

# ---------------------------------------------------------------------------
# GraphSAGE model definition
# ---------------------------------------------------------------------------

class GraphSAGEFraud(torch.nn.Module):
    """2-layer GraphSAGE producing GNN_OUT-dim node embeddings."""
    def __init__(self, in_channels: int, hidden: int = 16, out_channels: int = 8):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.1, training=self.training)
        return self.conv2(x, edge_index)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_transaction_graph(df: pd.DataFrame) -> tuple:
    """
    Build tri-partite graph: User <-> Transaction <-> Merchant.
    User/Merchant node features are zeros; Transaction nodes carry tabular features.
    Returns (node_features, edge_index, tx_mask).
    """
    users = {cc: i for i, cc in enumerate(df["cc_num"].astype(str).unique())}
    n_users = len(users)
    merchants = {m: n_users + i for i, m in enumerate(df["merchant"].astype(str).unique())}
    n_merchants = len(merchants)
    n_tx = len(df)
    tx_offset = n_users + n_merchants

    # Node feature matrix: zeros for user/merchant, tabular features for transactions
    available = [c for c in FEATURE_COLS if c in df.columns]
    tx_features = df[available].fillna(0.0).values.astype(np.float32)
    # Pad to N_TABULAR if any columns missing
    if len(available) < N_TABULAR:
        pad = np.zeros((n_tx, N_TABULAR - len(available)), dtype=np.float32)
        tx_features = np.hstack([tx_features, pad])
    zeros = np.zeros((n_users + n_merchants, N_TABULAR), dtype=np.float32)
    node_features = np.vstack([zeros, tx_features])

    # Build edge index vectorized (bidirectional: user↔tx, merchant↔tx)
    users_arr = df["cc_num"].astype(str).map(users).values.astype(np.int64)
    merch_arr = df["merchant"].astype(str).map(merchants).values.astype(np.int64)
    tx_ids    = np.arange(tx_offset, tx_offset + n_tx, dtype=np.int64)

    src = np.concatenate([users_arr, tx_ids, merch_arr, tx_ids])
    dst = np.concatenate([tx_ids, users_arr, tx_ids, merch_arr])
    edge_index = np.vstack([src, dst])

    tx_mask = np.zeros(n_users + n_merchants + n_tx, dtype=bool)
    tx_mask[tx_offset:] = True

    return node_features, edge_index, tx_mask


# ---------------------------------------------------------------------------
# GNN training
# ---------------------------------------------------------------------------

def train_gnn(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    tx_mask: np.ndarray,
    y_tx: np.ndarray,
    pos_weight: float,
) -> GraphSAGEFraud:
    """Train GraphSAGE with a temporary linear head (head discarded after training)."""
    x  = torch.tensor(node_features, dtype=torch.float32, device=DEVICE)
    ei = torch.tensor(edge_index,    dtype=torch.long,    device=DEVICE)
    y  = torch.tensor(y_tx,          dtype=torch.float32, device=DEVICE)
    tx_idx = torch.where(torch.tensor(tx_mask))[0].to(DEVICE)

    model = GraphSAGEFraud(N_TABULAR, GNN_HIDDEN, GNN_OUT).to(DEVICE)
    head  = torch.nn.Linear(GNN_OUT, 1).to(DEVICE)

    pw = torch.tensor([min(pos_weight, 100.0)], dtype=torch.float32, device=DEVICE)
    loss_fn   = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=GNN_LR
    )

    model.train()
    head.train()
    t0 = time.perf_counter()
    for epoch in range(GNN_EPOCHS):
        optimizer.zero_grad()
        emb    = model(x, ei)
        logits = head(emb[tx_idx]).squeeze(1)
        loss   = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 4 == 0:
            log.info("[INFO] GNN epoch %d/%d loss=%.4f", epoch + 1, GNN_EPOCHS, loss.item())

    train_time = time.perf_counter() - t0
    log.info("[INFO] GNN training complete: %.1fs, device=%s", train_time, DEVICE)
    model.eval()
    return model


def extract_gnn_embeddings(
    model: GraphSAGEFraud,
    node_features: np.ndarray,
    edge_index: np.ndarray,
    tx_mask: np.ndarray,
) -> np.ndarray:
    """Extract GNN_OUT-dim embeddings for transaction nodes only."""
    with torch.no_grad():
        x  = torch.tensor(node_features, dtype=torch.float32, device=DEVICE)
        ei = torch.tensor(edge_index,    dtype=torch.long,    device=DEVICE)
        all_emb = model(x, ei)
    return all_emb[tx_mask].cpu().numpy()


# ---------------------------------------------------------------------------
# Triton Python backend model.py template
# ---------------------------------------------------------------------------

_MODEL_PY_TEMPLATE = '''\
# AUTO-GENERATED by train.py — Triton Python backend for GNN + XGBoost fraud detection
import triton_python_backend_utils as pb_utils
import torch
import torch.nn.functional as F
import xgboost as xgb
import numpy as np
from torch_geometric.nn import SAGEConv
from pathlib import Path

N_TABULAR  = __N_TABULAR__
GNN_HIDDEN = __GNN_HIDDEN__
GNN_OUT    = __GNN_OUT__


class GraphSAGEFraud(torch.nn.Module):
    def __init__(self, in_channels, hidden=16, out_channels=8):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.1, training=self.training)
        return self.conv2(x, edge_index)


class TritonPythonModel:
    def initialize(self, args):
        model_dir = Path(args["model_repository"]) / args["model_version"]
        self.n_tabular = N_TABULAR
        self.gnn = GraphSAGEFraud(N_TABULAR, GNN_HIDDEN, GNN_OUT)
        state = torch.load(str(model_dir / "state_dict_gnn.pth"), map_location="cpu",
                           weights_only=True)
        self.gnn.load_state_dict(state)
        self.gnn.eval()
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / "embedding_xgboost.json"))

    def execute(self, requests):
        responses = []
        for request in requests:
            node_features = pb_utils.get_input_tensor_by_name(request, "NODE_FEATURES").as_numpy()
            edge_index    = pb_utils.get_input_tensor_by_name(request, "EDGE_INDEX").as_numpy()
            feature_mask  = pb_utils.get_input_tensor_by_name(request, "FEATURE_MASK").as_numpy().astype(bool)
            compute_shap  = bool(pb_utils.get_input_tensor_by_name(request, "COMPUTE_SHAP").as_numpy()[0])

            x  = torch.tensor(node_features, dtype=torch.float32)
            ei = torch.tensor(edge_index,    dtype=torch.long)
            n_tx = int(feature_mask.sum())

            with torch.no_grad():
                all_emb = self.gnn(x, ei)
            emb     = all_emb[feature_mask].numpy()
            tabular = node_features[feature_mask]
            combined = np.concatenate([tabular, emb], axis=1).astype(np.float32)

            dm    = xgb.DMatrix(combined)
            probs = self.booster.predict(dm).reshape(-1, 1).astype(np.float32)
            if compute_shap:
                shap = self.booster.predict(dm, pred_contribs=True)[:, :self.n_tabular].astype(np.float32)
            else:
                shap = np.zeros((n_tx, self.n_tabular), dtype=np.float32)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("PREDICTION",  probs),
                pb_utils.Tensor("SHAP_VALUES", shap),
            ]))
        return responses
'''


# ---------------------------------------------------------------------------
# Triton Python backend config.pbtxt
# ---------------------------------------------------------------------------

def write_python_backend_config(model_dir: Path, model_name: str, kind: str) -> None:
    """Write Triton Python backend config.pbtxt."""
    config = f"""name: "{model_name}"
backend: "python"
max_batch_size: 0
input [
  {{ name: "NODE_FEATURES"  data_type: TYPE_FP32  dims: [-1, {N_TABULAR}] }},
  {{ name: "EDGE_INDEX"     data_type: TYPE_INT64  dims: [2, -1] }},
  {{ name: "FEATURE_MASK"   data_type: TYPE_INT32  dims: [-1] }},
  {{ name: "COMPUTE_SHAP"   data_type: TYPE_BOOL   dims: [1] }}
]
output [
  {{ name: "PREDICTION"     data_type: TYPE_FP32   dims: [-1, 1] }},
  {{ name: "SHAP_VALUES"    data_type: TYPE_FP32   dims: [-1, {N_TABULAR}] }}
]
instance_group [{{ kind: {kind} count: 1 }}]
"""
    (model_dir / "config.pbtxt").write_text(config)


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    device: str,
    scale_pos_weight: float,
) -> tuple:
    """Train XGBoost classifier, return (booster, train_time_s)."""
    params = dict(XGB_PARAMS)
    params["device"] = device
    params["scale_pos_weight"] = scale_pos_weight

    model = xgb.XGBClassifier(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.perf_counter() - t0
    log.info("[INFO] XGBoost %s: %.2fs (%d trees)", device, train_time, model.best_iteration + 1)
    return model, train_time


# ---------------------------------------------------------------------------
# Evaluation + SHAP
# ---------------------------------------------------------------------------

def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_test, y_prob)),
        "auc_pr":    float(average_precision_score(y_test, y_prob)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": 0.5,
    }


def compute_shap(model: xgb.XGBClassifier, X_test: np.ndarray, feature_names: list) -> dict:
    """SHAP via XGBoost native pred_contribs (no extra library)."""
    booster  = model.get_booster()
    dmat     = xgb.DMatrix(X_test, feature_names=feature_names)
    shap_raw = booster.predict(dmat, pred_contribs=True)[:, :-1]  # drop bias
    # Only report SHAP for the first N_TABULAR features (tabular, not GNN embeddings)
    shap_tab       = shap_raw[:, :N_TABULAR]
    tab_names      = feature_names[:N_TABULAR]
    mean_abs_shap  = np.abs(shap_tab).mean(axis=0)
    importance     = dict(zip(tab_names, mean_abs_shap.tolist()))
    top_features   = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "top_features":       top_features,
        "shap_values_sample": shap_tab[:100].tolist(),
        "feature_names":      tab_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_feature_chunks() -> pd.DataFrame:
    """Load last MAX_FILES feature chunk parquets and return concatenated DataFrame."""
    chunk_files = sorted(
        f for f in INPUT_PATH.glob("*.parquet")
        if not f.name.endswith((".processing", ".done"))
    )
    if not chunk_files:
        log.error("[ERROR] No feature chunk files in %s", INPUT_PATH)
        sys.exit(1)

    selected = chunk_files[-MAX_FILES:]  # most recent N
    log.info("[INFO] Loading %d chunk files from %s", len(selected), INPUT_PATH)
    dfs = [pd.read_parquet(str(f)) for f in selected]
    df  = pd.concat(dfs, ignore_index=True)
    log.info("[INFO] Loaded %d rows total", len(df))
    return df


def _temporal_split(df: pd.DataFrame) -> tuple:
    df = df.sort_values("unix_time").reset_index(drop=True)
    n  = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_val].copy(), df.iloc[n_val:].copy()


def main() -> None:
    MODEL_REPO.mkdir(parents=True, exist_ok=True)

    # --- Load feature chunks ---
    df = _load_feature_chunks()

    # Drop rows missing critical columns
    required = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]
    before = len(df)
    df = df.dropna(subset=[c for c in required if c in df.columns])
    if len(df) < before:
        log.warning("[WARN] Dropped %d rows with NaN", before - len(df))

    if len(df) < 1000:
        log.error("[ERROR] Only %d rows after cleaning — aborting", len(df))
        sys.exit(1)

    # Cap total rows
    if len(df) > MAX_SAMPLES:
        log.info("[INFO] Capping %d → %d rows", len(df), MAX_SAMPLES)
        fraud = df[df[LABEL_COL] == 1]
        legit = df[df[LABEL_COL] == 0]
        f_cap = int(MAX_SAMPLES * len(fraud) / len(df))
        l_cap = MAX_SAMPLES - f_cap
        df = pd.concat([
            fraud.sample(min(f_cap, len(fraud)), random_state=42),
            legit.sample(min(l_cap, len(legit)), random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Temporal split (70/15/15)
    train_df, val_df, test_df = _temporal_split(df)
    log.info("[INFO] Split: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    if len(available_features) < N_TABULAR:
        log.warning("[WARN] Missing tabular features: %s",
                    set(FEATURE_COLS) - set(available_features))

    # Tabular arrays (21-dim)
    X_train_tab = train_df[available_features].fillna(0.0).values.astype(np.float32)
    y_train     = train_df[LABEL_COL].values.astype(np.int8)
    X_val_tab   = val_df[available_features].fillna(0.0).values.astype(np.float32)
    y_val       = val_df[LABEL_COL].values.astype(np.int8)
    X_test_tab  = test_df[available_features].fillna(0.0).values.astype(np.float32)
    y_test      = test_df[LABEL_COL].values.astype(np.int8)

    fraud_rate = float(y_train.mean())
    spw = (1 - fraud_rate) / max(fraud_rate, 1e-6)
    if spw > 100.0:
        log.warning("[WARN] scale_pos_weight=%.1f capped at 100 (fraud_rate=%.4f)", spw, fraud_rate)
        spw = 100.0
    log.info("[INFO] fraud_rate_train=%.4f scale_pos_weight=%.1f", fraud_rate, spw)

    # --- Build transaction graph for GNN (from train+val, capped at GNN_MAX_TX) ---
    gnn_df = pd.concat([train_df, val_df], ignore_index=True)
    has_graph_cols = "cc_num" in gnn_df.columns and "merchant" in gnn_df.columns
    if not has_graph_cols:
        log.warning("[WARN] cc_num/merchant columns missing — GNN graph will be trivial")

    if len(gnn_df) > GNN_MAX_TX:
        log.info("[INFO] Sampling %d rows for GNN graph (total %d)", GNN_MAX_TX, len(gnn_df))
        gnn_df = gnn_df.sample(GNN_MAX_TX, random_state=42)

    log.info("[INFO] Building transaction graph: %d rows", len(gnn_df))
    t_graph = time.perf_counter()
    node_features, edge_index, tx_mask = build_transaction_graph(gnn_df)
    log.info("[INFO] Graph: %d nodes, %d edges (%.1fs)",
             node_features.shape[0], edge_index.shape[1], time.perf_counter() - t_graph)

    # --- Train GNN ---
    log.info("[INFO] Training GraphSAGE: epochs=%d hidden=%d out=%d device=%s",
             GNN_EPOCHS, GNN_HIDDEN, GNN_OUT, DEVICE)
    y_gnn = gnn_df[LABEL_COL].values.astype(np.float32)
    gnn_model = train_gnn(node_features, edge_index, tx_mask, y_gnn, pos_weight=spw)

    # --- Extract embeddings for all splits ---
    # For each split we build a graph that includes gnn_df (context) + the split rows,
    # then take embeddings only for the split rows (last n_split nodes in tx_mask).
    def _get_split_embeddings(split_df: pd.DataFrame) -> np.ndarray:
        if not has_graph_cols:
            return np.zeros((len(split_df), GNN_OUT), dtype=np.float32)
        combined = pd.concat([gnn_df, split_df], ignore_index=True)
        n_nf, n_ei, n_mask = build_transaction_graph(combined)
        all_emb = extract_gnn_embeddings(gnn_model, n_nf, n_ei, n_mask)
        # split rows are the last len(split_df) transaction nodes
        return all_emb[-len(split_df):]

    log.info("[INFO] Extracting GNN embeddings for train/val/test splits...")
    emb_train = _get_split_embeddings(train_df)
    emb_val   = _get_split_embeddings(val_df)
    emb_test  = _get_split_embeddings(test_df)

    # --- 29-dim XGBoost features (tabular + GNN embeddings) ---
    X_train_29 = np.hstack([X_train_tab, emb_train])
    X_val_29   = np.hstack([X_val_tab,   emb_val])
    X_test_29  = np.hstack([X_test_tab,  emb_test])

    emb_feature_names = [f"gnn_emb_{i}" for i in range(GNN_OUT)]
    all_feature_names = available_features + emb_feature_names

    # --- CPU XGBoost training ---
    log.info("[INFO] Training XGBoost on CPU (29-dim features)...")
    cpu_model, cpu_train_time = train_xgboost(X_train_29, y_train, X_val_29, y_val, "cpu", spw)
    cpu_metrics = evaluate_model(cpu_model, X_test_29, y_test)
    log.info("[INFO] CPU metrics: F1=%.4f AUC-PR=%.4f", cpu_metrics["f1"], cpu_metrics["auc_pr"])

    # --- GPU XGBoost training ---
    gpu_model = None
    gpu_train_time = 0.0
    gpu_metrics: dict = {}
    gpu_succeeded = False
    try:
        log.info("[INFO] Training XGBoost on GPU (29-dim features)...")
        gpu_model, gpu_train_time = train_xgboost(X_train_29, y_train, X_val_29, y_val, "cuda", spw)
        gpu_metrics = evaluate_model(gpu_model, X_test_29, y_test)
        log.info("[INFO] GPU metrics: F1=%.4f AUC-PR=%.4f", gpu_metrics["f1"], gpu_metrics["auc_pr"])
        gpu_succeeded = True
    except Exception as exc:
        log.warning("[WARN] GPU XGBoost failed (%s: %s) — using CPU model for GPU slot",
                    type(exc).__name__, exc)
        gpu_model = cpu_model
        gpu_train_time = cpu_train_time
        gpu_metrics = cpu_metrics

    speedup = cpu_train_time / max(gpu_train_time, 1e-6)
    log.info("[INFO] XGBoost speedup: %.1fx", speedup)

    # --- SHAP (first N_TABULAR tabular features only) ---
    log.info("[INFO] Computing SHAP values...")
    shap_model = gpu_model if gpu_succeeded else cpu_model
    shap_data  = compute_shap(shap_model, X_test_29[:1000], all_feature_names)
    log.info("[INFO] SHAP top feature: %s (%.4f)",
             shap_data["top_features"][0][0], shap_data["top_features"][0][1])

    # --- Write model.py template (substituting architecture constants) ---
    model_py_content = (
        _MODEL_PY_TEMPLATE
        .replace("__N_TABULAR__",  str(N_TABULAR))
        .replace("__GNN_HIDDEN__", str(GNN_HIDDEN))
        .replace("__GNN_OUT__",    str(GNN_OUT))
    )

    # --- Write Triton model repository ---
    # Write ordering:
    #   1. config.pbtxt   (Triton discovers model)
    #   2. model.py       (Python backend code)
    #   3. embedding_xgboost.json  (XGBoost booster)
    #   4. state_dict_gnn.pth      (LAST — Triton start.sh polls this as readiness trigger)
    gnn_state_dict = {k: v.cpu() for k, v in gnn_model.state_dict().items()}

    for model_name, xgb_model_obj, kind in [
        ("fraud_gnn_gpu", gpu_model,  "KIND_GPU"),
        ("fraud_gnn_cpu", cpu_model,  "KIND_CPU"),
    ]:
        model_dir   = MODEL_REPO / model_name
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)

        # 1. config.pbtxt
        write_python_backend_config(model_dir, model_name, kind)
        log.info("[INFO] Wrote config.pbtxt for %s", model_name)

        # 2. model.py
        (version_dir / "model.py").write_text(model_py_content)
        log.info("[INFO] Wrote model.py for %s", model_name)

        # 3. embedding_xgboost.json
        xgb_path = version_dir / "embedding_xgboost.json"
        xgb_model_obj.get_booster().save_model(str(xgb_path))
        log.info("[INFO] Wrote embedding_xgboost.json for %s", model_name)

        # 4. state_dict_gnn.pth — LAST (triggers Triton readiness poll)
        pth_path = version_dir / "state_dict_gnn.pth"
        torch.save(gnn_state_dict, str(pth_path))
        log.info("[INFO] Wrote state_dict_gnn.pth for %s (TRIGGER)", model_name)

    # --- SHAP summary ---
    shap_path = MODEL_REPO / "shap_summary.json"
    with open(str(shap_path), "w") as f:
        json.dump(shap_data, f, indent=2)
    log.info("[INFO] SHAP summary saved: %s", shap_path)

    # --- Training metrics ---
    use_metrics = gpu_metrics if gpu_succeeded else cpu_metrics
    training_metrics = {
        **use_metrics,
        "n_train":         len(X_train_29),
        "n_val":           len(X_val_29),
        "n_test":          len(X_test_29),
        "fraud_rate_train": fraud_rate,
        "fraud_rate_test":  float(y_test.mean()),
        "cpu_train_time_s": cpu_train_time,
        "gpu_train_time_s": gpu_train_time,
        "speedup":          speedup,
        "gpu_available":    gpu_succeeded,
        "n_tabular_features": N_TABULAR,
        "n_gnn_features":   GNN_OUT,
        "n_total_features": N_TABULAR + GNN_OUT,
        "features":         all_feature_names,
        "f1_cpu":           cpu_metrics["f1"],
        "f1_gpu":           gpu_metrics.get("f1", cpu_metrics["f1"]),
        "auc_pr_cpu":       cpu_metrics["auc_pr"],
        "auc_pr_gpu":       gpu_metrics.get("auc_pr", cpu_metrics["auc_pr"]),
        "gnn_epochs":       GNN_EPOCHS,
        "gnn_hidden":       GNN_HIDDEN,
        "gnn_out":          GNN_OUT,
        "gnn_device":       DEVICE,
    }
    metrics_path = MODEL_REPO / "training_metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(training_metrics, f, indent=2)
    log.info("[INFO] Training metrics saved: %s", metrics_path)

    sys.stdout.write(
        f"[TELEMETRY] stage=train "
        f"cpu_train_time_s={cpu_train_time:.1f} "
        f"gpu_train_time_s={gpu_train_time:.1f} "
        f"speedup={speedup:.1f}x "
        f"f1_cpu={cpu_metrics['f1']:.3f} "
        f"f1_gpu={gpu_metrics.get('f1', cpu_metrics['f1']):.3f} "
        f"auc_pr={use_metrics['auc_pr']:.3f} "
        f"rows_trained={len(train_df)} "
        f"rows_evaluated={len(test_df)} "
        f"gnn_device={DEVICE} "
        f"shap_computed=true\n"
    )
    sys.stdout.flush()
    log.info("[INFO] Model build complete")


if __name__ == "__main__":
    main()
