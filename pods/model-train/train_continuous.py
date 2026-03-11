"""
Pod: model-train (v4)
Continuous GNN + XGBoost training loop. Reads feature chunks from INPUT_PATH
(written by data-prep), trains GraphSAGE embeddings + XGBoost classifier,
writes model artifacts to MODEL_REPO. No hot-swap to Triton yet — just writes
artifacts. Triton can pick up new models via repository polling if enabled.

Runs in a loop: every TRAIN_INTERVAL_SEC, checks for new feature files. When
MIN_NEW_FILES new files have accumulated, triggers a training cycle on the
last MAX_FILES chunks.
"""
import os
import sys
import json
import time
import logging
import signal
import threading
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

# Liveness heartbeat thread
def _liveness_heartbeat():
    while True:
        try:
            Path("/tmp/.healthy").touch()
        except OSError:
            pass
        time.sleep(10)

threading.Thread(target=_liveness_heartbeat, daemon=True, name="liveness").start()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH      = Path(os.environ.get("INPUT_PATH",      "/data/features"))
MODEL_REPO      = Path(os.environ.get("MODEL_REPO",      "/data/models"))
MAX_FILES        = int(os.environ.get("MAX_FILES",        "200"))
MIN_NEW_FILES    = int(os.environ.get("MIN_NEW_FILES",    "20"))
TRAIN_INTERVAL_SEC = int(os.environ.get("TRAIN_INTERVAL_SEC", "60"))
MAX_SAMPLES      = int(os.environ.get("MAX_SAMPLES",      "500000"))
GNN_EPOCHS       = int(os.environ.get("GNN_EPOCHS",       "16"))
GNN_HIDDEN       = int(os.environ.get("GNN_HIDDEN",       "16"))
GNN_OUT          = int(os.environ.get("GNN_OUT",          "8"))
GNN_LR           = float(os.environ.get("GNN_LR",         "0.01"))
GNN_MAX_TX       = int(os.environ.get("GNN_MAX_TX",       "100000"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COL = "is_fraud"
FEATURE_COLS = [
    "amt_log", "amt_scaled", "hour_of_day", "day_of_week", "is_weekend",
    "is_night", "distance_km", "category_encoded", "state_encoded",
    "gender_encoded", "city_pop_log", "zip_region", "amt", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long", "merch_zipcode", "zip",
    # Per-customer features
    "cust_txn_count", "cust_amt_mean", "cust_amt_std", "cust_velocity",
    # Per-category features
    "cat_amt_mean", "cat_amt_std", "cat_count", "cat_amt_zscore",
    # Per-merchant features
    "merch_txn_count", "merch_amt_mean", "merch_amt_std", "merch_amt_zscore",
    # Percentile ranks
    "amt_rank", "distance_rank",
]
N_TABULAR = len(FEATURE_COLS)  # 35

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
# GraphSAGE model
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
    """Build tri-partite graph: User <-> Transaction <-> Merchant."""
    users = {cc: i for i, cc in enumerate(df["cc_num"].astype(str).unique())}
    n_users = len(users)
    merchants = {m: n_users + i for i, m in enumerate(df["merchant"].astype(str).unique())}
    n_merchants = len(merchants)
    n_tx = len(df)
    tx_offset = n_users + n_merchants

    available = [c for c in FEATURE_COLS if c in df.columns]
    tx_features = df[available].fillna(0.0).values.astype(np.float32)
    if len(available) < N_TABULAR:
        pad = np.zeros((n_tx, N_TABULAR - len(available)), dtype=np.float32)
        tx_features = np.hstack([tx_features, pad])
    zeros = np.zeros((n_users + n_merchants, N_TABULAR), dtype=np.float32)
    node_features = np.vstack([zeros, tx_features])

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

def train_gnn(node_features, edge_index, tx_mask, y_tx, pos_weight):
    """Train GraphSAGE with a temporary linear head."""
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
    log.info("[INFO] GNN training: %.1fs, device=%s", train_time, DEVICE)
    model.eval()
    return model


def extract_gnn_embeddings(model, node_features, edge_index, tx_mask):
    """Extract GNN_OUT-dim embeddings for transaction nodes."""
    with torch.no_grad():
        x  = torch.tensor(node_features, dtype=torch.float32, device=DEVICE)
        ei = torch.tensor(edge_index,    dtype=torch.long,    device=DEVICE)
        all_emb = model(x, ei)
    return all_emb[tx_mask].cpu().numpy()


# ---------------------------------------------------------------------------
# Triton Python backend model.py template
# ---------------------------------------------------------------------------

_MODEL_PY_TEMPLATE = '''\
# AUTO-GENERATED by model-train — Triton Python backend for GNN + XGBoost fraud detection
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


def write_python_backend_config(model_dir: Path, model_name: str, kind: str) -> None:
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
# XGBoost training + evaluation
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val, device, scale_pos_weight):
    params = dict(XGB_PARAMS)
    params["device"] = device
    params["scale_pos_weight"] = scale_pos_weight
    model = xgb.XGBClassifier(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.perf_counter() - t0
    log.info("[INFO] XGBoost %s: %.2fs (%d trees)", device, train_time, model.best_iteration + 1)
    return model, train_time


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0,
        "auc_pr":    float(average_precision_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def compute_shap(model, X_test, feature_names):
    booster  = model.get_booster()
    dmat     = xgb.DMatrix(X_test, feature_names=feature_names)
    shap_raw = booster.predict(dmat, pred_contribs=True)[:, :-1]
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
# Training cycle
# ---------------------------------------------------------------------------

def _temporal_split(df):
    df = df.sort_values("unix_time").reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_val].copy(), df.iloc[n_val:].copy()


def run_training_cycle(chunk_files: list, cycle_num: int) -> dict:
    """Run one full training cycle on the given feature chunk files."""
    t_cycle = time.perf_counter()

    # Load data
    selected = chunk_files[-MAX_FILES:]
    log.info("[CYCLE %d] Loading %d chunk files", cycle_num, len(selected))
    dfs = [pd.read_parquet(str(f)) for f in selected]
    df  = pd.concat(dfs, ignore_index=True)
    log.info("[CYCLE %d] %d rows loaded", cycle_num, len(df))

    required = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    if len(df) < 1000:
        log.warning("[CYCLE %d] Only %d rows — skipping", cycle_num, len(df))
        return {}

    # Cap rows
    if len(df) > MAX_SAMPLES:
        fraud = df[df[LABEL_COL] == 1]
        legit = df[df[LABEL_COL] == 0]
        f_cap = int(MAX_SAMPLES * len(fraud) / len(df))
        l_cap = MAX_SAMPLES - f_cap
        df = pd.concat([
            fraud.sample(min(f_cap, len(fraud)), random_state=42),
            legit.sample(min(l_cap, len(legit)), random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, val_df, test_df = _temporal_split(df)
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]

    X_train_tab = train_df[available_features].fillna(0.0).values.astype(np.float32)
    y_train     = train_df[LABEL_COL].values.astype(np.int8)
    X_val_tab   = val_df[available_features].fillna(0.0).values.astype(np.float32)
    y_val       = val_df[LABEL_COL].values.astype(np.int8)
    X_test_tab  = test_df[available_features].fillna(0.0).values.astype(np.float32)
    y_test      = test_df[LABEL_COL].values.astype(np.int8)

    fraud_rate = float(y_train.mean())
    spw = min((1 - fraud_rate) / max(fraud_rate, 1e-6), 100.0)

    # Build graph
    gnn_df = pd.concat([train_df, val_df], ignore_index=True)
    has_graph_cols = "cc_num" in gnn_df.columns and "merchant" in gnn_df.columns

    if len(gnn_df) > GNN_MAX_TX:
        gnn_df = gnn_df.sample(GNN_MAX_TX, random_state=42)

    node_features, edge_index, tx_mask = build_transaction_graph(gnn_df)

    # Train GNN
    y_gnn = gnn_df[LABEL_COL].values.astype(np.float32)
    gnn_model = train_gnn(node_features, edge_index, tx_mask, y_gnn, pos_weight=spw)

    # Extract embeddings
    def _get_split_embeddings(split_df):
        if not has_graph_cols:
            return np.zeros((len(split_df), GNN_OUT), dtype=np.float32)
        combined = pd.concat([gnn_df, split_df], ignore_index=True)
        n_nf, n_ei, n_mask = build_transaction_graph(combined)
        all_emb = extract_gnn_embeddings(gnn_model, n_nf, n_ei, n_mask)
        return all_emb[-len(split_df):]

    emb_train = _get_split_embeddings(train_df)
    emb_val   = _get_split_embeddings(val_df)
    emb_test  = _get_split_embeddings(test_df)

    X_train_29 = np.hstack([X_train_tab, emb_train])
    X_val_29   = np.hstack([X_val_tab,   emb_val])
    X_test_29  = np.hstack([X_test_tab,  emb_test])

    emb_feature_names = [f"gnn_emb_{i}" for i in range(GNN_OUT)]
    all_feature_names = available_features + emb_feature_names

    # Train XGBoost (GPU)
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_model, gpu_train_time = train_xgboost(X_train_29, y_train, X_val_29, y_val, xgb_device, spw)
    metrics = evaluate_model(gpu_model, X_test_29, y_test)
    log.info("[CYCLE %d] F1=%.4f AUC-PR=%.4f", cycle_num, metrics["f1"], metrics["auc_pr"])

    # SHAP
    shap_data = compute_shap(gpu_model, X_test_29[:1000], all_feature_names)

    # Write model.py template
    model_py_content = (
        _MODEL_PY_TEMPLATE
        .replace("__N_TABULAR__",  str(N_TABULAR))
        .replace("__GNN_HIDDEN__", str(GNN_HIDDEN))
        .replace("__GNN_OUT__",    str(GNN_OUT))
    )

    # Write to Triton model repo (GPU model only)
    gnn_state_dict = {k: v.cpu() for k, v in gnn_model.state_dict().items()}
    model_name = "fraud_gnn_gpu"
    model_dir   = MODEL_REPO / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    write_python_backend_config(model_dir, model_name, "KIND_GPU")
    (version_dir / "model.py").write_text(model_py_content)
    gpu_model.get_booster().save_model(str(version_dir / "embedding_xgboost.json"))
    torch.save(gnn_state_dict, str(version_dir / "state_dict_gnn.pth"))
    log.info("[CYCLE %d] Model artifacts written to %s", cycle_num, model_dir)

    # Save SHAP + training metrics
    shap_path = MODEL_REPO / "shap_summary.json"
    with open(str(shap_path), "w") as f:
        json.dump(shap_data, f, indent=2)

    cycle_time = time.perf_counter() - t_cycle
    training_metrics = {
        **metrics,
        "cycle": cycle_num,
        "n_train": len(X_train_29),
        "n_val":   len(X_val_29),
        "n_test":  len(X_test_29),
        "fraud_rate_train": fraud_rate,
        "gpu_train_time_s": gpu_train_time,
        "cycle_time_s": cycle_time,
        "gnn_epochs": GNN_EPOCHS,
        "gnn_hidden": GNN_HIDDEN,
        "gnn_out":    GNN_OUT,
        "gnn_device": DEVICE,
        "features":   all_feature_names,
    }
    metrics_path = MODEL_REPO / "training_metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(training_metrics, f, indent=2)

    # Telemetry
    sys.stdout.write(
        f"[TELEMETRY] stage=train cycle={cycle_num} "
        f"rows_trained={len(train_df)} "
        f"f1={metrics['f1']:.3f} auc_pr={metrics['auc_pr']:.3f} "
        f"cycle_time_s={cycle_time:.1f} gnn_device={DEVICE}\n"
    )
    sys.stdout.flush()

    return training_metrics


# ---------------------------------------------------------------------------
# Main — continuous loop
# ---------------------------------------------------------------------------

def main() -> None:
    MODEL_REPO.mkdir(parents=True, exist_ok=True)
    log.info("[INFO] model-train started: INPUT=%s MODEL_REPO=%s device=%s",
             INPUT_PATH, MODEL_REPO, DEVICE)
    log.info("[INFO] Config: interval=%ds min_new=%d max_files=%d",
             TRAIN_INTERVAL_SEC, MIN_NEW_FILES, MAX_FILES)

    seen_files: set = set()
    cycle_num = 0

    while not _SHUTDOWN:
        Path("/tmp/.healthy").touch()

        # Scan for .done feature files (already processed by prep, safe to read)
        done_files = sorted(
            f for f in INPUT_PATH.glob("*.parquet.done")
        )
        # Also include .parquet files not being processed (pending ones are fair game for training reads)
        pending_files = sorted(
            f for f in INPUT_PATH.glob("*.parquet")
            if not f.name.endswith((".processing", ".done"))
        )
        all_files = done_files + pending_files
        all_names = {f.name for f in all_files}
        new_files = all_names - seen_files

        if len(new_files) >= MIN_NEW_FILES and len(all_files) >= MIN_NEW_FILES:
            cycle_num += 1
            log.info("[INFO] %d new files detected, starting training cycle %d",
                     len(new_files), cycle_num)
            try:
                run_training_cycle(all_files, cycle_num)
            except Exception as exc:
                log.error("[ERROR] Training cycle %d failed: %s", cycle_num, exc, exc_info=True)
            seen_files = all_names

        # Sleep in small increments for responsive shutdown
        for _ in range(TRAIN_INTERVAL_SEC * 2):
            if _SHUTDOWN:
                break
            time.sleep(0.5)

    log.info("[INFO] model-train shutdown after %d cycles", cycle_num)


if __name__ == "__main__":
    main()
