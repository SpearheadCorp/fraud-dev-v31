"""
Pod 3: Model Build
Trains XGBoost (CPU then GPU) on feature files from Pod 2.
Computes SHAP values via XGBoost native pred_contribs.
Writes Triton model repository and evaluation metrics JSON.
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
INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/data/features"))
MODEL_REPO = Path(os.environ.get("MODEL_REPO", "/data/models"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "500000"))

LABEL_COL = "is_fraud"
FEATURE_COLS = [
    "amt_log", "amt_scaled", "hour_of_day", "day_of_week", "is_weekend",
    "is_night", "distance_km", "category_encoded", "state_encoded",
    "gender_encoded", "city_pop_log", "zip_region", "amt", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long", "merch_zipcode", "zip",
]

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
# Triton config.pbtxt
# ---------------------------------------------------------------------------

def write_triton_config(model_dir: Path, model_name: str, kind: str, n_features: int) -> None:
    """Write Triton FIL backend config.pbtxt."""
    config = f"""name: "{model_name}"
backend: "fil"
max_batch_size: 8192
input [{{
  name: "input__0"
  data_type: TYPE_FP32
  dims: [ {n_features} ]
}}]
output [{{
  name: "output__0"
  data_type: TYPE_FP32
  dims: [ 1 ]
}}]
instance_group [{{ kind: {kind} count: 1 }}]
parameters [
  {{ key: "model_type"    value: {{ string_value: "xgboost_json" }} }},
  {{ key: "predict_proba" value: {{ string_value: "true" }} }},
  {{ key: "output_class"  value: {{ string_value: "false" }} }},
  {{ key: "threshold"     value: {{ string_value: "0.5" }} }}
]
"""
    (model_dir / "config.pbtxt").write_text(config)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    device: str,
    scale_pos_weight: float,
) -> tuple:
    """Train XGBoost, return (model, train_time_s)."""
    params = dict(XGB_PARAMS)
    params["device"] = device
    params["scale_pos_weight"] = scale_pos_weight

    model = xgb.XGBClassifier(**params)

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.perf_counter() - t0
    log.info("[INFO] XGBoost %s training complete: %.2fs (%d trees)", device, train_time, model.best_iteration + 1)
    return model, train_time


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return dict of evaluation metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "auc_pr": float(average_precision_score(y_test, y_prob)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": 0.5,
    }


# ---------------------------------------------------------------------------
# SHAP (XGBoost native)
# ---------------------------------------------------------------------------

def compute_shap(model: xgb.XGBClassifier, X_test: np.ndarray, feature_names: list) -> dict:
    """
    Compute SHAP values using XGBoost native pred_contribs.
    No extra library required — XGBoost 2.x built-in.
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_test, feature_names=feature_names)

    # Returns shape (n_samples, n_features + 1); last column is bias
    shap_values = booster.predict(dmat, pred_contribs=True)
    shap_values = shap_values[:, :-1]  # drop bias column

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs_shap.tolist()))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "top_features": top_features,
        "shap_values_sample": shap_values[:100].tolist(),
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    MODEL_REPO.mkdir(parents=True, exist_ok=True)

    # --- Load feature files ---
    log.info("[INFO] Loading feature files from %s", INPUT_PATH)
    splits = {}
    for split in ("train", "val", "test"):
        fpath = INPUT_PATH / f"features_{split}.parquet"
        if not fpath.exists():
            log.error("[ERROR] Missing %s", fpath)
            sys.exit(1)
        splits[split] = pd.read_parquet(str(fpath))
        log.info("[INFO] Loaded %s: %d rows", split, len(splits[split]))

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Guard: empty splits crash training silently
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            log.error("[ERROR] %s split is empty — aborting", split_name)
            sys.exit(1)
        if len(split_df) < 1000:
            log.warning("[WARN] %s split has only %d rows — metrics may be unreliable", split_name, len(split_df))

    # Cap training set if very large
    if len(train_df) > MAX_SAMPLES:
        log.info("[INFO] Capping train set from %d to %d rows", len(train_df), MAX_SAMPLES)
        # Stratified cap: preserve fraud rate
        fraud = train_df[train_df[LABEL_COL] == 1]
        legit = train_df[train_df[LABEL_COL] == 0]
        fraud_cap = int(MAX_SAMPLES * (len(fraud) / len(train_df)))
        legit_cap = MAX_SAMPLES - fraud_cap
        train_df = pd.concat([
            fraud.sample(min(fraud_cap, len(fraud)), random_state=42),
            legit.sample(min(legit_cap, len(legit)), random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare arrays
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    if len(available_features) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_features)
        log.warning("[WARN] Missing feature columns: %s", missing)

    X_train = train_df[available_features].values.astype(np.float32)
    y_train = train_df[LABEL_COL].values.astype(np.int8)
    X_val = val_df[available_features].values.astype(np.float32)
    y_val = val_df[LABEL_COL].values.astype(np.int8)
    X_test = test_df[available_features].values.astype(np.float32)
    y_test = test_df[LABEL_COL].values.astype(np.int8)

    fraud_rate_train = float(y_train.mean())
    scale_pos_weight = (1 - fraud_rate_train) / max(fraud_rate_train, 1e-6)
    if scale_pos_weight > 100.0:
        log.warning("[WARN] scale_pos_weight=%.1f capped at 100 (fraud_rate=%.4f)", scale_pos_weight, fraud_rate_train)
        scale_pos_weight = 100.0
    log.info("[INFO] scale_pos_weight=%.1f (fraud_rate_train=%.4f)", scale_pos_weight, fraud_rate_train)

    # --- CPU training ---
    log.info("[INFO] Training XGBoost on CPU...")
    cpu_model, cpu_train_time = train_xgboost(X_train, y_train, X_val, y_val, "cpu", scale_pos_weight)
    cpu_metrics = evaluate_model(cpu_model, X_test, y_test)
    log.info("[INFO] CPU metrics: F1=%.4f AUC-PR=%.4f", cpu_metrics["f1"], cpu_metrics["auc_pr"])

    # --- GPU training ---
    gpu_model = None
    gpu_train_time = 0.0
    gpu_metrics: dict = {}
    gpu_succeeded = False
    try:
        log.info("[INFO] Training XGBoost on GPU...")
        gpu_model, gpu_train_time = train_xgboost(X_train, y_train, X_val, y_val, "cuda", scale_pos_weight)
        gpu_metrics = evaluate_model(gpu_model, X_test, y_test)
        log.info("[INFO] GPU metrics: F1=%.4f AUC-PR=%.4f", gpu_metrics["f1"], gpu_metrics["auc_pr"])
        gpu_succeeded = True
    except Exception as exc:
        log.warning("[WARN] GPU training failed (%s: %s) — using CPU model for GPU slot", type(exc).__name__, exc)
        gpu_model = cpu_model
        gpu_train_time = cpu_train_time
        gpu_metrics = cpu_metrics

    speedup = cpu_train_time / max(gpu_train_time, 1e-6)
    log.info("[INFO] Speedup: %.1fx (CPU=%.1fs GPU=%.1fs)", speedup, cpu_train_time, gpu_train_time)

    # --- SHAP values ---
    log.info("[INFO] Computing SHAP values (XGBoost native)...")
    shap_model = gpu_model if gpu_succeeded else cpu_model
    shap_data = compute_shap(shap_model, X_test[:1000], available_features)
    log.info("[INFO] SHAP top feature: %s (%.4f)", shap_data["top_features"][0][0], shap_data["top_features"][0][1])

    # --- Write Triton model repository ---
    n_features = len(available_features)

    for model_name, model_obj, kind in [
        ("fraud_xgboost_cpu", cpu_model, "KIND_CPU"),
        ("fraud_xgboost_gpu", gpu_model, "KIND_GPU"),
    ]:
        model_dir = MODEL_REPO / model_name
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Write config.pbtxt FIRST so inference pods that poll for xgboost.json
        # always find a valid config already present when the trigger file appears.
        write_triton_config(model_dir, model_name, kind, n_features)

        model_json_path = version_dir / "xgboost.json"
        model_obj.get_booster().save_model(str(model_json_path))

        # Validate both files written
        if not model_json_path.exists():
            log.error("[ERROR] Model file not written at expected path: %s", model_json_path)
            sys.exit(1)

        log.info("[INFO] Wrote Triton model: %s (%d features)", model_name, n_features)

    # --- Save SHAP summary ---
    shap_path = MODEL_REPO / "shap_summary.json"
    with open(str(shap_path), "w") as f:
        json.dump(shap_data, f, indent=2)
    log.info("[INFO] SHAP summary saved: %s", shap_path)

    # --- Save training metrics ---
    use_metrics = gpu_metrics if gpu_succeeded else cpu_metrics
    training_metrics = {
        **use_metrics,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "fraud_rate_train": fraud_rate_train,
        "fraud_rate_test": float(y_test.mean()),
        "cpu_train_time_s": cpu_train_time,
        "gpu_train_time_s": gpu_train_time,
        "speedup": speedup,
        "gpu_available": gpu_succeeded,
        "n_features": n_features,
        "features": available_features,
        "f1_cpu": cpu_metrics["f1"],
        "f1_gpu": gpu_metrics.get("f1", cpu_metrics["f1"]),
        "auc_pr_cpu": cpu_metrics["auc_pr"],
        "auc_pr_gpu": gpu_metrics.get("auc_pr", cpu_metrics["auc_pr"]),
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
        f"shap_computed=true\n"
    )
    sys.stdout.flush()
    log.info("[INFO] Model build complete")


if __name__ == "__main__":
    main()
