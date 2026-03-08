"""
Pod: scoring-cpu
Continuous file-queue worker (CPU lane). Identical to scoring-gpu except:
- FEATURES_PATH defaults to /data/features-cpu
- SCORES_PATH defaults to /data/features-cpu/scores
- MODEL_NAME defaults to fraud_gnn_cpu
- Telemetry stage = scoring-cpu
"""
import os
import sys
import time
import signal
import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tritonclient.http as httpclient

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
FEATURES_PATH = Path(os.environ.get("FEATURES_PATH", "/data/features-cpu"))
SCORES_PATH   = Path(os.environ.get("SCORES_PATH",   "/data/features-cpu/scores"))
TRITON_URL    = os.environ.get("TRITON_URL",    "triton:8000")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "fraud_gnn_cpu")
WINDOW_CHUNKS = int(os.environ.get("WINDOW_CHUNKS", "10"))
TRITON_RETRIES = int(os.environ.get("TRITON_RETRIES", "10"))

FEATURE_COLS = [
    "amt_log", "amt_scaled", "hour_of_day", "day_of_week", "is_weekend",
    "is_night", "distance_km", "category_encoded", "state_encoded",
    "gender_encoded", "city_pop_log", "zip_region", "amt", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long", "merch_zipcode", "zip",
]
N_TABULAR = len(FEATURE_COLS)  # 21
SCORE_COLS = ["trans_num", "cc_num", "merchant", "amt", "category", "is_fraud"]

# Pod-unique prefix so multiple replicas don't overwrite each other's score files.
_POD_PREFIX = os.environ.get("HOSTNAME", str(os.getpid()))


# ---------------------------------------------------------------------------
# Sliding-window graph (identical to scoring-gpu)
# ---------------------------------------------------------------------------

class WindowedGraph:
    def __init__(self, max_chunks: int):
        self.max_chunks = max_chunks
        self.chunks: deque = deque()

    def _rebuild(self) -> tuple:
        user_ids: dict = {}
        merch_ids: dict = {}
        uid = 0
        mid = 0
        for df in self.chunks:
            if "cc_num" in df.columns:
                for cc in df["cc_num"].astype(str).unique():
                    if cc not in user_ids:
                        user_ids[cc] = uid
                        uid += 1
            if "merchant" in df.columns:
                for m in df["merchant"].astype(str).unique():
                    if m not in merch_ids:
                        merch_ids[m] = mid
                        mid += 1
        return user_ids, merch_ids, uid, mid

    def build_inference_graph(self, new_df: pd.DataFrame) -> tuple:
        user_ids, merch_ids, n_users, n_merchants = self._rebuild()

        if "cc_num" in new_df.columns:
            for cc in new_df["cc_num"].astype(str).unique():
                if cc not in user_ids:
                    user_ids[cc] = n_users
                    n_users += 1
        if "merchant" in new_df.columns:
            for m in new_df["merchant"].astype(str).unique():
                if m not in merch_ids:
                    merch_ids[m] = n_merchants
                    n_merchants += 1

        tx_offset = n_users + n_merchants
        all_df = pd.concat(list(self.chunks) + [new_df], ignore_index=True)
        n_tx = len(all_df)
        n_new_tx = len(new_df)

        avail = [c for c in FEATURE_COLS if c in all_df.columns]
        tx_features = all_df[avail].fillna(0.0).values.astype(np.float32)
        if len(avail) < N_TABULAR:
            pad = np.zeros((n_tx, N_TABULAR - len(avail)), dtype=np.float32)
            tx_features = np.hstack([tx_features, pad])
        zeros = np.zeros((tx_offset, N_TABULAR), dtype=np.float32)
        node_features = np.vstack([zeros, tx_features])

        cc_col = all_df["cc_num"].astype(str).map(user_ids).fillna(0).values.astype(np.int64) \
                 if "cc_num" in all_df.columns else np.zeros(n_tx, dtype=np.int64)
        merch_col = (all_df["merchant"].astype(str).map(merch_ids).fillna(0).values.astype(np.int64)
                     + n_users) if "merchant" in all_df.columns else np.full(n_tx, n_users, dtype=np.int64)
        tx_ids = np.arange(tx_offset, tx_offset + n_tx, dtype=np.int64)

        src = np.concatenate([cc_col, tx_ids, merch_col, tx_ids])
        dst = np.concatenate([tx_ids, cc_col, tx_ids,   merch_col])
        edge_index = np.vstack([src, dst])

        feature_mask = np.zeros(n_users + n_merchants + n_tx, dtype=np.int32)
        feature_mask[tx_offset:] = 1

        return node_features, edge_index, feature_mask, n_new_tx

    def add_chunk(self, df: pd.DataFrame) -> None:
        self.chunks.append(df)
        if len(self.chunks) > self.max_chunks:
            self.chunks.popleft()


# ---------------------------------------------------------------------------
# Triton inference
# ---------------------------------------------------------------------------

def _connect_triton(url: str, retries: int) -> httpclient.InferenceServerClient:
    for attempt in range(retries):
        try:
            client = httpclient.InferenceServerClient(url=url, verbose=False)
            if client.is_server_ready():
                log.info("[INFO] Triton ready at %s", url)
                return client
        except Exception as exc:
            log.info("[INFO] Triton not ready (%s), retry %d/%d in 5s...", exc, attempt + 1, retries)
            time.sleep(5)
    log.error("[ERROR] Triton not reachable after %d retries at %s", retries, url)
    sys.exit(1)


def score_chunk(
    df: pd.DataFrame,
    graph: WindowedGraph,
    client: httpclient.InferenceServerClient,
    model_name: str,
) -> np.ndarray:
    node_features, edge_index, feature_mask, n_new_tx = graph.build_inference_graph(df)

    inputs = [
        httpclient.InferInput("NODE_FEATURES", list(node_features.shape), "FP32"),
        httpclient.InferInput("EDGE_INDEX",    list(edge_index.shape),    "INT64"),
        httpclient.InferInput("FEATURE_MASK",  list(feature_mask.shape),  "INT32"),
        httpclient.InferInput("COMPUTE_SHAP",  [1],                       "BOOL"),
    ]
    inputs[0].set_data_from_numpy(node_features)
    inputs[1].set_data_from_numpy(edge_index)
    inputs[2].set_data_from_numpy(feature_mask)
    inputs[3].set_data_from_numpy(np.array([False]))

    outputs = [httpclient.InferRequestedOutput("PREDICTION")]
    response = client.infer(model_name, inputs=inputs, outputs=outputs)
    all_probs = response.as_numpy("PREDICTION").flatten()

    # Triton returns one probability per transaction node (window first, new_df last).
    return all_probs[-n_new_tx:]


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def emit_telemetry(chunk_id: int, rows: int, latency_ms: float, fraud_rate: float,
                   decision_latency_ms: float = 0.0) -> None:
    sys.stdout.write(
        f"[TELEMETRY] stage=scoring-cpu chunk_id={chunk_id} rows={rows} "
        f"latency_ms={latency_ms:.1f} fraud_rate={fraud_rate:.4f} "
        f"decision_latency_ms={decision_latency_ms:.0f}\n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FEATURES_PATH.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.mkdir(parents=True, exist_ok=True)
    log.info("[INFO] scoring-cpu started: FEATURES=%s SCORES=%s MODEL=%s TRITON=%s",
             FEATURES_PATH, SCORES_PATH, MODEL_NAME, TRITON_URL)

    graph  = WindowedGraph(WINDOW_CHUNKS)
    client = _connect_triton(TRITON_URL, TRITON_RETRIES)

    chunk_id = 0
    while not _SHUTDOWN:
        Path("/.healthy").touch()  # liveness heartbeat
        files = sorted(f for f in FEATURES_PATH.glob("*.parquet")
                       if not f.name.endswith((".processing", ".done")))
        claimed: Path | None = None
        for f in files:
            proc = Path(str(f) + ".processing")
            try:
                f.rename(proc)
                claimed = proc
                break
            except (FileNotFoundError, OSError):
                continue

        if claimed is None:
            time.sleep(0.5)
            continue

        try:
            df = pd.read_parquet(str(claimed))
        except Exception as exc:
            log.warning("[WARN] Failed to read %s: %s", claimed.name, exc)
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue

        if len(df) == 0:
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue

        chunk_ts = float(df["chunk_ts"].iloc[0]) if "chunk_ts" in df.columns else None

        t0 = time.perf_counter()
        try:
            probs = score_chunk(df, graph, client, MODEL_NAME)
        except Exception as exc:
            log.warning("[WARN] Triton inference failed: %s — reconnecting", exc)
            try:
                client.close()
            except Exception:
                pass
            client = _connect_triton(TRITON_URL, TRITON_RETRIES)
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        graph.add_chunk(df)

        result = pd.DataFrame()
        for col in SCORE_COLS:
            if col in df.columns:
                result[col] = df[col].values
        result["fraud_score"] = probs[:len(df)]
        result["scored_at"]   = time.time()

        base = claimed.name[:-len(".processing")]
        out_file = SCORES_PATH / base.replace("features_", "scores_", 1)
        pq.write_table(pa.Table.from_pandas(result, preserve_index=False), str(out_file))
        claimed.rename(str(claimed).replace(".processing", ".done"))

        decision_latency_ms = (time.time() - chunk_ts) * 1000 if chunk_ts else 0.0
        fraud_rate = float((probs > 0.5).mean())
        emit_telemetry(chunk_id=chunk_id, rows=len(df),
                       latency_ms=latency_ms, fraud_rate=fraud_rate,
                       decision_latency_ms=decision_latency_ms)
        log.info("[INFO] chunk %06d: %d rows, latency=%.1fms, fraud_rate=%.4f",
                 chunk_id, len(df), latency_ms, fraud_rate)
        chunk_id += 1

    log.info("[INFO] scoring-cpu shutdown after %d chunks", chunk_id)


if __name__ == "__main__":
    main()
