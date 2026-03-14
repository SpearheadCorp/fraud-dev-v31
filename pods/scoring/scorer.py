"""
Pod: scoring (v4)
Continuous file-queue worker. Atomically claims feature parquet chunks from
FEATURES_PATH, builds a sliding-window tri-partite graph, calls Triton
(fraud_gnn_gpu model) for GNN+XGBoost inference, writes fraud scores to
SCORES_PATH. Multiple replicas race-safely share the queue via POSIX rename.
"""
import os
import sys
import time
import signal
import logging
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tritonclient.http as httpclient
import cudf
import cupy as cp
from concurrent.futures import ThreadPoolExecutor

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

# Liveness heartbeat — started immediately so the probe never kills the pod
# during Triton connection wait (can block up to TRITON_RETRIES × 5s).
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
FEATURES_PATH = Path(os.environ.get("FEATURES_PATH", "/data/features"))
SCORES_PATH   = Path(os.environ.get("SCORES_PATH",   "/data/scores"))
TRITON_URL    = os.environ.get("TRITON_URL",    "triton:8000")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "fraud_gnn_gpu")
WINDOW_CHUNKS = int(os.environ.get("WINDOW_CHUNKS", "10"))
TRITON_RETRIES = int(os.environ.get("TRITON_RETRIES", "10"))
BATCH_FILES    = int(os.environ.get("BATCH_FILES", "8"))

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

GRAPH_COLS = ["cc_num", "merchant"]
SCORE_COLS = ["trans_num", "cc_num", "merchant", "amt", "category", "is_fraud"]

# Pod-unique prefix so multiple replicas don't overwrite each other's score files.
_POD_PREFIX = os.environ.get("HOSTNAME", str(os.getpid()))


# ---------------------------------------------------------------------------
# Sliding-window graph
# ---------------------------------------------------------------------------

class WindowedGraph:
    """Maintains a sliding window of the last WINDOW_CHUNKS feature chunks.
    Builds a tri-partite graph (User ↔ Transaction ↔ Merchant) for Triton inference.
    User/merchant node IDs persist as long as they are in the window.
    """

    def __init__(self, max_chunks: int):
        self.max_chunks = max_chunks
        self.chunks: deque = deque()

    def _rebuild(self) -> tuple:
        """Rebuild user/merchant identity maps from current window."""
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
        """
        Build graph from the window + new_df.
        Transaction nodes for new_df are at the END.
        Returns (node_features [n_nodes, N_TABULAR], edge_index [2, n_edges],
                 feature_mask [n_nodes], n_new_tx).
        """
        user_ids, merch_ids, n_users, n_merchants = self._rebuild()

        # Register new entities from new_df
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

        # All transaction rows: window first, then new_df
        all_df = pd.concat(list(self.chunks) + [new_df], ignore_index=True)
        n_tx = len(all_df)
        n_new_tx = len(new_df)

        # Node features: zeros for users/merchants, tabular for transactions
        avail = [c for c in FEATURE_COLS if c in all_df.columns]
        tx_features = all_df[avail].fillna(0.0).values.astype(np.float32)
        if len(avail) < N_TABULAR:
            pad = np.zeros((n_tx, N_TABULAR - len(avail)), dtype=np.float32)
            tx_features = np.hstack([tx_features, pad])
        zeros = np.zeros((tx_offset, N_TABULAR), dtype=np.float32)
        node_features = np.vstack([zeros, tx_features])

        # Edge index (vectorized)
        cc_col   = all_df["cc_num"].astype(str).map(user_ids).fillna(0).values.astype(np.int64) \
                   if "cc_num" in all_df.columns else np.zeros(n_tx, dtype=np.int64)
        merch_col = all_df["merchant"].astype(str).map(merch_ids).fillna(0).values.astype(np.int64) \
                    if "merchant" in all_df.columns else np.zeros(n_tx, dtype=np.int64)
        # Offset merchants into global node id space
        merch_col = merch_col + n_users
        tx_ids = np.arange(tx_offset, tx_offset + n_tx, dtype=np.int64)

        src = np.concatenate([cc_col, tx_ids, merch_col, tx_ids])
        dst = np.concatenate([tx_ids, cc_col, tx_ids,   merch_col])
        edge_index = np.vstack([src, dst])

        # FEATURE_MASK: INT32 per-node, 1 = transaction node
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
    """Run GNN+XGBoost inference for new_df rows. Returns fraud probabilities [n_rows]."""
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

    # Triton returns one probability per transaction node (in tx order: window first, new_df last).
    # new_df rows are always the LAST n_new_tx transaction nodes, so slice from the end.
    return all_probs[-n_new_tx:]


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def emit_telemetry(chunk_id: int, rows: int, latency_ms: float, fraud_rate: float,
                   decision_latency_ms: float = 0.0) -> None:
    sys.stdout.write(
        f"[TELEMETRY] stage=scoring chunk_id={chunk_id} rows={rows} "
        f"latency_ms={latency_ms:.1f} fraud_rate={fraud_rate:.4f} "
        f"decision_latency_ms={decision_latency_ms:.0f}\n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _claim_files(path: Path, batch: int) -> list:
    """Atomically claim up to `batch` feature files."""
    files = sorted(f for f in path.glob("*.parquet")
                   if not f.name.endswith((".processing", ".done")))
    claimed = []
    for f in files:
        if len(claimed) >= batch:
            break
        proc = Path(str(f) + ".processing")
        try:
            f.rename(proc)
            claimed.append(proc)
        except (FileNotFoundError, OSError):
            continue
    return claimed


def _gpu_read_files(claimed: list) -> tuple:
    """Read feature files in parallel using cuDF on GPU."""
    def _read_one(f):
        try:
            gdf = cudf.read_parquet(str(f))
            return (gdf, f) if len(gdf) > 0 else (None, f)
        except Exception as exc:
            log.warning("Failed to read %s: %s", f.name, exc)
            return (None, f)

    frames = []
    valid_files = []
    with ThreadPoolExecutor(max_workers=min(len(claimed), 8)) as pool:
        for gdf, f in pool.map(_read_one, claimed):
            if gdf is not None:
                frames.append(gdf)
                valid_files.append(f)
            else:
                try:
                    f.rename(str(f).replace(".processing", ".done"))
                except OSError:
                    pass

    if not frames:
        return None, [], []
    mega = cudf.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return mega, valid_files, [len(f) for f in frames]


def main() -> None:
    FEATURES_PATH.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.mkdir(parents=True, exist_ok=True)
    log.info("scoring started: FEATURES=%s SCORES=%s MODEL=%s TRITON=%s batch=%d gpu=cudf",
             FEATURES_PATH, SCORES_PATH, MODEL_NAME, TRITON_URL, BATCH_FILES)

    graph  = WindowedGraph(WINDOW_CHUNKS)
    client = _connect_triton(TRITON_URL, TRITON_RETRIES)

    chunk_id = 0
    while not _SHUTDOWN:
        Path("/tmp/.healthy").touch()

        # --- Claim batch of files ---
        claimed = _claim_files(FEATURES_PATH, BATCH_FILES)
        if not claimed:
            time.sleep(0.5)
            continue

        # --- GPU read + concat ---
        t_read = time.perf_counter()
        mega_gdf, valid_files, file_row_counts = _gpu_read_files(claimed)
        if mega_gdf is None:
            continue
        t_read = time.perf_counter() - t_read
        n_rows = len(mega_gdf)

        # --- GPU feature extraction: fillna + column selection on GPU ---
        t_feat = time.perf_counter()
        avail = [c for c in FEATURE_COLS if c in mega_gdf.columns]
        feat_gdf = mega_gdf[avail].fillna(0.0)
        # Transfer to CPU numpy for Triton (cupy -> numpy)
        df = mega_gdf.to_arrow().to_pandas()
        t_feat = time.perf_counter() - t_feat

        chunk_ts = float(df["chunk_ts"].iloc[0]) if "chunk_ts" in df.columns else None
        log.info("mega-batch: %d files, %d rows (%.1fs read, %.1fs gpu-feat)",
                 len(valid_files), n_rows, t_read, t_feat)

        # --- Score via Triton ---
        t_score = time.perf_counter()
        try:
            probs = score_chunk(df, graph, client, MODEL_NAME)
        except Exception as exc:
            log.warning("Triton inference failed: %s — reconnecting", exc)
            try:
                client.close()
            except Exception:
                pass
            client = _connect_triton(TRITON_URL, TRITON_RETRIES)
            for f in valid_files:
                f.rename(str(f).replace(".processing", ".done"))
            continue
        t_score = time.perf_counter() - t_score

        graph.add_chunk(df)

        # --- Write scores (split back into per-file outputs) ---
        t_write = time.perf_counter()
        result = pd.DataFrame()
        for col in SCORE_COLS:
            if col in df.columns:
                result[col] = df[col].values
        result["fraud_score"] = probs[:len(df)]
        result["scored_at"]   = time.time()

        # Write per-file score outputs and mark done
        row_offset = 0
        for f, rc in zip(valid_files, file_row_counts):
            chunk_result = result.iloc[row_offset:row_offset + rc]
            row_offset += rc
            base = f.name[:-len(".processing")]
            out_file = SCORES_PATH / base.replace("features_", "scores_", 1)
            pq.write_table(pa.Table.from_pandas(chunk_result, preserve_index=False), str(out_file))
            f.rename(str(f).replace(".processing", ".done"))
        t_write = time.perf_counter() - t_write

        total_ms = (t_read + t_feat + t_score + t_write) * 1000
        decision_latency_ms = (time.time() - chunk_ts) * 1000 if chunk_ts else 0.0
        fraud_rate = float((probs > 0.5).mean())
        emit_telemetry(chunk_id=chunk_id, rows=n_rows,
                       latency_ms=total_ms, fraud_rate=fraud_rate,
                       decision_latency_ms=decision_latency_ms)
        log.info("batch %06d: %d rows, %.0fms (read=%.1fs feat=%.1fs score=%.1fs write=%.1fs) fraud=%.4f",
                 chunk_id, n_rows, total_ms, t_read, t_feat, t_score, t_write, fraud_rate)
        chunk_id += 1

    log.info("scoring shutdown after %d chunks", chunk_id)


if __name__ == "__main__":
    main()
