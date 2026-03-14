"""
GPU feature engineering worker for data-prep (v4 — mega-batch).

Mega-batch processing: reads ALL files in a batch into a single cuDF
dataframe, runs feature engineering ONCE on the combined data (100M+ rows),
then writes one consolidated output. This fills L40S SMs with a single
large kernel launch instead of many tiny per-file launches.

Protocol (via multiprocessing.Queue):
  req_q receives: list of (proc_path: str, out_path: str, tmp_path: str) tuples
  res_q sends:    "ready"                              on startup
                  ("ok",    n_rows: int, timing: dict) on success
                  ("error", msg: str,    {})            on exception
  None on req_q → graceful shutdown
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GPU-WORKER] %(message)s",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
ALL_CATEGORIES = [
    "misc_net", "grocery_pos", "entertainment", "gas_transport", "misc_pos",
    "grocery_net", "shopping_net", "shopping_pos", "food_dining", "personal_care",
    "health_fitness", "travel", "kids_pets", "home",
]
CATEGORY_MAP = {cat: idx for idx, cat in enumerate(ALL_CATEGORIES)}

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]
STATE_MAP = {s: idx for idx, s in enumerate(US_STATES)}

FEATURE_COLS = [
    "amt_log", "amt_scaled", "hour_of_day", "day_of_week", "is_weekend",
    "is_night", "distance_km", "category_encoded", "state_encoded",
    "gender_encoded", "city_pop_log", "zip_region", "amt", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long", "merch_zipcode", "zip",
    "is_fraud",
    # Per-customer features (GPU sort + groupby)
    "cust_txn_count", "cust_amt_mean", "cust_amt_std", "cust_velocity",
    # Per-category features
    "cat_amt_mean", "cat_amt_std", "cat_count", "cat_amt_zscore",
    # Per-merchant features
    "merch_txn_count", "merch_amt_mean", "merch_amt_std", "merch_amt_zscore",
    # Percentile ranks
    "amt_rank", "distance_rank",
]

_OUTPUT_COLS = FEATURE_COLS + ["cc_num", "merchant", "trans_num", "category", "chunk_ts"]

_REQUIRED_COLS = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]


# ── Feature engineering (runs on one big concatenated dataframe) ─────────────

def _engineer_features(gdf, cudf):
    """Run all feature engineering on a single (potentially huge) cuDF dataframe.

    With 100M+ rows the sort/groupby/merge kernels actually fill the L40S SMs,
    unlike per-file processing where each 5M-row kernel finishes instantly.
    """
    # ── Clean ──
    gdf["merch_zipcode"] = gdf["merch_zipcode"].fillna(0.0)
    gdf["category"] = gdf["category"].fillna("misc_net")
    gdf["state"] = gdf["state"].fillna("CA")
    gdf["gender"] = gdf["gender"].fillna("F")
    gdf = gdf.dropna(subset=_REQUIRED_COLS)
    if len(gdf) == 0:
        return gdf

    # ── Categorical encoding ──
    gdf["category_encoded"] = gdf["category"].map(CATEGORY_MAP).fillna(0).astype("int8")
    gdf["state_encoded"] = gdf["state"].map(STATE_MAP).fillna(0).astype("int8")
    gdf["gender_encoded"] = (gdf["gender"] == "F").astype("int8")

    # ── Amount features ──
    gdf["amt_log"] = np.log1p(gdf["amt"])
    amt_mean = float(gdf["amt"].mean())
    amt_std = float(gdf["amt"].std())
    gdf["amt_scaled"] = (gdf["amt"] - amt_mean) / max(amt_std, 1e-9)

    # ── Temporal features ──
    ts = cudf.to_datetime(gdf["unix_time"], unit="s")
    gdf["hour_of_day"] = ts.dt.hour.astype("int8")
    gdf["day_of_week"] = ts.dt.dayofweek.astype("int8")
    gdf["is_weekend"] = (gdf["day_of_week"] >= 5).astype("int8")
    gdf["is_night"] = (gdf["hour_of_day"] <= 5).astype("int8")

    # ── Haversine distance ──
    R = 6371.0
    lat1 = np.radians(gdf["lat"])
    lon1 = np.radians(gdf["long"])
    lat2 = np.radians(gdf["merch_lat"])
    lon2 = np.radians(gdf["merch_long"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    gdf["distance_km"] = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    # ── Misc numeric ──
    gdf["city_pop_log"] = np.log1p(gdf["city_pop"])
    gdf["zip_region"] = (gdf["zip"] // 10000).astype("int8")

    # ── Per-customer features (sort + groupby + merge) ──
    gdf = gdf.sort_values(["cc_num", "unix_time"]).reset_index(drop=True)
    cust_stats = gdf.groupby("cc_num", sort=False).agg(
        cust_txn_count=("amt", "count"),
        cust_amt_mean=("amt", "mean"),
        cust_amt_std=("amt", "std"),
    ).reset_index()
    cust_stats["cust_amt_std"] = cust_stats["cust_amt_std"].fillna(0.0)
    gdf = gdf.merge(cust_stats, on="cc_num", how="left")
    del cust_stats
    gdf["_prev_time"] = gdf.groupby("cc_num")["unix_time"].shift(1)
    gdf["cust_velocity"] = (gdf["unix_time"] - gdf["_prev_time"]).fillna(0.0)
    gdf = gdf.drop(columns=["_prev_time"])

    # ── Per-category features ──
    gdf = gdf.sort_values(["category_encoded", "amt"]).reset_index(drop=True)
    cat_stats = gdf.groupby("category_encoded", sort=False).agg(
        cat_amt_mean=("amt", "mean"),
        cat_amt_std=("amt", "std"),
        cat_count=("amt", "count"),
    ).reset_index()
    cat_stats["cat_amt_std"] = cat_stats["cat_amt_std"].fillna(0.0)
    gdf = gdf.merge(cat_stats, on="category_encoded", how="left")
    del cat_stats
    gdf["cat_amt_zscore"] = ((gdf["amt"] - gdf["cat_amt_mean"]) /
                             gdf["cat_amt_std"].clip(lower=1e-9))

    # ── Per-merchant features ──
    gdf = gdf.sort_values(["merchant", "unix_time"]).reset_index(drop=True)
    merch_stats = gdf.groupby("merchant", sort=False).agg(
        merch_txn_count=("amt", "count"),
        merch_amt_mean=("amt", "mean"),
        merch_amt_std=("amt", "std"),
    ).reset_index()
    merch_stats["merch_amt_std"] = merch_stats["merch_amt_std"].fillna(0.0)
    gdf = gdf.merge(merch_stats, on="merchant", how="left")
    del merch_stats
    gdf["merch_amt_zscore"] = ((gdf["amt"] - gdf["merch_amt_mean"]) /
                               gdf["merch_amt_std"].clip(lower=1e-9))

    # ── Percentile ranks ──
    gdf["amt_rank"] = gdf["amt"].rank(pct=True)
    gdf["distance_rank"] = gdf["distance_km"].rank(pct=True)

    return gdf


# ── Mega-batch: concat all files, process once, write one output ─────────────

def _process_mega_batch(file_list: list, cudf) -> tuple:
    """Concat all files into one dataframe, run features once, write one output.

    This is the key change from v3: instead of 128 threads processing 5M rows
    each (GPU idle between launches), we process 100M+ rows in a single kernel
    launch that actually fills the L40S streaming multiprocessors.
    """
    t0 = time.perf_counter()

    # ── Read all files from NFS in parallel (CPU), then transfer to GPU ──
    t_read_start = time.perf_counter()
    import pyarrow.parquet as _pq
    from concurrent.futures import ThreadPoolExecutor

    def _read_one(entry):
        proc_path, out_path, tmp_path = entry
        try:
            table = _pq.read_table(proc_path)
            if table.num_rows > 0:
                return (table, entry, None)
            Path(proc_path).rename(proc_path.replace(".processing", ".done"))
            return (None, entry, None)
        except Exception as exc:
            log.warning("mega-batch: skipping %s: %s", proc_path, exc)
            try:
                Path(proc_path).rename(proc_path.replace(".processing", ".done"))
            except OSError:
                pass
            return (None, entry, exc)

    n_read_pipes = min(len(file_list), 8)
    frames = []
    valid_files = []
    with ThreadPoolExecutor(max_workers=n_read_pipes) as pool:
        for table, entry, err in pool.map(_read_one, file_list):
            if table is not None:
                gdf = cudf.DataFrame.from_arrow(table)
                frames.append(gdf)
                valid_files.append(entry)
                del table

    if not frames:
        # Mark remaining files done
        for proc_path, _, _ in file_list:
            try:
                Path(proc_path).rename(proc_path.replace(".processing", ".done"))
            except OSError:
                pass
        return 0, {"total": time.perf_counter() - t0, "read": 0, "features": 0, "write": 0}

    # Concat into one mega-dataframe
    if len(frames) == 1:
        mega = frames[0]
    else:
        mega = cudf.concat(frames, ignore_index=True)
    del frames
    t_read = time.perf_counter() - t_read_start
    n_rows = len(mega)
    log.info("mega-batch: loaded %d files, %d rows (%.1fs read)",
                 len(valid_files), n_rows, t_read)

    # ── Feature engineering on the full mega-dataframe ──
    t_feat_start = time.perf_counter()
    mega = _engineer_features(mega, cudf)
    t_feat = time.perf_counter() - t_feat_start
    n_rows = len(mega)
    log.info("mega-batch: features done — %d rows (%.1fs gpu)", n_rows, t_feat)

    # ── Convert to Arrow (GPU→CPU transfer) then free GPU memory ──
    t_arrow_start = time.perf_counter()
    out_cols = [c for c in _OUTPUT_COLS if c in mega.columns]
    arrow_out = mega[out_cols].to_arrow()
    del mega
    t_arrow = time.perf_counter() - t_arrow_start
    log.info("mega-batch: arrow conversion done (%.1fs), GPU free for next batch", t_arrow)

    # ── Write to NFS with parallel pipes (background, GPU is free) ──
    _, first_out, _ = valid_files[0]
    out_dir = Path(first_out).parent
    batch_id = Path(valid_files[0][0]).stem.replace(".processing", "")
    n_write_pipes = int(os.environ.get("WRITE_PIPES", "32"))

    # Split arrow table into chunks for parallel NFS writes
    total = len(arrow_out)
    chunk_size = max(1, total // n_write_pipes)
    chunks = []
    for i in range(0, total, chunk_size):
        chunks.append(arrow_out.slice(i, min(chunk_size, total - i)))

    def _write_chunk(idx, chunk):
        out_path = out_dir / f"features_{batch_id}_part{idx:03d}.parquet"
        tmp_path = out_path.with_suffix(".parquet.tmp")
        pq.write_table(chunk, str(tmp_path))
        Path(tmp_path).rename(out_path)

    def _bg_write():
        t_w0 = time.perf_counter()
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_write_pipes) as pool:
            futs = {pool.submit(_write_chunk, i, c): i for i, c in enumerate(chunks)}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as exc:
                    log.error("write pipe %d failed: %s", futs[f], exc)
        # Mark all input files done after all chunks written
        for proc_path, _, _ in valid_files:
            try:
                Path(proc_path).rename(proc_path.replace(".processing", ".done"))
            except OSError:
                pass
        log.info("mega-batch: NFS write done — %d pipes, %.1fs",
                     len(chunks), time.perf_counter() - t_w0)

    import threading
    write_thread = threading.Thread(target=_bg_write, daemon=True)
    write_thread.start()

    elapsed = time.perf_counter() - t0
    log.info("mega-batch: GPU DONE — %d files, %d rows, %.1fs "
                 "(read=%.1fs feat=%.1fs arrow=%.1fs, write=background)",
                 len(valid_files), n_rows, elapsed,
                 t_read, t_feat, t_arrow)

    return n_rows, {
        "total": elapsed,
        "read": t_read,
        "features": t_feat,
        "arrow": t_arrow,
        "files": len(valid_files),
        "write_thread": write_thread,  # caller can join if needed
    }


# ── GPU worker loop ──────────────────────────────────────────────────────────

def run_gpu_loop(req_q, res_q) -> None:
    """
    Long-lived GPU worker loop. cudf imported here so the main process
    can safely import this module without triggering CUDA initialisation.

    Pipelined: while NFS writes batch N in a background thread, GPU
    processes batch N+1. The write thread from the previous batch is
    joined before starting a new one (prevents unbounded memory growth).
    """
    import faulthandler
    import sys as _sys
    import pandas as pd
    faulthandler.enable(file=_sys.stderr, all_threads=True)
    import cudf  # deferred — CUDA only initialised in this fresh process
    log.info("GPU worker: cudf %s, CUDA device %d (mega-batch mode, pipelined writes)",
                 cudf.__version__, 0)
    # Warm-up: force CUDA context + libcudf init before signalling ready.
    pd.DataFrame({"_x": [1.0]}).to_parquet("/tmp/_warmup.parquet")
    _warmup = cudf.read_parquet("/tmp/_warmup.parquet")
    _warmup.to_arrow()
    del _warmup
    res_q.put("ready")

    prev_write_thread = None

    while True:
        msg = req_q.get()
        if msg is None:  # shutdown signal
            if prev_write_thread:
                prev_write_thread.join(timeout=120)
            break
        try:
            # Process current batch (GPU read + features + arrow convert).
            # NFS write from previous batch runs concurrently in background.
            n_rows, timing = _process_mega_batch(msg, cudf)

            # Now join the PREVIOUS batch's write thread before launching
            # this batch's write. This ensures at most 1 write in flight.
            if prev_write_thread:
                prev_write_thread.join(timeout=120)

            prev_write_thread = timing.pop("write_thread", None)
            res_q.put(("ok", n_rows, timing))
        except Exception as exc:
            log.exception("mega-batch error")
            res_q.put(("error", str(exc), {}))
