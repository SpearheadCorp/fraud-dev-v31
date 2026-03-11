"""
GPU feature engineering worker for data-prep-gpu.

128-pipe concurrent processing: each file is an independent pipeline (thread)
that reads from NFS → GPU feature eng → writes to NFS. cudf releases the GIL
during GPU kernels, so while some pipes wait on NFS I/O, others keep the GPU
busy. This overlaps I/O and compute, sustaining high GPU utilization.

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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GPU-WORKER] %(message)s",
    stream=sys.stderr,
    force=True,
)

# Number of concurrent processing pipelines (threads).
# Each pipe independently: NFS read → GPU feature eng → NFS write.
# GPU stays busy because cudf releases GIL — while some pipes do I/O,
# others run GPU kernels.
_PIPES = int(os.environ.get("GPU_PIPES", "128"))

# ── Constants (duplicated from prepare.py for subprocess isolation) ─────────
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

# Thread-safe counter for logging progress.
_completed = 0
_completed_lock = threading.Lock()


def _process_one_file(proc_path, out_path, tmp_path, cudf):
    """Process a single file end-to-end: NFS read → GPU features → NFS write.

    Each call is one 'pipe'. Multiple pipes run concurrently via ThreadPoolExecutor.
    cudf releases the GIL during GPU kernels, allowing true overlap of I/O and compute.
    """
    global _completed

    # ── Read directly to GPU ──
    try:
        gdf = cudf.read_parquet(proc_path)
    except Exception as exc:
        logging.warning("pipe: skipping %s: %s", proc_path, exc)
        Path(proc_path).rename(proc_path.replace(".processing", ".done"))
        return 0
    if len(gdf) == 0:
        Path(proc_path).rename(proc_path.replace(".processing", ".done"))
        return 0

    # ── Clean ──
    gdf["merch_zipcode"] = gdf["merch_zipcode"].fillna(0.0)
    gdf["category"] = gdf["category"].fillna("misc_net")
    gdf["state"] = gdf["state"].fillna("CA")
    gdf["gender"] = gdf["gender"].fillna("F")
    gdf = gdf.dropna(subset=_REQUIRED_COLS)
    n_rows = len(gdf)
    if n_rows == 0:
        Path(proc_path).rename(proc_path.replace(".processing", ".done"))
        return 0

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

    # ── Write output (GPU→Arrow→NFS) ──
    out_cols = [c for c in _OUTPUT_COLS if c in gdf.columns]
    arrow_out = gdf[out_cols].to_arrow()
    del gdf
    pq.write_table(arrow_out, str(tmp_path))
    del arrow_out
    Path(tmp_path).rename(out_path)
    Path(proc_path).rename(proc_path.replace(".processing", ".done"))

    with _completed_lock:
        _completed += 1

    return n_rows


def _process_batch(file_list: list, cudf) -> tuple:
    """Launch concurrent pipes — each file processed independently on GPU.

    128 pipes overlap NFS I/O with GPU compute: while some threads read/write,
    others run cudf sort/groupby/merge kernels. GPU stays busy continuously.
    """
    global _completed
    _completed = 0
    n_pipes = min(len(file_list), _PIPES)
    t0 = time.perf_counter()

    logging.info("launching %d concurrent pipes for %d files", n_pipes, len(file_list))

    total_rows = 0
    with ThreadPoolExecutor(max_workers=n_pipes) as pool:
        futures = {
            pool.submit(_process_one_file, proc, out, tmp, cudf): (proc, out, tmp)
            for proc, out, tmp in file_list
        }
        for future in as_completed(futures):
            try:
                n = future.result()
                total_rows += n
            except Exception as exc:
                proc, _, _ = futures[future]
                logging.error("pipe error on %s: %s", proc, exc)
                try:
                    Path(proc).rename(proc.replace(".processing", ".done"))
                except OSError:
                    pass

    elapsed = time.perf_counter() - t0
    logging.info("ALL PIPES DONE — %d files, %d rows, %.2fs (%.0f files/s, %.0f rows/s)",
                 len(file_list), total_rows, elapsed,
                 len(file_list) / max(elapsed, 0.001),
                 total_rows / max(elapsed, 0.001))

    return total_rows, {"total": elapsed, "pipes": n_pipes}


def run_gpu_loop(req_q, res_q) -> None:
    """
    Long-lived GPU worker loop. cudf imported here so the main process
    can safely import this module without triggering CUDA initialisation.
    """
    import faulthandler
    import sys as _sys
    import pandas as pd
    faulthandler.enable(file=_sys.stderr, all_threads=True)
    import cudf  # deferred — CUDA only initialised in this fresh process
    logging.info("GPU worker: cudf %s, CUDA device %d, pipes=%d",
                 cudf.__version__, 0, _PIPES)
    # Warm-up: force CUDA context + libcudf init before signalling ready.
    pd.DataFrame({"_x": [1.0]}).to_parquet("/tmp/_warmup.parquet")
    _warmup = cudf.read_parquet("/tmp/_warmup.parquet")
    _warmup.to_arrow()
    del _warmup
    res_q.put("ready")

    while True:
        msg = req_q.get()
        if msg is None:  # shutdown signal
            break
        try:
            n_rows, timing = _process_batch(msg, cudf)
            res_q.put(("ok", n_rows, timing))
        except Exception as exc:
            res_q.put(("error", str(exc), {}))
