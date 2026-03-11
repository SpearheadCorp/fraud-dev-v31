"""
GPU feature engineering worker for data-prep-gpu.

Runs as a persistent subprocess managed by prepare.py via multiprocessing
fork context. cudf is imported INSIDE run_gpu_loop (not at module level)
so that `import gpu_worker` from the main process is safe — no CUDA state
is created in the parent process.

Full GPU pipeline: parallel NFS read → GPU cleaning + encoding + feature eng
(incl. sort + groupby per-customer stats) → Arrow export → parallel NFS write.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GPU-WORKER] %(message)s",
    stream=sys.stderr,
    force=True,
)

# Number of I/O threads for parallel NFS read/write.
_IO_WORKERS = int(os.environ.get("IO_WORKERS", "32"))

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

# All output columns (FEATURE_COLS + passthrough) — _file_idx appended at runtime.
_OUTPUT_COLS = FEATURE_COLS + ["cc_num", "merchant", "trans_num", "category", "chunk_ts"]

_REQUIRED_COLS = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]


def _write_one(args):
    """Write one Arrow table partition to NFS and mark input done (runs in thread pool)."""
    arrow_part, proc_path, out_path, tmp_path = args
    pq.write_table(arrow_part, str(tmp_path))
    Path(tmp_path).rename(out_path)
    Path(proc_path).rename(proc_path.replace(".processing", ".done"))


def _process_batch(file_list: list, cudf) -> tuple:
    """Read directly to GPU, feature engineer as one batch, write in parallel.

    Pipeline: cudf.read_parquet (NFS→GPU direct) → GPU sort + groupby + feature eng →
    to_arrow → ThreadPool NFS write.
    """
    t: dict = {}
    t0 = time.perf_counter()

    # --- Read all files directly to GPU with cudf.read_parquet ---
    # Each file read via libcudf goes NFS→host→GPU in one pipeline (no intermediate copy).
    t1 = time.perf_counter()
    frames = []
    valid_files = []
    for idx, (proc_path, out_path, tmp_path) in enumerate(file_list):
        try:
            gdf_part = cudf.read_parquet(proc_path)
        except Exception as exc:
            logging.warning("batch read: skipping %s: %s", proc_path, exc)
            Path(proc_path).rename(proc_path.replace(".processing", ".done"))
            continue
        if len(gdf_part) == 0:
            Path(proc_path).rename(proc_path.replace(".processing", ".done"))
            continue
        gdf_part["_file_idx"] = np.int32(idx)
        frames.append(gdf_part)
        valid_files.append((idx, proc_path, out_path, tmp_path))

    if not frames:
        return 0, {"total": 0.0}

    gdf = cudf.concat(frames, ignore_index=True)
    del frames
    t["read"] = time.perf_counter() - t1
    logging.info("step 0: cudf.read_parquet %d files, %d rows direct to GPU (%.2fs)",
                 len(valid_files), len(gdf), time.perf_counter() - t0)

    # --- Clean (all on GPU) ---
    t1 = time.perf_counter()
    gdf["merch_zipcode"] = gdf["merch_zipcode"].fillna(0.0)
    gdf["category"] = gdf["category"].fillna("misc_net")
    gdf["state"] = gdf["state"].fillna("CA")
    gdf["gender"] = gdf["gender"].fillna("F")
    gdf = gdf.dropna(subset=_REQUIRED_COLS)
    n_rows = len(gdf)
    if n_rows == 0:
        for _, proc_path, _, _ in valid_files:
            Path(proc_path).rename(proc_path.replace(".processing", ".done"))
        return 0, {"total": 0.0}
    t["clean"] = time.perf_counter() - t1
    logging.info("step 1: GPU clean done — %d rows (%.2fs)", n_rows, time.perf_counter() - t0)

    # --- Categorical encoding on GPU ---
    t1 = time.perf_counter()
    gdf["category_encoded"] = gdf["category"].map(CATEGORY_MAP).fillna(0).astype("int8")
    gdf["state_encoded"] = gdf["state"].map(STATE_MAP).fillna(0).astype("int8")
    gdf["gender_encoded"] = (gdf["gender"] == "F").astype("int8")
    t["encoding"] = time.perf_counter() - t1
    logging.info("step 2: GPU encoding done (%.2fs)", time.perf_counter() - t0)

    # --- Amount features ---
    t1 = time.perf_counter()
    gdf["amt_log"] = np.log1p(gdf["amt"])
    amt_mean = float(gdf["amt"].mean())
    amt_std = float(gdf["amt"].std())
    gdf["amt_scaled"] = (gdf["amt"] - amt_mean) / max(amt_std, 1e-9)
    t["amount"] = time.perf_counter() - t1
    logging.info("step 3: amount features done (%.2fs)", time.perf_counter() - t0)

    # --- Temporal features ---
    t1 = time.perf_counter()
    ts = cudf.to_datetime(gdf["unix_time"], unit="s")
    gdf["hour_of_day"] = ts.dt.hour.astype("int8")
    gdf["day_of_week"] = ts.dt.dayofweek.astype("int8")
    gdf["is_weekend"] = (gdf["day_of_week"] >= 5).astype("int8")
    gdf["is_night"] = (gdf["hour_of_day"] <= 5).astype("int8")
    t["temporal"] = time.perf_counter() - t1
    logging.info("step 4: temporal features done (%.2fs)", time.perf_counter() - t0)

    # --- Haversine distance ---
    t1 = time.perf_counter()
    R = 6371.0
    lat1 = np.radians(gdf["lat"])
    lon1 = np.radians(gdf["long"])
    lat2 = np.radians(gdf["merch_lat"])
    lon2 = np.radians(gdf["merch_long"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    gdf["distance_km"] = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    t["distance"] = time.perf_counter() - t1
    logging.info("step 5: haversine done (%.2fs)", time.perf_counter() - t0)

    # --- Misc numeric ---
    t1 = time.perf_counter()
    gdf["city_pop_log"] = np.log1p(gdf["city_pop"])
    gdf["zip_region"] = (gdf["zip"] // 10000).astype("int8")
    t["misc"] = time.perf_counter() - t1
    logging.info("step 6: misc done (%.2fs)", time.perf_counter() - t0)

    # --- Per-customer features (GPU sort + groupby — heavy GPU work) ---
    t1 = time.perf_counter()
    # Sort by customer + time — GPU radix sort on 16M rows is substantial.
    gdf = gdf.sort_values(["cc_num", "unix_time"]).reset_index(drop=True)
    logging.info("step 7a: GPU sort by customer+time done (%.2fs)", time.perf_counter() - t0)

    # GroupBy aggregates: txn count, mean amt, std amt per customer.
    cust_stats = gdf.groupby("cc_num", sort=False).agg(
        cust_txn_count=("amt", "count"),
        cust_amt_mean=("amt", "mean"),
        cust_amt_std=("amt", "std"),
    ).reset_index()
    cust_stats["cust_amt_std"] = cust_stats["cust_amt_std"].fillna(0.0)

    # Merge back (GPU hash join).
    gdf = gdf.merge(cust_stats, on="cc_num", how="left")
    del cust_stats
    logging.info("step 7b: customer groupby+merge done (%.2fs)", time.perf_counter() - t0)

    # Transaction velocity: time since previous txn per customer (GPU diff on sorted data).
    gdf["_prev_time"] = gdf.groupby("cc_num")["unix_time"].shift(1)
    gdf["cust_velocity"] = (gdf["unix_time"] - gdf["_prev_time"]).fillna(0.0)
    gdf = gdf.drop(columns=["_prev_time"])
    logging.info("step 7c: customer velocity done (%.2fs)", time.perf_counter() - t0)

    # --- Per-category stats (another sort + groupby + merge pass) ---
    gdf = gdf.sort_values(["category_encoded", "amt"]).reset_index(drop=True)
    cat_stats = gdf.groupby("category_encoded", sort=False).agg(
        cat_amt_mean=("amt", "mean"),
        cat_amt_std=("amt", "std"),
        cat_count=("amt", "count"),
    ).reset_index()
    cat_stats["cat_amt_std"] = cat_stats["cat_amt_std"].fillna(0.0)
    gdf = gdf.merge(cat_stats, on="category_encoded", how="left")
    del cat_stats
    # Amount z-score within category — how anomalous is this txn for its category.
    gdf["cat_amt_zscore"] = ((gdf["amt"] - gdf["cat_amt_mean"]) /
                             gdf["cat_amt_std"].clip(lower=1e-9))
    logging.info("step 7d: per-category stats done (%.2fs)", time.perf_counter() - t0)

    # --- Per-merchant stats (another sort + groupby + merge pass) ---
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
    logging.info("step 7e: per-merchant stats done (%.2fs)", time.perf_counter() - t0)

    # --- Amount percentile rank (forces full GPU sort on amt column) ---
    gdf["amt_rank"] = gdf["amt"].rank(pct=True)
    gdf["distance_rank"] = gdf["distance_km"].rank(pct=True)
    logging.info("step 7f: percentile ranks done (%.2fs)", time.perf_counter() - t0)

    t["customer"] = time.perf_counter() - t1
    logging.info("step 7: ALL groupby/sort features done — %.2fs GPU time",
                 t["customer"])

    # --- Build Arrow output directly from GPU (libcudf C++ interop, no numba) ---
    t1 = time.perf_counter()
    out_cols = [c for c in _OUTPUT_COLS if c in gdf.columns] + ["_file_idx"]
    arrow_all = gdf[out_cols].to_arrow()
    del gdf
    t["to_arrow"] = time.perf_counter() - t1
    logging.info("step 8: to_arrow done — %d rows, %d cols (%.2fs)",
                 n_rows, arrow_all.num_columns, time.perf_counter() - t0)

    # --- Split by _file_idx ---
    t1 = time.perf_counter()
    file_idx_col = arrow_all.column("_file_idx")
    col_idx = arrow_all.schema.get_field_index("_file_idx")
    arrow_no_idx = arrow_all.remove_column(col_idx)

    # Pre-split into per-file Arrow tables.
    write_tasks = []
    for file_idx, proc_path, out_path, tmp_path in valid_files:
        mask = pa.compute.equal(file_idx_col, pa.scalar(np.int32(file_idx)))
        arrow_part = pa.compute.filter(arrow_no_idx, mask)
        write_tasks.append((arrow_part, proc_path, out_path, tmp_path))

    # --- Parallel NFS writes via thread pool ---
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
        list(pool.map(_write_one, write_tasks))

    t["write"] = time.perf_counter() - t1
    logging.info("step 9: parallel write %d files done (%.2fs)",
                 len(valid_files), time.perf_counter() - t0)

    t["total"] = time.perf_counter() - t0
    logging.info("BATCH DONE — %d files, %d rows, %.2fs total (read=%.1f gpu=%.1f write=%.1f)",
                 len(valid_files), n_rows, t["total"],
                 t.get("read", 0), t.get("customer", 0), t.get("write", 0))

    return n_rows, t


def run_gpu_loop(req_q, res_q) -> None:
    """
    Long-lived GPU worker loop. cudf imported here so the main process
    can safely import this module without triggering CUDA initialisation.

    Signals ready via res_q, then blocks on req_q for work items.
    req_q receives: list of (proc_path, out_path, tmp_path) tuples.
    """
    import faulthandler
    import sys as _sys
    faulthandler.enable(file=_sys.stderr, all_threads=True)
    import cudf  # deferred — CUDA only initialised in this fresh process
    # Warm-up: force CUDA context + libcudf parquet reader init before signalling ready.
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
