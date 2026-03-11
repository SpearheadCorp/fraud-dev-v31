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
_IO_WORKERS = int(os.environ.get("IO_WORKERS", "16"))

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
    # Per-customer rolling features (GPU sort + groupby)
    "cust_txn_count", "cust_amt_mean", "cust_amt_std", "cust_velocity",
]

# All output columns (FEATURE_COLS + passthrough) — _file_idx appended at runtime.
_OUTPUT_COLS = FEATURE_COLS + ["cc_num", "merchant", "trans_num", "category", "chunk_ts"]

_REQUIRED_COLS = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]


def _read_one(args):
    """Read a single parquet file into a pyarrow Table (runs in thread pool)."""
    idx, proc_path = args
    try:
        tbl = pq.read_table(proc_path)
        if tbl.num_rows == 0:
            return idx, proc_path, None
        return idx, proc_path, tbl
    except Exception as exc:
        logging.warning("batch read: skipping %s: %s", proc_path, exc)
        return idx, proc_path, None


def _write_one(args):
    """Write one Arrow table partition to NFS and mark input done (runs in thread pool)."""
    arrow_part, proc_path, out_path, tmp_path = args
    pq.write_table(arrow_part, str(tmp_path))
    Path(tmp_path).rename(out_path)
    Path(proc_path).rename(proc_path.replace(".processing", ".done"))


def _process_batch(file_list: list, cudf) -> tuple:
    """Read files in parallel, GPU feature engineer as one batch, write in parallel.

    Pipeline: ThreadPool NFS read → cudf.from_arrow (single GPU transfer) →
    GPU sort + groupby + feature eng → to_arrow → ThreadPool NFS write.
    """
    t: dict = {}
    t0 = time.perf_counter()

    # --- Parallel NFS reads via thread pool → pyarrow tables in host memory ---
    t1 = time.perf_counter()
    read_args = [(idx, proc_path) for idx, (proc_path, _, _) in enumerate(file_list)]
    tables = []
    valid_files = []
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
        for idx, proc_path, tbl in pool.map(_read_one, read_args):
            if tbl is None:
                Path(proc_path).rename(proc_path.replace(".processing", ".done"))
                continue
            # Tag rows with file index via an extra column.
            idx_col = pa.array(np.full(tbl.num_rows, idx, dtype=np.int32))
            tbl = tbl.append_column("_file_idx", idx_col)
            tables.append(tbl)
            valid_files.append((idx, *file_list[idx]))

    if not tables:
        return 0, {"total": 0.0}

    # Concat Arrow tables on host, then single transfer to GPU.
    arrow_combined = pa.concat_tables(tables)
    del tables
    t["read"] = time.perf_counter() - t1
    logging.info("step 0: parallel read %d files, %d rows (%.2fs)",
                 len(valid_files), arrow_combined.num_rows, time.perf_counter() - t0)

    # --- Single host→GPU transfer ---
    t1 = time.perf_counter()
    gdf = cudf.DataFrame.from_arrow(arrow_combined)
    del arrow_combined
    t["to_gpu"] = time.perf_counter() - t1
    logging.info("step 1: from_arrow → GPU done — %d rows (%.2fs)",
                 len(gdf), time.perf_counter() - t0)

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
    logging.info("step 2: GPU clean done — %d rows (%.2fs)", n_rows, time.perf_counter() - t0)

    # --- Categorical encoding on GPU ---
    t1 = time.perf_counter()
    gdf["category_encoded"] = gdf["category"].map(CATEGORY_MAP).fillna(0).astype("int8")
    gdf["state_encoded"] = gdf["state"].map(STATE_MAP).fillna(0).astype("int8")
    gdf["gender_encoded"] = (gdf["gender"] == "F").astype("int8")
    t["encoding"] = time.perf_counter() - t1
    logging.info("step 3: GPU encoding done (%.2fs)", time.perf_counter() - t0)

    # --- Amount features ---
    t1 = time.perf_counter()
    gdf["amt_log"] = np.log1p(gdf["amt"])
    amt_mean = float(gdf["amt"].mean())
    amt_std = float(gdf["amt"].std())
    gdf["amt_scaled"] = (gdf["amt"] - amt_mean) / max(amt_std, 1e-9)
    t["amount"] = time.perf_counter() - t1
    logging.info("step 4: amount features done (%.2fs)", time.perf_counter() - t0)

    # --- Temporal features ---
    t1 = time.perf_counter()
    ts = cudf.to_datetime(gdf["unix_time"], unit="s")
    gdf["hour_of_day"] = ts.dt.hour.astype("int8")
    gdf["day_of_week"] = ts.dt.dayofweek.astype("int8")
    gdf["is_weekend"] = (gdf["day_of_week"] >= 5).astype("int8")
    gdf["is_night"] = (gdf["hour_of_day"] <= 5).astype("int8")
    t["temporal"] = time.perf_counter() - t1
    logging.info("step 5: temporal features done (%.2fs)", time.perf_counter() - t0)

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
    logging.info("step 6: haversine done (%.2fs)", time.perf_counter() - t0)

    # --- Misc numeric ---
    t1 = time.perf_counter()
    gdf["city_pop_log"] = np.log1p(gdf["city_pop"])
    gdf["zip_region"] = (gdf["zip"] // 10000).astype("int8")
    t["misc"] = time.perf_counter() - t1
    logging.info("step 7: misc done (%.2fs)", time.perf_counter() - t0)

    # --- Per-customer features (GPU sort + groupby — heavy GPU work) ---
    t1 = time.perf_counter()
    # Sort by customer + time — GPU radix sort on 16M rows is substantial.
    gdf = gdf.sort_values(["cc_num", "unix_time"]).reset_index(drop=True)
    logging.info("step 8a: GPU sort done (%.2fs)", time.perf_counter() - t0)

    # GroupBy aggregates: txn count, mean amt, std amt per customer.
    cust_stats = gdf.groupby("cc_num", sort=False).agg(
        cust_txn_count=("amt", "count"),
        cust_amt_mean=("amt", "mean"),
        cust_amt_std=("amt", "std"),
    ).reset_index()
    cust_stats["cust_amt_std"] = cust_stats["cust_amt_std"].fillna(0.0)
    logging.info("step 8b: GPU groupby done — %d customers (%.2fs)",
                 len(cust_stats), time.perf_counter() - t0)

    # Merge back (GPU hash join).
    gdf = gdf.merge(cust_stats, on="cc_num", how="left")
    del cust_stats
    logging.info("step 8c: GPU merge done (%.2fs)", time.perf_counter() - t0)

    # Transaction velocity: time since previous txn per customer (GPU diff on sorted data).
    gdf["_prev_time"] = gdf.groupby("cc_num")["unix_time"].shift(1)
    gdf["cust_velocity"] = (gdf["unix_time"] - gdf["_prev_time"]).fillna(0.0)
    gdf = gdf.drop(columns=["_prev_time"])

    t["customer"] = time.perf_counter() - t1
    logging.info("step 8d: per-customer features done (%.2fs)", time.perf_counter() - t0)

    # --- Build Arrow output directly from GPU (libcudf C++ interop, no numba) ---
    t1 = time.perf_counter()
    out_cols = [c for c in _OUTPUT_COLS if c in gdf.columns] + ["_file_idx"]
    arrow_all = gdf[out_cols].to_arrow()
    del gdf
    t["to_arrow"] = time.perf_counter() - t1
    logging.info("step 9: to_arrow done — %d rows, %d cols (%.2fs)",
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
    logging.info("step 10: parallel write %d files done (%.2fs)",
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
