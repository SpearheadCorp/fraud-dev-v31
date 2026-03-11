"""
GPU feature engineering worker for data-prep-gpu.

Runs as a persistent subprocess managed by prepare.py via multiprocessing
fork context. cudf is imported INSIDE run_gpu_loop (not at module level)
so that `import gpu_worker` from the main process is safe — no CUDA state
is created in the parent process.

Protocol (via multiprocessing.Queue):
  req_q receives: list of (proc_path: str, out_path: str, tmp_path: str) tuples
  res_q sends:    "ready"                              on startup
                  ("ok",    n_rows: int, timing: dict) on success
                  ("error", msg: str,    {})            on exception
  None on req_q → graceful shutdown
"""
import logging
import sys
import time
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
]

_NUMERIC_COLS = [
    "amt", "unix_time", "lat", "long", "merch_lat", "merch_long",
    "city_pop", "zip", "merch_zipcode", "is_fraud",
]
_GPU_FEATURE_COLS = [
    "amt_log", "amt_scaled", "hour_of_day", "day_of_week", "is_weekend",
    "is_night", "distance_km", "city_pop_log", "zip_region",
    "amt", "lat", "long", "city_pop", "unix_time",
    "merch_lat", "merch_long", "merch_zipcode", "zip", "is_fraud",
]

_REQUIRED_COLS = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]

# Passthrough cols for scorer (graph construction). is_fraud + amt already in FEATURE_COLS.
_PASSTHROUGH_COLS = ["cc_num", "merchant", "trans_num", "category", "chunk_ts"]


def _process_batch(file_list: list, cudf) -> tuple:
    """Read multiple raw files from NFS, GPU feature engineer as one batch, write per-file outputs.

    file_list: list of (proc_path, out_path, tmp_path) string tuples.
    cudf passed as arg (imported in caller). Returns (total_rows, timing_dict).

    Reads all files, tags with _file_idx, concatenates into one DataFrame.
    One GPU pass over the combined data. Splits output by _file_idx for per-file writes.
    """
    t: dict = {}
    t0 = time.perf_counter()

    # --- Read and concat all files with _file_idx tag ---
    t1 = time.perf_counter()
    frames = []
    valid_files = []  # track which file_list entries had data
    for idx, (proc_path, out_path, tmp_path) in enumerate(file_list):
        try:
            df_part = pd.read_parquet(proc_path)
        except Exception as exc:
            logging.warning("batch read: skipping %s: %s", proc_path, exc)
            Path(proc_path).rename(proc_path.replace(".processing", ".done"))
            continue
        if len(df_part) == 0:
            Path(proc_path).rename(proc_path.replace(".processing", ".done"))
            continue
        df_part["_file_idx"] = np.int32(idx)
        frames.append(df_part)
        valid_files.append((idx, proc_path, out_path, tmp_path))

    if not frames:
        return 0, {"total": 0.0}

    df = pd.concat(frames, ignore_index=True)
    del frames
    t["read"] = time.perf_counter() - t1
    logging.info("step 0: read %d files, %d total rows (%.2fs)",
                 len(valid_files), len(df), time.perf_counter() - t0)

    # --- Clean ---
    df["merch_zipcode"] = df["merch_zipcode"].fillna(0.0)
    df["category"] = df["category"].fillna("misc_net")
    df["state"] = df["state"].fillna("CA")
    df["gender"] = df["gender"].fillna("F")
    df = df.dropna(subset=_REQUIRED_COLS)
    n_rows = len(df)
    if n_rows == 0:
        for _, proc_path, _, _ in valid_files:
            Path(proc_path).rename(proc_path.replace(".processing", ".done"))
        return 0, {"total": 0.0}

    # --- Categorical encodings in pandas (fast; avoids cuDF string ops) ---
    t1 = time.perf_counter()
    category_encoded = df["category"].map(CATEGORY_MAP).fillna(0).astype(np.int8)
    state_encoded = df["state"].map(STATE_MAP).fillna(0).astype(np.int8)
    gender_encoded = (df["gender"] == "F").astype(np.int8)
    t["encoding"] = time.perf_counter() - t1
    logging.info("step 1: pandas encoding done (%.2fs)", time.perf_counter() - t0)

    # --- Transfer numeric-only columns + _file_idx to GPU ---
    gdf = cudf.from_pandas(df[_NUMERIC_COLS + ["_file_idx"]])
    logging.info("step 2: from_pandas done — %d rows on GPU (%.2fs)", len(gdf), time.perf_counter() - t0)

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

    # --- Build combined Arrow table — GPU cols via libcudf C++ (no numba) ---
    t1 = time.perf_counter()
    arrow_all = gdf[_GPU_FEATURE_COLS + ["_file_idx"]].to_arrow()
    logging.info("step 7: to_arrow done — %d rows (%.2fs)", n_rows, time.perf_counter() - t0)

    # Append pandas-side columns (categoricals + passthrough) directly to Arrow table.
    arrow_all = arrow_all.append_column("category_encoded", pa.array(category_encoded.values))
    arrow_all = arrow_all.append_column("state_encoded",    pa.array(state_encoded.values))
    arrow_all = arrow_all.append_column("gender_encoded",   pa.array(gender_encoded.values))
    for col in _PASSTHROUGH_COLS:
        if col in df.columns and col not in arrow_all.schema.names:
            arrow_all = arrow_all.append_column(col, pa.array(df[col].values))

    # Reorder to canonical FEATURE_COLS order + passthrough + _file_idx at end.
    base_cols = [c for c in FEATURE_COLS if c in arrow_all.schema.names]
    extra_cols = [c for c in arrow_all.schema.names if c not in FEATURE_COLS and c != "_file_idx"]
    arrow_all = arrow_all.select(base_cols + extra_cols + ["_file_idx"])
    t["arrow_build"] = time.perf_counter() - t1
    logging.info("step 8: arrow table built — %d rows, %d cols (%.2fs)",
                 n_rows, arrow_all.num_columns, time.perf_counter() - t0)

    # --- Split by _file_idx and write per-file outputs ---
    t1 = time.perf_counter()
    file_idx_col = arrow_all.column("_file_idx")
    # Drop _file_idx before writing output files (scorer doesn't need it).
    col_idx = arrow_all.schema.get_field_index("_file_idx")
    arrow_no_idx = arrow_all.remove_column(col_idx)

    for file_idx, proc_path, out_path, tmp_path in valid_files:
        # Boolean mask for this file's rows.
        mask = pa.compute.equal(file_idx_col, pa.scalar(np.int32(file_idx)))
        arrow_part = pa.compute.filter(arrow_no_idx, mask)
        pq.write_table(arrow_part, str(tmp_path))
        Path(tmp_path).rename(out_path)
        Path(proc_path).rename(proc_path.replace(".processing", ".done"))
        logging.info("  wrote %d rows → %s", arrow_part.num_rows, out_path)

    t["write"] = time.perf_counter() - t1
    t["total"] = time.perf_counter() - t0
    logging.info("step 9: batch done — %d files, %d rows, %.2fs total",
                 len(valid_files), n_rows, t["total"])

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
    # Warm-up: force CUDA context creation before signalling ready.
    _warmup = cudf.from_pandas(pd.DataFrame({"_x": pd.Series([1.0], dtype="float32")}))
    _warmup.to_arrow().to_pandas()
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
