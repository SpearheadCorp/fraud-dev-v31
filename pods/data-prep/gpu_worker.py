"""
GPU feature engineering worker for data-prep-gpu.

Runs as a persistent subprocess managed by prepare.py via multiprocessing
fork context. cudf is imported INSIDE run_gpu_loop (not at module level)
so that `import gpu_worker` from the main process is safe — no CUDA state
is created in the parent process.

Protocol (via multiprocessing.Queue):
  req_q receives: (proc_path: str, out_path: str, tmp_path: str)
  res_q sends:    "ready"                              on startup
                  ("ok",    n_rows: int, timing: dict) on success
                  ("error", msg: str,    {})            on exception
  None on req_q → graceful shutdown
"""
import io
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


def _process_file(proc_path: str, out_path: str, tmp_path: str, cudf) -> tuple:
    """Read raw file from NFS, GPU feature engineer, write output to NFS, mark input done.

    cudf passed as arg (imported in caller). Returns (n_rows, timing_dict).

    Categorical strings are encoded in pandas to avoid cuDF string handling.
    Only numeric columns are transferred to GPU for vectorised operations.
    GPU data is exported via Arrow (libcudf C++ interop — no numba involvement).
    """
    t: dict = {}
    t0 = time.perf_counter()

    # --- Read full file (all columns needed for passthrough + categoricals) ---
    df = pd.read_parquet(proc_path)
    logging.info("step 0: read %d rows from %s (%.2fs)", len(df), proc_path, time.perf_counter() - t0)

    # --- Clean ---
    df["merch_zipcode"] = df["merch_zipcode"].fillna(0.0)
    df["category"] = df["category"].fillna("misc_net")
    df["state"] = df["state"].fillna("CA")
    df["gender"] = df["gender"].fillna("F")
    df = df.dropna(subset=_REQUIRED_COLS)
    n_rows = len(df)
    if n_rows == 0:
        Path(proc_path).rename(proc_path.replace(".processing", ".done"))
        return 0, {"total": 0.0}

    # --- Categorical encodings in pandas (fast; avoids cuDF string ops) ---
    t1 = time.perf_counter()
    category_encoded = df["category"].map(CATEGORY_MAP).fillna(0).astype(np.int8)
    state_encoded = df["state"].map(STATE_MAP).fillna(0).astype(np.int8)
    gender_encoded = (df["gender"] == "F").astype(np.int8)
    t["encoding"] = time.perf_counter() - t1
    logging.info("step 1: pandas encoding done (%.2fs)", time.perf_counter() - t0)

    # --- Transfer numeric-only columns to GPU ---
    gdf = cudf.from_pandas(df[_NUMERIC_COLS])
    logging.info("step 2: from_pandas done — %d rows on GPU (%.2fs)", len(gdf), time.perf_counter() - t0)

    # --- Amount features ---
    t1 = time.perf_counter()
    gdf["amt_log"] = np.log1p(gdf["amt"])
    logging.info("step 3a: log1p done (%.2fs)", time.perf_counter() - t0)
    amt_mean = float(gdf["amt"].mean())
    amt_std = float(gdf["amt"].std())
    gdf["amt_scaled"] = (gdf["amt"] - amt_mean) / max(amt_std, 1e-9)
    t["amount"] = time.perf_counter() - t1
    logging.info("step 3b: amount features done (%.2fs)", time.perf_counter() - t0)

    # --- Temporal features ---
    t1 = time.perf_counter()
    ts = cudf.to_datetime(gdf["unix_time"], unit="s")
    logging.info("step 4a: to_datetime done (%.2fs)", time.perf_counter() - t0)
    gdf["hour_of_day"] = ts.dt.hour.astype("int8")
    gdf["day_of_week"] = ts.dt.dayofweek.astype("int8")
    gdf["is_weekend"] = (gdf["day_of_week"] >= 5).astype("int8")
    gdf["is_night"] = (gdf["hour_of_day"] <= 5).astype("int8")
    t["temporal"] = time.perf_counter() - t1
    logging.info("step 4b: temporal features done (%.2fs)", time.perf_counter() - t0)

    # --- Haversine distance ---
    t1 = time.perf_counter()
    R = 6371.0
    lat1 = np.radians(gdf["lat"])
    logging.info("step 5a: radians(lat) done (%.2fs)", time.perf_counter() - t0)
    lon1 = np.radians(gdf["long"])
    lat2 = np.radians(gdf["merch_lat"])
    lon2 = np.radians(gdf["merch_long"])
    logging.info("step 5b: all radians done (%.2fs)", time.perf_counter() - t0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    logging.info("step 5c: sin/cos done (%.2fs)", time.perf_counter() - t0)
    gdf["distance_km"] = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    t["distance"] = time.perf_counter() - t1
    logging.info("step 5d: haversine done (%.2fs)", time.perf_counter() - t0)

    # --- Misc numeric ---
    t1 = time.perf_counter()
    gdf["city_pop_log"] = np.log1p(gdf["city_pop"])
    gdf["zip_region"] = (gdf["zip"] // 10000).astype("int8")
    t["misc"] = time.perf_counter() - t1
    logging.info("step 6: misc done (%.2fs)", time.perf_counter() - t0)

    # --- Build Arrow output — GPU cols via libcudf C++ (no numba), then append pandas cols ---
    # gdf.to_arrow() uses libcudf C++ Arrow interop: no numba_cuda.as_cuda_array → no SIGSEGV.
    arrow_out = gdf[_GPU_FEATURE_COLS].to_arrow()
    logging.info("step 7: to_arrow done (%.2fs)", time.perf_counter() - t0)

    # Append pandas-side columns (categoricals + passthrough) directly to Arrow table.
    arrow_out = arrow_out.append_column("category_encoded", pa.array(category_encoded.values))
    arrow_out = arrow_out.append_column("state_encoded",    pa.array(state_encoded.values))
    arrow_out = arrow_out.append_column("gender_encoded",   pa.array(gender_encoded.values))
    for col in _PASSTHROUGH_COLS:
        if col in df.columns and col not in arrow_out.schema.names:
            arrow_out = arrow_out.append_column(col, pa.array(df[col].values))

    # Reorder to canonical FEATURE_COLS order (extra passthrough cols appended after).
    base_cols = [c for c in FEATURE_COLS if c in arrow_out.schema.names]
    extra_cols = [c for c in arrow_out.schema.names if c not in FEATURE_COLS]
    arrow_out = arrow_out.select(base_cols + extra_cols)

    logging.info("step 8: arrow table built — %d rows, %d cols (%.2fs)",
                 n_rows, arrow_out.num_columns, time.perf_counter() - t0)

    # --- Atomic write: tmp → rename (scorer never sees partial file) ---
    pq.write_table(arrow_out, str(tmp_path))
    Path(tmp_path).rename(out_path)
    t["total"] = time.perf_counter() - t0  # includes NFS read + compute + NFS write
    logging.info("step 9: written to %s (%.2fs)", out_path, t["total"])

    # --- Mark input done ---
    Path(proc_path).rename(proc_path.replace(".processing", ".done"))

    return n_rows, t


def run_gpu_loop(req_q, res_q) -> None:
    """
    Long-lived GPU worker loop. cudf imported here so the main process
    can safely import this module without triggering CUDA initialisation.

    Signals ready via res_q, then blocks on req_q for work items.
    """
    import faulthandler
    import sys as _sys
    faulthandler.enable(file=_sys.stderr, all_threads=True)
    import cudf  # deferred — CUDA only initialised in this fresh process
    # Warm-up: force CUDA context creation before signalling ready.
    # `import cudf` is lazy — actual device init happens on the first GPU op.
    # If from_pandas() crashes (SIGSEGV) the worker dies without sending
    # "ready" → main times out (600 s) → GPU startup fails → pod exits.
    _warmup = cudf.from_pandas(pd.DataFrame({"_x": pd.Series([1.0], dtype="float32")}))
    # Exercise the GPU→CPU return path (to_arrow bypasses numba_cuda).
    # If this crashes, worker dies before sending "ready" → clean startup failure.
    _warmup.to_arrow().to_pandas()
    del _warmup
    res_q.put("ready")

    while True:
        msg = req_q.get()
        if msg is None:  # shutdown signal
            break
        proc_path, out_path, tmp_path = msg
        try:
            n_rows, timing = _process_file(proc_path, out_path, tmp_path, cudf)
            res_q.put(("ok", n_rows, timing))
        except Exception as exc:
            res_q.put(("error", str(exc), {}))
