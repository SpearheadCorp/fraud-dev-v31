"""
GPU feature engineering worker for data-prep-gpu.

Runs as a persistent subprocess managed by prepare.py via multiprocessing
spawn context. cudf is imported INSIDE run_gpu_loop (not at module level)
so that `import gpu_worker` from the main process is safe — no CUDA state
is created in the parent process.

Protocol (via multiprocessing.Queue):
  req_q receives: bytes (parquet-serialised input DataFrame)
  res_q sends:    ("ready",) on startup
                  ("ok",    bytes, dict) on success
                  ("error", str,   dict) on exception
  None on req_q → graceful shutdown
"""
import io
import logging
import sys
import time
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


def _engineer(df: pd.DataFrame, cudf) -> tuple:
    """Core GPU feature engineering. cudf passed as arg (imported in caller).

    Categorical strings are encoded in pandas to avoid cuDF string handling.
    Only numeric columns are transferred to GPU for vectorised operations.
    """
    t: dict = {}
    t0 = time.perf_counter()

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

    # --- Assemble: GPU numeric cols + pandas-encoded categoricals ---
    # Use to_arrow().to_pandas() instead of to_pandas() directly.
    # Direct to_pandas() calls numba_cuda.as_cuda_array → ensure_context →
    # safe_cuda_api_call → SIGSEGV in forked process (libcudf context ≠ numba context).
    # The Arrow path (libcudf C++ interop) copies GPU→CPU without any numba involvement.
    result = gdf[_GPU_FEATURE_COLS].to_arrow().to_pandas()
    logging.info("step 7: to_arrow().to_pandas() done (%.2fs)", time.perf_counter() - t0)
    result["category_encoded"] = category_encoded.values
    result["state_encoded"] = state_encoded.values
    result["gender_encoded"] = gender_encoded.values

    t["total"] = time.perf_counter() - t0
    logging.info("step 8: engineer complete — total %.2fs", t["total"])
    return result[FEATURE_COLS], t


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
    # "ready" → main times out (120 s) → GPU startup fails → pod exits.
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
        df_bytes: bytes = msg
        try:
            df = pd.read_parquet(io.BytesIO(df_bytes))
            result, timing = _engineer(df, cudf)
            buf = io.BytesIO()
            pq.write_table(pa.Table.from_pandas(result, preserve_index=False), buf)
            res_q.put(("ok", buf.getvalue(), timing))
        except Exception as exc:
            res_q.put(("error", str(exc), {}))
