"""
Pod: data-prep-gpu
Continuous file-queue worker. Atomically claims raw parquet chunks from INPUT_PATH,
engineers 21 features (GPU via cuDF), writes to OUTPUT_PATH.
Multiple replicas race-safely share the queue via POSIX rename atomicity.

GPU worker owns the full file lifecycle: reads raw file, does GPU feature engineering,
writes output to NFS, marks input done. Queue carries only path strings + timing dicts.
"""
import os
import sys
import time
import logging
import signal
import multiprocessing as mp
import queue as _queue_module
from pathlib import Path

import numpy as np
import pandas as pd

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
INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/data/raw/gpu"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/data/features/gpu"))

# ---------------------------------------------------------------------------
# Persistent GPU worker subprocess
# The main process NEVER imports cudf/cupy — any CUDA in the parent
# corrupts numba_cuda's context for subsequent use. All GPU work runs in
# a long-lived child (fork mode = fresh CUDA state, no __main__ reimport).
# ---------------------------------------------------------------------------
_gpu_worker_proc: "mp.Process | None" = None
_gpu_req_q: "mp.Queue | None" = None
_gpu_res_q: "mp.Queue | None" = None
GPU_AVAILABLE = False


def _start_gpu_worker() -> bool:
    """Start persistent GPU worker. Returns True when worker signals ready.

    Uses fork context (not spawn): parent never imports cudf/CUDA so fork is
    safe, and fork avoids reimporting __main__ which would cause recursive
    _start_gpu_worker() calls inside the worker process.
    """
    global _gpu_worker_proc, _gpu_req_q, _gpu_res_q
    try:
        import gpu_worker as _gw  # safe: cudf imported inside run_gpu_loop, not at module level
        ctx = mp.get_context("fork")  # fork: no __main__ reimport, clean CUDA state (parent has none)
        _gpu_req_q = ctx.Queue()
        _gpu_res_q = ctx.Queue()
        _gpu_worker_proc = ctx.Process(
            target=_gw.run_gpu_loop,
            args=(_gpu_req_q, _gpu_res_q),
            daemon=True,
        )
        _gpu_worker_proc.start()
        msg = _gpu_res_q.get(timeout=600)  # wait for cudf + libcudf init + warmup (Numba JIT cold start can exceed 2 min)
        return msg == "ready"
    except Exception as exc:
        log.warning("[WARN] GPU worker startup failed: %s", exc)
        return False


if _start_gpu_worker():
    GPU_AVAILABLE = True
    log.info("[INFO] GPU worker ready — GPU path enabled")
else:
    log.error("[ERROR] GPU worker failed to start — pod is GPU-only, exiting for K8s restart")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Category / state maps (global — used by CPU path for speedup reference)
# ---------------------------------------------------------------------------
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

_REQUIRED_COLS = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]
_PASSTHROUGH_COLS = ["cc_num", "merchant", "trans_num", "category", "chunk_ts"]


# ---------------------------------------------------------------------------
# Haversine distance (pandas/numpy) — CPU reference path for speedup metric
# ---------------------------------------------------------------------------

def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km (numpy)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# CPU feature engineering — full NFS lifecycle reference path for speedup metric
# ---------------------------------------------------------------------------

def engineer_features_cpu(raw_path: Path, cpu_tmp: Path) -> tuple:
    """CPU reference: NFS read → transform → NFS write → delete.

    Timing scope matches GPU worker (both cover read + compute + write) for a
    fair apples-to-apples speedup comparison. Output is deleted immediately —
    result unused, only timing matters. Storage I/O is intentional: FlashBlade
    NFS throughput is the critical pipeline enabler this demo showcases.
    """
    t: dict = {}
    t0 = time.perf_counter()

    # --- NFS read ---
    df = pd.read_parquet(str(raw_path))

    # --- Clean (same as GPU worker) ---
    df["merch_zipcode"] = df["merch_zipcode"].fillna(0.0)
    df["category"] = df["category"].fillna("misc_net")
    df["state"] = df["state"].fillna("CA")
    df["gender"] = df["gender"].fillna("F")
    df = df.dropna(subset=_REQUIRED_COLS)
    if len(df) == 0:
        t["total"] = time.perf_counter() - t0
        return {}, t

    # --- Categorical encodings ---
    t1 = time.perf_counter()
    category_encoded = df["category"].map(CATEGORY_MAP).fillna(0).astype(np.int8)
    state_encoded = df["state"].map(STATE_MAP).fillna(0).astype(np.int8)
    gender_encoded = (df["gender"] == "F").astype(np.int8)
    t["encoding"] = time.perf_counter() - t1

    # --- Amount features ---
    t1 = time.perf_counter()
    amt_log = np.log1p(df["amt"].values)
    amt_mean = df["amt"].mean()
    amt_std = df["amt"].std()
    amt_scaled = (df["amt"].values - amt_mean) / max(amt_std, 1e-9)
    t["amount"] = time.perf_counter() - t1

    # --- Temporal features ---
    t1 = time.perf_counter()
    dt = pd.to_datetime(df["unix_time"], unit="s")
    hour_of_day = dt.dt.hour.astype(np.int8)
    day_of_week = dt.dt.dayofweek.astype(np.int8)
    is_weekend = (day_of_week >= 5).astype(np.int8)
    is_night = (hour_of_day <= 5).astype(np.int8)
    t["temporal"] = time.perf_counter() - t1

    # --- Distance ---
    t1 = time.perf_counter()
    distance_km = haversine_np(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values,
    )
    t["distance"] = time.perf_counter() - t1

    # --- Misc ---
    t1 = time.perf_counter()
    city_pop_log = np.log1p(df["city_pop"].values)
    zip_region = (df["zip"].values // 10000).astype(np.int8)
    t["misc"] = time.perf_counter() - t1

    # --- Build output (same schema as GPU worker) ---
    out_df = pd.DataFrame({
        "amt_log": amt_log, "amt_scaled": amt_scaled,
        "hour_of_day": hour_of_day.values, "day_of_week": day_of_week.values,
        "is_weekend": is_weekend.values, "is_night": is_night.values,
        "distance_km": distance_km,
        "category_encoded": category_encoded.values,
        "state_encoded": state_encoded.values,
        "gender_encoded": gender_encoded.values,
        "city_pop_log": city_pop_log, "zip_region": zip_region,
        "amt": df["amt"].values, "lat": df["lat"].values, "long": df["long"].values,
        "city_pop": df["city_pop"].values, "unix_time": df["unix_time"].values,
        "merch_lat": df["merch_lat"].values, "merch_long": df["merch_long"].values,
        "merch_zipcode": df["merch_zipcode"].values.astype(np.int32),
        "zip": df["zip"].values.astype(np.int32),
        "is_fraud": df["is_fraud"].values,
    })
    for col in _PASSTHROUGH_COLS:
        if col in df.columns:
            out_df[col] = df[col].values

    # --- NFS write + cleanup (tmp deleted — reference timing only) ---
    out_df.to_parquet(str(cpu_tmp), index=False)
    cpu_tmp.unlink(missing_ok=True)

    t["total"] = time.perf_counter() - t0  # NFS read + compute + NFS write
    return {}, t


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def emit_telemetry(stage: str, chunk_id: int, rows: int,
                   cpu_time: float, gpu_time: float,
                   speedup: float, gpu_used: int) -> None:
    sys.stdout.write(
        f"[TELEMETRY] stage={stage} chunk_id={chunk_id} rows={rows} "
        f"cpu_time_s={cpu_time:.3f} gpu_time_s={gpu_time:.3f} "
        f"speedup={speedup:.1f}x gpu_used={gpu_used}\n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main — continuous file-queue loop
# ---------------------------------------------------------------------------

# Pod-unique prefix so multiple replicas don't overwrite each other's output files.
_POD_PREFIX = os.environ.get("HOSTNAME", str(os.getpid()))


def main() -> None:
    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    log.info("[INFO] data-prep-gpu started: INPUT=%s OUTPUT=%s gpu=%s pod=%s",
             INPUT_PATH, OUTPUT_PATH, GPU_AVAILABLE, _POD_PREFIX)

    chunk_id = 0
    while not _SHUTDOWN:
        Path("/.healthy").touch()  # liveness heartbeat
        # --- Claim next available chunk via atomic rename ---
        files = sorted(f for f in INPUT_PATH.glob("*.parquet")
                       if not f.name.endswith((".processing", ".done")))
        claimed: Path | None = None
        for f in files:
            proc = Path(str(f) + ".processing")
            try:
                f.rename(proc)
                claimed = proc
                break
            except (FileNotFoundError, OSError):
                continue  # another worker claimed it first

        if claimed is None:
            time.sleep(0.5)
            continue

        # --- Validate chunk (quick read — separate from timed CPU reference path) ---
        try:
            df_check = pd.read_parquet(str(claimed))
        except Exception as exc:
            log.warning("[WARN] Failed to read %s: %s — skipping", claimed.name, exc)
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue

        if len(df_check) == 0:
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue
        del df_check

        # --- Derive output paths from source filename (preserves chunk identity) ---
        # claimed = "raw_chunk_000042.parquet.processing" → raw_stem = "raw_chunk_000042"
        raw_stem = claimed.name[:-len(".parquet.processing")]
        out_file = OUTPUT_PATH / f"features_{raw_stem}.parquet"
        tmp_file = out_file.with_suffix(".parquet.tmp")
        cpu_tmp  = OUTPUT_PATH / f"cpu_ref_{raw_stem}.parquet.tmp"

        # --- CPU reference path: full NFS lifecycle (read + transform + write) ---
        # Runs before GPU so claimed (.processing) is still present for both reads.
        _, cpu_timing = engineer_features_cpu(claimed, cpu_tmp)

        # --- GPU worker path: send paths, collect result ---
        _gpu_req_q.put((str(claimed), str(out_file), str(tmp_file)))
        try:
            status, n_rows, gpu_timing = _gpu_res_q.get(timeout=600)
        except _queue_module.Empty:
            log.error("[ERROR] GPU worker timeout — exiting for K8s restart")
            sys.exit(1)

        if status != "ok":
            log.error("[ERROR] GPU worker error: %s — exiting for K8s restart", n_rows)
            sys.exit(1)

        # GPU worker handled: atomic write (tmp→rename), claimed.rename(.done).

        speedup = cpu_timing["total"] / max(gpu_timing.get("total", 0.0), 1e-6)
        emit_telemetry(
            stage="prep-gpu", chunk_id=chunk_id, rows=n_rows,
            cpu_time=cpu_timing["total"], gpu_time=gpu_timing.get("total", 0.0),
            speedup=speedup, gpu_used=1,
        )
        log.info("[INFO] chunk %06d: %d rows speedup=%.1fx gpu=1",
                 chunk_id, n_rows, speedup)
        chunk_id += 1

    log.info("[INFO] data-prep-gpu shutdown complete after %d chunks", chunk_id)


if __name__ == "__main__":
    main()
