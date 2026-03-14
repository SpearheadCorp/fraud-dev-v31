"""
Pod: data-prep (v4 — mega-batch)
Continuous file-queue worker. Atomically claims raw parquet chunks from INPUT_PATH,
concatenates them into one mega-dataframe, engineers features (GPU via cuDF) in a
single large kernel launch, writes consolidated output to OUTPUT_PATH.

Mega-batch approach: concat 20+ files into 100M+ rows and process once.
This fills L40S SMs with meaningful work per kernel launch.
"""
import os
import sys
import time
import logging
import signal
import threading
import multiprocessing as mp
import queue as _queue_module
from pathlib import Path

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
INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/data/raw"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/data/features"))
BATCH_FILES = int(os.environ.get("BATCH_FILES", "20"))

# ---------------------------------------------------------------------------
# Persistent GPU worker subprocess — mega-batch mode
# The main process NEVER imports cudf/cupy — any CUDA in the parent
# corrupts numba_cuda's context for subsequent use. All GPU work runs in
# a long-lived child (fork mode = fresh CUDA state, no __main__ reimport).
# GPU worker concats all files in a batch into one mega-dataframe and
# processes features in a single large kernel launch.
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


# Liveness heartbeat thread — started before GPU worker so the probe never
# kills the pod during Numba JIT cold-start (can exceed 4 min on first run).
# Main loop also touches /tmp/.healthy every iteration for tight steady-state detection.
def _liveness_heartbeat():
    while True:
        try:
            Path("/tmp/.healthy").touch()
        except OSError:
            pass
        time.sleep(10)

threading.Thread(target=_liveness_heartbeat, daemon=True, name="liveness").start()

if _start_gpu_worker():
    GPU_AVAILABLE = True
    log.info("[INFO] GPU worker ready — GPU path enabled")
else:
    log.error("[ERROR] GPU worker failed to start — pod is GPU-only, exiting for K8s restart")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Category / state maps
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
# Telemetry
# ---------------------------------------------------------------------------

def emit_telemetry(chunk_id: int, rows: int, gpu_time: float, feat_time: float, n_files: int) -> None:
    sys.stdout.write(
        f"[TELEMETRY] stage=prep chunk_id={chunk_id} rows={rows} "
        f"gpu_time_s={gpu_time:.3f} feat_time_s={feat_time:.3f} gpu_used=1 batch_files={n_files}\n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Claim helpers
# ---------------------------------------------------------------------------

# Pod-unique prefix so multiple replicas don't overwrite each other's output files.
_POD_PREFIX = os.environ.get("HOSTNAME", str(os.getpid()))


def _claim_files(max_files: int) -> list:
    """Claim up to max_files via atomic rename. Returns list of (proc_path, out_path, tmp_path) tuples."""
    files = sorted(f for f in INPUT_PATH.glob("*.parquet")
                   if not f.name.endswith((".processing", ".done")))
    claimed = []
    for f in files:
        if len(claimed) >= max_files:
            break
        proc = Path(str(f) + ".processing")
        try:
            f.rename(proc)
        except (FileNotFoundError, OSError):
            continue  # another worker claimed it first
        raw_stem = f.name[:-len(".parquet")]
        out_file = OUTPUT_PATH / f"features_{raw_stem}.parquet"
        tmp_file = out_file.with_suffix(".parquet.tmp")
        claimed.append((str(proc), str(out_file), str(tmp_file)))
    return claimed


# ---------------------------------------------------------------------------
# Main — continuous batch + prefetch loop
# ---------------------------------------------------------------------------


def main() -> None:
    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    log.info("[INFO] data-prep started: INPUT=%s OUTPUT=%s gpu=%s pod=%s batch=%d",
             INPUT_PATH, OUTPUT_PATH, GPU_AVAILABLE, _POD_PREFIX, BATCH_FILES)

    chunk_id = 0
    pending_batch = None  # True when we've sent a batch to GPU and haven't collected yet.

    while not _SHUTDOWN:
        Path("/tmp/.healthy").touch()  # liveness heartbeat

        # --- Claim a batch of files ---
        batch = _claim_files(BATCH_FILES)
        if not batch:
            # If GPU is still working on a previous batch, collect that first.
            if pending_batch is not None:
                try:
                    status, n_rows, gpu_timing = _gpu_res_q.get(timeout=600)
                except _queue_module.Empty:
                    log.error("[ERROR] GPU worker timeout — exiting for K8s restart")
                    sys.exit(1)
                if status != "ok":
                    log.error("[ERROR] GPU worker error: %s — exiting for K8s restart", n_rows)
                    sys.exit(1)
                gpu_time = gpu_timing.get("total", 0.0)
                feat_time = gpu_timing.get("features", 0.0)
                emit_telemetry(chunk_id=chunk_id, rows=n_rows,
                               gpu_time=gpu_time, feat_time=feat_time, n_files=pending_batch)
                log.info("[INFO] batch %06d: %d rows gpu_time=%.3fs (%d files)",
                         chunk_id, n_rows, gpu_time, pending_batch)
                chunk_id += 1
                pending_batch = None
                continue  # re-check for files immediately
            time.sleep(0.5)
            continue

        # --- If GPU is busy with previous batch, collect result first ---
        if pending_batch is not None:
            try:
                status, n_rows, gpu_timing = _gpu_res_q.get(timeout=600)
            except _queue_module.Empty:
                log.error("[ERROR] GPU worker timeout — exiting for K8s restart")
                sys.exit(1)
            if status != "ok":
                log.error("[ERROR] GPU worker error: %s — exiting for K8s restart", n_rows)
                sys.exit(1)
            gpu_time = gpu_timing.get("total", 0.0)
            feat_time = gpu_timing.get("features", 0.0)
            emit_telemetry(chunk_id=chunk_id, rows=n_rows,
                           gpu_time=gpu_time, feat_time=feat_time, n_files=pending_batch)
            log.info("[INFO] batch %06d: %d rows gpu_time=%.3fs (%d files)",
                     chunk_id, n_rows, gpu_time, pending_batch)
            chunk_id += 1
            pending_batch = None

        # --- Send batch to GPU worker (list of path tuples) ---
        _gpu_req_q.put(batch)
        pending_batch = len(batch)
        # Loop back to claim next batch (prefetch) while GPU processes this one.

    # --- Drain: collect last pending batch on shutdown ---
    if pending_batch is not None:
        try:
            status, n_rows, gpu_timing = _gpu_res_q.get(timeout=600)
            if status == "ok":
                gpu_time = gpu_timing.get("total", 0.0)
                feat_time = gpu_timing.get("features", 0.0)
                emit_telemetry(chunk_id=chunk_id, rows=n_rows,
                               gpu_time=gpu_time, feat_time=feat_time, n_files=pending_batch)
                log.info("[INFO] batch %06d (drain): %d rows gpu_time=%.3fs",
                         chunk_id, n_rows, gpu_time)
        except _queue_module.Empty:
            log.warning("[WARN] GPU worker timeout during drain")

    log.info("[INFO] data-prep shutdown complete after %d batches", chunk_id)


if __name__ == "__main__":
    main()
