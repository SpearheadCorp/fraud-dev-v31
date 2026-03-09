"""
Pod: backend
FastAPI control plane + real-time WebSocket dashboard.
Controls continuous pipeline Deployments via kubernetes Python client;
streams metrics to browser. model-build is run manually (offline).
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline as pl
import metrics as mt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

RAW_PATH_GPU      = Path(os.environ.get("OUTPUT_PATH_GPU",      "/data/raw/gpu"))
RAW_PATH_CPU      = Path(os.environ.get("OUTPUT_PATH_CPU",      "/data/raw/cpu"))
FEATURES_GPU_PATH = Path(os.environ.get("FEATURES_GPU_PATH",   "/data/features/gpu"))
FEATURES_CPU_PATH = Path(os.environ.get("FEATURES_CPU_DATA_PATH", "/data/features-cpu"))
SCORES_GPU_PATH   = Path(os.environ.get("SCORES_GPU_PATH",     "/data/features/scores"))
SCORES_CPU_PATH   = Path(os.environ.get("SCORES_CPU_PATH",     "/data/features-cpu/scores"))
MODEL_REPO_PATH   = Path(os.environ.get("MODEL_REPO_PATH",     "/data/models"))
STRESS_CONFIG_PATH = Path(os.environ.get("STRESS_CONFIG_PATH", "/data/raw/.stress.conf"))
STATIC_DIR = Path(__file__).parent / "static"

# Gather worker config written to STRESS_CONFIG_PATH for hot-reload by data-gather.
GATHER_STRESS_WORKERS = int(os.environ.get("GATHER_STRESS_WORKERS", "8"))
GATHER_STRESS_RATE    = int(os.environ.get("GATHER_STRESS_RATE",    "40000"))
GATHER_NORMAL_WORKERS = int(os.environ.get("GATHER_NORMAL_WORKERS", "2"))
GATHER_NORMAL_RATE    = int(os.environ.get("GATHER_NORMAL_RATE",    "10000"))


def _write_gather_config(workers: int, rate: int) -> None:
    """Write data-gather hot-reload config to shared NFS path."""
    try:
        STRESS_CONFIG_PATH.write_text(f"NUM_WORKERS={workers}\nTARGET_ROWS_PER_SEC={rate}\n")
    except Exception as exc:
        log.warning("[WARN] Failed to write gather config: %s", exc)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
state = mt.PipelineState()
collector = mt.MetricsCollector(state)

app = FastAPI(title="Fraud Detection Demo v3.2", version="3.2.0")

# Serve static files (dashboard.html)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = STATIC_DIR / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


@app.get("/api/status")
async def get_status():
    service_states = pl.get_service_states()
    replicas       = pl.get_replica_counts()
    return {
        "is_running":  state.is_running,
        "stress_mode": state.stress_mode,
        "elapsed_sec": state.elapsed_sec,
        "services":    service_states,
        "replicas":    replicas,
    }


@app.post("/api/control/start")
async def start_pipeline():
    if state.is_running:
        return {"status": "already_running", "message": "Pipeline is already running"}
    state.is_running = True
    state.start_time = time.time()
    _write_gather_config(GATHER_NORMAL_WORKERS, GATHER_NORMAL_RATE)  # ensure clean state on start
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pl.start_pipeline)
    return {"status": "started", "message": "Deployments scaled up"}


@app.post("/api/control/stop")
async def stop_pipeline():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pl.stop_pipeline)
    state.is_running = False
    return result


@app.post("/api/control/reset")
async def reset_pipeline():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: pl.reset_pipeline(
            RAW_PATH_GPU, RAW_PATH_CPU,
            FEATURES_GPU_PATH, FEATURES_CPU_PATH,
            SCORES_GPU_PATH, SCORES_CPU_PATH,
        )
    )
    telemetry_file = MODEL_REPO_PATH / "last_telemetry.json"
    telemetry_file.unlink(missing_ok=True)
    state.reset()
    return result


@app.post("/api/control/clear-data")
async def clear_data():
    """Delete data files only — call after pods are already stopped."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: pl.clear_data_files(
            RAW_PATH_GPU, RAW_PATH_CPU,
            FEATURES_GPU_PATH, FEATURES_CPU_PATH,
            SCORES_GPU_PATH, SCORES_CPU_PATH,
        )
    )
    telemetry_file = MODEL_REPO_PATH / "last_telemetry.json"
    telemetry_file.unlink(missing_ok=True)
    state.reset()
    return result


@app.post("/api/control/stress")
async def start_stress():
    state.stress_mode = True
    state.last_telemetry.pop("gather", None)
    _write_gather_config(GATHER_STRESS_WORKERS, GATHER_STRESS_RATE)
    pl.write_stress_config(True)
    return {"status": "stress mode activated"}


@app.post("/api/control/stress-stop")
async def stop_stress():
    state.stress_mode = False
    state.last_telemetry.pop("gather", None)
    _write_gather_config(GATHER_NORMAL_WORKERS, GATHER_NORMAL_RATE)
    pl.write_stress_config(False)
    return {"status": "stress mode deactivated"}


@app.get("/api/metrics/current")
async def get_current_metrics():
    return collector.collect()


@app.get("/api/metrics/shap")
async def get_shap():
    data = mt.load_shap_summary()
    if data is None:
        return JSONResponse({"error": "SHAP data not available yet"}, status_code=404)
    return data


@app.get("/api/metrics/training")
async def get_training_metrics():
    data = mt.load_training_metrics()
    if data is None:
        return JSONResponse({"error": "Training metrics not available yet"}, status_code=404)
    return data


@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus text-format metrics for scraping."""
    m = collector.collect()
    sys_m = m.get("system", {})
    gpu_m = m.get("gpu", {})
    gather = m.get("pipeline", {}).get("gather", {})
    lines = [
        "# HELP fraud_det_cpu_percent CPU utilization percent",
        "# TYPE fraud_det_cpu_percent gauge",
        f"fraud_det_cpu_percent {sys_m.get('cpu_percent', 0)}",
        "# HELP fraud_det_ram_percent RAM utilization percent",
        "# TYPE fraud_det_ram_percent gauge",
        f"fraud_det_ram_percent {sys_m.get('ram_percent', 0)}",
        "# HELP fraud_det_rows_generated Total rows generated",
        "# TYPE fraud_det_rows_generated counter",
        f"fraud_det_rows_generated {gather.get('rows_generated', 0)}",
        "# HELP fraud_det_throughput_mbps Current data throughput MB/s",
        "# TYPE fraud_det_throughput_mbps gauge",
        f"fraud_det_throughput_mbps {gather.get('throughput_mbps', 0)}",
        "# HELP fraud_det_is_running Pipeline running flag",
        "# TYPE fraud_det_is_running gauge",
        f"fraud_det_is_running {1 if state.is_running else 0}",
    ]
    for k, v in gpu_m.items():
        safe_key = k.replace("-", "_")
        lines += [
            f"# HELP fraud_det_{safe_key} GPU metric",
            f"# TYPE fraud_det_{safe_key} gauge",
            f"fraud_det_{safe_key} {v}",
        ]
    return HTMLResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    log.info("[INFO] WebSocket client connected")
    try:
        while True:
            try:
                payload = collector.collect()
            except Exception as exc:
                log.warning("[WARN] metrics collect error: %s", exc)
                await asyncio.sleep(1.0)
                continue
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        log.info("[INFO] WebSocket client disconnected")
    except Exception:
        log.info("[INFO] WebSocket connection closed")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    # Initialise psutil cpu_percent (first call returns 0.0)
    psutil.cpu_percent(interval=None)
    # Ensure data directories exist
    for p in (RAW_PATH_GPU, RAW_PATH_CPU, FEATURES_GPU_PATH, FEATURES_CPU_PATH,
              SCORES_GPU_PATH, SCORES_CPU_PATH):
        p.mkdir(parents=True, exist_ok=True)
    # Infer is_running from actual K8s deployment states (survives pod restarts)
    try:
        service_states = pl.get_service_states()
        if any(s not in ("Stopped", "NotFound") for s in service_states.values()):
            state.is_running = True
            state.start_time = state.start_time or time.time()
            log.info("[INFO] Inferred is_running=True from K8s deployment states")
    except Exception as exc:
        log.warning("[WARN] Could not infer pipeline state from K8s: %s", exc)
    log.info("[INFO] Backend started — dashboard at http://0.0.0.0:8080")
