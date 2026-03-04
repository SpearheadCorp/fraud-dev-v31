"""
Pod 5: Backend
FastAPI control plane + real-time WebSocket dashboard.
Controls K8s Jobs pipeline via kubernetes Python client; streams metrics to browser.
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

RAW_PATH = Path(os.environ.get("RAW_DATA_PATH", "/data/raw"))
FEATURES_PATH = Path(os.environ.get("FEATURES_DATA_PATH", "/data/features"))
FEATURES_CPU_PATH = Path(os.environ.get("FEATURES_CPU_DATA_PATH", "/data/features-cpu"))
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
state = mt.PipelineState()
collector = mt.MetricsCollector(state)

app = FastAPI(title="Fraud Detection Demo v3.1", version="3.1.0")

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
    return {
        "is_running": state.is_running,
        "stress_mode": state.stress_mode,
        "elapsed_sec": state.elapsed_sec,
        "services": service_states,
    }


async def _run_pipeline_task(overrides: dict) -> None:
    """Run pipeline in a thread so the async event loop stays responsive."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: pl.start_pipeline(overrides))
    state.is_running = False
    if result.get("status") == "error":
        log.error("[ERROR] Pipeline failed at stage '%s': %s", result.get("stage"), result.get("message"))
    else:
        log.info("[INFO] Pipeline completed: %s", result.get("message"))


@app.post("/api/control/start")
async def start_pipeline():
    if state.is_running:
        return {"status": "already_running", "message": "Pipeline is already running"}
    state.is_running = True
    state.start_time = time.time()
    asyncio.create_task(_run_pipeline_task({}))
    return {"status": "started", "message": "Pipeline started in background"}


@app.post("/api/control/stop")
async def stop_pipeline():
    result = pl.stop_pipeline()
    state.is_running = False
    return result


@app.post("/api/control/reset")
async def reset_pipeline():
    result = pl.reset_pipeline(RAW_PATH, FEATURES_PATH, FEATURES_CPU_PATH)
    state.reset()
    return result


@app.post("/api/control/stress")
async def start_stress():
    pl.write_stress_config(stress_on=True, num_workers=32, chunk_size=200000)
    state.stress_mode = True
    return {"status": "stress mode activated"}


@app.post("/api/control/stress-stop")
async def stop_stress():
    pl.write_stress_config(stress_on=False)
    state.stress_mode = False
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
                await websocket.send_json(payload)
            except Exception as exc:
                log.warning("[WARN] metrics collect error: %s", exc)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        log.info("[INFO] WebSocket client disconnected")
    except Exception as exc:
        log.warning("[WARN] WebSocket error: %s", exc)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    # Initialise psutil cpu_percent (first call returns 0.0)
    psutil.cpu_percent(interval=None)
    # Ensure data directories exist
    for p in (RAW_PATH, FEATURES_PATH, FEATURES_CPU_PATH):
        p.mkdir(parents=True, exist_ok=True)
    log.info("[INFO] Backend started — dashboard at http://0.0.0.0:8080")
