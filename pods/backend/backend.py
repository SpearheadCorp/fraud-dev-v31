"""
Pod: backend (v4)
FastAPI control plane + real-time WebSocket dashboard.
Controls continuous pipeline Deployments via kubernetes Python client;
streams metrics to browser.
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

RAW_PATH          = Path(os.environ.get("OUTPUT_PATH",       "/data/raw"))
FEATURES_PATH     = Path(os.environ.get("FEATURES_PATH",    "/data/features"))
SCORES_PATH       = Path(os.environ.get("SCORES_PATH",      "/data/scores"))
MODEL_REPO_PATH   = Path(os.environ.get("MODEL_REPO_PATH",  "/data/models"))
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
state = mt.PipelineState()
collector = mt.MetricsCollector(state)

app = FastAPI(title="Fraud Detection Demo v4", version="4.0.0")

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
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, pl.start_pipeline)
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
            RAW_PATH, FEATURES_PATH, SCORES_PATH,
        )
    )
    telemetry_file = MODEL_REPO_PATH / "last_telemetry.json"
    telemetry_file.unlink(missing_ok=True)
    state.reset()
    return result




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
    log.info("WebSocket client connected")
    loop = asyncio.get_event_loop()
    tick = 0
    _last_full: dict = {}
    try:
        while True:
            try:
                if tick % 10 == 0:
                    # Full collection every 10th tick (2s): pod logs, NFS globs, fraud metrics
                    _last_full = await loop.run_in_executor(None, collector.collect)
                    payload = _last_full
                else:
                    # Fast path (200ms): GPU, CPU, FlashBlade only — merge into last full
                    fast = await loop.run_in_executor(None, collector.collect_fast)
                    payload = {**_last_full, **fast}
            except Exception as exc:
                log.warning("Metrics collect error: %s", exc)
                tick += 1
                await asyncio.sleep(0.2)
                continue
            await websocket.send_json(payload)
            tick += 1
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception:
        log.exception("WebSocket connection closed unexpectedly")


# Alias — some corporate proxies rewrite /ws/ paths
@app.websocket("/data/dashboard")
async def dashboard_ws_alias(websocket: WebSocket):
    await dashboard_ws(websocket)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    # Initialise psutil cpu_percent (first call returns 0.0)
    psutil.cpu_percent(interval=None)
    # Ensure data directories exist
    for p in (RAW_PATH, FEATURES_PATH, SCORES_PATH):
        p.mkdir(parents=True, exist_ok=True)
    # Infer is_running from actual K8s deployment states (survives pod restarts)
    try:
        service_states = pl.get_service_states()
        if any(s not in ("Stopped", "NotFound") for s in service_states.values()):
            state.is_running = True
            state.start_time = state.start_time or time.time()
            log.info("Inferred is_running=True from K8s deployment states")
    except Exception as exc:
        log.warning("Could not infer pipeline state from K8s: %s", exc)
    log.info("Backend started — dashboard at http://0.0.0.0:8080")
