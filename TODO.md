# fraud-det-v31 — TODO / Current State

_Last updated: 2026-03-05_

---

## Current Status: Working

The full pipeline runs end-to-end:
- `Start` button → data-gather Job → data-prep (GPU+CPU in parallel) → model-build → inference pods scale up
- Dashboard shows live metrics: TX/s, CPU%, GPU%, FlashBlade read latency, SHAP, training metrics
- `Stress` button re-submits data-gather in `RUN_MODE=continuous` with 32 workers → sustained high TX/s
- GPU utilisation now visible (DCGM ServiceMonitor added)
- Stage values (prep rows, train F1) persist across backend restarts via `last_telemetry.json`

---

## Known Issues / Needs Verification

### 1. FlashBlade read latency showing 0.0 ms
**Status:** Likely — the `purefb_file_systems_performance_latency_usec` metric may not exist
in this cluster's purefb exporter version. The backend falls back to 0.0 gracefully.

**To check:**
```bash
kubectl -n fraud-det-v31 exec deployment/backend -- python3 -c "
import requests
r = requests.get('http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090/api/v1/label/__name__/values', timeout=5)
names = [n for n in r.json()['data'] if 'purefb' in n]
print('\n'.join(names))
"
```
**To fix:** Find the correct metric name from the output above and update `metrics.py`:
```python
# In _collect_flashblade(), change metric name:
("purefb_array_performance_latency_usec", "read", "read_latency_ms"),   # or whatever exists
```

### 2. data-prep GPU path — verify RAPIDS 24.12 works on L40S
**Status:** Untested after Dockerfile fix (changed from 24.02-cuda12.0 to 24.12-cuda12.5).
The subprocess GPU probe in `prepare.py` will catch a SIGSEGV and fall back to CPU.
The dashboard shows a red "(CPU fallback)" badge on the Prep GPU funnel cell when `gpu_used=0`.

**To verify:** Run the pipeline and check the funnel-prep-gpu cell. If it shows "(CPU fallback)",
the GPU path is still failing. Check:
```bash
kubectl -n fraud-det-v31 logs -l job-name=data-prep --tail=50
```

### 3. KAGGLE_SEED_PATH not set in Job YAML
The data-gather job falls back to hardcoded distributions if `KAGGLE_SEED_PATH` is not set.
The seed data file is at `seed-data/credit_card_transactions.csv.zip` in the repo.
To use it, the file needs to be in the data-gather container image OR mounted via a ConfigMap/PVC.
Currently **not mounted** — gather.py uses hardcoded fallback (acceptable for demo).

### 4. inference-cpu image not tested end-to-end
The inference-cpu pod was added recently. Verify it starts and serves predictions after model-build.

### 5. stress → off transition
When stress is turned off, a new `data-gather` job is submitted with `RUN_MODE=once, TARGET_ROWS=1000000`.
The old continuous job is deleted first. Verify the transition is clean (no orphaned pods).

---

## Potential Enhancements

### Dashboard
- [ ] **Alert banner** when GPU path falls back to CPU (currently shown as badge in funnel only)
- [ ] **Pipeline elapsed time** display on infra tab
- [ ] **Model accuracy delta** — show if retrained model improves vs previous run
- [ ] **Inference latency** — add p99 latency from Triton metrics endpoint
  ```
  GET http://inference:8002/metrics  (Triton Prometheus metrics)
  metric: nv_inference_request_duration_us
  ```

### Pipeline
- [ ] **Retrain button** — re-run model-build without re-gathering data (features already on NFS)
- [ ] **Continuous inference loop** — submit inference requests to Triton and show p50/p99 latency
- [ ] **Per-GPU telemetry** — DCGM returns 4 results (2 GPUs × 2 nodes); dashboard currently shows `gpu_0` only. Could show the L40S on node .44 specifically.

### Ops
- [ ] **Pre-pull images on .44** before a demo to avoid cold-pull delay on first Start:
  ```bash
  for img in data-gather data-prep data-prep-cpu model-build inference inference-cpu; do
    kubectl -n fraud-det-v31 create job prepull-$img --image=10.23.181.247:5000/fraud-det-v31/$img:latest -- echo ok
    kubectl -n fraud-det-v31 delete job prepull-$img
  done
  ```
- [ ] **Node .40 GPU worker** — `nodeSelector` currently hardcoded to `slc6-lg-n3-b30-29` (.44).
  If workload should run on .40, update the Job YAMLs and ensure containerd hosts.toml is configured on .40.
- [ ] **NFS disk monitoring** — add alert or dashboard indicator when raw-data-pvc > 80% full
  (stress mode continuous can fill 500Gi fast at high throughput)

---

## Pre-Demo Checklist

```bash
# 1. Backend running?
kubectl get pods -n fraud-det-v31

# 2. Registry healthy?
ssh -i ~/.ssh/id_rsa tduser@10.23.181.247 "curl -s http://localhost:5000/v2/_catalog | python3 -m json.tool"

# 3. Disk on build VM?
ssh -i ~/.ssh/id_rsa tduser@10.23.181.247 "df -h /"
# Should show > 50GB free. If low: docker builder prune -af && docker image prune -af

# 4. Open dashboard
# http://10.23.181.44:30880

# 5. Click Start — verify all 4 stages show green in status tab within ~5 min
```

---

## Commit History (this session — 2026-03-04/05)

```
a08377a Fix prep/train '--', enable GPU metrics, latency scale 2-3ms
d4b0d70 Fix latency Y-axis scale to 2–3 ms (expected FlashBlade range)
8b7b7d7 Fix latency chart: use read latency (hot path for ML workloads)
063022e Add FlashBlade latency chart, 3rd Y-axis, fix stress continuous mode
27da420 Fix RAPIDS base image tag: 24.12-cuda12.5-py3.11 (driver 580 / L40S compatible)
4ae284d Fix GPU crash: upgrade RAPIDS to 25.02/CUDA 12.6 for driver 580 (L40S)
93a828f Add TX/s chart, combined 4-line infra chart, pipeline funnel display
a6da4e4 Fix stop/reset UX and telemetry persistence across job restarts
76f9a62 Manage inference pod lifecycle via start/stop — not always-on
de3e550 Remove own Prometheus; use cluster kube-prometheus-stack instead
f7d25c6 Fix pipeline stability and demo UX issues (code review pass)
878ec0c Add CPU pods for GPU vs CPU side-by-side comparison demo
```
