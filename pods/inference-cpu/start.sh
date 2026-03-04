#!/bin/bash
set -e

MODEL_REPO="${MODEL_REPO:-/data/models}"

echo "[INFO] Waiting for CPU model at ${MODEL_REPO}/fraud_xgboost_cpu/1/xgboost.json..."
until [ -f "${MODEL_REPO}/fraud_xgboost_cpu/1/xgboost.json" ]; do
    echo "[INFO] CPU model not ready yet, waiting 10s..."
    sleep 10
done

echo "[INFO] CPU model found. Starting Triton (CPU-only)..."
exec tritonserver \
    --model-repository="${MODEL_REPO}" \
    --model-control-mode=explicit \
    --load-model=fraud_xgboost_cpu \
    --strict-model-config=false \
    --log-verbose=0 \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
