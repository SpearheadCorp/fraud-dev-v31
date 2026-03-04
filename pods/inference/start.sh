#!/bin/bash
set -e

MODEL_REPO="${MODEL_REPO:-/data/models}"

echo "[INFO] Waiting for model repository at ${MODEL_REPO}..."
until [ -f "${MODEL_REPO}/fraud_xgboost_gpu/1/xgboost.json" ]; do
    echo "[INFO] Models not ready yet, waiting 10s..."
    sleep 10
done

echo "[INFO] Models found. Starting Triton Inference Server..."
exec tritonserver \
    --model-repository="${MODEL_REPO}" \
    --model-control-mode=poll \
    --repository-poll-secs=30 \
    --strict-model-config=false \
    --log-verbose=0 \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
