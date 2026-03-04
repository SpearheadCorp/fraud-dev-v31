#!/bin/bash
set -e

MODEL_REPO="${MODEL_REPO:-/data/models}"

echo "[INFO] Waiting for model repository at ${MODEL_REPO}..."
until [ -s "${MODEL_REPO}/fraud_xgboost_gpu/1/xgboost.json" ] && \
      [ -f "${MODEL_REPO}/fraud_xgboost_gpu/config.pbtxt" ]; do
    echo "[INFO] GPU model not ready yet (waiting for non-empty xgboost.json + config.pbtxt), waiting 10s..."
    sleep 10
done

echo "[INFO] GPU model ready. Starting Triton Inference Server..."
exec tritonserver \
    --model-repository="${MODEL_REPO}" \
    --model-control-mode=poll \
    --repository-poll-secs=30 \
    --strict-model-config=false \
    --log-verbose=0 \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
