FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build context: repo root (docker build -f pods/backend/Dockerfile .)
COPY pods/backend/ .

# K8s Job templates for pipeline.py
COPY k8s/jobs/ /app/k8s/jobs/

# Seed data (optional — baked in if present in repo; falls back to hardcoded defaults)
COPY seed-data/ /app/seed-data/

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]
