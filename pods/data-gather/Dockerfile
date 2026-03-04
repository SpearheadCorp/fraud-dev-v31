# Build context: repo root (docker build -f pods/data-gather/Dockerfile .)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pods/data-gather/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pods/data-gather/gather.py .

# Seed data baked in at build time (optional — falls back to hardcoded defaults)
COPY seed-data/ /app/seed-data/

CMD ["python", "gather.py"]
