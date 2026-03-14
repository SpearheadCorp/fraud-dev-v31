FROM nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.11

WORKDIR /app
COPY pods/scoring/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pods/scoring/scorer.py .
CMD ["python", "scorer.py"]
