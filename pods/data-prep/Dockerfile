FROM nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.11

WORKDIR /app

COPY pods/data-prep/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pods/data-prep/prepare.py .
COPY pods/data-prep/gpu_worker.py .

CMD ["python", "prepare.py"]
