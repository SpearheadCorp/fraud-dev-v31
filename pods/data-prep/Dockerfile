FROM nvcr.io/nvidia/rapidsai/base:25.02-cuda12.6-py3.11

WORKDIR /app

COPY pods/data-prep/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pods/data-prep/prepare.py .

CMD ["python", "prepare.py"]
