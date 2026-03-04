FROM python:3.11-slim
WORKDIR /app
COPY pods/data-prep-cpu/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pods/data-prep-cpu/prepare_cpu.py .
CMD ["python", "prepare_cpu.py"]
