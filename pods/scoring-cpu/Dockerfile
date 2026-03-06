FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas \
    pyarrow \
    numpy \
    "tritonclient[http]" \
    torch \
    torch_geometric

COPY pods/scoring-cpu/scorer.py /app/scorer.py
WORKDIR /app
CMD ["python", "scorer.py"]
