FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    pyarrow==18.1.0 \
    numpy==1.26.4 \
    "tritonclient[http]==2.53.0" \
    torch==2.7.0 \
    torch_geometric==2.6.1

COPY pods/scoring/scorer.py /app/scorer.py
WORKDIR /app
CMD ["python", "scorer.py"]
