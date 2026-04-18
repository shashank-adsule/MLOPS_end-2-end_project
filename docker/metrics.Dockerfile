# docker/metrics.Dockerfile
# --------------------------
# Lightweight FastAPI sidecar that exposes /metrics for Prometheus
# and /health + /predict endpoints

FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

COPY src/ ./src/
COPY params.yaml .

RUN mkdir -p models data/processed logs

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
