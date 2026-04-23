# docker/airflow.Dockerfile
# Pre-bakes all heavy Python dependencies so the container starts in seconds.
# CUDA-enabled torch for NVIDIA RTX 3050 (CUDA 13.1 driver).
# cu121 wheel is forward-compatible with CUDA 13.1 drivers.
#
# Each dependency is in its OWN RUN layer so:
#   - Docker layer cache means a timeout on step N won't re-download steps 1..N-1
#   - --timeout 600 + --retries 10 handles slow/flaky PyPI connections

FROM apache/airflow:2.9.1-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Step 1: torch + torchvision from PyTorch CDN (large wheels ~2GB, high timeout)
RUN pip install --no-cache-dir --timeout 600 --retries 10 \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Step 2: lighter deps (separate layer — torch won't re-download if this fails)
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    Pillow \
    numpy \
    scikit-learn \
    tqdm

# Step 3: mlflow separately (20MB wheel — was the one that timed out)
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    mlflow==2.12.2

# Step 4: dvc
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    dvc
