# docker/streamlit.Dockerfile
# ----------------------------
# Streamlit frontend
# Uses python:3.10-slim base and installs torch via index URL
# to avoid timeout issues with large wheels

FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install torch FIRST using PyTorch's own CDN index
# This is faster and more reliable than PyPI for large wheels
RUN pip install --no-cache-dir \
    --timeout 600 \
    --retries 10 \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install remaining dependencies
COPY docker/requirements.api.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    -r requirements.api.txt

# Step 3: Install streamlit
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    streamlit\
    pandas==2.2.2 \
    prometheus-client==0.20.0

# Copy source
COPY src/ ./src/
COPY params.yaml .

RUN mkdir -p models data/processed reports logs

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]
