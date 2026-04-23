# docker/airflow.Dockerfile
FROM apache/airflow:2.9.1-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir --timeout 600 --retries 10 \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    Pillow numpy scikit-learn tqdm

RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    mlflow==2.12.2

RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    dvc psycopg2-binary