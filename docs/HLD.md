# High Level Design — MLOps End-to-End Surface Defect Detection

**Student:** DA25M005 (Shashank Adsule)
**Repository:** https://github.com/shashank-adsule/MLOPS_end-2-end_project
**Version:** 1.0 | **Date:** April 2026

---

## 1. Problem Statement

Manufacturing lines produce defective products that escape visual inspection due to operator fatigue, lighting inconsistency, and high throughput speeds. The goal is to build an automated surface defect detection system that:

- Classifies each product image as **normal** or **defective** in real time
- Produces a **pixel-level anomaly heatmap** localising the defect region
- Covers **15 product categories** from the MVTec AD benchmark dataset
- Operates with **no labelled defect examples** at training time (anomaly detection, not classification)
- Meets a **200 ms per-image SLA** under GPU inference
- Exposes real-time metrics for **production observability**

---

## 2. System Overview

The system is a full MLOps pipeline from raw data to monitored production inference. It is composed of six distinct planes:

```
┌──────────────────────────────────────────────────────────────────────┐
│  DATA PLANE          │  MODEL PLANE         │  SERVING PLANE         │
│  Raw MVTec AD        │  PatchCore Training  │  Streamlit App         │
│  DVC Versioning      │  MLflow Tracking     │  Prometheus Metrics    │
│  Airflow Pipeline    │  Model Registry      │  :8501  :8000          │
├──────────────────────┴──────────────────────┴────────────────────────┤
│  OBSERVABILITY PLANE                                                  │
│  Prometheus :9090  →  Alertmanager :9093  →  Grafana :3001           │
├──────────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION PLANE          │  INFRASTRUCTURE PLANE                │
│  Airflow :8081                │  Docker Compose / defect_net         │
│  2 DAGs                       │  Postgres :5432                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture Diagram

```
                          ┌─────────────────────────────────────────────────────┐
                          │               Docker Network: defect_net             │
                          │                                                      │
   User / Browser         │   ┌──────────────┐         ┌──────────────────┐    │
        │                 │   │  Streamlit   │         │    Prometheus    │    │
        │ :8501           │   │  app.py      │◄────────│    :9090         │    │
        ▼                 │   │  (local GPU) │         │                  │    │
   ┌─────────┐            │   │  :8501 :8000 │         │  scrapes         │    │
   │Streamlit│            │   └──────┬───────┘         │  host.docker     │    │
   │  UI     │            │          │ predict()        │  .internal:8000  │    │
   └────┬────┘            │          ▼                  └────────┬─────────┘    │
        │                 │   ┌──────────────┐                   │ alerts       │
        │ upload image    │   │  PatchCore   │                   ▼              │
        │                 │   │  .pt model   │         ┌──────────────────┐    │
        │                 │   │  (CUDA)      │         │  Alertmanager    │    │
        │                 │   └──────────────┘         │  :9093           │    │
        │                 │                             └──────────────────┘    │
        │                 │   ┌──────────────┐                                  │
        │                 │   │   MLflow     │         ┌──────────────────┐    │
        │                 │   │   :5000      │         │   Grafana        │    │
        │                 │   └──────────────┘         │   :3001          │    │
        │                 │                             │   13 panels      │    │
        │                 │   ┌──────────────┐         └──────────────────┘    │
        │                 │   │   Airflow    │                                  │
        │                 │   │   :8081      │──┐      ┌──────────────────┐    │
        │                 │   └──────────────┘  └─────►│   Postgres       │    │
        │                 │                             │   :5432          │    │
        │                 └─────────────────────────────┴──────────────────┘    │
        │                                                                        │
        │                 Remote Services:                                       │
        │                   DagsHub MLflow  ← experiment runs, artifacts        │
        └─────────────────  DagsHub DVC     ← raw + processed data versioning   │
```

---

## 4. Component Descriptions

### 4.1 Data Layer

**MVTec AD Dataset** — 15 industrial product categories, 6,644 images, 5.2 GB. Each category has only normal (defect-free) images in the training split, making this an unsupervised anomaly detection problem.

**DVC** tracks both raw and processed data with the remote store on DagsHub. Any team member can reproduce the exact data state with `dvc pull`.

**Airflow** (`mvtec_ad_data_pipeline` DAG) orchestrates the three-stage data pipeline: ingest → preprocess → feature engineering. It runs on PostgreSQL (not SQLite) to support concurrent scheduler + webserver access.

### 4.2 Model Layer

**PatchCore** is a memory-bank based anomaly detection algorithm. During training it:
1. Passes all normal training images through a frozen WideResNet50 backbone
2. Extracts multi-scale patch features from `layer2` and `layer3`
3. Applies greedy coreset subsampling (10%) to produce a compact memory bank

At inference, the anomaly score for an image is the maximum nearest-neighbour distance from its patches to the memory bank. A Gaussian-smoothed heatmap shows which regions are anomalous.

**MLflow** tracks every training run with parameters, metrics, and artifacts to both DagsHub (primary, full artifacts) and a local instance at `localhost:5000` (secondary, params + metrics only). If the local instance is down, DagsHub logging continues unaffected.

### 4.3 Serving Layer

**Streamlit** provides the user-facing interface with four pages: defect detection, pipeline dashboard, experiment metrics, and prediction history. It runs locally (not in Docker) to access the NVIDIA GPU directly via CUDA.

A custom **`_MetricsHandler` HTTP server** runs on port 8000 in a background thread and exposes Prometheus metrics. Because Streamlit re-executes the script in a new thread on each interaction, all metric state is persisted through a JSON file in `%TEMP%` — the only reliable cross-thread state mechanism in this architecture.

### 4.4 Observability Layer

**Prometheus** scrapes `host.docker.internal:8000` every 10 seconds. It reaches the locally-running Streamlit metrics server from inside Docker using Docker's host network bridge.

**Alertmanager** receives fired alerts from Prometheus. Currently configured with null receivers (alert logging only) — no email/webhook configured, which avoids dependency on external SMTP services.

**Grafana** auto-provisions a 13-panel dashboard at startup. All panels use `uid: prometheus` to reference the datasource provisioned by the `grafana/provisioning/datasources/prometheus.yml` file.

### 4.5 Orchestration Layer

**Airflow** manages two DAGs:

| DAG | Trigger | Purpose |
|---|---|---|
| `mvtec_ad_data_pipeline` | Manual | End-to-end data preparation |
| `app_down_response_dag` | Manual only | 5-step incident response workflow |

PostgreSQL replaces SQLite as the Airflow metadata database. SQLite's write lock caused gunicorn timeouts when the scheduler and webserver accessed it concurrently.

---

## 5. Data Flow

```
Raw PNG images (MVTec AD)
        │
        ▼ Airflow: ingest.py
   Validation + file listing
        │
        ▼ Airflow: preprocess.py
   Resize(256) → CenterCrop(224) → Normalize(ImageNet)
   Saved as .pt tensors → data/processed/{category}/
        │
        ▼ Airflow: feature_engineering.py
   Per-category baseline stats → data/processed/{category}/stats/baseline.json
   (used for drift detection at inference time)
        │
        ▼ train.py
   PatchCore.fit() — build coreset memory bank
   MLflow logging → DagsHub + localhost:5000
   Saved to models/patchcore_{category}.pt
        │
        ▼ evaluate.py
   AUROC, F1, pixel-AUROC, latency
   ROC curve + heatmap samples → reports/{category}/
        │
        ▼ app.py (Streamlit)
   User uploads image
   PatchCore.predict() → anomaly score + heatmap
   Drift check via check_drift()
   Metrics written to app_metrics_state.json
        │
        ▼ Prometheus scrapes :8000/metrics
   Grafana reads → dashboard panels update
```

---

## 6. Key Design Decisions

| Decision | Chosen Approach | Alternative Considered | Reason |
|---|---|---|---|
| Anomaly detection algorithm | PatchCore (memory bank) | Supervised CNN classifier | MVTec AD has no labelled defect training images; PatchCore requires only normal samples |
| Feature layers | layer2 + layer3 | layer3 only, layer4 | layer2 gives spatial resolution; layer3 gives semantic depth; layer4 is too abstract |
| Coreset ratio | 10% | 5% or 20% | Balances memory bank compactness vs accuracy; all categories achieve AUROC > 0.84 |
| Metrics state sharing | JSON file in %TEMP% | Redis, multiprocessing shared memory | Zero additional infrastructure; works across Streamlit's thread isolation model |
| Streamlit deployment | Local (not Docker) | Dockerised with GPU passthrough | Windows Docker GPU passthrough requires WSL2 config changes; local is simpler and equally observable |
| Airflow backend | PostgreSQL | SQLite | SQLite write lock causes gunicorn timeout under concurrent scheduler + webserver access |
| MLflow dual logging | DagsHub primary + local secondary | Single remote only | Allows offline inspection of metrics without internet; graceful degradation if local is down |

---

## 7. Technology Stack Summary

| Layer | Technology | Version |
|---|---|---|
| ML Algorithm | PatchCore (custom) | — |
| Backbone | WideResNet50 | torchvision 0.17 |
| Framework | PyTorch | 2.2.2 |
| Experiment Tracking | MLflow | 2.12.2 |
| Data Versioning | DVC | 3.50.0 |
| Pipeline Orchestration | Apache Airflow | 2.9.1 |
| Airflow Database | PostgreSQL | 15-alpine |
| Serving UI | Streamlit | 1.35.0 |
| Metrics Collection | Prometheus | 2.52.0 |
| Alerting | Alertmanager | 0.27.0 |
| Dashboarding | Grafana | 10.4.2 |
| Containerisation | Docker Compose | — |
| Remote Storage | DagsHub | — |
| GPU | NVIDIA RTX 3050 6 GB | CUDA 13.1 |
| Python | CPython | 3.11.9 |
| OS | Windows 10 | — |

---

## 8. Non-Functional Requirements

| Requirement | Target | Achieved |
|---|---|---|
| Inference latency (P95) | < 200 ms | ✅ All 15 categories < 200 ms (worst: 166 ms) |
| Detection accuracy (AUROC) | > 0.85 | ✅ Avg 0.9643; min 0.84 (grid) |
| Reproducibility | DVC + MLflow | ✅ Every run tracked with git hash + DVC version |
| Observability | Prometheus + Grafana | ✅ 9 metrics, 13 dashboard panels, 5 alert rules |
| Data versioning | DVC remote | ✅ Full dataset versioned on DagsHub |
| Pipeline automation | Airflow DAG | ✅ 2 DAGs; data pipeline + incident response |
| Zero-dependency deployment | Docker Compose | ✅ 6 services in single `docker compose up` |

---

## 9. Security and Configuration

- Secrets (DagsHub token, MLflow credentials) stored in `.env` and never committed (`.gitignore` enforced)
- `.env.example` provides the template without values
- All Docker services run on an isolated bridge network `defect_net` — only mapped ports are externally accessible
- Grafana admin password set via `GF_SECURITY_ADMIN_PASSWORD` env var
- Airflow uses a static `AIRFLOW__WEBSERVER__SECRET_KEY` (should be rotated in production)

---

**Prepared by:** Shashank Adsule — DA25M005
**Date:** April 2026
