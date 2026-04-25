# MLOps End-to-End Project — Surface Defect Detection

**Student:** DA25M005 (Shashank Adsule)  
**Repository:** https://github.com/shashank-adsule/MLOPS_end-2-end_project  
**DagsHub:** https://dagshub.com/da25m005/MLOPS_end-2-end_project

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Complete Project Structure](#2-complete-project-structure)
3. [Architecture](#3-architecture)
4. [Tech Stack](#4-tech-stack)
5. [How Every File Is Used](#5-how-every-file-is-used)
6. [Full Workflow — Step by Step](#6-full-workflow--step-by-step)
7. [ML Pipeline](#7-ml-pipeline)
8. [Unit Tests — How to Run](#8-unit-tests--how-to-run)
9. [CI/CD Pipeline — How It Works](#9-cicd-pipeline--how-it-works)
10. [Model Performance](#10-model-performance)
11. [Prometheus Metrics](#11-prometheus-metrics)
12. [Alerting Rules](#12-alerting-rules)
13. [Grafana Dashboard](#13-grafana-dashboard)
14. [MLflow Experiment Tracking](#14-mlflow-experiment-tracking)
15. [Airflow DAGs](#15-airflow-dags)
16. [Design Documents](#16-design-documents)
17. [Port Reference](#17-port-reference)
18. [Environment Variables](#18-environment-variables)
19. [All Run Commands Reference](#19-all-run-commands-reference)
20. [Completion Checklist](#20-completion-checklist)

---

## 1. Project Overview

| Field | Detail |
|---|---|
| Problem | Automated surface defect detection in manufacturing |
| Dataset | MVTec AD (15 categories, 6,644 images, 5.2 GB) |
| Algorithm | PatchCore (WideResNet50 backbone) |
| Local Path | `D:\code\repo\MLOPS_end-2-end_project` |
| Virtual Env | `E2E` (Python 3.11.9) |
| OS | Windows 10 |
| GPU | NVIDIA RTX 3050 6 GB (CUDA 13.1) |

This project builds a **production-grade MLOps pipeline** for automated visual quality control. A PatchCore model is trained on normal (defect-free) images for each of 15 MVTec AD product categories. At inference time, images that deviate from the learned normal distribution are flagged as defective, with a pixel-level heatmap showing the exact defect location.

The full stack covers: data versioning (DVC), pipeline orchestration (Airflow), experiment tracking (MLflow on DagsHub), real-time observability (Prometheus + Grafana + Alertmanager), a Streamlit frontend, automated testing (pytest), and CI/CD (GitHub Actions).

---

## 2. Complete Project Structure

```
MLOPS_end-2-end_project/
│
├── .github/
│   └── workflows/
│       └── ci.yml                        ← GitHub Actions CI/CD (5 jobs)
│
├── src/
│   ├── app.py                            ← Streamlit UI + Prometheus metrics server (:8501, :8000)
│   ├── __init__.py
│   ├── model/
│   │   ├── patchcore.py                  ← PatchCore model class (fit, predict, save, load)
│   │   ├── train.py                      ← Training script — dual MLflow logging
│   │   └── evaluate.py                   ← AUROC, F1, pixel-AUROC, latency
│   ├── pipeline/
│   │   ├── ingest.py                     ← Raw data validation
│   │   ├── preprocess.py                 ← Resize/crop/normalise → .pt tensors
│   │   ├── feature_engineering.py        ← Drift baseline computation
│   │   ├── airflow_dag.py                ← mvtec_ad_data_pipeline DAG
│   │   └── app_down_response_dag.py      ← Incident response DAG
│   └── monitoring/
│       └── drift_detector.py             ← Z-score drift detector
│
├── tests/                                ← Unit test suite (55 tests, 5 files)
│   ├── __init__.py
│   ├── conftest.py                       ← Shared fixtures (images, tensors, dirs, PatchCore)
│   ├── test_patchcore.py                 ← PatchCore model tests (20 tests)
│   ├── test_feature_engineering.py       ← Drift baseline + check_drift tests (15 tests)
│   ├── test_ingest.py                    ← Data validation tests (12 tests)
│   ├── test_preprocess.py                ← Preprocessing pipeline tests (10 tests)
│   └── test_evaluate.py                  ← Evaluation module tests (13 tests)
│
├── docs/
│   ├── HLD.md                            ← High Level Design document
│   └── LLD.md                            ← Low Level Design + API specifications
│
├── monitoring/
│   ├── prometheus.yml                    ← Scrape config (host.docker.internal:8000)
│   ├── alert_rules.yml                   ← 5 alerting rules
│   ├── alertmanager.yml                  ← Alert routing (null receivers)
│   └── grafana/
│       ├── dashboards/
│       │   └── defect_detection.json     ← 13-panel auto-provisioned dashboard
│       └── provisioning/
│           ├── dashboards/dashboards.yml
│           └── datasources/prometheus.yml
│
├── docker/
│   ├── airflow.Dockerfile                ← Pre-bakes CUDA torch at build time
│   ├── streamlit.Dockerfile
│   ├── metrics.Dockerfile
│   └── requirements.api.txt
│
├── data/
│   ├── raw/mvtec_ad/                     ← DVC-tracked (15 categories, 5.2 GB)
│   └── processed/                        ← DVC-tracked .pt tensors + baselines
│
├── models/                               ← Trained patchcore_{category}.pt files
├── reports/                              ← eval_metrics.json, roc_curve.csv, heatmaps
│   └── metrics.json                      ← Aggregated results (all 15 categories)
│
├── docker-compose.yml                    ← 6 Docker services on defect_net
├── params.yaml                           ← Pipeline parameters
├── requirements-pipeline.txt
├── .env                                  ← Secrets — never committed
├── .env.example
├── .dvcignore
└── .gitignore
```

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Docker Network: defect_net                      │
│                                                                    │
│   ┌───────────────┐  GET :8000/metrics  ┌──────────────────────┐ │
│   │  Streamlit    │◄────────────────────│                      │ │
│   │  app.py       │  host.docker        │    Prometheus        │ │
│   │  :8501 :8000  │  .internal:8000     │    :9090             │ │
│   │  (local GPU)  │                     │                      │ │
│   └───────────────┘                     └──────────┬───────────┘ │
│                                                     │ alerts      │
│   ┌───────────────┐                                 ▼             │
│   │   MLflow      │                      ┌──────────────────────┐ │
│   │   :5000       │                      │   Alertmanager       │ │
│   └───────────────┘                      │   :9093              │ │
│                                          └──────────────────────┘ │
│   ┌───────────────┐                                               │
│   │   Grafana     │──PromQL queries──────────────────────────────►│
│   │   :3001       │                                               │
│   └───────────────┘                                               │
│                                                                    │
│   ┌───────────────┐      ┌────────────────┐                      │
│   │   Airflow     │─────►│   Postgres     │                      │
│   │   :8081       │      │   :5432        │                      │
│   └───────────────┘      └────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘

CI/CD:   GitHub Actions → lint + tests + config validation + Docker build
Remote:  DagsHub (MLflow experiments + DVC data)
```

> **Key design decision:** Streamlit runs locally (not in Docker) so it can access the NVIDIA GPU via CUDA. Prometheus reaches it from inside Docker via `host.docker.internal:8000`.

---

## 4. Tech Stack

| Service | Image / Version | Port | Credentials |
|---|---|---|---|
| MLflow | `ghcr.io/mlflow/mlflow:v2.12.2` | 5000 | — |
| Prometheus | `prom/prometheus:v2.52.0` | 9090 | — |
| Alertmanager | `prom/alertmanager:v0.27.0` | 9093 | — |
| Grafana | `grafana/grafana:10.4.2` | 3001 | admin / admin |
| Airflow | Custom (`apache/airflow:2.9.1-python3.10`) | 8081 | admin / admin |
| Postgres | `postgres:15-alpine` | 5432 | airflow / airflow |
| Streamlit | Local — `python src/app.py` | 8501 | — |

---

## 5. How Every File Is Used

### 5.1 Source Code (`src/`)

| File | What it does | Called by | When |
|---|---|---|---|
| `src/model/patchcore.py` | PatchCore model — `fit()` builds memory bank, `predict()` returns score + heatmap | `train.py` (fit), `app.py` (predict), all tests | Training + every inference |
| `src/model/train.py` | Loads tensors, fits PatchCore, evaluates, logs to DagsHub + local MLflow, saves `.pt` | Developer runs manually | Once per category / `--category all` |
| `src/model/evaluate.py` | AUROC, F1, latency; writes `eval_metrics.json`, `roc_curve.csv`, heatmap `.npy` | Called inside `train.py` | Automatically after each training run |
| `src/pipeline/ingest.py` | Validates raw MVTec AD structure — image counts, readability | Airflow DAG task 1 | When Airflow DAG is triggered |
| `src/pipeline/preprocess.py` | Resize→CenterCrop→Normalise→saves `.pt` tensors | Airflow DAG task 2; standalone | When Airflow DAG runs |
| `src/pipeline/feature_engineering.py` | Baseline stats for drift detection; `check_drift()` at inference | Airflow DAG task 3; `app.py` per upload | DAG (baselines) + every inference (drift) |
| `src/pipeline/airflow_dag.py` | `mvtec_ad_data_pipeline` DAG definition | Airflow reads from `/opt/airflow/dags/` | Triggered manually from Airflow UI |
| `src/pipeline/app_down_response_dag.py` | 5-step incident response DAG | Airflow reads from `/opt/airflow/dags/` | Triggered manually when APP DOWN fires |
| `src/monitoring/drift_detector.py` | Wrapper around `check_drift()` for Airflow batch reporting | Airflow DAG | During pipeline runs |
| `src/app.py` | Streamlit frontend (4 pages) + Prometheus HTTP server on `:8000` + APP DOWN button | Developer runs it; Prometheus scrapes it | Always running during system operation |

### 5.2 Tests (`tests/`)

| File | What it tests | Run command |
|---|---|---|
| `conftest.py` | Shared fixtures — no tests here, just setup | Loaded automatically by pytest |
| `test_patchcore.py` | PatchCore construction, predict, batch predict, save/load | `pytest tests/test_patchcore.py -v` |
| `test_feature_engineering.py` | Baseline computation, load_baseline, check_drift logic | `pytest tests/test_feature_engineering.py -v` |
| `test_ingest.py` | validate_raw_data pass/fail, list_train_images, list_test_images | `pytest tests/test_ingest.py -v` |
| `test_preprocess.py` | Transform output, full pipeline tensor counts, corrupt image skip | `pytest tests/test_preprocess.py -v` |
| `test_evaluate.py` | _optimal_f1_threshold, evaluate_category keys/ranges, file outputs | `pytest tests/test_evaluate.py -v` |

### 5.3 CI/CD (`.github/workflows/ci.yml`)

| Job | What it does | When |
|---|---|---|
| `lint` | flake8 on `src/` and `tests/` | Every push to main/develop, every PR |
| `unit-tests` | CPU pytest with 60% coverage gate | After lint passes |
| `validate-configs` | Validates params.yaml, docker-compose.yml, alert_rules.yml, dashboard JSON | After lint, parallel with tests |
| `docker-build` | Builds Airflow image — no push, just confirms it compiles | After tests + configs pass |
| `ci-summary` | Prints pass/fail table; fails if lint/tests/configs failed | Always, even on failure |

### 5.4 Design Documents (`docs/`)

| File | Contains | Read it when |
|---|---|---|
| `docs/HLD.md` | System overview, architecture diagram, component descriptions, data flow, design decisions table, NFRs | Starting development, project review, understanding why things are built this way |
| `docs/LLD.md` | Every function signature + params/returns, PatchCore tensor shape trace, Prometheus endpoint spec, all metrics, alert PromQL, Grafana panel queries, Airflow task graphs, Docker service spec, MLflow schema, file I/O table | Debugging, extending the system, understanding exact internal data flows |

---

## 6. Full Workflow — Step by Step

This is the correct order to run the entire project from scratch.

### Step 1 — Clone and set up environment

```powershell
git clone https://github.com/shashank-adsule/MLOPS_end-2-end_project
cd MLOPS_end-2-end_project

conda activate E2E

pip install -r requirements-pipeline.txt
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

copy .env.example .env
# Edit .env — fill in DAGSHUB_TOKEN and other credentials
```

### Step 2 — Pull data from DagsHub

```powershell
dvc pull
# Downloads data/raw/mvtec_ad/ (5.2 GB) and data/processed/ from DagsHub
```

Skip this if data is already present locally.

### Step 3 — Start all Docker services

```powershell
docker compose up -d
# First run builds the Airflow image and downloads CUDA torch — takes ~10 min
# Subsequent starts take under 30 seconds
```

Wait ~60 seconds, then verify:

| URL | Expected |
|---|---|
| http://localhost:5000 | MLflow UI loads |
| http://localhost:9090/targets | Prometheus targets page |
| http://localhost:9093 | Alertmanager UI |
| http://localhost:3001 | Grafana (admin / admin) |
| http://localhost:8081 | Airflow (admin / admin) |

### Step 4 — Run the Airflow data pipeline

> Only needed if `data/processed/` is empty or you want to regenerate tensors.

1. Open http://localhost:8081 — login admin / admin
2. Find DAG **`mvtec_ad_data_pipeline`** → toggle it ON
3. Click ▶ **Trigger DAG**
4. Watch: `ingest_data → preprocess_data → compute_feature_baselines` all turn green

This populates `data/processed/{category}/train/`, `test/`, and `stats/baseline.json` for all 15 categories.

**Alternative — run preprocessing directly (faster, no Airflow overhead):**
```powershell
$env:PYTHONPATH = "."
python -m src.pipeline.preprocess
```

### Step 5 — Train models

```powershell
$env:PYTHONPATH = "."

# Single category (~5 min on GPU)
python -m src.model.train --category bottle

# All 15 categories (~90 min on RTX 3050)
python -m src.model.train --category all
```

After training completes:
- `models/patchcore_{category}.pt` — saved model files
- `reports/{category}/eval_metrics.json` — per-category metrics
- `reports/metrics.json` — aggregated table
- DagsHub MLflow — full run with artifacts: http://dagshub.com/da25m005/MLOPS_end-2-end_project/experiments
- Local MLflow — params + metrics: http://localhost:5000

### Step 6 — Start Streamlit (inference)

```powershell
$env:PYTHONPATH = "."
python src/app.py
```

Open http://localhost:8501. The app automatically starts the Prometheus metrics server on `:8000`. Prometheus (running in Docker) scrapes it within 10 seconds.

**Using the app:**
1. Select a product category from the dropdown
2. Upload a product image (PNG/JPG/BMP)
3. Click **Run Detection** — get anomaly score, verdict, heatmap overlay
4. Check Grafana at http://localhost:3001 — panels update within 15 seconds
5. Use the **🔴 Trigger APP DOWN Alert** sidebar button to test the full alert pipeline

### Step 7 — Run unit tests (any time, independently)

```powershell
$env:PYTHONPATH = "."
pytest tests/ -v

# No dataset, no Docker, no GPU needed — tests are fully self-contained
```

### Step 8 — Push code → CI runs automatically

```powershell
git add .
git commit -m "your changes"
git push origin main
```

GitHub Actions runs `.github/workflows/ci.yml` automatically. View results at:
`https://github.com/shashank-adsule/MLOPS_end-2-end_project/actions`

---

## 7. ML Pipeline

### Data Flow

```
MVTec AD raw PNGs
      │
      ▼  ingest.py     — validates folder structure and image counts
      │
      ▼  preprocess.py — Resize(256) → CenterCrop(224) → Normalize(ImageNet)
                         saves [3,224,224] float32 .pt tensors
      │
      ▼  feature_engineering.py — per-channel means/stds, global mean/std,
                                   drift_low/drift_high (±3σ), saved to baseline.json
      │
      ▼  train.py      — PatchCore.fit() builds coreset memory bank (10%)
                         evaluate_category() → AUROC, F1, latency
                         logs to DagsHub MLflow + localhost:5000
                         saves models/patchcore_{category}.pt
      │
      ▼  app.py        — PatchCore.load() → predict() → score + heatmap
                         check_drift() compares image to training baseline
                         _record_prediction() writes app_metrics_state.json
      │
      ▼  Prometheus scrapes :8000/metrics every 10s
         Grafana reads → 13 dashboard panels update
```

### PatchCore Configuration (`params.yaml`)

| Parameter | Value | Purpose |
|---|---|---|
| Backbone | `wide_resnet50_2` | Feature extractor — frozen, ImageNet pretrained |
| Input size | 256 → CenterCrop 224 | Matches backbone pretraining resolution |
| Feature layers | `layer2`, `layer3` | Multi-scale: spatial detail + semantic depth |
| Coreset ratio | 10% | Memory bank compactness vs accuracy trade-off |
| Anomaly threshold | 0.5 | Adjustable per session in Streamlit UI |
| Latency SLA | 200 ms | Enforced by `HighInferenceLatency` alert rule |

---

## 8. Unit Tests — How to Run

Tests are **completely self-contained** — they create all data they need via `conftest.py` fixtures. No dataset, no Docker, no GPU, no internet required.

### Install test dependencies

```powershell
pip install pytest==8.2.0 pytest-cov==5.0.0

# CPU-only PyTorch (if running on a machine without GPU, e.g. CI)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

### Run all 55 tests

```powershell
$env:PYTHONPATH = "."
pytest tests/ -v
```

### Run with coverage report

```powershell
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run one file or one test

```powershell
pytest tests/test_patchcore.py -v
pytest tests/test_feature_engineering.py::TestCheckDrift -v
pytest tests/ -v -k "test_heatmap_shape"
```

### Expected output (all pass)

```
tests/test_patchcore.py::TestPatchCoreConstruction::test_default_construction              PASSED
tests/test_patchcore.py::TestPatchCorePredict::test_heatmap_shape_matches_image_size       PASSED
tests/test_feature_engineering.py::TestCheckDrift::test_strongly_shifted_image_flagged    PASSED
tests/test_ingest.py::TestValidateRawData::test_valid_structure_passes                    PASSED
tests/test_preprocess.py::TestPreprocessAllCategories::test_creates_train_tensors          PASSED
tests/test_evaluate.py::TestEvaluateCategory::test_returns_dict_with_required_keys        PASSED
...
55 passed in ~120s
```

### What the conftest.py fixtures provide

| Fixture | Scope | Provides |
|---|---|---|
| `random_rgb_image` | function | 224×224 PIL image |
| `random_tensor` | function | `[3,224,224]` float tensor |
| `batch_tensor` | function | `[4,3,224,224]` batch |
| `tmp_dir` | function | Clean temp directory |
| `processed_dir` | function | Temp tree with 10 train + 5 good + 5 defect `.pt` tensors for 'bottle' |
| `raw_mvtec_dir` | function | Temp raw structure — 55 train PNGs + 15 test/good + 10 test/broken_large |
| `baseline_dict` | function | Pre-built drift baseline stats dict |
| `baseline_file` | function | baseline_dict written to disk as `baseline.json` |
| `tiny_patchcore` | **session** | PatchCore on CPU with 50-patch synthetic memory bank — no `fit()` needed |

> `tiny_patchcore` is `scope="session"` so the WideResNet50 backbone loads **once per pytest run**, not once per test. This keeps the full suite under 2 minutes.

---

## 9. CI/CD Pipeline — How It Works

**File:** `.github/workflows/ci.yml`
**Triggers:** Every push to `main` or `develop`, every pull request to `main`

### Job dependency graph

```
lint ──────────────────────────────────────┐
  │                                         │
  ├──► unit-tests ───────────────────────┐  │
  │                                      │  │
  └──► validate-configs ─────────────────┤  │
                                         ▼  ▼
                                    docker-build
                                         │
                                    ci-summary
```

### What each job does

**`lint`** — blocks everything if it fails
- Runs `flake8 src/ tests/` with `--max-line-length=120`
- Catches syntax errors, undefined names, unused imports, common bug patterns

**`unit-tests`** — runs after lint
- Installs CPU-only PyTorch (GitHub runners have no GPU)
- Writes a dummy `.env` — tests mock all external services
- `pytest tests/ -m "not integration and not gpu" --cov-fail-under=60`
- Uploads `coverage.xml` as a downloadable artifact in the Actions tab

**`validate-configs`** — runs after lint, parallel with unit-tests
- `params.yaml` — checks all 4 top-level keys, exactly 15 categories, valid coreset + image sizes
- `docker-compose.yml` — all 6 required services present
- `monitoring/alert_rules.yml` — every rule has `alert` + `expr`, at least 5 rules
- `monitoring/grafana/dashboards/defect_detection.json` — at least 13 panels, each has a title

**`docker-build`** — runs after tests + configs pass
- Builds `docker/airflow.Dockerfile` with Buildx (uses GitHub cache for speed)
- `push: false` — build check only, no image is published

**`ci-summary`** — always runs, even on failure
- Prints pass/fail table for all jobs
- Fails the overall workflow with `exit 1` if lint, unit-tests, or validate-configs failed

### Running CI checks locally before pushing

```powershell
# Same lint check as CI
flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503

# Same test command as CI
$env:PYTHONPATH = "."
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=60 -m "not integration and not gpu"

# Same config checks as CI
python -c "import yaml; p=yaml.safe_load(open('params.yaml')); assert len(p['data']['categories'])==15; print('params.yaml OK')"
```

---

## 10. Model Performance

Results from `reports/metrics.json` — all 15 categories trained:

| Category | AUROC | F1 Score | Avg Latency |
|---|---|---|---|
| bottle | 1.0000 | 1.0000 | 71.9 ms |
| cable | 0.9865 | 0.9574 | 69.5 ms |
| capsule | 0.9198 | 0.9600 | 69.3 ms |
| carpet | 0.9904 | 0.9724 | 82.0 ms |
| grid | 0.8404 | 0.8793 | 78.9 ms |
| hazelnut | 1.0000 | 1.0000 | 103.8 ms |
| leather | 1.0000 | 1.0000 | 73.6 ms |
| metal_nut | 1.0000 | 1.0000 | 70.2 ms |
| pill | 0.9580 | 0.9606 | 76.7 ms |
| screw | 0.8780 | 0.8973 | 97.8 ms |
| tile | 0.9978 | 0.9880 | 75.3 ms |
| toothbrush | 0.9444 | 0.9524 | 45.1 ms |
| transistor | 0.9962 | 0.9639 | 112.6 ms |
| wood | 0.9807 | 0.9667 | 166.2 ms |
| zipper | 0.9716 | 0.9667 | 105.7 ms |
| **Average** | **0.9643** | **0.9643** | **86.6 ms** |

All 15 categories meet the 200 ms SLA. 6 categories achieve perfect AUROC of 1.0.

---

## 11. Prometheus Metrics

Exposed at `localhost:8000/metrics` by a background thread in `src/app.py`. Prometheus (Docker) scrapes via `host.docker.internal:8000` every 10 seconds.

All metrics are **Gauges** — because values come from reading `app_metrics_state.json` on each scrape, not from in-process increments. Streamlit's thread model makes in-process Counters unreliable across reruns.

| Metric | Labels | Description |
|---|---|---|
| `predictions_total` | — | Total inference count |
| `defects_detected_total` | — | Total defective predictions |
| `normal_detected_total` | — | Total normal predictions |
| `drift_detected_total` | — | Drift events detected |
| `inference_latency_avg_ms` | — | Rolling average latency |
| `inference_latency_p95_ms` | — | Last observed latency (P95 proxy) |
| `anomaly_score_last` | `category` | Last anomaly score per category |
| `predictions_by_category` | `category` | Total predictions per category |
| `app_down_manual` | — | Manual APP DOWN flag (0=up, 1=down) |

---

## 12. Alerting Rules

Defined in `monitoring/alert_rules.yml`, evaluated every 15 seconds.

| Alert | Expression | For | Severity |
|---|---|---|---|
| `HighInferenceLatency` | P95 latency > 200 ms | 1 m | warning |
| `HighErrorRate` | API 5xx rate > 5% | 2 m | critical |
| `DataDriftDetected` | `increase(drift_detected_total[5m]) > 0` | 0 m | warning |
| `HighDefectRate` | Defect rate > 80% over 10 m | 5 m | warning |
| `AppDown` | `app_down_manual == 1` | 0 m | critical |

The **AppDown** alert is fired via the Streamlit sidebar button. When it fires: Prometheus detects it → Alertmanager routes it → Grafana shows it red → trigger `app_down_response_dag` in Airflow for the 5-step incident response.

---

## 13. Grafana Dashboard

Auto-provisioned at startup — no manual setup required. Grafana rescans the dashboards folder every 10 seconds.

**URL:** http://localhost:3001 (admin / admin)

| Panel | Metric |
|---|---|
| Total Predictions | `predictions_total` |
| Defects Found | `defects_detected_total` |
| Normal Images | `normal_detected_total` |
| P95 Latency (ms) | `inference_latency_p95_ms` |
| Drift Events | `drift_detected_total` |
| SLA Status (< 200 ms) | `inference_latency_p95_ms * 1000 < 200` |
| Inference Latency P50/P95/P99 | `inference_latency_avg_ms`, `inference_latency_p95_ms` |
| Prediction Rate | `predictions_total` by category |
| Anomaly Score per Category | `anomaly_score_last{category}` |
| Defect vs Normal | `defects_detected_total` + `normal_detected_total` |
| Predictions by Category | `predictions_by_category{category}` |
| Data Drift Z-Score | `drift_z_score_last` |
| Active Prometheus Alerts | `ALERTS{alertstate="firing"}` |

---

## 14. MLflow Experiment Tracking

| Endpoint | URL |
|---|---|
| DagsHub MLflow (primary) | https://dagshub.com/da25m005/MLOPS_end-2-end_project/experiments |
| Local MLflow (secondary) | http://localhost:5000 |

Each training run logs parameters, metrics (AUROC, F1, latency), git commit hash, and full artifacts (model file, ROC curve, heatmaps) to DagsHub. Local MLflow logs params + metrics only. If local MLflow is down, DagsHub logging continues unaffected.

---

## 15. Airflow DAGs

Both DAGs live in `src/pipeline/` — mounted into Airflow at `/opt/airflow/dags/`.

**Access:** http://localhost:8081 (admin / admin)

**`mvtec_ad_data_pipeline`** — manual trigger:
```
ingest_data → preprocess_data → compute_feature_baselines
```

**`app_down_response_dag`** — manual trigger only:
```
log_alert_received → check_services → notify_team → attempt_recovery → log_resolution
```

---

## 16. Design Documents

**`docs/HLD.md`** — High Level Design. Read this to understand the overall system: 6-plane architecture, component responsibilities, data flow, design decisions (why PatchCore over supervised, why Postgres over SQLite, why file-based metrics), and non-functional requirements.

**`docs/LLD.md`** — Low Level Design + API Specs. Read this to understand implementation details: every function signature with parameter/return types, PatchCore internal tensor shape trace, Prometheus `/metrics` endpoint spec, all alert PromQL with threshold justification, Grafana panel-by-panel queries, both Airflow DAG task graphs, Docker service specification, MLflow run schema, and the complete file I/O table.

---

## 17. Port Reference

| Service | Port | URL |
|---|---|---|
| Streamlit UI | 8501 | http://localhost:8501 |
| App Prometheus metrics | 8000 | http://localhost:8000/metrics |
| MLflow | 5000 | http://localhost:5000 |
| Prometheus | 9090 | http://localhost:9090 |
| Alertmanager | 9093 | http://localhost:9093 |
| Grafana | 3001 | http://localhost:3001 |
| Airflow | 8081 | http://localhost:8081 |
| Postgres | 5432 | internal only |

---

## 18. Environment Variables

Create `.env` in the project root (copy from `.env.example`):

```env
DAGSHUB_TOKEN=<your_dagshub_token>
DAGSHUB_USERNAME=da25m005
DAGSHUB_REPO=MLOPS_end-2-end_project
MLFLOW_TRACKING_URI=https://dagshub.com/da25m005/MLOPS_end-2-end_project.mlflow
MLFLOW_TRACKING_USERNAME=da25m005
MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
MLFLOW_LOCAL_URI=http://localhost:5000
DVC_REMOTE_ACCESS_KEY_ID=<your_dagshub_token>
DVC_REMOTE_SECRET_ACCESS_KEY=<your_dagshub_token>
RAW_DATA_DIR=data/raw/mvtec_ad
PROCESSED_DATA_DIR=data/processed
MODEL_CATEGORY=bottle
DEVICE=cuda
```

Get your DagsHub token: https://dagshub.com/user/settings/tokens

---

## 19. All Run Commands Reference

```powershell
# ── Setup ──────────────────────────────────────────────────────────────────
conda activate E2E
copy .env.example .env
dvc pull                                    # pull dataset from DagsHub

# ── Docker ─────────────────────────────────────────────────────────────────
docker compose up -d                        # start all 6 services
docker compose up --build                   # rebuild after Dockerfile changes
docker compose down -v --remove-orphans     # full reset (wipes all volumes)
docker compose ps                           # check status
docker compose logs airflow --tail=50
docker compose logs mlflow  --tail=50
docker compose logs grafana --tail=30

# ── Streamlit (local, GPU) ──────────────────────────────────────────────────
$env:PYTHONPATH = "."
python src/app.py                           # starts :8501 UI + :8000 metrics

# ── Training ───────────────────────────────────────────────────────────────
$env:PYTHONPATH = "."
python -m src.model.train --category bottle  # single category (~5 min)
python -m src.model.train --category all     # all 15 (~90 min on RTX 3050)

# ── Preprocessing (standalone, without Airflow) ─────────────────────────────
$env:PYTHONPATH = "."
python -m src.pipeline.preprocess

# ── Unit Tests ─────────────────────────────────────────────────────────────
$env:PYTHONPATH = "."
pytest tests/ -v                                         # all 55 tests
pytest tests/ -v --cov=src --cov-report=term-missing     # with coverage
pytest tests/test_patchcore.py -v                        # single file
pytest tests/ -v -k "test_heatmap"                       # single test by name

# ── DVC ────────────────────────────────────────────────────────────────────
dvc push                                    # push data to DagsHub
dvc pull                                    # pull data from DagsHub

# ── Airflow ────────────────────────────────────────────────────────────────
docker exec -it airflow airflow users create `
  --username admin --password admin `
  --firstname Admin --lastname User `
  --role Admin --email admin@example.com
docker exec -it airflow airflow users reset-password --username admin --password admin

# ── Grafana — force dashboard reload ───────────────────────────────────────
$cred = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:admin"))
Invoke-WebRequest -Method POST `
  -Uri "http://localhost:3001/api/admin/provisioning/dashboards/reload" `
  -Headers @{Authorization="Basic $cred"} `
  -UseBasicParsing

# ── Verify Prometheus is scraping metrics ──────────────────────────────────
curl "http://localhost:9090/api/v1/query?query=predictions_total" -UseBasicParsing

# ── Verify metrics file is written after predictions ───────────────────────
type $env:TEMP\app_metrics_state.json
```

---

## 20. Completion Checklist

| Component | Status | Notes |
|---|---|---|
| MVTec AD dataset (DVC) | ✅ Complete | 15 categories, 6,644 images |
| PatchCore training (all 15) | ✅ Complete | Avg AUROC 0.9643 |
| MLflow on DagsHub | ✅ Complete | All runs logged with artifacts |
| Local MLflow (dual logging) | ✅ Complete | localhost:5000 |
| DVC data versioning | ✅ Complete | Remote on DagsHub |
| Airflow data pipeline DAG | ✅ Complete | localhost:8081 |
| Airflow incident response DAG | ✅ Complete | Manual trigger |
| Prometheus metrics server | ✅ Complete | localhost:8000/metrics |
| Prometheus scraping | ✅ Complete | host.docker.internal:8000 |
| Alert rules (5 rules) | ✅ Complete | localhost:9090/alerts |
| Alertmanager | ✅ Complete | localhost:9093 |
| APP DOWN manual alert | ✅ Complete | Sidebar button in Streamlit |
| Grafana dashboard (13 panels) | ✅ Complete | Auto-provisioned |
| Grafana prediction panels bug | ✅ Fixed | fcntl + thread isolation resolved |
| Streamlit app (CUDA) | ✅ Complete | localhost:8501 |
| GitHub Actions CI/CD | ✅ Complete | `.github/workflows/ci.yml` — 5 jobs |
| Unit tests (55 tests) | ✅ Complete | `tests/` — 5 files, all modules covered |
| HLD document | ✅ Complete | `docs/HLD.md` |
| LLD + API specs | ✅ Complete | `docs/LLD.md` |

---

**Repository:** https://github.com/shashank-adsule/MLOPS_end-2-end_project  
**DagsHub:** https://dagshub.com/da25m005/MLOPS_end-2-end_project