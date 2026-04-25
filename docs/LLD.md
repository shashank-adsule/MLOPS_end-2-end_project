# Low Level Design & API Specifications — MLOps End-to-End Surface Defect Detection

**Student:** DA25M005 (Shashank Adsule)
**Repository:** https://github.com/shashank-adsule/MLOPS_end-2-end_project
**Version:** 1.0 | **Date:** April 2026

---

## 1. Module Breakdown

### 1.1 `src/model/patchcore.py` — PatchCore Model

**Class:** `PatchCore(nn.Module)`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backbone` | `str` | `"wide_resnet50_2"` | torchvision backbone name |
| `layers` | `List[str]` | `["layer2","layer3"]` | Feature extraction hook targets |
| `coreset_ratio` | `float` | `0.1` | Fraction of patches to keep in memory bank |
| `image_size` | `int` | `224` | Heatmap output resolution |
| `device` | `str` | `"cuda"` | Falls back to CPU if CUDA unavailable |

#### Public Methods

```python
def fit(dataloader: DataLoader) -> None
```
Iterates all batches, extracts patch features via hooks on `layer2`+`layer3`, concatenates them, applies greedy coreset subsampling, stores the result in `self.memory_bank: torch.Tensor [N_keep, D]`.

```python
def predict(image: torch.Tensor) -> Tuple[float, np.ndarray]
```
Accepts `[3,H,W]` or `[1,3,H,W]`. Returns `(anomaly_score: float, heatmap: np.ndarray[H,W])`. Heatmap is normalised to `[0,1]` and upsampled to `image_size × image_size` via bilinear interpolation + Gaussian smoothing (`sigma=4`).

```python
def predict_batch(images: torch.Tensor) -> Tuple[List[float], List[np.ndarray]]
```
Loops `predict()` over each image in the batch. Returns parallel lists of scores and heatmaps.

```python
def save(path: str) -> None
def load(path: str, device: str) -> PatchCore   # classmethod
```
Saves/loads `{memory_bank, backbone_name, layers, coreset_ratio, image_size}` as a PyTorch checkpoint.

#### Internal Data Flow

```
Input batch [B, 3, H, W]
    │
    ▼ feature_extractor forward pass (frozen WideResNet50)
Forward hooks populate _features dict:
    _features["layer2"]  →  [B, 512, 28, 28]
    _features["layer3"]  →  [B, 1024, 14, 14]
    │
    ▼ Resize layer3 to layer2 spatial size via bilinear interpolation
    ▼ Concatenate → [B, 1536, 28, 28]
    ▼ avg_pool2d(kernel=3, stride=1, padding=1)  → [B, 1536, 28, 28]
    ▼ Permute + reshape → [B*28*28, 1536]  (patch vectors)
    ▼ L2 normalise each patch
    │
    ▼ (training) greedy coreset subsampling
         Random projection → [N, 128] for fast distance computation
         Greedy farthest-point sampling
         Keep 10% → memory_bank [N_keep, 1536]
    │
    ▼ (inference) cdist(query_patches, memory_bank) → nearest-neighbour distance per patch
    ▼ max over patches → anomaly_score
    ▼ reshape distances to grid [28, 28]
    ▼ bilinear upsample to [224, 224]
    ▼ Gaussian smooth (sigma=4)
    ▼ min-max normalise → heatmap [0, 1]
```

#### Coreset Subsampling Detail

The coreset uses random projection (D=1536 → 128 dims) before greedy farthest-point sampling to avoid OOM on the RTX 3050 6 GB. Distances are computed in chunks of 1000 rows to limit peak RAM.

---

### 1.2 `src/model/evaluate.py` — Evaluation

**Public Function:**

```python
def evaluate_category(
    model: PatchCore,
    processed_dir: str,
    category: str,
    reports_dir: str,
    save_heatmaps: bool = True,
    n_heatmap_samples: int = 5,
) -> Dict
```

**Returns:**

| Key | Type | Description |
|---|---|---|
| `auroc` | `float` | Image-level AUROC (sklearn roc_auc_score) |
| `f1_score` | `float` | F1 at optimal threshold |
| `pixel_auroc` | `float` | Proxy pixel-AUROC (image score used as proxy when GT masks absent) |
| `avg_latency_ms` | `float` | Mean inference time per image in milliseconds |
| `threshold` | `float` | Score threshold that maximised F1 |
| `n_normal` | `int` | Count of normal test images |
| `n_defect` | `int` | Count of defective test images |

**Side effects:**
- Writes `reports/{category}/eval_metrics.json`
- Writes `reports/{category}/roc_curve.csv` (columns: `fpr,tpr`)
- Writes `reports/{category}/heatmaps/{defect_type}_{idx:03d}.npy` for defect samples

**Internal helper:**

```python
def _optimal_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]
```
Scans 200 evenly-spaced percentile thresholds over `scores`. Returns `(best_threshold, best_f1)`.

---

### 1.3 `src/pipeline/ingest.py` — Data Validation

**Public Functions:**

```python
def validate_raw_data(raw_dir: str, categories: List[str]) -> Dict[str, bool]
```
Validates directory structure, image counts, and PIL readability (first 3 images) for each category. Returns pass/fail dict.

**Constraints enforced:**
- `train/good/` must exist with ≥ 50 images
- `test/good/` must exist with ≥ 10 images
- `ground_truth/` directory must exist
- First 3 training images must be PIL-readable (not corrupt)

```python
def list_train_images(raw_dir: str, category: str) -> List[Path]
def list_test_images(raw_dir: str, category: str) -> Dict[str, List[Path]]
def list_ground_truth_masks(raw_dir: str, category: str, defect_type: str) -> List[Path]
```

---

### 1.4 `src/pipeline/preprocess.py` — Image Preprocessing

**Public Functions:**

```python
def preprocess_all_categories(
    raw_dir: str,
    processed_dir: str,
    categories: List[str],
    image_size: int = 256,
    patch_size: int = 224,
) -> Dict[str, Dict]
```

Returns per-category stats: `{train_count, test_count, elapsed_sec}`.

**Transform pipeline:**
```
PIL Image (any size)
    → Resize(256, BILINEAR)
    → CenterCrop(224)
    → ToTensor()                         # [0, 255] uint8 → [0.0, 1.0] float32
    → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])   # ImageNet stats
Output: torch.Tensor [3, 224, 224] float32
```

**Output layout:**
```
data/processed/{category}/
    train/0001.pt, 0002.pt, ...          ← normal training images only
    test/
        good/0001.pt, ...                ← normal test images
        broken_large/0001.pt, ...        ← defect test images (one dir per defect type)
    stats/baseline.json                  ← written by feature_engineering
```

**Corrupt image handling:** `_load_and_transform()` wraps PIL open in try/except. Corrupt files are logged as warnings and skipped — they do not abort the pipeline.

```python
def load_tensor(tensor_path: str) -> torch.Tensor
```

---

### 1.5 `src/pipeline/feature_engineering.py` — Drift Baselines

**Public Functions:**

```python
def compute_baseline_statistics(processed_dir: str, category: str) -> Dict
```

Loads all training `.pt` tensors for a category, stacks them into `[N, 3, 224, 224]`, and computes:

| Statistic | Description |
|---|---|
| `channel_means` | Per-channel mean across all training images `[R, G, B]` |
| `channel_stds` | Per-channel std across all training images |
| `global_mean` | Scalar mean over all pixels across all images |
| `global_std` | Scalar std over all pixels |
| `global_min / max` | Dataset pixel range |
| `intensity_p5 / p95` | 5th and 95th percentile of per-image mean intensity |
| `drift_low / drift_high` | `global_mean ± 3 × global_std` |

Saves result to `data/processed/{category}/stats/baseline.json`.

```python
def load_baseline(processed_dir: str, category: str) -> Dict
```
Loads from disk. Raises `FileNotFoundError` if baseline does not exist.

```python
def check_drift(image_tensor: torch.Tensor, baseline: Dict, sigma_threshold: float = 3.0) -> Dict
```

**Algorithm:**
```
img_mean = mean(image_tensor)
z_score  = |img_mean - baseline.global_mean| / baseline.global_std
drifted  = z_score > sigma_threshold
```

**Returns:**

| Key | Type | Description |
|---|---|---|
| `drifted` | `bool` | True if z_score exceeds sigma_threshold |
| `z_score` | `float` | Normalised distance from training distribution |
| `img_mean` | `float` | Per-pixel mean of the inference image |
| `baseline_mean` | `float` | Training set global mean |
| `sigma_threshold` | `float` | Threshold used for this call |
| `channel_deltas` | `List[float]` | Per-channel `|img_mean - baseline_mean|` |

---

### 1.6 `src/monitoring/drift_detector.py`

Thin wrapper around `feature_engineering.check_drift()` used by the Airflow DAG for batch drift reporting. Exposes `DriftDetector` class with `detect(tensor, category)` → drift result dict.

---

### 1.7 `src/app.py` — Streamlit Application

#### Prometheus Metrics Server

The metrics server runs on port **8000** in a background daemon thread. It starts once per process using a `globals()` guard to prevent duplicate servers on Streamlit script reruns.

**Endpoint:**

| Path | Method | Response |
|---|---|---|
| `/metrics` | GET | Prometheus text format (`text/plain; version=0.0.4`) |
| Any other path | GET | 404 |

**Metric sync on scrape:** On every `/metrics` request, `_MetricsHandler.do_GET()` reads `app_metrics_state.json` from `%TEMP%` and synchronises all Gauge values before calling `generate_latest()`. This ensures the scrape always reflects the latest prediction state written by any Streamlit thread.

#### Metrics Exposed (`/metrics`)

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `predictions_total` | Gauge | — | Total inference count |
| `defects_detected_total` | Gauge | — | Total defective predictions |
| `normal_detected_total` | Gauge | — | Total normal predictions |
| `drift_detected_total` | Gauge | — | Drift events detected |
| `inference_latency_avg_ms` | Gauge | — | Rolling average latency ms |
| `inference_latency_p95_ms` | Gauge | — | Last observed latency (P95 proxy) |
| `anomaly_score_last` | Gauge | `category` | Last anomaly score per category |
| `predictions_by_category` | Gauge | `category` | Total predictions per category |
| `app_down_manual` | Gauge | — | Manual APP DOWN flag (0=up, 1=down) |

**All metrics use a private `CollectorRegistry` (`_APP_REGISTRY`)** — not the default global registry — to prevent duplicate registration errors from other Prometheus client imports.

#### State Files

| File | Location | Contents |
|---|---|---|
| `app_metrics_state.json` | `%TEMP%/` | JSON dict with all prediction counters and scores |
| `app_down_state.txt` | `%TEMP%/` | `"0"` or `"1"` — current APP DOWN flag |

#### Prediction Flow (per image upload)

```
User clicks "Run Detection"
    │
    ▼ load_model(category)        — @st.cache_resource, loaded once
    ▼ preprocess_image(PIL)       — Resize+CenterCrop+Normalize → tensor
    ▼ run_drift_check(tensor)     — load_baseline() + check_drift()
    ▼ model.predict(tensor)       — PatchCore nearest-neighbour scoring
    ▼ _record_prediction(...)     — read/update/write app_metrics_state.json
    ▼ create_heatmap_overlay()    — blend original + heat colour map
    ▼ st.session_state.history.append(result)
    ▼ Render results UI
```

---

## 2. Prometheus Alert Rules — Detailed Specification

**File:** `monitoring/alert_rules.yml`
**Evaluation interval:** 15s (matches group `interval` setting)

| Alert Name | PromQL Expression | `for` | Severity | Justification |
|---|---|---|---|---|
| `HighInferenceLatency` | `histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m])) > 0.2` | 1m | warning | 200 ms = `latency_target_ms` in params.yaml; all trained categories achieve < 200 ms on CUDA |
| `HighErrorRate` | `rate(api_requests_total{status=~"5.."}[5m]) / rate(api_requests_total[5m]) > 0.05` | 2m | critical | 5% threshold; manufacturing inspection has high cost per missed defect |
| `DataDriftDetected` | `increase(drift_detected_total[5m]) > 0` | 0m | warning | Immediate — drift may indicate camera calibration change or new product variant |
| `HighDefectRate` | `rate(defects_detected_total[10m]) / (rate(defects_detected_total[10m]) + rate(normal_detected_total[10m])) > 0.8` | 5m | warning | >80% defect rate for 5 min is statistically anomalous; suggests model drift or batch labelling issue |
| `AppDown` | `app_down_manual == 1` | 0m | critical | Operator-triggered; no delay needed |

---

## 3. Grafana Dashboard Panel Specification

**File:** `monitoring/grafana/dashboards/defect_detection.json`
**Datasource UID:** `prometheus` (must match provisioned datasource)

| # | Panel Title | Panel Type | PromQL Query |
|---|---|---|---|
| 1 | Total Predictions | Stat | `predictions_total` |
| 2 | Defects Found | Stat | `defects_detected_total` |
| 3 | Normal Images | Stat | `normal_detected_total` |
| 4 | P95 Latency (ms) | Stat | `inference_latency_p95_ms` |
| 5 | Drift Events | Stat | `drift_detected_total` |
| 6 | SLA Status (< 200ms) | Stat | `inference_latency_p95_ms * 1000 < 200` |
| 7 | Inference Latency P50/P95/P99 | Time series | `inference_latency_avg_ms`, `inference_latency_p95_ms` |
| 8 | Prediction Rate | Time series | `rate(predictions_total[5m])` by category |
| 9 | Anomaly Score per Category | Bar chart | `anomaly_score_last{category}` |
| 10 | Defect vs Normal | Pie chart | `defects_detected_total` + `normal_detected_total` |
| 11 | Predictions by Category | Bar chart | `predictions_by_category{category}` |
| 12 | Data Drift Z-Score | Time series | `drift_z_score_last` |
| 13 | Active Prometheus Alerts | Alert list | `ALERTS{alertstate="firing"}` |

---

## 4. Airflow DAG Specifications

### 4.1 `mvtec_ad_data_pipeline`

**File:** `src/pipeline/airflow_dag.py`
**Schedule:** Manual trigger (`schedule_interval=None`)

```
ingest_data
    calls: validate_raw_data() + list_train_images() for all 15 categories
    output: XCom with pass/fail dict
    │
    ▼
preprocess_data
    calls: preprocess_all_categories()
    reads params.yaml for image_size, patch_size
    output: .pt tensor files in data/processed/
    │
    ▼
compute_feature_baselines
    calls: compute_baseline_statistics() for each category
    output: baseline.json per category in data/processed/{cat}/stats/
```

### 4.2 `app_down_response_dag`

**File:** `src/pipeline/app_down_response_dag.py`
**Schedule:** Manual only (`schedule_interval=None`) — triggered by Alertmanager webhook or operator

```
log_alert_received
    Logs alert metadata (timestamp, severity, source)
    │
    ▼
check_services
    Queries Docker compose service health endpoints
    Sets XCom: {mlflow: up/down, prometheus: up/down, grafana: up/down, airflow: up/down}
    │
    ▼
notify_team
    Logs structured notification (team channel, severity, service status summary)
    │
    ▼
attempt_recovery
    For each down service: logs recommended recovery action
    Does not restart services directly (avoids Docker-in-Docker complexity)
    │
    ▼
log_resolution
    Logs final resolution status and elapsed time
```

---

## 5. Docker Compose Service Specification

| Service | Image | Internal Port | Host Port | Volumes | Network |
|---|---|---|---|---|---|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.12.2` | 5000 | 5000 | `mlflow_data:/mlflow` | defect_net |
| `prometheus` | `prom/prometheus:v2.52.0` | 9090 | 9090 | `prometheus_data:/prometheus`, config files (ro) | defect_net |
| `alertmanager` | `prom/alertmanager:v0.27.0` | 9093 | 9093 | `alertmanager_data:/alertmanager`, config (ro) | defect_net |
| `grafana` | `grafana/grafana:10.4.2` | 3000 | 3001 | `grafana_data`, provisioning dirs (ro), dashboards (ro) | defect_net |
| `postgres` | `postgres:15-alpine` | 5432 | 5432 | `postgres_data:/var/lib/postgresql/data` | defect_net |
| `airflow` | Custom `docker/airflow.Dockerfile` | 8080 | 8081 | DAGs, src, data, logs | defect_net |

**Streamlit:** Runs on the Windows host (not in Docker) at `:8501` and `:8000`.

---

## 6. MLflow Run Schema

Every training run logs the following to DagsHub (primary) and localhost:5000 (secondary):

**Parameters:**

| Key | Type | Example |
|---|---|---|
| `category` | str | `"bottle"` |
| `backbone` | str | `"wide_resnet50_2"` |
| `layers` | str | `"['layer2', 'layer3']"` |
| `coreset_ratio` | float | `0.1` |
| `image_size` | int | `256` |
| `patch_size` | int | `224` |
| `batch_size` | int | `32` |
| `device` | str | `"cuda"` |

**Metrics:**

| Key | Type | Description |
|---|---|---|
| `auroc` | float | Image-level AUROC |
| `f1_score` | float | Best F1 at optimal threshold |
| `pixel_auroc` | float | Pixel-level AUROC |
| `avg_latency_ms` | float | Average inference latency |
| `threshold` | float | Optimal F1 threshold |
| `train_time_sec` | float | Training duration |
| `memory_bank_size` | int | Coreset patch count |

**Tags:**

| Key | Description |
|---|---|
| `git_commit` | Short git hash for reproducibility |
| `category` | Category name (also in params) |
| `algorithm` | `"PatchCore"` |
| `dataset` | `"MVTec AD"` |
| `sla_breach` | `"none"` or `"latency"` |
| `dagshub_run_id` | Cross-reference tag in local MLflow runs |

**Artifacts (DagsHub only):**
- `models/patchcore_{category}.pt`
- `reports/{category}/roc_curve.csv`
- `heatmaps/*.npy`

---

## 7. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DAGSHUB_TOKEN` | Yes | DagsHub access token (used as MLflow + DVC password) |
| `DAGSHUB_USERNAME` | Yes | DagsHub username |
| `DAGSHUB_REPO` | No | Repository name (hardcoded in train.py) |
| `MLFLOW_TRACKING_URI` | Yes | DagsHub MLflow URI |
| `MLFLOW_TRACKING_USERNAME` | Yes | DagsHub username (same as DAGSHUB_USERNAME) |
| `MLFLOW_TRACKING_PASSWORD` | Yes | DagsHub token (same as DAGSHUB_TOKEN) |
| `MLFLOW_LOCAL_URI` | No | Local MLflow URI (default: `http://localhost:5000`) |
| `DVC_REMOTE_ACCESS_KEY_ID` | Yes | DagsHub token for DVC remote |
| `DVC_REMOTE_SECRET_ACCESS_KEY` | Yes | DagsHub token for DVC remote |
| `RAW_DATA_DIR` | No | Path to raw MVTec AD data (default: `data/raw/mvtec_ad`) |
| `PROCESSED_DATA_DIR` | No | Path for processed tensors (default: `data/processed`) |
| `MODEL_CATEGORY` | No | Default category for training (default: `bottle`) |
| `DEVICE` | No | `cuda` or `cpu` (default: `cuda`) |

---

## 8. File I/O Summary

| File / Dir | Writer | Reader | Format |
|---|---|---|---|
| `data/processed/{cat}/train/*.pt` | `preprocess.py` | `train.py` (MVTecTrainDataset) | PyTorch tensor |
| `data/processed/{cat}/test/**/*.pt` | `preprocess.py` | `evaluate.py` (MVTecTestDataset) | PyTorch tensor |
| `data/processed/{cat}/stats/baseline.json` | `feature_engineering.py` | `app.py` (run_drift_check) | JSON |
| `models/patchcore_{cat}.pt` | `patchcore.save()` in `train.py` | `patchcore.load()` in `app.py` | PyTorch checkpoint |
| `reports/{cat}/eval_metrics.json` | `evaluate.py` | `app.py` Pipeline Dashboard | JSON |
| `reports/{cat}/roc_curve.csv` | `evaluate.py` | MLflow / DVC plots | CSV |
| `reports/{cat}/heatmaps/*.npy` | `evaluate.py` | MLflow artifacts | NumPy binary |
| `reports/metrics.json` | `train.py` (main) | `app.py` Experiment Metrics page | JSON (list) |
| `%TEMP%/app_metrics_state.json` | `_record_prediction()` in `app.py` | `_MetricsHandler.do_GET()` | JSON |
| `%TEMP%/app_down_state.txt` | `_set_app_down()` in `app.py` | `_MetricsHandler.do_GET()` | Plain text `"0"`/`"1"` |
| `monitoring/grafana/dashboards/defect_detection.json` | (static) | Grafana provisioner | Grafana JSON |
| `monitoring/grafana/provisioning/**/*.yml` | (static) | Grafana on startup | YAML |

---

**Prepared by:** Shashank Adsule — DA25M005
**Date:** April 2026
