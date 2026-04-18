"""
src/app.py
-----------
Streamlit frontend for Surface Defect Detection System.

Features:
  - Image upload for any of 15 MVTec AD categories
  - Real-time anomaly detection with PatchCore
  - Heatmap visualization overlaid on original image
  - Prediction history table
  - MLflow experiment metrics viewer
  - Pipeline status dashboard
  - Prometheus metrics exposed on port 8000

Usage:
  streamlit run src/app.py
  python src/app.py
"""

import base64
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image
import torchvision.transforms as T

load_dotenv()

# ---------------------------------------------------------------------------
# Prometheus metrics server — starts in background on port 8000
# ---------------------------------------------------------------------------
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge,
    Info,
)

def _start_metrics_server(port: int = 8000):
    """Start Prometheus metrics HTTP server in a background thread."""
    try:
        start_http_server(port)
        logging.getLogger(__name__).info(
            "Prometheus metrics server started on port %d", port
        )
    except OSError:
        logging.getLogger(__name__).warning(
            "Metrics server already running on port %d", port
        )

# Only start once — use module-level flag
if not globals().get("_METRICS_SERVER_STARTED", False):
    threading.Thread(target=_start_metrics_server, daemon=True).start()
    globals()["_METRICS_SERVER_STARTED"] = True

# Metric definitions
PREDICT_COUNTER = Counter(
    "predictions_total",
    "Total predictions made by the app",
    ["category", "verdict"],
)
LATENCY_HIST = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    ["category"],
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0],
)
ANOMALY_GAUGE = Gauge(
    "anomaly_score_last",
    "Most recent anomaly score per category",
    ["category"],
)
DRIFT_COUNTER = Counter(
    "drift_detected_total",
    "Number of drift detections",
    ["category"],
)
DEFECT_COUNTER = Counter(
    "defects_detected_total",
    "Total defects detected",
    ["category"],
)
NORMAL_COUNTER = Counter(
    "normal_detected_total",
    "Total normal images detected",
    ["category"],
)
APP_INFO = Info("app", "Application information")
APP_INFO.info({
    "version":   "1.0.0",
    "algorithm": "PatchCore",
    "dataset":   "MVTec AD",
    "backbone":  "wide_resnet50_2",
})

# ---------------------------------------------------------------------------
# Streamlit page config — MUST be first st call
# ---------------------------------------------------------------------------
import streamlit as st

st.set_page_config(
    page_title="Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
@st.cache_resource
def load_config():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

PARAMS        = load_config()
MODELS_DIR    = os.environ.get("MODEL_PATH", "models")
PROCESSED_DIR = os.environ.get("PROCESSED_DATA_DIR", PARAMS["data"]["processed_dir"])
IMAGE_SIZE    = PARAMS["model"]["image_size"]
PATCH_SIZE    = PARAMS["model"]["patch_size"]
CATEGORIES    = PARAMS["data"]["categories"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(category: str):
    """Load PatchCore model for a category. Cached in session."""
    import sys
    sys.path.insert(0, ".")
    from src.model.patchcore import PatchCore

    model_path = Path(MODELS_DIR) / f"patchcore_{category}.pt"
    if not model_path.exists():
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PatchCore.load(str(model_path), device=device)


def get_available_categories() -> List[str]:
    return [
        c for c in CATEGORIES
        if (Path(MODELS_DIR) / f"patchcore_{c}.pt").exists()
    ]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(PATCH_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image.convert("RGB"))


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
def run_drift_check(tensor: torch.Tensor, category: str) -> Dict:
    """Check for data drift against training baseline."""
    try:
        from src.pipeline.feature_engineering import load_baseline, check_drift
        baseline = load_baseline(PROCESSED_DIR, category)
        result = check_drift(tensor, baseline)
        if result.get("drifted"):
            DRIFT_COUNTER.labels(category=category).inc()
        return result
    except Exception:
        return {"drifted": False, "z_score": 0.0}


# ---------------------------------------------------------------------------
# Heatmap overlay
# ---------------------------------------------------------------------------
def create_heatmap_overlay(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    orig_resized = original.resize((PATCH_SIZE, PATCH_SIZE)).convert("RGB")
    orig_array   = np.array(orig_resized, dtype=np.float32)
    h, w         = heatmap.shape
    heat_rgb     = np.zeros((h, w, 3), dtype=np.float32)
    heat_rgb[:, :, 0] = np.clip(heatmap * 2,     0, 1) * 255
    heat_rgb[:, :, 1] = np.clip(heatmap * 2 - 1, 0, 1) * 255
    blended = (1 - alpha) * orig_array + alpha * heat_rgb
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        color: #1f77b4; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #666; margin-bottom: 2rem;
    }
    .defect-banner {
        background: #ffe0e0; border: 2px solid #ff4b4b;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .normal-banner {
        background: #e0ffe0; border: 2px solid #00cc44;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .badge-defect {
        background: #ff4b4b; color: white;
        padding: 0.3rem 1.2rem; border-radius: 20px;
        font-weight: bold; font-size: 1.2rem;
    }
    .badge-normal {
        background: #00cc44; color: white;
        padding: 0.3rem 1.2rem; border-radius: 20px;
        font-weight: bold; font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Defect Detection")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🔍 Detect Defects",
            "📊 Pipeline Dashboard",
            "📈 Experiment Metrics",
            "📋 Prediction History",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### System Status")

    device    = "CUDA" if torch.cuda.is_available() else "CPU"
    available = get_available_categories()

    st.success(f"Device: {device}")
    st.info(f"Models: {len(available)}/15 available")

    if torch.cuda.is_available():
        vram_used  = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.metric("VRAM", f"{vram_used:.1f} / {vram_total:.1f} GB")

    st.markdown("---")
    st.markdown("### Monitoring")
    st.markdown("📊 [Grafana](http://localhost:3001)")
    st.markdown("🔥 [Prometheus](http://localhost:9090)")
    st.markdown("🔔 [Alertmanager](http://localhost:9093)")

    st.markdown("---")
    st.markdown("### Links")
    st.markdown("🔗 [MLflow](http://localhost:5000)")
    st.markdown("🔗 [DagsHub](https://dagshub.com/da25m005/MLOPS_end-2-end_project)")
    st.markdown("🔗 [GitHub](https://github.com/shashank-adsule/MLOPS_end-2-end_project)")


# ===========================================================================
# PAGE 1: Detect Defects
# ===========================================================================
if page == "🔍 Detect Defects":

    st.markdown('<p class="main-header">🔍 Surface Defect Detection</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a product image to detect surface '
        'defects using PatchCore anomaly detection</p>',
        unsafe_allow_html=True,
    )

    # Controls row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        category = st.selectbox(
            "Product Category",
            options=get_available_categories(),
            help="Select the product type matching your image",
        )
    with col2:
        threshold = st.slider(
            "Anomaly Threshold",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Score above this = DEFECTIVE",
        )
    with col3:
        heatmap_alpha = st.slider(
            "Heatmap Opacity",
            min_value=0.1, max_value=0.9, value=0.5, step=0.1,
        )

    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["png", "jpg", "jpeg", "bmp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        if st.button("🚀 Run Detection", type="primary", use_container_width=True):

            with st.spinner(f"Running PatchCore detection on {category}..."):

                # Load model
                model = load_model(category)
                if model is None:
                    st.error(f"Model not found for: {category}")
                    st.stop()

                # Preprocess
                tensor = preprocess_image(image)

                # Drift check
                drift_info = run_drift_check(tensor, category)

                # Inference
                t0 = time.perf_counter()
                anomaly_score, heatmap = model.predict(tensor)
                latency_ms = (time.perf_counter() - t0) * 1000

                # Verdict
                is_defective = anomaly_score > threshold
                verdict = "defective" if is_defective else "normal"

                # Record Prometheus metrics
                PREDICT_COUNTER.labels(category=category, verdict=verdict).inc()
                LATENCY_HIST.labels(category=category).observe(latency_ms / 1000)
                ANOMALY_GAUGE.labels(category=category).set(anomaly_score)
                if is_defective:
                    DEFECT_COUNTER.labels(category=category).inc()
                else:
                    NORMAL_COUNTER.labels(category=category).inc()

            # ----------------------------------------------------------------
            # Results
            # ----------------------------------------------------------------
            st.markdown("---")
            st.markdown("### Detection Results")

            if is_defective:
                st.markdown(
                    f'<div class="defect-banner">'
                    f'<span class="badge-defect">⚠️ DEFECTIVE</span>'
                    f'<p style="margin-top:0.5rem;color:#cc0000;">'
                    f'Anomaly score {anomaly_score:.4f} exceeds threshold {threshold:.2f}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="normal-banner">'
                    f'<span class="badge-normal">✅ NORMAL</span>'
                    f'<p style="margin-top:0.5rem;color:#006622;">'
                    f'Anomaly score {anomaly_score:.4f} below threshold {threshold:.2f}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Anomaly Score", f"{anomaly_score:.4f}")
            m2.metric("Threshold",     f"{threshold:.2f}")
            m3.metric("Latency",       f"{latency_ms:.1f}ms")
            m4.metric("Category",      category.title())

            # Score bar
            st.markdown("**Anomaly Score Bar**")
            st.progress(min(float(anomaly_score), 1.0))

            # Drift warning
            if drift_info.get("drifted"):
                st.warning(
                    f"⚠️ Data drift detected — z-score: "
                    f"{drift_info.get('z_score', 0):.2f} "
                    f"(threshold: 3.0)"
                )

            # Images
            st.markdown("### Image Analysis")
            ic1, ic2, ic3 = st.columns(3)

            with ic1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)

            with ic2:
                st.markdown("**Anomaly Heatmap**")
                heatmap_rgb = np.stack([
                    np.clip(heatmap * 2,     0, 1),
                    np.clip(heatmap * 2 - 1, 0, 1),
                    np.zeros_like(heatmap),
                ], axis=2)
                heatmap_img = Image.fromarray(
                    (heatmap_rgb * 255).astype(np.uint8)
                )
                st.image(heatmap_img, use_container_width=True)

            with ic3:
                st.markdown("**Overlay**")
                overlay = create_heatmap_overlay(image, heatmap, heatmap_alpha)
                st.image(overlay, use_container_width=True)

            # Save to history
            st.session_state.history.append({
                "timestamp":     time.strftime("%H:%M:%S"),
                "filename":      uploaded_file.name,
                "category":      category,
                "anomaly_score": round(float(anomaly_score), 4),
                "verdict":       "DEFECTIVE" if is_defective else "NORMAL",
                "latency_ms":    round(latency_ms, 1),
                "threshold":     threshold,
                "drift":         drift_info.get("drifted", False),
            })

        else:
            pc1, pc2 = st.columns([1, 2])
            with pc1:
                st.image(image, caption=uploaded_file.name,
                         use_container_width=True)
            with pc2:
                st.info(
                    f"**File:** {uploaded_file.name}\n\n"
                    f"**Size:** {image.size[0]}×{image.size[1]} px\n\n"
                    f"**Category:** {category}\n\n"
                    f"**Threshold:** {threshold}\n\n"
                    f"Click **Run Detection** to analyze."
                )
    else:
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;padding:3rem;'
            'background:#f8f9fa;border-radius:10px;">'
            '<h3>Upload an image to get started</h3>'
            '<p>Supports PNG, JPG, JPEG, BMP</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Available Categories")
        cats = get_available_categories()
        cols = st.columns(5)
        for i, cat in enumerate(cats):
            cols[i % 5].success(f"✓ {cat}")


# ===========================================================================
# PAGE 2: Pipeline Dashboard
# ===========================================================================
elif page == "📊 Pipeline Dashboard":

    st.markdown('<p class="main-header">📊 ML Pipeline Dashboard</p>',
                unsafe_allow_html=True)

    st.markdown("### Pipeline Stages")
    stages = [
        ("Data Ingestion",       "Airflow DAG",      "src/pipeline/ingest.py",              True),
        ("Preprocessing",        "Custom",           "src/pipeline/preprocess.py",           True),
        ("Feature Engineering",  "Custom",           "src/pipeline/feature_engineering.py",  True),
        ("Model Training",       "PatchCore",        "src/model/train.py",                   True),
        ("Model Evaluation",     "Scikit-learn",     "src/model/evaluate.py",                True),
        ("Model Registry",       "MLflow",           "DagsHub MLflow",                       True),
        ("App Serving",          "Streamlit",        "src/app.py",                           True),
        ("Metrics Exposure",     "Prometheus Client","src/app.py :8000",                     True),
        ("Monitoring",           "Prometheus",       "monitoring/prometheus.yml",             True),
        ("Alerting",             "Alertmanager",     "monitoring/alertmanager.yml",           True),
        ("Dashboards",           "Grafana",          "monitoring/grafana/",                  True),
    ]
    for i, (stage, tool, path, done) in enumerate(stages):
        c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
        c1.write(f"**{i+1}. {stage}**")
        c2.write(f"`{tool}`")
        c3.write(f"`{path}`")
        c4.write("✅" if done else "⏳")

    st.markdown("---")
    st.markdown("### Trained Models")
    available = get_available_categories()
    missing   = [c for c in CATEGORIES if c not in available]

    c1, c2 = st.columns(2)
    with c1:
        st.success(f"**{len(available)} models trained**")
        for cat in available:
            size = Path(MODELS_DIR, f"patchcore_{cat}.pt").stat().st_size / 1024**2
            st.write(f"✅ {cat} — {size:.0f} MB")
    with c2:
        if missing:
            st.warning(f"**{len(missing)} models missing**")
            for cat in missing:
                st.write(f"❌ {cat}")
        else:
            st.success("All 15 categories trained!")

    st.markdown("---")
    st.markdown("### DVC Data Versioning")
    d1, d2, d3 = st.columns(3)
    d1.metric("DVC Remote", "DagsHub")
    d2.metric("Dataset",    "MVTec AD")
    d3.metric("Files",      "6,644")

    st.markdown("---")
    st.markdown("### Evaluation Reports")
    summary = Path("reports") / "metrics.json"
    if summary.exists():
        with open(summary) as f:
            data = json.load(f)
        if data:
            import pandas as pd
            df = pd.DataFrame(data)
            cols = [c for c in
                    ["category","auroc","f1_score","pixel_auroc","avg_latency_ms"]
                    if c in df.columns]
            if cols:
                st.dataframe(
                    df[cols].style.format({
                        "auroc":          "{:.4f}",
                        "f1_score":       "{:.4f}",
                        "pixel_auroc":    "{:.4f}",
                        "avg_latency_ms": "{:.1f}",
                    }),
                    use_container_width=True,
                )
    else:
        st.info("Run training to generate reports.")


# ===========================================================================
# PAGE 3: Experiment Metrics
# ===========================================================================
elif page == "📈 Experiment Metrics":

    st.markdown('<p class="main-header">📈 MLflow Experiment Metrics</p>',
                unsafe_allow_html=True)

    st.info(
        "👉 Full experiment tracking on DagsHub MLflow:\n\n"
        "https://dagshub.com/da25m005/MLOPS_end-2-end_project/experiments"
    )

    summary = Path("reports") / "metrics.json"
    if summary.exists():
        with open(summary) as f:
            data = json.load(f)

        if data:
            import pandas as pd
            df = pd.DataFrame(data)

            s1, s2, s3, s4 = st.columns(4)
            if "auroc" in df.columns:
                s1.metric("Avg AUROC",  f"{df['auroc'].mean():.4f}")
                s2.metric("Best AUROC", f"{df['auroc'].max():.4f}")
            if "f1_score" in df.columns:
                s3.metric("Avg F1",     f"{df['f1_score'].mean():.4f}")
            if "avg_latency_ms" in df.columns:
                s4.metric("Avg Latency", f"{df['avg_latency_ms'].mean():.1f}ms")

            st.markdown("---")

            if "auroc" in df.columns:
                st.markdown("**AUROC by Category**")
                st.bar_chart(df.set_index("category")[["auroc"]]
                             .sort_values("auroc", ascending=False))

            if "f1_score" in df.columns:
                st.markdown("**F1 Score by Category**")
                st.bar_chart(df.set_index("category")[["f1_score"]]
                             .sort_values("f1_score", ascending=False))

            if "avg_latency_ms" in df.columns:
                st.markdown("**Inference Latency by Category (ms)**")
                st.bar_chart(df.set_index("category")[["avg_latency_ms"]]
                             .sort_values("avg_latency_ms"))

            st.markdown("---")
            st.markdown("### Full Results Table")
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("No metrics found. Run: `python -m src.model.train --category all`")


# ===========================================================================
# PAGE 4: Prediction History
# ===========================================================================
elif page == "📋 Prediction History":

    st.markdown('<p class="main-header">📋 Prediction History</p>',
                unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.info("No predictions yet. Go to **Detect Defects** and upload an image.")
    else:
        total     = len(history)
        defective = sum(1 for h in history if h["verdict"] == "DEFECTIVE")
        normal    = total - defective
        avg_lat   = sum(h["latency_ms"] for h in history) / total
        drifts    = sum(1 for h in history if h.get("drift"))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total",     total)
        c2.metric("Defective", defective)
        c3.metric("Normal",    normal)
        c4.metric("Avg Latency", f"{avg_lat:.1f}ms")
        c5.metric("Drift Events", drifts)

        st.markdown("---")

        import pandas as pd
        df = pd.DataFrame(history)

        def color_verdict(val):
            if val == "DEFECTIVE":
                return "background-color: #ffe0e0"
            elif val == "NORMAL":
                return "background-color: #e0ffe0"
            return ""

        st.dataframe(
            df.style.applymap(color_verdict, subset=["verdict"]),
            use_container_width=True,
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import subprocess
    import sys

    os.chdir(Path(__file__).parent.parent)
    os.environ["PYTHONPATH"] = "."

    os.system(
        f"streamlit run src/app.py "
        f"--server.port 8501 "
        f"--server.headless false"
    )