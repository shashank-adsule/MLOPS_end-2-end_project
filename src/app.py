"""
src/app.py
-----------
Streamlit frontend for Surface Defect Detection System.
"""

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
# Prometheus — lazy singleton pattern to survive Streamlit reloads
# ---------------------------------------------------------------------------
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge,
    REGISTRY,
    CollectorRegistry,
)

_METRICS = {}

def _get_metrics():
    """Return metrics dict, creating once and reusing across reloads."""
    global _METRICS
    if _METRICS:
        return _METRICS

    def _safe_counter(name, desc, labels):
        try:
            return Counter(name, desc, labels)
        except ValueError:
            # Already registered — find and return it
            for n, c in REGISTRY._names_to_collectors.items():
                if n == name + "_total":
                    return c
            return Counter(name, desc, labels)

    def _safe_histogram(name, desc, labels, buckets):
        try:
            return Histogram(name, desc, labels, buckets=buckets)
        except ValueError:
            for n, c in REGISTRY._names_to_collectors.items():
                if n == name + "_bucket":
                    return c
            return Histogram(name, desc, labels, buckets=buckets)

    def _safe_gauge(name, desc, labels):
        try:
            return Gauge(name, desc, labels)
        except ValueError:
            for n, c in REGISTRY._names_to_collectors.items():
                if n == name:
                    return c
            return Gauge(name, desc, labels)

    _METRICS = {
        "predict":  _safe_counter("predictions_total", "Total predictions", ["category", "verdict"]),
        "latency":  _safe_histogram("inference_latency_seconds", "Inference latency", ["category"], [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]),
        "score":    _safe_gauge("anomaly_score_last", "Last anomaly score", ["category"]),
        "drift":    _safe_counter("drift_detected_total", "Drift detections", ["category"]),
        "defects":  _safe_counter("defects_detected_total", "Total defects", ["category"]),
        "normal":   _safe_counter("normal_detected_total", "Total normal", ["category"]),
    }
    return _METRICS


def _start_metrics_server(port: int = 8000):
    try:
        start_http_server(port)
    except OSError:
        pass  # Already running


# Start metrics server once per process
if not globals().get("_METRICS_STARTED", False):
    threading.Thread(target=_start_metrics_server, daemon=True).start()
    globals()["_METRICS_STARTED"] = True

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
    import sys
    sys.path.insert(0, ".")
    from src.model.patchcore import PatchCore
    model_path = Path(MODELS_DIR) / f"patchcore_{category}.pt"
    if not model_path.exists():
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PatchCore.load(str(model_path), device=device)


def get_available_categories() -> List[str]:
    return [c for c in CATEGORIES
            if (Path(MODELS_DIR) / f"patchcore_{c}.pt").exists()]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE),
                 interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(PATCH_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image.convert("RGB"))


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
def run_drift_check(tensor: torch.Tensor, category: str) -> Dict:
    try:
        from src.pipeline.feature_engineering import load_baseline, check_drift
        baseline = load_baseline(PROCESSED_DIR, category)
        return check_drift(tensor, baseline)
    except Exception:
        return {"drifted": False, "z_score": 0.0}


# ---------------------------------------------------------------------------
# Heatmap overlay
# ---------------------------------------------------------------------------
def create_heatmap_overlay(original, heatmap, alpha=0.5):
    orig = np.array(original.resize((PATCH_SIZE, PATCH_SIZE)).convert("RGB"),
                    dtype=np.float32)
    h, w = heatmap.shape
    heat = np.zeros((h, w, 3), dtype=np.float32)
    heat[:, :, 0] = np.clip(heatmap * 2,     0, 1) * 255
    heat[:, :, 1] = np.clip(heatmap * 2 - 1, 0, 1) * 255
    blended = (1 - alpha) * orig + alpha * heat
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.main-header { font-size:2.2rem; font-weight:700; color:#1f77b4; margin-bottom:0.2rem; }
.sub-header  { font-size:1rem; color:#666; margin-bottom:2rem; }
.defect-banner { background:#ffe0e0; border:2px solid #ff4b4b; border-radius:10px; padding:1rem; text-align:center; }
.normal-banner { background:#e0ffe0; border:2px solid #00cc44; border-radius:10px; padding:1rem; text-align:center; }
.badge-defect { background:#ff4b4b; color:white; padding:0.3rem 1.2rem; border-radius:20px; font-weight:bold; font-size:1.2rem; }
.badge-normal { background:#00cc44; color:white; padding:0.3rem 1.2rem; border-radius:20px; font-weight:bold; font-size:1.2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Defect Detection")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🔍 Detect Defects",
        "📊 Pipeline Dashboard",
        "📈 Experiment Metrics",
        "📋 Prediction History",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### System Status")
    device    = "CUDA" if torch.cuda.is_available() else "CPU"
    available = get_available_categories()
    st.success(f"Device: {device}")
    st.info(f"Models: {len(available)}/15 available")
    if torch.cuda.is_available():
        vu = torch.cuda.memory_allocated(0) / 1024**3
        vt = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.metric("VRAM", f"{vu:.1f} / {vt:.1f} GB")

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
    st.markdown('<p class="main-header">🔍 Surface Defect Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a product image to detect surface defects using PatchCore</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        category = st.selectbox("Product Category", options=get_available_categories())
    with col2:
        threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.01)
    with col3:
        heatmap_alpha = st.slider("Heatmap Opacity", 0.1, 0.9, 0.5, 0.1)

    uploaded_file = st.file_uploader("Upload product image", type=["png","jpg","jpeg","bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        if st.button("🚀 Run Detection", type="primary", width="stretch"):
            with st.spinner(f"Running PatchCore detection on {category}..."):
                model = load_model(category)
                if model is None:
                    st.error(f"Model not found for: {category}")
                    st.stop()

                tensor     = preprocess_image(image)
                drift_info = run_drift_check(tensor, category)

                t0 = time.perf_counter()
                anomaly_score, heatmap = model.predict(tensor)
                latency_ms = (time.perf_counter() - t0) * 1000

                is_defective = anomaly_score > threshold
                verdict      = "defective" if is_defective else "normal"

                # Record metrics safely
                try:
                    m = _get_metrics()
                    m["predict"].labels(category=category, verdict=verdict).inc()
                    m["latency"].labels(category=category).observe(latency_ms / 1000)
                    m["score"].labels(category=category).set(anomaly_score)
                    if is_defective:
                        m["defects"].labels(category=category).inc()
                    else:
                        m["normal"].labels(category=category).inc()
                    if drift_info.get("drifted"):
                        m["drift"].labels(category=category).inc()
                except Exception as e:
                    log.warning("Metrics error: %s", e)

            st.markdown("---")
            st.markdown("### Detection Results")

            if is_defective:
                st.markdown(
                    f'<div class="defect-banner"><span class="badge-defect">⚠️ DEFECTIVE</span>'
                    f'<p style="margin-top:0.5rem;color:#cc0000;">Score {anomaly_score:.4f} exceeds threshold {threshold:.2f}</p></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="normal-banner"><span class="badge-normal">✅ NORMAL</span>'
                    f'<p style="margin-top:0.5rem;color:#006622;">Score {anomaly_score:.4f} below threshold {threshold:.2f}</p></div>',
                    unsafe_allow_html=True)

            st.markdown("")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Anomaly Score", f"{anomaly_score:.4f}")
            m2.metric("Threshold",     f"{threshold:.2f}")
            m3.metric("Latency",       f"{latency_ms:.1f}ms")
            m4.metric("Category",      category.title())
            st.markdown("**Anomaly Score Bar**")
            st.progress(min(float(anomaly_score), 1.0))

            if drift_info.get("drifted"):
                st.warning(f"⚠️ Data drift detected — z-score: {drift_info.get('z_score', 0):.2f}")

            st.markdown("### Image Analysis")
            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                st.markdown("**Original Image**")
                st.image(image, width="stretch")
            with ic2:
                st.markdown("**Anomaly Heatmap**")
                hmap_rgb = np.stack([
                    np.clip(heatmap * 2, 0, 1),
                    np.clip(heatmap * 2 - 1, 0, 1),
                    np.zeros_like(heatmap),
                ], axis=2)
                st.image(Image.fromarray((hmap_rgb * 255).astype(np.uint8)), width="stretch")
            with ic3:
                st.markdown("**Overlay**")
                st.image(create_heatmap_overlay(image, heatmap, heatmap_alpha), width="stretch")

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
                st.image(image, caption=uploaded_file.name, width="stretch")
            with pc2:
                st.info(
                    f"**File:** {uploaded_file.name}\n\n"
                    f"**Size:** {image.size[0]}x{image.size[1]} px\n\n"
                    f"**Category:** {category}\n\n"
                    f"Click **Run Detection** to analyze.")
    else:
        st.markdown("---")
        st.markdown('<div style="text-align:center;padding:3rem;background:#f8f9fa;border-radius:10px;"><h3>Upload an image to get started</h3><p>Supports PNG, JPG, JPEG, BMP</p></div>', unsafe_allow_html=True)
        cats = get_available_categories()
        cols = st.columns(5)
        for i, cat in enumerate(cats):
            cols[i % 5].success(f"✓ {cat}")


# ===========================================================================
# PAGE 2: Pipeline Dashboard
# ===========================================================================
elif page == "📊 Pipeline Dashboard":
    st.markdown('<p class="main-header">📊 ML Pipeline Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Pipeline Stages")
    stages = [
        ("Data Ingestion",      "Airflow DAG",       "src/pipeline/ingest.py",             True),
        ("Preprocessing",       "Custom",            "src/pipeline/preprocess.py",          True),
        ("Feature Engineering", "Custom",            "src/pipeline/feature_engineering.py", True),
        ("Model Training",      "PatchCore",         "src/model/train.py",                  True),
        ("Model Evaluation",    "Scikit-learn",      "src/model/evaluate.py",               True),
        ("Model Registry",      "MLflow",            "DagsHub MLflow",                      True),
        ("App Serving",         "Streamlit",         "src/app.py",                          True),
        ("Metrics",             "Prometheus Client", "src/app.py :8000",                    True),
        ("Monitoring",          "Prometheus",        "monitoring/prometheus.yml",            True),
        ("Alerting",            "Alertmanager",      "monitoring/alertmanager.yml",          True),
        ("Dashboards",          "Grafana",           "monitoring/grafana/",                 True),
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
            df   = pd.DataFrame(data)
            cols = [c for c in ["category","auroc","f1_score","pixel_auroc","avg_latency_ms"] if c in df.columns]
            if cols:
                st.dataframe(df[cols].style.format({"auroc":"{:.4f}","f1_score":"{:.4f}","pixel_auroc":"{:.4f}","avg_latency_ms":"{:.1f}"}), width="stretch")
    else:
        st.info("Run training to generate reports.")


# ===========================================================================
# PAGE 3: Experiment Metrics
# ===========================================================================
elif page == "📈 Experiment Metrics":
    st.markdown('<p class="main-header">📈 MLflow Experiment Metrics</p>', unsafe_allow_html=True)
    st.info("👉 Full experiment tracking:\nhttps://dagshub.com/da25m005/MLOPS_end-2-end_project/experiments")

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
                s3.metric("Avg F1", f"{df['f1_score'].mean():.4f}")
            if "avg_latency_ms" in df.columns:
                s4.metric("Avg Latency", f"{df['avg_latency_ms'].mean():.1f}ms")
            st.markdown("---")
            if "auroc" in df.columns:
                st.markdown("**AUROC by Category**")
                st.bar_chart(df.set_index("category")[["auroc"]].sort_values("auroc", ascending=False))
            if "f1_score" in df.columns:
                st.markdown("**F1 Score by Category**")
                st.bar_chart(df.set_index("category")[["f1_score"]].sort_values("f1_score", ascending=False))
            if "avg_latency_ms" in df.columns:
                st.markdown("**Latency by Category (ms)**")
                st.bar_chart(df.set_index("category")[["avg_latency_ms"]].sort_values("avg_latency_ms"))
            st.markdown("---")
            st.dataframe(df, width="stretch")
    else:
        st.warning("No metrics found. Run: `python -m src.model.train --category all`")


# ===========================================================================
# PAGE 4: Prediction History
# ===========================================================================
elif page == "📋 Prediction History":
    st.markdown('<p class="main-header">📋 Prediction History</p>', unsafe_allow_html=True)
    history = st.session_state.history
    if not history:
        st.info("No predictions yet. Go to Detect Defects and upload an image.")
    else:
        total     = len(history)
        defective = sum(1 for h in history if h["verdict"] == "DEFECTIVE")
        normal    = total - defective
        avg_lat   = sum(h["latency_ms"] for h in history) / total
        drifts    = sum(1 for h in history if h.get("drift"))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total",        total)
        c2.metric("Defective",    defective)
        c3.metric("Normal",       normal)
        c4.metric("Avg Latency",  f"{avg_lat:.1f}ms")
        c5.metric("Drift Events", drifts)
        st.markdown("---")
        import pandas as pd
        df = pd.DataFrame(history)
        def color_verdict(val):
            if val == "DEFECTIVE": return "background-color: #ffe0e0"
            elif val == "NORMAL":  return "background-color: #e0ffe0"
            return ""
        st.dataframe(df.style.applymap(color_verdict, subset=["verdict"]), width="stretch")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    os.environ["PYTHONPATH"] = "."
    os.system("streamlit run src/app.py --server.port 8501 --server.headless false")