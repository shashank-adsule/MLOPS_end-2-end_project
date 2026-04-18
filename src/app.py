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
  - Data drift monitoring

Usage:
  streamlit run src/app.py
"""

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image
import torchvision.transforms as T

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
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
MODELS_DIR    = "models"
PROCESSED_DIR = PARAMS["data"]["processed_dir"]
IMAGE_SIZE    = PARAMS["model"]["image_size"]
PATCH_SIZE    = PARAMS["model"]["patch_size"]
CATEGORIES    = PARAMS["data"]["categories"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Model loading — cached so it only loads once per session
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
    model  = PatchCore.load(str(model_path), device=device)
    return model


def get_available_categories() -> List[str]:
    """Return categories with trained model files."""
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
# Heatmap overlay
# ---------------------------------------------------------------------------
def create_heatmap_overlay(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay coloured heatmap on original image."""
    orig_resized = original.resize((PATCH_SIZE, PATCH_SIZE)).convert("RGB")
    orig_array   = np.array(orig_resized, dtype=np.float32)

    # Build RGB heatmap (red = high anomaly)
    h, w   = heatmap.shape
    heat_rgb = np.zeros((h, w, 3), dtype=np.float32)
    heat_rgb[:, :, 0] = np.clip(heatmap * 2,       0, 1) * 255   # Red
    heat_rgb[:, :, 1] = np.clip(heatmap * 2 - 1,   0, 1) * 255   # Green (high only)
    heat_rgb[:, :, 2] = 0                                          # No blue

    # Blend
    blended = (1 - alpha) * orig_array + alpha * heat_rgb
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Prediction history in session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .defect-badge {
        background: #ff4b4b;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .normal-badge {
        background: #00cc44;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stProgress > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/factory.png", width=60)
    st.title("Defect Detection")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["🔍 Detect Defects", "📊 Pipeline Dashboard", "📈 Experiment Metrics", "📋 Prediction History"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # System status
    st.markdown("### System Status")
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    available = get_available_categories()

    st.success(f"Device: {device}")
    st.info(f"Models available: {len(available)}/15")

    if torch.cuda.is_available():
        vram_used  = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.metric("VRAM", f"{vram_used:.1f} / {vram_total:.1f} GB")

    st.markdown("---")
    st.markdown("### Links")
    st.markdown("🔗 [MLflow on DagsHub](https://dagshub.com/da25m005/MLOPS_end-2-end_project/experiments)")
    st.markdown("🔗 [GitHub Repo](https://github.com/shashank-adsule/MLOPS_end-2-end_project)")
    st.markdown("🔗 [DagsHub Repo](https://dagshub.com/da25m005/MLOPS_end-2-end_project)")


# ===========================================================================
# PAGE 1: Detect Defects
# ===========================================================================
if page == "🔍 Detect Defects":

    st.markdown('<p class="main-header">🔍 Surface Defect Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a product image to detect surface defects using PatchCore anomaly detection</p>', unsafe_allow_html=True)

    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        available = get_available_categories()
        category = st.selectbox(
            "Product Category",
            options=available,
            help="Select the product type matching your image",
        )

    with col2:
        threshold = st.slider(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Score above this = DEFECTIVE",
        )

    with col3:
        heatmap_alpha = st.slider(
            "Heatmap Opacity",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
        )

    # Upload
    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Upload an image of the product to inspect",
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Run detection button
        if st.button("🚀 Run Detection", type="primary", use_container_width=True):

            with st.spinner(f"Loading {category} model and running detection..."):

                # Load model
                model = load_model(category)
                if model is None:
                    st.error(f"Model not found for category: {category}")
                    st.stop()

                # Preprocess + predict
                t0     = time.perf_counter()
                tensor = preprocess_image(image)
                anomaly_score, heatmap = model.predict(tensor)
                latency_ms = (time.perf_counter() - t0) * 1000

                is_defective = anomaly_score > threshold

            # ----------------------------------------------------------------
            # Results
            # ----------------------------------------------------------------
            st.markdown("---")
            st.markdown("### Detection Results")

            # Verdict banner
            if is_defective:
                st.markdown(
                    f'<div style="text-align:center; padding:1rem; background:#ffe0e0; border-radius:10px; border:2px solid #ff4b4b;">'
                    f'<span class="defect-badge">⚠️ DEFECTIVE</span>'
                    f'<p style="margin-top:0.5rem; color:#cc0000;">Anomaly detected — score {anomaly_score:.4f} exceeds threshold {threshold:.2f}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="text-align:center; padding:1rem; background:#e0ffe0; border-radius:10px; border:2px solid #00cc44;">'
                    f'<span class="normal-badge">✅ NORMAL</span>'
                    f'<p style="margin-top:0.5rem; color:#006622;">No defect detected — score {anomaly_score:.4f} below threshold {threshold:.2f}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Anomaly Score",  f"{anomaly_score:.4f}")
            m2.metric("Threshold",      f"{threshold:.2f}")
            m3.metric("Latency",        f"{latency_ms:.1f}ms")
            m4.metric("Category",       category.title())

            # Anomaly score bar
            st.markdown("**Anomaly Score**")
            score_color = "normal" if not is_defective else "inverse"
            st.progress(min(anomaly_score, 1.0))

            # Images side by side
            st.markdown("### Image Analysis")
            img_col1, img_col2, img_col3 = st.columns(3)

            with img_col1:
                st.markdown("**Original Image**")
                st.image(image, use_column_width=True)

            with img_col2:
                st.markdown("**Anomaly Heatmap**")
                heatmap_img = Image.fromarray(
                    (np.stack([
                        np.clip(heatmap * 2, 0, 1),
                        np.clip(heatmap * 2 - 1, 0, 1),
                        np.zeros_like(heatmap),
                    ], axis=2) * 255).astype(np.uint8)
                )
                st.image(heatmap_img, use_column_width=True)

            with img_col3:
                st.markdown("**Overlay**")
                overlay = create_heatmap_overlay(image, heatmap, alpha=heatmap_alpha)
                st.image(overlay, use_column_width=True)

            # Save to history
            st.session_state.history.append({
                "timestamp":     time.strftime("%H:%M:%S"),
                "filename":      uploaded_file.name,
                "category":      category,
                "anomaly_score": round(anomaly_score, 4),
                "verdict":       "DEFECTIVE" if is_defective else "NORMAL",
                "latency_ms":    round(latency_ms, 1),
                "threshold":     threshold,
            })

        else:
            # Show preview before running
            st.markdown("### Image Preview")
            prev_col1, prev_col2 = st.columns([1, 2])
            with prev_col1:
                st.image(image, caption=uploaded_file.name, use_column_width=True)
            with prev_col2:
                st.info(
                    f"**File:** {uploaded_file.name}\n\n"
                    f"**Size:** {image.size[0]}×{image.size[1]} px\n\n"
                    f"**Category:** {category}\n\n"
                    f"**Threshold:** {threshold}\n\n"
                    f"Click **Run Detection** to analyze."
                )

    else:
        # Empty state
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; padding:3rem; background:#f8f9fa; border-radius:10px;">'
            '<h3>Upload an image to get started</h3>'
            '<p>Supports PNG, JPG, JPEG, BMP</p>'
            '<p>Select the matching product category from the dropdown above</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Sample categories
        st.markdown("### Available Categories")
        cats = get_available_categories()
        cols = st.columns(5)
        for i, cat in enumerate(cats):
            cols[i % 5].success(f"✓ {cat}")


# ===========================================================================
# PAGE 2: Pipeline Dashboard
# ===========================================================================
elif page == "📊 Pipeline Dashboard":

    st.markdown('<p class="main-header">📊 ML Pipeline Dashboard</p>', unsafe_allow_html=True)

    # Pipeline stages
    st.markdown("### Pipeline Stages")

    stages = [
        ("Data Ingestion",      "Airflow DAG",    "src/pipeline/ingest.py",           True),
        ("Preprocessing",       "Custom",         "src/pipeline/preprocess.py",        True),
        ("Feature Engineering", "Custom",         "src/pipeline/feature_engineering.py", True),
        ("Model Training",      "PatchCore",      "src/model/train.py",                True),
        ("Model Evaluation",    "Scikit-learn",   "src/model/evaluate.py",             True),
        ("Model Registry",      "MLflow",         "DagsHub MLflow",                    True),
        ("API Serving",         "FastAPI/Streamlit", "src/app.py",                     True),
        ("Monitoring",          "Prometheus",     "src/api/metrics.py",                True),
    ]

    for i, (stage, tool, path, done) in enumerate(stages):
        col1, col2, col3, col4 = st.columns([3, 2, 3, 1])
        with col1:
            st.write(f"**{i+1}. {stage}**")
        with col2:
            st.write(f"`{tool}`")
        with col3:
            st.write(f"`{path}`")
        with col4:
            st.write("✅" if done else "⏳")

    st.markdown("---")

    # Model status
    st.markdown("### Trained Models")
    available = get_available_categories()
    missing   = [c for c in CATEGORIES if c not in available]

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**{len(available)} models trained**")
        for cat in available:
            size = Path(MODELS_DIR, f"patchcore_{cat}.pt").stat().st_size / 1024**2
            st.write(f"✅ {cat} — {size:.0f} MB")

    with col2:
        if missing:
            st.warning(f"**{len(missing)} models missing**")
            for cat in missing:
                st.write(f"❌ {cat}")
        else:
            st.success("All 15 categories trained!")

    st.markdown("---")

    # DVC status
    st.markdown("### Data Version Control")
    c1, c2, c3 = st.columns(3)
    c1.metric("DVC Remote",  "DagsHub")
    c2.metric("Dataset",     "MVTec AD")
    c3.metric("Files",       "6,644")

    st.markdown("---")

    # Reports
    st.markdown("### Evaluation Reports")
    reports_dir = Path("reports")
    if reports_dir.exists():
        summary = reports_dir / "metrics.json"
        if summary.exists():
            with open(summary) as f:
                metrics_data = json.load(f)

            if metrics_data:
                import pandas as pd
                df = pd.DataFrame(metrics_data)
                cols_to_show = [
                    c for c in
                    ["category", "auroc", "f1_score", "pixel_auroc", "avg_latency_ms"]
                    if c in df.columns
                ]
                if cols_to_show:
                    st.dataframe(
                        df[cols_to_show].style.format({
                            "auroc":          "{:.4f}",
                            "f1_score":       "{:.4f}",
                            "pixel_auroc":    "{:.4f}",
                            "avg_latency_ms": "{:.1f}",
                        }),
                        use_container_width=True,
                    )
        else:
            st.info("Run training to generate reports: `python -m src.model.train --category all`")
    else:
        st.info("No reports found yet.")


# ===========================================================================
# PAGE 3: Experiment Metrics
# ===========================================================================
elif page == "📈 Experiment Metrics":

    st.markdown('<p class="main-header">📈 MLflow Experiment Metrics</p>', unsafe_allow_html=True)

    st.info(
        "View full experiment tracking on DagsHub MLflow:\n\n"
        "👉 https://dagshub.com/da25m005/MLOPS_end-2-end_project/experiments"
    )

    # Load local metrics if available
    reports_dir = Path("reports")
    summary     = reports_dir / "metrics.json"

    if summary.exists():
        with open(summary) as f:
            metrics_data = json.load(f)

        if metrics_data:
            import pandas as pd
            df = pd.DataFrame(metrics_data)

            # Summary metrics
            st.markdown("### Summary")
            s1, s2, s3, s4 = st.columns(4)
            if "auroc" in df.columns:
                s1.metric("Avg AUROC",    f"{df['auroc'].mean():.4f}")
                s2.metric("Best AUROC",   f"{df['auroc'].max():.4f}")
            if "f1_score" in df.columns:
                s3.metric("Avg F1",       f"{df['f1_score'].mean():.4f}")
            if "avg_latency_ms" in df.columns:
                s4.metric("Avg Latency",  f"{df['avg_latency_ms'].mean():.1f}ms")

            st.markdown("---")
            st.markdown("### Per-Category Results")

            # AUROC chart
            if "auroc" in df.columns and "category" in df.columns:
                st.markdown("**Image-level AUROC by Category**")
                chart_df = df.set_index("category")[["auroc"]].sort_values("auroc", ascending=False)
                st.bar_chart(chart_df)

            # F1 chart
            if "f1_score" in df.columns and "category" in df.columns:
                st.markdown("**F1 Score by Category**")
                chart_df2 = df.set_index("category")[["f1_score"]].sort_values("f1_score", ascending=False)
                st.bar_chart(chart_df2)

            # Latency chart
            if "avg_latency_ms" in df.columns and "category" in df.columns:
                st.markdown("**Inference Latency by Category (ms)**")
                lat_df = df.set_index("category")[["avg_latency_ms"]].sort_values("avg_latency_ms")
                st.bar_chart(lat_df)

            st.markdown("---")
            st.markdown("### Full Results Table")
            st.dataframe(df, use_container_width=True)

    else:
        st.warning("No local metrics found. Run training first.")
        st.code("python -m src.model.train --category all")


# ===========================================================================
# PAGE 4: Prediction History
# ===========================================================================
elif page == "📋 Prediction History":

    st.markdown('<p class="main-header">📋 Prediction History</p>', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.info("No predictions made yet. Go to **Detect Defects** and upload an image.")
    else:
        # Summary
        total     = len(history)
        defective = sum(1 for h in history if h["verdict"] == "DEFECTIVE")
        normal    = total - defective
        avg_lat   = sum(h["latency_ms"] for h in history) / total

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", total)
        c2.metric("Defective",  defective, delta=f"{defective/total*100:.0f}%")
        c3.metric("Normal",     normal,    delta=f"{normal/total*100:.0f}%")
        c4.metric("Avg Latency", f"{avg_lat:.1f}ms")

        st.markdown("---")

        # History table
        import pandas as pd
        df = pd.DataFrame(history)
        st.dataframe(
            df.style.applymap(
                lambda v: "background-color: #ffe0e0" if v == "DEFECTIVE"
                          else "background-color: #e0ffe0" if v == "NORMAL"
                          else "",
                subset=["verdict"],
            ),
            use_container_width=True,
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


if __name__ == "__main__":
    import subprocess
    import sys
    import os

    os.chdir(Path(__file__).parent.parent)  # go to project root

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        __file__,
        "--server.port", "8501",
        "--server.headless", "false",
    ], check=True)