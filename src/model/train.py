"""
src/model/train.py
-------------------
PatchCore training script with full MLflow experiment tracking.

Tracks:
  - Parameters: backbone, layers, coreset_ratio, image_size, category
  - Metrics: AUROC, F1-score, pixel-AUROC, inference latency
  - Artifacts: model file, heatmap samples, ROC curve
  - Tags: git commit hash, DVC data version

MLflow tracking URI points to DagsHub hosted MLflow server:
  https://dagshub.com/da25m005/MLOPS_end-2-end_project.mlflow

Usage:
  python -m src.model.train                        # uses params.yaml
  python -m src.model.train --category bottle      # single category
  python -m src.model.train --category all         # all 15 categories
"""

import argparse
import json
import logging
import multiprocessing
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.model.patchcore import PatchCore
from src.model.evaluate import evaluate_category

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MVTecTrainDataset(Dataset):
    """Loads preprocessed .pt tensors from data/processed/{category}/train/"""

    def __init__(self, processed_dir: str, category: str):
        self.tensor_paths = sorted(
            Path(processed_dir, category, "train").glob("*.pt")
        )
        if not self.tensor_paths:
            raise FileNotFoundError(
                f"No tensors found in {processed_dir}/{category}/train/. "
                "Run the data pipeline first: python -m src.pipeline.preprocess"
            )
        log.info("  Found %d training tensors for '%s'", len(self.tensor_paths), category)

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        return torch.load(
            self.tensor_paths[idx],
            map_location="cpu",
            weights_only=True,
        )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_category(
    category: str,
    params: Dict,
    processed_dir: str,
    models_dir: str,
    reports_dir: str,
    category_index: int,
    total_categories: int,
    local_client=None,
    local_exp_id=None,
) -> Dict:
    """Train PatchCore for one category and log to DagsHub + local MLflow."""

    print(f"\n{'='*65}")
    print(f"  [{category_index}/{total_categories}] Starting: {category.upper()}")
    print(f"{'='*65}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_used  = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  GPU      : {gpu_name}")
        print(f"  VRAM     : {vram_used:.1f}GB used / {vram_total:.1f}GB total")
    else:
        print(f"  Device   : CPU")

    print(f"  Category : {category}")
    print(f"  Backbone : {params['model']['backbone']}")
    print(f"  Coreset  : {params['model']['coreset_ratio']}")
    print(f"  Batch    : {params['training']['batch_size']}")
    print(f"{'-'*65}")

    with mlflow.start_run(run_name=f"patchcore_{category}"):

        # ----------------------------------------------------------------
        # Log parameters
        # ----------------------------------------------------------------
        print(f"  [1/5] Logging parameters to MLflow...")
        mlflow.log_params({
            "category":      category,
            "backbone":      params["model"]["backbone"],
            "layers":        str(params["model"]["layers"]),
            "coreset_ratio": params["model"]["coreset_ratio"],
            "image_size":    params["model"]["image_size"],
            "patch_size":    params["model"]["patch_size"],
            "batch_size":    params["training"]["batch_size"],
            "device":        device,
        })

        # Log git commit hash for reproducibility
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            mlflow.set_tag("git_commit", git_hash)
            print(f"       Git commit : {git_hash}")
        except Exception:
            mlflow.set_tag("git_commit", "unknown")

        mlflow.set_tag("category", category)
        mlflow.set_tag("algorithm", "PatchCore")
        mlflow.set_tag("dataset", "MVTec AD")
        print(f"  [1/5] Parameters logged OK")

        # ----------------------------------------------------------------
        # Build dataloader
        # ----------------------------------------------------------------
        print(f"\n  [2/5] Loading dataset...")
        dataset = MVTecTrainDataset(processed_dir, category)

        # Windows fix: num_workers must be 0 on Windows
        num_workers = 0 if platform.system() == "Windows" else params["training"]["num_workers"]
        if platform.system() == "Windows":
            print(f"       Windows detected — using num_workers=0")

        dataloader = DataLoader(
            dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        print(f"       Batches : {len(dataloader)}")
        print(f"  [2/5] Dataset loaded OK")

        # ----------------------------------------------------------------
        # Initialize PatchCore
        # ----------------------------------------------------------------
        print(f"\n  [3/5] Building PatchCore memory bank...")
        print(f"       Extracting features from {params['model']['backbone']}...")
        print(f"       Layers: {params['model']['layers']}")
        print(f"       This may take a few minutes...")

        model = PatchCore(
            backbone=params["model"]["backbone"],
            layers=params["model"]["layers"],
            coreset_ratio=params["model"]["coreset_ratio"],
            image_size=params["model"]["patch_size"],
            device=device,
        )

        t_start = time.time()

        # Fit with progress updates
        model.fit(dataloader)

        train_time = round(time.time() - t_start, 2)
        memory_bank_size = len(model.memory_bank)

        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("memory_bank_size", memory_bank_size)

        print(f"       Memory bank size : {memory_bank_size} patches")
        print(f"       Time taken       : {train_time}s")
        print(f"  [3/5] Memory bank built OK")

        # ----------------------------------------------------------------
        # Evaluate
        # ----------------------------------------------------------------
        print(f"\n  [4/5] Evaluating on test set...")
        print(f"       Running inference on all test images...")

        metrics = evaluate_category(
            model=model,
            processed_dir=processed_dir,
            category=category,
            reports_dir=reports_dir,
        )

        mlflow.log_metrics({
            "auroc":          round(metrics["auroc"], 4),
            "f1_score":       round(metrics["f1_score"], 4),
            "pixel_auroc":    round(metrics["pixel_auroc"], 4),
            "avg_latency_ms": round(metrics["avg_latency_ms"], 2),
            "threshold":      round(metrics["threshold"], 4),
        })

        print(f"\n       +--------------------------+----------+")
        print(f"       | Metric                   | Value    |")
        print(f"       +--------------------------+----------+")
        print(f"       | Image AUROC              | {metrics['auroc']:.4f}   |")
        print(f"       | F1 Score                 | {metrics['f1_score']:.4f}   |")
        print(f"       | Pixel AUROC              | {metrics['pixel_auroc']:.4f}   |")
        print(f"       | Avg Latency              | {metrics['avg_latency_ms']:.1f}ms   |")
        print(f"       | Threshold                | {metrics['threshold']:.4f}   |")
        print(f"       | Normal samples           | {metrics['n_normal']}      |")
        print(f"       | Defect samples           | {metrics['n_defect']}      |")
        print(f"       +--------------------------+----------+")

        # Latency SLA check
        if metrics["avg_latency_ms"] > 200:
            print(f"       WARNING: Latency {metrics['avg_latency_ms']:.1f}ms exceeds 200ms SLA!")
            mlflow.set_tag("sla_breach", "latency")
        else:
            print(f"       Latency SLA: PASS ({metrics['avg_latency_ms']:.1f}ms < 200ms)")
            mlflow.set_tag("sla_breach", "none")

        print(f"  [4/5] Evaluation complete")

        # ----------------------------------------------------------------
        # Save model + artifacts
        # ----------------------------------------------------------------
        print(f"\n  [5/5] Saving model and artifacts...")

        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model_path = str(Path(models_dir) / f"patchcore_{category}.pt")
        model.save(model_path)
        print(f"       Model saved  : {model_path}")

        mlflow.log_artifact(model_path, artifact_path="models")
        print(f"       Artifact logged to MLflow")

        # Log ROC curve
        roc_path = Path(reports_dir) / category / "roc_curve.csv"
        if roc_path.exists():
            mlflow.log_artifact(str(roc_path), artifact_path="reports")
            print(f"       ROC curve logged")

        # Log heatmaps
        heatmap_dir = Path(reports_dir) / category / "heatmaps"
        if heatmap_dir.exists():
            mlflow.log_artifacts(str(heatmap_dir), artifact_path="heatmaps")
            print(f"       Heatmaps logged")

        # Register in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/models"
        try:
            mlflow.register_model(model_uri=model_uri, name=f"patchcore_{category}")
            print(f"       Registered in MLflow registry: patchcore_{category}")
        except Exception as e:
            print(f"       Registry warning: {e}")

        run_id = mlflow.active_run().info.run_id
        print(f"       MLflow run ID: {run_id}")
        print(f"  [5/5] Artifacts saved OK")

        print(f"\n  DONE: {category} — AUROC={metrics['auroc']:.4f} | "
              f"F1={metrics['f1_score']:.4f} | Latency={metrics['avg_latency_ms']:.1f}ms")

        metrics["category"] = category
        metrics["run_id"]   = run_id

        # ── Log to LOCAL MLflow as well ──────────────────────────────────
        if local_client and local_exp_id:
            try:
                local_run = local_client.create_run(
                    experiment_id=local_exp_id,
                    run_name=f"patchcore_{category}",
                )
                lid = local_run.info.run_id

                # Params
                for k, v in {
                    "category":      category,
                    "backbone":      params["model"]["backbone"],
                    "layers":        str(params["model"]["layers"]),
                    "coreset_ratio": params["model"]["coreset_ratio"],
                    "image_size":    params["model"]["image_size"],
                    "batch_size":    params["training"]["batch_size"],
                    "device":        "cuda" if torch.cuda.is_available() else "cpu",
                }.items():
                    local_client.log_param(lid, k, v)

                # Metrics
                for k, v in {
                    "auroc":          metrics["auroc"],
                    "f1_score":       metrics["f1_score"],
                    "pixel_auroc":    metrics["pixel_auroc"],
                    "avg_latency_ms": metrics["avg_latency_ms"],
                    "threshold":      metrics["threshold"],
                }.items():
                    local_client.log_metric(lid, k, v)

                # Tags
                local_client.set_tag(lid, "category",  category)
                local_client.set_tag(lid, "algorithm", "PatchCore")
                local_client.set_tag(lid, "dataset",   "MVTec AD")
                local_client.set_tag(lid, "dagshub_run_id", run_id)

                local_client.set_terminated(lid, status="FINISHED")
                print(f"       Local MLflow run logged: {lid}")
            except Exception as e:
                print(f"       Local MLflow logging failed: {e}")

        return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec AD")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category to train. Use 'all' for all 15. Default: params.yaml value.",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="Path to params.yaml",
    )
    args = parser.parse_args()

    # Load params
    with open(args.params) as f:
        params = yaml.safe_load(f)

    # Force num_workers=0 on Windows
    if platform.system() == "Windows":
        params["training"]["num_workers"] = 0

    processed_dir = params["data"]["processed_dir"]
    models_dir    = "models"
    reports_dir   = "reports"

    # Determine categories
    if args.category == "all":
        categories = CATEGORIES
    elif args.category:
        categories = [args.category]
    else:
        categories = [params["model"]["category"]]

    # ----------------------------------------------------------------
    # Startup banner
    # ----------------------------------------------------------------
    print(f"\n{'#'*65}")
    print(f"#  PatchCore Training — MVTec AD Anomaly Detection")
    print(f"#  Categories : {len(categories)}")
    print(f"#  Device     : {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"#  System     : {platform.system()}")
    print(f"{'#'*65}\n")

    # ----------------------------------------------------------------
    # Configure MLflow — logs to BOTH DagsHub AND local MLflow
    # ----------------------------------------------------------------
    dagshub_token = os.environ.get("DAGSHUB_TOKEN", "")
    dagshub_uri   = "https://dagshub.com/da25m005/MLOPS_end-2-end_project.mlflow"
    local_uri     = os.environ.get("MLFLOW_LOCAL_URI", "http://localhost:5000")

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "da25m005"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        print(f"MLflow auth     : DagsHub token loaded from .env")
    else:
        print(f"WARNING: DAGSHUB_TOKEN not found — DagsHub logging may fail")

    # Primary tracking URI = DagsHub (used by mlflow.start_run)
    mlflow.set_tracking_uri(dagshub_uri)
    mlflow.set_experiment("patchcore-mvtec-ad")

    # Secondary client for local MLflow
    local_client = mlflow.tracking.MlflowClient(tracking_uri=local_uri)
    try:
        exps = local_client.search_experiments()
        exp_names = [e.name for e in exps]
        if "patchcore-mvtec-ad" not in exp_names:
            local_client.create_experiment("patchcore-mvtec-ad")
        local_exp = local_client.get_experiment_by_name("patchcore-mvtec-ad")
        local_exp_id = local_exp.experiment_id
        print(f"Local MLflow    : {local_uri} — experiment ready")
    except Exception as e:
        local_client  = None
        local_exp_id  = None
        print(f"Local MLflow    : UNAVAILABLE ({e}) — skipping local logging")

    print(f"DagsHub MLflow  : {dagshub_uri}")
    print(f"Local MLflow    : {local_uri}")
    print(f"Categories      : {categories}\n")

    # ----------------------------------------------------------------
    # Train all categories
    # ----------------------------------------------------------------
    all_metrics = []
    t_total = time.time()

    for i, category in enumerate(categories, start=1):
        try:
            metrics = train_category(
                category=category,
                params=params,
                processed_dir=processed_dir,
                models_dir=models_dir,
                reports_dir=reports_dir,
                category_index=i,
                total_categories=len(categories),
                local_client=local_client,
                local_exp_id=local_exp_id,
            )
            all_metrics.append(metrics)
        except Exception as e:
            log.error("Failed category '%s': %s", category, e)
            print(f"\n  ERROR in {category}: {e}\n")
            continue

    total_time = round(time.time() - t_total, 1)

    # ----------------------------------------------------------------
    # Save summary + print table
    # ----------------------------------------------------------------
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(reports_dir) / "metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n\n{'#'*65}")
    print(f"#  TRAINING COMPLETE — {len(all_metrics)}/{len(categories)} categories")
    print(f"#  Total time: {total_time}s ({total_time/60:.1f} mins)")
    print(f"{'#'*65}")
    print(f"\n{'='*65}")
    print(f"{'Category':<15} {'AUROC':>8} {'F1':>8} {'Pixel-AUROC':>12} {'Latency':>12}")
    print(f"{'-'*65}")
    for m in all_metrics:
        print(
            f"{m['category']:<15} "
            f"{m['auroc']:>8.4f} "
            f"{m['f1_score']:>8.4f} "
            f"{m['pixel_auroc']:>12.4f} "
            f"{m['avg_latency_ms']:>10.1f}ms"
        )
    print(f"{'='*65}")
    print(f"\nSummary saved  : {summary_path}")
    print(f"MLflow runs    : {dagshub_uri }")
    print(f"Next step      : python -m src.api.main")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print(f"Starting training with PyTorch {torch.__version__} on {platform.system()}")
    main()