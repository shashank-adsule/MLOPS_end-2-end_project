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
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dotenv import load_dotenv
load_dotenv()

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
        log.info("Dataset: %d training tensors for '%s'", len(self.tensor_paths), category)

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        return torch.load(self.tensor_paths[idx], map_location="cpu")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_category(
    category: str,
    params: Dict,
    processed_dir: str,
    models_dir: str,
    reports_dir: str,
) -> Dict:
    """
    Train PatchCore for one category and log everything to MLflow.

    Returns dict of metrics.
    """
    log.info("=" * 60)
    log.info("Training category: %s", category)
    log.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    with mlflow.start_run(run_name=f"patchcore_{category}"):

        # ----------------------------------------------------------------
        # Log parameters
        # ----------------------------------------------------------------
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
        except Exception:
            mlflow.set_tag("git_commit", "unknown")

        mlflow.set_tag("category", category)
        mlflow.set_tag("algorithm", "PatchCore")
        mlflow.set_tag("dataset", "MVTec AD")

        # ----------------------------------------------------------------
        # Build dataloader
        # ----------------------------------------------------------------
        dataset = MVTecTrainDataset(processed_dir, category)
        dataloader = DataLoader(
            dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
            num_workers=params["training"]["num_workers"],
            pin_memory=(device == "cuda"),
        )

        # ----------------------------------------------------------------
        # Initialize and fit PatchCore
        # ----------------------------------------------------------------
        model = PatchCore(
            backbone=params["model"]["backbone"],
            layers=params["model"]["layers"],
            coreset_ratio=params["model"]["coreset_ratio"],
            image_size=params["model"]["patch_size"],
            device=device,
        )

        t_start = time.time()
        model.fit(dataloader)
        train_time = time.time() - t_start

        mlflow.log_metric("train_time_sec", round(train_time, 2))
        mlflow.log_metric("memory_bank_size", len(model.memory_bank))
        log.info("Training complete in %.1fs", train_time)

        # ----------------------------------------------------------------
        # Evaluate
        # ----------------------------------------------------------------
        log.info("Evaluating on test set...")
        metrics = evaluate_category(
            model=model,
            processed_dir=processed_dir,
            category=category,
            reports_dir=reports_dir,
        )

        # Log all evaluation metrics
        mlflow.log_metrics({
            "auroc":            round(metrics["auroc"], 4),
            "f1_score":         round(metrics["f1_score"], 4),
            "pixel_auroc":      round(metrics["pixel_auroc"], 4),
            "avg_latency_ms":   round(metrics["avg_latency_ms"], 2),
            "threshold":        round(metrics["threshold"], 4),
        })

        log.info(
            "Metrics — AUROC: %.4f | F1: %.4f | Pixel-AUROC: %.4f | Latency: %.1fms",
            metrics["auroc"],
            metrics["f1_score"],
            metrics["pixel_auroc"],
            metrics["avg_latency_ms"],
        )

        # Latency SLA check (must be < 200ms per problem statement)
        if metrics["avg_latency_ms"] > 200:
            log.warning(
                "Latency SLA BREACH: %.1fms > 200ms target",
                metrics["avg_latency_ms"],
            )
            mlflow.set_tag("sla_breach", "latency")
        else:
            mlflow.set_tag("sla_breach", "none")

        # ----------------------------------------------------------------
        # Save model artifact
        # ----------------------------------------------------------------
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model_path = str(Path(models_dir) / f"patchcore_{category}.pt")
        model.save(model_path)

        # Log model file as MLflow artifact
        mlflow.log_artifact(model_path, artifact_path="models")

        # Log ROC curve if it exists
        roc_path = Path(reports_dir) / category / "roc_curve.csv"
        if roc_path.exists():
            mlflow.log_artifact(str(roc_path), artifact_path="reports")

        # Log sample heatmaps
        heatmap_dir = Path(reports_dir) / category / "heatmaps"
        if heatmap_dir.exists():
            mlflow.log_artifacts(str(heatmap_dir), artifact_path="heatmaps")

        # Register model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/models"
        try:
            mlflow.register_model(
                model_uri=model_uri,
                name=f"patchcore_{category}",
            )
            log.info("Model registered in MLflow registry: patchcore_%s", category)
        except Exception as e:
            log.warning("Could not register model: %s", e)

        metrics["category"] = category
        metrics["run_id"] = mlflow.active_run().info.run_id
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
        help="Category to train on. Use 'all' for all 15. Defaults to params.yaml value.",
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

    processed_dir = params["data"]["processed_dir"]
    models_dir    = "models"
    reports_dir   = "reports"

    # Determine categories to train
    if args.category == "all":
        categories = CATEGORIES
    elif args.category:
        categories = [args.category]
    else:
        categories = [params["model"]["category"]]

    # ----------------------------------------------------------------
    # Configure MLflow → DagsHub
    # ----------------------------------------------------------------
    dagshub_token = os.environ.get("DAGSHUB_TOKEN", "")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "da25m005"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/da25m005/MLOPS_end-2-end_project.mlflow"
    )
    mlflow.set_experiment("patchcore-mvtec-ad")
    log.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())

    # ----------------------------------------------------------------
    # Train all requested categories
    # ----------------------------------------------------------------
    all_metrics = []
    for category in categories:
        try:
            metrics = train_category(
                category=category,
                params=params,
                processed_dir=processed_dir,
                models_dir=models_dir,
                reports_dir=reports_dir,
            )
            all_metrics.append(metrics)
        except Exception as e:
            log.error("Failed to train category '%s': %s", category, e)
            continue

    # Save summary metrics
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(reports_dir) / "metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    log.info("Training complete. Summary saved to %s", summary_path)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Category':<15} {'AUROC':>8} {'F1':>8} {'Pixel-AUROC':>12} {'Latency':>10}")
    print("-" * 70)
    for m in all_metrics:
        print(
            f"{m['category']:<15} "
            f"{m['auroc']:>8.4f} "
            f"{m['f1_score']:>8.4f} "
            f"{m['pixel_auroc']:>12.4f} "
            f"{m['avg_latency_ms']:>9.1f}ms"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()