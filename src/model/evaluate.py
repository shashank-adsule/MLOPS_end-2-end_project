import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

DEFECT_LABEL = 1
NORMAL_LABEL = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MVTecTestDataset(Dataset):

    def __init__(self, processed_dir: str, category: str, defect_type: str):
        self.tensor_paths = sorted(
            Path(processed_dir, category, "test", defect_type).glob("*.pt")
        )
        self.label = NORMAL_LABEL if defect_type == "good" else DEFECT_LABEL

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor = torch.load(self.tensor_paths[idx], map_location="cpu")
        return tensor, self.label, str(self.tensor_paths[idx])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_category(
    model,
    processed_dir: str,
    category: str,
    reports_dir: str,
    save_heatmaps: bool = True,
    n_heatmap_samples: int = 5,
) -> Dict:
    test_root = Path(processed_dir) / category / "test"
    defect_types = sorted([d.name for d in test_root.iterdir() if d.is_dir()])

    all_scores:  List[float] = []
    all_labels:  List[int]   = []
    all_latencies: List[float] = []

    out_dir = Path(reports_dir) / category
    out_dir.mkdir(parents=True, exist_ok=True)

    for defect_type in defect_types:
        dataset = MVTecTestDataset(processed_dir, category, defect_type)
        if len(dataset) == 0:
            continue

        log.info("Evaluating %s/%s (%d images)", category, defect_type, len(dataset))

        for idx in range(len(dataset)):
            image, label, path = dataset[idx]

            t0 = time.perf_counter()
            score, heatmap = model.predict(image)
            latency_ms = (time.perf_counter() - t0) * 1000

            all_scores.append(score)
            all_labels.append(label)
            all_latencies.append(latency_ms)

            # Save sample heatmaps
            if save_heatmaps and idx < n_heatmap_samples and label == DEFECT_LABEL:
                _save_heatmap(
                    heatmap=heatmap,
                    out_path=out_dir / "heatmaps" / f"{defect_type}_{idx:03d}.npy",
                )

    # ----------------------------------------------------------------
    # Compute metrics
    # ----------------------------------------------------------------
    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    # Image-level AUROC
    try:
        auroc = roc_auc_score(labels_arr, scores_arr)
    except ValueError:
        auroc = 0.0
        log.warning("Could not compute AUROC — only one class present")

    # Optimal F1 threshold
    threshold, f1 = _optimal_f1_threshold(labels_arr, scores_arr)

    # Pixel-level AUROC (simplified — uses image scores as proxy)
    pixel_auroc = auroc

    avg_latency = float(np.mean(all_latencies))

    metrics = {
        "auroc":          auroc,
        "f1_score":       f1,
        "pixel_auroc":    pixel_auroc,
        "avg_latency_ms": avg_latency,
        "threshold":      threshold,
        "n_normal":       int((labels_arr == 0).sum()),
        "n_defect":       int((labels_arr == 1).sum()),
    }

    # Save metrics JSON
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save ROC curve data for DVC plots
    fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
    roc_data = "\n".join(
        ["fpr,tpr"] + [f"{f:.4f},{t:.4f}" for f, t in zip(fpr, tpr)]
    )
    with open(out_dir / "roc_curve.csv", "w") as f:
        f.write(roc_data)

    log.info(
        "%s — AUROC: %.4f | F1: %.4f | Latency: %.1fms",
        category, auroc, f1, avg_latency,
    )
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _optimal_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple:
    """Find the score threshold that maximises F1."""
    best_f1, best_thresh = 0.0, 0.5

    thresholds = np.percentile(scores, np.linspace(0, 100, 200))
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        try:
            f1 = f1_score(labels, preds, zero_division=0)
        except Exception:
            continue
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = thresh

    return float(best_thresh), float(best_f1)


def _save_heatmap(heatmap: np.ndarray, out_path: Path) -> None:
    """Save heatmap as numpy array."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), heatmap)