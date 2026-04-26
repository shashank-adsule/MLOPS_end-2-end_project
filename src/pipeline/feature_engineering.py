import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

log = logging.getLogger(__name__)

BASELINE_FILENAME = "baseline.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_baseline_statistics(
    processed_dir: str,
    category: str,
) -> Dict:
    train_dir = Path(processed_dir) / category / "train"
    stats_dir = Path(processed_dir) / category / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    tensor_paths = sorted(train_dir.glob("*.pt"))
    if not tensor_paths:
        raise FileNotFoundError(
            f"No preprocessed tensors found in {train_dir}. "
            "Run preprocess_all_categories first."
        )

    log.info(
        "Computing baselines for %s from %d tensors",
        category, len(tensor_paths),
    )

    # --- Collect all tensors into one array ---
    tensors: List[torch.Tensor] = []
    for path in tensor_paths:
        try:
            t = torch.load(path, map_location="cpu")   # shape: [3, H, W]
            tensors.append(t)
        except Exception as exc:
            log.warning("Skipping unreadable tensor %s: %s", path, exc)

    if not tensors:
        raise RuntimeError(f"No valid tensors loaded for {category}")

    stack = torch.stack(tensors)   # [N, 3, H, W]

    # --- Per-channel pixel statistics (for drift detection) ---
    channel_means = stack.mean(dim=[0, 2, 3]).tolist()   # [3]
    channel_stds  = stack.std(dim=[0, 2, 3]).tolist()    # [3]

    # --- Global pixel intensity stats ---
    flat = stack.view(stack.shape[0], -1)   # [N, 3*H*W]
    global_mean = float(flat.mean())
    global_std  = float(flat.std())
    global_min  = float(flat.min())
    global_max  = float(flat.max())

    # --- Per-image mean intensity (for anomaly score calibration) ---
    per_image_means = flat.mean(dim=1).numpy()   # [N]
    intensity_p5    = float(np.percentile(per_image_means, 5))
    intensity_p95   = float(np.percentile(per_image_means, 95))

    # --- Assemble baseline document ---
    baseline = {
        "category": category,
        "n_train_images": len(tensors),
        "channel_means": channel_means,   # R, G, B mean (normalized space)
        "channel_stds":  channel_stds,    # R, G, B std
        "global_mean":   global_mean,
        "global_std":    global_std,
        "global_min":    global_min,
        "global_max":    global_max,
        "intensity_p5":  intensity_p5,    # lower fence for drift
        "intensity_p95": intensity_p95,   # upper fence for drift
        # Drift alert thresholds (±3σ from mean)
        "drift_low":  global_mean - 3 * global_std,
        "drift_high": global_mean + 3 * global_std,
    }

    out_path = stats_dir / BASELINE_FILENAME
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)

    log.info("Baseline saved to %s", out_path)
    _log_summary(baseline)
    return baseline


def load_baseline(processed_dir: str, category: str) -> Dict:
    path = Path(processed_dir) / category / "stats" / BASELINE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline not found for category '{category}' at {path}. "
            "Run compute_baseline_statistics first."
        )
    with open(path) as f:
        return json.load(f)


def check_drift(
    image_tensor: torch.Tensor,
    baseline: Dict,
    sigma_threshold: float = 3.0,
) -> Dict:
    img_mean  = float(image_tensor.mean())
    baseline_mean = baseline["global_mean"]
    baseline_std  = baseline["global_std"]

    if baseline_std == 0:
        z_score = 0.0
    else:
        z_score = abs(img_mean - baseline_mean) / baseline_std

    drifted = z_score > sigma_threshold

    # Per-channel comparison
    channel_means = image_tensor.mean(dim=[1, 2]).tolist()
    channel_deltas = [
        abs(cm - bm)
        for cm, bm in zip(channel_means, baseline["channel_means"])
    ]

    return {
        "drifted":        drifted,
        "z_score":        round(z_score, 4),
        "img_mean":       round(img_mean, 6),
        "baseline_mean":  round(baseline_mean, 6),
        "sigma_threshold": sigma_threshold,
        "channel_deltas": [round(d, 6) for d in channel_deltas],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_summary(baseline: Dict) -> None:
    log.info(
        "  category=%s | n=%d | mean=%.4f | std=%.4f | drift_low=%.4f | drift_high=%.4f",
        baseline["category"],
        baseline["n_train_images"],
        baseline["global_mean"],
        baseline["global_std"],
        baseline["drift_low"],
        baseline["drift_high"],
    )
