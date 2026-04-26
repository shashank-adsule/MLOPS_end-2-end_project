import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Basic image / tensor fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def random_rgb_image() -> Image.Image:
    """224×224 random RGB PIL image."""
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def random_tensor() -> torch.Tensor:
    """Preprocessed image tensor [3, 224, 224] with values ~N(0,1)."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def batch_tensor() -> torch.Tensor:
    """Batch of 4 preprocessed tensors [4, 3, 224, 224]."""
    return torch.randn(4, 3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# Directory fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Alias for pytest's built-in tmp_path."""
    return tmp_path


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    cat = "bottle"
    train_dir = tmp_path / cat / "train"
    good_dir   = tmp_path / cat / "test" / "good"
    defect_dir = tmp_path / cat / "test" / "broken_large"
    stats_dir  = tmp_path / cat / "stats"

    for d in [train_dir, good_dir, defect_dir, stats_dir]:
        d.mkdir(parents=True)

    for i in range(10):
        torch.save(torch.randn(3, 224, 224), train_dir / f"{i+1:04d}.pt")
    for i in range(5):
        torch.save(torch.randn(3, 224, 224), good_dir / f"{i+1:04d}.pt")
    for i in range(5):
        # Defect tensors have a large mean shift so AUROC is non-trivial
        torch.save(torch.randn(3, 224, 224) + 3.0, defect_dir / f"{i+1:04d}.pt")

    return tmp_path


@pytest.fixture
def raw_mvtec_dir(tmp_path: Path) -> Path:
    cat = "bottle"
    dirs = {
        "train_good":  tmp_path / cat / "train" / "good",
        "test_good":   tmp_path / cat / "test" / "good",
        "test_defect": tmp_path / cat / "test" / "broken_large",
        "gt":          tmp_path / cat / "ground_truth" / "broken_large",
    }
    for d in dirs.values():
        d.mkdir(parents=True)

    # Synthetic 64×64 PNG (PIL can open it; small to keep CI fast)
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    for i in range(55):
        img.save(dirs["train_good"] / f"{i:03d}.png")
    for i in range(15):
        img.save(dirs["test_good"] / f"{i:03d}.png")
    for i in range(10):
        img.save(dirs["test_defect"] / f"{i:03d}.png")

    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# Drift baseline fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def baseline_dict() -> Dict:
    """Pre-built baseline statistics dict — matches feature_engineering output schema."""
    return {
        "category":       "bottle",
        "n_train_images": 10,
        "channel_means":  [0.01, -0.02, 0.03],
        "channel_stds":   [0.98, 1.01, 0.99],
        "global_mean":    0.004,
        "global_std":     0.99,
        "global_min":     -3.5,
        "global_max":      3.5,
        "intensity_p5":   -0.80,
        "intensity_p95":   0.82,
        "drift_low":      -2.966,
        "drift_high":      2.974,
    }


@pytest.fixture
def baseline_file(tmp_path: Path, baseline_dict: Dict) -> Path:
    """Write baseline_dict to tmp_path/bottle/stats/baseline.json. Returns tmp_path."""
    stats_dir = tmp_path / "bottle" / "stats"
    stats_dir.mkdir(parents=True)
    with open(stats_dir / "baseline.json", "w") as f:
        json.dump(baseline_dict, f)
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# PatchCore fixture (session-scoped — backbone loads once per pytest run)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_patchcore():
    from src.model.patchcore import PatchCore
    model = PatchCore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_ratio=0.1,
        image_size=224,
        device="cpu",
    )
    model.memory_bank = torch.randn(50, 1536)
    return model
