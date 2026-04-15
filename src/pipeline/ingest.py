"""
src/pipeline/ingest.py
-----------------------
Raw data validation for the MVTec AD dataset.

Responsibilities:
  - Verify expected folder structure exists for each category
  - Count and log image files per split
  - Check image readability (not corrupt)
  - Enforce minimum image counts per split
  - Return per-category pass/fail dict consumed by Airflow DAG
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected MVTec AD directory schema
# ---------------------------------------------------------------------------
EXPECTED_TRAIN_SUBDIRS = ["good"]          # unsupervised — only normal images
EXPECTED_TEST_SUBDIRS  = ["good"]          # at minimum; defect dirs are optional
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

MIN_TRAIN_IMAGES = 50    # sanity lower-bound per category
MIN_TEST_IMAGES  = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_raw_data(
    raw_dir: str,
    categories: List[str],
) -> Dict[str, bool]:
    """
    Validate the raw MVTec AD dataset structure.

    Args:
        raw_dir:    Path to the root mvtec_ad directory.
        categories: List of category names to validate.

    Returns:
        Dict mapping category name → True (passed) / False (failed).
    """
    raw_path = Path(raw_dir)
    results: Dict[str, bool] = {}

    for category in categories:
        try:
            _validate_category(raw_path, category)
            log.info("[PASS] %s", category)
            results[category] = True
        except (ValueError, FileNotFoundError) as exc:
            log.error("[FAIL] %s — %s", category, exc)
            results[category] = False

    total = len(categories)
    passed = sum(results.values())
    log.info("Validation summary: %d/%d categories passed", passed, total)
    return results


def list_train_images(
    raw_dir: str,
    category: str,
) -> List[Path]:
    """Return sorted list of training (normal) image paths for a category."""
    train_good = Path(raw_dir) / category / "train" / "good"
    return sorted(_collect_images(train_good))


def list_test_images(
    raw_dir: str,
    category: str,
) -> Dict[str, List[Path]]:
    """
    Return dict of {defect_type: [image_paths]} for all test subdirectories.
    'good' key contains normal test images.
    """
    test_dir = Path(raw_dir) / category / "test"
    result: Dict[str, List[Path]] = {}
    for subdir in sorted(test_dir.iterdir()):
        if subdir.is_dir():
            images = sorted(_collect_images(subdir))
            if images:
                result[subdir.name] = images
    return result


def list_ground_truth_masks(
    raw_dir: str,
    category: str,
    defect_type: str,
) -> List[Path]:
    """Return sorted list of ground-truth mask paths for a defect type."""
    gt_dir = Path(raw_dir) / category / "ground_truth" / defect_type
    if not gt_dir.exists():
        return []
    return sorted(_collect_images(gt_dir))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_category(raw_path: Path, category: str) -> None:
    """Raise if this category fails any structural check."""
    cat_dir = raw_path / category
    _require_dir(cat_dir, f"Category root missing: {cat_dir}")

    # --- train/good ---
    train_good = cat_dir / "train" / "good"
    _require_dir(train_good, f"Missing train/good in {category}")
    train_images = list(_collect_images(train_good))
    if len(train_images) < MIN_TRAIN_IMAGES:
        raise ValueError(
            f"{category}/train/good has only {len(train_images)} images "
            f"(minimum {MIN_TRAIN_IMAGES})"
        )
    log.debug("%s: %d training images", category, len(train_images))

    # --- test/ ---
    test_dir = cat_dir / "test"
    _require_dir(test_dir, f"Missing test/ in {category}")
    test_good = test_dir / "good"
    _require_dir(test_good, f"Missing test/good in {category}")
    test_images = list(_collect_images(test_good))
    if len(test_images) < MIN_TEST_IMAGES:
        raise ValueError(
            f"{category}/test/good has only {len(test_images)} images "
            f"(minimum {MIN_TEST_IMAGES})"
        )

    # --- ground_truth/ ---
    gt_dir = cat_dir / "ground_truth"
    _require_dir(gt_dir, f"Missing ground_truth/ in {category}")

    # --- spot-check: first 3 images are readable ---
    for img_path in train_images[:3]:
        _check_readable(img_path)


def _require_dir(path: Path, msg: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(msg)


def _collect_images(directory: Path) -> List[Path]:
    """Yield all image files in a directory (non-recursive)."""
    return [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]


def _check_readable(path: Path) -> None:
    """Raise if PIL cannot open the image (corrupt / truncated)."""
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        raise ValueError(f"Corrupt image {path}: {exc}") from exc
