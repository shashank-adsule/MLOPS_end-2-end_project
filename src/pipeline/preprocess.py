"""
src/pipeline/preprocess.py
---------------------------
Image preprocessing pipeline for MVTec AD dataset.

Responsibilities:
  - Resize images to a configurable size (default 256×256)
  - Center-crop to patch size (default 224×224) for backbone compatibility
  - Normalize with ImageNet mean/std (matches WideResNet50 pretraining)
  - Save processed tensors as .pt files for fast DataLoader loading
  - Return throughput statistics consumed by Airflow DAG

Output layout:
  data/processed/{category}/
      train/{0001.pt, 0002.pt, ...}   ← normal-only
      test/{good/0001.pt, broken_large/0001.pt, ...}
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet statistics — matches WideResNet50 / ResNet backbone pretraining
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_all_categories(
    raw_dir: str,
    processed_dir: str,
    categories: List[str],
    image_size: int = 256,
    patch_size: int = 224,
) -> Dict[str, Dict]:
    """
    Preprocess all categories and return per-category statistics.

    Args:
        raw_dir:       Root of raw MVTec AD data.
        processed_dir: Root output directory for processed tensors.
        categories:    List of category names to process.
        image_size:    Resize target before center crop.
        patch_size:    Center crop size (must match CNN input).

    Returns:
        Dict of {category: {train_count, test_count, elapsed_sec}}
    """
    transform = _build_transform(image_size, patch_size)
    stats: Dict[str, Dict] = {}

    for category in categories:
        log.info("Preprocessing category: %s", category)
        t0 = time.time()

        train_count = _preprocess_split(
            src_dir=Path(raw_dir) / category / "train" / "good",
            dst_dir=Path(processed_dir) / category / "train",
            transform=transform,
            split_label=f"{category}/train",
        )

        test_count = _preprocess_test_split(
            src_test_dir=Path(raw_dir) / category / "test",
            dst_dir=Path(processed_dir) / category / "test",
            transform=transform,
            category=category,
        )

        elapsed = round(time.time() - t0, 2)
        stats[category] = {
            "train_count": train_count,
            "test_count": test_count,
            "elapsed_sec": elapsed,
        }
        log.info(
            "%s done — train: %d, test: %d, time: %.1fs",
            category, train_count, test_count, elapsed,
        )

    return stats


def load_tensor(tensor_path: str) -> torch.Tensor:
    """Load a preprocessed image tensor from disk."""
    return torch.load(tensor_path, map_location="cpu")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_transform(image_size: int, patch_size: int) -> T.Compose:
    """
    Build the preprocessing transform pipeline.
    Resize → CenterCrop → ToTensor → Normalize (ImageNet stats).
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(patch_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _preprocess_split(
    src_dir: Path,
    dst_dir: Path,
    transform: T.Compose,
    split_label: str,
) -> int:
    """
    Preprocess all images in src_dir, save tensors to dst_dir.
    Returns count of processed images.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _collect_images(src_dir)

    if not image_paths:
        log.warning("No images found in %s", src_dir)
        return 0

    count = 0
    for img_path in tqdm(image_paths, desc=split_label, leave=False):
        tensor = _load_and_transform(img_path, transform)
        if tensor is None:
            continue
        # Zero-padded filename: 0001.pt, 0002.pt, ...
        out_path = dst_dir / f"{count + 1:04d}.pt"
        torch.save(tensor, out_path)
        count += 1

    log.debug("%s: saved %d tensors to %s", split_label, count, dst_dir)
    return count


def _preprocess_test_split(
    src_test_dir: Path,
    dst_dir: Path,
    transform: T.Compose,
    category: str,
) -> int:
    """
    Preprocess all test subdirectories (good, broken_large, etc.).
    Preserves subfolder structure under dst_dir.
    """
    total = 0
    for subdir in sorted(src_test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        count = _preprocess_split(
            src_dir=subdir,
            dst_dir=dst_dir / subdir.name,
            transform=transform,
            split_label=f"{category}/test/{subdir.name}",
        )
        total += count
    return total


def _load_and_transform(
    img_path: Path,
    transform: T.Compose,
) -> torch.Tensor:
    """
    Open image, convert to RGB, apply transform.
    Returns None on failure (logs warning instead of crashing pipeline).
    """
    try:
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
        return transform(rgb)
    except Exception as exc:
        log.warning("Skipping corrupt image %s: %s", img_path, exc)
        return None


def _collect_images(directory: Path) -> List[Path]:
    """Return sorted list of image files in a directory."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
