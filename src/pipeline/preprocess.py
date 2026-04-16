"""
src/pipeline/preprocess.py
---------------------------
Image preprocessing pipeline for MVTec AD dataset.

Responsibilities:
  - Resize images to a configurable size (default 256x256)
  - Center-crop to patch size (default 224x224) for backbone compatibility
  - Normalize with ImageNet mean/std (matches WideResNet50 pretraining)
  - Save processed tensors as .pt files for fast DataLoader loading
  - Return throughput statistics consumed by Airflow DAG

Output layout:
  data/processed/{category}/
      train/{0001.pt, 0002.pt, ...}   <- normal-only
      test/{good/0001.pt, broken_large/0001.pt, ...}

Usage:
  python -m src.pipeline.preprocess
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet statistics -- matches WideResNet50 / ResNet backbone pretraining
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
            "test_count":  test_count,
            "elapsed_sec": elapsed,
        }
        log.info(
            "%s done -- train: %d, test: %d, time: %.1fs",
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
    """Resize -> CenterCrop -> ToTensor -> Normalize (ImageNet stats)."""
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
    """Preprocess all images in src_dir, save tensors to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _collect_images(src_dir)

    if not image_paths:
        log.warning("No images found in %s", src_dir)
        return 0

    count = 0
    for img_path in tqdm(image_paths, desc=split_label, leave=True):
        tensor = _load_and_transform(img_path, transform)
        if tensor is None:
            continue
        out_path = dst_dir / f"{count + 1:04d}.pt"
        torch.save(tensor, out_path)
        count += 1

    return count


def _preprocess_test_split(
    src_test_dir: Path,
    dst_dir: Path,
    transform: T.Compose,
    category: str,
) -> int:
    """Preprocess all test subdirectories (good, broken_large, etc.)."""
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
    """Open image, convert to RGB, apply transform. Returns None on failure."""
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # Load params.yaml
    params_path = Path("params.yaml")
    if not params_path.exists():
        raise FileNotFoundError(
            "params.yaml not found. Make sure you run from the project root:\n"
            "  cd D:\\code\\repo\\MLOPS_end-2-end_project\n"
            "  python -m src.pipeline.preprocess"
        )

    with open(params_path) as f:
        params = yaml.safe_load(f)

    raw_dir       = os.environ.get("RAW_DATA_DIR",       params["data"]["raw_dir"])
    processed_dir = os.environ.get("PROCESSED_DATA_DIR", params["data"]["processed_dir"])
    categories    = params["data"]["categories"]
    image_size    = params["model"]["image_size"]
    patch_size    = params["model"]["patch_size"]

    log.info("=" * 55)
    log.info("MVTec AD Preprocessing Pipeline")
    log.info("=" * 55)
    log.info("Raw data dir   : %s", raw_dir)
    log.info("Processed dir  : %s", processed_dir)
    log.info("Categories     : %d total", len(categories))
    log.info("Image size     : %d -> %d (crop)", image_size, patch_size)
    log.info("=" * 55)

    stats = preprocess_all_categories(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        categories=categories,
        image_size=image_size,
        patch_size=patch_size,
    )

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'Category':<15} {'Train':>8} {'Test':>8} {'Time(s)':>10}")
    print("-" * 55)
    for cat, s in stats.items():
        print(
            f"{cat:<15} {s['train_count']:>8} "
            f"{s['test_count']:>8} {s['elapsed_sec']:>10.1f}"
        )
    print("=" * 55)
    print(f"\nProcessed data saved to: {processed_dir}")
    # print("Next step: python -m src.model.train --category bottle")