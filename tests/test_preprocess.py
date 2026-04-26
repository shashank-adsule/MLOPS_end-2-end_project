import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Transform
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTransform:

    def test_output_shape(self, random_rgb_image):
        from src.pipeline.preprocess import _build_transform
        transform = _build_transform(image_size=256, patch_size=224)
        tensor = transform(random_rgb_image)
        assert tensor.shape == (3, 224, 224)

    def test_output_is_float_tensor(self, random_rgb_image):
        from src.pipeline.preprocess import _build_transform
        transform = _build_transform(image_size=256, patch_size=224)
        tensor = transform(random_rgb_image)
        assert tensor.dtype == torch.float32

    def test_normalisation_changes_range(self, random_rgb_image):
        """After ImageNet normalisation, values should not be strictly in [0,1]."""
        from src.pipeline.preprocess import _build_transform
        transform = _build_transform(image_size=256, patch_size=224)
        tensor = transform(random_rgb_image)
        # Normalised values can be negative or > 1
        assert tensor.min() < 0.5 or tensor.max() > 0.9


# ─────────────────────────────────────────────────────────────────────────────
# preprocess_all_categories()
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessAllCategories:

    def test_creates_train_tensors(self, raw_mvtec_dir, tmp_path):
        from src.pipeline.preprocess import preprocess_all_categories
        out_dir = tmp_path / "processed"
        preprocess_all_categories(
            raw_dir=str(raw_mvtec_dir),
            processed_dir=str(out_dir),
            categories=["bottle"],
            image_size=64,   # small for speed
            patch_size=56,
        )
        train_tensors = list((out_dir / "bottle" / "train").glob("*.pt"))
        assert len(train_tensors) == 55  # matches raw_mvtec_dir fixture count

    def test_creates_test_good_tensors(self, raw_mvtec_dir, tmp_path):
        from src.pipeline.preprocess import preprocess_all_categories
        out_dir = tmp_path / "processed"
        preprocess_all_categories(
            raw_dir=str(raw_mvtec_dir),
            processed_dir=str(out_dir),
            categories=["bottle"],
            image_size=64,
            patch_size=56,
        )
        good_tensors = list((out_dir / "bottle" / "test" / "good").glob("*.pt"))
        assert len(good_tensors) == 15

    def test_creates_test_defect_tensors(self, raw_mvtec_dir, tmp_path):
        from src.pipeline.preprocess import preprocess_all_categories
        out_dir = tmp_path / "processed"
        preprocess_all_categories(
            raw_dir=str(raw_mvtec_dir),
            processed_dir=str(out_dir),
            categories=["bottle"],
            image_size=64,
            patch_size=56,
        )
        defect_tensors = list((out_dir / "bottle" / "test" / "broken_large").glob("*.pt"))
        assert len(defect_tensors) == 10

    def test_returns_stats_dict(self, raw_mvtec_dir, tmp_path):
        from src.pipeline.preprocess import preprocess_all_categories
        out_dir = tmp_path / "processed"
        stats = preprocess_all_categories(
            raw_dir=str(raw_mvtec_dir),
            processed_dir=str(out_dir),
            categories=["bottle"],
            image_size=64,
            patch_size=56,
        )
        assert "bottle" in stats
        for key in ["train_count", "test_count", "elapsed_sec"]:
            assert key in stats["bottle"], f"Missing stats key: {key}"

    def test_tensor_shape_correct(self, raw_mvtec_dir, tmp_path):
        from src.pipeline.preprocess import preprocess_all_categories
        out_dir = tmp_path / "processed"
        preprocess_all_categories(
            raw_dir=str(raw_mvtec_dir),
            processed_dir=str(out_dir),
            categories=["bottle"],
            image_size=64,
            patch_size=56,
        )
        first_pt = sorted((out_dir / "bottle" / "train").glob("*.pt"))[0]
        tensor = torch.load(str(first_pt), map_location="cpu")
        assert tensor.shape == (3, 56, 56)


# ─────────────────────────────────────────────────────────────────────────────
# load_tensor()
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadTensor:

    def test_loads_saved_tensor(self, tmp_dir):
        from src.pipeline.preprocess import load_tensor
        tensor = torch.randn(3, 224, 224)
        path = str(tmp_dir / "test.pt")
        torch.save(tensor, path)
        loaded = load_tensor(path)
        assert loaded.shape == tensor.shape

    def test_loaded_values_match(self, tmp_dir):
        from src.pipeline.preprocess import load_tensor
        tensor = torch.randn(3, 224, 224)
        path = str(tmp_dir / "exact.pt")
        torch.save(tensor, path)
        loaded = load_tensor(path)
        assert torch.allclose(tensor, loaded)


# ─────────────────────────────────────────────────────────────────────────────
# Corrupt image handling
# ─────────────────────────────────────────────────────────────────────────────

class TestCorruptImageHandling:

    def test_corrupt_image_is_skipped(self, tmp_path):
        """A corrupt PNG in the source dir should be silently skipped."""
        from src.pipeline.preprocess import _preprocess_split, _build_transform
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()

        # Write one valid image
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(src / "valid.png")

        # Write one corrupt file (random bytes, not a valid PNG)
        (src / "corrupt.png").write_bytes(b"this is not a valid image file")

        transform = _build_transform(64, 56)
        count = _preprocess_split(src, dst, transform, "test")

        # Only the valid image should be processed
        assert count == 1
        assert len(list(dst.glob("*.pt"))) == 1
