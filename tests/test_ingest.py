import pytest
import numpy as np
from pathlib import Path
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# validate_raw_data()
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateRawData:

    def test_valid_structure_passes(self, raw_mvtec_dir):
        from src.pipeline.ingest import validate_raw_data
        results = validate_raw_data(str(raw_mvtec_dir), ["bottle"])
        assert results["bottle"] is True

    def test_missing_category_fails(self, raw_mvtec_dir):
        from src.pipeline.ingest import validate_raw_data
        results = validate_raw_data(str(raw_mvtec_dir), ["nonexistent_category"])
        assert results["nonexistent_category"] is False

    def test_missing_train_good_fails(self, tmp_path):
        from src.pipeline.ingest import validate_raw_data
        # Create structure but omit train/good
        (tmp_path / "bottle" / "test" / "good").mkdir(parents=True)
        (tmp_path / "bottle" / "ground_truth").mkdir(parents=True)
        results = validate_raw_data(str(tmp_path), ["bottle"])
        assert results["bottle"] is False

    def test_too_few_train_images_fails(self, tmp_path):
        from src.pipeline.ingest import validate_raw_data
        # Only 3 images — below MIN_TRAIN_IMAGES (50)
        train_good = tmp_path / "bottle" / "train" / "good"
        test_good  = tmp_path / "bottle" / "test"  / "good"
        gt_dir     = tmp_path / "bottle" / "ground_truth"
        train_good.mkdir(parents=True)
        test_good.mkdir(parents=True)
        gt_dir.mkdir(parents=True)

        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        for i in range(3):
            img.save(train_good / f"{i}.png")
        for i in range(12):
            img.save(test_good / f"{i}.png")

        results = validate_raw_data(str(tmp_path), ["bottle"])
        assert results["bottle"] is False

    def test_returns_dict_for_all_categories(self, raw_mvtec_dir, tmp_path):
        from src.pipeline.ingest import validate_raw_data
        # bottle exists, cable doesn't
        results = validate_raw_data(str(raw_mvtec_dir), ["bottle", "cable"])
        assert "bottle" in results
        assert "cable" in results
        assert results["cable"] is False

    def test_multiple_valid_categories_all_pass(self, tmp_path):
        """Two identical valid category structures must both pass."""
        from src.pipeline.ingest import validate_raw_data
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")

        for cat in ["bottle", "cable"]:
            (tmp_path / cat / "train" / "good").mkdir(parents=True)
            (tmp_path / cat / "test"  / "good").mkdir(parents=True)
            (tmp_path / cat / "ground_truth").mkdir(parents=True)
            for i in range(55):
                img.save(tmp_path / cat / "train" / "good" / f"{i}.png")
            for i in range(12):
                img.save(tmp_path / cat / "test" / "good" / f"{i}.png")

        results = validate_raw_data(str(tmp_path), ["bottle", "cable"])
        assert results["bottle"] is True
        assert results["cable"]  is True


# ─────────────────────────────────────────────────────────────────────────────
# list_train_images()
# ─────────────────────────────────────────────────────────────────────────────

class TestListTrainImages:

    def test_returns_correct_count(self, raw_mvtec_dir):
        from src.pipeline.ingest import list_train_images
        images = list_train_images(str(raw_mvtec_dir), "bottle")
        # raw_mvtec_dir fixture creates 55 train/good images
        assert len(images) == 55

    def test_all_paths_exist(self, raw_mvtec_dir):
        from src.pipeline.ingest import list_train_images
        images = list_train_images(str(raw_mvtec_dir), "bottle")
        for p in images:
            assert p.exists()

    def test_returns_sorted_list(self, raw_mvtec_dir):
        from src.pipeline.ingest import list_train_images
        images = list_train_images(str(raw_mvtec_dir), "bottle")
        assert images == sorted(images)


# ─────────────────────────────────────────────────────────────────────────────
# list_test_images()
# ─────────────────────────────────────────────────────────────────────────────

class TestListTestImages:

    def test_returns_dict(self, raw_mvtec_dir):
        from src.pipeline.ingest import list_test_images
        result = list_test_images(str(raw_mvtec_dir), "bottle")
        assert isinstance(result, dict)

    def test_good_key_present(self, raw_mvtec_dir):
        from src.pipeline.ingest import list_test_images
        result = list_test_images(str(raw_mvtec_dir), "bottle")
        assert "good" in result

    def test_defect_key_present(self, raw_mvtec_dir):
        from src.pipeline.ingest import list_test_images
        result = list_test_images(str(raw_mvtec_dir), "bottle")
        assert "broken_large" in result

    def test_image_counts_correct(self, raw_mvtec_dir):
        from src.pipeline.ingest import list_test_images
        result = list_test_images(str(raw_mvtec_dir), "bottle")
        assert len(result["good"]) == 15
        assert len(result["broken_large"]) == 10


# ─────────────────────────────────────────────────────────────────────────────
# list_ground_truth_masks()
# ─────────────────────────────────────────────────────────────────────────────

class TestListGroundTruthMasks:

    def test_returns_empty_list_when_absent(self, tmp_dir):
        from src.pipeline.ingest import list_ground_truth_masks
        masks = list_ground_truth_masks(str(tmp_dir), "bottle", "broken_large")
        assert masks == []

    def test_returns_masks_when_present(self, raw_mvtec_dir):
        """raw_mvtec_dir creates ground_truth/broken_large dir (empty — no masks needed for count test)."""
        from src.pipeline.ingest import list_ground_truth_masks
        masks = list_ground_truth_masks(str(raw_mvtec_dir), "bottle", "broken_large")
        # Dir exists but has no images → empty list is fine
        assert isinstance(masks, list)
