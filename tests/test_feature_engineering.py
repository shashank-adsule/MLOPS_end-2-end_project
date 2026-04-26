import json
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# compute_baseline_statistics()
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBaselineStatistics:

    def test_returns_dict_with_required_keys(self, processed_dir):
        from src.pipeline.feature_engineering import compute_baseline_statistics
        baseline = compute_baseline_statistics(str(processed_dir), "bottle")
        required = [
            "category", "n_train_images",
            "channel_means", "channel_stds",
            "global_mean", "global_std",
            "global_min", "global_max",
            "intensity_p5", "intensity_p95",
            "drift_low", "drift_high",
        ]
        for key in required:
            assert key in baseline, f"Missing key: {key}"

    def test_category_name_matches(self, processed_dir):
        from src.pipeline.feature_engineering import compute_baseline_statistics
        baseline = compute_baseline_statistics(str(processed_dir), "bottle")
        assert baseline["category"] == "bottle"

    def test_n_train_images_correct(self, processed_dir):
        from src.pipeline.feature_engineering import compute_baseline_statistics
        baseline = compute_baseline_statistics(str(processed_dir), "bottle")
        # processed_dir fixture creates 10 training tensors
        assert baseline["n_train_images"] == 10

    def test_channel_means_length_3(self, processed_dir):
        from src.pipeline.feature_engineering import compute_baseline_statistics
        baseline = compute_baseline_statistics(str(processed_dir), "bottle")
        assert len(baseline["channel_means"]) == 3
        assert len(baseline["channel_stds"]) == 3

    def test_drift_bounds_are_3_sigma(self, processed_dir):
        from src.pipeline.feature_engineering import compute_baseline_statistics
        baseline = compute_baseline_statistics(str(processed_dir), "bottle")
        expected_low  = baseline["global_mean"] - 3 * baseline["global_std"]
        expected_high = baseline["global_mean"] + 3 * baseline["global_std"]
        assert abs(baseline["drift_low"]  - expected_low)  < 1e-5
        assert abs(baseline["drift_high"] - expected_high) < 1e-5

    def test_baseline_json_written_to_disk(self, processed_dir):
        from src.pipeline.feature_engineering import compute_baseline_statistics
        compute_baseline_statistics(str(processed_dir), "bottle")
        json_path = processed_dir / "bottle" / "stats" / "baseline.json"
        assert json_path.exists()

    def test_baseline_json_is_valid_json(self, processed_dir):
        from src.pipeline.feature_engineering import compute_baseline_statistics
        compute_baseline_statistics(str(processed_dir), "bottle")
        json_path = processed_dir / "bottle" / "stats" / "baseline.json"
        with open(json_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_raises_if_no_tensors(self, tmp_dir):
        from src.pipeline.feature_engineering import compute_baseline_statistics
        # tmp_dir has no tensors at all
        with pytest.raises((FileNotFoundError, RuntimeError)):
            compute_baseline_statistics(str(tmp_dir), "bottle")


# ─────────────────────────────────────────────────────────────────────────────
# load_baseline()
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadBaseline:

    def test_loads_correctly(self, baseline_file, baseline_dict):
        from src.pipeline.feature_engineering import load_baseline
        loaded = load_baseline(str(baseline_file), "bottle")
        assert loaded["category"] == baseline_dict["category"]
        assert loaded["global_mean"] == pytest.approx(baseline_dict["global_mean"])

    def test_raises_if_file_missing(self, tmp_dir):
        from src.pipeline.feature_engineering import load_baseline
        with pytest.raises(FileNotFoundError):
            load_baseline(str(tmp_dir), "nonexistent_category")


# ─────────────────────────────────────────────────────────────────────────────
# check_drift()
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckDrift:

    def test_normal_image_not_flagged(self, baseline_dict, random_tensor):
        """A tensor drawn from ~N(0,1) should match baseline ~N(0.004, 0.99)."""
        from src.pipeline.feature_engineering import check_drift
        result = check_drift(random_tensor, baseline_dict, sigma_threshold=3.0)
        # With high sigma threshold and similar distribution, should not drift
        assert isinstance(result["drifted"], bool)
        assert isinstance(result["z_score"], float)
        assert result["z_score"] >= 0.0

    def test_strongly_shifted_image_flagged(self, baseline_dict):
        """A tensor with mean=100 vs baseline mean≈0 must trigger drift."""
        from src.pipeline.feature_engineering import check_drift
        shifted = torch.ones(3, 224, 224) * 100.0   # far from baseline mean≈0
        result = check_drift(shifted, baseline_dict, sigma_threshold=3.0)
        assert result["drifted"] is True
        assert result["z_score"] > 3.0

    def test_output_has_required_keys(self, baseline_dict, random_tensor):
        from src.pipeline.feature_engineering import check_drift
        result = check_drift(random_tensor, baseline_dict)
        for key in ["drifted", "z_score", "img_mean", "baseline_mean",
                    "sigma_threshold", "channel_deltas"]:
            assert key in result, f"Missing key: {key}"

    def test_channel_deltas_length_3(self, baseline_dict, random_tensor):
        from src.pipeline.feature_engineering import check_drift
        result = check_drift(random_tensor, baseline_dict)
        assert len(result["channel_deltas"]) == 3

    def test_z_score_increases_with_shift(self, baseline_dict):
        """Larger mean shift → larger z-score."""
        from src.pipeline.feature_engineering import check_drift
        t_small = torch.ones(3, 224, 224) * 1.0
        t_large = torch.ones(3, 224, 224) * 10.0
        z_small = check_drift(t_small, baseline_dict)["z_score"]
        z_large = check_drift(t_large, baseline_dict)["z_score"]
        assert z_large > z_small

    def test_sigma_threshold_respected(self, baseline_dict):
        """Same tensor, lower sigma_threshold should drift; higher should not."""
        from src.pipeline.feature_engineering import check_drift
        tensor = torch.ones(3, 224, 224) * 3.5  # z≈3.5 from baseline≈0
        result_strict = check_drift(tensor, baseline_dict, sigma_threshold=2.0)
        result_loose  = check_drift(tensor, baseline_dict, sigma_threshold=10.0)
        assert result_strict["drifted"] is True
        assert result_loose["drifted"]  is False

    def test_zero_std_baseline_does_not_crash(self, baseline_dict, random_tensor):
        """If global_std=0, z_score must return 0.0 (no division by zero)."""
        from src.pipeline.feature_engineering import check_drift
        zero_std_baseline = {**baseline_dict, "global_std": 0.0}
        result = check_drift(random_tensor, zero_std_baseline)
        assert result["z_score"] == 0.0
