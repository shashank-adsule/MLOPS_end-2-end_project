"""
tests/test_evaluate.py
-----------------------
Unit tests for src/model/evaluate.py

Covers:
  - _optimal_f1_threshold() returns float threshold and f1
  - _optimal_f1_threshold() picks the right threshold on a known case
  - evaluate_category() returns dict with required keys
  - evaluate_category() produces eval_metrics.json on disk
  - evaluate_category() produces roc_curve.csv on disk
  - evaluate_category() saves heatmap .npy files for defects
"""

import json
import pytest
import numpy as np
import torch
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# _optimal_f1_threshold()
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimalF1Threshold:

    def test_returns_two_floats(self):
        from src.model.evaluate import _optimal_f1_threshold
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        thresh, f1 = _optimal_f1_threshold(labels, scores)
        assert isinstance(thresh, float)
        assert isinstance(f1, float)

    def test_f1_in_zero_one(self):
        from src.model.evaluate import _optimal_f1_threshold
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        _, f1 = _optimal_f1_threshold(labels, scores)
        assert 0.0 <= f1 <= 1.0

    def test_perfect_separation_gives_f1_one(self):
        """When normal scores < 0.5 and defect scores > 0.5, F1 should be 1.0."""
        from src.model.evaluate import _optimal_f1_threshold
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        _, f1 = _optimal_f1_threshold(labels, scores)
        assert f1 == pytest.approx(1.0, abs=1e-4)

    def test_threshold_between_score_ranges(self):
        """Threshold must fall between the two score clusters."""
        from src.model.evaluate import _optimal_f1_threshold
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.3, 0.7, 0.8])
        thresh, _ = _optimal_f1_threshold(labels, scores)
        assert 0.3 <= thresh <= 0.7

    def test_all_same_label_returns_default(self):
        """Edge case: only one class present should not crash."""
        from src.model.evaluate import _optimal_f1_threshold
        labels = np.array([0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3])
        thresh, f1 = _optimal_f1_threshold(labels, scores)
        assert isinstance(thresh, float)
        assert isinstance(f1, float)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_category()
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateCategory:

    def test_returns_dict_with_required_keys(self, tiny_patchcore, processed_dir, tmp_dir):
        from src.model.evaluate import evaluate_category
        metrics = evaluate_category(
            model=tiny_patchcore,
            processed_dir=str(processed_dir),
            category="bottle",
            reports_dir=str(tmp_dir / "reports"),
            save_heatmaps=False,
        )
        for key in ["auroc", "f1_score", "pixel_auroc",
                    "avg_latency_ms", "threshold", "n_normal", "n_defect"]:
            assert key in metrics, f"Missing key: {key}"

    def test_auroc_in_zero_one(self, tiny_patchcore, processed_dir, tmp_dir):
        from src.model.evaluate import evaluate_category
        metrics = evaluate_category(
            model=tiny_patchcore,
            processed_dir=str(processed_dir),
            category="bottle",
            reports_dir=str(tmp_dir / "reports"),
            save_heatmaps=False,
        )
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_f1_score_in_zero_one(self, tiny_patchcore, processed_dir, tmp_dir):
        from src.model.evaluate import evaluate_category
        metrics = evaluate_category(
            model=tiny_patchcore,
            processed_dir=str(processed_dir),
            category="bottle",
            reports_dir=str(tmp_dir / "reports"),
            save_heatmaps=False,
        )
        assert 0.0 <= metrics["f1_score"] <= 1.0

    def test_latency_is_positive(self, tiny_patchcore, processed_dir, tmp_dir):
        from src.model.evaluate import evaluate_category
        metrics = evaluate_category(
            model=tiny_patchcore,
            processed_dir=str(processed_dir),
            category="bottle",
            reports_dir=str(tmp_dir / "reports"),
            save_heatmaps=False,
        )
        assert metrics["avg_latency_ms"] > 0.0

    def test_n_normal_and_n_defect_correct(self, tiny_patchcore, processed_dir, tmp_dir):
        """processed_dir fixture has 5 normal + 5 defect test images."""
        from src.model.evaluate import evaluate_category
        metrics = evaluate_category(
            model=tiny_patchcore,
            processed_dir=str(processed_dir),
            category="bottle",
            reports_dir=str(tmp_dir / "reports"),
            save_heatmaps=False,
        )
        assert metrics["n_normal"] == 5
        assert metrics["n_defect"] == 5

    def test_writes_eval_metrics_json(self, tiny_patchcore, processed_dir, tmp_dir):
        from src.model.evaluate import evaluate_category
        reports = tmp_dir / "reports"
        evaluate_category(
            model=tiny_patchcore,
            processed_dir=str(processed_dir),
            category="bottle",
            reports_dir=str(reports),
            save_heatmaps=False,
        )
        json_path = reports / "bottle" / "eval_metrics.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "auroc" in data

    def test_writes_roc_curve_csv(self, tiny_patchcore, processed_dir, tmp_dir):
        from src.model.evaluate import evaluate_category
        reports = tmp_dir / "reports"
        evaluate_category(
            model=tiny_patchcore,
            processed_dir=str(processed_dir),
            category="bottle",
            reports_dir=str(reports),
            save_heatmaps=False,
        )
        csv_path = reports / "bottle" / "roc_curve.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert content.startswith("fpr,tpr")

    def test_saves_heatmap_npy_files(self, tiny_patchcore, processed_dir, tmp_dir):
        from src.model.evaluate import evaluate_category
        reports = tmp_dir / "reports"
        evaluate_category(
            model=tiny_patchcore,
            processed_dir=str(processed_dir),
            category="bottle",
            reports_dir=str(reports),
            save_heatmaps=True,
            n_heatmap_samples=3,
        )
        heatmap_dir = reports / "bottle" / "heatmaps"
        npy_files = list(heatmap_dir.glob("*.npy"))
        # At most n_heatmap_samples=3 per defect type; we have 1 defect type with 5 images
        assert len(npy_files) <= 3
        assert len(npy_files) >= 1

    def test_heatmap_npy_loadable(self, tiny_patchcore, processed_dir, tmp_dir):
        from src.model.evaluate import evaluate_category
        reports = tmp_dir / "reports"
        evaluate_category(
            model=tiny_patchcore,
            processed_dir=str(processed_dir),
            category="bottle",
            reports_dir=str(reports),
            save_heatmaps=True,
            n_heatmap_samples=1,
        )
        npy_files = list((reports / "bottle" / "heatmaps").glob("*.npy"))
        arr = np.load(str(npy_files[0]))
        assert arr.ndim == 2   # heatmap is 2D
