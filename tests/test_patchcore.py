import pytest
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchCoreConstruction:

    def test_default_construction(self):
        from src.model.patchcore import PatchCore
        model = PatchCore(device="cpu")
        assert model.backbone_name == "wide_resnet50_2"
        assert model.layers == ["layer2", "layer3"]
        assert model.coreset_ratio == 0.1
        assert model.image_size == 224

    def test_memory_bank_starts_none(self):
        from src.model.patchcore import PatchCore
        model = PatchCore(device="cpu")
        assert model.memory_bank is None

    def test_hooks_registered_for_each_layer(self):
        """Forward hooks must be registered so _features is populated on forward pass."""
        from src.model.patchcore import PatchCore
        model = PatchCore(layers=["layer2", "layer3"], device="cpu")
        dummy = torch.randn(1, 3, 224, 224)
        model.feature_extractor(dummy)
        for layer_name in ["layer2", "layer3"]:
            assert layer_name in model._features, f"Hook not registered for {layer_name}"

    def test_cpu_fallback(self, monkeypatch):
        """When CUDA is not available, model must use CPU regardless of device arg."""
        import src.model.patchcore as pc_module
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        from src.model.patchcore import PatchCore
        model = PatchCore(device="cuda")
        assert str(model.device) == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchCorePredict:

    def test_predict_returns_float_score(self, tiny_patchcore, random_tensor):
        score, _ = tiny_patchcore.predict(random_tensor)
        assert isinstance(score, float)

    def test_predict_score_is_non_negative(self, tiny_patchcore, random_tensor):
        score, _ = tiny_patchcore.predict(random_tensor)
        assert score >= 0.0

    def test_predict_returns_numpy_heatmap(self, tiny_patchcore, random_tensor):
        _, heatmap = tiny_patchcore.predict(random_tensor)
        assert isinstance(heatmap, np.ndarray)

    def test_heatmap_shape_matches_image_size(self, tiny_patchcore, random_tensor):
        """Heatmap must be upsampled to image_size × image_size."""
        _, heatmap = tiny_patchcore.predict(random_tensor)
        sz = tiny_patchcore.image_size
        assert heatmap.shape == (sz, sz), f"Expected ({sz},{sz}), got {heatmap.shape}"

    def test_heatmap_values_in_zero_one(self, tiny_patchcore, random_tensor):
        """Heatmap is normalised to [0, 1]."""
        _, heatmap = tiny_patchcore.predict(random_tensor)
        assert heatmap.min() >= 0.0 - 1e-6
        assert heatmap.max() <= 1.0 + 1e-6

    def test_predict_accepts_3d_tensor(self, tiny_patchcore, random_tensor):
        """predict() should accept [3,H,W] as well as [1,3,H,W]."""
        score, heatmap = tiny_patchcore.predict(random_tensor)  # [3,224,224]
        assert isinstance(score, float)

    def test_predict_accepts_4d_tensor(self, tiny_patchcore, random_tensor):
        """predict() should accept [1,3,H,W]."""
        score, heatmap = tiny_patchcore.predict(random_tensor.unsqueeze(0))
        assert isinstance(score, float)

    def test_predict_raises_if_memory_bank_none(self, random_tensor):
        from src.model.patchcore import PatchCore
        model = PatchCore(device="cpu")
        with pytest.raises(RuntimeError, match="(?i)memory bank"):
            model.predict(random_tensor)


# ─────────────────────────────────────────────────────────────────────────────
# predict_batch()
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchCorePredictBatch:

    def test_batch_output_length_matches_input(self, tiny_patchcore, batch_tensor):
        scores, heatmaps = tiny_patchcore.predict_batch(batch_tensor)
        assert len(scores) == len(batch_tensor)
        assert len(heatmaps) == len(batch_tensor)

    def test_batch_scores_are_floats(self, tiny_patchcore, batch_tensor):
        scores, _ = tiny_patchcore.predict_batch(batch_tensor)
        assert all(isinstance(s, float) for s in scores)

    def test_batch_heatmaps_correct_shape(self, tiny_patchcore, batch_tensor):
        _, heatmaps = tiny_patchcore.predict_batch(batch_tensor)
        sz = tiny_patchcore.image_size
        for h in heatmaps:
            assert h.shape == (sz, sz)


# ─────────────────────────────────────────────────────────────────────────────
# save() / load()
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchCoreSaveLoad:

    def test_save_creates_file(self, tiny_patchcore, tmp_dir):
        path = str(tmp_dir / "patchcore_test.pt")
        tiny_patchcore.save(path)
        assert (tmp_dir / "patchcore_test.pt").exists()

    def test_load_restores_memory_bank(self, tiny_patchcore, tmp_dir):
        from src.model.patchcore import PatchCore
        path = str(tmp_dir / "patchcore_roundtrip.pt")
        tiny_patchcore.save(path)
        loaded = PatchCore.load(path, device="cpu")
        assert loaded.memory_bank is not None
        assert loaded.memory_bank.shape == tiny_patchcore.memory_bank.shape

    def test_load_restores_config(self, tiny_patchcore, tmp_dir):
        from src.model.patchcore import PatchCore
        path = str(tmp_dir / "patchcore_config.pt")
        tiny_patchcore.save(path)
        loaded = PatchCore.load(path, device="cpu")
        assert loaded.backbone_name == tiny_patchcore.backbone_name
        assert loaded.layers == tiny_patchcore.layers
        assert loaded.coreset_ratio == tiny_patchcore.coreset_ratio
        assert loaded.image_size == tiny_patchcore.image_size

    def test_loaded_model_can_predict(self, tiny_patchcore, tmp_dir, random_tensor):
        from src.model.patchcore import PatchCore
        path = str(tmp_dir / "patchcore_predict.pt")
        tiny_patchcore.save(path)
        loaded = PatchCore.load(path, device="cpu")
        score, heatmap = loaded.predict(random_tensor)
        assert isinstance(score, float)
        assert heatmap.shape == (loaded.image_size, loaded.image_size)
