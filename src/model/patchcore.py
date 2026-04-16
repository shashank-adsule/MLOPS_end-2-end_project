"""
src/model/patchcore.py
-----------------------
PatchCore anomaly detection algorithm.

Algorithm summary:
  1. Extract patch-level features from a pretrained WideResNet50 backbone
     at layers layer2 and layer3 (multi-scale)
  2. Aggregate features via adaptive average pooling into a flat memory bank
  3. Apply greedy coreset subsampling to keep only the most representative
     patches (reduces memory + inference time)
  4. At inference: compute nearest-neighbor distance from each patch to the
     memory bank — high distance = anomaly
  5. Produce pixel-level anomaly heatmap via Gaussian smoothing

Reference: Roth et al. "Towards Total Recall in Industrial Anomaly Detection"
           https://arxiv.org/abs/2106.08265
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import Wide_ResNet50_2_Weights
from scipy.ndimage import gaussian_filter
import numpy as np

log = logging.getLogger(__name__)


class PatchCore(nn.Module):
    """
    PatchCore anomaly detector.

    Args:
        backbone:      torchvision model name (default: wide_resnet50_2)
        layers:        list of layer names to extract features from
        coreset_ratio: fraction of patches to keep in memory bank (0.0-1.0)
        image_size:    input image size (used for heatmap upsampling)
        device:        torch device string ('cuda' or 'cpu')
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] = ["layer2", "layer3"],
        coreset_ratio: float = 0.1,
        image_size: int = 224,
        device: str = "cuda",
    ):
        super().__init__()

        self.backbone_name = backbone
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if str(self.device) == "cuda":
            log.info("PatchCore using GPU: %s", torch.cuda.get_device_name(0))
        else:
            log.info("PatchCore using CPU")

        # Memory bank — populated during fit()
        self.memory_bank: torch.Tensor = None

        # Build feature extractor
        self.feature_extractor = self._build_feature_extractor()
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)

        # Hook storage
        self._features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Build the memory bank from normal training images.

        Args:
            dataloader: DataLoader yielding preprocessed normal image tensors
        """
        log.info("Building PatchCore memory bank...")
        all_patches = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(self.device)
                patches = self._extract_patches(images)   # [N, D]
                all_patches.append(patches.cpu())

        memory_bank = torch.cat(all_patches, dim=0)       # [Total, D]
        log.info("Raw memory bank: %d patches, dim=%d", *memory_bank.shape)

        # Coreset subsampling
        self.memory_bank = self._coreset_subsample(memory_bank)
        log.info(
            "After coreset subsampling (ratio=%.2f): %d patches",
            self.coreset_ratio,
            len(self.memory_bank),
        )

    def predict(
        self,
        image: torch.Tensor,
    ) -> Tuple[float, np.ndarray]:
        """
        Predict anomaly score and heatmap for a single image.

        Args:
            image: Preprocessed image tensor [1, 3, H, W] or [3, H, W]

        Returns:
            anomaly_score: float — max patch distance (higher = more anomalous)
            heatmap:       np.ndarray [H, W] — pixel-level anomaly map [0, 1]
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty. Call fit() first.")

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            patch_features = self._extract_patches(image)   # [P, D]

        # Nearest-neighbor distances to memory bank
        distances = self._nearest_neighbor_distances(
            patch_features.cpu(),
            self.memory_bank,
        )   # [P]

        # Anomaly score = max patch distance
        anomaly_score = float(distances.max())

        # Build spatial heatmap
        heatmap = self._build_heatmap(distances)

        return anomaly_score, heatmap

    def predict_batch(
        self,
        images: torch.Tensor,
    ) -> Tuple[List[float], List[np.ndarray]]:
        """Predict anomaly scores and heatmaps for a batch of images."""
        scores, heatmaps = [], []
        for i in range(len(images)):
            score, heatmap = self.predict(images[i])
            scores.append(score)
            heatmaps.append(heatmap)
        return scores, heatmaps

    def save(self, path: str) -> None:
        """Save memory bank and config to disk."""
        torch.save({
            "memory_bank":    self.memory_bank,
            "backbone_name":  self.backbone_name,
            "layers":         self.layers,
            "coreset_ratio":  self.coreset_ratio,
            "image_size":     self.image_size,
        }, path)
        log.info("PatchCore saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PatchCore":
        """Load a saved PatchCore model from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            backbone=checkpoint["backbone_name"],
            layers=checkpoint["layers"],
            coreset_ratio=checkpoint["coreset_ratio"],
            image_size=checkpoint["image_size"],
            device=device,
        )
        model.memory_bank = checkpoint["memory_bank"]
        log.info("PatchCore loaded from %s — %d patches in memory bank",
                 path, len(model.memory_bank))
        return model

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _build_feature_extractor(self) -> nn.Module:
        """Load pretrained WideResNet50, freeze all weights."""
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = models.wide_resnet50_2(weights=weights)
        # Freeze all parameters — we only use it as a feature extractor
        for param in backbone.parameters():
            param.requires_grad = False
        return backbone

    def _register_hooks(self) -> None:
        """Register forward hooks on the specified layers."""
        def make_hook(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook

        for name, module in self.feature_extractor.named_modules():
            if name in self.layers:
                module.register_forward_hook(make_hook(name))
                log.debug("Registered hook on layer: %s", name)

    def _extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone, extract + aggregate multi-scale features.

        Returns:
            patches: [N*H*W, D] flattened patch feature vectors
        """
        self._features.clear()
        _ = self.feature_extractor(images)

        # Collect features from all registered layers
        feature_maps = []
        target_size = None

        for layer_name in self.layers:
            feat = self._features[layer_name]   # [B, C, H, W]
            if target_size is None:
                target_size = feat.shape[-2:]    # use first layer's spatial size
            else:
                # Upsample to match first layer's spatial resolution
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            feature_maps.append(feat)

        # Concatenate along channel dimension → [B, C_total, H, W]
        combined = torch.cat(feature_maps, dim=1)

        # Adaptive average pooling per patch (neighbourhood aggregation)
        combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)

        # Reshape to [B*H*W, C_total]
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # L2 normalize patch vectors
        patches = F.normalize(patches, p=2, dim=1)

        return patches

    # ------------------------------------------------------------------
    # Coreset subsampling (greedy farthest-point sampling)
    # ------------------------------------------------------------------

    def _coreset_subsample(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Greedy coreset subsampling — keeps patches that maximally cover
        the feature space, discarding near-duplicates.

        Uses an approximation for speed: random initial selection then
        iterative farthest-point sampling.
        """
        n_total = len(patches)
        n_keep  = max(1, int(n_total * self.coreset_ratio))

        if n_keep >= n_total:
            return patches

        log.info("Coreset: selecting %d from %d patches...", n_keep, n_total)

        # Move to GPU for speed if available
        patches_gpu = patches.to(self.device)

        selected_indices = []

        # Random start
        idx = torch.randint(0, n_total, (1,)).item()
        selected_indices.append(idx)

        # Track min distance from each point to the selected set
        min_distances = torch.full((n_total,), float("inf"), device=self.device)

        for _ in range(n_keep - 1):
            # Distance from last selected point to all others
            last = patches_gpu[selected_indices[-1]].unsqueeze(0)   # [1, D]
            dists = torch.cdist(patches_gpu, last).squeeze(1)        # [N]
            min_distances = torch.minimum(min_distances, dists)

            # Select the farthest point
            idx = int(min_distances.argmax().item())
            selected_indices.append(idx)

        selected = patches_gpu[selected_indices].cpu()
        log.info("Coreset selection complete.")
        return selected

    # ------------------------------------------------------------------
    # Nearest-neighbor distance
    # ------------------------------------------------------------------

    def _nearest_neighbor_distances(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """
        Compute distance from each query patch to its nearest neighbour
        in the memory bank. Batched to avoid OOM.
        """
        memory = memory.to(self.device)
        distances = []

        for i in range(0, len(query), batch_size):
            q_batch = query[i : i + batch_size].to(self.device)
            # [Q, M] pairwise L2 distances
            dists = torch.cdist(q_batch, memory)
            nn_dists, _ = dists.min(dim=1)   # [Q]
            distances.append(nn_dists.cpu())

        return torch.cat(distances, dim=0)   # [P]

    # ------------------------------------------------------------------
    # Heatmap generation
    # ------------------------------------------------------------------

    def _build_heatmap(self, patch_distances: torch.Tensor) -> np.ndarray:
        """
        Reshape flat patch distances into a spatial heatmap and upsample
        to the original image size with Gaussian smoothing.
        """
        # Infer spatial grid size from number of patches
        n_patches = len(patch_distances)
        grid_size = int(n_patches ** 0.5)

        scores_2d = patch_distances.numpy().reshape(grid_size, grid_size)

        # Upsample to image size
        heatmap = np.array(
            torch.nn.functional.interpolate(
                torch.tensor(scores_2d).unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
        )

        # Gaussian smoothing for cleaner localization
        heatmap = gaussian_filter(heatmap, sigma=4)

        # Normalize to [0, 1]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)

        return heatmap.astype(np.float32)