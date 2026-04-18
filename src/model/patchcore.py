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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.ndimage import gaussian_filter
from torchvision.models import Wide_ResNet50_2_Weights

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

        # Memory bank populated during fit()
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
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(self.device)
                patches = self._extract_patches(images)   # [N, D]
                all_patches.append(patches.cpu())
                if (i + 1) % 5 == 0 or (i + 1) == len(dataloader):
                    print(f"         Batch {i+1}/{len(dataloader)} processed...",
                          flush=True)

        memory_bank = torch.cat(all_patches, dim=0)       # [Total, D]
        print(f"         Raw patches    : {len(memory_bank)}", flush=True)

        # Coreset subsampling
        self.memory_bank = self._coreset_subsample(memory_bank)
        print(f"         After coreset  : {len(self.memory_bank)}", flush=True)

    def predict(
        self,
        image: torch.Tensor,
    ) -> Tuple[float, np.ndarray]:
        """
        Predict anomaly score and heatmap for a single image.

        Args:
            image: Preprocessed image tensor [1, 3, H, W] or [3, H, W]

        Returns:
            anomaly_score: float
            heatmap:       np.ndarray [H, W] in [0, 1]
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty. Call fit() first.")

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            patch_features = self._extract_patches(image)   # [P, D]

        distances = self._nearest_neighbor_distances(
            patch_features.cpu(),
            self.memory_bank,
        )

        anomaly_score = float(distances.max())
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
            "memory_bank":   self.memory_bank,
            "backbone_name": self.backbone_name,
            "layers":        self.layers,
            "coreset_ratio": self.coreset_ratio,
            "image_size":    self.image_size,
        }, path)
        log.info("PatchCore saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PatchCore":
        """Load a saved PatchCore model from disk."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            backbone=checkpoint["backbone_name"],
            layers=checkpoint["layers"],
            coreset_ratio=checkpoint["coreset_ratio"],
            image_size=checkpoint["image_size"],
            device=device,
        )
        model.memory_bank = checkpoint["memory_bank"]
        log.info(
            "PatchCore loaded from %s — %d patches in memory bank",
            path, len(model.memory_bank),
        )
        return model

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _build_feature_extractor(self) -> nn.Module:
        """Load pretrained WideResNet50, freeze all weights."""
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = models.wide_resnet50_2(weights=weights)
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

    def _extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass → extract + aggregate multi-scale features.
        Returns: [N*H*W, D] patch feature vectors
        """
        self._features.clear()
        _ = self.feature_extractor(images)

        feature_maps = []
        target_size = None

        for layer_name in self.layers:
            feat = self._features[layer_name]
            if target_size is None:
                target_size = feat.shape[-2:]
            else:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            feature_maps.append(feat)

        combined = torch.cat(feature_maps, dim=1)
        combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)

        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)
        patches = F.normalize(patches, p=2, dim=1)

        return patches

    # ------------------------------------------------------------------
    # Coreset subsampling — CPU-safe random projection approximation
    # ------------------------------------------------------------------

    def _coreset_subsample(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Fast coreset subsampling using random projection + mini-batch
        k-means approximation.

        This avoids the full N×N distance matrix that causes OOM/hangs
        on GPUs with limited VRAM (e.g. RTX 3050 6GB).
        """
        n_total = len(patches)
        n_keep  = max(1, int(n_total * self.coreset_ratio))

        if n_keep >= n_total:
            return patches

        print(f"         Coreset: {n_total} → {n_keep} patches...", flush=True)

        # Move to CPU for coreset (avoids VRAM pressure)
        patches_cpu = patches.cpu().float()

        # Use random projection to reduce dimensionality first
        # This makes distance computations much faster
        D = patches_cpu.shape[1]
        proj_dim = min(128, D)
        torch.manual_seed(42)
        projection = torch.randn(D, proj_dim) / (proj_dim ** 0.5)
        projected = patches_cpu @ projection           # [N, 128]

        selected_indices = self._greedy_coreset_cpu(projected, n_keep)
        result = patches_cpu[selected_indices]

        print(f"         Coreset done.", flush=True)
        return result

    def _greedy_coreset_cpu(
        self,
        patches: torch.Tensor,
        n_keep: int,
        chunk_size: int = 1000,
    ) -> torch.Tensor:
        """
        Greedy farthest-point sampling on CPU using chunked distance
        computation — avoids building the full N×N matrix at once.

        Args:
            patches:    [N, D] projected patch features on CPU
            n_keep:     number of patches to select
            chunk_size: process distances in chunks to limit RAM usage

        Returns:
            selected_indices: [n_keep] LongTensor
        """
        n_total = len(patches)
        selected = []

        # Random start
        idx = torch.randint(0, n_total, (1,)).item()
        selected.append(idx)

        # min_distances[i] = distance from patch i to nearest selected patch
        min_distances = torch.full((n_total,), float("inf"))

        for step in range(n_keep - 1):
            last = patches[selected[-1]].unsqueeze(0)   # [1, D]

            # Update min_distances in chunks to avoid OOM
            for start in range(0, n_total, chunk_size):
                end   = min(start + chunk_size, n_total)
                chunk = patches[start:end]              # [chunk, D]
                dists = torch.cdist(chunk, last).squeeze(1)  # [chunk]
                min_distances[start:end] = torch.minimum(
                    min_distances[start:end], dists
                )

            # Pick farthest point
            idx = int(min_distances.argmax().item())
            selected.append(idx)

            # Progress every 20%
            if (step + 1) % max(1, n_keep // 5) == 0:
                pct = (step + 1) / n_keep * 100
                print(f"         Coreset progress: {pct:.0f}%", flush=True)

        return torch.tensor(selected, dtype=torch.long)

    # ------------------------------------------------------------------
    # Nearest-neighbor distance
    # ------------------------------------------------------------------

    def _nearest_neighbor_distances(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        Distance from each query patch to its nearest neighbour in memory.
        Batched to avoid OOM.
        """
        memory = memory.to(self.device)
        distances = []

        for i in range(0, len(query), batch_size):
            q_batch = query[i : i + batch_size].to(self.device)
            dists = torch.cdist(q_batch, memory)
            nn_dists, _ = dists.min(dim=1)
            distances.append(nn_dists.cpu())

        return torch.cat(distances, dim=0)

    # ------------------------------------------------------------------
    # Heatmap generation
    # ------------------------------------------------------------------

    def _build_heatmap(self, patch_distances: torch.Tensor) -> np.ndarray:
        """Reshape patch distances into a smoothed spatial heatmap."""
        n_patches = len(patch_distances)
        grid_size = int(n_patches ** 0.5)

        scores_2d = patch_distances.numpy().reshape(grid_size, grid_size)

        heatmap = np.array(
            torch.nn.functional.interpolate(
                torch.tensor(scores_2d).unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
        )

        heatmap = gaussian_filter(heatmap, sigma=4)

        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)

        return heatmap.astype(np.float32)