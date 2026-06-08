"""PatchCore anomaly detector.

Implements "Towards Total Recall in Industrial Anomaly Detection"
(Roth et al., CVPR 2022).

The core idea: extract patch-level embeddings from a frozen pretrained backbone
for all normal training images, subsample them into a compact coreset memory
bank, then score test images by the nearest-neighbour distance of each patch
embedding to that bank.  No gradient updates anywhere.

Architecture summary
--------------------
    fit()
        ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
        │  Train images│────▶│ WideResNet-50  │────▶│  All patches  │
        │  (normal)    │     │ layer2 + layer3│     │  (N × 1536-d) │
        └──────────────┘     └────────────────┘     └──────┬────────┘
                                                           │ greedy coreset
                                                    ┌──────▼────────┐
                                                    │ Memory bank   │
                                                    │ FAISS IndexL2 │
                                                    └───────────────┘
    predict()
        ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
        │  Test image  │────▶│ WideResNet-50  │────▶│ Patch scores  │
        └──────────────┘     └────────────────┘     │ (k-NN dist)   │
                                                    └──────┬────────┘
                                                           │ upsample + smooth
                                                    ┌──────▼────────┐
                                                    │ Anomaly map   │
                                                    │ (H × W)       │
                                                    └───────────────┘

Reference
---------
Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P.
"Towards Total Recall in Industrial Anomaly Detection."
IEEE/CVF CVPR, 2022.  https://arxiv.org/abs/2106.08265
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from scipy.ndimage import gaussian_filter
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Feature extraction ────────────────────────────────────────────────────────

class _FeatureExtractor(nn.Module):
    """Wraps a pretrained backbone and captures intermediate feature maps.

    Forward hooks are registered on the named layers in *layers*.  A forward
    pass through the full backbone populates ``self._features``; only the
    hooked layers' outputs are retained (no gradients, backbone frozen).

    Args:
        backbone_name: Any ``torchvision.models`` callable name.
        layers: Names of submodules to hook (as returned by
            ``model.named_modules()``).  E.g. ``['layer2', 'layer3']``.
    """

    def __init__(self, backbone_name: str, layers: list[str]) -> None:
        super().__init__()
        weights_enum = f"{backbone_name.upper()}_Weights" if False else None
        self.backbone: nn.Module = getattr(tvm, backbone_name)(
            weights="IMAGENET1K_V1"
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        self.layers = layers
        self._features: dict[str, Tensor] = {}
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.backbone.named_modules():
            if name in self.layers:
                handle = module.register_forward_hook(
                    lambda _m, _i, output, n=name: self._features.__setitem__(n, output)
                )
                self._hooks.append(handle)

    @torch.no_grad()
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        self._features.clear()
        self.backbone(x)
        # Return features in the order specified by self.layers
        return {name: self._features[name] for name in self.layers}

    def remove_hooks(self) -> None:
        """Clean up forward hooks (call when done to avoid memory leaks)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _pool_and_concat(
    feature_maps: dict[str, Tensor],
    target_size: int,
) -> Tensor:
    """Merge multi-scale feature maps into a flat patch-feature matrix.

    Each feature map is bilinearly resized to *target_size* × *target_size*,
    then all maps are concatenated along the channel axis.  The result is
    reshaped from ``(B, C_total, H, W)`` to ``(B × H × W, C_total)`` so that
    each row represents one spatial patch embedding.

    Args:
        feature_maps: Ordered dict of ``{layer_name: (B, C, H, W)}``.
        target_size: Target spatial resolution for all maps.

    Returns:
        ``(B × target_size², C_total)`` float32 tensor.
    """
    resized: list[Tensor] = []
    for feat in feature_maps.values():
        if feat.shape[-1] != target_size or feat.shape[-2] != target_size:
            feat = F.interpolate(
                feat,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        resized.append(feat)

    combined = torch.cat(resized, dim=1)          # (B, C_total, H, W)
    B, C, H, W = combined.shape
    patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)
    return patches.contiguous()


# ── Coreset subsampling ───────────────────────────────────────────────────────

def _greedy_coreset(
    features: np.ndarray,
    ratio: float,
    min_size: int = 300,
    max_candidates: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Greedy furthest-point coreset subsampling.

    Selects a subset of patch embeddings that maximally covers the feature
    space — i.e., minimises the maximum distance from any feature to its
    nearest coreset representative.  This is the coreset construction used
    in the original PatchCore paper.

    For efficiency, when the total patch count exceeds *max_candidates*, an
    initial random subsample is drawn first.  This follows standard practice
    in PatchCore implementations and has negligible impact on downstream AUROC.

    Args:
        features: ``(N, D)`` float32 array of patch embeddings.
        ratio: Target coreset size as a fraction of *N*.  E.g. ``0.01`` = 1%.
        min_size: Coreset will be at least this many points regardless of ratio.
        max_candidates: Random pre-subsample threshold (see above).
        seed: RNG seed for reproducibility.

    Returns:
        ``(M, D)`` float32 array where ``M ≈ max(ratio × N, min_size)``.
    """
    rng = np.random.default_rng(seed)
    N = len(features)
    target_size = max(int(ratio * N), min_size)

    # Pre-subsample to keep greedy loop tractable
    if N > max_candidates:
        idx = rng.choice(N, size=max_candidates, replace=False)
        candidates = features[idx].copy()
    else:
        candidates = features.copy()

    M = len(candidates)
    target_size = min(target_size, M)

    # ── Greedy loop ───────────────────────────────────────────────────────────
    # min_distances[i] = squared L2 distance from candidates[i] to its
    # nearest already-selected coreset point.
    min_distances = np.full(M, np.inf, dtype=np.float64)
    selected_idx: list[int] = [int(rng.integers(M))]  # random seed point

    for _ in tqdm(range(target_size - 1), desc="  Coreset subsampling", leave=False):
        last = candidates[selected_idx[-1]]                    # (D,)
        dists = np.sum((candidates - last) ** 2, axis=1)      # (M,) squared L2
        np.minimum(min_distances, dists, out=min_distances)
        selected_idx.append(int(np.argmax(min_distances)))

    return candidates[selected_idx].astype(np.float32)


# ── Anomaly map post-processing ───────────────────────────────────────────────

def _smooth_and_upsample(
    patch_scores: np.ndarray,
    patch_size: int,
    output_size: tuple[int, int],
    sigma: float = 4.0,
) -> Tensor:
    """Convert flat patch scores to a smooth full-resolution anomaly map.

    Pipeline:
        1. Reshape ``(patch_size², )`` → ``(patch_size, patch_size)``
        2. Bilinear upsample to *output_size*
        3. Gaussian blur with std-dev *sigma* (improves visual quality and
           pixel-level AUROC by ~0.5–1%)

    Args:
        patch_scores: 1-D array of per-patch anomaly scores.
        patch_size: Spatial resolution of the patch grid.
        output_size: ``(H, W)`` of the original input image.
        sigma: Gaussian blur std-dev in pixels (post-upsample).

    Returns:
        ``(1, H, W)`` float32 tensor.
    """
    score_grid = patch_scores.reshape(patch_size, patch_size).astype(np.float32)

    # Upsample via torch for sub-pixel accuracy
    score_tensor = torch.from_numpy(score_grid).unsqueeze(0).unsqueeze(0)  # (1,1,P,P)
    upsampled = F.interpolate(
        score_tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()  # (H, W)

    smoothed = gaussian_filter(upsampled, sigma=sigma)
    return torch.from_numpy(smoothed).unsqueeze(0).float()   # (1, H, W)


# ── PatchCore ─────────────────────────────────────────────────────────────────

class PatchCore:
    """PatchCore anomaly detector (Roth et al., CVPR 2022).

    No backbone training is performed.  The memory bank is built in a single
    forward pass through the defect-free training images.

    Args:
        backbone_name: torchvision model name.  Default: ``'wide_resnet50_2'``.
        layers: Backbone submodule names to hook for feature extraction.
            ``['layer2', 'layer3']`` yields a 1536-d patch embedding for
            WideResNet-50 and captures both mid-level texture and higher-level
            semantic features.
        coreset_ratio: Fraction of all extracted patches to retain in the
            memory bank after coreset subsampling.  Lower values → smaller
            bank → faster inference; 0.01 (1%) is the paper default.
        k_neighbors: Number of nearest neighbours used to compute the patch
            anomaly score.  ``k=1`` (nearest neighbour distance) is standard.
        patch_size: Spatial resolution of the patch grid (must match the
            backbone layer2 output at the chosen input resolution — 28 for
            224×224 input with WideResNet-50).
        smooth_sigma: Gaussian blur std-dev applied to the anomaly map before
            returning it.  Higher values → smoother maps.
        device: ``'cuda'``, ``'cpu'``, or ``'auto'``.

    Example::

        from torch.utils.data import DataLoader
        from anomaly_det.data import (
            MVTecDataset, get_train_transform, get_test_transform, get_mask_transform
        )
        from anomaly_det.models.patchcore import PatchCore

        model = PatchCore(coreset_ratio=0.01)

        train_dl = DataLoader(
            MVTecDataset("data/mvtec", "bottle", "train",
                         transform=get_train_transform()),
            batch_size=32, num_workers=4, pin_memory=True,
        )
        model.fit(train_dl)

        test_ds = MVTecDataset("data/mvtec", "bottle", "test",
                               transform=get_test_transform(),
                               mask_transform=get_mask_transform())
        score, heatmap = model.predict(test_ds[0]["image"])
        # score   : float  — image-level anomaly score
        # heatmap : (1, H, W) Tensor — pixel-level anomaly map
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: Optional[list[str]] = None,
        coreset_ratio: float = 0.01,
        k_neighbors: int = 1,
        patch_size: int = 28,
        smooth_sigma: float = 4.0,
        device: str = "auto",
    ) -> None:
        self.backbone_name = backbone_name
        self.layers: list[str] = layers if layers is not None else ["layer2", "layer3"]
        self.coreset_ratio = coreset_ratio
        self.k_neighbors = k_neighbors
        self.patch_size = patch_size
        self.smooth_sigma = smooth_sigma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._extractor: Optional[_FeatureExtractor] = None
        self._index: Optional[faiss.Index] = None
        self._feature_dim: Optional[int] = None
        self._is_fitted: bool = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_extractor(self) -> _FeatureExtractor:
        if self._extractor is None:
            self._extractor = (
                _FeatureExtractor(self.backbone_name, self.layers)
                .to(self.device)
                .eval()
            )
        return self._extractor

    @torch.no_grad()
    def _embed_batch(self, images: Tensor) -> np.ndarray:
        """Return ``(B × patch_size², D)`` float32 patch embeddings for a batch."""
        extractor = self._get_extractor()
        feature_maps = extractor(images.to(self.device))
        patches = _pool_and_concat(feature_maps, self.patch_size)
        return patches.cpu().numpy().astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, dataloader: DataLoader) -> "PatchCore":
        """Build the memory bank from normal training images.

        Args:
            dataloader: Yields dicts (or tuples) containing ``'image'`` tensors
                of shape ``(B, 3, H, W)``.  Only normal images should be
                included (the MVTec training split guarantees this).

        Returns:
            ``self`` for method chaining.
        """
        print(f"[PatchCore] Building memory bank  ({self.backbone_name} · "
              f"layers={self.layers} · device={self.device})")

        all_patches: list[np.ndarray] = []

        for batch in tqdm(dataloader, desc="  Extracting patch features"):
            images = batch["image"] if isinstance(batch, dict) else batch[0]
            all_patches.append(self._embed_batch(images))

        features = np.concatenate(all_patches, axis=0)  # (N, D)
        self._feature_dim = features.shape[1]

        n_patches = len(features)
        print(f"  Total patches : {n_patches:>10,}")
        print(f"  Feature dim   : {self._feature_dim:>10,}")

        # ── Coreset subsampling ───────────────────────────────────────────────
        coreset = _greedy_coreset(features, ratio=self.coreset_ratio)
        print(f"  Coreset size  : {len(coreset):>10,}  "
              f"({100 * len(coreset) / n_patches:.2f}% of patches)")

        # ── FAISS index ───────────────────────────────────────────────────────
        # IndexFlatL2 gives exact nearest-neighbour search.  At our scale
        # (~1k–10k coreset vectors) this is fast and avoids approximation error.
        self._index = faiss.IndexFlatL2(self._feature_dim)
        self._index.add(coreset)
        self._is_fitted = True

        print(f"[PatchCore] Memory bank ready — {self._index.ntotal:,} vectors indexed.\n")
        return self

    @torch.no_grad()
    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        """Score a single image and produce a pixel-level anomaly heatmap.

        Args:
            image: ``(3, H, W)`` normalised image tensor (no batch dimension).

        Returns:
            A tuple ``(score, heatmap)`` where:

            *   ``score`` is a float — image-level anomaly score computed as
                the max patch score, following the paper.
            *   ``heatmap`` is a ``(1, H, W)`` float32 tensor — upsampled,
                Gaussian-smoothed per-pixel anomaly scores.  Values are raw
                squared L2 distances; they are *not* normalised to [0, 1] so
                that scores are comparable across images within a category.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "PatchCore has not been fitted yet.  Call fit() first."
            )

        H, W = image.shape[-2], image.shape[-1]
        patches = self._embed_batch(image.unsqueeze(0))         # (P, D)

        # k-NN distance query
        distances, _ = self._index.search(patches, k=self.k_neighbors)  # (P, k)
        patch_scores = (
            distances[:, 0] if self.k_neighbors == 1
            else distances.mean(axis=1)
        )                                                        # (P,)

        heatmap = _smooth_and_upsample(
            patch_scores,
            patch_size=self.patch_size,
            output_size=(H, W),
            sigma=self.smooth_sigma,
        )                                                        # (1, H, W)

        image_score = float(heatmap.max())
        return image_score, heatmap

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the fitted memory bank and config to *path* directory.

        Creates two files:
            ``index.faiss``  — FAISS flat index (the memory bank)
            ``config.pt``    — hyperparameter dict
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        torch.save(
            {
                "backbone_name": self.backbone_name,
                "layers": self.layers,
                "coreset_ratio": self.coreset_ratio,
                "k_neighbors": self.k_neighbors,
                "patch_size": self.patch_size,
                "smooth_sigma": self.smooth_sigma,
                "feature_dim": self._feature_dim,
            },
            path / "config.pt",
        )
        print(f"[PatchCore] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PatchCore":
        """Load a previously saved PatchCore model from *path* directory."""
        path = Path(path)
        config = torch.load(path / "config.pt", weights_only=True)
        model = cls(
            backbone_name=config["backbone_name"],
            layers=config["layers"],
            coreset_ratio=config["coreset_ratio"],
            k_neighbors=config["k_neighbors"],
            patch_size=config["patch_size"],
            smooth_sigma=config["smooth_sigma"],
        )
        model._feature_dim = config["feature_dim"]
        model._index = faiss.read_index(str(path / "index.faiss"))
        model._is_fitted = True
        print(f"[PatchCore] Loaded from {path} "
              f"({model._index.ntotal:,} vectors, dim={model._feature_dim})")
        return model

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = (
            f"fitted, {self._index.ntotal:,} vectors"
            if self._is_fitted
            else "not fitted"
        )
        return (
            f"PatchCore("
            f"backbone={self.backbone_name!r}, "
            f"layers={self.layers}, "
            f"coreset_ratio={self.coreset_ratio}, "
            f"k={self.k_neighbors}, "
            f"device={self.device}, "
            f"status={status!r}"
            f")"
        )