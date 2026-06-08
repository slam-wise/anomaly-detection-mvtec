"""Evaluation metrics for anomaly detection on MVTec AD.

Provides image-level and pixel-level AUROC computation, following the
standard MVTec AD evaluation protocol (Bergmann et al., CVPR 2019).

Image-level AUROC
    Treats the maximum patch anomaly score as the image-level score.
    AUC is computed over all test images (normal + anomalous).

Pixel-level AUROC
    Treats each pixel's heatmap value as a score and each pixel's
    ground-truth mask value (binarised at 0.5) as the label.
    AUC is computed over all pixels in all test images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """All outputs from evaluating one model on one category's test set.

    Stores raw predictions alongside final metrics so that visualisation
    and further analysis can be done without re-running inference.

    Attributes:
        category:     MVTec AD category name.
        image_auroc:  Image-level ROC-AUC (0–1, higher is better).
        pixel_auroc:  Pixel-level ROC-AUC (0–1, higher is better).
                      ``float('nan')`` if no anomalous pixels exist in the
                      test set (degenerate case, not present in MVTec AD).
        scores:       Per-image anomaly scores (max patch distance).
        labels:       Per-image ground-truth labels (0=normal, 1=anomaly).
        heatmaps:     Per-image anomaly maps, each ``(1, H, W)`` float32.
        masks:        Per-image GT binary masks, each ``(1, H, W)`` float32.
        image_paths:  Absolute paths to the source images.
        defect_types: Defect-type subdirectory name for each image.
    """

    category: str
    image_auroc: float
    pixel_auroc: float

    scores: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    heatmaps: list[Tensor] = field(default_factory=list)
    masks: list[Tensor] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    defect_types: list[str] = field(default_factory=list)

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def n_normal(self) -> int:
        return sum(1 for l in self.labels if l == 0)

    @property
    def n_anomalous(self) -> int:
        return sum(1 for l in self.labels if l == 1)

    def anomalous_indices(self) -> list[int]:
        """Return indices of anomalous test samples."""
        return [i for i, l in enumerate(self.labels) if l == 1]

    def normal_indices(self) -> list[int]:
        """Return indices of normal test samples."""
        return [i for i, l in enumerate(self.labels) if l == 0]

    def summary(self) -> str:
        """Return a human-readable one-block summary of results."""
        pixel_str = (
            f"{self.pixel_auroc:.4f}"
            if not np.isnan(self.pixel_auroc)
            else "n/a (no anomalous pixels)"
        )
        return (
            f"{'─' * 40}\n"
            f"  Category    : {self.category}\n"
            f"  Image AUROC : {self.image_auroc:.4f}\n"
            f"  Pixel AUROC : {pixel_str}\n"
            f"  Test images : {len(self.labels)} "
            f"({self.n_normal} normal, {self.n_anomalous} anomalous)\n"
            f"{'─' * 40}"
        )


# ── Evaluation function ───────────────────────────────────────────────────────

def evaluate_category(
    model,
    test_dataloader: DataLoader,
    category: str,
) -> EvalResult:
    """Evaluate a fitted anomaly detector on one MVTec AD category.

    Iterates over *test_dataloader*, calls ``model.predict()`` for each
    image, then computes image-level and pixel-level AUROC.

    Args:
        model: A fitted model with a ``predict(image: Tensor) ->
            (float, Tensor)`` interface (e.g. :class:`~anomaly_det.models.PatchCore`).
        test_dataloader: DataLoader over the test split.  Must yield dicts
            with keys ``'image'``, ``'label'``, ``'mask'``,
            ``'image_path'``, and ``'defect_type'``.
        category: Category name used to label the result.

    Returns:
        :class:`EvalResult` containing metrics and raw predictions.
    """
    scores: list[float] = []
    labels: list[int] = []
    heatmaps: list[Tensor] = []
    masks: list[Tensor] = []
    image_paths: list[str] = []
    defect_types: list[str] = []

    for batch in tqdm(test_dataloader, desc=f"  Evaluating {category}"):
        batch_images = batch["image"]            # (B, 3, H, W)
        batch_labels = batch["label"].tolist()   # list[int]
        batch_masks = batch["mask"]              # (B, 1, H, W)

        for i in range(len(batch_images)):
            score, heatmap = model.predict(batch_images[i])
            scores.append(score)
            heatmaps.append(heatmap.cpu())

        labels.extend(batch_labels)
        masks.extend([batch_masks[i].cpu() for i in range(len(batch_images))])
        image_paths.extend(batch["image_path"])
        defect_types.extend(batch["defect_type"])

    # ── Image-level AUROC ─────────────────────────────────────────────────────
    image_auroc = float(roc_auc_score(labels, scores))

    # ── Pixel-level AUROC ─────────────────────────────────────────────────────
    # Flatten all heatmaps and masks to single 1-D arrays.
    heatmaps_flat = np.concatenate(
        [h.squeeze().numpy().ravel() for h in heatmaps]
    )
    masks_flat = np.concatenate(
        [m.squeeze().numpy().ravel() for m in masks]
    )
    binary_masks = (masks_flat > 0.5).astype(np.int32)

    if binary_masks.sum() == 0:
        # No anomalous pixels in this split — pixel AUROC is undefined.
        pixel_auroc = float("nan")
    else:
        pixel_auroc = float(roc_auc_score(binary_masks, heatmaps_flat))

    return EvalResult(
        category=category,
        image_auroc=image_auroc,
        pixel_auroc=pixel_auroc,
        scores=scores,
        labels=labels,
        heatmaps=heatmaps,
        masks=masks,
        image_paths=image_paths,
        defect_types=defect_types,
    )


# ── Multi-category benchmark ──────────────────────────────────────────────────

def print_benchmark_table(results: dict[str, EvalResult]) -> None:
    """Print a formatted benchmark table to stdout.

    Args:
        results: Mapping of ``category_name → EvalResult``.
    """
    header = f"{'Category':<14} {'Image AUROC':>12} {'Pixel AUROC':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    image_aurocs = []
    pixel_aurocs = []

    for cat, result in sorted(results.items()):
        pixel_str = (
            f"{result.pixel_auroc:.4f}"
            if not np.isnan(result.pixel_auroc)
            else "     n/a"
        )
        print(f"{cat:<14} {result.image_auroc:>12.4f} {pixel_str:>12}")
        image_aurocs.append(result.image_auroc)
        if not np.isnan(result.pixel_auroc):
            pixel_aurocs.append(result.pixel_auroc)

    print("-" * len(header))
    mean_pixel = np.mean(pixel_aurocs) if pixel_aurocs else float("nan")
    print(
        f"{'Mean':<14} {np.mean(image_aurocs):>12.4f} {mean_pixel:>12.4f}"
    )
    print("=" * len(header) + "\n")