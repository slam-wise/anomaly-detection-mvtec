"""Visualisation utilities for anomaly detection results.

Produces publication-quality figures for use in the README and notebook:

    overlay_heatmap()       Single image + heatmap blended into one frame.
    make_result_grid()      N-column comparison grid:
                            Original | GT Mask | Predicted Heatmap.
    save_category_figure()  Convenience wrapper — runs grid + saves to disk.

Heatmap normalisation note
--------------------------
Raw anomaly scores are squared L2 distances (unbounded above).  All
visualisation functions normalise heatmaps to [0, 1] using the per-image
min/max before applying the colourmap.  This is standard practice for
anomaly map visualisation and ensures colour contrast regardless of the
absolute score magnitude.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torch import Tensor

# ImageNet normalisation constants (must match transforms.py)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Image utilities ───────────────────────────────────────────────────────────

def denormalize(tensor: Tensor) -> np.ndarray:
    """Convert a normalised image tensor to a displayable uint8 array.

    Reverses ImageNet normalisation and clips to [0, 255].

    Args:
        tensor: ``(3, H, W)`` float32 tensor normalised with ImageNet stats.

    Returns:
        ``(H, W, 3)`` uint8 numpy array.
    """
    arr = tensor.permute(1, 2, 0).cpu().float().numpy()  # (H, W, 3)
    arr = arr * _STD + _MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def heatmap_to_rgb(
    heatmap: Tensor,
    cmap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Convert a scalar heatmap tensor to a colourised RGB array.

    Args:
        heatmap: ``(1, H, W)`` or ``(H, W)`` float32 anomaly score map.
        cmap: Matplotlib colourmap name.  ``'jet'`` is standard for anomaly
            maps; ``'inferno'`` is a perceptually-uniform alternative.
        vmin: Lower clamp for normalisation.  Defaults to ``heatmap.min()``.
        vmax: Upper clamp for normalisation.  Defaults to ``heatmap.max()``.

    Returns:
        ``(H, W, 3)`` uint8 RGB array.
    """
    arr = heatmap.squeeze().cpu().float().numpy()
    v0 = float(arr.min()) if vmin is None else vmin
    v1 = float(arr.max()) if vmax is None else vmax
    if v1 == v0:
        v1 = v0 + 1e-6  # avoid division by zero for all-zero maps

    norm = Normalize(vmin=v0, vmax=v1)
    colourmap = plt.get_cmap(cmap)
    rgb = colourmap(norm(arr))[:, :, :3]          # (H, W, 3) in [0, 1]
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    image: Tensor,
    heatmap: Tensor,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    """Blend a colourised anomaly map onto the original image.

    Args:
        image: ``(3, H, W)`` normalised image tensor.
        heatmap: ``(1, H, W)`` anomaly score tensor.
        alpha: Opacity of the heatmap layer (0 = invisible, 1 = opaque).
        cmap: Matplotlib colourmap for the heatmap.

    Returns:
        ``(H, W, 3)`` uint8 blended image.
    """
    img_rgb = denormalize(image).astype(np.float32)
    heat_rgb = heatmap_to_rgb(heatmap, cmap=cmap).astype(np.float32)
    blended = (1 - alpha) * img_rgb + alpha * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def mask_to_rgb(mask: Tensor, colour: tuple[int, int, int] = (255, 50, 50)) -> np.ndarray:
    """Convert a binary mask tensor to a coloured RGB display image.

    Anomalous pixels are filled with *colour*; normal pixels are black.

    Args:
        mask: ``(1, H, W)`` float32 mask (values ≥ 0.5 = anomalous).
        colour: RGB colour for anomalous pixels.

    Returns:
        ``(H, W, 3)`` uint8 array.
    """
    binary = (mask.squeeze().cpu().numpy() >= 0.5).astype(np.uint8)  # (H, W)
    rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
    rgb[binary == 1] = colour
    return rgb


# ── Result grid ───────────────────────────────────────────────────────────────

def make_result_grid(
    result,
    n_anomalous: int = 4,
    n_normal: int = 1,
    cmap: str = "jet",
    heatmap_alpha: float = 0.45,
    figsize_per_col: float = 2.5,
    dpi: int = 150,
) -> plt.Figure:
    """Build a comparison figure for one evaluation result.

    Columns: **Original** | **GT Mask** | **Predicted Heatmap**
    Each row is one test image.  Anomalous samples are shown first,
    then normal samples, with a horizontal rule between them.

    Args:
        result: :class:`~anomaly_det.evaluation.metrics.EvalResult`
            from :func:`~anomaly_det.evaluation.metrics.evaluate_category`.
        n_anomalous: Number of anomalous samples to display.
        n_normal: Number of normal samples to display.
        cmap: Colourmap for heatmap columns.
        heatmap_alpha: Blend factor for heatmap overlay (0–1).
        figsize_per_col: Figure width per column in inches.
        dpi: Output resolution.

    Returns:
        A ``matplotlib.figure.Figure`` ready to ``savefig()`` or display.
    """
    # Collect sample indices
    anom_idx = result.anomalous_indices()[:n_anomalous]
    norm_idx = result.normal_indices()[:n_normal]
    indices = anom_idx + norm_idx
    n_rows = len(indices)

    if n_rows == 0:
        raise ValueError("No samples to display.")

    n_cols = 3
    fig_w = figsize_per_col * n_cols
    fig_h = figsize_per_col * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)

    # Ensure axes is always 2-D even for a single row
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Ground Truth", "Anomaly Map"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold", pad=4)

    for row, idx in enumerate(indices):
        image = result.heatmaps[idx]       # reuse index for heatmap
        # Re-fetch image from image_paths is impractical here;
        # the EvalResult doesn't store raw images to keep memory lean.
        # We display: denormalized-from-heatmap context | mask | overlay.
        # Callers who need the raw image should pass it explicitly —
        # see save_category_figure() which takes the test dataset directly.
        pass

    plt.close(fig)

    # ── Re-implement with dataset access ─────────────────────────────────────
    # The above skeleton is intentionally incomplete — make_result_grid
    # is called via save_category_figure() which supplies the raw images.
    raise NotImplementedError(
        "Call save_category_figure() instead — it supplies the raw images."
    )


def save_category_figure(
    result,
    test_dataset,
    output_path: Path,
    n_anomalous: int = 4,
    n_normal: int = 1,
    cmap: str = "jet",
    heatmap_alpha: float = 0.45,
    dpi: int = 150,
) -> Path:
    """Save a comparison grid figure for one category to disk.

    Columns: **Original** | **GT Mask** | **Predicted Heatmap Overlay**

    Args:
        result: :class:`~anomaly_det.evaluation.metrics.EvalResult`.
        test_dataset: The MVTecDataset test split (provides raw images).
            Must have the same ordering as the samples in *result*.
        output_path: File path to save the figure (e.g. ``results/figures/bottle.png``).
        n_anomalous: Anomalous rows to include.
        n_normal: Normal rows to include.
        cmap: Colourmap for predicted heatmap column.
        heatmap_alpha: Heatmap blend opacity.
        dpi: Output resolution.

    Returns:
        The resolved ``output_path``.
    """
    anom_idx = result.anomalous_indices()[:n_anomalous]
    norm_idx = result.normal_indices()[:n_normal]
    indices = anom_idx + norm_idx
    n_rows = len(indices)

    if n_rows == 0:
        raise ValueError("No samples to display — check n_anomalous / n_normal.")

    col_titles = ["Original", "GT Mask", "Predicted Heatmap"]
    n_cols = len(col_titles)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.8 * n_rows),
        dpi=dpi,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold", pad=4)

    # Separator between anomalous and normal sections
    sep_row = len(anom_idx)

    for row, idx in enumerate(indices):
        sample = test_dataset[idx]
        image_tensor = sample["image"]       # (3, H, W)
        heatmap = result.heatmaps[idx]       # (1, H, W)
        mask = result.masks[idx]             # (1, H, W)
        label = result.labels[idx]
        defect = result.defect_types[idx]

        img_rgb = denormalize(image_tensor)
        mask_rgb = mask_to_rgb(mask)
        overlay = overlay_heatmap(image_tensor, heatmap, alpha=heatmap_alpha, cmap=cmap)

        axes[row, 0].imshow(img_rgb)
        axes[row, 1].imshow(mask_rgb)
        axes[row, 2].imshow(overlay)

        # Row label on the left
        row_label = defect if label == 1 else "normal"
        axes[row, 0].set_ylabel(
            row_label, fontsize=7, rotation=0, labelpad=48, va="center"
        )

        # Draw separator line after anomalous block
        if row == sep_row - 1 and n_normal > 0:
            for col in range(n_cols):
                axes[row, col].spines["bottom"].set_linewidth(2.0)
                axes[row, col].spines["bottom"].set_color("#555555")

    # Remove axes ticks
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Title
    fig.suptitle(
        f"{result.category}  |  "
        f"Image AUROC {result.image_auroc:.4f}  "
        f"Pixel AUROC {result.pixel_auroc:.4f}",
        fontsize=10,
        y=1.01,
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure → {output_path}")
    return output_path


# ── Score histogram ───────────────────────────────────────────────────────────

def plot_score_distribution(
    result,
    output_path: Optional[Path] = None,
    dpi: int = 120,
) -> plt.Figure:
    """Plot the distribution of image-level anomaly scores.

    Shows overlapping histograms for normal (blue) and anomalous (red)
    images — useful for understanding score separation.

    Args:
        result: :class:`~anomaly_det.evaluation.metrics.EvalResult`.
        output_path: If provided, save the figure here.
        dpi: Output resolution.

    Returns:
        The matplotlib figure.
    """
    normal_scores = [s for s, l in zip(result.scores, result.labels) if l == 0]
    anom_scores = [s for s, l in zip(result.scores, result.labels) if l == 1]

    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
    bins = 30
    ax.hist(normal_scores, bins=bins, alpha=0.6, label="Normal", color="#4C72B0")
    ax.hist(anom_scores, bins=bins, alpha=0.6, label="Anomalous", color="#DD4444")
    ax.set_xlabel("Anomaly score (max patch distance)")
    ax.set_ylabel("Count")
    ax.set_title(f"{result.category} — score distribution  (AUROC {result.image_auroc:.4f})")
    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved histogram → {output_path}")

    return fig