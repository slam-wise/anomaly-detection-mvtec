"""Fit pipeline: build a PatchCore memory bank for one MVTec AD category.

Can be used as a library function or invoked via scripts/run_benchmark.py.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from anomaly_det.data import MVTecDataset, get_train_transform
from anomaly_det.models import PatchCore


def fit_category(
    category: str,
    data_root: str | Path = "data/mvtec",
    checkpoint_dir: str | Path = "results/checkpoints",
    image_size: int = 224,
    coreset_ratio: float = 0.01,
    batch_size: int = 32,
    num_workers: int = 0,
    force_refit: bool = False,
) -> PatchCore:
    """Fit PatchCore on the normal training images for *category*.

    If a saved checkpoint already exists and *force_refit* is False, loads
    and returns the cached model without re-running feature extraction.

    Args:
        category: MVTec AD category name (e.g. ``'bottle'``).
        data_root: Path to the MVTec AD root directory.
        checkpoint_dir: Directory where fitted models are saved.
        image_size: Spatial resolution fed to the backbone (224 = standard).
        coreset_ratio: Fraction of patches to keep in the memory bank.
        batch_size: DataLoader batch size for feature extraction.
        num_workers: DataLoader worker processes (0 = main process, safe on Windows).
        force_refit: If True, ignore any existing checkpoint and refit.

    Returns:
        A fitted :class:`~anomaly_det.models.PatchCore` model.
    """
    checkpoint_path = Path(checkpoint_dir) / category

    # ── Load cached model if available ───────────────────────────────────────
    if not force_refit and (checkpoint_path / "index.faiss").exists():
        print(f"[fit] Loading cached checkpoint for '{category}'  ({checkpoint_path})")
        return PatchCore.load(checkpoint_path)

    # ── Build memory bank ─────────────────────────────────────────────────────
    print(f"[fit] Fitting PatchCore on '{category}' ...")
    model = PatchCore(coreset_ratio=coreset_ratio)

    train_ds = MVTecDataset(
        root=data_root,
        category=category,
        split="train",
        transform=get_train_transform(image_size),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.fit(train_dl)
    model.save(checkpoint_path)
    return model