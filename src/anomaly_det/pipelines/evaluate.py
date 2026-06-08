"""Evaluate pipeline: score a fitted model on one MVTec AD category.

Can be used as a library function or invoked via scripts/run_benchmark.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

from anomaly_det.data import MVTecDataset, get_mask_transform, get_test_transform
from anomaly_det.evaluation import (
    EvalResult,
    evaluate_category,
    plot_score_distribution,
    save_category_figure,
)
from anomaly_det.models import PatchCore


def eval_category(
    model: PatchCore,
    category: str,
    data_root: str | Path = "data/mvtec",
    figures_dir: str | Path = "results/figures",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    n_anomalous_figures: int = 4,
    n_normal_figures: int = 1,
    save_figures: bool = True,
) -> EvalResult:
    """Evaluate *model* on the test split of *category*.

    Produces an anomaly heatmap comparison figure and a score distribution
    histogram, both saved under *figures_dir*.

    Args:
        model: A fitted :class:`~anomaly_det.models.PatchCore`.
        category: MVTec AD category name.
        data_root: Path to the MVTec AD root directory.
        figures_dir: Directory to write output figures.
        image_size: Must match the value used during fitting.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker processes.
        n_anomalous_figures: Anomalous rows in the comparison grid.
        n_normal_figures: Normal rows in the comparison grid.
        save_figures: If False, skip figure generation (faster, useful for CI).

    Returns:
        :class:`~anomaly_det.evaluation.EvalResult` with metrics and
        raw predictions.
    """
    print(f"[eval] Evaluating '{category}' ...")

    test_ds = MVTecDataset(
        root=data_root,
        category=category,
        split="test",
        transform=get_test_transform(image_size),
        mask_transform=get_mask_transform(image_size),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    result = evaluate_category(model, test_dl, category)
    print(result.summary())

    if save_figures:
        figures_dir = Path(figures_dir)
        save_category_figure(
            result,
            test_ds,
            figures_dir / f"{category}_comparison.png",
            n_anomalous=n_anomalous_figures,
            n_normal=n_normal_figures,
        )
        plot_score_distribution(
            result,
            output_path=figures_dir / f"{category}_score_dist.png",
        )
        import matplotlib.pyplot as plt
        plt.close("all")

    return result