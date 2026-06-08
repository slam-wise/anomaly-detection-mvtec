"""Benchmark pipeline: sweep PatchCore across all benchmark categories.

Fits and evaluates PatchCore on each of the five benchmark categories,
writes per-category figures, and serialises results to JSON.

Can be used as a library function or invoked via scripts/run_benchmark.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anomaly_det.data import BENCHMARK_CATEGORIES
from anomaly_det.evaluation import EvalResult, print_benchmark_table
from anomaly_det.pipelines.evaluate import eval_category
from anomaly_det.pipelines.fit import fit_category


def run_benchmark(
    categories: list[str] | None = None,
    data_root: str | Path = "data/mvtec",
    results_dir: str | Path = "results",
    image_size: int = 224,
    coreset_ratio: float = 0.01,
    batch_size: int = 32,
    num_workers: int = 0,
    force_refit: bool = False,
    save_figures: bool = True,
) -> dict[str, EvalResult]:
    """Fit and evaluate PatchCore on each benchmark category.

    Checkpoints are cached under ``results/checkpoints/`` so individual
    categories can be re-evaluated without re-running feature extraction.

    Args:
        categories: List of category names to benchmark.  Defaults to
            :data:`~anomaly_det.data.BENCHMARK_CATEGORIES`.
        data_root: Path to MVTec AD root directory.
        results_dir: Root directory for checkpoints, figures, and metrics.
        image_size: Input resolution for the backbone.
        coreset_ratio: Coreset subsampling ratio for PatchCore.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers (0 = safe on Windows).
        force_refit: Re-run feature extraction even if a checkpoint exists.
        save_figures: Generate and save comparison figures and histograms.

    Returns:
        Dict mapping category name → :class:`~anomaly_det.evaluation.EvalResult`.
    """
    if categories is None:
        categories = BENCHMARK_CATEGORIES

    results_dir = Path(results_dir)
    checkpoint_dir = results_dir / "checkpoints"
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"

    for d in (checkpoint_dir, figures_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 50}")
    print(f"  PatchCore Benchmark  —  {len(categories)} categories")
    print(f"  coreset_ratio={coreset_ratio}  image_size={image_size}")
    print(f"{'═' * 50}\n")

    all_results: dict[str, EvalResult] = {}

    for category in categories:
        print(f"\n{'─' * 50}")
        print(f"  {category.upper()}")
        print(f"{'─' * 50}")

        model = fit_category(
            category=category,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            image_size=image_size,
            coreset_ratio=coreset_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            force_refit=force_refit,
        )

        result = eval_category(
            model=model,
            category=category,
            data_root=data_root,
            figures_dir=figures_dir,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            save_figures=save_figures,
        )

        all_results[category] = result

    # ── Print summary table ───────────────────────────────────────────────────
    print_benchmark_table(all_results)

    # ── Serialise metrics to JSON ─────────────────────────────────────────────
    metrics_payload = {
        cat: {
            "image_auroc": round(r.image_auroc, 6),
            "pixel_auroc": (
                round(r.pixel_auroc, 6) if not np.isnan(r.pixel_auroc) else None
            ),
            "n_normal": r.n_normal,
            "n_anomalous": r.n_anomalous,
        }
        for cat, r in all_results.items()
    }
    image_aurocs = [v["image_auroc"] for v in metrics_payload.values()]
    pixel_aurocs = [v["pixel_auroc"] for v in metrics_payload.values() if v["pixel_auroc"]]
    metrics_payload["_mean"] = {
        "image_auroc": round(float(np.mean(image_aurocs)), 6),
        "pixel_auroc": round(float(np.mean(pixel_aurocs)), 6) if pixel_aurocs else None,
    }
    metrics_payload["_config"] = {
        "coreset_ratio": coreset_ratio,
        "image_size": image_size,
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
    }

    out_path = metrics_dir / "patchcore_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Results saved → {out_path}\n")

    return all_results
