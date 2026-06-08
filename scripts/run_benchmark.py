"""Run the full PatchCore benchmark across MVTec AD categories.

Usage
-----
    # Benchmark all 5 categories (default)
    python scripts/run_benchmark.py

    # Benchmark specific categories
    python scripts/run_benchmark.py --categories bottle hazelnut

    # Force re-fit even if checkpoints exist
    python scripts/run_benchmark.py --force-refit

    # Skip figure generation (faster, useful for quick metric checks)
    python scripts/run_benchmark.py --no-figures

    # Full options
    python scripts/run_benchmark.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anomaly_det.data import ALL_CATEGORIES, BENCHMARK_CATEGORIES
from anomaly_det.pipelines.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PatchCore benchmark over MVTec AD categories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        choices=ALL_CATEGORIES + ["all"],
        metavar="CATEGORY",
        help=(
            "Categories to benchmark. "
            f"Defaults to the 5-category benchmark subset: {BENCHMARK_CATEGORIES}. "
            "Pass 'all' for all 15 MVTec AD categories."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/mvtec"),
        help="MVTec AD root directory. Default: data/mvtec",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory for checkpoints, figures, and metrics. Default: results/",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input resolution fed to the backbone. Default: 224",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.01,
        help="Fraction of patches to keep in the memory bank. Default: 0.01",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader batch size. Default: 32",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Default: 0 (safe on Windows)",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Re-run feature extraction even if a checkpoint already exists.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation (faster for metric-only runs).",
    )
    args = parser.parse_args()

    categories = None
    if args.categories == ["all"]:
        categories = ALL_CATEGORIES
    elif args.categories is not None:
        categories = args.categories

    run_benchmark(
        categories=categories,
        data_root=args.data_root,
        results_dir=args.results_dir,
        image_size=args.image_size,
        coreset_ratio=args.coreset_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_refit=args.force_refit,
        save_figures=not args.no_figures,
    )


if __name__ == "__main__":
    main()