"""Fit PatchCore on a single MVTec AD category and save the checkpoint.

For running the full 5-category benchmark, use scripts/run_benchmark.py instead.
This script is useful when you want to fit one category quickly — for example
to inspect a checkpoint, test a hyperparameter change, or pre-build the memory
bank before running evaluation separately.

Usage
-----
    # Fit on bottle (default)
    python scripts/train_patchcore.py --category bottle

    # Custom coreset ratio
    python scripts/train_patchcore.py --category screw --coreset-ratio 0.05

    # Force refit even if a checkpoint exists
    python scripts/train_patchcore.py --category cable --force-refit

    # Print model summary after fitting
    python scripts/train_patchcore.py --category hazelnut --summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anomaly_det.data import ALL_CATEGORIES
from anomaly_det.pipelines.fit import fit_category


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit PatchCore on one MVTec AD category.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--category", required=True,
        choices=ALL_CATEGORIES, metavar="CATEGORY",
        help=f"Category to fit. One of: {', '.join(ALL_CATEGORIES)}",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/mvtec"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("results/checkpoints"),
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
    )
    parser.add_argument(
        "--coreset-ratio", type=float, default=0.01,
        help="Fraction of patches to keep in the memory bank. Default: 0.01",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
    )
    parser.add_argument(
        "--force-refit", action="store_true",
        help="Re-run feature extraction even if a checkpoint already exists.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a model summary after fitting.",
    )
    args = parser.parse_args()

    model = fit_category(
        category=args.category,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        image_size=args.image_size,
        coreset_ratio=args.coreset_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_refit=args.force_refit,
    )

    if args.summary:
        print(model)


if __name__ == "__main__":
    main()