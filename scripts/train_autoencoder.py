"""Train and evaluate the autoencoder baseline on MVTec AD categories.

Usage
-----
    # Train + evaluate on all 5 benchmark categories
    python scripts/train_autoencoder.py

    # Single category
    python scripts/train_autoencoder.py --categories bottle

    # Fewer epochs for a quick test
    python scripts/train_autoencoder.py --epochs 50 --categories bottle

    # Force retrain even if a checkpoint exists
    python scripts/train_autoencoder.py --force-retrain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from anomaly_det.data import ALL_CATEGORIES, BENCHMARK_CATEGORIES, MVTecDataset, get_train_transform
from anomaly_det.evaluation import print_benchmark_table
from anomaly_det.models.autoencoder import Autoencoder
from anomaly_det.pipelines.evaluate import eval_category


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the autoencoder anomaly detection baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        choices=ALL_CATEGORIES + ["all"],
        metavar="CATEGORY",
        help="Categories to train on. Default: 5-category benchmark subset.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/mvtec"),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Maximum training epochs per category. Default: 200",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Adam learning rate. Default: 2e-4",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early stopping patience. Default: 20",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Retrain even if a checkpoint already exists.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
    )
    args = parser.parse_args()

    categories = (
        ALL_CATEGORIES if args.categories == ["all"]
        else (args.categories or BENCHMARK_CATEGORIES)
    )

    results_dir = args.results_dir
    checkpoint_dir = results_dir / "checkpoints_ae"
    figures_dir = results_dir / "figures_ae"
    metrics_dir = results_dir / "metrics"
    for d in (checkpoint_dir, figures_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 50}")
    print(f"  Autoencoder Baseline  —  {len(categories)} categories")
    print(f"  epochs={args.epochs}  lr={args.lr}  patience={args.patience}")
    print(f"{'═' * 50}\n")

    all_results = {}

    for category in categories:
        print(f"\n{'─' * 50}")
        print(f"  {category.upper()}")
        print(f"{'─' * 50}")

        checkpoint_path = checkpoint_dir / category

        # Load cached model or train from scratch
        if not args.force_retrain and (checkpoint_path / "weights.pt").exists():
            print(f"[train_ae] Loading cached checkpoint for '{category}'")
            model = Autoencoder.load(checkpoint_path)
        else:
            model = Autoencoder()
            train_ds = MVTecDataset(
                args.data_root, category, "train",
                transform=get_train_transform(args.image_size),
            )
            train_dl = DataLoader(
                train_ds, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers, pin_memory=True,
            )
            model.fit(train_dl, epochs=args.epochs, lr=args.lr, patience=args.patience)
            model.save(checkpoint_path)

        result = eval_category(
            model=model,
            category=category,
            data_root=args.data_root,
            figures_dir=figures_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_figures=not args.no_figures,
        )
        all_results[category] = result

    print("\n── Autoencoder Results ──")
    print_benchmark_table(all_results)

    # Save metrics
    metrics_payload = {
        cat: {
            "image_auroc": round(r.image_auroc, 6),
            "pixel_auroc": round(r.pixel_auroc, 6) if not np.isnan(r.pixel_auroc) else None,
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
        "epochs": args.epochs,
        "lr": args.lr,
        "base_channels": 64,
        "architecture": "conv_autoencoder",
    }

    out_path = metrics_dir / "autoencoder_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Results saved → {out_path}\n")


if __name__ == "__main__":
    main()