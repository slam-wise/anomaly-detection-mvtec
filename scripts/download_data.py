"""Download and extract the MVTec Anomaly Detection dataset.

Usage
-----
    # Download full archive and verify the 5-category benchmark subset (default)
    python scripts/download_data.py

    # Verify all 15 categories after extraction
    python scripts/download_data.py --categories all

    # Point to a custom data directory
    python scripts/download_data.py --root /path/to/data/mvtec

    # Skip re-downloading if the archive already exists on disk
    python scripts/download_data.py --skip-download

Dataset structure after extraction
-----------------------------------
    data/mvtec/
    ├── bottle/
    │   ├── train/
    │   │   └── good/          ← defect-free training images
    │   ├── test/
    │   │   ├── good/          ← defect-free test images
    │   │   ├── broken_large/  ← anomalous test images (one dir per defect type)
    │   │   └── broken_small/
    │   └── ground_truth/
    │       ├── broken_large/  ← binary masks (*_mask.png) aligned to test images
    │       └── broken_small/
    ├── cable/
    ├── hazelnut/
    └── ...                    ← 15 categories total

License
-------
MVTec AD is released under CC BY-NC-SA 4.0.
Non-commercial use only. Cite the original paper if you use this dataset:
    Bergmann et al., "MVTec AD – A Comprehensive Real-World Dataset for
    Unsupervised Anomaly Detection", CVPR 2019.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

# ── Dataset constants ─────────────────────────────────────────────────────────

# NOTE: The historical mydrive.ch mirror has been unreliable since late 2025.
# The script will attempt it but will print clear manual instructions if it fails.
MVTEC_URL = (
    "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282"
    "/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
)

MANUAL_DOWNLOAD_MSG = """
Automatic download failed. The mirror URL has been intermittently broken since late 2025.

To get the dataset:
  Option A (recommended):
    1. Visit  https://www.mvtec.com/research-teaching/datasets/mvtec-ad
    2. Fill in your credentials and agree to the terms to access the download page.
    3. Download  mvtec_anomaly_detection.tar.xz  (~4.9 GB)
    4. Place it at  {archive_path}
    5. Re-run:  python scripts/download_data.py --skip-download

  Option B (Hugging Face mirror):
    https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split
"""

ALL_CATEGORIES: list[str] = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

# The 5-category benchmark subset used in this project.
# Chosen to represent visual variety and a range of difficulty levels.
BENCHMARK_CATEGORIES: list[str] = [
    "bottle",    # rigid object, clean backgrounds — good for README hero figures
    "cable",     # deformable, structural defects — closest to NDT/X-ray domain
    "hazelnut",  # organic texture, natural variation
    "leather",   # texture category, different failure modes from objects
    "screw",     # fine-grained, high difficulty — impressive when it works
]

ARCHIVE_NAME = "mvtec_anomaly_detection.tar.xz"


# ── Download helpers ──────────────────────────────────────────────────────────

class _TqdmHook(tqdm):
    """tqdm progress hook for urllib.request.urlretrieve."""

    def update_to(self, n_blocks: int = 1, block_size: int = 1, total: int = -1) -> None:
        if total > 0:
            self.total = total
        self.update(n_blocks * block_size - self.n)


def _download(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    print(f"Downloading: {url}")
    print(f"        → {dest}\n")
    with _TqdmHook(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as hook:
        urllib.request.urlretrieve(url, filename=dest, reporthook=hook.update_to)
    print()


def _extract(archive: Path, dest: Path) -> None:
    """Extract a .tar.xz archive to *dest* with a per-file progress bar."""
    print(f"Extracting {archive.name} → {dest} ...")
    with tarfile.open(archive, "r:xz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting", unit="files", leave=False):
            tar.extract(member, path=dest, filter="data")
    print("Extraction complete.\n")


# ── Verification ──────────────────────────────────────────────────────────────

def _verify(root: Path, categories: list[str]) -> bool:
    """Check that each category has the expected train/test/ground_truth layout."""
    print("Verifying directory structure ...")
    all_ok = True
    for cat in categories:
        train_good = root / cat / "train" / "good"
        test_dir = root / cat / "test"
        gt_dir = root / cat / "ground_truth"
        n_train = len(list(train_good.glob("*.png"))) if train_good.exists() else 0
        ok = train_good.exists() and test_dir.exists() and gt_dir.exists() and n_train > 0
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {cat:<14}  ({n_train} training images)")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All categories verified — dataset ready.")
    else:
        print("One or more categories failed verification. Try re-running without --skip-download.")
    return all_ok


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract the MVTec AD dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/mvtec"),
        help="Directory to extract the dataset into. Default: data/mvtec",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["benchmark"],
        choices=ALL_CATEGORIES + ["all", "benchmark"],
        metavar="CATEGORY",
        help=(
            "Which categories to verify after extraction. "
            "'benchmark' (default) = the 5 categories used in this project. "
            "'all' = all 15 categories. "
            f"Individual names: {', '.join(ALL_CATEGORIES)}"
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step and use an existing archive on disk.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Do not prompt to delete the archive after extraction.",
    )
    args = parser.parse_args()

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    # Resolve which categories to verify
    if args.categories == ["all"]:
        verify_cats = ALL_CATEGORIES
    elif args.categories == ["benchmark"]:
        verify_cats = BENCHMARK_CATEGORIES
    else:
        verify_cats = args.categories

    # ── Fast-path: already extracted ─────────────────────────────────────────
    if all((root / cat / "train" / "good").exists() for cat in verify_cats):
        print("Dataset already extracted. Skipping download and extraction.\n")
        _verify(root, verify_cats)
        return

    # ── Download ─────────────────────────────────────────────────────────────
    archive = root / ARCHIVE_NAME
    if args.skip_download:
        if not archive.exists():
            print(
                f"Error: --skip-download was set but archive not found at {archive}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using existing archive at {archive}\n")
    else:
        if archive.exists():
            print(f"Archive already present at {archive} — skipping download.\n")
        else:
            try:
                _download(MVTEC_URL, archive)
            except Exception as exc:
                print(f"\nDownload failed: {exc}", file=sys.stderr)
                print(MANUAL_DOWNLOAD_MSG.format(archive_path=archive), file=sys.stderr)
                sys.exit(1)

    # ── Extract ──────────────────────────────────────────────────────────────
    _extract(archive, root)

    # ── Verify ───────────────────────────────────────────────────────────────
    ok = _verify(root, verify_cats)

    # ── Clean up archive ─────────────────────────────────────────────────────
    if not args.keep_archive and archive.exists():
        try:
            answer = input("\nDelete archive to reclaim ~4.9 GB? [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer == "y":
            archive.unlink()
            print(f"Deleted {archive.name}.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()