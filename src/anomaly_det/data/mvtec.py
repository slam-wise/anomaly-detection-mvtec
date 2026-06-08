"""PyTorch Dataset for the MVTec Anomaly Detection benchmark.

The MVTec AD dataset contains 15 industrial categories. Each category has:
    - A training set of defect-free images only (one-class setting).
    - A test set with both normal and anomalous images.
    - Pixel-precise ground-truth masks for all anomalous test images.

Reference
---------
Bergmann et al., "MVTec AD – A Comprehensive Real-World Dataset for
Unsupervised Anomaly Detection", CVPR 2019.
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Expected on-disk layout
-----------------------
    <root>/
    ├── bottle/
    │   ├── train/
    │   │   └── good/             ← PNG files, defect-free
    │   ├── test/
    │   │   ├── good/             ← PNG files, defect-free
    │   │   ├── broken_large/     ← PNG files, anomalous (one dir per defect type)
    │   │   └── broken_small/
    │   └── ground_truth/
    │       ├── broken_large/     ← *_mask.png binary masks aligned to test images
    │       └── broken_small/
    └── cable/
        └── ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

# ── Category constants ────────────────────────────────────────────────────────

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

BENCHMARK_CATEGORIES: list[str] = [
    "bottle",
    "cable",
    "hazelnut",
    "leather",
    "screw",
]


# ── Dataset class ─────────────────────────────────────────────────────────────

class MVTecDataset(Dataset):
    """One-class anomaly detection dataset wrapping MVTec AD.

    Args:
        root: Path to the MVTec AD root directory (e.g. ``data/mvtec``).
        category: One of the 15 MVTec AD category names.
        split: ``'train'`` returns only defect-free images (used to build the
            PatchCore memory bank). ``'test'`` returns all test images with
            labels and ground-truth masks.
        transform: Transform applied to each PIL image. If ``None``, the raw
            PIL image is returned — callers should always provide a transform.
        mask_transform: Transform applied to ground-truth masks (test split
            only). If ``None``, raw PIL images are returned for masks.

    Returns (per ``__getitem__``):
        A dict with the following keys:

        ``image`` : torch.Tensor
            Transformed image tensor, shape ``(3, H, W)``.
        ``label`` : torch.Tensor
            Scalar long tensor. ``0`` = normal, ``1`` = anomaly.
        ``mask`` : torch.Tensor
            Binary mask tensor, shape ``(1, H, W)``. All zeros for normal
            images; pixel value > 0 indicates an anomalous region.
        ``image_path`` : str
            Absolute path to the source image file (useful for debugging).
        ``defect_type`` : str
            Subdirectory name, e.g. ``'good'``, ``'broken_large'``.

    Example::

        from anomaly_det.data.mvtec import MVTecDataset
        from anomaly_det.data.transforms import get_train_transform, get_mask_transform

        train_ds = MVTecDataset(
            root="data/mvtec",
            category="bottle",
            split="train",
            transform=get_train_transform(),
        )
        sample = train_ds[0]
        print(sample["image"].shape)   # torch.Size([3, 224, 224])
        print(sample["label"])         # tensor(0)
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ) -> None:
        if category not in ALL_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Valid categories: {ALL_CATEGORIES}"
            )
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'.")

        self.root = Path(root).resolve()
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        self._samples: list[dict] = self._build_sample_list()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_sample_list(self) -> list[dict]:
        """Scan the dataset directory tree and return a flat list of sample dicts.

        Each dict contains:
            image_path  : Path  — absolute path to the image
            label       : int   — 0 = normal, 1 = anomaly
            mask_path   : Path | None — mask file for anomalous images
            defect_type : str   — subdirectory name under split/
        """
        samples: list[dict] = []
        split_dir = self.root / self.category / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                "Run `python scripts/download_data.py` to download the dataset."
            )

        if self.split == "train":
            # Training set: only the single 'good' subdirectory
            good_dir = split_dir / "good"
            for img_path in sorted(good_dir.glob("*.png")):
                samples.append(
                    {
                        "image_path": img_path,
                        "label": 0,
                        "mask_path": None,
                        "defect_type": "good",
                    }
                )
        else:
            # Test set: 'good' + one subdirectory per defect type
            gt_dir = self.root / self.category / "ground_truth"
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                is_anomaly = defect_dir.name != "good"
                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path: Optional[Path] = None
                    if is_anomaly:
                        # MVTec AD mask naming: <stem>_mask.png
                        mask_path = gt_dir / defect_dir.name / f"{img_path.stem}_mask.png"
                    samples.append(
                        {
                            "image_path": img_path,
                            "label": int(is_anomaly),
                            "mask_path": mask_path,
                            "defect_type": defect_dir.name,
                        }
                    )

        if len(samples) == 0:
            raise RuntimeError(
                f"No PNG images found in {split_dir}. "
                "The dataset may not have extracted correctly."
            )

        return samples

    def _load_mask(self, mask_path: Optional[Path], image_size: tuple[int, int]) -> Image.Image:
        """Load an anomaly mask, or return a blank (all-zero) mask for normal images.

        Args:
            mask_path: Path to the mask PNG, or ``None`` for normal images.
            image_size: ``(width, height)`` of the corresponding image, used to
                create a correctly-sized blank mask when there is no GT mask.

        Returns:
            A grayscale PIL Image (mode ``'L'``).
        """
        if mask_path is not None and mask_path.exists():
            return Image.open(mask_path).convert("L")
        # Normal image — return an all-zero mask matching the image dimensions
        return Image.new("L", image_size, color=0)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        mask = self._load_mask(sample["mask_path"], image.size)

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            # If no mask_transform, convert to tensor so collation always works
            mask = torch.zeros(1, 1, 1, dtype=torch.float32)

        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "mask": mask,
            "image_path": str(sample["image_path"]),
            "defect_type": sample["defect_type"],
        }

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def anomaly_count(self) -> int:
        """Number of anomalous samples (test split only; always 0 for train)."""
        return sum(s["label"] for s in self._samples)

    @property
    def normal_count(self) -> int:
        """Number of normal samples."""
        return sum(1 - s["label"] for s in self._samples)

    @property
    def defect_types(self) -> list[str]:
        """Sorted list of unique defect type names present in this split."""
        return sorted({s["defect_type"] for s in self._samples})

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"category={self.category!r}, "
            f"split={self.split!r}, "
            f"n_samples={len(self)}, "
            f"n_anomalous={self.anomaly_count}, "
            f"defect_types={self.defect_types}"
            f")"
        )