"""Tests for MVTecDataset and transforms.

Run after downloading the dataset:
    pytest tests/test_dataset.py -v

Tests that require the dataset on disk are automatically skipped if the
data directory does not exist, so CI passes without needing the ~5 GB archive.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from anomaly_det.data import (
    BENCHMARK_CATEGORIES,
    MVTecDataset,
    get_mask_transform,
    get_test_transform,
    get_train_transform,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

DATA_ROOT = Path("data/mvtec")
REQUIRES_DATA = pytest.mark.skipif(
    not (DATA_ROOT / "bottle" / "train" / "good").exists(),
    reason="MVTec AD dataset not found at data/mvtec — run scripts/download_data.py",
)


# ── Transform tests (no data needed) ─────────────────────────────────────────

class TestTransforms:
    def test_train_transform_output_shape(self):
        from PIL import Image
        img = Image.new("RGB", (900, 900))
        transform = get_train_transform(224)
        tensor = transform(img)
        assert tensor.shape == torch.Size([3, 224, 224])

    def test_test_transform_output_shape(self):
        from PIL import Image
        img = Image.new("RGB", (1024, 768))
        transform = get_test_transform(224)
        tensor = transform(img)
        assert tensor.shape == torch.Size([3, 224, 224])

    def test_mask_transform_output_shape(self):
        from PIL import Image
        mask = Image.new("L", (900, 900))
        transform = get_mask_transform(224)
        tensor = transform(mask)
        assert tensor.shape == torch.Size([1, 224, 224])

    def test_mask_transform_preserves_binary_values(self):
        """NEAREST resize must not introduce grey-valued artefacts."""
        import numpy as np
        from PIL import Image
        arr = (torch.randint(0, 2, (900, 900)) * 255).numpy().astype("uint8")
        mask = Image.fromarray(arr, mode="L")
        transform = get_mask_transform(224)
        tensor = transform(mask)
        unique_vals = tensor.unique()
        assert all(v.item() in {0.0, 1.0} for v in unique_vals), (
            f"Mask contains non-binary values after transform: {unique_vals}"
        )

    def test_custom_image_size(self):
        from PIL import Image
        img = Image.new("RGB", (800, 800))
        for size in (256, 320):
            tensor = get_train_transform(size)(img)
            assert tensor.shape == torch.Size([3, size, size])


# ── Dataset tests (require downloaded data) ───────────────────────────────────

@REQUIRES_DATA
class TestMVTecDataset:
    def test_train_split_labels_all_zero(self):
        """Training set must contain only normal (label=0) images."""
        ds = MVTecDataset(DATA_ROOT, "bottle", split="train",
                          transform=get_train_transform())
        labels = [ds[i]["label"].item() for i in range(len(ds))]
        assert all(l == 0 for l in labels), "Train split contains anomalous labels."

    def test_test_split_contains_anomalies(self):
        """Test set must contain at least one anomalous image."""
        ds = MVTecDataset(DATA_ROOT, "bottle", split="test",
                          transform=get_test_transform(),
                          mask_transform=get_mask_transform())
        assert ds.anomaly_count > 0

    def test_sample_dict_keys(self):
        ds = MVTecDataset(DATA_ROOT, "bottle", split="test",
                          transform=get_test_transform(),
                          mask_transform=get_mask_transform())
        sample = ds[0]
        assert set(sample.keys()) == {"image", "label", "mask", "image_path", "defect_type"}

    def test_image_tensor_shape(self):
        ds = MVTecDataset(DATA_ROOT, "bottle", split="train",
                          transform=get_train_transform())
        assert ds[0]["image"].shape == torch.Size([3, 224, 224])

    def test_mask_tensor_shape(self):
        ds = MVTecDataset(DATA_ROOT, "bottle", split="test",
                          transform=get_test_transform(),
                          mask_transform=get_mask_transform())
        assert ds[0]["mask"].shape == torch.Size([1, 224, 224])

    def test_normal_test_image_has_zero_mask(self):
        """Normal test images must have all-zero masks."""
        ds = MVTecDataset(DATA_ROOT, "bottle", split="test",
                          transform=get_test_transform(),
                          mask_transform=get_mask_transform())
        for i in range(len(ds)):
            sample = ds[i]
            if sample["label"].item() == 0:
                assert sample["mask"].sum().item() == 0.0, (
                    f"Normal image has non-zero mask at index {i}"
                )

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Unknown category"):
            MVTecDataset(DATA_ROOT, "not_a_real_category")

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="split must be"):
            MVTecDataset(DATA_ROOT, "bottle", split="val")

    def test_repr(self):
        ds = MVTecDataset(DATA_ROOT, "bottle", split="train",
                          transform=get_train_transform())
        r = repr(ds)
        assert "bottle" in r and "train" in r

    @pytest.mark.parametrize("category", BENCHMARK_CATEGORIES)
    def test_benchmark_categories_load(self, category):
        """Smoke-test that each benchmark category dataset constructs without error."""
        ds = MVTecDataset(DATA_ROOT, category, split="train",
                          transform=get_train_transform())
        assert len(ds) > 0
