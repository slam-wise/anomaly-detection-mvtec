"""Image transforms for MVTec AD.

All transforms target a 224×224 output to match ImageNet-pretrained backbones
(WideResNet-50 used by PatchCore). Three variants are provided:

    get_train_transform   — used during PatchCore memory-bank construction
    get_test_transform    — used during evaluation (identical to train for PatchCore)
    get_mask_transform    — ground-truth binary masks; no normalisation, NEAREST resize

ImageNet normalisation statistics are applied to image transforms so that
pretrained backbone features are in the expected input distribution.
"""

from __future__ import annotations

from torchvision import transforms

# ── ImageNet statistics ───────────────────────────────────────────────────────
# Standard mean/std used when WideResNet-50 (and most torchvision models) were
# pre-trained on ImageNet. Apply to all image transforms.
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Return the standard image transform used during memory-bank construction.

    Pipeline:
        1. Resize shortest edge to *image_size* (bicubic, preserves aspect ratio).
        2. Centre-crop to *image_size* × *image_size*.
        3. Convert PIL Image → float32 tensor in [0, 1].
        4. Normalise with ImageNet mean/std.

    Args:
        image_size: Target spatial resolution. 224 is the standard for
            ImageNet-pretrained backbones. Increase to 256 or 320 if GPU
            memory allows — larger inputs produce denser patch grids and
            generally improve pixel-level AUROC.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_test_transform(image_size: int = 224) -> transforms.Compose:
    """Return the image transform used during evaluation.

    Identical to :func:`get_train_transform` — PatchCore performs no
    data augmentation at any stage, so train and test transforms are the same.
    Keeping them as separate functions makes the pipeline explicit and leaves
    a clean extension point if augmentation-based methods are added later.

    Args:
        image_size: Target spatial resolution. Must match the value used
            during memory-bank construction.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return get_train_transform(image_size)


def get_mask_transform(image_size: int = 224) -> transforms.Compose:
    """Return the transform applied to ground-truth anomaly masks.

    Pipeline:
        1. Resize to *image_size* using NEAREST interpolation (preserves
           binary values — bicubic would create grey-valued border artefacts).
        2. Centre-crop to *image_size* × *image_size*.
        3. Convert PIL Image → float32 tensor; pixel values become 0.0 or ~1.0.

    No normalisation is applied. Downstream code should threshold at > 0
    to recover the binary mask used for pixel-level AUROC computation.

    Args:
        image_size: Target spatial resolution. Must match the image transforms.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )