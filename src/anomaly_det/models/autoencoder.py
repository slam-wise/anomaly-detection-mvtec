"""Convolutional autoencoder for reconstruction-based anomaly detection.

Trains on defect-free images only. At inference, anomalies produce high
per-pixel reconstruction error because the model has only learned to
reconstruct normal appearance.

Architecture
------------
A symmetric encoder-decoder with 5 stride-2 convolutional blocks.
Input 224×224 is compressed to a 7×7×512 spatial bottleneck — 32× smaller
in each spatial dimension — forcing the model to discard local detail.
Normal textures can be recovered from this compressed representation;
anomalous local deviations cannot.

    Encoder  :  (3,224,224) → (64,112,112) → (128,56,56) → (256,28,28)
                            → (512,14,14)  → (512,7,7)   ← bottleneck
    Decoder  :  (512,7,7)  → (512,14,14)  → (256,28,28) → (128,56,56)
                            → (64,112,112) → (3,224,224)

Anomaly scoring
---------------
Per-pixel MSE between input and reconstruction is computed channel-wise,
averaged to a single (H, W) map, then Gaussian-smoothed.  The image-level
score is the maximum pixel score — consistent with PatchCore's convention.

Interface
---------
Mirrors :class:`~anomaly_det.models.PatchCore` exactly so that both models
can be passed to the same evaluation functions without modification:

    model.fit(dataloader)               → trains the autoencoder
    score, heatmap = model.predict(img) → reconstruction-error scoring
    model.save(path) / model.load(path) → persistence
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# ── Building blocks ───────────────────────────────────────────────────────────

class _EncoderBlock(nn.Module):
    """Conv(stride=2) + BatchNorm + ReLU — halves spatial dimensions."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    """ConvTranspose(stride=2) + BatchNorm + ReLU — doubles spatial dimensions."""

    def __init__(self, in_ch: int, out_ch: int, activation: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size=3, stride=2, padding=1, output_padding=1,
                bias=False,
            ),
        ]
        if activation:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ── Encoder / Decoder ─────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    """Five-block encoder: (3,224,224) → (512,7,7)."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        ch = base_channels
        self.blocks = nn.Sequential(
            _EncoderBlock(3, ch),           # → (ch,  112, 112)
            _EncoderBlock(ch, ch * 2),      # → (ch*2, 56,  56)
            _EncoderBlock(ch * 2, ch * 4),  # → (ch*4, 28,  28)
            _EncoderBlock(ch * 4, ch * 8),  # → (ch*8, 14,  14)
            _EncoderBlock(ch * 8, ch * 8),  # → (ch*8,  7,   7)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class _Decoder(nn.Module):
    """Five-block decoder: (512,7,7) → (3,224,224).

    No activation on the final layer — input images are ImageNet-normalised
    and live roughly in [-2.5, 2.5], outside the range of Sigmoid or Tanh.
    """

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        ch = base_channels
        self.blocks = nn.Sequential(
            _DecoderBlock(ch * 8, ch * 8),      # → (ch*8, 14,  14)
            _DecoderBlock(ch * 8, ch * 4),      # → (ch*4, 28,  28)
            _DecoderBlock(ch * 4, ch * 2),      # → (ch*2, 56,  56)
            _DecoderBlock(ch * 2, ch),          # → (ch,  112, 112)
            _DecoderBlock(ch, 3, activation=False),  # → (3,  224, 224)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


# ── Autoencoder ───────────────────────────────────────────────────────────────

class Autoencoder:
    """Reconstruction-based anomaly detector.

    Trains a convolutional autoencoder on defect-free images. At inference,
    the per-pixel MSE between input and reconstruction serves as the anomaly
    heatmap; high error → likely anomaly.

    This model serves as an interpretable baseline for PatchCore: the residual
    map is computed entirely in pixel space, making the "why is this flagged?"
    question directly answerable by inspecting the reconstruction.

    Args:
        base_channels: Channel multiplier controlling model width.
            Default 64 gives ~12M parameters; increase for more capacity.
        smooth_sigma: Gaussian blur std-dev applied to the residual map.
        device: ``'cuda'``, ``'cpu'``, or ``'auto'``.

    Example::

        from torch.utils.data import DataLoader
        from anomaly_det.data import MVTecDataset, get_train_transform
        from anomaly_det.models.autoencoder import Autoencoder

        model = Autoencoder()
        train_dl = DataLoader(
            MVTecDataset("data/mvtec", "bottle", "train",
                         transform=get_train_transform()),
            batch_size=32,
        )
        history = model.fit(train_dl, epochs=200)

        score, heatmap = model.predict(test_image_tensor)
    """

    def __init__(
        self,
        base_channels: int = 64,
        smooth_sigma: float = 4.0,
        device: str = "auto",
    ) -> None:
        self.base_channels = base_channels
        self.smooth_sigma = smooth_sigma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._net: Optional[nn.Module] = None
        self._is_fitted: bool = False
        self._train_history: dict = {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_net(self) -> nn.Module:
        net = nn.Sequential(
            _Encoder(self.base_channels),
            _Decoder(self.base_channels),
        )
        return net.to(self.device)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int = 200,
        lr: float = 2e-4,
        val_split: float = 0.1,
        patience: int = 20,
    ) -> dict:
        """Train the autoencoder on defect-free images.

        Holds out a random *val_split* fraction of training images for early
        stopping.  Saves the best checkpoint (lowest validation loss) and
        restores it at the end of training.

        Args:
            dataloader: Yields dicts with an ``'image'`` key or ``(image, ...)``
                tuples.  Should contain *only* normal training images.
            epochs: Maximum training epochs.
            lr: Initial learning rate for Adam.
            val_split: Fraction of training images to use for validation.
            patience: Early-stopping patience in epochs.

        Returns:
            History dict with keys ``'train_loss'`` and ``'val_loss'``
            (lists of per-epoch mean losses).
        """
        print(f"[Autoencoder] Training on {self.device}  "
              f"(epochs={epochs}, lr={lr}, patience={patience})")

        self._net = self._build_net()

        # ── Train / val split ─────────────────────────────────────────────────
        dataset = dataloader.dataset
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
        )
        print(f"  Train images : {n_train}   Val images : {n_val}")

        # ── Optimiser + scheduler ─────────────────────────────────────────────
        optimiser = torch.optim.Adam(
            self._net.parameters(), lr=lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=lr * 0.01
        )
        criterion = nn.MSELoss()

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_loss = float("inf")
        best_state: dict = {}
        epochs_no_improve = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        epoch_bar = tqdm(range(1, epochs + 1), desc="  Training", unit="epoch")
        for epoch in epoch_bar:
            # ── Train step ────────────────────────────────────────────────────
            self._net.train()
            train_loss = 0.0
            for batch in train_dl:
                images = batch["image"] if isinstance(batch, dict) else batch[0]
                images = images.to(self.device)
                recon = self._net(images)
                loss = criterion(recon, images)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * len(images)
            train_loss /= n_train

            # ── Validation step ───────────────────────────────────────────────
            self._net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    images = batch["image"] if isinstance(batch, dict) else batch[0]
                    images = images.to(self.device)
                    recon = self._net(images)
                    val_loss += criterion(recon, images).item() * len(images)
            val_loss /= n_val

            scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            epoch_bar.set_postfix(
                train=f"{train_loss:.5f}", val=f"{val_loss:.5f}"
            )

            # ── Early stopping ────────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n  Early stop at epoch {epoch} "
                          f"(no improvement for {patience} epochs).")
                    break

        # Restore best weights
        self._net.load_state_dict(
            {k: v.to(self.device) for k, v in best_state.items()}
        )
        self._net.eval()
        self._is_fitted = True

        self._train_history = {"train_loss": train_losses, "val_loss": val_losses}
        print(f"[Autoencoder] Training complete — best val loss: {best_val_loss:.6f}\n")
        return self._train_history

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        """Score a single image and produce a reconstruction-error heatmap.

        Args:
            image: ``(3, H, W)`` normalised image tensor (no batch dimension).

        Returns:
            A tuple ``(score, heatmap)`` where:

            *   ``score`` is a float — image-level anomaly score (max pixel error).
            *   ``heatmap`` is a ``(1, H, W)`` float32 tensor — per-pixel MSE
                between input and reconstruction, Gaussian-smoothed.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        self._net.eval()
        x = image.unsqueeze(0).to(self.device)   # (1, 3, H, W)
        recon = self._net(x).squeeze(0)           # (3, H, W)
        image_cpu = image.cpu()

        # Per-pixel MSE averaged over channels → (H, W)
        residual = (image_cpu - recon.cpu()).pow(2).mean(dim=0).numpy()

        # Gaussian smoothing
        smoothed = gaussian_filter(residual, sigma=self.smooth_sigma)
        heatmap = torch.from_numpy(smoothed).unsqueeze(0).float()  # (1, H, W)

        return float(heatmap.max()), heatmap

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model weights and config to *path* directory."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / "weights.pt")
        torch.save(
            {
                "base_channels": self.base_channels,
                "smooth_sigma": self.smooth_sigma,
                "train_history": self._train_history,
            },
            path / "config.pt",
        )
        print(f"[Autoencoder] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Autoencoder":
        """Load a previously saved autoencoder from *path* directory."""
        path = Path(path)
        config = torch.load(path / "config.pt", weights_only=False)
        model = cls(
            base_channels=config["base_channels"],
            smooth_sigma=config["smooth_sigma"],
        )
        model._net = model._build_net()
        model._net.load_state_dict(
            torch.load(path / "weights.pt", map_location=model.device, weights_only=True)
        )
        model._net.eval()
        model._is_fitted = True
        model._train_history = config.get("train_history", {})
        print(f"[Autoencoder] Loaded from {path}")
        return model

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_params = (
            sum(p.numel() for p in self._net.parameters())
            if self._net is not None else 0
        )
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"Autoencoder("
            f"base_channels={self.base_channels}, "
            f"params={n_params:,}, "
            f"device={self.device}, "
            f"status={status!r}"
            f")"
        )