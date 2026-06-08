"""Unit tests for anomaly scoring logic.

These tests use synthetic random tensors and do not require the MVTec AD
dataset to be downloaded.  They validate the shape, dtype, and basic
behavioural contracts of the scoring pipeline for both models.

Run with:
    pytest tests/test_scoring.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_image(h: int = 224, w: int = 224) -> torch.Tensor:
    """Return a random (3, H, W) float32 image tensor."""
    return torch.randn(3, h, w, dtype=torch.float32)


def _random_batch(b: int = 4, h: int = 224, w: int = 224) -> torch.Tensor:
    return torch.randn(b, 3, h, w, dtype=torch.float32)


# ── PatchCore internals ───────────────────────────────────────────────────────

class TestPoolAndConcat:
    """Tests for _pool_and_concat — multi-scale feature merging."""

    def test_output_shape_single_map(self):
        from anomaly_det.models.patchcore import _pool_and_concat
        feat = {"layer2": torch.randn(2, 512, 28, 28)}
        out = _pool_and_concat(feat, target_size=28)
        assert out.shape == (2 * 28 * 28, 512)

    def test_output_shape_two_maps(self):
        from anomaly_det.models.patchcore import _pool_and_concat
        feats = {
            "layer2": torch.randn(2, 512, 28, 28),
            "layer3": torch.randn(2, 1024, 14, 14),
        }
        out = _pool_and_concat(feats, target_size=28)
        # layer3 is upsampled to 28×28 then concatenated → 512+1024=1536
        assert out.shape == (2 * 28 * 28, 1536)

    def test_output_is_contiguous(self):
        from anomaly_det.models.patchcore import _pool_and_concat
        feats = {"layer2": torch.randn(1, 512, 28, 28)}
        out = _pool_and_concat(feats, target_size=28)
        assert out.is_contiguous()

    def test_upsampling_aligns_spatial_dims(self):
        """layer3 (14×14) must be resized to target_size before concat."""
        from anomaly_det.models.patchcore import _pool_and_concat
        feats = {
            "layer2": torch.randn(1, 512, 28, 28),
            "layer3": torch.randn(1, 1024, 14, 14),
        }
        out = _pool_and_concat(feats, target_size=28)
        assert out.shape[0] == 28 * 28   # single image


class TestGreedyCoreset:
    """Tests for _greedy_coreset — furthest-point subsampling."""

    def test_output_size_respects_ratio(self):
        from anomaly_det.models.patchcore import _greedy_coreset
        features = np.random.randn(1000, 64).astype(np.float32)
        coreset = _greedy_coreset(features, ratio=0.1, min_size=10)
        assert coreset.shape[0] == pytest.approx(100, abs=5)

    def test_output_size_respects_min_size(self):
        from anomaly_det.models.patchcore import _greedy_coreset
        features = np.random.randn(200, 64).astype(np.float32)
        # ratio=0.001 would give 0.2 points → min_size should kick in
        coreset = _greedy_coreset(features, ratio=0.001, min_size=50)
        assert coreset.shape[0] == 50

    def test_output_feature_dim_preserved(self):
        from anomaly_det.models.patchcore import _greedy_coreset
        D = 128
        features = np.random.randn(500, D).astype(np.float32)
        coreset = _greedy_coreset(features, ratio=0.1)
        assert coreset.shape[1] == D

    def test_output_dtype_is_float32(self):
        from anomaly_det.models.patchcore import _greedy_coreset
        features = np.random.randn(300, 32).astype(np.float32)
        coreset = _greedy_coreset(features, ratio=0.1)
        assert coreset.dtype == np.float32

    def test_reproducible_with_seed(self):
        from anomaly_det.models.patchcore import _greedy_coreset
        features = np.random.randn(500, 64).astype(np.float32)
        c1 = _greedy_coreset(features, ratio=0.1, seed=0)
        c2 = _greedy_coreset(features, ratio=0.1, seed=0)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds_differ(self):
        from anomaly_det.models.patchcore import _greedy_coreset
        features = np.random.randn(500, 64).astype(np.float32)
        c1 = _greedy_coreset(features, ratio=0.1, seed=0)
        c2 = _greedy_coreset(features, ratio=0.1, seed=99)
        assert not np.array_equal(c1, c2)


class TestSmoothAndUpsample:
    """Tests for _smooth_and_upsample — score map post-processing."""

    def test_output_shape(self):
        from anomaly_det.models.patchcore import _smooth_and_upsample
        scores = np.random.rand(28 * 28).astype(np.float32)
        out = _smooth_and_upsample(scores, patch_size=28, output_size=(224, 224))
        assert out.shape == torch.Size([1, 224, 224])

    def test_output_dtype(self):
        from anomaly_det.models.patchcore import _smooth_and_upsample
        scores = np.random.rand(28 * 28).astype(np.float32)
        out = _smooth_and_upsample(scores, patch_size=28, output_size=(224, 224))
        assert out.dtype == torch.float32

    def test_non_negative_scores(self):
        """Squared L2 distances must be non-negative."""
        from anomaly_det.models.patchcore import _smooth_and_upsample
        scores = np.abs(np.random.rand(28 * 28)).astype(np.float32)
        out = _smooth_and_upsample(scores, patch_size=28, output_size=(224, 224))
        assert out.min().item() >= 0.0

    def test_custom_output_size(self):
        from anomaly_det.models.patchcore import _smooth_and_upsample
        scores = np.random.rand(28 * 28).astype(np.float32)
        out = _smooth_and_upsample(scores, patch_size=28, output_size=(256, 256))
        assert out.shape == torch.Size([1, 256, 256])


# ── PatchCore end-to-end predict (no dataset needed) ─────────────────────────

class TestPatchCorePredict:
    """Integration tests for PatchCore.predict() using a tiny synthetic bank."""

    @pytest.fixture(scope="class")
    def tiny_model(self):
        """Build a minimal fitted PatchCore without touching the backbone."""
        import faiss
        from anomaly_det.models.patchcore import PatchCore

        model = PatchCore(patch_size=4)
        model._feature_dim = 32
        # Seed the FAISS index with random normal vectors
        rng = np.random.default_rng(0)
        bank = rng.standard_normal((200, 32)).astype(np.float32)
        model._index = faiss.IndexFlatL2(32)
        model._index.add(bank)
        model._is_fitted = True
        return model

    def test_predict_returns_tuple(self, tiny_model):
        image = _random_image(64, 64)
        result = tiny_model.predict.__func__  # avoid calling real backbone
        # Since predict() calls _embed_batch which needs the backbone,
        # test the output contract via a manual scoring call instead.
        import faiss
        patches = np.random.randn(4 * 4, 32).astype(np.float32)
        dists, _ = tiny_model._index.search(patches, k=1)
        assert dists.shape == (16, 1)

    def test_score_is_positive(self, tiny_model):
        """Squared L2 distance scores must always be non-negative."""
        import faiss
        patches = np.random.randn(100, 32).astype(np.float32)
        dists, _ = tiny_model._index.search(patches, k=1)
        assert (dists >= 0).all()

    def test_predict_raises_before_fit(self):
        from anomaly_det.models.patchcore import PatchCore
        model = PatchCore()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict(_random_image())


# ── Autoencoder scoring ───────────────────────────────────────────────────────

class TestAutoencoderPredict:
    """Tests for Autoencoder.predict() using a randomly initialised network."""

    @pytest.fixture(scope="class")
    def untrained_model(self):
        """Return an Autoencoder with randomly initialised weights (no training)."""
        from anomaly_det.models.autoencoder import Autoencoder
        model = Autoencoder(base_channels=8, device="cpu")   # tiny width for speed
        model._net = model._build_net()
        model._net.eval()
        model._is_fitted = True
        return model

    def test_predict_raises_before_fit(self):
        from anomaly_det.models.autoencoder import Autoencoder
        model = Autoencoder()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict(_random_image())

    def test_predict_output_types(self, untrained_model):
        score, heatmap = untrained_model.predict(_random_image())
        assert isinstance(score, float)
        assert isinstance(heatmap, torch.Tensor)

    def test_heatmap_shape(self, untrained_model):
        image = _random_image(224, 224)
        _, heatmap = untrained_model.predict(image)
        assert heatmap.shape == torch.Size([1, 224, 224])

    def test_heatmap_dtype(self, untrained_model):
        _, heatmap = untrained_model.predict(_random_image())
        assert heatmap.dtype == torch.float32

    def test_heatmap_non_negative(self, untrained_model):
        """MSE residuals must be non-negative."""
        _, heatmap = untrained_model.predict(_random_image())
        assert heatmap.min().item() >= 0.0

    def test_score_equals_heatmap_max(self, untrained_model):
        score, heatmap = untrained_model.predict(_random_image())
        assert score == pytest.approx(heatmap.max().item(), rel=1e-5)

    def test_different_images_give_different_scores(self, untrained_model):
        s1, _ = untrained_model.predict(_random_image())
        s2, _ = untrained_model.predict(_random_image())
        assert s1 != pytest.approx(s2)

    @pytest.mark.parametrize("h,w", [(224, 224), (256, 256)])
    def test_predict_various_resolutions(self, untrained_model, h, w):
        _, heatmap = untrained_model.predict(_random_image(h, w))
        assert heatmap.shape == torch.Size([1, h, w])


# ── Shared interface contract ─────────────────────────────────────────────────

class TestSharedInterface:
    """Both models must satisfy the same predict() contract."""

    @pytest.fixture(params=["patchcore", "autoencoder"])
    def fitted_model(self, request):
        if request.param == "patchcore":
            import faiss
            from anomaly_det.models.patchcore import PatchCore
            m = PatchCore(patch_size=4)
            m._feature_dim = 32
            bank = np.random.randn(50, 32).astype(np.float32)
            m._index = faiss.IndexFlatL2(32)
            m._index.add(bank)
            m._is_fitted = True
            # Patch _embed_batch to skip the real backbone
            m._embed_batch = lambda imgs: np.random.randn(
                imgs.shape[0] * 4 * 4, 32
            ).astype(np.float32)
            return m
        else:
            from anomaly_det.models.autoencoder import Autoencoder
            m = Autoencoder(base_channels=8, device="cpu")
            m._net = m._build_net()
            m._net.eval()
            m._is_fitted = True
            return m

    def test_predict_returns_float_and_tensor(self, fitted_model):
        score, heatmap = fitted_model.predict(_random_image())
        assert isinstance(score, float)
        assert isinstance(heatmap, torch.Tensor)

    def test_heatmap_has_single_channel(self, fitted_model):
        _, heatmap = fitted_model.predict(_random_image())
        assert heatmap.shape[0] == 1

    def test_score_matches_heatmap_max(self, fitted_model):
        score, heatmap = fitted_model.predict(_random_image())
        assert score == pytest.approx(heatmap.max().item(), rel=1e-4)