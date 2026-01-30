"""Tests for the hierarchical temporal pyramid encoder."""

import pytest
import torch

from spinlock.encoding.temporal_pyramid import TemporalPyramid
from spinlock.encoding.encoders.pyramid_temporal import PyramidTemporalEncoder


class TestTemporalPyramid:
    """Tests for TemporalPyramid downsampling."""

    def test_default_factors(self):
        pyramid = TemporalPyramid()
        assert pyramid.downsample_factors == [1, 2, 4, 8]

    def test_custom_factors(self):
        pyramid = TemporalPyramid([1, 3, 9])
        assert pyramid.downsample_factors == [1, 3, 9]

    def test_empty_factors_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            TemporalPyramid([])

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            TemporalPyramid([1, 0])

    @pytest.mark.parametrize("T", [64, 128, 256, 512])
    def test_output_lengths(self, T):
        """Verify each level has the expected temporal length."""
        pyramid = TemporalPyramid([1, 2, 4, 8])
        x = torch.randn(4, T, 32)
        levels = pyramid(x)

        assert len(levels) == 4
        assert levels[0].shape == (4, T, 32)        # factor 1: full
        assert levels[1].shape == (4, T // 2, 32)   # factor 2: half
        assert levels[2].shape == (4, T // 4, 32)   # factor 4: quarter
        assert levels[3].shape == (4, T // 8, 32)   # factor 8: eighth

    def test_preserves_batch_and_feature_dims(self):
        pyramid = TemporalPyramid([1, 4])
        x = torch.randn(8, 100, 64)
        levels = pyramid(x)

        for level in levels:
            assert level.shape[0] == 8   # batch
            assert level.shape[2] == 64  # features

    def test_factor_1_is_identity(self):
        """Factor 1 should return the original tensor."""
        pyramid = TemporalPyramid([1])
        x = torch.randn(2, 50, 16)
        levels = pyramid(x)
        assert torch.equal(levels[0], x)

    def test_small_T_min_length_1(self):
        """Even very small T should produce at least length 1."""
        pyramid = TemporalPyramid([1, 2, 4, 8])
        x = torch.randn(2, 4, 16)
        levels = pyramid(x)

        assert levels[0].shape[1] == 4   # T/1
        assert levels[1].shape[1] == 2   # T/2
        assert levels[2].shape[1] == 1   # T/4
        # T/8 = 0, but clamped to 1
        assert levels[3].shape[1] >= 1


class TestPyramidTemporalEncoder:
    """Tests for PyramidTemporalEncoder."""

    @pytest.fixture
    def encoder(self):
        return PyramidTemporalEncoder(
            input_dim=64,
            level_dims=[32, 64, 96, 128],
            downsample_factors=[1, 2, 4, 8],
        )

    def test_output_shape(self, encoder):
        x = torch.randn(8, 256, 64)
        out = encoder(x)
        assert out.shape == (8, 320)  # 32 + 64 + 96 + 128

    def test_output_dim_property(self, encoder):
        assert encoder.output_dim == 320

    def test_output_dims_per_level(self, encoder):
        assert encoder.output_dims_per_level == [32, 64, 96, 128]

    def test_forward_per_level(self, encoder):
        x = torch.randn(4, 128, 64)
        levels = encoder.forward_per_level(x)

        assert len(levels) == 4
        assert levels[0].shape == (4, 32)
        assert levels[1].shape == (4, 64)
        assert levels[2].shape == (4, 96)
        assert levels[3].shape == (4, 128)

    def test_forward_matches_per_level_concat(self, encoder):
        """forward() should equal concat of forward_per_level()."""
        encoder.eval()
        x = torch.randn(4, 128, 64)
        with torch.no_grad():
            concat_out = encoder(x)
            per_level = encoder.forward_per_level(x)
            manual_concat = torch.cat(per_level, dim=1)
        assert torch.allclose(concat_out, manual_concat, atol=1e-5)

    @pytest.mark.parametrize("input_dim", [32, 64, 128, 345])
    def test_variable_input_dims(self, input_dim):
        """Should work with different input feature dimensions."""
        enc = PyramidTemporalEncoder(
            input_dim=input_dim,
            level_dims=[16, 32],
            downsample_factors=[1, 2],
        )
        x = torch.randn(4, 100, input_dim)
        out = enc(x)
        assert out.shape == (4, 48)  # 16 + 32

    @pytest.mark.parametrize("T", [32, 64, 128, 256, 500])
    def test_variable_sequence_lengths(self, T):
        """Should work with different temporal lengths."""
        enc = PyramidTemporalEncoder(
            input_dim=64,
            level_dims=[32, 64],
            downsample_factors=[1, 4],
        )
        x = torch.randn(2, T, 64)
        out = enc(x)
        assert out.shape == (2, 96)

    def test_mismatched_dims_factors_raises(self):
        with pytest.raises(ValueError, match="must match"):
            PyramidTemporalEncoder(
                input_dim=64,
                level_dims=[32, 64, 96],
                downsample_factors=[1, 2],
            )

    def test_invalid_architecture_raises(self):
        with pytest.raises(ValueError, match="resnet1d_3"):
            PyramidTemporalEncoder(
                input_dim=64,
                architecture="invalid",
            )

    def test_gradient_flows(self, encoder):
        """Verify gradients flow through all levels."""
        x = torch.randn(4, 128, 64, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_is_base_encoder(self, encoder):
        """Should be a proper BaseEncoder subclass."""
        from spinlock.encoding.encoders.base import BaseEncoder
        assert isinstance(encoder, BaseEncoder)


class TestEncoderRegistry:
    """Tests for encoder registry integration."""

    def test_registered_by_name(self):
        from spinlock.encoding.encoders import get_encoder
        encoder = get_encoder("pyramid_temporal", input_dim=64)
        assert isinstance(encoder, PyramidTemporalEncoder)

    def test_registered_by_class_name(self):
        from spinlock.encoding.encoders import get_encoder
        encoder = get_encoder("PyramidTemporalEncoder", input_dim=64)
        assert isinstance(encoder, PyramidTemporalEncoder)

    def test_with_custom_params(self):
        from spinlock.encoding.encoders import get_encoder
        encoder = get_encoder(
            "pyramid_temporal",
            input_dim=128,
            level_dims=[16, 32, 48],
            downsample_factors=[1, 2, 4],
        )
        assert encoder.output_dim == 96
        assert encoder.output_dims_per_level == [16, 32, 48]
