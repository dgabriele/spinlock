"""
Unit tests for inverse decoder models.

Tests ThetaInverseMLP and InitialInverseCNN architectures.
"""

import pytest
import torch

from spinlock.tokens.inverse_models import ThetaInverseMLP, InitialInverseCNN


class TestThetaInverseMLP:
    """Tests for ThetaInverseMLP."""

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        model = ThetaInverseMLP(encoded_dim=32, param_dim=14)
        x = torch.randn(10, 32)

        out = model(x)

        assert out.shape == (10, 14), f"Expected (10, 14), got {out.shape}"

    def test_output_range(self):
        """Test output is in [0, 1] range due to sigmoid."""
        model = ThetaInverseMLP(encoded_dim=32, param_dim=14)
        x = torch.randn(10, 32) * 10  # Large values

        out = model(x)

        assert torch.all(out >= 0.0), "Output contains values < 0"
        assert torch.all(out <= 1.0), "Output contains values > 1"

    def test_different_dimensions(self):
        """Test with different encoder/parameter dimensions."""
        model = ThetaInverseMLP(encoded_dim=64, param_dim=20)
        x = torch.randn(5, 64)

        out = model(x)

        assert out.shape == (5, 20)

    def test_batch_size_1(self):
        """Test with batch size 1."""
        model = ThetaInverseMLP(encoded_dim=32, param_dim=14)
        x = torch.randn(1, 32)

        out = model(x)

        assert out.shape == (1, 14)

    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = ThetaInverseMLP(encoded_dim=32, param_dim=14)
        x = torch.randn(10, 32, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert torch.all(torch.isfinite(x.grad)), "Gradient contains NaN/Inf"


class TestInitialInverseCNN:
    """Tests for InitialInverseCNN."""

    def test_forward_shape_64(self):
        """Test forward pass for 64x64 output."""
        model = InitialInverseCNN(encoded_dim=426, channels=3, spatial_size=64)
        x = torch.randn(10, 426)

        out = model(x)

        assert out.shape == (10, 3, 64, 64), f"Expected (10, 3, 64, 64), got {out.shape}"

    def test_forward_shape_32(self):
        """Test forward pass for 32x32 output."""
        model = InitialInverseCNN(encoded_dim=426, channels=3, spatial_size=32)
        x = torch.randn(10, 426)

        out = model(x)

        assert out.shape == (10, 3, 32, 32), f"Expected (10, 3, 32, 32), got {out.shape}"

    def test_single_channel(self):
        """Test with single channel output."""
        model = InitialInverseCNN(encoded_dim=426, channels=1, spatial_size=64)
        x = torch.randn(10, 426)

        out = model(x)

        assert out.shape == (10, 1, 64, 64)

    def test_different_encoded_dim(self):
        """Test with different encoded dimension."""
        model = InitialInverseCNN(encoded_dim=512, channels=3, spatial_size=64)
        x = torch.randn(5, 512)

        out = model(x)

        assert out.shape == (5, 3, 64, 64)

    def test_batch_size_1(self):
        """Test with batch size 1."""
        model = InitialInverseCNN(encoded_dim=426, channels=3, spatial_size=64)
        x = torch.randn(1, 426)

        out = model(x)

        assert out.shape == (1, 3, 64, 64)

    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = InitialInverseCNN(encoded_dim=426, channels=3, spatial_size=64)
        x = torch.randn(10, 426, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert torch.all(torch.isfinite(x.grad)), "Gradient contains NaN/Inf"

    def test_invalid_spatial_size(self):
        """Test invalid spatial size raises error."""
        with pytest.raises(AssertionError):
            InitialInverseCNN(encoded_dim=426, channels=3, spatial_size=48)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
