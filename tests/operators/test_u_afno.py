"""Tests for U-AFNO neural operator architecture.

Tests for:
1. SpectralMixingBlock: FFT-based spectral mixing
2. AFNOBlock: Full AFNO block with spectral + feedforward
3. UNetEncoder/Decoder: Multi-scale encoding/decoding
4. UAFNOOperator: Complete U-AFNO architecture
5. OperatorBuilder: U-AFNO factory methods
"""

import torch
import pytest
import numpy as np

from spinlock.operators.afno import SpectralMixingBlock, AFNOBlock
from spinlock.operators.u_afno import UNetEncoder, UNetDecoder, UAFNOOperator
from spinlock.operators.builder import OperatorBuilder


# =============================================================================
# SpectralMixingBlock Tests
# =============================================================================


class TestSpectralMixingBlock:
    """Test SpectralMixingBlock FFT-based spectral mixing."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        block = SpectralMixingBlock(64, 64, modes=16)
        x = torch.randn(4, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_output_shape_channel_change(self):
        """Test output shape when changing channels."""
        block = SpectralMixingBlock(64, 128, modes=16, use_residual=False)
        x = torch.randn(4, 64, 32, 32)
        out = block(x)
        assert out.shape == (4, 128, 32, 32)

    def test_mode_truncation(self):
        """Test that mode truncation handles small grid sizes."""
        block = SpectralMixingBlock(32, 32, modes=64)  # modes > grid size
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        assert out.shape == x.shape  # Should work without error

    def test_gradient_flow(self):
        """Test that gradients flow through FFT operations."""
        block = SpectralMixingBlock(32, 32, modes=8)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_residual_connection(self):
        """Test that residual connection works."""
        # With residual
        block_res = SpectralMixingBlock(32, 32, modes=8, use_residual=True)
        # Without residual
        block_no_res = SpectralMixingBlock(32, 32, modes=8, use_residual=False)

        x = torch.randn(2, 32, 16, 16)
        out_res = block_res(x)
        out_no_res = block_no_res(x)

        # Outputs should differ
        assert not torch.allclose(out_res, out_no_res)


# =============================================================================
# AFNOBlock Tests
# =============================================================================


class TestAFNOBlock:
    """Test AFNOBlock (spectral + feedforward with pre-norm)."""

    def test_output_shape(self):
        """Test that output shape equals input shape."""
        block = AFNOBlock(64, modes=16, mlp_ratio=4.0)
        x = torch.randn(4, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connections(self):
        """Test that residual connections are present."""
        block = AFNOBlock(32, modes=8)
        x = torch.ones(1, 32, 8, 8)
        out = block(x)
        # With residual connections, output should not be zero
        # (unless spectral and MLP outputs cancel out, which is unlikely)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_gradient_flow(self):
        """Test gradient flow through block."""
        block = AFNOBlock(32, modes=8)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_dropout(self):
        """Test that dropout parameter works."""
        block = AFNOBlock(32, modes=8, dropout=0.5)
        x = torch.randn(4, 32, 16, 16)

        block.train()
        out_train1 = block(x)
        out_train2 = block(x)
        # Due to dropout, outputs should differ
        assert not torch.allclose(out_train1, out_train2, atol=1e-6)

        block.eval()
        out_eval1 = block(x)
        out_eval2 = block(x)
        # In eval mode, outputs should be identical
        assert torch.allclose(out_eval1, out_eval2)


# =============================================================================
# UNetEncoder Tests
# =============================================================================


class TestUNetEncoder:
    """Test UNetEncoder multi-scale downsampling."""

    def test_output_shape(self):
        """Test bottleneck and skip shapes."""
        encoder = UNetEncoder(3, base_channels=32, num_levels=3)
        x = torch.randn(2, 3, 64, 64)
        bottleneck, skips = encoder(x)

        # Bottleneck should be downsampled 2^3 = 8x
        assert bottleneck.shape == (2, encoder.out_channels, 8, 8)

        # Should have 3 skips (one per level before downsample)
        assert len(skips) == 3

    def test_skip_channels(self):
        """Test that skip_channels are tracked correctly."""
        encoder = UNetEncoder(3, base_channels=32, num_levels=3)
        assert len(encoder.skip_channels) == 3

    def test_various_input_sizes(self):
        """Test encoder works with various input sizes."""
        encoder = UNetEncoder(3, base_channels=16, num_levels=2)

        for size in [32, 64, 128]:
            x = torch.randn(1, 3, size, size)
            bottleneck, skips = encoder(x)
            expected_size = size // (2 ** 2)  # 2 levels of downsampling
            assert bottleneck.shape[-1] == expected_size


# =============================================================================
# UNetDecoder Tests
# =============================================================================


class TestUNetDecoder:
    """Test UNetDecoder with skip connections."""

    def test_output_shape(self):
        """Test decoder restores original resolution."""
        encoder = UNetEncoder(3, base_channels=32, num_levels=3)
        decoder = UNetDecoder(
            in_channels=encoder.out_channels,
            out_channels=3,
            skip_channels=encoder.skip_channels,
        )

        x = torch.randn(2, 3, 64, 64)
        bottleneck, skips = encoder(x)
        out = decoder(bottleneck, skips)

        # Output should have same spatial size as input (and 3 channels)
        assert out.shape == (2, 3, 64, 64)

    def test_skip_fusion(self):
        """Test that skip connections are used."""
        encoder = UNetEncoder(3, base_channels=16, num_levels=2)
        decoder = UNetDecoder(
            in_channels=encoder.out_channels,
            out_channels=3,
            skip_channels=encoder.skip_channels,
        )

        x = torch.randn(1, 3, 32, 32)
        bottleneck, skips = encoder(x)

        # Zero out skips
        zero_skips = [torch.zeros_like(s) for s in skips]

        out_normal = decoder(bottleneck, skips)
        out_zero_skips = decoder(bottleneck, zero_skips)

        # Outputs should differ if skips are used
        assert not torch.allclose(out_normal, out_zero_skips)


# =============================================================================
# UAFNOOperator Tests
# =============================================================================


class TestUAFNOOperator:
    """Test complete UAFNOOperator architecture."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
        )
        x = torch.randn(2, 3, 32, 32)
        out = op(x)
        assert out.shape == x.shape

    def test_various_grid_sizes(self):
        """Test with various grid sizes."""
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
        )

        for size in [32, 64, 128]:
            x = torch.randn(1, 3, size, size)
            out = op(x)
            assert out.shape == x.shape

    def test_stochastic_noise(self):
        """Test stochastic noise injection."""
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            noise_type="gaussian",
            noise_scale=0.1,
        )

        x = torch.randn(2, 3, 32, 32)
        torch.manual_seed(42)
        out1 = op(x)
        torch.manual_seed(123)
        out2 = op(x)

        # Outputs should differ due to stochastic noise
        assert not torch.allclose(out1, out2)

    def test_no_noise_deterministic(self):
        """Test that without noise, outputs are deterministic."""
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            # No noise
        )
        op.eval()

        x = torch.randn(1, 3, 32, 32)
        out1 = op(x)
        out2 = op(x)

        assert torch.allclose(out1, out2)

    def test_gradient_flow(self):
        """Test gradient flow through entire architecture."""
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
        )

        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = op(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_get_config(self):
        """Test config serialization."""
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=32,
            encoder_levels=3,
            modes=16,
            afno_blocks=4,
        )

        config = op.get_config()

        assert config["in_channels"] == 3
        assert config["out_channels"] == 3
        assert config["base_channels"] == 32
        assert config["encoder_levels"] == 3
        assert config["modes"] == 16
        assert config["afno_blocks"] == 4

    def test_from_config(self):
        """Test reconstruction from config."""
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 32,
            "encoder_levels": 3,
            "modes": 16,
            "afno_blocks": 4,
        }

        op = UAFNOOperator.from_config(config)

        assert op.config["base_channels"] == 32
        assert op.config["modes"] == 16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self):
        """Test that operator works on GPU."""
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
        ).cuda()

        x = torch.randn(2, 3, 32, 32).cuda()
        out = op(x)

        assert out.device.type == "cuda"
        assert out.shape == x.shape


# =============================================================================
# OperatorBuilder Tests
# =============================================================================


class TestOperatorBuilderUAFNO:
    """Test OperatorBuilder U-AFNO factory methods."""

    def test_build_u_afno_basic(self):
        """Test basic U-AFNO building."""
        builder = OperatorBuilder()

        params = {
            "input_channels": 3,
            "output_channels": 3,
            "base_channels": 32,
            "encoder_levels": 3,
            "modes": 16,
            "afno_blocks": 4,
        }

        model = builder.build_u_afno(params)

        # Verify it's a UAFNOOperator
        assert isinstance(model, UAFNOOperator)

        # Verify forward pass works
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == x.shape

    def test_build_u_afno_with_noise(self):
        """Test U-AFNO building with noise parameters."""
        builder = OperatorBuilder()

        params = {
            "input_channels": 3,
            "output_channels": 3,
            "base_channels": 16,
            "encoder_levels": 2,
            "modes": 8,
            "afno_blocks": 2,
            "noise_type": "gaussian",
            "noise_scale": 0.05,
        }

        model = builder.build_u_afno(params)
        assert model.stochastic is not None

    def test_build_u_afno_defaults(self):
        """Test that defaults work."""
        builder = OperatorBuilder()

        params = {
            "input_channels": 3,
            "output_channels": 3,
            # Use defaults for everything else
        }

        model = builder.build_u_afno(params)
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == x.shape

    def test_build_from_sampled_params_u_afno(self):
        """Test build_from_sampled_params with operator_type='u_afno'."""
        builder = OperatorBuilder()

        # Minimal spec for testing
        spec = {
            "base_channels": {"type": "integer", "bounds": [16, 64]},
            "input_channels": {"type": "integer", "bounds": [3, 3]},
            "output_channels": {"type": "integer", "bounds": [3, 3]},
            "num_layers": {"type": "integer", "bounds": [2, 4]},
            "kernel_size": {"type": "integer", "bounds": [3, 3]},
            "activation": {"type": "choice", "choices": ["gelu"]},
            "normalization": {"type": "choice", "choices": ["instance"]},
            "modes": {"type": "integer", "bounds": [8, 32]},
            "encoder_levels": {"type": "integer", "bounds": [2, 4]},
            "afno_blocks": {"type": "integer", "bounds": [2, 4]},
        }

        # Sample parameters
        unit_params = np.array([0.5] * len(spec))

        # Build U-AFNO
        model = builder.build_from_sampled_params(
            unit_params, spec, operator_type="u_afno"
        )

        assert isinstance(model, UAFNOOperator)

    def test_build_from_sampled_params_cnn(self):
        """Test build_from_sampled_params with default operator_type='cnn'."""
        builder = OperatorBuilder()

        spec = {
            "num_layers": {"type": "integer", "bounds": [2, 4]},
            "base_channels": {"type": "integer", "bounds": [16, 32]},
            "input_channels": {"type": "integer", "bounds": [3, 3]},
            "output_channels": {"type": "integer", "bounds": [3, 3]},
            "kernel_size": {"type": "integer", "bounds": [3, 3]},
            "activation": {"type": "choice", "choices": ["gelu"]},
            "normalization": {"type": "choice", "choices": ["instance"]},
        }

        unit_params = np.array([0.5] * len(spec))

        # Default operator_type should be "cnn"
        model = builder.build_from_sampled_params(unit_params, spec)

        # Should be Sequential (CNN), not UAFNOOperator
        assert not isinstance(model, UAFNOOperator)
        assert isinstance(model, torch.nn.Sequential)

    def test_block_registry_includes_afno(self):
        """Test that BLOCK_REGISTRY includes AFNO classes."""
        assert "SpectralMixingBlock" in OperatorBuilder.BLOCK_REGISTRY
        assert "AFNOBlock" in OperatorBuilder.BLOCK_REGISTRY
        assert "UAFNOOperator" in OperatorBuilder.BLOCK_REGISTRY


# =============================================================================
# Integration Tests
# =============================================================================


class TestUAFNOIntegration:
    """Integration tests for U-AFNO with rollout engine."""

    def test_temporal_rollout_compatible(self):
        """Test that U-AFNO works with temporal rollout (multiple timesteps)."""
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
        )
        op.eval()

        # Simulate autoregressive rollout
        x = torch.randn(1, 3, 32, 32)
        trajectory = [x]

        for _ in range(5):  # 5 timesteps
            x = op(x)
            trajectory.append(x)

        assert len(trajectory) == 6  # Initial + 5 steps
        assert all(t.shape == trajectory[0].shape for t in trajectory)

    def test_stochastic_realizations(self):
        """Test generating multiple stochastic realizations."""
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            noise_type="gaussian",
            noise_scale=0.1,
        )

        x = torch.randn(1, 3, 32, 32)
        realizations = []

        for i in range(5):
            torch.manual_seed(i)
            out = op(x)
            realizations.append(out)

        # Realizations should be different
        for i in range(4):
            assert not torch.allclose(realizations[i], realizations[i + 1])
