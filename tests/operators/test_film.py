"""Tests for FiLM (Feature-wise Linear Modulation) implementation.

Tests for:
1. FiLMConfig: Configuration dataclass
2. FiLMLayer: Feature-wise modulation layer
3. FiLMProjectionHead: Embedding to gamma/beta projection
4. FiLMGenerator: Multi-layer FiLM parameter generation
5. FiLMResidualBlock: Residual block with FiLM support
6. FiLMAFNOBlock: AFNO block with post-spectral FiLM
7. FiLMUAFNOOperator: Complete U-AFNO with FiLM
"""

import torch
import pytest
from typing import Dict

from spinlock.operators.film import (
    FiLMConfig,
    FiLMLayerSpec,
    FiLMLayer,
    FiLMProjectionHead,
    FiLMGenerator,
    build_film_layer_specs,
    estimate_film_parameter_overhead,
)
from spinlock.operators.blocks import FiLMResidualBlock
from spinlock.operators.afno import FiLMAFNOBlock
from spinlock.operators.u_afno import (
    FiLMUNetEncoder,
    FiLMUNetDecoder,
    FiLMUAFNOOperator,
)


# =============================================================================
# FiLMConfig Tests
# =============================================================================


class TestFiLMConfig:
    """Test FiLMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FiLMConfig()
        assert config.enabled is False
        assert config.embed_dim == 64
        assert config.hidden_dim == 128
        assert config.init_gamma == 1.0
        assert config.init_beta == 0.0
        assert config.post_norm is True
        assert config.modulate_encoder is True
        assert config.modulate_decoder is True
        assert config.modulate_afno_post is False

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "enabled": True,
            "embed_dim": 128,
            "hidden_dim": 256,
            "post_norm": False,
            "modulate_afno_post": True,
            "afno_blocks": [2, 3],
        }
        config = FiLMConfig.from_dict(config_dict)
        assert config.enabled is True
        assert config.embed_dim == 128
        assert config.hidden_dim == 256
        assert config.post_norm is False
        assert config.modulate_afno_post is True
        assert config.afno_blocks == [2, 3]

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = FiLMConfig(
            enabled=True,
            embed_dim=64,
            modulate_encoder=True,
            modulate_decoder=False,
        )
        d = config.to_dict()
        assert d["enabled"] is True
        assert d["embed_dim"] == 64
        assert d["modulate_encoder"] is True
        assert d["modulate_decoder"] is False

    def test_validate_warns_on_afno(self):
        """Test that validate() warns when AFNO modulation is enabled."""
        config = FiLMConfig(
            enabled=True,
            modulate_afno_post=True,
            _warn_afno=True,
        )
        with pytest.warns(UserWarning, match="AFNO post-spectral modulation"):
            config.validate()


# =============================================================================
# FiLMLayer Tests
# =============================================================================


class TestFiLMLayer:
    """Test FiLMLayer modulation."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        layer = FiLMLayer(64, post_norm=True)
        x = torch.randn(4, 64, 16, 16)
        gamma = torch.ones(4, 64)
        beta = torch.zeros(4, 64)
        out = layer(x, gamma, beta)
        assert out.shape == x.shape

    def test_identity_transformation(self):
        """Test that gamma=1, beta=0 is (nearly) identity."""
        layer = FiLMLayer(32, post_norm=False)  # No norm for exact identity
        x = torch.randn(2, 32, 8, 8)
        gamma = torch.ones(2, 32)
        beta = torch.zeros(2, 32)
        out = layer(x, gamma, beta)
        assert torch.allclose(out, x, atol=1e-6)

    def test_scaling(self):
        """Test that gamma scales features correctly."""
        layer = FiLMLayer(16, post_norm=False)
        x = torch.ones(1, 16, 4, 4)
        gamma = torch.full((1, 16), 2.0)
        beta = torch.zeros(1, 16)
        out = layer(x, gamma, beta)
        assert torch.allclose(out, x * 2.0, atol=1e-6)

    def test_shifting(self):
        """Test that beta shifts features correctly."""
        layer = FiLMLayer(16, post_norm=False)
        x = torch.ones(1, 16, 4, 4)
        gamma = torch.ones(1, 16)
        beta = torch.full((1, 16), 3.0)
        out = layer(x, gamma, beta)
        assert torch.allclose(out, x + 3.0, atol=1e-6)

    def test_post_norm_enabled(self):
        """Test that post_norm normalizes output."""
        layer = FiLMLayer(32, post_norm=True)
        x = torch.randn(2, 32, 8, 8)
        gamma = torch.ones(2, 32) * 10  # Large scale
        beta = torch.zeros(2, 32)
        out = layer(x, gamma, beta)
        # Post-norm should constrain output magnitude
        assert out.std() < x.std() * 10

    def test_gradient_flow(self):
        """Test gradient flow through FiLM layer."""
        layer = FiLMLayer(16, post_norm=True)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        gamma = torch.ones(2, 16, requires_grad=True)
        beta = torch.zeros(2, 16, requires_grad=True)
        out = layer(x, gamma, beta)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert gamma.grad is not None
        assert beta.grad is not None


# =============================================================================
# FiLMProjectionHead Tests
# =============================================================================


class TestFiLMProjectionHead:
    """Test FiLMProjectionHead."""

    def test_output_shape(self):
        """Test gamma/beta output shapes."""
        head = FiLMProjectionHead(64, 128, 256)
        embedding = torch.randn(8, 64)
        gamma, beta = head(embedding)
        assert gamma.shape == (8, 256)
        assert beta.shape == (8, 256)

    def test_identity_init(self):
        """Test that initial gamma=1, beta=0."""
        head = FiLMProjectionHead(64, 128, 32, init_gamma=1.0, init_beta=0.0)
        embedding = torch.zeros(1, 64)  # Zero input
        gamma, beta = head(embedding)
        assert torch.allclose(gamma, torch.ones(1, 32), atol=1e-6)
        assert torch.allclose(beta, torch.zeros(1, 32), atol=1e-6)

    def test_custom_init(self):
        """Test custom init_gamma/init_beta values."""
        head = FiLMProjectionHead(64, 128, 32, init_gamma=0.5, init_beta=0.1)
        embedding = torch.zeros(1, 64)
        gamma, beta = head(embedding)
        assert torch.allclose(gamma, torch.full((1, 32), 0.5), atol=1e-6)
        assert torch.allclose(beta, torch.full((1, 32), 0.1), atol=1e-6)

    def test_reinit_identity(self):
        """Test re-initialization to identity."""
        head = FiLMProjectionHead(64, 128, 32)
        # Modify weights
        with torch.no_grad():
            for p in head.parameters():
                p.fill_(1.0)
        # Re-init
        head.reinit_identity()
        # Check identity output
        embedding = torch.zeros(1, 64)
        gamma, beta = head(embedding)
        assert torch.allclose(gamma, torch.ones(1, 32), atol=1e-6)
        assert torch.allclose(beta, torch.zeros(1, 32), atol=1e-6)


# =============================================================================
# FiLMGenerator Tests
# =============================================================================


class TestFiLMGenerator:
    """Test FiLMGenerator for multi-layer FiLM parameter generation."""

    def test_generates_all_layers(self):
        """Test that generator produces params for all specified layers."""
        specs = [
            FiLMLayerSpec("encoder_0", 64, "encoder", 0),
            FiLMLayerSpec("encoder_1", 128, "encoder", 1),
            FiLMLayerSpec("decoder_0", 128, "decoder", 0),
        ]
        generator = FiLMGenerator(specs, embed_dim=64, hidden_dim=128)
        embedding = torch.randn(4, 64)
        film_params = generator(embedding)

        assert "encoder_0" in film_params
        assert "encoder_1" in film_params
        assert "decoder_0" in film_params

        gamma_0, beta_0 = film_params["encoder_0"]
        assert gamma_0.shape == (4, 64)
        assert beta_0.shape == (4, 64)

        gamma_1, beta_1 = film_params["encoder_1"]
        assert gamma_1.shape == (4, 128)
        assert beta_1.shape == (4, 128)

    def test_identity_init_all_layers(self):
        """Test that all layers initialize to identity."""
        specs = [
            FiLMLayerSpec("layer_0", 32, "encoder", 0),
            FiLMLayerSpec("layer_1", 64, "encoder", 1),
        ]
        generator = FiLMGenerator(specs, embed_dim=64, hidden_dim=128)
        embedding = torch.zeros(1, 64)
        film_params = generator(embedding)

        for name, (gamma, beta) in film_params.items():
            assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-6), f"{name} gamma not 1"
            assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6), f"{name} beta not 0"

    def test_get_layer_names(self):
        """Test get_layer_names returns all layer names."""
        specs = [
            FiLMLayerSpec("a", 32, "encoder", 0),
            FiLMLayerSpec("b", 64, "encoder", 1),
        ]
        generator = FiLMGenerator(specs, embed_dim=64, hidden_dim=128)
        names = generator.get_layer_names()
        assert names == ["a", "b"]

    def test_count_parameters(self):
        """Test parameter counting."""
        specs = [FiLMLayerSpec("layer_0", 64, "encoder", 0)]
        generator = FiLMGenerator(specs, embed_dim=64, hidden_dim=128)
        count = generator.count_parameters()
        assert count > 0


# =============================================================================
# build_film_layer_specs Tests
# =============================================================================


class TestBuildFilmLayerSpecs:
    """Test build_film_layer_specs function."""

    def test_encoder_only(self):
        """Test building specs for encoder only."""
        config = FiLMConfig(
            enabled=True,
            modulate_encoder=True,
            modulate_decoder=False,
            modulate_afno_post=False,
        )
        specs = build_film_layer_specs(
            config,
            encoder_channels=[64, 128, 256],
            decoder_channels=[256, 128, 64],
            afno_channels=256,
            num_afno_blocks=4,
        )
        names = [s.name for s in specs]
        assert "encoder_0" in names
        assert "encoder_1" in names
        assert "encoder_2" in names
        assert all("decoder" not in n for n in names)
        assert all("afno" not in n for n in names)

    def test_specific_levels(self):
        """Test building specs for specific levels."""
        config = FiLMConfig(
            enabled=True,
            modulate_encoder=True,
            modulate_decoder=True,
            encoder_levels=[0, 2],
            decoder_levels=[1],
        )
        specs = build_film_layer_specs(
            config,
            encoder_channels=[64, 128, 256],
            decoder_channels=[256, 128, 64],
            afno_channels=256,
            num_afno_blocks=4,
        )
        names = [s.name for s in specs]
        assert "encoder_0" in names
        assert "encoder_1" not in names
        assert "encoder_2" in names
        assert "decoder_0" not in names
        assert "decoder_1" in names
        assert "decoder_2" not in names

    def test_afno_blocks(self):
        """Test AFNO block spec generation."""
        config = FiLMConfig(
            enabled=True,
            modulate_encoder=False,
            modulate_decoder=False,
            modulate_afno_post=True,
            afno_blocks=[1, 3],
        )
        specs = build_film_layer_specs(
            config,
            encoder_channels=[64, 128, 256],
            decoder_channels=[256, 128, 64],
            afno_channels=256,
            num_afno_blocks=4,
        )
        names = [s.name for s in specs]
        assert "afno_post_1" in names
        assert "afno_post_3" in names
        assert "afno_post_0" not in names
        assert "afno_post_2" not in names


# =============================================================================
# FiLMResidualBlock Tests
# =============================================================================


class TestFiLMResidualBlock:
    """Test FiLMResidualBlock."""

    def test_output_shape(self):
        """Test output shape with channel change."""
        block = FiLMResidualBlock(32, 64, post_norm=True)
        x = torch.randn(2, 32, 16, 16)
        gamma = torch.ones(2, 64)
        beta = torch.zeros(2, 64)
        out = block(x, gamma=gamma, beta=beta)
        assert out.shape == (2, 64, 16, 16)

    def test_without_film(self):
        """Test that block works without FiLM parameters."""
        block = FiLMResidualBlock(32, 64)
        x = torch.randn(2, 32, 16, 16)
        out = block(x)  # No gamma/beta
        assert out.shape == (2, 64, 16, 16)

    def test_film_affects_output(self):
        """Test that different gamma/beta produce different outputs."""
        block = FiLMResidualBlock(32, 64, post_norm=False)
        block.eval()
        x = torch.randn(2, 32, 16, 16)

        gamma1 = torch.ones(2, 64)
        beta1 = torch.zeros(2, 64)
        out1 = block(x, gamma=gamma1, beta=beta1)

        gamma2 = torch.ones(2, 64) * 2
        beta2 = torch.ones(2, 64)
        out2 = block(x, gamma=gamma2, beta=beta2)

        assert not torch.allclose(out1, out2)

    def test_get_film_channels(self):
        """Test get_film_channels returns correct value."""
        block = FiLMResidualBlock(32, 128)
        assert block.get_film_channels() == 128


# =============================================================================
# FiLMAFNOBlock Tests
# =============================================================================


class TestFiLMAFNOBlock:
    """Test FiLMAFNOBlock with post-spectral FiLM."""

    def test_output_shape(self):
        """Test output shape preservation."""
        block = FiLMAFNOBlock(64, modes=8, post_norm=True)
        x = torch.randn(2, 64, 16, 16)
        gamma = torch.ones(2, 64)
        beta = torch.zeros(2, 64)
        out = block(x, gamma=gamma, beta=beta)
        assert out.shape == x.shape

    def test_without_film(self):
        """Test that block works without FiLM parameters."""
        block = FiLMAFNOBlock(64, modes=8)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)  # No gamma/beta
        assert out.shape == x.shape

    def test_film_affects_output(self):
        """Test that different FiLM params produce different outputs."""
        block = FiLMAFNOBlock(32, modes=8, post_norm=False)
        block.eval()
        x = torch.randn(1, 32, 16, 16)

        out_no_film = block(x)
        out_with_film = block(x, gamma=torch.ones(1, 32) * 2, beta=torch.ones(1, 32))

        assert not torch.allclose(out_no_film, out_with_film)

    def test_gradient_flow(self):
        """Test gradient flow through FiLM AFNO block."""
        block = FiLMAFNOBlock(32, modes=8)
        x = torch.randn(1, 32, 16, 16, requires_grad=True)
        gamma = torch.ones(1, 32, requires_grad=True)
        beta = torch.zeros(1, 32, requires_grad=True)
        out = block(x, gamma=gamma, beta=beta)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert gamma.grad is not None


# =============================================================================
# FiLMUAFNOOperator Tests
# =============================================================================


class TestFiLMUAFNOOperator:
    """Test complete FiLMUAFNOOperator."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        film_config = {
            "enabled": True,
            "embed_dim": 64,
            "hidden_dim": 128,
            "post_norm": True,
            "modulate_encoder": True,
            "modulate_decoder": True,
            "modulate_afno_post": False,
        }
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )
        x = torch.randn(2, 1, 32, 32)
        conditioning = torch.randn(2, 64)
        out = op(x, conditioning=conditioning)
        assert out.shape == x.shape

    def test_without_conditioning(self):
        """Test that operator works without conditioning."""
        film_config = {"enabled": True, "embed_dim": 64}
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )
        x = torch.randn(2, 1, 32, 32)
        out = op(x)  # No conditioning
        assert out.shape == x.shape

    def test_conditioning_affects_output(self):
        """Test that different conditioning produces different outputs.

        Note: FiLM is identity-initialized (gamma=1, beta=0), so we need to
        perturb the weights slightly to see different outputs from different
        conditioning inputs.
        """
        film_config = {
            "enabled": True,
            "embed_dim": 64,
            "modulate_encoder": True,
            "modulate_decoder": True,
        }
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )

        # Perturb FiLM generator weights so different conditioning produces different output
        # (At identity init, all conditioning produces the same gamma=1, beta=0)
        with torch.no_grad():
            for name, param in op.film_generator.named_parameters():
                if "weight" in name:
                    param.add_(torch.randn_like(param) * 0.1)

        op.eval()

        x = torch.randn(1, 1, 32, 32)
        cond1 = torch.randn(1, 64)
        cond2 = torch.randn(1, 64) * 10  # Different conditioning

        out1 = op(x, conditioning=cond1)
        out2 = op(x, conditioning=cond2)

        assert not torch.allclose(out1, out2)

    def test_film_disabled(self):
        """Test operator with FiLM disabled."""
        film_config = {"enabled": False}
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )
        assert op.film_generator is None
        x = torch.randn(2, 1, 32, 32)
        out = op(x)
        assert out.shape == x.shape

    def test_afno_post_modulation(self):
        """Test with AFNO post-spectral modulation enabled."""
        film_config = {
            "enabled": True,
            "embed_dim": 64,
            "modulate_encoder": True,
            "modulate_decoder": True,
            "modulate_afno_post": True,
            "afno_blocks": [0, 1],  # Modulate first two blocks
            "_warn_afno": False,  # Suppress warning in test
        }
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )
        x = torch.randn(1, 1, 32, 32)
        conditioning = torch.randn(1, 64)
        out = op(x, conditioning=conditioning)
        assert out.shape == x.shape

    def test_get_config(self):
        """Test configuration retrieval."""
        film_config = {"enabled": True, "embed_dim": 64, "hidden_dim": 128}
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            encoder_levels=3,
            modes=16,
            afno_blocks=4,
            film_config=film_config,
        )
        config = op.get_config()
        assert config["in_channels"] == 1
        assert config["base_channels"] == 32
        assert config["film_config"]["enabled"] is True

    def test_reinit_film_layers(self):
        """Test re-initialization of FiLM layers."""
        film_config = {"enabled": True, "embed_dim": 64}
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )
        op.reinit_film_layers()
        # Should not raise

    def test_get_film_parameter_count(self):
        """Test FiLM parameter counting."""
        film_config = {"enabled": True, "embed_dim": 64, "hidden_dim": 128}
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )
        count = op.get_film_parameter_count()
        assert count > 0

    def test_gradient_flow(self):
        """Test gradient flow through entire architecture."""
        film_config = {"enabled": True, "embed_dim": 64}
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )
        x = torch.randn(1, 1, 32, 32, requires_grad=True)
        conditioning = torch.randn(1, 64, requires_grad=True)
        out = op(x, conditioning=conditioning)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert conditioning.grad is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestFiLMIntegration:
    """Integration tests for FiLM with NOA system."""

    def test_from_config_roundtrip(self):
        """Test creating operator from config and back."""
        film_config = {
            "enabled": True,
            "embed_dim": 64,
            "hidden_dim": 128,
            "modulate_encoder": True,
            "modulate_decoder": True,
        }
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )
        config = op.get_config()
        op2 = FiLMUAFNOOperator.from_config(config)

        # Compare forward pass
        x = torch.randn(1, 1, 32, 32)
        cond = torch.randn(1, 64)

        op.eval()
        op2.eval()

        # Copy weights for deterministic comparison
        op2.load_state_dict(op.state_dict())

        out1 = op(x, conditioning=cond)
        out2 = op2(x, conditioning=cond)
        assert torch.allclose(out1, out2)

    def test_temporal_rollout_with_film(self):
        """Test autoregressive rollout with FiLM conditioning."""
        film_config = {"enabled": True, "embed_dim": 64}
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        )
        op.eval()

        x = torch.randn(1, 1, 32, 32)
        conditioning = torch.randn(1, 64)
        trajectory = [x]

        for _ in range(5):
            x = op(x, conditioning=conditioning)
            trajectory.append(x)

        assert len(trajectory) == 6
        assert all(t.shape == trajectory[0].shape for t in trajectory)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self):
        """Test FiLM operator on GPU."""
        film_config = {"enabled": True, "embed_dim": 64}
        op = FiLMUAFNOOperator(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
            film_config=film_config,
        ).cuda()

        x = torch.randn(1, 1, 32, 32).cuda()
        conditioning = torch.randn(1, 64).cuda()
        out = op(x, conditioning=conditioning)

        assert out.device.type == "cuda"
        assert out.shape == x.shape
