"""Unit tests for TD (Temporal Dynamics) feature extraction and encoding."""

import pytest
import torch
import numpy as np
from spinlock.features.td import TDExtractor, TDConfig
from spinlock.encoding.encoders.td_cnn import TDCNNEncoder, ResidualBlock1D


class TestTDConfig:
    """Test TD configuration."""

    def test_default_config(self):
        """Test default TD configuration."""
        config = TDConfig()

        assert config.version == "1.0.0"
        assert config.enabled is True
        assert config.include_per_timestep is True
        assert config.include_derived_curves is True
        assert config.store_sequences is True
        assert config.store_context is True
        assert len(config.derived_features) == 3

    def test_custom_config(self):
        """Test custom TD configuration."""
        config = TDConfig(
            include_per_timestep=False,
            include_derived_curves=True,
            derived_features=["energy_trajectory"]
        )

        assert config.include_per_timestep is False
        assert len(config.derived_features) == 1


class TestTDExtractor:
    """Test TD feature extractor."""

    def test_extractor_initialization(self):
        """Test TD extractor initializes correctly."""
        config = TDConfig()
        extractor = TDExtractor(config=config, device=torch.device('cpu'))

        assert extractor.config == config
        assert extractor.registry.family == "td"

    def test_derived_curves_extraction(self):
        """Test derived temporal curve extraction."""
        config = TDConfig(
            include_per_timestep=False,
            include_derived_curves=True,
            derived_features=["energy_trajectory", "variance_trajectory"]
        )
        extractor = TDExtractor(config=config, device=torch.device('cpu'))

        # Create synthetic trajectories [N, M, T, C, H, W]
        N, M, T, C, H, W = 5, 3, 100, 3, 64, 64
        trajectories = torch.randn(N, M, T, C, H, W)

        # Extract
        result = extractor.extract_all(trajectories=trajectories)

        # Check shape
        sequences = result['sequences']
        assert sequences.shape == (N, M, T, 2), f"Expected (5, 3, 100, 2), got {sequences.shape}"

        # Check no NaNs
        assert not torch.isnan(sequences).any()

    def test_combined_extraction(self):
        """Test per-timestep + derived concatenation."""
        config = TDConfig(
            include_per_timestep=True,
            include_derived_curves=True,
            derived_features=["energy_trajectory"]
        )
        extractor = TDExtractor(config=config, device=torch.device('cpu'))

        # Create inputs
        N, M, T = 5, 3, 100
        trajectories = torch.randn(N, M, T, 3, 64, 64)
        per_timestep = torch.randn(N, M, T, 46)  # SDF per-timestep features

        # Extract
        result = extractor.extract_all(
            trajectories=trajectories,
            per_timestep_features=per_timestep
        )

        # Check shape
        sequences = result['sequences']
        assert sequences.shape == (N, M, T, 47), f"Expected (5, 3, 100, 47), got {sequences.shape}"

    def test_context_computation(self):
        """Test context feature computation."""
        config = TDConfig(
            include_per_timestep=False,
            include_derived_curves=True,
            derived_features=["energy_trajectory"],
            store_context=True
        )
        extractor = TDExtractor(config=config, device=torch.device('cpu'))

        # Create trajectories
        N, M, T = 5, 3, 100
        trajectories = torch.randn(N, M, T, 3, 64, 64)

        # Extract
        result = extractor.extract_all(trajectories=trajectories)

        # Check context shape
        context = result['context']
        assert context is not None
        assert context.shape == (N, 2), f"Expected (5, 2), got {context.shape}"  # 2 = mean + std

    def test_energy_trajectory_validity(self):
        """Test energy trajectory values are reasonable."""
        config = TDConfig(
            include_per_timestep=False,
            include_derived_curves=True,
            derived_features=["energy_trajectory"]
        )
        extractor = TDExtractor(config=config, device=torch.device('cpu'))

        N, M, T = 2, 1, 50
        trajectories = torch.ones(N, M, T, 3, 16, 16)  # All ones

        result = extractor.extract_all(trajectories=trajectories)
        energy = result['sequences']  # [N, M, T, 1]

        # Energy should be constant for constant field
        assert torch.allclose(energy[0, 0, 0], energy[0, 0, -1], atol=1e-5)

        # Energy should be positive
        assert (energy >= 0).all()


class TestResidualBlock1D:
    """Test 1D residual block."""

    def test_residual_block_forward(self):
        """Test 1D residual block forward pass."""
        block = ResidualBlock1D(in_channels=32, out_channels=64, stride=2)

        B, C, T = 8, 32, 100
        x = torch.randn(B, C, T)

        output = block(x)

        assert output.shape == (B, 64, 50), f"Expected (8, 64, 50), got {output.shape}"

    def test_residual_block_identity(self):
        """Test residual block with stride=1 (no downsampling)."""
        block = ResidualBlock1D(in_channels=64, out_channels=64, stride=1)

        B, C, T = 8, 64, 100
        x = torch.randn(B, C, T)

        output = block(x)

        assert output.shape == (B, 64, 100), f"Expected (8, 64, 100), got {output.shape}"


class TestTDCNNEncoder:
    """Test TD 1D CNN encoder."""

    def test_encoder_initialization(self):
        """Test TD CNN encoder initializes correctly."""
        encoder = TDCNNEncoder(input_dim=56, embedding_dim=64)

        assert encoder.output_dim == 64
        assert encoder._input_dim == 56

    def test_encoder_forward(self):
        """Test TD encoder forward pass with normal sequence."""
        encoder = TDCNNEncoder(input_dim=56, embedding_dim=64)

        B, T, D_in = 32, 500, 56
        x = torch.randn(B, T, D_in)

        embeddings = encoder(x)

        assert embeddings.shape == (B, 64), f"Expected (32, 64), got {embeddings.shape}"
        assert not torch.isnan(embeddings).any()

    def test_encoder_short_sequence(self):
        """Test TD encoder handles short sequences."""
        encoder = TDCNNEncoder(input_dim=56, embedding_dim=64)

        # Very short sequence (edge case)
        B, T, D_in = 8, 50, 56
        x = torch.randn(B, T, D_in)

        embeddings = encoder(x)
        assert embeddings.shape == (B, 64), f"Expected (8, 64), got {embeddings.shape}"

    def test_encoder_different_embedding_dims(self):
        """Test encoder with different output dimensions."""
        for embedding_dim in [32, 64, 128]:
            encoder = TDCNNEncoder(input_dim=56, embedding_dim=embedding_dim)

            B, T, D_in = 16, 500, 56
            x = torch.randn(B, T, D_in)

            embeddings = encoder(x)
            assert embeddings.shape == (B, embedding_dim)

    def test_encoder_gradient_flow(self):
        """Test gradients flow through encoder."""
        encoder = TDCNNEncoder(input_dim=56, embedding_dim=64)

        B, T, D_in = 16, 500, 56
        x = torch.randn(B, T, D_in, requires_grad=True)

        embeddings = encoder(x)
        loss = embeddings.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_encoder_deterministic(self):
        """Test encoder produces same output for same input."""
        encoder = TDCNNEncoder(input_dim=56, embedding_dim=64)
        encoder.eval()

        B, T, D_in = 8, 500, 56
        x = torch.randn(B, T, D_in)

        with torch.no_grad():
            output1 = encoder(x)
            output2 = encoder(x)

        assert torch.allclose(output1, output2)


class TestTDIntegration:
    """Integration tests for TD feature extraction."""

    def test_end_to_end_extraction(self):
        """Test complete TD extraction pipeline."""
        config = TDConfig(
            include_per_timestep=True,
            include_derived_curves=True,
            derived_features=["energy_trajectory", "variance_trajectory", "smoothness_trajectory"]
        )
        extractor = TDExtractor(config=config, device=torch.device('cpu'))

        # Create realistic test data
        N, M, T, C, H, W = 10, 3, 500, 3, 128, 128
        trajectories = torch.randn(N, M, T, C, H, W)
        per_timestep = torch.randn(N, M, T, 46)

        # Extract
        result = extractor.extract_all(
            trajectories=trajectories,
            per_timestep_features=per_timestep
        )

        # Validate output
        sequences = result['sequences']
        context = result['context']

        assert sequences.shape == (N, M, T, 49)  # 46 + 3 derived
        assert context.shape == (N, 98)  # 2 * 49
        assert not torch.isnan(sequences).any()
        assert not torch.isnan(context).any()

    def test_td_encoder_integration(self):
        """Test TD extraction → encoder pipeline."""
        # Extract TD features
        config = TDConfig(
            include_per_timestep=False,
            include_derived_curves=True,
            derived_features=["energy_trajectory", "variance_trajectory"]
        )
        extractor = TDExtractor(config=config, device=torch.device('cpu'))

        N, M, T = 8, 3, 500
        trajectories = torch.randn(N, M, T, 3, 128, 128)

        result = extractor.extract_all(trajectories=trajectories)
        sequences = result['sequences']  # [N, M, T, 2]

        # Aggregate across realizations
        sequences_agg = sequences.mean(dim=1)  # [N, T, 2]

        # Encode with TD CNN
        encoder = TDCNNEncoder(input_dim=2, embedding_dim=64)
        embeddings = encoder(sequences_agg)

        assert embeddings.shape == (N, 64)
        assert not torch.isnan(embeddings).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
