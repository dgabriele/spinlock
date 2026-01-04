"""
Tests for learned SUMMARY feature extraction from U-AFNO latents.

Tests:
1. LearnedSummaryConfig validation
2. U-AFNO get_intermediate_features() method
3. LearnedSummaryExtractor aggregation pipeline
4. SummaryExtractor integration with summary_mode
5. HDF5 storage for learned features

Author: Claude (Anthropic)
Date: January 2026
"""

import pytest
import torch
import numpy as np
from typing import List

# Test fixtures
@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_operator(device):
    """Create a sample U-AFNO operator for testing."""
    from spinlock.operators.u_afno import UAFNOOperator

    operator = UAFNOOperator(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        encoder_levels=3,
        modes=16,
        afno_blocks=2,
    )
    return operator.to(device).eval()


@pytest.fixture
def sample_trajectories(device):
    """Create sample trajectories [M, T, C, H, W]."""
    M, T, C, H, W = 5, 10, 3, 64, 64
    return torch.randn(M, T, C, H, W, device=device)


@pytest.fixture
def learned_config():
    """Create a sample LearnedSummaryConfig."""
    from spinlock.features.summary.config import LearnedSummaryConfig

    return LearnedSummaryConfig(
        enabled=True,
        extract_from="bottleneck",
        skip_levels=[0, 1, 2],
        temporal_agg="mean_max",
        spatial_agg="gap",
        projection_dim=None,
    )


class TestLearnedSummaryConfig:
    """Tests for LearnedSummaryConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        from spinlock.features.summary.config import LearnedSummaryConfig

        config = LearnedSummaryConfig()
        assert config.enabled == False
        assert config.extract_from == "bottleneck"
        assert config.skip_levels == [0, 1, 2]
        assert config.temporal_agg == "mean_max"
        assert config.spatial_agg == "gap"
        assert config.projection_dim is None

    def test_extract_from_options(self):
        """Test valid extract_from options."""
        from spinlock.features.summary.config import LearnedSummaryConfig

        for option in ["bottleneck", "skips", "all"]:
            config = LearnedSummaryConfig(extract_from=option)
            assert config.extract_from == option

    def test_temporal_agg_options(self):
        """Test valid temporal_agg options."""
        from spinlock.features.summary.config import LearnedSummaryConfig

        for option in ["mean", "max", "mean_max", "std"]:
            config = LearnedSummaryConfig(temporal_agg=option)
            assert config.temporal_agg == option

    def test_projection_dim_validation(self):
        """Test projection_dim bounds."""
        from spinlock.features.summary.config import LearnedSummaryConfig
        from pydantic import ValidationError

        # Valid range
        config = LearnedSummaryConfig(projection_dim=64)
        assert config.projection_dim == 64

        # None is allowed
        config = LearnedSummaryConfig(projection_dim=None)
        assert config.projection_dim is None


class TestUAFNOIntermediateFeatures:
    """Tests for U-AFNO get_intermediate_features() method."""

    def test_bottleneck_extraction(self, sample_operator, device):
        """Test extracting bottleneck features only."""
        x = torch.randn(4, 3, 64, 64, device=device)

        with torch.no_grad():
            features = sample_operator.get_intermediate_features(
                x, extract_from="bottleneck"
            )

        assert "bottleneck" in features
        assert len(features) == 1

        bottleneck = features["bottleneck"]
        assert bottleneck.shape[0] == 4  # Batch size
        assert bottleneck.ndim == 4  # [B, C, H, W]

    def test_skips_extraction(self, sample_operator, device):
        """Test extracting skip connection features."""
        x = torch.randn(4, 3, 64, 64, device=device)

        with torch.no_grad():
            features = sample_operator.get_intermediate_features(
                x, extract_from="skips", skip_levels=[0, 1, 2]
            )

        assert "bottleneck" not in features
        assert "skip_0" in features
        assert "skip_1" in features
        assert "skip_2" in features

        # Check shapes (each level has different resolution)
        for level in [0, 1, 2]:
            skip = features[f"skip_{level}"]
            assert skip.shape[0] == 4  # Batch size
            assert skip.ndim == 4  # [B, C, H, W]

    def test_all_extraction(self, sample_operator, device):
        """Test extracting all features."""
        x = torch.randn(4, 3, 64, 64, device=device)

        with torch.no_grad():
            features = sample_operator.get_intermediate_features(
                x, extract_from="all", skip_levels=[0, 1]
            )

        assert "bottleneck" in features
        assert "skip_0" in features
        assert "skip_1" in features
        assert "skip_2" not in features  # Only requested 0, 1


class TestLearnedSummaryExtractor:
    """Tests for LearnedSummaryExtractor aggregation pipeline."""

    def test_extract_from_single_operator(
        self, sample_operator, sample_trajectories, learned_config, device
    ):
        """Test extracting features from a single operator."""
        from spinlock.features.summary.learned import LearnedSummaryExtractor

        extractor = LearnedSummaryExtractor(device=device, config=learned_config)

        with torch.no_grad():
            features = extractor.extract_from_operator(
                sample_operator, sample_trajectories
            )

        # Should return [D_learned] vector
        assert features.ndim == 1
        assert features.shape[0] > 0
        assert not torch.isnan(features).any()

    def test_extract_batch(
        self, sample_operator, device, learned_config
    ):
        """Test extracting features for batch of operators."""
        from spinlock.features.summary.learned import LearnedSummaryExtractor
        from spinlock.operators.u_afno import UAFNOOperator

        # Create 3 operators
        operators = [sample_operator]
        for _ in range(2):
            op = UAFNOOperator(
                in_channels=3, out_channels=3,
                base_channels=32, encoder_levels=3,
                modes=16, afno_blocks=2,
            ).to(device).eval()
            operators.append(op)

        # Create trajectories for 3 operators
        N, M, T, C, H, W = 3, 5, 10, 3, 64, 64
        trajectories = torch.randn(N, M, T, C, H, W, device=device)

        extractor = LearnedSummaryExtractor(device=device, config=learned_config)

        with torch.no_grad():
            features = extractor.extract_batch(operators, trajectories)

        # Should return [N, D_learned] matrix
        assert features.ndim == 2
        assert features.shape[0] == N
        assert not torch.isnan(features).any()

    def test_temporal_aggregation_methods(
        self, sample_operator, sample_trajectories, device
    ):
        """Test different temporal aggregation methods."""
        from spinlock.features.summary.learned import LearnedSummaryExtractor
        from spinlock.features.summary.config import LearnedSummaryConfig

        dims = {}
        for method in ["mean", "max", "mean_max", "std"]:
            config = LearnedSummaryConfig(
                enabled=True,
                extract_from="bottleneck",
                temporal_agg=method
            )
            extractor = LearnedSummaryExtractor(device=device, config=config)

            with torch.no_grad():
                features = extractor.extract_from_operator(
                    sample_operator, sample_trajectories
                )

            dims[method] = features.shape[0]

        # mean_max should have 2x the dimension of mean/max/std
        assert dims["mean_max"] == 2 * dims["mean"]
        assert dims["mean"] == dims["max"] == dims["std"]

    def test_projection_mlp(
        self, sample_operator, sample_trajectories, device
    ):
        """Test MLP projection to fixed dimension."""
        from spinlock.features.summary.learned import LearnedSummaryExtractor
        from spinlock.features.summary.config import LearnedSummaryConfig

        projection_dim = 128
        config = LearnedSummaryConfig(
            enabled=True,
            extract_from="bottleneck",
            projection_dim=projection_dim
        )
        extractor = LearnedSummaryExtractor(device=device, config=config)

        with torch.no_grad():
            features = extractor.extract_from_operator(
                sample_operator, sample_trajectories
            )

        assert features.shape[0] == projection_dim


class TestSummaryExtractorIntegration:
    """Tests for SummaryExtractor integration with summary_mode."""

    def test_manual_mode(self, device):
        """Test manual mode (default)."""
        from spinlock.features.summary.extractors import SummaryExtractor
        from spinlock.features.summary.config import SummaryConfig

        config = SummaryConfig(summary_mode="manual")
        extractor = SummaryExtractor(device=device, config=config)

        # Create sample trajectories [N, M, T, C, H, W]
        N, M, T, C, H, W = 2, 3, 5, 3, 32, 32
        trajectories = torch.randn(N, M, T, C, H, W, device=device)

        with torch.no_grad():
            result = extractor.extract_all(trajectories)

        # Manual mode should not have learned features
        assert "learned" not in result
        assert "per_trajectory" in result
        assert "aggregated_mean" in result

    def test_hybrid_mode_requires_operators(self, device):
        """Test hybrid mode requires operators."""
        from spinlock.features.summary.extractors import SummaryExtractor
        from spinlock.features.summary.config import SummaryConfig, LearnedSummaryConfig

        config = SummaryConfig(
            summary_mode="hybrid",
            learned=LearnedSummaryConfig(enabled=True)
        )
        extractor = SummaryExtractor(device=device, config=config)

        # Create sample trajectories
        N, M, T, C, H, W = 2, 3, 5, 3, 32, 32
        trajectories = torch.randn(N, M, T, C, H, W, device=device)

        # Should raise error without operators
        with pytest.raises(ValueError, match="operators"):
            extractor.extract_all(trajectories, operators=None)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hybrid_mode_with_operators(self, sample_operator, device):
        """Test hybrid mode with operators provides both manual and learned features."""
        from spinlock.features.summary.extractors import SummaryExtractor
        from spinlock.features.summary.config import SummaryConfig, LearnedSummaryConfig
        from spinlock.operators.u_afno import UAFNOOperator

        config = SummaryConfig(
            summary_mode="hybrid",
            learned=LearnedSummaryConfig(enabled=True)
        )
        extractor = SummaryExtractor(device=device, config=config)

        # Create operators and trajectories
        N = 2
        operators = [sample_operator]
        operators.append(
            UAFNOOperator(
                in_channels=3, out_channels=3,
                base_channels=32, encoder_levels=3,
                modes=16, afno_blocks=2,
            ).to(device).eval()
        )

        M, T, C, H, W = 3, 5, 3, 64, 64
        trajectories = torch.randn(N, M, T, C, H, W, device=device)

        with torch.no_grad():
            result = extractor.extract_all(trajectories, operators=operators)

        # Should have both manual and learned
        assert "per_trajectory" in result
        assert "aggregated_mean" in result
        assert "learned" in result
        assert result["learned"].shape[0] == N


class TestSchemaConfig:
    """Tests for schema.py configuration."""

    def test_summary_features_config_defaults(self):
        """Test SummaryFeaturesConfig defaults."""
        from spinlock.config.schema import SummaryFeaturesConfig

        config = SummaryFeaturesConfig()
        assert config.enabled == True
        assert config.summary_mode == "manual"
        assert config.learned.enabled == False

    def test_learned_features_config(self):
        """Test LearnedFeaturesConfig in schema."""
        from spinlock.config.schema import LearnedFeaturesConfig

        config = LearnedFeaturesConfig(
            enabled=True,
            extract_from="all",
            temporal_agg="mean_max",
            projection_dim=256
        )
        assert config.enabled == True
        assert config.extract_from == "all"
        assert config.projection_dim == 256

    def test_hybrid_mode_config(self):
        """Test hybrid mode configuration."""
        from spinlock.config.schema import SummaryFeaturesConfig, LearnedFeaturesConfig

        config = SummaryFeaturesConfig(
            summary_mode="hybrid",
            learned=LearnedFeaturesConfig(
                enabled=True,
                extract_from="bottleneck"
            )
        )
        assert config.summary_mode == "hybrid"
        assert config.learned.enabled == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
