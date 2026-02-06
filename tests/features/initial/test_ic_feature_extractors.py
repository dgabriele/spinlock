"""
Unit tests for InitialConditionsFeatureExtractor.

Tests variance preservation, correct dimensions, and feature behavior on diverse inputs.
"""

import pytest
import torch
import numpy as np
from spinlock.features.initial.ic_feature_extractors import InitialConditionsFeatureExtractor


@pytest.fixture
def extractor():
    """Create IC feature extractor instance."""
    return InitialConditionsFeatureExtractor(device='cpu')


@pytest.fixture
def sample_ics():
    """Create diverse sample initial conditions."""
    torch.manual_seed(42)

    # High variance IC
    ic_high_var = torch.randn(2, 3, 64, 64) * 10

    # Low variance IC
    ic_low_var = torch.randn(2, 3, 64, 64) * 0.1

    # Zero IC
    ic_zero = torch.zeros(2, 3, 64, 64)

    # Structured IC (gradient)
    ic_structured = torch.zeros(2, 3, 64, 64)
    for i in range(64):
        ic_structured[:, :, i, :] = i / 64.0

    return {
        'high_var': ic_high_var,
        'low_var': ic_low_var,
        'zero': ic_zero,
        'structured': ic_structured
    }


class TestDimensions:
    """Test output dimensions for various input shapes."""

    def test_single_batch_3_channels(self, extractor):
        """Test [B, C, H, W] input with C=3."""
        ic = torch.randn(4, 3, 64, 64)
        features = extractor(ic)

        # For C=3: 30 (dist) + 24 (spatial) + 9 (energy) = 63
        assert features.shape == (4, 63), f"Expected (4, 63), got {features.shape}"

    def test_multi_realization_3_channels(self, extractor):
        """Test [B, M, C, H, W] input with C=3."""
        ic = torch.randn(2, 5, 3, 64, 64)
        features = extractor(ic)

        # Should handle multi-realization input
        assert features.shape == (2, 5, 63), f"Expected (2, 5, 63), got {features.shape}"

    def test_single_channel(self, extractor):
        """Test [B, C, H, W] input with C=1."""
        ic = torch.randn(4, 1, 64, 64)
        features = extractor(ic)

        # For C=1: 10 (dist) + 8 (spatial) + 2 (energy) + 0 (cross-corr) = 20
        assert features.shape == (4, 20), f"Expected (4, 20), got {features.shape}"

    def test_two_channels(self, extractor):
        """Test [B, C, H, W] input with C=2."""
        ic = torch.randn(4, 2, 64, 64)
        features = extractor(ic)

        # For C=2: 20 (dist) + 16 (spatial) + 4 (energy) + 1 (cross-corr) = 41
        assert features.shape == (4, 41), f"Expected (4, 41), got {features.shape}"

    def test_different_spatial_sizes(self, extractor):
        """Test different H, W dimensions."""
        ic_small = torch.randn(2, 3, 32, 32)
        ic_large = torch.randn(2, 3, 128, 128)

        features_small = extractor(ic_small)
        features_large = extractor(ic_large)

        # Output dimension should be same regardless of spatial size
        assert features_small.shape == features_large.shape == (2, 63)


class TestVariancePreservation:
    """Test that features preserve natural variance without collapsing."""

    def test_variance_not_collapsed(self, extractor, sample_ics):
        """Verify features don't collapse to ~0.00001 variance."""
        ic_high = sample_ics['high_var']
        ic_low = sample_ics['low_var']

        feat_high = extractor(ic_high)
        feat_low = extractor(ic_low)

        # Combine to compute variance
        combined = torch.cat([feat_high, feat_low], dim=0)
        feat_std = torch.std(combined, dim=0)

        # Check variance is NOT collapsed (not all features near zero variance)
        assert feat_std.mean() > 0.01, f"Features collapsed: mean std = {feat_std.mean()}"
        assert feat_std.max() > 0.1, f"No high-variance features: max std = {feat_std.max()}"

        # At least 50% of features should have reasonable variance
        reasonable_variance_count = (feat_std > 0.01).sum()
        assert reasonable_variance_count > len(feat_std) * 0.5, \
            f"Only {reasonable_variance_count}/{len(feat_std)} features have variance > 0.01"

    def test_high_variance_ic_produces_high_variance_features(self, extractor, sample_ics):
        """High-variance ICs should produce features with higher magnitude."""
        ic_high = sample_ics['high_var']
        ic_low = sample_ics['low_var']

        feat_high = extractor(ic_high)
        feat_low = extractor(ic_low)

        # Statistical features (mean, std, norms) should reflect IC variance
        # Check that some features have higher absolute values for high-var IC
        high_abs_mean = torch.abs(feat_high).mean()
        low_abs_mean = torch.abs(feat_low).mean()

        assert high_abs_mean > low_abs_mean, \
            f"High-var IC features ({high_abs_mean}) not larger than low-var ({low_abs_mean})"


class TestDistributional:
    """Test distributional feature extraction."""

    def test_basic_statistics(self, extractor):
        """Test mean, std, min, max, median."""
        # Create IC with known statistics
        ic = torch.zeros(1, 1, 64, 64)
        ic[0, 0, :, :] = torch.arange(64*64).float().view(64, 64)  # 0 to 4095

        features = extractor.extract_distributional(ic)

        # Features: mean, std, min, max, median, p05, p25, p75, p95, skew, kurt
        # For C=1: 10 features
        assert features.shape == (1, 10)

        # Check reasonable values (not all zeros or ones)
        assert torch.abs(features).max() > 1.0, "Features should have reasonable scale"

    def test_zero_ic(self, extractor, sample_ics):
        """Test on zero IC (edge case)."""
        ic = sample_ics['zero']
        features = extractor.extract_distributional(ic)

        # Should not crash and should return zeros/near-zeros for mean, std
        assert not torch.isnan(features).any(), "NaN in features for zero IC"
        assert not torch.isinf(features).any(), "Inf in features for zero IC"


class TestSpatial:
    """Test spatial feature extraction."""

    def test_gradients_on_structured_ic(self, extractor, sample_ics):
        """Test gradients on IC with known structure."""
        ic = sample_ics['structured']  # Linear gradient in y-direction

        features = extractor.extract_spatial(ic)

        # Should detect gradient (not all zeros)
        assert torch.abs(features).max() > 0.001, "Should detect gradient in structured IC"

    def test_center_of_mass_normalized(self, extractor):
        """Test center of mass is normalized to [0, 1]."""
        ic = torch.randn(2, 3, 64, 64)
        features = extractor.extract_spatial(ic)

        # CoM features are last 2 per channel (x, y)
        # For C=3: features are C*8 = 24 dims
        # CoM x, y are indices 6, 7 for each channel
        com_indices = []
        for c in range(3):
            com_indices.extend([c * 8 + 6, c * 8 + 7])

        com_values = features[:, com_indices]

        # Should be in [0, 1] range (normalized by H, W)
        assert torch.all(com_values >= -0.1), "CoM should be >= 0 (with small tolerance)"
        assert torch.all(com_values <= 1.1), "CoM should be <= 1 (with small tolerance)"

    def test_laplacian_computation(self, extractor):
        """Test Laplacian is computed correctly."""
        # Create IC with smooth gradient (Laplacian ~ 0) vs sharp edge (Laplacian != 0)
        ic_smooth = torch.zeros(1, 1, 64, 64)
        for i in range(64):
            ic_smooth[0, 0, i, :] = i  # Linear gradient

        ic_edge = torch.zeros(1, 1, 64, 64)
        ic_edge[0, 0, 30:34, :] = 10  # Sharp edge

        feat_smooth = extractor.extract_spatial(ic_smooth)
        feat_edge = extractor.extract_spatial(ic_edge)

        # Laplacian std (index 5 for C=1) should be higher for edge IC
        laplacian_std_smooth = feat_smooth[0, 5]
        laplacian_std_edge = feat_edge[0, 5]

        assert laplacian_std_edge > laplacian_std_smooth * 2, \
            "Laplacian should detect edges more strongly"


class TestEnergy:
    """Test energy/conservation feature extraction."""

    def test_norms_scaling(self, extractor):
        """Test L1 and L2 norms scale with IC magnitude."""
        ic_small = torch.randn(2, 3, 64, 64) * 0.1
        ic_large = torch.randn(2, 3, 64, 64) * 10.0

        feat_small = extractor.extract_energy(ic_small)
        feat_large = extractor.extract_energy(ic_large)

        # L2 and L1 norms are first 2 features per channel (indices 0, 1, 2, 3, 4, 5)
        norms_small = feat_small[:, :6].abs().mean()
        norms_large = feat_large[:, :6].abs().mean()

        assert norms_large > norms_small * 10, \
            f"Norms should scale with IC magnitude: {norms_large} vs {norms_small}"

    def test_cross_channel_correlation(self, extractor):
        """Test cross-channel correlation detection."""
        # Create ICs with known correlation
        ic_correlated = torch.randn(2, 3, 64, 64)
        ic_correlated[:, 1, :, :] = ic_correlated[:, 0, :, :] * 0.9  # Ch1 ~ Ch0

        ic_uncorrelated = torch.randn(2, 3, 64, 64)  # Independent channels

        feat_corr = extractor.extract_energy(ic_correlated)
        feat_uncorr = extractor.extract_energy(ic_uncorrelated)

        # Cross-channel correlations are last 3 features (indices 6, 7, 8)
        corr_values_corr = feat_corr[:, 6:9].abs().mean()
        corr_values_uncorr = feat_uncorr[:, 6:9].abs().mean()

        assert corr_values_corr > corr_values_uncorr, \
            "Should detect higher correlation in correlated IC"


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_no_nans(self, extractor, sample_ics):
        """Test that no NaNs are produced."""
        for name, ic in sample_ics.items():
            features = extractor(ic)
            assert not torch.isnan(features).any(), \
                f"NaN detected in features for {name} IC"

    def test_no_infs(self, extractor, sample_ics):
        """Test that no Infs are produced."""
        for name, ic in sample_ics.items():
            features = extractor(ic)
            assert not torch.isinf(features).any(), \
                f"Inf detected in features for {name} IC"

    def test_constant_ic(self, extractor):
        """Test on constant IC (all same value)."""
        ic = torch.ones(2, 3, 64, 64) * 5.0
        features = extractor(ic)

        # Should not crash, std should be ~0
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()

    def test_very_small_values(self, extractor):
        """Test on IC with very small values."""
        ic = torch.randn(2, 3, 64, 64) * 1e-8
        features = extractor(ic)

        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()

    def test_very_large_values(self, extractor):
        """Test on IC with very large values."""
        ic = torch.randn(2, 3, 64, 64) * 1e6
        features = extractor(ic)

        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()


class TestExtractAll:
    """Test the combined extract_all method."""

    def test_extract_all_combines_correctly(self, extractor):
        """Test that extract_all combines all feature categories."""
        ic = torch.randn(2, 3, 64, 64)

        features = extractor.extract_all(ic)

        # For C=3: 30 + 24 + 9 = 63
        assert features.shape == (2, 63)

        # Verify same as concatenating individual extractions
        dist = extractor.extract_distributional(ic)
        spatial = extractor.extract_spatial(ic)
        energy = extractor.extract_energy(ic)

        expected = torch.cat([dist, spatial, energy], dim=-1)

        assert torch.allclose(features, expected, atol=1e-6), \
            "extract_all should match concatenated individual extractions"


class TestBatchProcessing:
    """Test batch processing capabilities."""

    def test_single_vs_batch(self, extractor):
        """Test that batching gives same results as individual processing."""
        ic1 = torch.randn(1, 3, 64, 64)
        ic2 = torch.randn(1, 3, 64, 64)

        # Process individually
        feat1 = extractor(ic1)
        feat2 = extractor(ic2)

        # Process as batch
        ic_batch = torch.cat([ic1, ic2], dim=0)
        feat_batch = extractor(ic_batch)

        # Should match
        assert torch.allclose(feat_batch[0], feat1[0], atol=1e-6)
        assert torch.allclose(feat_batch[1], feat2[0], atol=1e-6)

    def test_large_batch(self, extractor):
        """Test on large batch size."""
        ic = torch.randn(128, 3, 64, 64)
        features = extractor(ic)

        assert features.shape == (128, 63)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()


class TestDeviceCompatibility:
    """Test device compatibility (CPU/CUDA)."""

    def test_cpu_device(self):
        """Test on CPU device."""
        extractor = InitialConditionsFeatureExtractor(device='cpu')
        ic = torch.randn(2, 3, 64, 64)
        features = extractor(ic)

        assert features.device.type == 'cpu'
        assert features.shape == (2, 63)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test on CUDA device."""
        extractor = InitialConditionsFeatureExtractor(device='cuda')
        ic = torch.randn(2, 3, 64, 64, device='cuda')
        features = extractor(ic)

        assert features.device.type == 'cuda'
        assert features.shape == (2, 63)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
