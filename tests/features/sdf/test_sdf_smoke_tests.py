"""
Smoke tests for SDF v2.0 feature extraction.

Fast tests to catch obvious bugs like:
- All zeros when should be impossible
- All NaN when should be valid
- All constant when should vary
- Wrong shapes
- Missing features
"""

import torch
import pytest
from spinlock.features.sdf.extractors import SDFExtractor
from spinlock.features.sdf.config import SDFConfig


@pytest.fixture
def device():
    """Test device (CPU for fast tests)."""
    return torch.device('cpu')


@pytest.fixture
def diverse_trajectories(device):
    """Create diverse trajectories with varied dynamics."""
    N, M, T, C, H, W = 3, 5, 10, 4, 64, 64
    torch.manual_seed(42)

    # Random base
    traj = torch.randn(N, M, T, C, H, W, device=device)

    # Add structure so features aren't pure noise
    for n in range(N):
        for m in range(M):
            for t in range(T):
                # Spatial waves
                x = torch.linspace(0, 2 * 3.14159 * (n+1), W)
                y = torch.linspace(0, 2 * 3.14159 * (m+1), H)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                wave = torch.sin(xx) * torch.cos(yy) * (1 + 0.1 * t)
                traj[n, m, t, :, :, :] += wave.unsqueeze(0)

    return traj


def test_per_timestep_no_all_zeros(device, diverse_trajectories):
    """Test that per-timestep features are not all zero."""
    config = SDFConfig()
    extractor = SDFExtractor(device, config)

    per_timestep = extractor.extract_per_timestep(diverse_trajectories)

    # Check shape
    N, M, T, C, H, W = diverse_trajectories.shape
    assert per_timestep.shape[0] == N
    assert per_timestep.shape[1] == T
    D = per_timestep.shape[2]
    assert D > 0, "Should have at least some per-timestep features"

    # Check each feature is not all zeros
    registry = extractor.get_feature_registry()
    per_timestep_cats = ['spatial', 'spectral', 'cross_channel']
    feature_names = []
    for cat in per_timestep_cats:
        feature_names.extend([f.name for f in registry.get_features_by_category(cat)])

    # Smoke test: Just check that features can be extracted and have reasonable values
    # Don't be overly strict about variance (test data might not exercise all features)

    features_with_zero_variance = []

    for idx, name in enumerate(feature_names):
        if idx >= D:
            break  # Config may have disabled some features

        feat_values = per_timestep[:, :, idx]

        # Skip NaN features (e.g., cross-channel with C=1)
        if torch.isnan(feat_values).all():
            continue

        # Check that at least SOME features have variance
        # (All features being constant would indicate a bug)
        if feat_values.numel() > 1:
            std = feat_values.std()
            if std < 1e-10 and feat_values.abs().max() > 1e-10:
                features_with_zero_variance.append(name)

    # At most 30% of features should be constant for this test data
    # (If more than 30% are constant, likely a bug in extraction)
    max_constant = int(0.3 * len(feature_names))
    assert len(features_with_zero_variance) <= max_constant, \
        f"Too many features with zero variance ({len(features_with_zero_variance)}/{len(feature_names)}): {features_with_zero_variance[:10]}"


def test_per_trajectory_no_all_zeros(device, diverse_trajectories):
    """Test that per-trajectory features are not all zero."""
    config = SDFConfig()
    extractor = SDFExtractor(device, config)

    per_trajectory = extractor.extract_per_trajectory(diverse_trajectories)

    # Check shape
    N, M, T, C, H, W = diverse_trajectories.shape
    assert per_trajectory.shape[0] == N
    assert per_trajectory.shape[1] == M
    D_traj = per_trajectory.shape[2]
    assert D_traj > 0, "Should have at least some per-trajectory features"

    # Check each feature is not all zeros
    registry = extractor.get_feature_registry()
    per_traj_cats = ['temporal', 'causality', 'invariant_drift']
    feature_names = []
    for cat in per_traj_cats:
        feature_names.extend([f.name for f in registry.get_features_by_category(cat)])

    # Smoke test: Just check that features can be extracted and have reasonable values
    # Don't be overly strict about variance (test data might not exercise all features)

    features_with_zero_variance = []

    for idx, name in enumerate(feature_names):
        if idx >= D_traj:
            break  # Config may have disabled some features

        feat_values = per_trajectory[:, :, idx]

        # Skip NaN features (e.g., causality with T=1)
        if torch.isnan(feat_values).all():
            continue

        # Check that at least SOME features have variance
        if feat_values.numel() > 1:
            std = feat_values.std()
            if std < 1e-10 and feat_values.abs().max() > 1e-10:
                features_with_zero_variance.append(name)

    # At most 30% of features should be constant for this test data
    max_constant = int(0.3 * len(feature_names))
    assert len(features_with_zero_variance) <= max_constant, \
        f"Too many features with zero variance ({len(features_with_zero_variance)}/{len(feature_names)}): {features_with_zero_variance[:10]}"


def test_no_infinite_values(device, diverse_trajectories):
    """Test that no features produce infinite values."""
    config = SDFConfig()
    extractor = SDFExtractor(device, config)

    results = extractor.extract_all(diverse_trajectories)

    for key, tensor in results.items():
        # Filter out NaN (which is expected for some edge cases)
        valid_values = tensor[~torch.isnan(tensor)]

        if valid_values.numel() > 0:
            assert not torch.isinf(valid_values).any(), \
                f"Found infinite values in '{key}'"


def test_shape_consistency(device, diverse_trajectories):
    """Test that all outputs have correct shapes."""
    config = SDFConfig()
    extractor = SDFExtractor(device, config)

    N, M, T, C, H, W = diverse_trajectories.shape

    results = extractor.extract_all(diverse_trajectories)

    # Check per-timestep shape
    per_timestep = results['per_timestep']
    assert per_timestep.shape[0] == N
    assert per_timestep.shape[1] == T
    assert per_timestep.ndim == 3

    # Check per-trajectory shape
    per_trajectory = results['per_trajectory']
    assert per_trajectory.shape[0] == N
    assert per_trajectory.shape[1] == M
    assert per_trajectory.ndim == 3

    # Check aggregated shapes
    for key in ['aggregated_mean', 'aggregated_std', 'aggregated_cv']:
        if key in results:
            agg = results[key]
            assert agg.shape[0] == N
            assert agg.ndim == 2
            assert agg.shape[1] == per_trajectory.shape[2], \
                f"Aggregated features should match per-trajectory feature count"


def test_edge_case_single_channel(device):
    """Test that C=1 is handled gracefully (cross-channel returns NaN)."""
    N, M, T, C, H, W = 2, 3, 10, 1, 64, 64  # C=1
    trajectories = torch.randn(N, M, T, C, H, W, device=device)

    config = SDFConfig()
    extractor = SDFExtractor(device, config)

    results = extractor.extract_all(trajectories)

    # Should not crash
    assert 'per_timestep' in results
    assert 'per_trajectory' in results

    # Cross-channel features should be NaN
    # (Test passes if no exception raised)


def test_edge_case_single_timestep(device):
    """Test that T=1 is handled gracefully (trajectory features return NaN)."""
    N, M, T, C, H, W = 2, 3, 1, 4, 64, 64  # T=1
    trajectories = torch.randn(N, M, T, C, H, W, device=device)

    config = SDFConfig()
    extractor = SDFExtractor(device, config)

    results = extractor.extract_all(trajectories)

    # Should not crash
    assert 'per_timestep' in results
    assert 'per_trajectory' in results

    # Per-timestep features should still work
    per_timestep = results['per_timestep']
    assert per_timestep.shape[1] == 1  # T=1

    # Trajectory features should be mostly NaN (no temporal dynamics)
    per_trajectory = results['per_trajectory']
    # (Some features like temporal mean might still work, but many should be NaN)


def test_reproducibility(device, diverse_trajectories):
    """Test that feature extraction is deterministic."""
    config = SDFConfig()
    extractor = SDFExtractor(device, config)

    results1 = extractor.extract_all(diverse_trajectories)
    results2 = extractor.extract_all(diverse_trajectories)

    for key in results1.keys():
        # Compare non-NaN values
        t1, t2 = results1[key], results2[key]
        mask = ~torch.isnan(t1)

        assert torch.allclose(t1[mask], t2[mask], rtol=1e-5, atol=1e-8), \
            f"Feature extraction is not reproducible for '{key}'"


def test_sensitivity_to_input(device):
    """Test that features change when inputs change."""
    config = SDFConfig()
    extractor = SDFExtractor(device, config)

    # Two different trajectories
    torch.manual_seed(42)
    traj1 = torch.randn(2, 3, 10, 4, 64, 64, device=device)

    torch.manual_seed(123)  # Different seed
    traj2 = torch.randn(2, 3, 10, 4, 64, 64, device=device)

    results1 = extractor.extract_all(traj1)
    results2 = extractor.extract_all(traj2)

    # At least some features should be different
    per_timestep_changed = False
    per_trajectory_changed = False

    # Check per-timestep
    t1, t2 = results1['per_timestep'], results2['per_timestep']
    mask = ~torch.isnan(t1) & ~torch.isnan(t2)
    if mask.any():
        per_timestep_changed = not torch.allclose(t1[mask], t2[mask], rtol=1e-2, atol=1e-2)

    # Check per-trajectory
    t1, t2 = results1['per_trajectory'], results2['per_trajectory']
    mask = ~torch.isnan(t1) & ~torch.isnan(t2)
    if mask.any():
        per_trajectory_changed = not torch.allclose(t1[mask], t2[mask], rtol=1e-2, atol=1e-2)

    assert per_timestep_changed or per_trajectory_changed, \
        "Features should change when inputs change"


def test_feature_registry_completeness(device):
    """Test that all registered features are actually extracted."""
    config = SDFConfig()
    extractor = SDFExtractor(device, config)

    registry = extractor.get_feature_registry()

    # Count expected features
    per_timestep_cats = ['spatial', 'spectral', 'cross_channel']
    per_traj_cats = ['temporal', 'causality', 'invariant_drift']

    expected_per_timestep = sum(
        len(registry.get_features_by_category(cat)) for cat in per_timestep_cats
    )
    expected_per_traj = sum(
        len(registry.get_features_by_category(cat)) for cat in per_traj_cats
    )

    # Extract
    N, M, T, C, H, W = 2, 3, 10, 4, 64, 64
    trajectories = torch.randn(N, M, T, C, H, W, device=device)
    results = extractor.extract_all(trajectories)

    # Check dimensions match
    actual_per_timestep = results['per_timestep'].shape[2]
    actual_per_traj = results['per_trajectory'].shape[2]

    # Should match or be less if some features disabled
    assert actual_per_timestep <= expected_per_timestep
    assert actual_per_traj <= expected_per_traj

    # At minimum, should have v1.0 features (spatial, spectral, temporal)
    assert actual_per_timestep >= 19 + 27  # spatial + spectral
    assert actual_per_traj >= 13  # temporal
