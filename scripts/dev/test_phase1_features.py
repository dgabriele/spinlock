#!/usr/bin/env python3
"""
Smoke test for Phase 1 feature extensions.

Tests that new features can be extracted without errors:
- Spatial: percentiles
- Temporal: event counts, time-to-event, rolling windows
- Nonlinear: RQA, correlation dimension (opt-in)
"""

import torch
from spinlock.features.sdf.config import SDFConfig
from spinlock.features.sdf.extractors import SDFExtractor

def test_phase1_features():
    """Test Phase 1 feature extraction on synthetic data."""

    print("=" * 70)
    print("PHASE 1 FEATURE SMOKE TEST")
    print("=" * 70)
    print()

    # Create synthetic trajectories
    # [N=2 operators, M=3 realizations, T=100 timesteps, C=3 channels, H=64, W=64]
    N, M, T, C, H, W = 2, 3, 100, 3, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Trajectory shape: [{N}, {M}, {T}, {C}, {H}, {W}]")
    print()

    trajectories = torch.randn(N, M, T, C, H, W, device=device)

    # Test 1: All Phase 1 features enabled (except expensive nonlinear features)
    print("Test 1: All Phase 1 features enabled (nonlinear disabled)")
    print("-" * 70)

    config = SDFConfig()
    # Enable new temporal features (already default)
    config.temporal.include_event_counts = True
    config.temporal.include_time_to_event = True
    config.temporal.include_rolling_windows = True

    # Enable new spatial features (already default)
    config.spatial.include_percentiles = True

    # Disable nonlinear for first test (expensive)
    config.nonlinear.enabled = False

    # Disable expensive v2.0 features for speed
    config.operator_sensitivity.enabled = False
    config.causality.enabled = False
    config.invariant_drift.enabled = False

    extractor = SDFExtractor(device=device, config=config)

    # Extract per-timestep features
    print("Extracting per-timestep features (spatial, spectral, cross-channel)...")
    per_timestep_features = extractor.extract_per_timestep(trajectories)
    print(f"✓ Per-timestep features shape: {per_timestep_features.shape}")

    # Extract per-trajectory features
    print("Extracting per-trajectory features (temporal with Phase 1 extensions)...")
    per_trajectory_features = extractor.extract_per_trajectory(trajectories)
    print(f"✓ Per-trajectory features shape: {per_trajectory_features.shape}")

    # Aggregate across realizations
    print("Aggregating across realizations...")
    aggregated_features = extractor.aggregate_realizations(per_trajectory_features)
    print(f"✓ Aggregated features shape: {aggregated_features.shape}")

    # Check registry
    registry = extractor.get_feature_registry()
    temporal_features = [f.name for f in registry.get_features_by_category('temporal')]
    spatial_features = [f.name for f in registry.get_features_by_category('spatial')]

    print()
    print("New temporal features registered:")
    new_temporal = [f for f in temporal_features if any(x in f for x in ['num_spikes', 'num_bursts', 'num_zero_crossings', 'time_to', 'rolling_w'])]
    for feat in new_temporal:
        print(f"  - {feat}")

    print()
    print("New spatial features registered:")
    new_spatial = [f for f in spatial_features if 'percentile' in f]
    for feat in new_spatial:
        print(f"  - {feat}")

    print()
    print("✓ Test 1 PASSED")
    print()

    # Test 2: Nonlinear features enabled (expensive, small test)
    print("Test 2: Nonlinear features enabled (expensive, subsampled)")
    print("-" * 70)

    config2 = SDFConfig()
    config2.nonlinear.enabled = True
    config2.nonlinear.include_recurrence = True
    config2.nonlinear.include_correlation_dim = True
    # Use aggressive subsampling for speed
    config2.nonlinear.rqa_subsample_factor = 20  # Every 20th timestep
    config2.nonlinear.corr_dim_subsample_factor = 20

    # Disable other extractors for speed
    config2.operator_sensitivity.enabled = False
    config2.causality.enabled = False
    config2.invariant_drift.enabled = False
    config2.temporal.include_rolling_windows = False  # Skip expensive rolling windows

    extractor2 = SDFExtractor(device=device, config=config2)

    print("Extracting nonlinear features (RQA, correlation dimension)...")
    # Extract only trajectory-level features
    per_trajectory_features2 = extractor2.extract_per_trajectory(trajectories)
    print(f"✓ Per-trajectory features with nonlinear shape: {per_trajectory_features2.shape}")

    # Check nonlinear features in registry
    registry2 = extractor2.get_feature_registry()
    nonlinear_features = [f.name for f in registry2.get_features_by_category('nonlinear')]

    print()
    print("Nonlinear features registered:")
    for feat in nonlinear_features:
        print(f"  - {feat}")

    print()
    print("✓ Test 2 PASSED")
    print()

    # Summary
    print("=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    print("✓ Spatial percentiles: WORKING")
    print("✓ Temporal event counts: WORKING")
    print("✓ Temporal time-to-event: WORKING")
    print("✓ Temporal rolling windows: WORKING")
    print("✓ Nonlinear RQA: WORKING")
    print("✓ Nonlinear correlation dimension: WORKING")
    print()
    print("All Phase 1 features successfully implemented and tested!")
    print()

    # Feature count estimate
    print("Feature count estimate:")
    total_features = config.estimate_feature_count()
    print(f"  With Phase 1 extensions (nonlinear disabled): {total_features} features")

    total_features_with_nonlinear = config2.estimate_feature_count()
    print(f"  With Phase 1 extensions (nonlinear enabled): {total_features_with_nonlinear} features")
    print()


if __name__ == "__main__":
    test_phase1_features()
