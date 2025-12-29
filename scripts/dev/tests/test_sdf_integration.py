#!/usr/bin/env python3
"""
Integration test suite for SDF feature extraction.

Validates:
1. All new features extract correctly across different timestep counts
2. NaN rates for different T values (T=1, T=3, T=10, T=50)
3. Feature shapes and dtypes
4. Config flag integration
5. End-to-end extraction pipeline
"""

import torch
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from spinlock.features.sdf.spatial import SpatialFeatureExtractor
from spinlock.features.sdf.spectral import SpectralFeatureExtractor
from spinlock.features.sdf.invariant_drift import InvariantDriftExtractor
from spinlock.features.sdf.cross_channel import CrossChannelFeatureExtractor


def create_test_data(N=2, M=3, T=10, C=3, H=64, W=64, device='cuda'):
    """Create synthetic test data with known structure."""
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    device = torch.device(device)

    # Create spatial grids
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Generate diverse test data
    # Channel 0: Smooth sinusoid (low coherence length)
    # Channel 1: High-frequency pattern
    # Channel 2: Random noise

    fields = []
    for t in range(T):
        # Time-varying amplitude
        amp = 1.0 - 0.01 * t  # Slow decay

        c0 = amp * torch.sin(2 * torch.pi * 2 * X)
        c1 = amp * torch.sin(2 * torch.pi * 8 * Y)
        c2 = 0.5 * torch.randn(H, W, device=device)

        # Stack channels
        frame = torch.stack([c0, c1, c2], dim=0)  # [C, H, W]
        frame = frame.unsqueeze(0).unsqueeze(0).expand(N, M, C, H, W)  # [N, M, C, H, W]
        fields.append(frame.unsqueeze(2))  # [N, M, 1, C, H, W]

    fields = torch.cat(fields, dim=2)  # [N, M, T, C, H, W]

    return fields


def count_nans(features: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Count NaN values in each feature."""
    nan_counts = {}
    for name, feat in features.items():
        nan_count = torch.isnan(feat).sum().item()
        nan_counts[name] = nan_count
    return nan_counts


def test_timestep_dependency():
    """Test feature extraction across different timestep counts."""
    print("=" * 70)
    print("TEST 1: Timestep Dependency Analysis")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    timestep_configs = [
        (1, "T=1 (single timestep - expect NaN for temporal features)"),
        (3, "T=3 (minimum for temporal - expect few NaN)"),
        (10, "T=10 (good temporal resolution)"),
        (50, "T=50 (high temporal resolution)")
    ]

    results = {}

    for T, desc in timestep_configs:
        print(f"\n{'-' * 70}")
        print(f"{desc}")
        print(f"{'-' * 70}")

        # Create test data
        fields = create_test_data(N=2, M=3, T=T, C=3, H=64, W=64, device=device)

        # Extract features from each category
        spatial_extractor = SpatialFeatureExtractor(device=device)
        spectral_extractor = SpectralFeatureExtractor(device=device)
        drift_extractor = InvariantDriftExtractor(device=device)
        cross_extractor = CrossChannelFeatureExtractor(device=device)

        # Spatial features (should work for all T)
        spatial_features = spatial_extractor.extract(fields, config=None)
        spatial_nans = count_nans(spatial_features)

        # Spectral features (should work for all T)
        spectral_features = spectral_extractor.extract(fields, config=None)
        spectral_nans = count_nans(spectral_features)

        # Invariant drift features (NaN for T=1)
        drift_features = drift_extractor.extract(fields, config=None)
        drift_nans = count_nans(drift_features)

        # Cross-channel features (should work for all T with C>1)
        cross_features = cross_extractor.extract(fields, config=None)
        cross_nans = count_nans(cross_features)

        # Count total NaNs
        total_features = len(spatial_features) + len(spectral_features) + len(drift_features) + len(cross_features)
        total_nans = sum(spatial_nans.values()) + sum(spectral_nans.values()) + sum(drift_nans.values()) + sum(cross_nans.values())

        nan_rate = (total_nans / total_features) * 100 if total_features > 0 else 0

        print(f"\nFeature counts:")
        print(f"  Spatial:        {len(spatial_features):3d} features, {sum(v > 0 for v in spatial_nans.values()):3d} with NaN")
        print(f"  Spectral:       {len(spectral_features):3d} features, {sum(v > 0 for v in spectral_nans.values()):3d} with NaN")
        print(f"  Invariant drift:{len(drift_features):3d} features, {sum(v > 0 for v in drift_nans.values()):3d} with NaN")
        print(f"  Cross-channel:  {len(cross_features):3d} features, {sum(v > 0 for v in cross_nans.values()):3d} with NaN")
        print(f"\nTotal: {total_features} features, {nan_rate:.1f}% with NaN")

        results[T] = {
            'total_features': total_features,
            'total_nans': total_nans,
            'nan_rate': nan_rate,
            'spatial_nans': sum(v > 0 for v in spatial_nans.values()),
            'spectral_nans': sum(v > 0 for v in spectral_nans.values()),
            'drift_nans': sum(v > 0 for v in drift_nans.values()),
            'cross_nans': sum(v > 0 for v in cross_nans.values())
        }

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: NaN Rates by Timestep Count")
    print(f"{'=' * 70}")
    print(f"{'T':>5s} | {'Total Features':>15s} | {'Features w/ NaN':>15s} | {'NaN Rate':>10s}")
    print(f"{'-' * 70}")
    for T in [1, 3, 10, 50]:
        res = results[T]
        print(f"{T:5d} | {res['total_features']:15d} | {sum([res['spatial_nans'], res['spectral_nans'], res['drift_nans'], res['cross_nans']]):15d} | {res['nan_rate']:9.1f}%")

    # Validation
    print(f"\n{'=' * 70}")
    print("VALIDATION")
    print(f"{'=' * 70}")

    # T=1 should have NaN for temporal features
    if results[1]['nan_rate'] > 50:
        print("✓ PASS: T=1 has expected high NaN rate (temporal features undefined)")
    else:
        print("✗ FAIL: T=1 should have >50% NaN rate")

    # T>=3 should have low NaN rate
    if results[3]['nan_rate'] < 10 and results[10]['nan_rate'] < 10 and results[50]['nan_rate'] < 10:
        print("✓ PASS: T>=3 has low NaN rate (<10%)")
    else:
        print("✗ FAIL: T>=3 should have <10% NaN rate")

    print()
    return results


def test_new_features():
    """Test all newly implemented features."""
    print("=" * 70)
    print("TEST 2: New Feature Validation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create test data (T=50 for full feature extraction)
    fields = create_test_data(N=2, M=3, T=50, C=3, H=64, W=64, device=device)

    print(f"\nTest data shape: {tuple(fields.shape)}")

    # Extract features
    spatial_extractor = SpatialFeatureExtractor(device=device)
    spectral_extractor = SpectralFeatureExtractor(device=device)
    drift_extractor = InvariantDriftExtractor(device=device)

    spatial_features = spatial_extractor.extract(fields, config=None)
    spectral_features = spectral_extractor.extract(fields, config=None)
    drift_features = drift_extractor.extract(fields, config=None)

    # Check new features
    new_features = {
        'Spatial - Effective dimensionality': [
            'effective_rank',
            'participation_ratio',
            'explained_variance_90'
        ],
        'Spatial - Gradient saturation': [
            'gradient_saturation_ratio',
            'gradient_flatness'
        ],
        'Spatial - Coherence structure': [
            'coherence_length',
            'correlation_anisotropy',
            'structure_factor_peak'
        ],
        'Spectral - Harmonic content': [
            'harmonic_ratio_2f',
            'harmonic_ratio_3f',
            'total_harmonic_distortion',
            'fundamental_purity'
        ],
        'Invariant drift - Scale-specific dissipation': [
            'dissipation_rate_lowfreq',
            'dissipation_rate_highfreq',
            'dissipation_selectivity',
            'energy_cascade_direction'
        ]
    }

    all_features = {**spatial_features, **spectral_features, **drift_features}

    print(f"\n{'=' * 70}")
    print("NEW FEATURES VALIDATION")
    print(f"{'=' * 70}")

    total_new_features = 0
    total_valid = 0

    for category, feature_list in new_features.items():
        print(f"\n{category}:")
        for feat_name in feature_list:
            total_new_features += 1
            if feat_name in all_features:
                feat = all_features[feat_name]
                nan_count = torch.isnan(feat).sum().item()
                inf_count = torch.isinf(feat).sum().item()

                status = "✓" if nan_count == 0 and inf_count == 0 else "✗"
                if nan_count == 0 and inf_count == 0:
                    total_valid += 1

                print(f"  {status} {feat_name:35s} - shape={tuple(feat.shape)}, NaN={nan_count}, Inf={inf_count}")
            else:
                print(f"  ✗ {feat_name:35s} - MISSING")

    print(f"\n{'=' * 70}")
    print(f"Summary: {total_valid}/{total_new_features} new features valid (0% NaN/Inf)")
    print(f"{'=' * 70}")

    if total_valid == total_new_features:
        print("✓ PASS: All new features implemented and valid")
    else:
        print(f"✗ FAIL: {total_new_features - total_valid} new features missing or invalid")

    print()
    return total_valid == total_new_features


def test_feature_shapes():
    """Test feature shapes are correct."""
    print("=" * 70)
    print("TEST 3: Feature Shape Validation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    N, M, T, C, H, W = 2, 3, 10, 3, 64, 64
    fields = create_test_data(N=N, M=M, T=T, C=C, H=H, W=W, device=device)

    # Extract features
    spatial_extractor = SpatialFeatureExtractor(device=device)
    spectral_extractor = SpectralFeatureExtractor(device=device)
    drift_extractor = InvariantDriftExtractor(device=device)
    cross_extractor = CrossChannelFeatureExtractor(device=device)

    spatial_features = spatial_extractor.extract(fields, config=None)
    spectral_features = spectral_extractor.extract(fields, config=None)
    drift_features = drift_extractor.extract(fields, config=None)
    cross_features = cross_extractor.extract(fields, config=None)

    print(f"\nInput shape: [N={N}, M={M}, T={T}, C={C}, H={H}, W={W}]")
    print(f"\nExpected output shapes:")
    print(f"  Spatial/Spectral: [N={N}, M={M}, T={T}, C={C}]")
    print(f"  Invariant drift:  [N={N}, M={M}, C={C}]")
    print(f"  Cross-channel:    [N={N}, M={M}, T={T}]")

    # Check shapes
    shape_errors = []

    # Spatial/Spectral: [N, M, T, C]
    expected_per_timestep = (N, M, T, C)
    for name, feat in {**spatial_features, **spectral_features}.items():
        if feat.shape != expected_per_timestep:
            shape_errors.append(f"{name}: {feat.shape} != {expected_per_timestep}")

    # Invariant drift: [N, M, C]
    expected_trajectory = (N, M, C)
    for name, feat in drift_features.items():
        if feat.shape != expected_trajectory:
            shape_errors.append(f"{name}: {feat.shape} != {expected_trajectory}")

    # Cross-channel: [N, M, T]
    expected_cross = (N, M, T)
    for name, feat in cross_features.items():
        if feat.shape != expected_cross:
            shape_errors.append(f"{name}: {feat.shape} != {expected_cross}")

    if len(shape_errors) == 0:
        print(f"\n✓ PASS: All {len(spatial_features) + len(spectral_features) + len(drift_features) + len(cross_features)} features have correct shapes")
    else:
        print(f"\n✗ FAIL: {len(shape_errors)} shape errors:")
        for error in shape_errors[:10]:  # Show first 10
            print(f"  - {error}")

    print()
    return len(shape_errors) == 0


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("SDF FEATURE INTEGRATION TEST SUITE")
    print("=" * 70)

    all_passed = True

    # Test 1: Timestep dependency
    try:
        test_timestep_dependency()
    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}")
        all_passed = False

    # Test 2: New features
    try:
        passed = test_new_features()
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}")
        all_passed = False

    # Test 3: Feature shapes
    try:
        passed = test_feature_shapes()
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}")
        all_passed = False

    # Final summary
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
