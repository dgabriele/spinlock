#!/usr/bin/env python
"""
Empirical validation of SDF v2.1 features on 100-operator dataset.

Tests Phase 1 (numerical fixes) and Phase 2 (new extractors) in practice.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import h5py
import numpy as np
import torch
from typing import Dict, List


def load_features(dataset_path: str) -> Dict[str, np.ndarray]:
    """Load features from HDF5 dataset."""
    with h5py.File(dataset_path, 'r') as f:
        features = {
            'aggregated': f['features/sdf/aggregated/features'][:],
            'per_timestep': f['features/sdf/per_timestep/features'][:],
            'per_trajectory': f['features/sdf/per_trajectory/features'][:],
        }
    return features


def validate_no_extreme_values(features: np.ndarray, threshold: float = 1e6) -> Dict:
    """Check for extreme values (Phase 1 fix validation)."""
    print("\n" + "="*70)
    print("VALIDATION 1: No Extreme Values (FFT Power, Kurtosis Fixes)")
    print("="*70)

    max_abs = np.abs(features).max()
    extreme_mask = np.abs(features) > threshold
    num_extreme = extreme_mask.sum()

    if num_extreme > 0:
        extreme_indices = np.where(extreme_mask.any(axis=0))[0]
        print(f"  ❌ Found {num_extreme} extreme values (|x| > {threshold:.0e})")
        print(f"  Max absolute value: {max_abs:.2e}")
        print(f"  Extreme feature indices: {extreme_indices[:20].tolist()}")
        passed = False
    else:
        print(f"  ✅ No extreme values found")
        print(f"  Max absolute value: {max_abs:.2e}")
        passed = True

    return {
        'passed': passed,
        'max_abs_value': max_abs,
        'num_extreme': num_extreme
    }


def validate_nan_count(features: np.ndarray, threshold_pct: float = 1.0) -> Dict:
    """Check NaN count (Phase 1 fix validation)."""
    print("\n" + "="*70)
    print("VALIDATION 2: NaN Count (Rolling Window, PACF, Distributional Fixes)")
    print("="*70)

    nan_count = np.isnan(features).sum()
    total_values = features.size
    nan_pct = 100.0 * nan_count / total_values

    # Per-feature NaN counts
    nan_per_feature = np.isnan(features).sum(axis=0)
    num_features_with_nan = (nan_per_feature > 0).sum()

    print(f"  Total NaN values: {nan_count:,} / {total_values:,} ({nan_pct:.3f}%)")
    print(f"  Features with any NaN: {num_features_with_nan} / {features.shape[1]}")

    if num_features_with_nan > 0:
        # Show features with most NaN
        sorted_indices = np.argsort(nan_per_feature)[::-1]
        print(f"\n  Top 10 features by NaN count:")
        for i in range(min(10, num_features_with_nan)):
            idx = sorted_indices[i]
            if nan_per_feature[idx] > 0:
                print(f"    Feature {idx:3d}: {nan_per_feature[idx]:3d}/{features.shape[0]} NaN ({100*nan_per_feature[idx]/features.shape[0]:.1f}%)")

    passed = nan_pct < threshold_pct

    if passed:
        print(f"\n  ✅ NaN percentage {nan_pct:.3f}% < {threshold_pct}% threshold")
    else:
        print(f"\n  ❌ NaN percentage {nan_pct:.3f}% >= {threshold_pct}% threshold")

    return {
        'passed': passed,
        'nan_count': nan_count,
        'nan_pct': nan_pct,
        'num_features_with_nan': num_features_with_nan
    }


def validate_zero_variance(features: np.ndarray, threshold: float = 1e-12) -> Dict:
    """Check for zero-variance features (Phase 2 fix validation)."""
    print("\n" + "="*70)
    print("VALIDATION 3: Zero-Variance Features (Missing Extractor Fixes)")
    print("="*70)

    variance = np.nanvar(features, axis=0)
    zero_var_mask = variance < threshold
    num_zero_var = zero_var_mask.sum()

    if num_zero_var > 0:
        zero_var_indices = np.where(zero_var_mask)[0]
        print(f"  ❌ Found {num_zero_var} zero-variance features")
        print(f"  Zero-variance feature indices: {zero_var_indices.tolist()}")
        passed = False
    else:
        print(f"  ✅ No zero-variance features found")
        passed = True

    print(f"  Variance range: [{variance.min():.2e}, {variance.max():.2e}]")

    return {
        'passed': passed,
        'num_zero_var': num_zero_var
    }


def validate_feature_count(features: np.ndarray, expected_min: int = 300) -> Dict:
    """Check feature count matches expectations."""
    print("\n" + "="*70)
    print("VALIDATION 4: Feature Count (Phase 2 New Extractors)")
    print("="*70)

    num_features = features.shape[1]

    print(f"  Total aggregated features: {num_features}")
    print(f"  Expected minimum (with Phase 2): {expected_min}")

    if num_features >= expected_min:
        print(f"  ✅ Feature count meets expectations")
        passed = True
    else:
        print(f"  ❌ Feature count below expected minimum")
        passed = False

    return {
        'passed': passed,
        'num_features': num_features
    }


def validate_feature_statistics(features: np.ndarray) -> Dict:
    """Compute and display feature statistics."""
    print("\n" + "="*70)
    print("VALIDATION 5: Feature Statistics Summary")
    print("="*70)

    # Overall statistics
    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0)
    median = np.nanmedian(features, axis=0)

    print(f"  Mean of means: {mean.mean():.3f} ± {mean.std():.3f}")
    print(f"  Mean of stds: {std.mean():.3f} ± {std.std():.3f}")
    print(f"  Median of medians: {np.median(median):.3f}")

    # Check for potential issues
    issues = []

    # Very high std (potential outliers after clipping)
    high_std_mask = std > 100
    if high_std_mask.sum() > 0:
        issues.append(f"{high_std_mask.sum()} features with std > 100")

    # Very low std (potential zero-variance)
    low_std_mask = (std < 1e-6) & (~np.isnan(std))
    if low_std_mask.sum() > 0:
        issues.append(f"{low_std_mask.sum()} features with std < 1e-6")

    if issues:
        print(f"\n  ⚠️  Potential issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ✅ No statistical anomalies detected")

    return {
        'mean_of_means': mean.mean(),
        'mean_of_stds': std.mean(),
        'issues': issues
    }


def validate_phase2_features(dataset_path: str) -> Dict:
    """Check that Phase 2 features are present in metadata."""
    print("\n" + "="*70)
    print("VALIDATION 6: Phase 2 Feature Presence")
    print("="*70)

    with h5py.File(dataset_path, 'r') as f:
        # Get feature metadata if available
        if 'features/sdf/aggregated/metadata' in f:
            metadata_group = f['features/sdf/aggregated/metadata']

            # Check for feature names
            if 'feature_names' in metadata_group:
                feature_names = [name.decode() if isinstance(name, bytes) else name
                                for name in metadata_group['feature_names'][:]]

                # Look for Phase 2 category features
                phase2_categories = {
                    'distributional': ['entropy', 'participation', 'compression', 'multiscale'],
                    'structural': ['connected', 'edge', 'glcm'],
                    'physics': ['correlation', 'structure_factor', 'density_fluctuation', 'clustering'],
                    'morphological': ['area_fraction', 'circularity', 'centroid', 'hu_moment', 'granule'],
                    'multiscale': ['wavelet', 'pyramid']
                }

                found_categories = {}
                for category, keywords in phase2_categories.items():
                    matches = []
                    for keyword in keywords:
                        matches.extend([name for name in feature_names if keyword in name.lower()])
                    found_categories[category] = len(matches)
                    print(f"  {category.capitalize():20s}: {len(matches):3d} features")

                all_found = all(count > 0 for count in found_categories.values())

                if all_found:
                    print(f"\n  ✅ All Phase 2 categories present")
                    passed = True
                else:
                    missing = [cat for cat, count in found_categories.items() if count == 0]
                    print(f"\n  ❌ Missing categories: {missing}")
                    passed = False

                return {
                    'passed': passed,
                    'found_categories': found_categories
                }
            else:
                print("  ⚠️  Feature names not found in metadata")
                return {'passed': None, 'found_categories': {}}
        else:
            print("  ⚠️  Metadata not found (may be okay for some configs)")
            return {'passed': None, 'found_categories': {}}


def main():
    print("="*70)
    print("SDF v2.1 Feature Validation - 100 Operator Dataset")
    print("="*70)

    dataset_path = Path("datasets/test_100_v2_1_validation.h5")

    # Check if dataset exists
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("\nTo generate the dataset, run:")
        print("  poetry run spinlock generate --config configs/experiments/test_100_v2_1_validation.yaml")
        return 1

    print(f"\n✓ Loading dataset: {dataset_path}")

    # Load features
    features_dict = load_features(str(dataset_path))
    features = features_dict['aggregated']

    print(f"✓ Loaded features: {features.shape[0]} operators × {features.shape[1]} features")

    # Run validations
    results = {}

    results['extreme_values'] = validate_no_extreme_values(features)
    results['nan_count'] = validate_nan_count(features)
    results['zero_variance'] = validate_zero_variance(features)
    results['feature_count'] = validate_feature_count(features, expected_min=300)
    results['statistics'] = validate_feature_statistics(features)
    results['phase2_presence'] = validate_phase2_features(str(dataset_path))

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    validations = [
        ('No Extreme Values', results['extreme_values']['passed']),
        ('NaN Count', results['nan_count']['passed']),
        ('No Zero-Variance', results['zero_variance']['passed']),
        ('Feature Count', results['feature_count']['passed']),
    ]

    if results['phase2_presence']['passed'] is not None:
        validations.append(('Phase 2 Features', results['phase2_presence']['passed']))

    passed_count = sum(1 for _, passed in validations if passed)
    total_count = len(validations)

    for name, passed in validations:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    print("\n" + "="*70)
    print(f"Overall: {passed_count}/{total_count} validations passed")
    print("="*70)

    if passed_count == total_count:
        print("\n🎉 All validations passed! SDF v2.1 features are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} validation(s) failed. Review above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
