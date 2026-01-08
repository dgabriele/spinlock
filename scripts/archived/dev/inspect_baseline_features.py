"""Inspect baseline_10k dataset features for quality validation.

Checks:
1. NaN/inf values across all features
2. Feature value ranges and distributions
3. Completeness across operators
4. Per-category statistics
"""

import h5py
import numpy as np
from pathlib import Path


def inspect_dataset(dataset_path: str):
    """Comprehensive dataset inspection."""

    print("=" * 80)
    print("BASELINE_10K DATASET INSPECTION")
    print("=" * 80)
    print()

    # Open with SWMR (Single Writer Multiple Reader) mode to avoid lock conflicts
    try:
        f = h5py.File(dataset_path, 'r', swmr=True)
    except BlockingIOError:
        print("⚠ Dataset is locked (being written to). Trying without SWMR...")
        try:
            f = h5py.File(dataset_path, 'r', libver='latest')
        except Exception as e:
            print(f"✗ Cannot open dataset: {e}")
            print("  The dataset is currently being written to.")
            print("  Please try again when generation pauses or completes.")
            return

    with f:
        # Dataset structure
        print("Dataset Structure:")
        print(f"  File size: {Path(dataset_path).stat().st_size / 1e6:.1f} MB")
        print(f"  Groups: {list(f.keys())}")
        print()

        # Check features group
        if 'features' not in f:
            print("⚠ No 'features' group found!")
            return

        features_grp = f['features']
        if 'sdf' not in features_grp:
            print("⚠ No 'sdf' subgroup found!")
            return

        sdf_grp = features_grp['sdf']
        print("SDF Feature Groups:")
        for key in sdf_grp.keys():
            print(f"  {key}: {sdf_grp[key].shape if hasattr(sdf_grp[key], 'shape') else 'group'}")
        print()

        # Load aggregated features (main feature matrix for VQ-VAE)
        if 'aggregated' not in sdf_grp:
            print("⚠ No 'aggregated' features found!")
            return

        agg_features = sdf_grp['aggregated'][:]  # [N, F]
        N, F = agg_features.shape

        print("=" * 80)
        print("AGGREGATED FEATURES (for VQ-VAE training)")
        print("=" * 80)
        print(f"Shape: {agg_features.shape} (N={N} operators, F={F} features)")
        print()

        # 1. NaN/Inf Check
        print("1. NaN/Inf Detection:")
        print("-" * 40)
        nan_mask = np.isnan(agg_features)
        inf_mask = np.isinf(agg_features)

        nan_count = nan_mask.sum()
        inf_count = inf_mask.sum()

        print(f"  NaN values: {nan_count:,} / {agg_features.size:,} ({100*nan_count/agg_features.size:.2e}%)")
        print(f"  Inf values: {inf_count:,} / {agg_features.size:,} ({100*inf_count/agg_features.size:.2e}%)")

        if nan_count > 0:
            # Per-feature NaN counts
            nan_per_feature = nan_mask.sum(axis=0)
            worst_features_nan = np.argsort(nan_per_feature)[-10:][::-1]
            print(f"\n  Top 10 features with NaN:")
            for idx in worst_features_nan:
                if nan_per_feature[idx] > 0:
                    print(f"    Feature {idx:3d}: {nan_per_feature[idx]:4d}/{N} ({100*nan_per_feature[idx]/N:.1f}%)")

        if inf_count > 0:
            inf_per_feature = inf_mask.sum(axis=0)
            worst_features_inf = np.argsort(inf_per_feature)[-10:][::-1]
            print(f"\n  Top 10 features with Inf:")
            for idx in worst_features_inf:
                if inf_per_feature[idx] > 0:
                    print(f"    Feature {idx:3d}: {inf_per_feature[idx]:4d}/{N} ({100*inf_per_feature[idx]/N:.1f}%)")

        print()

        # 2. Value Range Check
        print("2. Value Range Statistics:")
        print("-" * 40)

        # Filter out NaN/Inf for stats
        valid_mask = ~(nan_mask | inf_mask)
        valid_features = np.where(valid_mask, agg_features, np.nan)

        # Global stats
        print(f"  Global min: {np.nanmin(valid_features):.3e}")
        print(f"  Global max: {np.nanmax(valid_features):.3e}")
        print(f"  Global mean: {np.nanmean(valid_features):.3e}")
        print(f"  Global std: {np.nanstd(valid_features):.3e}")
        print()

        # Per-feature stats
        feature_mins = np.nanmin(valid_features, axis=0)
        feature_maxs = np.nanmax(valid_features, axis=0)
        feature_means = np.nanmean(valid_features, axis=0)
        feature_stds = np.nanstd(valid_features, axis=0)

        # Identify problematic features
        zero_variance = feature_stds < 1e-10
        large_range = (feature_maxs - feature_mins) > 1e6

        print(f"  Features with zero variance: {zero_variance.sum()}/{F}")
        if zero_variance.sum() > 0:
            print(f"    Indices: {np.where(zero_variance)[0][:10].tolist()}...")

        print(f"  Features with large range (>1e6): {large_range.sum()}/{F}")
        if large_range.sum() > 0:
            large_range_idx = np.where(large_range)[0][:10]
            for idx in large_range_idx:
                print(f"    Feature {idx:3d}: [{feature_mins[idx]:.2e}, {feature_maxs[idx]:.2e}]")
        print()

        # 3. Completeness Check
        print("3. Completeness Check:")
        print("-" * 40)

        # Check per-operator completeness
        valid_per_operator = valid_mask.sum(axis=1)
        fully_complete = (valid_per_operator == F).sum()
        partially_complete = ((valid_per_operator > 0) & (valid_per_operator < F)).sum()
        empty = (valid_per_operator == 0).sum()

        print(f"  Fully complete operators: {fully_complete}/{N} ({100*fully_complete/N:.1f}%)")
        print(f"  Partially complete: {partially_complete}/{N} ({100*partially_complete/N:.1f}%)")
        print(f"  Empty operators: {empty}/{N} ({100*empty/N:.1f}%)")

        if partially_complete > 0:
            incomplete_idx = np.where((valid_per_operator > 0) & (valid_per_operator < F))[0][:5]
            print(f"\n  Sample incomplete operators:")
            for idx in incomplete_idx:
                print(f"    Operator {idx:4d}: {valid_per_operator[idx]:3d}/{F} valid features")
        print()

        # 4. Distribution Analysis
        print("4. Distribution Analysis:")
        print("-" * 40)

        # Percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        print("  Percentiles of feature means:")
        for p in percentiles:
            val = np.nanpercentile(feature_means, p)
            print(f"    {p:2d}%: {val:10.3e}")
        print()

        # Check for outliers
        q1 = np.nanpercentile(valid_features, 25)
        q3 = np.nanpercentile(valid_features, 75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        outliers = ((valid_features < lower_bound) | (valid_features > upper_bound)) & valid_mask
        outlier_count = outliers.sum()

        print(f"  Outliers (>3 IQR from quartiles): {outlier_count:,} / {valid_mask.sum():,} ({100*outlier_count/valid_mask.sum():.2f}%)")
        print()

        # 5. Per-Timestep and Per-Trajectory Features (if available)
        if 'per_timestep' in sdf_grp:
            per_timestep = sdf_grp['per_timestep']
            print("5. Per-Timestep Features:")
            print("-" * 40)
            print(f"  Shape: {per_timestep.shape}")

            # Check a sample
            sample = per_timestep[:10, :, :]  # First 10 operators
            nan_count_ts = np.isnan(sample).sum()
            print(f"  NaN count (sample): {nan_count_ts} / {sample.size}")
            print()

        if 'per_trajectory' in sdf_grp:
            per_trajectory = sdf_grp['per_trajectory']
            print("6. Per-Trajectory Features:")
            print("-" * 40)
            print(f"  Shape: {per_trajectory.shape}")

            # Check a sample
            sample = per_trajectory[:10, :, :]  # First 10 operators
            nan_count_traj = np.isnan(sample).sum()
            print(f"  NaN count (sample): {nan_count_traj} / {sample.size}")
            print()

        # 7. Feature Registry (if available)
        if 'feature_names' in sdf_grp.attrs:
            feature_names = sdf_grp.attrs['feature_names']
            print("7. Feature Registry:")
            print("-" * 40)
            print(f"  Total features: {len(feature_names)}")
            print(f"  Sample names: {feature_names[:5].tolist() if hasattr(feature_names, 'tolist') else feature_names[:5]}")
            print()

        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        issues = []
        if nan_count > 0:
            issues.append(f"⚠ {nan_count:,} NaN values detected")
        if inf_count > 0:
            issues.append(f"⚠ {inf_count:,} Inf values detected")
        if zero_variance.sum() > 0:
            issues.append(f"⚠ {zero_variance.sum()} features with zero variance")
        if empty > 0:
            issues.append(f"⚠ {empty} operators with no valid features")

        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✓ No critical issues detected!")
            print(f"✓ {fully_complete}/{N} operators fully complete")
            print(f"✓ All {F} features have valid data")

        print()


if __name__ == "__main__":
    dataset_path = "datasets/baseline_10k.h5"
    inspect_dataset(dataset_path)
