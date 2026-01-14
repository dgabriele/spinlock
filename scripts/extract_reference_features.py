#!/usr/bin/env python3
"""
Extract reference features from full dataset for MNO dataset alignment.

For VQ-VAE training with reference feature regularization, this script:
1. Checks if reference dataset has matching (params, IC) pairs
2. If match found: extracts features directly (fast)
3. If no match: reports unmatched samples for manual CNO generation

This ensures reference features correspond to the EXACT same problem instances
(same parameters + same initial conditions) as MNO features.

Usage:
    python scripts/extract_reference_features.py \\
        --mno-dataset datasets/noa_features/mno_v3_10k.h5 \\
        --reference-dataset datasets/local_100k_optimized.h5 \\
        [--generate-unmatched]  # Future: generate CNO rollouts for unmatched
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm


def find_matching_sample(
    target_params: np.ndarray,
    target_ic: np.ndarray,
    reference_params: np.ndarray,
    reference_ics: np.ndarray,
    param_tol: float = 1e-6,
    ic_tol: float = 1e-5,
) -> Optional[int]:
    """Find reference sample matching target params and IC.

    Args:
        target_params: Target parameter vector [D]
        target_ic: Target initial condition [H, W]
        reference_params: Reference parameter array [N, D]
        reference_ics: Reference IC array [N, M, H, W]
        param_tol: Tolerance for parameter matching
        ic_tol: Tolerance for IC matching

    Returns:
        Index of matching sample, or None if no match
    """
    # Find samples with matching parameters
    param_diffs = np.abs(reference_params - target_params[None, :])
    param_matches = np.all(param_diffs < param_tol, axis=1)

    if not param_matches.any():
        return None

    # Among param matches, find IC match
    matching_indices = np.where(param_matches)[0]

    for idx in matching_indices:
        # Check all realizations for this sample
        for r in range(reference_ics.shape[1]):
            ref_ic = reference_ics[idx, r]
            ic_diff = np.abs(ref_ic - target_ic)
            if np.all(ic_diff < ic_tol):
                return int(idx)

    return None


def extract_reference_features(
    mno_dataset_path: Path,
    reference_dataset_path: Path,
) -> int:
    """Extract reference features with IC/param matching.

    Args:
        mno_dataset_path: Path to MNO feature dataset
        reference_dataset_path: Path to reference dataset (100K CNO/UAFNO/etc.)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Validate inputs
    if not mno_dataset_path.exists():
        print(f"Error: MNO dataset not found: {mno_dataset_path}", file=sys.stderr)
        return 1

    if not reference_dataset_path.exists():
        print(f"Error: Reference dataset not found: {reference_dataset_path}", file=sys.stderr)
        return 1

    print(f"{'='*70}")
    print(f"REFERENCE FEATURE EXTRACTION WITH IC/PARAM MATCHING")
    print(f"{'='*70}")
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"Reference dataset: {reference_dataset_path}\n")

    # Load MNO dataset
    print("Loading MNO dataset...")
    with h5py.File(mno_dataset_path, 'r') as f:
        # Check required fields
        required_fields = [
            'metadata/generation_indices',
            'metadata/is_interpolated',
            'inputs/fields',
            'features/summary/aggregated/features',
        ]

        for field in required_fields:
            if field not in f:
                print(f"Error: MNO dataset missing '{field}'", file=sys.stderr)
                return 1

        # Check parameters (can be at two locations)
        has_params = ('parameters/params' in f) or ('features/parameters' in f)
        if not has_params:
            print("Error: MNO dataset missing parameters", file=sys.stderr)
            return 1

        # Load MNO data
        mno_ics = f['inputs/fields'][:, 0]  # [N, H, W] - first realization

        # Parameters can be at /parameters/params or /features/parameters
        if 'parameters/params' in f:
            mno_params = f['parameters/params'][:]  # [N, D]
        elif 'features/parameters' in f:
            mno_params = f['features/parameters'][:]  # [N, D]
        else:
            print("Error: MNO dataset missing parameters", file=sys.stderr)
            return 1

        generation_indices = f['metadata/generation_indices'][:]
        is_interpolated = f['metadata/is_interpolated'][:]
        mno_feature_shape = f['features/summary/aggregated/features'].shape

        n_samples = len(mno_ics)

        print(f"  MNO samples: {n_samples}")
        print(f"  Exact training points: {(~is_interpolated).sum()}")
        print(f"  Interpolated points: {is_interpolated.sum()}")
        print(f"  Feature dimension: {mno_feature_shape[1]}")
        print(f"  IC shape: {mno_ics.shape}")

    # Load reference dataset
    print(f"\nLoading reference dataset...")
    with h5py.File(reference_dataset_path, 'r') as f:
        if 'inputs/fields' not in f:
            print(f"Error: Reference dataset missing inputs/fields", file=sys.stderr)
            return 1

        # Check for aggregated features - try both locations
        ref_features_path = None
        if 'features/summary/aggregated/features' in f:
            ref_features_path = 'features/summary/aggregated/features'
        elif 'features/sdf/aggregated/features' in f:
            ref_features_path = 'features/sdf/aggregated/features'
        else:
            print(f"Error: Reference dataset missing aggregated features", file=sys.stderr)
            return 1

        # Parameters can be at /parameters/params or /features/parameters
        has_params = ('parameters/params' in f) or ('features/parameters' in f)
        if not has_params:
            print("Error: Reference dataset missing parameters", file=sys.stderr)
            return 1

        reference_ics = f['inputs/fields'][:]  # [N, M, H, W]

        # Load parameters from whichever location exists
        if 'parameters/params' in f:
            reference_params = f['parameters/params'][:]  # [N, D]
        else:
            reference_params = f['features/parameters'][:]  # [N, D]

        reference_features_raw = f[ref_features_path][:]  # [N, D_feat]

        ref_n_samples = len(reference_ics)
        ref_feature_dim_raw = reference_features_raw.shape[1]

        print(f"  Reference samples: {ref_n_samples}")
        print(f"  Raw feature dimension: {ref_feature_dim_raw}")
        print(f"  IC shape: {reference_ics.shape}")

        # Check if we need to exclude operator_sensitivity features
        # Reference has 360 aggregated features = 120 per-traj × 3 stats
        # MNO has 330 aggregated features = 110 per-traj × 3 stats
        # Difference: 10 operator_sensitivity features (indices 87-96 per-traj)

        # Compute per-trajectory feature count
        ref_per_traj = ref_feature_dim_raw // 3  # Assumes 3 aggregations (mean, std, cv)
        mno_per_traj = mno_feature_shape[1] // 3

        if ref_per_traj != mno_per_traj:
            print(f"\n  Feature dimension mismatch detected:")
            print(f"    Reference: {ref_per_traj} per-traj features → {ref_feature_dim_raw} aggregated")
            print(f"    MNO: {mno_per_traj} per-traj features → {mno_feature_shape[1]} aggregated")
            print(f"    Difference: {ref_per_traj - mno_per_traj} features")

            if ref_per_traj == 120 and mno_per_traj == 110:
                print(f"\n  Excluding operator_sensitivity features (indices 87-96)...")
                print(f"    These require operator access and cannot be computed from MNO rollouts.")

                # Create mask to exclude operator_sensitivity features (indices 87-96)
                # For aggregated features with 3 stats, this means excluding:
                # - Mean: 87*3+0, 87*3+1, ..., 96*3+2
                # - Std: same offset pattern
                # - CV: same offset pattern

                # Build list of indices to KEEP (0-86, 97-119 per-traj → expanded for 3 stats)
                keep_indices = []
                for i in range(120):
                    if i < 87 or i > 96:  # Exclude operator_sensitivity (87-96)
                        # Add all 3 aggregation variants for this feature
                        keep_indices.extend([i*3, i*3+1, i*3+2])

                # Filter features
                reference_features = reference_features_raw[:, keep_indices]
                print(f"    Filtered: {reference_features_raw.shape[1]} → {reference_features.shape[1]} features")
            else:
                print(f"\n  WARNING: Unexpected feature mismatch pattern!", file=sys.stderr)
                print(f"    Expected (120→110), got ({ref_per_traj}→{mno_per_traj})", file=sys.stderr)
                reference_features = reference_features_raw
        else:
            reference_features = reference_features_raw
            print(f"  Feature dimensions match: {ref_per_traj} per-trajectory features")

    # Match MNO samples to reference dataset
    print(f"\nMatching MNO samples to reference dataset...")
    print(f"  This may take a few minutes for 10K samples...\n")

    matched_indices = []  # (mno_idx, ref_idx) pairs
    unmatched_indices = []  # MNO indices for unmatched samples

    for i in tqdm(range(n_samples), desc="Matching samples"):
        match_idx = find_matching_sample(
            mno_params[i],
            mno_ics[i],
            reference_params,
            reference_ics,
        )

        if match_idx is not None:
            matched_indices.append((i, match_idx))
        else:
            unmatched_indices.append(i)

    print(f"\n{'='*70}")
    print(f"MATCHING RESULTS")
    print(f"{'='*70}")
    print(f"  Matched samples: {len(matched_indices)} ({100*len(matched_indices)/n_samples:.1f}%)")
    print(f"  Unmatched samples: {len(unmatched_indices)} ({100*len(unmatched_indices)/n_samples:.1f}%)")

    if len(unmatched_indices) > 0:
        print(f"\nWARNING: {len(unmatched_indices)} samples need CNO generation!")
        print(f"  These samples have params/ICs not in reference dataset.")
        print(f"  For proper regularization, these need CNO rollouts generated.")
        print(f"  (Feature generation not yet implemented in this script)\n")

    # Extract features for matched samples
    if len(matched_indices) == 0:
        print("\nNo matched samples found! Cannot proceed.", file=sys.stderr)
        return 1

    output_features = np.zeros((n_samples, mno_feature_shape[1]), dtype=np.float32)

    print(f"\nExtracting features for {len(matched_indices)} matched samples...")
    # Note: reference_features already loaded and filtered above
    for mno_idx, ref_idx in tqdm(matched_indices, desc="Extracting features"):
        output_features[mno_idx] = reference_features[ref_idx]

    # For unmatched samples, set to NaN as placeholder
    if len(unmatched_indices) > 0:
        for idx in unmatched_indices:
            output_features[idx, :] = np.nan

    # Store in MNO dataset
    print(f"\nStoring reference features in MNO dataset...")
    with h5py.File(mno_dataset_path, 'a') as f:
        # Check if already exists
        if 'features/reference_features' in f:
            print("  Warning: 'features/reference_features' already exists, overwriting...")
            del f['features/reference_features']

        # Create dataset
        f.create_dataset(
            'features/reference_features',
            data=output_features,
            compression='gzip',
            compression_opts=4,
        )

        # Add metadata
        f['features/reference_features'].attrs['description'] = (
            f'Reference solver features with IC/param matching. '
            f'{len(matched_indices)} extracted from reference dataset, '
            f'{len(unmatched_indices)} unmatched (set to NaN).'
        )
        f['features/reference_features'].attrs['source_dataset'] = str(reference_dataset_path)
        f['features/reference_features'].attrs['matched_samples'] = len(matched_indices)
        f['features/reference_features'].attrs['unmatched_samples'] = len(unmatched_indices)

    print(f"\n{'=' * 70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"Reference features stored at: 'features/reference_features'")
    print(f"Shape: {output_features.shape}")
    print(f"Matched (extracted): {len(matched_indices)}")
    print(f"Unmatched (NaN placeholders): {len(unmatched_indices)}")
    print(f"Interpolated samples: {is_interpolated.sum()} / {len(is_interpolated)}")

    if len(unmatched_indices) > 0:
        print(f"\nNOTE: VQ-VAE training should skip NaN samples in reference regularization.")
    print(f"{'=' * 70}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference features with IC/param matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mno-dataset",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to MNO feature dataset (.h5 file)",
    )

    parser.add_argument(
        "--reference-dataset",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to reference dataset (100K CNO/UAFNO/etc., .h5 file)",
    )

    args = parser.parse_args()

    return extract_reference_features(
        args.mno_dataset,
        args.reference_dataset,
    )


if __name__ == "__main__":
    sys.exit(main())
