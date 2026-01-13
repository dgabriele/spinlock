#!/usr/bin/env python3
"""
Extract reference features from full dataset for MNO dataset alignment.

For VQ-VAE training with reference feature regularization, this script extracts
reference solver (CNO/UAFNO/etc.) features at the same parameter points as the
MNO feature dataset. This enables balanced regularization that guides VQ-VAE
toward physics consistency while still learning MNO's distribution.

Usage:
    python scripts/extract_reference_features.py \\
        --mno-dataset datasets/mno_features_10k.h5 \\
        --reference-dataset datasets/100k_full_features.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def extract_reference_features(mno_dataset_path: Path, reference_dataset_path: Path) -> int:
    """Extract reference features at MNO generation indices.

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

    print(f"Extracting reference features from: {reference_dataset_path}")
    print(f"For MNO dataset: {mno_dataset_path}\n")

    # Load MNO metadata
    with h5py.File(mno_dataset_path, 'r') as f:
        # Check for required metadata
        if 'metadata/generation_indices' not in f:
            print(
                f"Error: MNO dataset missing 'metadata/generation_indices'",
                file=sys.stderr,
            )
            print(
                "  This metadata is required for reference feature extraction.",
                file=sys.stderr,
            )
            print(
                "  Please regenerate MNO dataset with updated generation pipeline.",
                file=sys.stderr,
            )
            return 1

        if 'metadata/is_interpolated' not in f:
            print(
                f"Error: MNO dataset missing 'metadata/is_interpolated'",
                file=sys.stderr,
            )
            print(
                "  This metadata is required for reference feature extraction.",
                file=sys.stderr,
            )
            print(
                "  Please regenerate MNO dataset with updated generation pipeline.",
                file=sys.stderr,
            )
            return 1

        # Load metadata
        generation_indices = f['metadata/generation_indices'][:]
        is_interpolated = f['metadata/is_interpolated'][:]

        print(f"MNO Dataset Information:")
        print(f"  Total samples: {len(generation_indices)}")
        print(f"  Interpolated points: {is_interpolated.sum()}")
        print(f"  Exact training points: {(~is_interpolated).sum()}")
        print(f"  Index range: [{generation_indices.min()}, {generation_indices.max()}]\n")

        # Check for MNO features to match dimensions
        if 'features/summary/aggregated/features' not in f:
            print(
                f"Error: MNO dataset missing 'features/summary/aggregated/features'",
                file=sys.stderr,
            )
            return 1

        mno_features = f['features/summary/aggregated/features']
        mno_shape = mno_features.shape
        print(f"MNO features shape: {mno_shape}")

    # Extract reference features
    with h5py.File(reference_dataset_path, 'r') as f:
        # Check for reference features
        if 'features/summary/aggregated/features' not in f:
            print(
                f"Error: Reference dataset missing 'features/summary/aggregated/features'",
                file=sys.stderr,
            )
            print(
                "  Ensure reference dataset uses same feature extraction pipeline.",
                file=sys.stderr,
            )
            return 1

        reference_features_dataset = f['features/summary/aggregated/features']
        ref_total_samples = reference_features_dataset.shape[0]
        ref_feature_dim = reference_features_dataset.shape[1]

        print(f"\nReference Dataset Information:")
        print(f"  Total samples: {ref_total_samples}")
        print(f"  Feature dimension: {ref_feature_dim}")

        # Validate indices
        if generation_indices.max() >= ref_total_samples:
            print(
                f"Error: Generation indices exceed reference dataset size",
                file=sys.stderr,
            )
            print(
                f"  Max index: {generation_indices.max()}, Reference size: {ref_total_samples}",
                file=sys.stderr,
            )
            return 1

        # Validate feature dimensions match
        if ref_feature_dim != mno_shape[1]:
            print(
                f"Error: Feature dimension mismatch",
                file=sys.stderr,
            )
            print(
                f"  MNO features: {mno_shape[1]}, Reference features: {ref_feature_dim}",
                file=sys.stderr,
            )
            print(
                "  Ensure both datasets use same feature extraction pipeline.",
                file=sys.stderr,
            )
            return 1

        # Extract features at generation indices
        print(f"\nExtracting reference features...")
        reference_features = reference_features_dataset[generation_indices]
        print(f"  Extracted shape: {reference_features.shape}")

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
            data=reference_features,
            compression='gzip',
            compression_opts=4,
        )

        # Add description
        f['features/reference_features'].attrs['description'] = (
            f'Reference solver features from {reference_dataset_path.name} at same '
            'parameter points as MNO features. Used for reference feature regularization '
            'during VQ-VAE training to guide toward physics consistency.'
        )

        f['features/reference_features'].attrs['source_dataset'] = str(reference_dataset_path)
        f['features/reference_features'].attrs['extraction_indices'] = 'metadata/generation_indices'

    print(f"\n{'=' * 70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"Reference features stored at: 'features/reference_features'")
    print(f"Shape: {reference_features.shape}")
    print(f"Interpolated samples: {is_interpolated.sum()} / {len(is_interpolated)}")
    print(f"{'=' * 70}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference features for MNO dataset alignment",
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

    return extract_reference_features(args.mno_dataset, args.reference_dataset)


if __name__ == "__main__":
    sys.exit(main())
