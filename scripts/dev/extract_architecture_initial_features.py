#!/usr/bin/env python3
"""
Extract ARCHITECTURE and INITIAL features from baseline dataset.

ARCHITECTURE features: Derived from /parameters/params (12D raw parameters)
INITIAL features: Derived from /inputs/fields (3×128×128 initial conditions)

This script adds /features/architecture/ and /features/initial/ groups to the dataset.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse


def extract_architecture_features(params: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extract ARCHITECTURE features from raw parameters.

    Args:
        params: (N, 12) array of raw operator parameters

    Returns:
        features: (N, D) array of architecture features
        feature_names: List of feature names
    """
    N = params.shape[0]

    # Use raw parameters directly (12D)
    features = params.copy()
    feature_names = [f"param_{i}" for i in range(12)]

    # Add simple derived features to expand feature space
    # Pairwise products of select parameters (add 9 more features → 21D total)
    derived_pairs = [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11),  # Adjacent pairs
        (0, 11), (1, 10), (2, 9)  # Cross-pairs
    ]

    for i, j in derived_pairs:
        product = params[:, i] * params[:, j]
        features = np.column_stack([features, product])
        feature_names.append(f"param_{i}_x_param_{j}")

    print(f"  ARCHITECTURE: {params.shape[0]} samples × {features.shape[1]} features")
    return features.astype(np.float32), feature_names


def extract_initial_features(fields: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extract INITIAL features from input fields.

    Args:
        fields: (N, C, H, W) array of initial condition grids

    Returns:
        features: (N, D) array of initial condition features
        feature_names: List of feature names
    """
    N, C, H, W = fields.shape
    features_list = []
    feature_names = []

    # Per-channel statistics (6 features × 3 channels = 18 features)
    for c in range(C):
        channel_data = fields[:, c, :, :]

        # Basic statistics
        features_list.append(np.mean(channel_data, axis=(1, 2)))
        feature_names.append(f"ch{c}_mean")

        features_list.append(np.std(channel_data, axis=(1, 2)))
        feature_names.append(f"ch{c}_std")

        features_list.append(np.min(channel_data, axis=(1, 2)))
        feature_names.append(f"ch{c}_min")

        features_list.append(np.max(channel_data, axis=(1, 2)))
        feature_names.append(f"ch{c}_max")

        features_list.append(np.median(channel_data, axis=(1, 2)))
        feature_names.append(f"ch{c}_median")

        # Spatial structure: center vs edge variance
        center = channel_data[:, H//4:3*H//4, W//4:3*W//4]
        features_list.append(np.std(center, axis=(1, 2)))
        feature_names.append(f"ch{c}_center_std")

    # Cross-channel statistics (6 features)
    mean_per_channel = np.mean(fields, axis=(2, 3))  # (N, C)

    features_list.append(np.std(mean_per_channel, axis=1))  # Variance across channels
    feature_names.append("cross_channel_std")

    features_list.append(np.max(mean_per_channel, axis=1) - np.min(mean_per_channel, axis=1))
    feature_names.append("cross_channel_range")

    # Spatial gradients (6 features - 2 per channel)
    for c in range(C):
        channel_data = fields[:, c, :, :]
        grad_x = np.diff(channel_data, axis=2)
        grad_y = np.diff(channel_data, axis=1)

        features_list.append(np.mean(np.abs(grad_x), axis=(1, 2)))
        feature_names.append(f"ch{c}_grad_x_mean")

        features_list.append(np.mean(np.abs(grad_y), axis=(1, 2)))
        feature_names.append(f"ch{c}_grad_y_mean")

    # Global spatial structure (12 features)
    # Divide into quadrants and compute std in each
    h_mid, w_mid = H // 2, W // 2
    quadrants = [
        (0, h_mid, 0, w_mid),      # Top-left
        (0, h_mid, w_mid, W),      # Top-right
        (h_mid, H, 0, w_mid),      # Bottom-left
        (h_mid, H, w_mid, W),      # Bottom-right
    ]

    for c in range(C):
        for q_idx, (y1, y2, x1, x2) in enumerate(quadrants):
            quad = fields[:, c, y1:y2, x1:x2]
            features_list.append(np.std(quad, axis=(1, 2)))
            feature_names.append(f"ch{c}_quad{q_idx}_std")

    features = np.column_stack(features_list).astype(np.float32)

    print(f"  INITIAL: {N} samples × {features.shape[1]} features")
    return features, feature_names


def main():
    parser = argparse.ArgumentParser(
        description="Extract ARCHITECTURE and INITIAL features and add to dataset"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to HDF5 dataset"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing features if present"
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        return 1

    print(f"Extracting features from: {args.dataset}")
    print()

    with h5py.File(args.dataset, 'r+') as f:
        # Check if features group exists
        if 'features' not in f:
            print("Creating /features/ group")
            f.create_group('features')

        features_group = f['features']

        # Extract ARCHITECTURE features
        print("Extracting ARCHITECTURE features...")
        params = np.array(f['parameters/params'])
        arch_features, arch_names = extract_architecture_features(params)

        # Save to /features/architecture/aggregated/
        if 'architecture' in features_group:
            if args.overwrite:
                print("  Removing existing architecture group")
                del features_group['architecture']
            else:
                print("  Skipping (already exists, use --overwrite to replace)")
                arch_features = None

        if arch_features is not None:
            arch_group = features_group.create_group('architecture')
            agg_group = arch_group.create_group('aggregated')
            agg_group.create_dataset('features', data=arch_features, compression='gzip')
            agg_group.attrs['feature_names'] = [name.encode('utf-8') for name in arch_names]
            print(f"  ✓ Saved to /features/architecture/aggregated/")

        print()

        # Extract INITIAL features
        print("Extracting INITIAL features...")
        fields = np.array(f['inputs/fields'])
        init_features, init_names = extract_initial_features(fields)

        # Save to /features/initial/aggregated/
        if 'initial' in features_group:
            if args.overwrite:
                print("  Removing existing initial group")
                del features_group['initial']
            else:
                print("  Skipping (already exists, use --overwrite to replace)")
                init_features = None

        if init_features is not None:
            init_group = features_group.create_group('initial')
            agg_group = init_group.create_group('aggregated')
            agg_group.create_dataset('features', data=init_features, compression='gzip')
            agg_group.attrs['feature_names'] = [name.encode('utf-8') for name in init_names]
            print(f"  ✓ Saved to /features/initial/aggregated/")

    print()
    print("✓ Feature extraction complete!")
    print()
    print("Dataset now contains:")
    with h5py.File(args.dataset, 'r') as f:
        for family in f['features'].keys():
            family_group = f['features'][family]
            for granularity in family_group.keys():
                n_samples, n_features = family_group[granularity]['features'].shape
                print(f"  /features/{family}/{granularity}/: {n_samples} × {n_features}")

    return 0


if __name__ == "__main__":
    exit(main())
