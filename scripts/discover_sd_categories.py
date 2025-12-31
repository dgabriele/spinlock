#!/usr/bin/env python3
"""
Discover SD feature categories for VQ-VAE training.

Creates a category-to-indices mapping for GroupedFeatureExtractor by organizing
aggregated SDF v2.0 features into semantic groups based on the feature registry.

For each category (spatial, spectral, temporal, etc.), we have 3 aggregated versions:
- mean (across realizations)
- std (standard deviation)
- cv (coefficient of variation)

So each category's features are replicated 3× in the aggregated feature tensor.

Usage:
    python scripts/discover_sd_categories.py \\
        --dataset datasets/test_1k_inline_features.h5 \\
        --feature-path "/features/sdf/aggregated/features" \\
        --output configs/vqvae/sd_category_mapping.json
"""

import argparse
import json
import h5py
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for development (same pattern as cli.py)
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from spinlock.features.registry import FeatureRegistry


def discover_categories(
    dataset_path: Path,
    feature_path: str,
    registry_path: str = "/features/sdf"
) -> Dict[str, List[int]]:
    """
    Discover feature categories from dataset.

    Args:
        dataset_path: Path to HDF5 dataset
        feature_path: HDF5 path to aggregated features
        registry_path: HDF5 path to SDF group containing registry attribute

    Returns:
        Dict mapping category name → list of feature indices
    """
    with h5py.File(dataset_path, 'r') as f:
        # Load feature registry from SDF group attributes
        if registry_path not in f:
            raise ValueError(f"SDF group not found at {registry_path} in dataset")

        sdf_group = f[registry_path]
        if 'feature_registry' not in sdf_group.attrs:
            raise ValueError(f"feature_registry attribute not found in {registry_path}")

        registry_json = sdf_group.attrs['feature_registry']
        if isinstance(registry_json, bytes):
            registry_json = registry_json.decode('utf-8')
        print(f"Loaded feature registry from {registry_path} attributes")

        # Load registry
        registry = FeatureRegistry.from_json(registry_json, family_name="sdf")
        print(f"\nFeature Registry:")
        print(f"  Total features: {registry.num_features}")
        print(f"  Categories: {', '.join(registry.categories)}")

        # Get aggregated features shape
        if feature_path not in f:
            raise ValueError(f"Features not found at {feature_path} in dataset")

        features_shape = f[feature_path].shape
        print(f"\nAggregated features shape: {features_shape}")
        print(f"  Expected: (num_samples, 120 trajectory features × 3 aggregations)")

        # Build category mapping
        category_indices: Dict[str, List[int]] = {}

        # Trajectory features are in categories:
        # - temporal
        # - causality
        # - invariant_drift
        # - operator_sensitivity
        # - nonlinear (if enabled)
        # - integrated (trajectory-level spectral features)

        # Per-timestep categories (spatial, spectral, cross_channel) are NOT included
        # in aggregated features - they're already aggregated in per_timestep

        trajectory_categories = [
            'temporal',
            'causality',
            'invariant_drift',
            'operator_sensitivity',
            'nonlinear',
            'integrated'  # Trajectory-level spectral features
        ]

        # Count trajectory features per category
        print(f"\nTrajectory feature counts by category:")
        trajectory_feature_offset = 0

        for category in trajectory_categories:
            features = registry.get_features_by_category(category)

            if len(features) == 0:
                print(f"  {category}: 0 features (skipped)")
                continue

            num_features = len(features)
            print(f"  {category}: {num_features} features")

            # Each trajectory feature has 3 aggregations: mean, std, cv
            # So indices are:
            # - mean: [offset + 0, offset + num_features)
            # - std:  [offset + num_features, offset + 2*num_features)
            # - cv:   [offset + 2*num_features, offset + 3*num_features)

            # We combine all 3 aggregations for this category
            indices = []

            # Mean aggregation
            indices.extend(range(
                trajectory_feature_offset,
                trajectory_feature_offset + num_features
            ))

            # Std aggregation
            indices.extend(range(
                trajectory_feature_offset + num_features,
                trajectory_feature_offset + 2 * num_features
            ))

            # CV aggregation
            indices.extend(range(
                trajectory_feature_offset + 2 * num_features,
                trajectory_feature_offset + 3 * num_features
            ))

            category_indices[category] = indices
            trajectory_feature_offset += 3 * num_features

        # Validate total feature count
        total_indices = sum(len(indices) for indices in category_indices.values())
        expected_features = features_shape[1]  # (N, F) -> F

        print(f"\nValidation:")
        print(f"  Total indices mapped: {total_indices}")
        print(f"  Expected from dataset: {expected_features}")

        if total_indices != expected_features:
            raise ValueError(
                f"Index count mismatch: {total_indices} mapped vs "
                f"{expected_features} in dataset"
            )

        # Print summary
        print(f"\nCategory mapping summary:")
        for category, indices in sorted(category_indices.items()):
            print(f"  {category}: {len(indices)} indices")

        return category_indices


def main():
    parser = argparse.ArgumentParser(
        description="Discover SD feature categories for VQ-VAE training"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to HDF5 dataset"
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        default="/features/sdf/aggregated/features",
        help="HDF5 path to aggregated features"
    )
    parser.add_argument(
        "--registry-path",
        type=str,
        default="/features/sdf",
        help="HDF5 path to SDF group (registry read from attributes)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # Validate dataset exists
    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)

    # Discover categories
    print(f"Discovering categories from {args.dataset}")
    print(f"Feature path: {args.feature_path}")
    print("=" * 60)

    category_mapping = discover_categories(
        args.dataset,
        args.feature_path,
        args.registry_path
    )

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(category_mapping, f, indent=2)

    print(f"\n✓ Category mapping saved to {args.output}")
    print(f"\nUse this mapping in VQ-VAE config:")
    print(f"  category_mapping_file: \"{args.output}\"")


if __name__ == "__main__":
    main()
