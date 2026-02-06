"""Example usage of V2 feature grouping system.

This script demonstrates how to use the V2 grouping system with a real dataset.
"""

import numpy as np
from pathlib import Path

from spinlock.v2.data import SpinlockDataset, SanitizationParams
from spinlock.features.grouping import (
    create_grouper,
    TemporalGroupingConfig,
    InitialGroupingConfig,
    ClusteringParams,
    GradientParams,
)


def example_basic_usage():
    """Example 1: Basic usage with default config."""
    print("=" * 80)
    print("Example 1: Basic Usage (Default Config)")
    print("=" * 80)

    # Load and sanitize dataset FIRST
    dataset_path = Path(__file__).parent.parent / "data" / "50k_baseline.h5"

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Skipping example...")
        return

    dataset = SpinlockDataset.from_file(str(dataset_path))
    san_params = SanitizationParams(
        remove_nans=True,
        remove_zero_variance=True,
    )
    clean_dataset = dataset.sanitize(params=san_params, inplace=False)

    # Extract temporal features
    with clean_dataset.open():
        temporal_features = clean_dataset.features.temporal.load_all()  # [N, T, D]

    # Aggregate over time (mean)
    temporal_agg = temporal_features.mean(axis=1)  # [N, D]

    # Get feature names
    feature_names = [f"temporal_{i}" for i in range(temporal_agg.shape[1])]

    # Create grouper with default config
    grouper = create_grouper("temporal")

    # Group features (clustering → gradient pipeline)
    result = grouper.group_features(temporal_agg, feature_names)

    # Print results
    print(f"\nDiscovered {result.num_groups} groups for {result.total_features} features")
    for name, group in result.groups.items():
        print(f"  {name}: {group.size} features")
        print(f"    Indices: {group.feature_indices[:5]}..." if group.size > 5 else f"    Indices: {group.feature_indices}")

    # Convert to dict for VQ-VAE
    group_dict = result.to_dict()
    print(f"\nConverted to dict: {len(group_dict)} groups")


def example_custom_config():
    """Example 2: Custom configuration."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)

    # Load dataset
    dataset_path = Path(__file__).parent.parent / "data" / "50k_baseline.h5"

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Skipping example...")
        return

    dataset = SpinlockDataset.from_file(str(dataset_path))
    san_params = SanitizationParams(
        remove_nans=True,
        remove_zero_variance=True,
    )
    clean_dataset = dataset.sanitize(params=san_params, inplace=False)

    # Extract temporal features
    with clean_dataset.open():
        temporal_features = clean_dataset.features.temporal.load_all().mean(axis=1)

    feature_names = [f"temporal_{i}" for i in range(temporal_features.shape[1])]

    # Custom config for temporal features
    config = TemporalGroupingConfig()
    config.clustering = ClusteringParams(
        min_groups=10,
        max_groups=25,
        use_gpu=True,
    )
    config.gradient = GradientParams(
        num_epochs=1000,  # More epochs for refinement
        orthogonality_target=0.10,  # Stricter orthogonality
    )
    config.skip_gradient_refinement = False  # Run full pipeline

    # Create grouper
    grouper = create_grouper("temporal", config=config)

    # Group features
    result = grouper.group_features(temporal_features, feature_names)

    print(f"\nDiscovered {result.num_groups} groups (custom config)")
    for name, group in result.groups.items():
        print(f"  {name}: {group.size} features")


def example_per_family_pipeline():
    """Example 3: Per-family grouping pipeline."""
    print("\n" + "=" * 80)
    print("Example 3: Per-Family Grouping Pipeline")
    print("=" * 80)

    # Load dataset
    dataset_path = Path(__file__).parent.parent / "data" / "50k_baseline.h5"

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Skipping example...")
        return

    dataset = SpinlockDataset.from_file(str(dataset_path))
    san_params = SanitizationParams(
        remove_nans=True,
        remove_zero_variance=True,
    )
    clean_dataset = dataset.sanitize(params=san_params, inplace=False)

    # Process each family
    families = ["temporal", "initial"]
    all_results = {}

    with clean_dataset.open():
        for family in families:
            print(f"\nProcessing {family.upper()} features...")

            # Get features for this family
            if family == "temporal":
                features = clean_dataset.features.temporal.load_all().mean(axis=1)
            elif family == "initial":
                features = clean_dataset.features.initial.load_all()

            # Get feature names
            feature_names = [f"{family}_{i}" for i in range(features.shape[1])]

            # Group with family-specific defaults
            grouper = create_grouper(family)
            result = grouper.group_features(features, feature_names)

            all_results[family] = result

            print(f"  Discovered {result.num_groups} groups")
            for name, group in result.groups.items():
                print(f"    {name}: {group.size} features")

    # Combine for VQ-VAE training
    combined_groups = {}
    for family, result in all_results.items():
        for name, group in result.groups.items():
            combined_groups[f"{family}_{name}"] = group.feature_indices

    print(f"\nCombined: {len(combined_groups)} total groups across families")


def example_clustering_only():
    """Example 4: Clustering-only (skip gradient refinement)."""
    print("\n" + "=" * 80)
    print("Example 4: Clustering-Only Mode")
    print("=" * 80)

    # Create synthetic data for quick demo
    np.random.seed(42)
    features = np.random.randn(200, 30).astype(np.float32)

    # Add correlation structure
    for i in range(1, 10):
        features[:, i] = features[:, 0] + np.random.randn(200) * 0.3

    feature_names = [f"feat_{i}" for i in range(30)]

    # Create config with gradient disabled
    config = TemporalGroupingConfig()
    config.clustering = ClusteringParams(num_groups=3, use_gpu=True)
    config.skip_gradient_refinement = True  # Skip gradient

    grouper = create_grouper("temporal", config=config)
    result = grouper.group_features(features, feature_names)

    print(f"\nClustering-only: {result.num_groups} groups")
    for name, group in result.groups.items():
        print(f"  {name}: {group.size} features")


def main():
    """Run all examples."""
    try:
        example_basic_usage()
        example_custom_config()
        example_per_family_pipeline()
        example_clustering_only()
    except ImportError as e:
        print(f"\nError: {e}")
        print("Make sure CuPy is installed for GPU clustering:")
        print("  pip install cupy-cuda12x")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
