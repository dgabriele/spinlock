"""
Feature Extraction Tutorial

Demonstrates how to extract and use SDF features from Spinlock datasets.

This tutorial covers:
1. Basic feature extraction
2. Reading and inspecting features
3. Feature analysis and visualization
4. Preparing features for ML tasks
"""

from pathlib import Path
import numpy as np
import h5py

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping visualizations")


def example_1_basic_extraction():
    """
    Example 1: Basic feature extraction from command line.

    This is typically done via CLI, not Python API.
    """
    print("=" * 60)
    print("Example 1: Basic Feature Extraction")
    print("=" * 60)
    print()

    print("To extract features from a dataset, use the CLI:")
    print()
    print("  spinlock extract-features --dataset datasets/test_diverse_100.h5")
    print()
    print("This will:")
    print("  - Extract all SDF features (spatial, spectral, temporal)")
    print("  - Store features in /features/sdf/ group")
    print("  - Create feature registry with name-to-index mapping")
    print()
    print("Options:")
    print("  --verbose       Show detailed progress")
    print("  --overwrite     Replace existing features")
    print("  --batch-size N  Adjust GPU batch size (default: 32)")
    print("  --dry-run       Show config without extracting")
    print()


def example_2_reading_features():
    """
    Example 2: Reading extracted features in Python.
    """
    print("=" * 60)
    print("Example 2: Reading Features")
    print("=" * 60)
    print()

    dataset_path = Path("datasets/test_diverse_100.h5")

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run: spinlock generate --shape 256x256 --count 100 --output datasets/test_diverse_100.h5")
        return

    # Method 1: Using HDF5FeatureReader (recommended)
    from spinlock.features.storage import HDF5FeatureReader

    print("Method 1: Using HDF5FeatureReader")
    print("-" * 40)

    with HDF5FeatureReader(dataset_path) as reader:
        # Check if features exist
        if not reader.has_sdf():
            print("No SDF features found. Run feature extraction first:")
            print(f"  spinlock extract-features --dataset {dataset_path}")
            return

        # Get feature registry
        registry = reader.get_sdf_registry()
        print(f"Total features: {registry.num_features}")
        print()

        # List features by category
        for category in ['spatial', 'spectral', 'temporal']:
            features = registry.get_features_by_category(category)
            print(f"{category.capitalize()} features: {len(features)}")
            for feat in features[:3]:
                print(f"  - {feat.name} (index {feat.index})")
            if len(features) > 3:
                print(f"  ... and {len(features) - 3} more")
            print()

        # Read aggregated features (most compact representation)
        aggregated = reader.get_sdf_aggregated()
        print(f"Aggregated features shape: {aggregated.shape}")
        print(f"  [N={aggregated.shape[0]}, D={aggregated.shape[1]}]")
        print()

        # Read per-timestep features
        per_timestep = reader.get_sdf_per_timestep()
        if per_timestep is not None:
            print(f"Per-timestep features shape: {per_timestep.shape}")
            print(f"  [N={per_timestep.shape[0]}, T={per_timestep.shape[1]}, D={per_timestep.shape[2]}]")
        print()

    # Method 2: Direct HDF5 access (for advanced users)
    print("Method 2: Direct HDF5 Access")
    print("-" * 40)

    with h5py.File(dataset_path, 'r') as f:
        if 'features/sdf/aggregated/features' in f:
            features = f['features/sdf/aggregated/features'][:]
            print(f"Features shape: {features.shape}")
            print(f"Features dtype: {features.dtype}")
            print()

            # Get feature statistics
            print("Feature statistics:")
            print(f"  Min:  {features.min():.3f}")
            print(f"  Max:  {features.max():.3f}")
            print(f"  Mean: {features.mean():.3f}")
            print(f"  Std:  {features.std():.3f}")
            print()


def example_3_feature_analysis():
    """
    Example 3: Analyzing feature distributions.
    """
    print("=" * 60)
    print("Example 3: Feature Analysis")
    print("=" * 60)
    print()

    dataset_path = Path("datasets/test_diverse_100.h5")

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    from spinlock.features.storage import HDF5FeatureReader

    with HDF5FeatureReader(dataset_path) as reader:
        if not reader.has_sdf():
            print("No features found")
            return

        registry = reader.get_sdf_registry()
        per_timestep = reader.get_sdf_per_timestep()

        if per_timestep is None:
            print("No per-timestep features")
            return

        # Analyze per-timestep features
        features_2d = per_timestep.squeeze(1)  # [N, T, D] -> [N, D] for T=1

        # Get spatial features
        spatial_features = registry.get_features_by_category('spatial')

        print("Spatial Feature Statistics:")
        print("-" * 40)
        print(f"{'Feature':<30} {'Min':>10} {'Max':>10} {'Mean':>10}")
        print("-" * 40)

        for feat_meta in spatial_features[:10]:
            idx = feat_meta.index
            values = features_2d[:, idx]
            print(f"{feat_meta.name:<30} {values.min():>10.3f} {values.max():>10.3f} {values.mean():>10.3f}")

        print()

        # Check for NaN values
        nan_count = np.isnan(features_2d).sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values")
            print("This is expected if dataset has T=1 (temporal features undefined)")
        else:
            print("✓ No NaN values found")
        print()


def example_4_visualization():
    """
    Example 4: Visualizing feature distributions.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping visualization example (matplotlib not available)")
        return

    print("=" * 60)
    print("Example 4: Feature Visualization")
    print("=" * 60)
    print()

    dataset_path = Path("datasets/test_diverse_100.h5")

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    from spinlock.features.storage import HDF5FeatureReader

    with HDF5FeatureReader(dataset_path) as reader:
        if not reader.has_sdf():
            print("No features found")
            return

        registry = reader.get_sdf_registry()
        per_timestep = reader.get_sdf_per_timestep()

        if per_timestep is None:
            return

        features_2d = per_timestep.squeeze(1)

        # Get a few interesting features to visualize
        spatial_features = registry.get_features_by_category('spatial')
        spectral_features = registry.get_features_by_category('spectral')

        # Select features
        feature_names = [
            'spatial_mean',
            'spatial_std',
            'gradient_magnitude_mean',
            'fft_power_scale_0_mean',
            'fft_power_scale_2_mean',
            'dominant_freq_magnitude'
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, feat_name in enumerate(feature_names):
            # Find feature index
            feat_meta = None
            for f in spatial_features + spectral_features:
                if f.name == feat_name:
                    feat_meta = f
                    break

            if feat_meta is None:
                continue

            values = features_2d[:, feat_meta.index]

            # Histogram
            axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(feat_name)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Count')
            axes[i].grid(True, alpha=0.3)

            # Add statistics
            stats_text = f"μ={values.mean():.2f}\nσ={values.std():.2f}"
            axes[i].text(0.95, 0.95, stats_text,
                        transform=axes[i].transAxes,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = Path("feature_distributions.png")
        plt.savefig(output_path, dpi=150)
        print(f"Saved feature distributions to: {output_path}")
        print()


def example_5_ml_preparation():
    """
    Example 5: Preparing features for machine learning.
    """
    print("=" * 60)
    print("Example 5: ML Preparation")
    print("=" * 60)
    print()

    dataset_path = Path("datasets/test_diverse_100.h5")

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    from spinlock.features.storage import HDF5FeatureReader

    with HDF5FeatureReader(dataset_path) as reader:
        if not reader.has_sdf():
            print("No features found")
            return

        # Read aggregated features (best for ML)
        features = reader.get_sdf_aggregated()

        print(f"Raw features shape: {features.shape}")
        print()

        # Step 1: Handle NaN values
        print("Step 1: Handle NaN values")
        print("-" * 40)

        nan_mask = np.isnan(features)
        nan_count = nan_mask.sum()

        if nan_count > 0:
            print(f"Found {nan_count} NaN values")

            # Option A: Remove NaN features
            valid_features = ~np.isnan(features).any(axis=0)
            features_clean = features[:, valid_features]
            print(f"After removing NaN features: {features_clean.shape}")

            # Option B: Impute NaN values (if you prefer)
            # features_imputed = np.nan_to_num(features, nan=0.0)
        else:
            features_clean = features
            print("No NaN values found")

        print()

        # Step 2: Normalize features
        print("Step 2: Normalize features")
        print("-" * 40)

        # Standardization (zero mean, unit variance)
        mean = features_clean.mean(axis=0, keepdims=True)
        std = features_clean.std(axis=0, keepdims=True)
        features_normalized = (features_clean - mean) / (std + 1e-8)

        print(f"Normalized features shape: {features_normalized.shape}")
        print(f"  Mean: {features_normalized.mean(axis=0).mean():.6f} (should be ~0)")
        print(f"  Std:  {features_normalized.std(axis=0).mean():.6f} (should be ~1)")
        print()

        # Step 3: Feature selection (optional)
        print("Step 3: Feature selection (optional)")
        print("-" * 40)

        # Remove low-variance features
        variances = features_normalized.var(axis=0)
        high_variance_mask = variances > 0.01  # Threshold
        features_selected = features_normalized[:, high_variance_mask]

        print(f"Features after variance filtering: {features_selected.shape}")
        print(f"  Removed {(~high_variance_mask).sum()} low-variance features")
        print()

        # Step 4: Save processed features
        print("Step 4: Save for ML")
        print("-" * 40)

        output_path = Path("features_ml_ready.npz")
        np.savez(
            output_path,
            features=features_selected,
            mean=mean[:, valid_features][:, high_variance_mask],
            std=std[:, valid_features][:, high_variance_mask],
            feature_mask=high_variance_mask
        )

        print(f"Saved ML-ready features to: {output_path}")
        print()
        print("Usage:")
        print("  data = np.load('features_ml_ready.npz')")
        print("  X = data['features']  # [N, D]")
        print("  # Use X for clustering, classification, VQ-VAE training, etc.")
        print()


def example_6_feature_importance():
    """
    Example 6: Computing feature importance (simple variance-based).
    """
    print("=" * 60)
    print("Example 6: Feature Importance")
    print("=" * 60)
    print()

    dataset_path = Path("datasets/test_diverse_100.h5")

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    from spinlock.features.storage import HDF5FeatureReader

    with HDF5FeatureReader(dataset_path) as reader:
        if not reader.has_sdf():
            print("No features found")
            return

        registry = reader.get_sdf_registry()
        per_timestep = reader.get_sdf_per_timestep()

        if per_timestep is None:
            return

        features_2d = per_timestep.squeeze(1)

        # Remove NaN features
        valid_mask = ~np.isnan(features_2d).any(axis=0)
        features_valid = features_2d[:, valid_mask]

        # Normalize
        features_norm = (features_valid - features_valid.mean(axis=0)) / (features_valid.std(axis=0) + 1e-8)

        # Compute variance (simple importance measure)
        variances = features_norm.var(axis=0)

        # Get feature names
        all_features = registry.get_features_by_category('spatial') + \
                      registry.get_features_by_category('spectral')

        valid_features = [f for f, valid in zip(all_features, valid_mask) if valid]

        # Sort by variance
        sorted_indices = np.argsort(variances)[::-1]

        print("Top 10 Most Important Features (by variance):")
        print("-" * 60)
        print(f"{'Rank':<6} {'Feature':<30} {'Variance':>10} {'Category':<12}")
        print("-" * 60)

        for rank, idx in enumerate(sorted_indices[:10], 1):
            feat = valid_features[idx]
            var = variances[idx]
            print(f"{rank:<6} {feat.name:<30} {var:>10.4f} {feat.category:<12}")

        print()


def main():
    """Run all examples."""
    examples = [
        example_1_basic_extraction,
        example_2_reading_features,
        example_3_feature_analysis,
        example_4_visualization,
        example_5_ml_preparation,
        example_6_feature_importance
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()

        input("Press Enter to continue to next example...")
        print("\n" * 2)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║      Spinlock Feature Extraction Tutorial               ║
    ║                                                          ║
    ║  This tutorial demonstrates how to extract and use       ║
    ║  SDF features from Spinlock datasets.                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    main()
