#!/usr/bin/env python3
"""Diagnose dataset homogeneity/diversity."""

import numpy as np
import h5py
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

def analyze_parameter_coverage(dataset_path, output_dir="diagnostics/homogeneity"):
    """Analyze parameter space coverage."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("DATASET HOMOGENEITY ANALYSIS")
    print("="*70)

    with h5py.File(dataset_path, 'r') as f:
        # Load parameters and features
        params = f['parameters/params'][:]

        # Check for incomplete dataset
        non_zero_mask = np.any(params != 0, axis=1)
        actual_samples = non_zero_mask.sum()
        allocated_samples = len(params)

        if actual_samples < allocated_samples:
            print(f"\nTruncating to {actual_samples} valid samples (from {allocated_samples} allocated)")
            params = params[non_zero_mask]

        n_samples, n_params = params.shape
        print(f"\nDataset: {n_samples} samples × {n_params} parameters")

        # 1. PARAMETER SPACE COVERAGE
        print(f"\n{'='*70}")
        print("1. PARAMETER SPACE COVERAGE")
        print(f"{'='*70}")

        param_names = [f"param_{i}" for i in range(n_params)]

        for i in range(n_params):
            values = params[:, i]
            unique_vals = len(np.unique(values))
            min_val, max_val = values.min(), values.max()
            std_val = values.std()

            # Check if parameter varies
            if std_val < 1e-10:
                status = "⚠️  CONSTANT (no variation)"
            elif unique_vals < 5:
                status = f"⚠️  DISCRETE ({unique_vals} unique values)"
            elif std_val < (max_val - min_val) * 0.1:
                status = "⚠️  LOW VARIANCE (clustered)"
            else:
                status = "✓  Good variance"

            print(f"  {param_names[i]}: [{min_val:.3f}, {max_val:.3f}], std={std_val:.3f} - {status}")

        # 2. PAIRWISE DISTANCE ANALYSIS
        print(f"\n{'='*70}")
        print("2. SAMPLE DIVERSITY (Pairwise Distances)")
        print(f"{'='*70}")

        # Compute pairwise distances in parameter space
        # Sample subset if too large
        sample_size = min(1000, n_samples)
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            params_sample = params[indices]
            print(f"  Analyzing {sample_size} random samples (from {n_samples} total)")
        else:
            params_sample = params

        distances = pdist(params_sample, metric='euclidean')

        mean_dist = distances.mean()
        std_dist = distances.std()
        min_dist = distances.min()
        max_dist = distances.max()

        print(f"\n  Pairwise distance statistics:")
        print(f"    Mean:   {mean_dist:.3f}")
        print(f"    Std:    {std_dist:.3f}")
        print(f"    Min:    {min_dist:.3f}")
        print(f"    Max:    {max_dist:.3f}")
        print(f"    Range:  {max_dist - min_dist:.3f}")

        # Coefficient of variation
        cv = std_dist / mean_dist if mean_dist > 0 else 0
        print(f"    Coeff of variation: {cv:.3f}")

        # Interpret
        if cv < 0.2:
            print(f"  ⚠️  LOW DIVERSITY: Samples are very similar (CV < 0.2)")
        elif cv < 0.5:
            print(f"  ⚠️  MODERATE DIVERSITY: Some clustering (CV < 0.5)")
        else:
            print(f"  ✓  GOOD DIVERSITY: Well-distributed samples (CV ≥ 0.5)")

        # Check for duplicate/near-duplicate samples
        duplicate_threshold = mean_dist * 0.1  # Within 10% of mean distance
        near_duplicates = (distances < duplicate_threshold).sum()
        duplicate_rate = near_duplicates / len(distances)

        print(f"\n  Near-duplicate analysis:")
        print(f"    Threshold: {duplicate_threshold:.3f} (10% of mean distance)")
        print(f"    Near-duplicates: {near_duplicates}/{len(distances)} ({duplicate_rate*100:.2f}%)")

        if duplicate_rate > 0.1:
            print(f"  ⚠️  HIGH DUPLICATION: {duplicate_rate*100:.1f}% of pairs are very similar")
        elif duplicate_rate > 0.05:
            print(f"  ⚠️  MODERATE DUPLICATION: {duplicate_rate*100:.1f}% of pairs are similar")
        else:
            print(f"  ✓  LOW DUPLICATION: Dataset has good uniqueness")

        # 3. PCA ANALYSIS - Effective dimensionality
        print(f"\n{'='*70}")
        print("3. EFFECTIVE DIMENSIONALITY (PCA)")
        print(f"{'='*70}")

        # Normalize parameters for PCA
        params_norm = (params - params.mean(axis=0)) / (params.std(axis=0) + 1e-10)

        pca = PCA()
        pca.fit(params_norm)

        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)

        # Find number of components for 90% and 95% variance
        n_90 = np.argmax(cumsum_var >= 0.90) + 1
        n_95 = np.argmax(cumsum_var >= 0.95) + 1
        n_99 = np.argmax(cumsum_var >= 0.99) + 1

        print(f"\n  Components needed for variance thresholds:")
        print(f"    90% variance: {n_90}/{n_params} components")
        print(f"    95% variance: {n_95}/{n_params} components")
        print(f"    99% variance: {n_99}/{n_params} components")

        # Effective dimensionality
        # Shannon entropy of explained variance
        effective_dim = np.exp(-np.sum(explained_var * np.log(explained_var + 1e-10)))

        print(f"\n  Effective dimensionality: {effective_dim:.1f}/{n_params}")
        print(f"    (Shannon entropy-based measure)")

        if effective_dim < n_params * 0.3:
            print(f"  ⚠️  LOW EFFECTIVE DIM: Parameters are highly correlated")
            print(f"     → Dataset may be restricted to low-dim manifold")
        elif effective_dim < n_params * 0.7:
            print(f"  ⚠️  MODERATE EFFECTIVE DIM: Some parameter correlation")
        else:
            print(f"  ✓  HIGH EFFECTIVE DIM: Parameters are relatively independent")

        # Plot PCA explained variance
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Scree plot
        axes[0].bar(range(1, len(explained_var) + 1), explained_var)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA Scree Plot')
        axes[0].grid(True, alpha=0.3)

        # Cumulative variance
        axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'b-', linewidth=2)
        axes[1].axhline(y=0.90, color='r', linestyle='--', label='90%')
        axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95%')
        axes[1].axhline(y=0.99, color='g', linestyle='--', label='99%')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Variance Explained')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        pca_plot_path = output_dir / "pca_analysis.png"
        plt.savefig(pca_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved PCA plot: {pca_plot_path}")

        # 4. LOAD FEATURES FOR FEATURE SPACE ANALYSIS
        print(f"\n{'='*70}")
        print("4. FEATURE SPACE DIVERSITY")
        print(f"{'='*70}")

        # Load temporal features as proxy for trajectory diversity
        if 'features/temporal/features' in f:
            temporal = f['features/temporal/features'][:]
            if actual_samples < allocated_samples:
                temporal = temporal[non_zero_mask]

            # Flatten: [N, R, T, F] -> [N, R*T*F]
            N = temporal.shape[0]
            temporal_flat = temporal.reshape(N, -1)

            # Sample for distance computation
            if N > sample_size:
                indices = np.random.choice(N, sample_size, replace=False)
                temporal_sample = temporal_flat[indices]
            else:
                temporal_sample = temporal_flat

            # Compute feature space distances
            print(f"  Analyzing temporal features: {temporal_flat.shape}")
            feat_distances = pdist(temporal_sample, metric='euclidean')

            mean_feat_dist = feat_distances.mean()
            std_feat_dist = feat_distances.std()
            cv_feat = std_feat_dist / mean_feat_dist if mean_feat_dist > 0 else 0

            print(f"\n  Feature space distance statistics:")
            print(f"    Mean:   {mean_feat_dist:.3f}")
            print(f"    Std:    {std_feat_dist:.3f}")
            print(f"    CV:     {cv_feat:.3f}")

            if cv_feat < 0.2:
                print(f"  ⚠️  LOW DIVERSITY: Trajectories are very similar")
            elif cv_feat < 0.5:
                print(f"  ⚠️  MODERATE DIVERSITY: Some trajectory clustering")
            else:
                print(f"  ✓  GOOD DIVERSITY: Diverse trajectories")

        # 5. TRAIN/VAL SIMILARITY CHECK
        print(f"\n{'='*70}")
        print("5. TRAIN/VAL SPLIT SIMILARITY")
        print(f"{'='*70}")

        # Simulate train/val split
        n_train = int(0.9 * n_samples)
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_samples)

        train_params = params[indices[:n_train]]
        val_params = params[indices[n_train:]]

        # Compare distributions
        print(f"\n  Split sizes: {len(train_params)} train, {len(val_params)} val")

        train_mean = train_params.mean(axis=0)
        val_mean = val_params.mean(axis=0)
        train_std = train_params.std(axis=0)
        val_std = val_params.std(axis=0)

        mean_diff = np.abs(train_mean - val_mean).mean()
        std_diff = np.abs(train_std - val_std).mean()

        print(f"\n  Distribution similarity:")
        print(f"    Mean difference: {mean_diff:.4f}")
        print(f"    Std difference:  {std_diff:.4f}")

        # Normalized difference
        scale = params.std(axis=0).mean()
        normalized_mean_diff = mean_diff / (scale + 1e-10)

        if normalized_mean_diff < 0.05:
            print(f"  ✓  VERY SIMILAR: Train/val distributions are nearly identical")
            print(f"     → This explains the small train/val loss gap!")
        elif normalized_mean_diff < 0.15:
            print(f"  ✓  SIMILAR: Train/val distributions are well-matched")
        else:
            print(f"  ⚠️  DIFFERENT: Train/val distributions differ significantly")

        # FINAL VERDICT
        print(f"\n{'='*70}")
        print("FINAL ASSESSMENT")
        print(f"{'='*70}")

        issues = []

        # Check each criterion
        if cv < 0.3:
            issues.append("Low sample diversity in parameter space")
        if duplicate_rate > 0.1:
            issues.append(f"High near-duplication rate ({duplicate_rate*100:.1f}%)")
        if effective_dim < n_params * 0.5:
            issues.append(f"Low effective dimensionality ({effective_dim:.1f}/{n_params})")
        if normalized_mean_diff < 0.05:
            issues.append("Train/val splits are extremely similar (may indicate homogeneity)")

        if not issues:
            print("✓ HEALTHY DIVERSITY")
            print("  Dataset has good variation across parameters and features")
            print("  No pathological homogeneity detected")
        else:
            print("⚠️  POTENTIAL HOMOGENEITY ISSUES DETECTED:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

            print("\nRECOMMENDATIONS:")
            if cv < 0.3:
                print("  • Increase parameter sampling range")
                print("  • Use more diverse initial conditions")
            if duplicate_rate > 0.1:
                print("  • Check if dataset generation creates duplicates")
                print("  • Increase sampling resolution")
            if effective_dim < n_params * 0.5:
                print("  • Parameters may be constrained by physics")
                print("  • Consider if some parameters can vary independently")
            if normalized_mean_diff < 0.05:
                print("  • Random split is fine, but dataset is quite uniform")
                print("  • Consider stratified sampling if expanding dataset")

        print(f"\n{'='*70}")
        print(f"Analysis saved to: {output_dir}/")
        print(f"{'='*70}")

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "datasets/cno_50k_3channel_10000k.h5"
    analyze_parameter_coverage(dataset)
