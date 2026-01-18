#!/usr/bin/env python3
"""
Validate enhanced temporal feature extraction pipeline.

Tests the new 193D per-timestep feature architecture:
1. Loads sample trajectories
2. Extracts features using TemporalFeatureOrchestrator
3. Validates dimensions, NaN/Inf handling, and value ranges
4. Generates distribution plots for inspection
"""

import sys
from pathlib import Path
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spinlock.features.temporal import (
    TemporalFeatureOrchestrator,
    TemporalFeatureConfig,
    SpatialConfig,
    SpectralConfig,
    CrossChannelConfig,
    TemporalConfig,
)


def create_test_trajectories(
    num_samples: int = 10,
    num_realizations: int = 3,
    num_timesteps: int = 100,
    num_channels: int = 2,
    grid_size: int = 64,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create synthetic test trajectories.

    Args:
        num_samples: Number of samples
        num_realizations: Number of realizations per sample
        num_timesteps: Number of timesteps
        num_channels: Number of channels
        grid_size: Spatial grid size
        device: Device for computation

    Returns:
        Test trajectories [N, M, T, C, H, W]
    """
    print("Creating synthetic test trajectories...")
    print(f"  Shape: [{num_samples}, {num_realizations}, {num_timesteps}, {num_channels}, {grid_size}, {grid_size}]")

    # Create smooth random fields with temporal evolution
    trajectories = []
    for n in range(num_samples):
        sample_traj = []
        for m in range(num_realizations):
            # Initialize with smooth random field
            state = torch.randn(num_channels, grid_size, grid_size, device=device)

            # Apply low-pass filter
            for c in range(num_channels):
                fft = torch.fft.rfft2(state[c])
                H_freq, W_freq = fft.shape
                mask = torch.zeros_like(fft)
                mask[:H_freq//4, :W_freq//4] = 1.0
                fft_filtered = fft * mask
                state[c] = torch.fft.irfft2(fft_filtered, s=(grid_size, grid_size))

            # Evolve in time with simple dynamics
            realization_traj = []
            for t in range(num_timesteps):
                # Simple diffusion-like evolution
                state = state + 0.01 * torch.randn_like(state)
                state = 0.99 * state  # Decay
                realization_traj.append(state.clone())

            sample_traj.append(torch.stack(realization_traj))  # [T, C, H, W]

        trajectories.append(torch.stack(sample_traj))  # [M, T, C, H, W]

    trajectories = torch.stack(trajectories)  # [N, M, T, C, H, W]
    print(f"  ✓ Created synthetic trajectories\n")

    return trajectories


def validate_features(
    features: torch.Tensor,
    registry,
    output_dir: Path,
) -> Dict[str, bool]:
    """
    Validate extracted features.

    Args:
        features: Extracted features [N, T, D]
        registry: Feature registry
        output_dir: Output directory for plots

    Returns:
        Validation results
    """
    print("Validating extracted features...")
    N, T, D = features.shape
    results = {}

    # 1. Dimension check
    expected_dim = registry.num_features
    results['dimension_check'] = (D == expected_dim)
    print(f"  Dimension: {D} (expected {expected_dim}) - {'✓' if results['dimension_check'] else '✗'}")

    # 2. NaN/Inf check
    has_nan = torch.isnan(features).any().item()
    has_inf = torch.isinf(features).any().item()
    results['nan_check'] = not has_nan
    results['inf_check'] = not has_inf
    print(f"  NaN check: {'✓ No NaNs' if results['nan_check'] else '✗ Contains NaNs'}")
    print(f"  Inf check: {'✓ No Infs' if results['inf_check'] else '✗ Contains Infs'}")

    if has_nan:
        nan_counts = torch.isnan(features).sum(dim=(0, 1))
        problematic = (nan_counts > 0).nonzero(as_tuple=True)[0]
        print(f"    Warning: {len(problematic)} features contain NaNs")

    if has_inf:
        inf_counts = torch.isinf(features).sum(dim=(0, 1))
        problematic = (inf_counts > 0).nonzero(as_tuple=True)[0]
        print(f"    Warning: {len(problematic)} features contain Infs")

    # 3. Value range check
    features_np = features.cpu().numpy()
    mean_vals = np.nanmean(features_np, axis=(0, 1))
    std_vals = np.nanstd(features_np, axis=(0, 1))
    min_vals = np.nanmin(features_np, axis=(0, 1))
    max_vals = np.nanmax(features_np, axis=(0, 1))

    # Check for reasonable ranges (not too extreme)
    reasonable = (np.abs(mean_vals) < 1e6) & (std_vals < 1e6)
    results['range_check'] = reasonable.all()
    print(f"  Range check: {'✓ All features in reasonable range' if results['range_check'] else '✗ Some features have extreme values'}")

    if not results['range_check']:
        extreme = (~reasonable).nonzero()[0]
        print(f"    Warning: {len(extreme)} features have extreme values")

    # 4. Variability check (features should vary across samples/time)
    zero_var_features = (std_vals < 1e-10).sum()
    results['variability_check'] = (zero_var_features < D * 0.1)  # Less than 10% zero variance
    print(f"  Variability: {D - zero_var_features}/{D} features vary - {'✓' if results['variability_check'] else '✗'}")

    # 5. Category breakdown
    print("\n  Feature breakdown by category:")
    for category in ['spatial', 'spectral', 'cross_channel', 'temporal']:
        cat_features = registry.get_features_by_category(category)
        if cat_features:
            print(f"    {category}: {len(cat_features)}D")

    # 6. Generate distribution plots
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_feature_distributions(features_np, registry, output_dir)
        print(f"\n  ✓ Distribution plots saved to {output_dir}")

    print()
    return results


def plot_feature_distributions(
    features: np.ndarray,
    registry,
    output_dir: Path,
):
    """
    Plot feature distributions for visual inspection.

    Args:
        features: Features [N, T, D]
        registry: Feature registry
        output_dir: Output directory
    """
    N, T, D = features.shape

    # Plot 1: Overall distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean values
    mean_vals = np.nanmean(features, axis=(0, 1))
    axes[0, 0].hist(mean_vals, bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Mean value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of feature means')
    axes[0, 0].grid(True, alpha=0.3)

    # Std values
    std_vals = np.nanstd(features, axis=(0, 1))
    axes[0, 1].hist(std_vals, bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Std deviation')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of feature stds')
    axes[0, 1].grid(True, alpha=0.3)

    # Min/max ranges
    min_vals = np.nanmin(features, axis=(0, 1))
    max_vals = np.nanmax(features, axis=(0, 1))
    ranges = max_vals - min_vals
    axes[1, 0].hist(ranges, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Range (max - min)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of feature ranges')
    axes[1, 0].grid(True, alpha=0.3)

    # Category comparison
    categories = ['spatial', 'spectral', 'cross_channel', 'temporal']
    cat_means = []
    cat_labels = []
    for cat in categories:
        cat_features = registry.get_features_by_category(cat)
        if cat_features:
            cat_indices = [f.index for f in cat_features]
            cat_mean = np.nanmean(np.abs(mean_vals[cat_indices]))
            cat_means.append(cat_mean)
            cat_labels.append(f"{cat}\n({len(cat_features)}D)")

    axes[1, 1].bar(range(len(cat_means)), cat_means, edgecolor='black')
    axes[1, 1].set_xticks(range(len(cat_means)))
    axes[1, 1].set_xticklabels(cat_labels, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Mean |value|')
    axes[1, 1].set_title('Mean absolute values by category')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Temporal evolution (sample features)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sample_indices = [0, D//4, D//2, 3*D//4]
    sample_names = [
        registry.features[i].name if i < len(registry.features) else f"feature_{i}"
        for i in sample_indices
    ]

    for ax, idx, name in zip(axes.flat, sample_indices, sample_names):
        if idx < D:
            # Plot temporal evolution for first sample
            ax.plot(features[0, :, idx], alpha=0.7)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Value')
            ax.set_title(f'{name[:30]}...' if len(name) > 30 else name, fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_evolution_samples.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate enhanced temporal feature extraction")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test samples",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=100,
        help="Number of timesteps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_output"),
        help="Output directory for plots",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Enhanced Temporal Feature Extraction Validation")
    print("=" * 80)
    print()

    # Create test trajectories
    trajectories = create_test_trajectories(
        num_samples=args.num_samples,
        num_timesteps=args.num_timesteps,
        device=args.device,
    )

    # Initialize feature extractor
    print("Initializing feature extractor...")
    config = TemporalFeatureConfig(
        spatial=SpatialConfig(enabled=True, include_percentiles=True),
        spectral=SpectralConfig(enabled=True, num_fft_scales=5),
        cross_channel=CrossChannelConfig(enabled=True),
        temporal=TemporalConfig(
            enabled=True,
            window_size=5,
            short_window=5,
            medium_window=20,
            long_window=50,
        ),
    )

    extractor = TemporalFeatureOrchestrator(device=torch.device(args.device), config=config)
    registry = extractor.get_feature_registry()
    print(f"  Expected features: {registry.num_features}D\n")

    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        # Reset buffers
        extractor.temporal_extractor.reset()
        features = extractor.extract_per_timestep(trajectories)  # [N, T, D]

    print(f"  ✓ Extracted features: {list(features.shape)}\n")

    # Validate
    results = validate_features(features, registry, args.output_dir)

    # Summary
    print("=" * 80)
    print("Validation Summary")
    print("=" * 80)
    all_passed = all(results.values())
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check}: {status}")

    print()
    if all_passed:
        print("✓ All validation checks passed!")
    else:
        print("✗ Some validation checks failed - review output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
