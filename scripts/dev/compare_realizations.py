#!/usr/bin/env python3
"""
Empirical comparison of M=3 vs M=5 realizations for VQ-VAE tokenization.

Tests whether 3 realizations provide sufficient feature statistics quality
for coarse-grained operator characterization compared to 5 realizations.

Usage:
    poetry run python scripts/dev/compare_realizations.py
"""

import h5py
import numpy as np
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


def run_generation(num_realizations: int, output_path: str) -> float:
    """
    Run dataset generation with specified number of realizations.

    Returns:
        Generation time in seconds
    """
    # Create temporary config with specified num_realizations
    config_template = Path("configs/experiments/test_realizations_50/dataset.yaml")
    config_path = f"/tmp/test_m{num_realizations}.yaml"

    # Read template and modify num_realizations and output_path
    with open(config_template, 'r') as f:
        config_text = f.read()

    # Replace num_realizations
    config_text = config_text.replace(
        "num_realizations: 5",
        f"num_realizations: {num_realizations}"
    )

    # Replace output path
    config_text = config_text.replace(
        "./datasets/test_realizations_m5_100.h5",
        f"./datasets/test_realizations_m{num_realizations}_100.h5"
    )

    with open(config_path, 'w') as f:
        f.write(config_text)

    print(f"\n{'='*70}")
    print(f"GENERATING DATASET: M={num_realizations} realizations, N=100 operators")
    print(f"{'='*70}\n")

    # Run generation
    start = time.time()
    result = subprocess.run(
        ["poetry", "run", "python", "scripts/spinlock.py", "generate", "--config", config_path],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: Generation failed for M={num_realizations}")
        print(result.stderr)
        sys.exit(1)

    print(f"\n✓ Generation complete: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    return elapsed


def load_features(dataset_path: str) -> Dict[str, np.ndarray]:
    """
    Load all features from HDF5 dataset.

    Returns:
        Dictionary of feature_name -> array
    """
    features = {}
    with h5py.File(dataset_path, 'r') as f:
        if 'features' not in f:
            return {}

        for feature_name in f['features'].keys():
            features[feature_name] = f['features'][feature_name][:]

    return features


def compare_feature_statistics(
    features_m5: Dict[str, np.ndarray],
    features_m3: Dict[str, np.ndarray]
) -> None:
    """
    Compare feature statistics between M=5 and M=3 datasets.

    Analyzes:
    1. Mean absolute difference (should be small)
    2. Correlation (should be high, >0.95)
    3. Relative std difference (acceptable if <20%)
    """
    print(f"\n{'='*70}")
    print("FEATURE STATISTICS COMPARISON: M=5 vs M=3")
    print(f"{'='*70}\n")

    if not features_m5 or not features_m3:
        print("⚠ No features extracted - check extract_operator_features setting")
        return

    # Check feature alignment
    common_features = set(features_m5.keys()) & set(features_m3.keys())

    if not common_features:
        print("⚠ No common features found between datasets")
        return

    print(f"Comparing {len(common_features)} features across 100 operators\n")

    results = []

    for feat_name in sorted(common_features):
        f5 = features_m5[feat_name]
        f3 = features_m3[feat_name]

        # Handle different shapes (features might be multi-dimensional)
        if f5.shape != f3.shape:
            print(f"⚠ Shape mismatch for {feat_name}: {f5.shape} vs {f3.shape}")
            continue

        # Flatten for statistics
        f5_flat = f5.flatten()
        f3_flat = f3.flatten()

        # Filter out NaN/Inf
        valid_mask = np.isfinite(f5_flat) & np.isfinite(f3_flat)
        f5_valid = f5_flat[valid_mask]
        f3_valid = f3_flat[valid_mask]

        if len(f5_valid) == 0:
            continue

        # Statistics
        mean_abs_diff = np.mean(np.abs(f5_valid - f3_valid))
        mean_abs_value = np.mean(np.abs(f5_valid))
        relative_diff = (mean_abs_diff / (mean_abs_value + 1e-8)) * 100

        correlation = np.corrcoef(f5_valid, f3_valid)[0, 1]

        std_m5 = np.std(f5_valid)
        std_m3 = np.std(f3_valid)
        std_diff_pct = ((std_m3 - std_m5) / (std_m5 + 1e-8)) * 100

        results.append({
            'name': feat_name,
            'relative_diff': relative_diff,
            'correlation': correlation,
            'std_diff_pct': std_diff_pct,
            'mean_m5': np.mean(f5_valid),
            'mean_m3': np.mean(f3_valid)
        })

    # Sort by correlation (descending)
    results.sort(key=lambda x: x['correlation'], reverse=True)

    # Print summary
    print(f"{'Feature':<40} {'Corr':>6} {'Rel Diff':>9} {'Std Diff':>9}")
    print(f"{'-'*40} {'-'*6} {'-'*9} {'-'*9}")

    for r in results[:20]:  # Top 20 features
        name_short = r['name'][:40]
        print(f"{name_short:<40} {r['correlation']:>6.3f} {r['relative_diff']:>8.1f}% {r['std_diff_pct']:>8.1f}%")

    # Overall statistics
    avg_corr = np.mean([r['correlation'] for r in results])
    avg_rel_diff = np.mean([r['relative_diff'] for r in results])
    avg_std_diff = np.mean([r['std_diff_pct'] for r in results])

    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS (across {len(results)} features)")
    print(f"{'='*70}")
    print(f"Average correlation:       {avg_corr:.4f}")
    print(f"Average relative diff:     {avg_rel_diff:.2f}%")
    print(f"Average std difference:    {avg_std_diff:.2f}%")
    print()

    # Quality assessment
    print(f"{'='*70}")
    print("QUALITY ASSESSMENT")
    print(f"{'='*70}")

    if avg_corr > 0.95:
        print("✓ EXCELLENT: Correlation > 0.95 - features are highly consistent")
    elif avg_corr > 0.90:
        print("✓ GOOD: Correlation > 0.90 - features are sufficiently consistent")
    elif avg_corr > 0.85:
        print("⚠ ACCEPTABLE: Correlation > 0.85 - marginal consistency")
    else:
        print("✗ POOR: Correlation < 0.85 - M=3 may be insufficient")

    if avg_rel_diff < 5.0:
        print("✓ EXCELLENT: Mean difference < 5% - very stable")
    elif avg_rel_diff < 10.0:
        print("✓ GOOD: Mean difference < 10% - stable")
    elif avg_rel_diff < 20.0:
        print("⚠ ACCEPTABLE: Mean difference < 20% - acceptable for coarse-graining")
    else:
        print("✗ POOR: Mean difference > 20% - M=3 may be insufficient")

    if abs(avg_std_diff) < 10.0:
        print("✓ EXCELLENT: Std difference < 10% - variance well captured")
    elif abs(avg_std_diff) < 20.0:
        print("✓ GOOD: Std difference < 20% - variance adequately captured")
    else:
        print("⚠ CAUTION: Std difference > 20% - variance less stable with M=3")

    print()


def main():
    """Run comparison experiment."""

    print(f"\n{'='*70}")
    print("EMPIRICAL VALIDATION: M=3 vs M=5 REALIZATIONS")
    print(f"{'='*70}")
    print("\nObjective: Determine if M=3 realizations provide sufficient")
    print("           feature quality for VQ-VAE tokenization compared to M=5")
    print("\nDataset: 100 operators, T=500 timesteps, 128×128 grids")
    print(f"{'='*70}\n")

    # Run M=5 baseline
    time_m5 = run_generation(5, "./datasets/test_realizations_m5_100.h5")

    # Run M=3 test
    time_m3 = run_generation(3, "./datasets/test_realizations_m3_100.h5")

    # Load features
    print("\nLoading features...")
    features_m5 = load_features("./datasets/test_realizations_m5_100.h5")
    features_m3 = load_features("./datasets/test_realizations_m3_100.h5")

    # Compare
    compare_feature_statistics(features_m5, features_m3)

    # Timing summary
    speedup = time_m5 / time_m3
    print(f"{'='*70}")
    print("TIMING COMPARISON")
    print(f"{'='*70}")
    print(f"M=5: {time_m5:.1f}s ({time_m5/60:.2f} min)")
    print(f"M=3: {time_m3:.1f}s ({time_m3/60:.2f} min)")
    print(f"Speedup: {speedup:.2f}× (expected: 1.67×)")
    print()

    # Projected 10K times
    projected_m5 = (time_m5 / 100) * 10000 / 3600
    projected_m3 = (time_m3 / 100) * 10000 / 3600

    print(f"{'='*70}")
    print("PROJECTED 10K DATASET TIMES")
    print(f"{'='*70}")
    print(f"M=5: {projected_m5:.1f} hours")
    print(f"M=3: {projected_m3:.1f} hours")
    print(f"Savings: {projected_m5 - projected_m3:.1f} hours ({((projected_m5-projected_m3)/projected_m5)*100:.1f}% reduction)")
    print()

    # Final recommendation
    print(f"{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print("Based on the empirical results above:")
    print("- If correlation > 0.90 AND relative diff < 15%: Use M=3 ✓")
    print("- If correlation > 0.85 AND relative diff < 25%: M=3 acceptable ⚠")
    print("- Otherwise: Stick with M=5 ✗")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
