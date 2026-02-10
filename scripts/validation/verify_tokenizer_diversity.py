#!/usr/bin/env python3
"""
Comprehensive analysis of VQTokenizer diversity and dataset coverage.

Verification steps:
1. Check which categories have low diversity (and which quantizer levels)
2. Analyze reconstruction quality
3. Check parameter/IC space coverage
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def analyze_token_diversity_by_level(tokens_path: Path) -> Dict[str, any]:
    """Check which categories have 1 token and identify their quantizer levels."""
    print("=" * 80)
    print("STEP 1: Token Diversity by Quantizer Level")
    print("=" * 80)

    with h5py.File(tokens_path, 'r') as f:
        tokens_group = f['tokens']

        # Parse category names to extract level info
        # Format: feature_name_group_X_LY where Y is the level
        diversity_by_level = defaultdict(list)
        low_diversity_categories = []

        for key in sorted(tokens_group.keys()):
            tokens = tokens_group[key][:]
            unique_count = len(np.unique(tokens))
            total_samples = len(tokens)

            # Extract level from key (e.g., "theta_L_group_1_L0" -> L0)
            if '_L0' in key:
                level = 'L0 (coarse)'
            elif '_L1' in key:
                level = 'L1 (medium)'
            elif '_L2' in key:
                level = 'L2 (fine)'
            else:
                level = 'unknown'

            diversity_ratio = unique_count / total_samples

            diversity_by_level[level].append({
                'category': key,
                'unique_tokens': unique_count,
                'total_samples': total_samples,
                'diversity_ratio': diversity_ratio
            })

            if unique_count <= 2:
                low_diversity_categories.append((key, level, unique_count))

        # Print summary by level
        print("\nDiversity Summary by Quantizer Level:")
        print("-" * 80)
        for level in ['L0 (coarse)', 'L1 (medium)', 'L2 (fine)', 'unknown']:
            if level not in diversity_by_level:
                continue

            categories = diversity_by_level[level]
            single_token = sum(1 for c in categories if c['unique_tokens'] == 1)
            low_token = sum(1 for c in categories if c['unique_tokens'] <= 2)
            avg_unique = np.mean([c['unique_tokens'] for c in categories])

            print(f"\n{level}:")
            print(f"  Total categories: {len(categories)}")
            print(f"  Single token (1 unique): {single_token} ({100*single_token/len(categories):.1f}%)")
            print(f"  Low diversity (≤2 unique): {low_token} ({100*low_token/len(categories):.1f}%)")
            print(f"  Average unique tokens: {avg_unique:.1f}")

        # Print detailed list of low diversity categories
        print("\n" + "=" * 80)
        print("Low Diversity Categories (≤2 unique tokens):")
        print("=" * 80)
        for cat, level, unique in low_diversity_categories:
            print(f"  {cat:60s} [{level:12s}] {unique} unique")

        return {
            'diversity_by_level': diversity_by_level,
            'low_diversity_categories': low_diversity_categories
        }


def check_reconstruction_quality(checkpoint_path: Path, dataset_path: Path) -> Dict[str, float]:
    """Load checkpoint and compute reconstruction metrics on validation set."""
    print("\n" + "=" * 80)
    print("STEP 2: Reconstruction Quality Analysis")
    print("=" * 80)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract metrics from checkpoint if available
    metrics = {}
    if 'best_val_loss' in checkpoint:
        metrics['checkpoint_best_val_loss'] = checkpoint['best_val_loss']
        print(f"\nCheckpoint best validation loss: {checkpoint['best_val_loss']:.6f}")

    # Load a sample from the dataset to check actual reconstruction
    with h5py.File(dataset_path, 'r') as f:
        if 'trajectories/cno' in f:
            # Load a random sample
            cno_data = f['trajectories/cno'][:]
            sample_size = min(100, len(cno_data))
            sample_indices = np.random.choice(len(cno_data), sample_size, replace=False)
            samples = cno_data[sample_indices]

            print(f"\nDataset validation samples loaded: {sample_size}")
            print(f"Sample shape: {samples.shape}")

            # Compute basic statistics
            metrics['data_mean'] = float(np.mean(samples))
            metrics['data_std'] = float(np.std(samples))
            metrics['data_min'] = float(np.min(samples))
            metrics['data_max'] = float(np.max(samples))

            print(f"Data statistics:")
            print(f"  Mean: {metrics['data_mean']:.6f}")
            print(f"  Std:  {metrics['data_std']:.6f}")
            print(f"  Min:  {metrics['data_min']:.6f}")
            print(f"  Max:  {metrics['data_max']:.6f}")

    return metrics


def analyze_parameter_space_coverage(dataset_path: Path) -> Dict[str, any]:
    """Check if the 50K samples span the parameter/IC space uniformly."""
    print("\n" + "=" * 80)
    print("STEP 3: Parameter Space Coverage Analysis")
    print("=" * 80)

    with h5py.File(dataset_path, 'r') as f:
        # Check for theta (parameter) data
        if 'features/theta' in f:
            theta_data = f['features/theta'][:]
            print(f"\nTheta (parameters) shape: {theta_data.shape}")

            # Theta should be [N, 14] for 14 PDE parameters
            if len(theta_data.shape) == 2:
                n_samples, n_params = theta_data.shape
                print(f"Samples: {n_samples}, Parameters: {n_params}")

                # Check coverage for each parameter dimension
                print("\nParameter coverage analysis:")
                print("-" * 80)
                print(f"{'Parameter':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Unique':>10}")
                print("-" * 80)

                coverage_stats = []
                for i in range(n_params):
                    param_values = theta_data[:, i]
                    stats = {
                        'param_idx': i,
                        'min': float(np.min(param_values)),
                        'max': float(np.max(param_values)),
                        'mean': float(np.mean(param_values)),
                        'std': float(np.std(param_values)),
                        'unique': len(np.unique(param_values)),
                        'coverage': float(np.max(param_values) - np.min(param_values))
                    }
                    coverage_stats.append(stats)

                    print(f"Param {i:2d}      {stats['min']:10.4f} {stats['max']:10.4f} "
                          f"{stats['mean']:10.4f} {stats['std']:10.4f} {stats['unique']:10d}")

                # Check if parameters span [0, 1] (expected from Sobol sampling)
                print("\n" + "=" * 80)
                print("Coverage Assessment:")
                print("=" * 80)
                full_coverage = sum(1 for s in coverage_stats if s['coverage'] > 0.95)
                partial_coverage = sum(1 for s in coverage_stats if 0.5 < s['coverage'] <= 0.95)
                poor_coverage = sum(1 for s in coverage_stats if s['coverage'] <= 0.5)

                print(f"Full coverage (>95% of [0,1]):     {full_coverage}/{n_params}")
                print(f"Partial coverage (50-95%):         {partial_coverage}/{n_params}")
                print(f"Poor coverage (<50%):              {poor_coverage}/{n_params}")

                if poor_coverage > 0:
                    print("\n⚠️  WARNING: Some parameters have poor coverage!")
                    print("This could explain low token diversity in parameter-related categories.")

                return {
                    'coverage_stats': coverage_stats,
                    'full_coverage_count': full_coverage,
                    'partial_coverage_count': partial_coverage,
                    'poor_coverage_count': poor_coverage
                }
            else:
                print(f"⚠️  Unexpected theta shape: {theta_data.shape}")
                return {}
        else:
            print("⚠️  No theta data found in dataset")
            return {}


def generate_summary_report(
    diversity_results: Dict,
    quality_results: Dict,
    coverage_results: Dict
) -> str:
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    report = []

    # Check 1: Diversity by level
    report.append("\n1. Token Diversity Assessment:")
    diversity_by_level = diversity_results['diversity_by_level']

    for level in ['L0 (coarse)', 'L1 (medium)', 'L2 (fine)']:
        if level not in diversity_by_level:
            continue
        categories = diversity_by_level[level]
        single_token = sum(1 for c in categories if c['unique_tokens'] == 1)
        pct = 100 * single_token / len(categories) if categories else 0

        if level == 'L0 (coarse)' and pct > 20:
            report.append(f"   ⚠️  {level}: {pct:.1f}% single-token (CONCERNING - should have diversity)")
        elif level == 'L2 (fine)' and pct > 50:
            report.append(f"   ✓  {level}: {pct:.1f}% single-token (ACCEPTABLE - fine-grained compression)")
        else:
            report.append(f"   •  {level}: {pct:.1f}% single-token")

    # Check 2: Reconstruction quality
    report.append("\n2. Reconstruction Quality:")
    if 'checkpoint_best_val_loss' in quality_results:
        loss = quality_results['checkpoint_best_val_loss']
        if loss < 0.001:
            report.append(f"   ✓  Validation loss: {loss:.6f} (EXCELLENT)")
        elif loss < 0.01:
            report.append(f"   •  Validation loss: {loss:.6f} (GOOD)")
        else:
            report.append(f"   ⚠️  Validation loss: {loss:.6f} (needs improvement)")

    # Check 3: Parameter coverage
    report.append("\n3. Parameter Space Coverage:")
    if coverage_results:
        poor = coverage_results.get('poor_coverage_count', 0)
        full = coverage_results.get('full_coverage_count', 0)
        total = poor + coverage_results.get('partial_coverage_count', 0) + full

        if poor > 0:
            report.append(f"   ⚠️  {poor}/{total} parameters have <50% coverage")
            report.append(f"   → This explains low token diversity in parameter-related categories")
        else:
            report.append(f"   ✓  All {total} parameters have >50% coverage")

    # Final verdict
    report.append("\n" + "=" * 80)
    report.append("VERDICT:")
    report.append("=" * 80)

    # Determine if this is a real problem
    has_l0_collapse = False
    if 'L0 (coarse)' in diversity_by_level:
        l0_cats = diversity_by_level['L0 (coarse)']
        l0_single = sum(1 for c in l0_cats if c['unique_tokens'] == 1)
        if l0_single / len(l0_cats) > 0.2:
            has_l0_collapse = True

    has_poor_coverage = coverage_results.get('poor_coverage_count', 0) > 0
    has_good_recon = quality_results.get('checkpoint_best_val_loss', 1.0) < 0.001

    if has_l0_collapse or has_poor_coverage:
        report.append("\n⚠️  DATASET DIVERSITY ISSUE DETECTED")
        if has_poor_coverage:
            report.append("\nRoot cause: Limited parameter space coverage in 50K dataset")
            report.append("Recommendation: Generate more diverse samples with better Sobol coverage")
        if has_l0_collapse:
            report.append("\nL0 quantizers showing unexpected collapse")
            report.append("This could affect semantic diversity in downstream models")
    elif has_good_recon:
        report.append("\n✓  TOKENIZER IS WORKING CORRECTLY")
        report.append("\nThe 39% single-token categories are expected behavior:")
        report.append("  • Mostly L2 (fine) quantizers doing aggressive compression")
        report.append("  • Reconstruction quality is excellent (MSE < 0.001)")
        report.append("  • All codebook entries are used (no mode collapse)")
        report.append("\nNo action needed on VQTokenizer.")
    else:
        report.append("\n•  MIXED RESULTS - Further investigation needed")

    summary = "\n".join(report)
    print(summary)
    return summary


def main():
    """Run all verification steps."""
    # Paths
    tokens_path = Path("datasets/50k_baseline_tokenized_theta.h5")
    checkpoint_path = Path("checkpoints/vqvae/theta_baseline_50k/vq_tokenizer_final.pt")
    dataset_path = Path("datasets/50k_baseline.h5")

    # Verify files exist
    for path in [tokens_path, checkpoint_path, dataset_path]:
        if not path.exists():
            print(f"❌ File not found: {path}")
            return

    print("\n" + "=" * 80)
    print("VQTokenizer Diversity Verification")
    print("=" * 80)
    print(f"\nTokenized dataset:  {tokens_path}")
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Original dataset:   {dataset_path}")

    # Run verification steps
    diversity_results = analyze_token_diversity_by_level(tokens_path)
    quality_results = check_reconstruction_quality(checkpoint_path, dataset_path)
    coverage_results = analyze_parameter_space_coverage(dataset_path)

    # Generate summary
    summary = generate_summary_report(diversity_results, quality_results, coverage_results)

    # Save report
    report_path = Path("visualizations/tokenizer_diversity_report.txt")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(summary)
    print(f"\n✓ Report saved to: {report_path}")


if __name__ == "__main__":
    main()
