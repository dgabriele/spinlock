"""
Visualization generation for MNO diagnostics.

Provides plotting functions for physics fidelity, tokenization quality,
and distribution analysis.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


def plot_physics_fidelity(
    results: Dict[str, Any],
    output_path: Path,
):
    """Generate physics fidelity visualization.

    Creates a figure with:
    - Per-timestep MSE (with std bands if available)
    - Error distribution histogram
    - Error growth curve

    Args:
        results: Physics evaluation results
        output_path: Path to save figure
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Per-timestep MSE
    ax = axes[0, 0]
    per_timestep_mse = np.array(results.get('per_timestep_mse', []))
    if len(per_timestep_mse) > 0:
        timesteps = np.arange(len(per_timestep_mse))
        ax.plot(timesteps, per_timestep_mse, label='Mean MSE', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('MSE')
        ax.set_title('Per-Timestep MSE')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 2: Per-timestep relative L2
    ax = axes[0, 1]
    per_timestep_rel_l2 = np.array(results.get('per_timestep_rel_l2', []))
    if len(per_timestep_rel_l2) > 0:
        timesteps = np.arange(len(per_timestep_rel_l2))
        ax.plot(timesteps, per_timestep_rel_l2, label='Relative L2', linewidth=2, color='orange')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Relative L2 Error')
        ax.set_title('Per-Timestep Relative L2')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 3: Error distribution
    ax = axes[1, 0]
    per_sample = results.get('per_sample_results', [])
    if per_sample:
        final_mses = [s['final_mse'] for s in per_sample]
        ax.hist(final_mses, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Final MSE')
        ax.set_ylabel('Count')
        ax.set_title(f'Final MSE Distribution (n={len(final_mses)})')
        ax.axvline(np.mean(final_mses), color='red', linestyle='--',
                  label=f'Mean={np.mean(final_mses):.4f}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 4: Error growth stats
    ax = axes[1, 1]
    error_growth = results.get('error_growth', {})
    growth_rate = error_growth.get('growth_rate', 0.0)
    growth_type = error_growth.get('growth_type', 'linear')
    r_squared = error_growth.get('r_squared', 0.0)

    if len(per_timestep_mse) > 0:
        timesteps = np.arange(len(per_timestep_mse))
        ax.plot(timesteps, per_timestep_mse, 'o', alpha=0.5, label='Actual')

        # Plot fitted curve
        if growth_type == 'linear':
            fitted = growth_rate * timesteps + per_timestep_mse[0]
            ax.plot(timesteps, fitted, 'r--', label=f'Linear fit (R²={r_squared:.3f})')
        else:
            # Exponential
            fitted = per_timestep_mse[0] * np.exp(growth_rate * timesteps)
            ax.plot(timesteps, fitted, 'r--', label=f'Exponential fit (R²={r_squared:.3f})')

        ax.set_xlabel('Timestep')
        ax.set_ylabel('MSE')
        ax.set_title(f'Error Growth ({growth_type})')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Physics fidelity plot saved to {output_path}")


def plot_tokenization_quality(
    mno_metrics: Dict[str, Any],
    cno_metrics: Dict[str, Any],
    dist_comparison: Dict[str, Any],
    output_path: Path,
):
    """Generate tokenization quality visualization.

    Creates a figure with:
    - Reconstruction MSE comparison
    - Token statistics comparison
    - Distribution shift metrics

    Args:
        mno_metrics: MNO tokenization metrics
        cno_metrics: CNO tokenization metrics
        dist_comparison: Distribution comparison metrics
        output_path: Path to save figure
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Reconstruction MSE
    ax = axes[0, 0]
    sources = ['MNO', 'CNO']
    recon_mses = [
        mno_metrics.get('reconstruction_mse', 0.0),
        cno_metrics.get('reconstruction_mse', 0.0),
    ]
    bars = ax.bar(sources, recon_mses, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('VQ-VAE Reconstruction Quality')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, recon_mses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom')

    # Plot 2: Token statistics
    ax = axes[0, 1]
    metrics_names = ['Perplexity', 'Entropy', 'Unique Tokens']
    mno_vals = [
        mno_metrics.get('perplexity', 0.0),
        mno_metrics.get('token_entropy', 0.0),
        mno_metrics.get('unique_tokens', 0),
    ]
    cno_vals = [
        cno_metrics.get('perplexity', 0.0),
        cno_metrics.get('token_entropy', 0.0),
        cno_metrics.get('unique_tokens', 0),
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    # Normalize for visualization (different scales)
    mno_vals_norm = np.array(mno_vals) / (np.array(cno_vals) + 1e-10)

    bars1 = ax.bar(x, [1.0, 1.0, 1.0], width, label='CNO (baseline)', alpha=0.7,
                   color='#ff7f0e', edgecolor='black')
    bars2 = ax.bar(x + width, mno_vals_norm, width, label='MNO (normalized)', alpha=0.7,
                   color='#1f77b4', edgecolor='black')

    ax.set_ylabel('Ratio to CNO')
    ax.set_title('Token Statistics (normalized to CNO)')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

    # Plot 3: Distribution shift metrics
    ax = axes[1, 0]
    dist_metrics = ['KL(MNO→CNO)', 'KL(CNO→MNO)', 'JS Div', 'Wasserstein']
    dist_vals = [
        dist_comparison.get('kl_mno_to_cno', 0.0),
        dist_comparison.get('kl_cno_to_mno', 0.0),
        dist_comparison.get('js_divergence', 0.0),
        dist_comparison.get('wasserstein_distance', 0.0),
    ]

    bars = ax.bar(dist_metrics, dist_vals, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Divergence')
    ax.set_title('Distribution Shift Metrics')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels
    for bar, val in zip(bars, dist_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom')

    # Plot 4: Per-category reconstruction MSE (if available)
    ax = axes[1, 1]
    mno_per_cat = mno_metrics.get('per_category_mse', {})
    cno_per_cat = cno_metrics.get('per_category_mse', {})

    if mno_per_cat and cno_per_cat:
        categories = sorted(mno_per_cat.keys())
        mno_cat_vals = [mno_per_cat[c] for c in categories]
        cno_cat_vals = [cno_per_cat[c] for c in categories]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width/2, cno_cat_vals, width, label='CNO', alpha=0.7,
                      color='#ff7f0e', edgecolor='black')
        bars2 = ax.bar(x + width/2, mno_cat_vals, width, label='MNO', alpha=0.7,
                      color='#1f77b4', edgecolor='black')

        ax.set_ylabel('Reconstruction MSE')
        ax.set_title('Per-Category Reconstruction MSE')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No per-category data available',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, style='italic')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Tokenization quality plot saved to {output_path}")


def plot_split_comparison(
    train_results: Dict[str, Any],
    holdout_results: Dict[str, Any],
    output_path: Path,
):
    """Generate train vs holdout comparison visualization.

    Args:
        train_results: Training split results
        holdout_results: Holdout split results
        output_path: Path to save figure
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Per-timestep MSE comparison
    ax = axes[0]
    train_mse = np.array(train_results.get('per_timestep_mse', []))
    holdout_mse = np.array(holdout_results.get('per_timestep_mse', []))

    if len(train_mse) > 0 and len(holdout_mse) > 0:
        timesteps = np.arange(len(train_mse))
        ax.plot(timesteps, train_mse, label='Training', linewidth=2, alpha=0.8)
        ax.plot(timesteps, holdout_mse, label='Holdout', linewidth=2, alpha=0.8)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('MSE')
        ax.set_title('Per-Timestep MSE: Train vs Holdout')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 2: Summary metrics comparison
    ax = axes[1]
    metrics_names = ['Mean MSE', 'Final MSE', 'Relative L2']
    train_vals = [
        train_results.get('mean_mse', 0.0),
        train_results.get('final_mse', 0.0),
        train_results.get('relative_l2', 0.0),
    ]
    holdout_vals = [
        holdout_results.get('mean_mse', 0.0),
        holdout_results.get('final_mse', 0.0),
        holdout_results.get('relative_l2', 0.0),
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_vals, width, label='Training', alpha=0.7,
                   color='#1f77b4', edgecolor='black')
    bars2 = ax.bar(x + width/2, holdout_vals, width, label='Holdout', alpha=0.7,
                   color='#ff7f0e', edgecolor='black')

    ax.set_ylabel('Value')
    ax.set_title('Summary Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Split comparison plot saved to {output_path}")
