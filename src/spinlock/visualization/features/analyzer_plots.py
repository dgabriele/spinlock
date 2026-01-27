"""Visualization plots for feature analyzer.

Generates diagnostic plots for feature quality analysis:
- Variance distribution across features
- Correlation heatmap for redundancy detection
- NaN frequency bar chart
"""

import numpy as np
from pathlib import Path
from typing import List

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def plot_variance_distribution(
    variances: np.ndarray,
    threshold: float,
    output_path: Path,
    max_features: int = 1000
) -> None:
    """Plot variance distribution with threshold line.

    Args:
        variances: Per-feature variance values [D]
        threshold: Variance threshold for pruning
        output_path: Output file path
        max_features: Max features to plot (sample if more)
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn required for plotting")

    # Sample if too many features
    if len(variances) > max_features:
        indices = np.random.choice(len(variances), max_features, replace=False)
        variances = variances[indices]

    # Use log scale for better visualization
    log_variances = np.log10(variances + 1e-15)  # Add small constant to handle zeros

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(log_variances, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.log10(threshold), color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.2e})')
    ax1.set_xlabel('log10(Variance)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Feature Variance Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Sorted variance plot
    sorted_variances = np.sort(variances)
    ax2.plot(sorted_variances, linewidth=1)
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.2e})')
    ax2.set_xlabel('Feature Index (sorted)', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_title('Sorted Feature Variances', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    max_features: int = 100
) -> None:
    """Plot correlation heatmap for subset of features.

    Args:
        corr_matrix: Correlation matrix [D, D]
        feature_names: List of feature names
        output_path: Output file path
        max_features: Max features to show in heatmap
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn required for plotting")

    # Sample features if too many
    if len(feature_names) > max_features:
        # Sample uniformly across feature space
        indices = np.linspace(0, len(feature_names) - 1, max_features, dtype=int)
        corr_matrix = corr_matrix[indices][:, indices]
        feature_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap with better color scheme
    sns.heatmap(
        corr_matrix,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation', 'shrink': 0.8},
        ax=ax
    )

    ax.set_title(
        f'Feature Correlation Matrix\n(Sampled {len(feature_names)} features)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Remove tick labels if too many features
    if len(feature_names) > 50:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_nan_frequency(
    nan_fractions: np.ndarray,
    threshold: float,
    output_path: Path,
    max_features: int = 500
) -> None:
    """Plot NaN frequency bar chart.

    Args:
        nan_fractions: Per-feature NaN fractions [D]
        threshold: NaN threshold for pruning
        output_path: Output file path
        max_features: Max features to show
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn required for plotting")

    # Sample if too many features
    if len(nan_fractions) > max_features:
        indices = np.linspace(0, len(nan_fractions) - 1, max_features, dtype=int)
        nan_fractions = nan_fractions[indices]

    # Filter to only show features with some NaNs for clarity
    has_nans = nan_fractions > 0
    if has_nans.any():
        indices_with_nans = np.where(has_nans)[0]
        nan_fractions_filtered = nan_fractions[has_nans]
    else:
        # If no NaNs, show all zeros
        indices_with_nans = np.arange(min(20, len(nan_fractions)))
        nan_fractions_filtered = nan_fractions[indices_with_nans]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Color bars based on threshold
    colors = ['red' if frac > threshold else 'steelblue'
              for frac in nan_fractions_filtered]

    ax.bar(range(len(nan_fractions_filtered)), nan_fractions_filtered,
           color=colors, edgecolor='black', linewidth=0.5)

    ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold:.2%})')

    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('NaN Fraction', fontsize=12)
    ax.set_title('NaN Frequency per Feature\n(Only showing features with NaNs)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, min(1.0, max(nan_fractions_filtered) * 1.1))
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_analysis_summary(
    results: dict,
    output_path: Path
) -> None:
    """Plot comprehensive analysis summary with multiple subplots.

    Args:
        results: Full analysis results dictionary
        output_path: Output file path
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn required for plotting")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Pie chart of feature categories
    ax1 = fig.add_subplot(gs[0, 0])
    summary = results["summary"]
    labels = ['Keep', 'Zero Variance', 'High NaN', 'Redundant']
    sizes = [
        summary['recommended_keep'],
        summary['zero_variance_features'],
        summary['high_nan_features'],
        summary['redundant_features']
    ]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90)
    ax1.set_title('Feature Distribution', fontsize=12, fontweight='bold')

    # 2. Variance histogram
    ax2 = fig.add_subplot(gs[0, 1])
    variances = np.array(results["analysis"]["variance"]["per_feature_variance"])
    log_variances = np.log10(variances + 1e-15)
    threshold = results["metadata"]["thresholds"]["variance"]
    ax2.hist(log_variances, bins=40, edgecolor='black', alpha=0.7)
    ax2.axvline(np.log10(threshold), color='red', linestyle='--')
    ax2.set_xlabel('log10(Variance)')
    ax2.set_ylabel('Count')
    ax2.set_title('Variance Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Coefficient of variation
    ax3 = fig.add_subplot(gs[0, 2])
    cv = np.array(results["analysis"]["variance"]["per_feature_cv"])
    cv_filtered = cv[cv < 10]  # Filter extreme values for visualization
    ax3.hist(cv_filtered, bins=40, edgecolor='black', alpha=0.7, color='green')
    ax3.set_xlabel('Coefficient of Variation (CV)')
    ax3.set_ylabel('Count')
    ax3.set_title('Feature Stability (CV)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Correlation pairs histogram
    ax4 = fig.add_subplot(gs[1, 0])
    pairs = results["analysis"]["correlation"]["highly_correlated_pairs"]
    if pairs:
        correlations = [abs(p['correlation']) for p in pairs]
        ax4.hist(correlations, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax4.set_xlabel('|Correlation|')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Highly Correlated Pairs (n={len(pairs)})',
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No highly correlated pairs',
                ha='center', va='center', fontsize=12)
        ax4.set_title('Highly Correlated Pairs', fontsize=12, fontweight='bold')

    # 5. NaN frequency (top features)
    ax5 = fig.add_subplot(gs[1, 1])
    nan_fracs = np.array(results["analysis"]["nan_analysis"]["per_feature_nan_fraction"])
    top_nan_indices = np.argsort(nan_fracs)[-20:]  # Top 20 worst
    top_nan_values = nan_fracs[top_nan_indices]
    threshold_nan = results["metadata"]["thresholds"]["nan"]

    colors = ['red' if v > threshold_nan else 'steelblue' for v in top_nan_values]
    ax5.barh(range(len(top_nan_values)), top_nan_values, color=colors, edgecolor='black')
    ax5.axvline(threshold_nan, color='red', linestyle='--', label=f'Threshold ({threshold_nan:.1%})')
    ax5.set_xlabel('NaN Fraction')
    ax5.set_ylabel('Feature Rank')
    ax5.set_title('Top 20 Features by NaN Frequency', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, axis='x', alpha=0.3)

    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    stats_text = [
        ['Metric', 'Value'],
        ['─' * 20, '─' * 10],
        ['Total Features', f"{summary['features_analyzed']}"],
        ['Features to Keep', f"{summary['recommended_keep']}"],
        ['Features to Prune', f"{summary['recommended_prune']}"],
        ['', ''],
        ['Zero Variance', f"{summary['zero_variance_features']}"],
        ['High NaN', f"{summary['high_nan_features']}"],
        ['Redundant', f"{summary['redundant_features']}"],
        ['', ''],
        ['Compression', f"{summary['compression_ratio']:.1%}"],
    ]

    table = ax6.table(cellText=stats_text, cellLoc='left',
                     loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax6.set_title('Analysis Summary', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle(
        f"Feature Analysis Report - {results['metadata']['feature_family']}",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
