#!/usr/bin/env python3
"""
Generate Jaccard similarity visualization with binning for large datasets.

Bins 50K samples into ~5K bins for visualization while preserving full dataset coverage.
"""

import sys
sys.path.insert(0, 'src')

import h5py
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def load_token_sets(tokenized_dataset_path: str, max_samples: int = None):
    """Load token sets from pretokenized dataset."""
    print(f"Loading token sets from {tokenized_dataset_path}...")

    token_sets = []
    with h5py.File(tokenized_dataset_path, 'r') as f:
        # Access tokens group
        tokens_group = f['tokens']

        # Get all token keys (temporal, initial, theta - all contribute to diversity)
        token_keys = list(tokens_group.keys())

        n_samples = tokens_group[token_keys[0]].shape[0]
        if max_samples:
            n_samples = min(n_samples, max_samples)

        print(f"  Found {len(token_keys)} token categories")
        print(f"  Processing {n_samples:,} samples...")

        for i in range(n_samples):
            # Collect all tokens for this sample across categories
            sample_tokens = set()
            for key in token_keys:
                tokens = tokens_group[key][i]
                # Each token is a single integer (shape is (50000,) not (50000, n))
                sample_tokens.add(f"{key}_{tokens}")
            token_sets.append(sample_tokens)

    print(f"  ✓ Loaded {len(token_sets):,} token sets")
    return token_sets


def bin_token_sets(token_sets, n_bins=5000):
    """
    Bin token sets into groups for visualization.

    Each bin contains ~(len(token_sets) / n_bins) consecutive samples.
    Representative token set = union of all tokens in bin.
    """
    print(f"\nBinning {len(token_sets):,} samples into {n_bins:,} bins...")

    bin_size = max(1, len(token_sets) // n_bins)
    actual_n_bins = (len(token_sets) + bin_size - 1) // bin_size

    print(f"  Bin size: ~{bin_size} samples/bin")
    print(f"  Actual bins: {actual_n_bins:,}")

    binned_sets = []
    bin_indices = []

    for i in range(0, len(token_sets), bin_size):
        end_idx = min(i + bin_size, len(token_sets))

        # Union of all tokens in this bin
        bin_tokens = set()
        for j in range(i, end_idx):
            bin_tokens.update(token_sets[j])

        binned_sets.append(bin_tokens)
        bin_indices.append((i, end_idx))

    print(f"  ✓ Created {len(binned_sets):,} bins")
    return binned_sets, bin_indices


def compute_jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_similarity_matrix(token_sets):
    """Compute pairwise Jaccard similarity matrix."""
    n = len(token_sets)
    print(f"\nComputing {n:,}×{n:,} Jaccard similarity matrix...")

    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n} ({100*i/n:.1f}%)")

        for j in range(i, n):
            sim = compute_jaccard_similarity(token_sets[i], token_sets[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    avg_sim = np.mean(similarity_matrix[np.triu_indices(n, k=1)])
    print(f"  ✓ Average pairwise similarity: {avg_sim:.3f}")

    return similarity_matrix


def plot_similarity_dendrogram(similarity_matrix, output_path, bin_indices=None):
    """Generate hierarchical clustering dendrogram with similarity heatmap."""
    print(f"\nGenerating visualization...")

    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix
    condensed_dist = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')

    # Create figure with adjusted layout (no top dendrogram)
    fig = plt.figure(figsize=(20, 16))

    # Dendrogram on left only
    ax_dendro_left = plt.subplot2grid((1, 6), (0, 0), rowspan=1)
    dendro_left = dendrogram(linkage_matrix, ax=ax_dendro_left,
                             orientation='left', no_labels=True,
                             color_threshold=0,
                             link_color_func=lambda k: 'black')
    ax_dendro_left.set_xlabel('Jaccard Distance', fontsize=12)
    ax_dendro_left.set_title(f'Hierarchical Clustering\n({len(similarity_matrix):,} bins, 50K samples)',
                             fontsize=12, fontweight='bold', pad=10)

    # Make dendrogram lines thinner with opacity
    for line in ax_dendro_left.collections:
        line.set_linewidth(0.5)
        line.set_alpha(0.66)

    # Similarity heatmap
    ax_heatmap = plt.subplot2grid((1, 6), (0, 1), colspan=5)

    # Reorder matrix according to dendrogram
    idx = dendro_left['leaves']
    similarity_reordered = similarity_matrix[idx, :][:, idx]

    # Normalize color scale to actual data range for better contrast
    vmin = np.percentile(similarity_reordered, 1)  # Use 1st percentile to avoid outliers
    vmax = np.percentile(similarity_reordered, 99)  # Use 99th percentile

    im = ax_heatmap.imshow(similarity_reordered, cmap='viridis', aspect='auto',
                           interpolation='nearest', vmin=vmin, vmax=vmax)
    ax_heatmap.set_xlabel(f'Bin Index (reordered by clustering)', fontsize=12)
    ax_heatmap.set_ylabel(f'Bin Index (reordered by clustering)', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Jaccard Similarity', fontsize=12)

    plt.tight_layout()

    # Add stats annotation below the figure
    avg_sim = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
    stats_parts = [f"Avg Similarity: {avg_sim:.3f}", f"Range: [{vmin:.3f}, {vmax:.3f}]"]
    if bin_indices:
        total_samples = bin_indices[-1][1]
        stats_parts.extend([
            f"Total Samples: {total_samples:,}",
            f"Bins: {len(bin_indices):,}",
            f"~{total_samples // len(bin_indices)} samples/bin"
        ])

    stats_text = " | ".join(stats_parts)
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    print(f"  ✓ Color scale normalized to [{vmin:.3f}, {vmax:.3f}] for better contrast")

    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ Saved vector version to {pdf_path}")

    plt.close()


def main():
    # Configuration
    tokenized_dataset = "datasets/qbm_50k_pretokenized.h5"
    output_dir = Path("visualizations/vqvae_fixed_metric_50k_binned")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_bins = 5000  # Bin 50K samples into 5K bins for visualization

    print("="*80)
    print("BINNED JACCARD SIMILARITY ANALYSIS (50K → 5K bins)")
    print("="*80)

    # Load all 50K token sets
    token_sets = load_token_sets(tokenized_dataset, max_samples=50000)

    # Bin into groups
    binned_sets, bin_indices = bin_token_sets(token_sets, n_bins=n_bins)

    # Compute similarity matrix for bins
    similarity_matrix = compute_similarity_matrix(binned_sets)

    # Generate visualization
    output_path = output_dir / "rollout_similarity_binned.png"
    plot_similarity_dendrogram(similarity_matrix, output_path, bin_indices)

    print("\n" + "="*80)
    print(f"✓ COMPLETE! Visualizations saved to {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
