"""Binned token similarity dashboard.

Single-figure dashboard: hierarchical-clustering dendrogram + reordered similarity
heatmap for binned token sets.  Wraps compute_binned_similarity() from utils and
plot_similarity_dendrogram_heatmap() from components into one callable.

Usage::

    from spinlock.visualization.vqvae import create_binned_similarity_dashboard

    fig = create_binned_similarity_dashboard(
        tokenized_h5_path="datasets/qbm_50k_tokenized.h5",
        output_path="visualizations/binned_similarity_jaccard.png",
        metric="jaccard",
        n_bins=5000,
    )
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def create_binned_similarity_dashboard(
    tokenized_h5_path: str,
    output_path: Optional[str | Path] = None,
    n_bins: int = 5000,
    metric: str = "jaccard",
    families: Optional[list] = None,
    figsize: Tuple[int, int] = (20, 16),
    dpi: int = 150,
    checkpoint_path: Optional[str] = None,
) -> Figure:
    """Single-figure dashboard: dendrogram + similarity heatmap for binned token sets.

    Args:
        tokenized_h5_path: Path to pretokenized HDF5 dataset
        output_path: If given, save PNG (and PDF) to this path
        n_bins: Target number of bins (consecutive sample groups)
        metric: 'jaccard' (set-based), 'js' (Jensen-Shannon frequency-based),
            or 'weighted-hamming' (hierarchy-weighted embedding distance)
        families: Optional list of token families to include (e.g. ['temporal']).
            Excludes theta/IC tokens when set, isolating the temporal encoder's diversity.
            None (default) includes all families.
        figsize: Figure size in inches
        dpi: Figure resolution
        checkpoint_path: Path to VQ tokenizer checkpoint directory (required for
            weighted-hamming metric, which needs codebook embeddings)

    Returns:
        Matplotlib Figure object
    """
    from .utils import compute_binned_similarity
    from .components import plot_similarity_dendrogram_heatmap

    metric_label = {"jaccard": "Jaccard", "js": "JS", "weighted-hamming": "Weighted Hamming"}[metric]
    families_label = "+".join(sorted(families)) if families else "all"

    print("=" * 70)
    print(
        f"BINNED {metric_label.upper()} SIMILARITY DASHBOARD  "
        f"(→ {n_bins:,} bins, families={families_label})"
    )
    print("=" * 70)

    similarity_matrix, bin_indices = compute_binned_similarity(
        tokenized_h5_path=tokenized_h5_path,
        n_bins=n_bins,
        metric=metric,
        families=families,
        checkpoint_path=checkpoint_path,
    )

    # Layout: narrow dendrogram (1/6) + wide heatmap (5/6)
    fig = plt.figure(figsize=figsize)
    ax_dendro = plt.subplot2grid((1, 6), (0, 0), rowspan=1)
    ax_heatmap = plt.subplot2grid((1, 6), (0, 1), colspan=5)

    plot_similarity_dendrogram_heatmap(
        ax_dendro=ax_dendro,
        ax_heatmap=ax_heatmap,
        similarity_matrix=similarity_matrix,
        metric_name=metric_label,
        bin_indices=bin_indices,
    )

    # Footer stats
    n = len(similarity_matrix)
    avg = float(np.mean(similarity_matrix[np.triu_indices(n, k=1)]))
    total_samples = bin_indices[-1][1] if bin_indices else n
    stats_parts = [
        f"Metric: {metric_label}",
        f"Families: {families_label}",
        f"Avg Similarity: {avg:.3f}",
        f"Total Samples: {total_samples:,}",
        f"Bins: {n:,}",
        f"~{total_samples // n} samples/bin",
    ]
    fig.text(
        0.5, 0.01,
        " | ".join(stats_parts),
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure .png extension
        png_path = output_path.with_suffix(".png") if output_path.suffix == "" else output_path
        if png_path.suffix != ".png":
            png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        print(f"  ✓ Saved PNG: {png_path}")
        pdf_path = png_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  ✓ Saved PDF: {pdf_path}")

    print("=" * 70)
    print("✓ Binned similarity dashboard complete")
    print("=" * 70)
    return fig
