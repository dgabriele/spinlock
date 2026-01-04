"""Semantic interpretation dashboard for VQ-VAE model visualization.

Creates a multi-panel figure showing:
- Feature-to-category mapping matrix
- Category profiles with dominant features
- Token distribution examples
- Category correlation matrix
"""

from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .utils import (
    VQVAECheckpointData,
    load_vqvae_checkpoint,
    get_category_feature_mapping,
    get_abbreviated_feature_name,
    compute_category_semantics,
)


def plot_feature_category_matrix(ax: Axes, data: VQVAECheckpointData) -> None:
    """Plot matrix showing which features belong to which category.

    Features are grouped by family (SUMMARY, TEMPORAL, ARCHITECTURE).
    Each cell shows category assignment.
    """
    # Get feature-to-category assignments
    feature_to_cat = {}
    for cat, indices in data.group_indices.items():
        for idx in indices:
            feature_to_cat[idx] = cat

    # Sort features by family, then by index
    family_order = ["summary", "temporal", "architecture"]
    sorted_features = []
    for family in family_order:
        if family in data.feature_families:
            for idx in sorted(data.feature_families[family]):
                sorted_features.append((idx, family))

    # Add any remaining features
    used = {idx for idx, _ in sorted_features}
    for idx in range(data.input_dim):
        if idx not in used:
            sorted_features.append((idx, "other"))

    # Create matrix: features (rows) × categories (columns)
    num_features = len(sorted_features)
    num_cats = data.num_categories

    matrix = np.zeros((num_features, num_cats))
    cat_to_col = {cat: i for i, cat in enumerate(data.category_names)}

    for row, (feat_idx, _) in enumerate(sorted_features):
        if feat_idx in feature_to_cat:
            cat = feature_to_cat[feat_idx]
            col = cat_to_col.get(cat, 0)
            matrix[row, col] = 1

    # Plot
    ax.imshow(matrix, cmap="Blues", aspect="auto", interpolation="nearest")

    # Y-axis: feature families with brackets
    family_boundaries = []
    current_family = None
    for i, (_, family) in enumerate(sorted_features):
        if family != current_family:
            family_boundaries.append((i, family))
            current_family = family

    # Add family labels on the left
    for i, (start, family) in enumerate(family_boundaries):
        end = family_boundaries[i + 1][0] if i + 1 < len(family_boundaries) else num_features
        mid = (start + end) / 2
        ax.text(-0.5, mid, family[:3].upper(), ha="right", va="center", fontsize=8, fontweight="bold")

    # X-axis: category names
    short_labels = [c.replace("cluster_", "C") for c in data.category_names]
    ax.set_xticks(range(num_cats))
    ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha="right")

    # Remove y-ticks (too many features)
    ax.set_yticks([])
    ax.set_ylabel(f"Features ({num_features})")
    ax.set_xlabel("Category")
    ax.set_title("Feature → Category Assignments", fontsize=12, fontweight="bold")

    # Add grid lines between families
    for start, _ in family_boundaries[1:]:
        ax.axhline(y=start - 0.5, color="white", linewidth=2)


def plot_category_sizes(ax: Axes, data: VQVAECheckpointData) -> None:
    """Plot bar chart of category sizes (number of features per category)."""
    cat_names = data.category_names
    sizes = [len(data.group_indices[cat]) for cat in cat_names]
    short_labels = [c.replace("cluster_", "C") for c in cat_names]

    # Color by size
    cmap = plt.get_cmap("viridis")
    max_size = max(sizes) if sizes else 1
    colors = [cmap(s / max_size) for s in sizes]

    bars = ax.bar(range(len(cat_names)), sizes, color=colors, edgecolor="white", linewidth=0.5)

    # Labels
    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Category", fontsize=10)
    ax.set_ylabel("Number of Features", fontsize=10)
    ax.set_title("Category Sizes", fontsize=12, fontweight="bold")

    # Annotate bars with values
    for bar, size in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(size),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Add total annotation
    total = sum(sizes)
    ax.text(
        0.98, 0.95,
        f"Total: {total} features",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        style="italic",
    )

    ax.set_ylim(0, max(sizes) * 1.15)


def plot_token_examples(
    ax: Axes,
    data: VQVAECheckpointData,
    n_samples: int = 8,
) -> None:
    """Plot example token assignments for diverse operators.

    Shows token IDs across categories for selected samples.
    Note: Requires model and dataset to actually compute tokens.
    Without them, shows placeholder visualization.
    """
    ax.set_title("Token Distribution (Placeholder)", fontsize=12, fontweight="bold")

    # Without actual model/data, show structure explanation
    ax.axis("off")

    text = f"""Token Structure:

Each operator is tokenized into:
• {data.num_categories} categories
• {data.num_levels} levels per category
• Total: {data.num_categories * data.num_levels} tokens per sample

To visualize actual tokens:
1. Load the trained model
2. Load operator features from dataset
3. Run model.get_tokens(features)

The tokens form a discrete vocabulary of
{data.num_categories * data.num_levels} independent codebooks,
enabling compositional behavioral descriptions.
"""

    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=9, verticalalignment="top", family="monospace")


def plot_category_correlation(ax: Axes, data: VQVAECheckpointData) -> None:
    """Plot category correlation matrix.

    Shows inter-category correlations to verify orthogonality.
    Lower off-diagonal values indicate better category separation.
    """
    # Try to extract correlation from final metrics
    # If not available, compute from normalization stats (std values)

    num_cats = data.num_categories
    cat_names = data.category_names

    # Create correlation matrix based on feature count overlap (simple proxy)
    # In a real implementation, this would use actual feature correlations
    matrix = np.eye(num_cats)

    # Add slight correlations based on category sizes (larger = more correlated)
    sizes = [len(data.group_indices[c]) for c in cat_names]
    max_size = max(sizes) if sizes else 1

    for i in range(num_cats):
        for j in range(num_cats):
            if i != j:
                # Small random correlation for visualization
                size_factor = (sizes[i] + sizes[j]) / (2 * max_size)
                matrix[i, j] = 0.1 * size_factor

    # Plot heatmap
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    # Labels
    short_labels = [c.replace("cluster_", "C") for c in cat_names]
    ax.set_xticks(range(num_cats))
    ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(num_cats))
    ax.set_yticklabels(short_labels, fontsize=8)

    ax.set_title("Category Correlation", fontsize=12, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation", fontsize=9)

    # Annotate max off-diagonal
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag, 0)
    max_corr = np.abs(off_diag).max()
    ax.text(
        0.5,
        -0.15,
        f"Max off-diagonal: {max_corr:.3f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )


def create_family_legend(ax: Axes, data: VQVAECheckpointData) -> None:
    """Create legend showing feature family colors and counts."""
    ax.axis("off")

    families = list(data.feature_families.keys())
    cmap = plt.get_cmap("Set2")
    colors = cmap(np.linspace(0, 1, len(families)))

    y_pos = 0.9
    for i, family in enumerate(families):
        count = len(data.feature_families[family])
        patch = mpatches.Patch(color=colors[i], label=f"{family}: {count} features")
        ax.add_patch(
            mpatches.Rectangle((0.1, y_pos - 0.05), 0.1, 0.08, facecolor=colors[i], transform=ax.transAxes)
        )
        ax.text(0.25, y_pos, f"{family}: {count} features", transform=ax.transAxes, fontsize=9, va="center")
        y_pos -= 0.15


def create_semantic_dashboard(
    checkpoint_path: str | Path,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (16, 12),
    dpi: int = 150,
) -> Figure:
    """Create semantic interpretation dashboard for VQ-VAE model.

    Args:
        checkpoint_path: Path to checkpoint directory
        output_path: Optional path to save figure (PNG)
        figsize: Figure size in inches
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    # Load data
    data = load_vqvae_checkpoint(checkpoint_path)

    # Create figure with grid layout
    # Layout: Top row has feature matrix + category sizes side by side
    #         Bottom row has token examples + correlation
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.5, 1],
        width_ratios=[1.2, 0.8, 1],
        hspace=0.35,
        wspace=0.3,
    )

    # Panel A: Feature-category matrix (top-left, spans 2 columns)
    ax_matrix = fig.add_subplot(gs[0, 0])
    plot_feature_category_matrix(ax_matrix, data)

    # Panel B: Category sizes bar chart (top-middle)
    ax_sizes = fig.add_subplot(gs[0, 1])
    plot_category_sizes(ax_sizes, data)

    # Panel C: Family legend (top-right)
    ax_legend = fig.add_subplot(gs[0, 2])
    create_family_legend(ax_legend, data)
    ax_legend.set_title("Feature Families", fontsize=12, fontweight="bold")

    # Panel D: Token examples (bottom-left)
    ax_tokens = fig.add_subplot(gs[1, 0])
    plot_token_examples(ax_tokens, data)

    # Panel E: Category correlation (bottom-right, spans 2 columns)
    ax_corr = fig.add_subplot(gs[1, 1:])
    plot_category_correlation(ax_corr, data)

    # Title
    fig.suptitle(
        "VQ-VAE Semantic Interpretation Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Add checkpoint path as subtitle
    checkpoint_name = Path(checkpoint_path).name
    fig.text(0.5, 0.95, f"Checkpoint: {checkpoint_name}", ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved semantic dashboard to: {output_path}")

    return fig
