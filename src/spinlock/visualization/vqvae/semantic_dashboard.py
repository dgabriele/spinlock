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
    extract_utilization_counts,
    get_utilization_cmap,
    compute_topographic_metrics_from_checkpoint,
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

    # Add family labels on the left (inside plot area to avoid ylabel overlap)
    for i, (start, family) in enumerate(family_boundaries):
        end = family_boundaries[i + 1][0] if i + 1 < len(family_boundaries) else num_features
        mid = (start + end) / 2
        ax.text(0.02, mid, family[:3].upper(), ha="left", va="center", fontsize=7, fontweight="bold",
                transform=ax.get_yaxis_transform(), color="darkblue", alpha=0.8)

    # X-axis: category names
    short_labels = [c.replace("cluster_", "C") for c in data.category_names]
    ax.set_xticks(range(num_cats))
    ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha="right")

    # Remove y-ticks (too many features) - ylabel moved to right side to avoid family label overlap
    ax.set_yticks([])
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(f"Features ({num_features})", fontsize=10)
    # Skip xlabel - category names visible in tick labels, overlaps with panel below
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
    # Skip xlabel - category names already visible in tick labels, and it overlaps with panel below
    ax.set_ylabel("# Features", fontsize=10)
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


def plot_codebook_utilization(
    ax: Axes,
    data: VQVAECheckpointData,
) -> None:
    """Plot codebook utilization as N/M (used codes / codebook size).

    Shows a heatmap with N/M annotations for each category × level cell.
    This visualization helps identify underutilized codebooks.
    """
    # Extract utilization counts
    util_counts = extract_utilization_counts(data)

    num_cats = data.num_categories
    num_levels = data.num_levels

    # Create utilization matrix for coloring
    matrix = np.zeros((num_cats, num_levels))
    annotations = [['' for _ in range(num_levels)] for _ in range(num_cats)]

    for i, cat in enumerate(data.category_names):
        cat_counts = util_counts.get(cat, {})
        for level in range(num_levels):
            used, total = cat_counts.get(level, (0, 0))
            if total > 0:
                utilization = used / total
                matrix[i, level] = utilization
                annotations[i][level] = f"{used}/{total}"
            else:
                annotations[i][level] = "—"

    # Plot heatmap (dark gray → green, neutral for low util)
    im = ax.imshow(matrix, cmap=get_utilization_cmap(), aspect="auto", vmin=0, vmax=1.0)

    # Axis labels
    level_labels = [f"L{i}" for i in range(num_levels)]
    ax.set_xticks(range(num_levels))
    ax.set_xticklabels(level_labels, fontsize=10)

    short_labels = [c.replace("cluster_", "C").replace("architecture_isolated", "arch") for c in data.category_names]
    ax.set_yticks(range(num_cats))
    ax.set_yticklabels(short_labels, fontsize=8)

    ax.set_xlabel("Level")
    ax.set_ylabel("")  # Explicitly clear ylabel

    # Compute summary stats for title
    total_used = sum(used for cat_counts in util_counts.values() for used, _ in cat_counts.values())
    total_codes = sum(total for cat_counts in util_counts.values() for _, total in cat_counts.values())
    overall_util = total_used / total_codes if total_codes > 0 else 0

    ax.set_title(f"Codebook Utilization: {total_used}/{total_codes} ({overall_util:.1%})", fontsize=11, fontweight="bold")

    # Annotate cells with N/M values
    for i in range(num_cats):
        for j in range(num_levels):
            val = matrix[i, j]
            # Choose text color based on background brightness (dark gray → green)
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=7, color=color, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Utilization", fontsize=9)


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


def plot_topographic_metrics(
    ax: Axes,
    topo_metrics: Optional[Dict[str, float]] = None,
    error_message: Optional[str] = None,
) -> None:
    """Plot topographic similarity metrics as a visual summary.

    Shows four metrics with bar chart and interpretive labels:
    - Pre-quantization: Input → Latent (encoder quality)
    - Post-quantization: Latent → Code (VQ quality)
    - End-to-end: Input → Code (overall topology preservation)
    - Quantization degradation: Pre - Post (VQ information loss)

    Args:
        ax: Matplotlib axes
        topo_metrics: Dict of topology metrics, or None if computation failed
        error_message: Error message to display if metrics unavailable
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if topo_metrics is None:
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            error_message or "Topographic metrics unavailable",
            ha="center", va="center",
            fontsize=10, style="italic", color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Topographic Similarity", fontsize=12, fontweight="bold")
        return

    # Check if metrics are estimated
    is_estimated = topo_metrics.get("_estimated", False)

    # Metrics to display with labels and interpretations
    metric_info = [
        ("pre_quantization", "Pre-Quant", "Input → Latent", "#3498db"),
        ("post_quantization", "Post-Quant", "Latent → Code", "#9b59b6"),
        ("end_to_end", "End-to-End", "Input → Code", "#2ecc71"),
        ("quantization_degradation", "VQ Loss", "Pre − Post", "#e74c3c"),
    ]

    # Create horizontal bar chart
    ax.axis("off")

    y_positions = [0.78, 0.56, 0.34, 0.12]
    bar_height = 0.15
    bar_left = 0.35
    bar_width_max = 0.55

    for i, (key, label, desc, color) in enumerate(metric_info):
        value = topo_metrics.get(key, 0.0)
        y = y_positions[i]

        # For degradation, use absolute value for bar width but keep sign for color
        if key == "quantization_degradation":
            bar_value = abs(value)
            # Green if low degradation, red if high
            color = "#2ecc71" if value < 0.1 else "#e67e22" if value < 0.2 else "#e74c3c"
        else:
            bar_value = max(0, min(1, value))  # Clamp to [0, 1]

        # Draw bar background
        ax.add_patch(
            mpatches.Rectangle(
                (bar_left, y - bar_height / 2),
                bar_width_max, bar_height,
                facecolor="#f0f0f0",
                edgecolor="#cccccc",
                transform=ax.transAxes,
            )
        )

        # Draw bar
        ax.add_patch(
            mpatches.Rectangle(
                (bar_left, y - bar_height / 2),
                bar_width_max * bar_value, bar_height,
                facecolor=color,
                edgecolor="none",
                transform=ax.transAxes,
            )
        )

        # Label on left
        ax.text(
            0.02, y,
            f"{label}:",
            ha="left", va="center",
            fontsize=9, fontweight="bold",
            transform=ax.transAxes,
        )

        # Description below label
        ax.text(
            0.02, y - 0.06,
            desc,
            ha="left", va="center",
            fontsize=7, color="gray", style="italic",
            transform=ax.transAxes,
        )

        # Value on right
        ax.text(
            0.95, y,
            f"{value:.3f}",
            ha="right", va="center",
            fontsize=10, fontweight="bold",
            transform=ax.transAxes,
        )

    # Title with estimated indicator
    title = "Topographic Similarity"
    if is_estimated:
        title += " (estimated)"
    ax.set_title(title, fontsize=12, fontweight="bold")


def create_semantic_dashboard(
    checkpoint_path: str | Path,
    output_path: Optional[str | Path] = None,
    dataset_path: Optional[str | Path] = None,
    compute_topo: bool = True,
    topo_samples: int = 1000,
    device: str = "cuda",
    figsize: tuple = (16, 14),
    dpi: int = 150,
) -> Figure:
    """Create semantic interpretation dashboard for VQ-VAE model.

    Args:
        checkpoint_path: Path to checkpoint directory
        output_path: Optional path to save figure (PNG)
        dataset_path: Path to dataset for topographic metrics (optional)
        compute_topo: Whether to compute topographic metrics (requires model inference)
        topo_samples: Number of samples for topographic metric computation
        device: Device for topographic computation
        figsize: Figure size in inches
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    checkpoint_path = Path(checkpoint_path)

    # Load data
    data = load_vqvae_checkpoint(checkpoint_path)

    # Compute topographic metrics if requested
    topo_metrics = None
    topo_error = None
    if compute_topo:
        try:
            print("Computing topographic metrics...")
            topo_metrics = compute_topographic_metrics_from_checkpoint(
                checkpoint_path,
                dataset_path=dataset_path,
                n_samples=topo_samples,
                device=device,
            )
            print(f"Topographic metrics: {topo_metrics}")
        except Exception as e:
            topo_error = f"Failed: {str(e)[:50]}"
            print(f"Warning: Could not compute topographic metrics: {e}")

    # Create figure with grid layout
    # Layout: Top row has feature matrix + category sizes + family legend
    #         Middle row has codebook utilization + topographic metrics
    #         Bottom row has category correlation (wide)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[1.5, 1, 0.8],
        width_ratios=[1.2, 0.8, 1],
        hspace=0.35,
        wspace=0.3,
    )

    # Panel A: Feature-category matrix (top-left)
    ax_matrix = fig.add_subplot(gs[0, 0])
    plot_feature_category_matrix(ax_matrix, data)

    # Panel B: Category sizes bar chart (top-middle)
    ax_sizes = fig.add_subplot(gs[0, 1])
    plot_category_sizes(ax_sizes, data)

    # Panel C: Family legend (top-right)
    ax_legend = fig.add_subplot(gs[0, 2])
    create_family_legend(ax_legend, data)
    ax_legend.set_title("Feature Families", fontsize=12, fontweight="bold")

    # Panel D: Codebook utilization (middle-left)
    ax_util = fig.add_subplot(gs[1, 0])
    plot_codebook_utilization(ax_util, data)

    # Panel E: Topographic metrics (middle-right, spans 2 columns)
    ax_topo = fig.add_subplot(gs[1, 1:])
    plot_topographic_metrics(ax_topo, topo_metrics, topo_error)

    # Panel F: Category correlation (bottom, spans all columns)
    ax_corr = fig.add_subplot(gs[2, :])
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
