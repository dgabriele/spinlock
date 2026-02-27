"""Reusable visualization components for VQ-VAE dashboards.

This module provides DRY plotting utilities extracted from existing dashboards
to eliminate code duplication and provide consistent styling across all visualizations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable


def plot_metric_heatmap(
    ax: Axes,
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    cmap: str | LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    annotations: Optional[List[List[str]]] = None,
    colorbar_label: str = "",
    fmt: str = ".3f",
) -> None:
    """Generic heatmap with annotations (N/M or floats).

    Replaces:
    - semantic_dashboard.plot_codebook_utilization() (lines 160-225)
    - engineering_dashboard.plot_utilization_heatmap() (lines 306-334)

    Features:
    - Adaptive text color (white on dark, black on light)
    - Custom colormap support
    - N/M or float annotations

    Args:
        ax: Matplotlib axes to plot on
        matrix: 2D array of values to display
        row_labels: Labels for Y-axis (rows)
        col_labels: Labels for X-axis (columns)
        title: Plot title
        cmap: Colormap name or object
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        annotations: Optional custom annotations (if None, use matrix values with fmt)
        colorbar_label: Label for colorbar
        fmt: Format string for numeric annotations (if annotations is None)
    """
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=8)

    # Rotate column labels if too many
    if len(col_labels) > 5:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add annotations
    if annotations is None:
        # Generate from matrix with format string
        annotations = [[f"{val:{fmt}}" for val in row] for row in matrix]

    # Determine text color based on background
    norm = Normalize(vmin=vmin, vmax=vmax)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            # Calculate luminance to choose text color
            normalized_val = norm(matrix[i, j])
            text_color = "white" if normalized_val < 0.5 else "black"

            ax.text(
                j,
                i,
                annotations[i][j],
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold" if annotations[i][j] else "normal",
            )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label, rotation=270, labelpad=15, fontsize=10)

    # Title
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)


def plot_dual_axis_curves(
    ax: Axes,
    epochs: List[int],
    primary_data: Dict[str, List[float]],
    secondary_data: Optional[Dict[str, List[float]]] = None,
    primary_ylabel: str = "Loss",
    secondary_ylabel: str = "Metric",
    title: str = "",
    log_scale_primary: bool = False,
    mark_best_epoch: Optional[int] = None,
) -> None:
    """Training curves with optional secondary axis.

    Replaces:
    - engineering_dashboard.plot_training_curves() (lines 210-236)

    Features:
    - Dual Y-axes for different scales
    - Best epoch marker
    - Log scale support

    Args:
        ax: Matplotlib axes to plot on
        epochs: List of epoch numbers
        primary_data: Dict mapping label → values for primary axis
        secondary_data: Optional dict for secondary axis (different scale)
        primary_ylabel: Label for primary Y-axis
        secondary_ylabel: Label for secondary Y-axis
        title: Plot title
        log_scale_primary: Use log scale for primary axis
        mark_best_epoch: Epoch number to mark with vertical line
    """
    # Plot primary data
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (label, values) in enumerate(primary_data.items()):
        color = colors[idx % len(colors)]
        ax.plot(epochs, values, label=label, color=color, linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(primary_ylabel, fontsize=10)
    if log_scale_primary:
        ax.set_yscale("log")

    # Plot secondary data if provided
    if secondary_data:
        ax2 = ax.twinx()
        for idx, (label, values) in enumerate(secondary_data.items(), start=len(primary_data)):
            color = colors[idx % len(colors)]
            ax2.plot(
                epochs,
                values,
                label=label,
                color=color,
                linewidth=2,
                linestyle="--",
                alpha=0.7,
            )
        ax2.set_ylabel(secondary_ylabel, fontsize=10)
        ax2.legend(loc="upper right", fontsize=8)

    # Mark best epoch if provided
    if mark_best_epoch is not None:
        ax.axvline(
            x=mark_best_epoch,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Best Epoch ({mark_best_epoch})",
        )

    # Legend
    ax.legend(loc="upper left" if secondary_data else "best", fontsize=8)

    # Title
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)


def plot_loss_components(
    ax: Axes,
    loss_history: List[Dict[str, float]],
    components: Dict[str, dict],
    title: str = "Loss Evolution",
    ylabel: str = "Loss Value",
) -> None:
    """Plot multiple loss components over training.

    Replaces:
    - engineering_dashboard.plot_loss_component_evolution() (lines 239-303)

    Args:
        ax: Matplotlib axes to plot on
        loss_history: List of dicts containing loss values per epoch
        components: Dict mapping component name → {color, label, linestyle}
        title: Plot title
        ylabel: Y-axis label
    """
    epochs = list(range(len(loss_history)))

    for comp_name, comp_config in components.items():
        values = [epoch_losses.get(comp_name, 0.0) for epoch_losses in loss_history]

        ax.plot(
            epochs,
            values,
            label=comp_config.get("label", comp_name),
            color=comp_config.get("color", "blue"),
            linestyle=comp_config.get("linestyle", "-"),
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)


def plot_metric_bars(
    ax: Axes,
    labels: List[str],
    values: List[float],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    colormap: str = "RdYlGn",
    horizontal: bool = False,
    annotate: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Bar chart with value-based coloring and annotations.

    Replaces:
    - engineering_dashboard.plot_reconstruction_bars() (lines 337-365)
    - semantic_dashboard.plot_category_sizes() (lines 111-157)

    Args:
        ax: Matplotlib axes to plot on
        labels: Bar labels
        values: Bar values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colormap: Colormap for value-based coloring
        horizontal: Use horizontal bars instead of vertical
        annotate: Add value annotations on bars
        vmin: Minimum value for colormap normalization (auto if None)
        vmax: Maximum value for colormap normalization (auto if None)
    """
    # Normalize values for coloring
    vmin = vmin if vmin is not None else min(values)
    vmax = vmax if vmax is not None else max(values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap)

    # Create bars
    positions = np.arange(len(labels))
    colors = [cmap(norm(v)) for v in values]

    if horizontal:
        bars = ax.barh(positions, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(xlabel if xlabel else "Value", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        # Annotations
        if annotate:
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    f" {value:.3f}",
                    ha="left",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
    else:
        bars = ax.bar(positions, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel(ylabel if ylabel else "Value", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)

        # Annotations
        if annotate:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="x" if horizontal else "y")


def plot_metric_card(
    ax: Axes,
    metrics: Dict[str, Tuple[str, str]],
    title: str = "Summary",
) -> None:
    """Display key metrics in a card format.

    Args:
        ax: Matplotlib axes to plot on
        metrics: Dict mapping metric name → (value_str, unit_str)
        title: Card title
    """
    # Turn off axis
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.95,
        title,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="top",
    )

    # Display metrics vertically
    num_metrics = len(metrics)
    y_positions = np.linspace(0.80, 0.15, num_metrics)

    for (metric_name, (value_str, unit_str)), y_pos in zip(metrics.items(), y_positions):
        # Metric name
        ax.text(
            0.05,
            y_pos,
            f"{metric_name}:",
            transform=ax.transAxes,
            fontsize=11,
            ha="left",
            va="center",
            fontweight="bold",
        )

        # Value
        value_display = f"{value_str} {unit_str}".strip()
        ax.text(
            0.95,
            y_pos,
            value_display,
            transform=ax.transAxes,
            fontsize=11,
            ha="right",
            va="center",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
        )

    # Border
    rect = plt.Rectangle(
        (0.02, 0.02),
        0.96,
        0.96,
        transform=ax.transAxes,
        fill=False,
        edgecolor="gray",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(rect)


def plot_discrimination_bars(
    ax: Axes,
    param_names: List[str],
    ratios: Dict[str, float],
    title: str = "Token Discrimination Ratio per Parameter",
) -> None:
    """Bar chart of discrimination ratios with a dashed reference line at 1.0.

    Interpretation: ratio = between-group distance / within-group distance (Hamming).
    Bars below 1.0 mean the tokenizer discriminates samples along that parameter.

    Args:
        ax: Matplotlib axes to plot on
        param_names: Ordered parameter names
        ratios: Dict mapping param_name → discrimination ratio
        title: Plot title
    """
    values = [ratios.get(n, 1.0) for n in param_names]
    colors = ["#2196F3" if v < 1.0 else "#FF5722" for v in values]

    positions = np.arange(len(param_names))
    bars = ax.bar(positions, values, color=colors, edgecolor="black", linewidth=0.5)

    # Reference line at 1.0 (random / no discrimination)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.8, label="No discrimination (1.0)")

    ax.set_xticks(positions)
    ax.set_xticklabels(param_names, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Discrimination Ratio (between / within)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="y")

    # Annotate bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )


def plot_category_scatter(
    ax: Axes,
    n_unique: np.ndarray,
    max_mi: np.ndarray,
    classifications: List[str],
    title: str = "Category Profile: Utilization vs Physical Signal",
) -> None:
    """Scatter: x = unique codes used, y = max normalized MI, color = classification.

    Classes:
      'collapsed'           → red  (single-code, no signal possible)
      'conserved'           → gray (multi-code but low MI — physics invariant)
      'physics-informative' → blue (multi-code AND high MI)

    Args:
        ax: Matplotlib axes to plot on
        n_unique: [n_cats] number of unique codes per category
        max_mi: [n_cats] max normalized MI value per category
        classifications: [n_cats] string classification per category
        title: Plot title
    """
    color_map = {
        "collapsed": "#F44336",          # red
        "conserved": "#9E9E9E",          # gray
        "physics-informative": "#2196F3", # blue
    }
    marker_map = {
        "collapsed": "x",
        "conserved": "o",
        "physics-informative": "^",
    }

    for cls in ["collapsed", "conserved", "physics-informative"]:
        mask = np.array([c == cls for c in classifications])
        if not mask.any():
            continue
        ax.scatter(
            n_unique[mask],
            max_mi[mask],
            c=color_map[cls],
            marker=marker_map[cls],
            label=f"{cls} ({mask.sum()})",
            alpha=0.7,
            s=40,
            edgecolors="none" if cls == "collapsed" else "black",
            linewidths=0.3,
        )

    ax.set_xlabel("Unique Codes Used", fontsize=10)
    ax.set_ylabel("Max Normalized MI (with any parameter)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)


# ── Discrete colormap helpers for similarity heatmaps ────────────────────────


# Base palette with high inter-stop contrast; subsampled/interpolated at runtime
# to match the dendrogram's merge-level count.
_SIMILARITY_PALETTE = [
    "#08000a", "#1a0060", "#0058b8", "#008880",
    "#00b060", "#60d000", "#d8d800", "#f0c000",
    "#f08000", "#f03000", "#d00060", "#a000a0",
    "#c060d0", "#e0a8c0", "#f0e0e0", "#fffff0",
]


def _dendrogram_tier_boundaries(
    linkage_matrix: np.ndarray,
    vmin: float,
    vmax: float,
    max_levels: int = 64,
) -> np.ndarray:
    """Extract non-uniform color boundaries from dendrogram merge heights.

    Each merge in the linkage matrix has a distance (column 2).  We convert
    these to similarity (1 - dist), keep those within [vmin, vmax], group
    nearby values into tiers (tolerance = 0.5% of range), and return the
    tier midpoints as boundaries.  This ensures each discrete color band
    in the heatmap corresponds to a real merge level in the dendrogram.

    Args:
        linkage_matrix: scipy linkage matrix [n-1, 4].
        vmin: Lower similarity bound (from percentile clipping).
        vmax: Upper similarity bound.
        max_levels: Ceiling on tier count.

    Returns:
        Sorted boundary array of length (n_tiers + 1), spanning [vmin, vmax].
    """
    merge_sims = np.sort(1.0 - linkage_matrix[:, 2])
    visible = merge_sims[(merge_sims >= vmin) & (merge_sims <= vmax)]

    if len(visible) < 2:
        return np.linspace(vmin, vmax, 9)  # fallback: 8 uniform bands

    # Group into tiers: merge sims within tolerance are the same level
    span = vmax - vmin
    tol = span * 0.005
    tier_centers = [visible[0]]
    for s in visible[1:]:
        if s - tier_centers[-1] > tol:
            tier_centers.append(s)

    # Cap at max_levels by subsampling evenly
    if len(tier_centers) > max_levels:
        indices = np.linspace(0, len(tier_centers) - 1, max_levels).astype(int)
        tier_centers = [tier_centers[i] for i in indices]

    # Build boundaries: midpoints between consecutive tier centers,
    # bookended by vmin and vmax.
    centers = np.array(tier_centers)
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    boundaries = np.concatenate([[vmin], midpoints, [vmax]])
    return boundaries


def _build_discrete_cmap(
    boundaries: np.ndarray,
) -> tuple:
    """Build a discrete (step-function) ListedColormap + BoundaryNorm.

    The number of colors = len(boundaries) - 1.  Colors are sampled from
    ``_SIMILARITY_PALETTE`` (16 high-contrast base colors), interpolating
    via a continuous colormap when more than 16 bands are needed.

    Args:
        boundaries: Non-uniform boundary array from _dendrogram_tier_boundaries.

    Returns:
        (ListedColormap, BoundaryNorm) ready for imshow.
    """
    n_colors = len(boundaries) - 1
    base = _SIMILARITY_PALETTE

    if n_colors <= len(base):
        # Subsample from the 16-color base
        indices = np.linspace(0, len(base) - 1, n_colors).astype(int)
        colors = [base[i] for i in indices]
    else:
        # More tiers than base colors: interpolate via a continuous version
        continuous = LinearSegmentedColormap.from_list("_sim_cont", base, N=256)
        colors = [continuous(i / (n_colors - 1)) for i in range(n_colors)]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def plot_similarity_dendrogram_heatmap(
    ax_dendro: Axes,
    ax_heatmap: Axes,
    similarity_matrix: np.ndarray,
    metric_name: str,
    bin_indices: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """Hierarchical clustering dendrogram (left) + reordered similarity heatmap (right).

    Canonical DRY implementation migrated from scripts/visualize_binned_jaccard.py.
    Uses hierarchical_cluster_rollouts() from utils for the linkage computation.

    Args:
        ax_dendro: Axes for the dendrogram (narrow, left panel)
        ax_heatmap: Axes for the reordered heatmap (wide, right panel)
        similarity_matrix: [N, N] pairwise similarity matrix
        metric_name: Human-readable metric name (e.g. "Jaccard", "JS")
        bin_indices: Optional list of (start, end) sample indices for annotation
    """
    from scipy.cluster.hierarchy import dendrogram
    from .utils import hierarchical_cluster_rollouts

    n = len(similarity_matrix)
    # Clip negative values from float rounding before converting to distance
    distance_matrix = np.clip(1.0 - similarity_matrix, 0.0, None)
    linkage_matrix = hierarchical_cluster_rollouts(1.0 - distance_matrix)  # pass similarity

    dendro = dendrogram(
        linkage_matrix,
        ax=ax_dendro,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        link_color_func=lambda k: "black",
    )
    total_samples = bin_indices[-1][1] if bin_indices else n
    ax_dendro.set_xlabel(f"{metric_name} Distance", fontsize=10)
    ax_dendro.set_title(
        f"Hierarchical Clustering\n({n:,} bins, {total_samples:,} samples)",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
    for line in ax_dendro.collections:
        line.set_linewidth(0.5)
        line.set_alpha(0.66)

    # Reorder heatmap by dendrogram leaf order
    idx = dendro["leaves"]
    sim_reordered = similarity_matrix[idx, :][:, idx]
    vmin = float(np.percentile(sim_reordered, 1))
    vmax = float(np.percentile(sim_reordered, 99))

    # ── Dynamic discrete colormap from dendrogram merge heights ──
    # Boundaries are placed at actual dendrogram tier transitions so each
    # color band in the heatmap maps to a real merge level in the tree.
    boundaries = _dendrogram_tier_boundaries(linkage_matrix, vmin, vmax)
    n_colors = len(boundaries) - 1
    cmap, norm = _build_discrete_cmap(boundaries)

    im = ax_heatmap.imshow(
        sim_reordered,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax_heatmap.set_xlabel("Bin Index (reordered by clustering)", fontsize=10)
    ax_heatmap.set_ylabel("Bin Index (reordered by clustering)", fontsize=10)

    # Show a subset of boundary ticks to avoid label overlap
    max_ticks = 12
    if len(boundaries) > max_ticks:
        tick_idx = np.linspace(0, len(boundaries) - 1, max_ticks).astype(int)
        tick_vals = boundaries[tick_idx]
    else:
        tick_vals = boundaries
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04,
                        spacing="proportional", ticks=tick_vals)
    cbar.ax.set_yticklabels([f"{b:.3f}" for b in tick_vals], fontsize=8)
    label_map = {
        "Jaccard": "Jaccard Similarity",
        "JS": "JS Similarity (1 − JS distance)",
    }
    cbar.set_label(label_map.get(metric_name, f"{metric_name} Similarity"), fontsize=10)

    avg = float(np.mean(similarity_matrix[np.triu_indices(n, k=1)]))
    ax_heatmap.set_title(
        f"{metric_name} Similarity Heatmap  |  avg={avg:.3f}  |  "
        f"range=[{vmin:.3f}, {vmax:.3f}]  |  {n_colors} dendrogram levels",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
