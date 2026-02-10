"""Reusable visualization components for VQ-VAE dashboards.

This module provides DRY plotting utilities extracted from existing dashboards
to eliminate code duplication and provide consistent styling across all visualizations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
