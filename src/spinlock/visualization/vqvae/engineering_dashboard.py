"""Engineering dashboard for VQ-VAE model visualization.

Creates a multi-panel figure showing:
- Model architecture schematic
- Training curves (loss, quality)
- Loss component evolution (recon, VQ, ortho, info, topo, entropy)
- Per-category reconstruction MSE
- Summary metrics table
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .utils import (
    VQVAECheckpointData,
    load_vqvae_checkpoint,
    extract_utilization_matrix,
    extract_reconstruction_mse,
)


def create_architecture_diagram(ax: Axes, data: VQVAECheckpointData) -> None:
    """Create architecture schematic on given axes.

    Shows flow: Input → Encoders → Categories → Levels → Decoder
    For pyramid models, shows multi-scale temporal encoding.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Model Architecture", fontsize=12, fontweight="bold")

    # Colors
    input_color = "#e1f5e1"
    encoder_color = "#fff4e1"
    pyramid_colors = ["#4a9c3f", "#6ece5a", "#8fd97e", "#b0e49c"]  # P0-P3
    cat_color = "#e1e8f5"
    level_color = "#f5e1e8"
    decoder_color = "#e8f5e1"

    # Input box
    rect = mpatches.FancyBboxPatch(
        (0.5, 2.5), 1.5, 1, boxstyle="round,pad=0.05", facecolor=input_color, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(1.25, 3, f"Input\n{data.input_dim}D", ha="center", va="center", fontsize=9)

    if data.is_pyramid:
        # Pyramid temporal encoder structure
        _draw_pyramid_architecture(ax, data, encoder_color, pyramid_colors, cat_color, level_color, decoder_color)
    else:
        # Standard encoder structure
        _draw_standard_architecture(ax, data, encoder_color, cat_color, level_color, decoder_color)


def _draw_standard_architecture(ax: Axes, data: VQVAECheckpointData,
                                encoder_color: str, cat_color: str,
                                level_color: str, decoder_color: str) -> None:
    """Draw standard (non-pyramid) architecture diagram."""
    # Encoder box
    families = list(data.feature_families.keys())
    # Abbreviate family names for display
    family_abbrev = {"temporal": "temp", "theta": "θ", "initial": "init", "architecture": "arch"}
    family_labels = [family_abbrev.get(f, f) for f in families]
    encoder_text = "Encoders\n" + "\n".join(family_labels[:4])  # Show up to 4 families
    rect = mpatches.FancyBboxPatch(
        (2.5, 2), 1.8, 2, boxstyle="round,pad=0.05", facecolor=encoder_color, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(3.4, 3, encoder_text, ha="center", va="center", fontsize=8)

    # Categories box
    rect = mpatches.FancyBboxPatch(
        (5, 2), 1.5, 2, boxstyle="round,pad=0.05", facecolor=cat_color, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(
        5.75,
        3,
        f"Categories\n{data.num_categories}\n({data.group_embedding_dim}D)",
        ha="center",
        va="center",
        fontsize=8,
    )

    # Levels box
    rect = mpatches.FancyBboxPatch(
        (7, 2), 1.2, 2, boxstyle="round,pad=0.05", facecolor=level_color, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(
        7.6,
        3,
        f"Levels\n×{data.num_levels}\n(VQ)",
        ha="center",
        va="center",
        fontsize=8,
    )

    # Decoder box
    rect = mpatches.FancyBboxPatch(
        (8.5, 2.5), 1.2, 1, boxstyle="round,pad=0.05", facecolor=decoder_color, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(9.1, 3, f"Decoder\n{data.input_dim}D", ha="center", va="center", fontsize=9)

    # Arrows
    arrow_props = dict(arrowstyle="->", color="black", lw=1.5)
    ax.annotate("", xy=(2.5, 3), xytext=(2, 3), arrowprops=arrow_props)
    ax.annotate("", xy=(5, 3), xytext=(4.3, 3), arrowprops=arrow_props)
    ax.annotate("", xy=(7, 3), xytext=(6.5, 3), arrowprops=arrow_props)
    ax.annotate("", xy=(8.5, 3), xytext=(8.2, 3), arrowprops=arrow_props)

    # Total codebooks annotation
    total_codebooks = data.num_categories * data.num_levels
    ax.text(
        5,
        0.8,
        f"Total: {data.num_categories} categories × {data.num_levels} levels = {total_codebooks} codebooks",
        ha="center",
        fontsize=9,
        style="italic",
    )


def _draw_pyramid_architecture(ax: Axes, data: VQVAECheckpointData,
                               encoder_color: str, pyramid_colors: list,
                               cat_color: str, level_color: str,
                               decoder_color: str) -> None:
    """Draw pyramid temporal encoding architecture diagram."""
    # Pyramid encoder box (temporal)
    rect = mpatches.FancyBboxPatch(
        (2.5, 1), 2.2, 4, boxstyle="round,pad=0.05", facecolor=encoder_color, edgecolor="black", linewidth=2
    )
    ax.add_patch(rect)
    ax.text(3.6, 4.5, "Pyramid Temporal", ha="center", va="center", fontsize=9, fontweight="bold")

    # Draw 4 pyramid levels as stacked boxes
    level_heights = [0.6, 0.6, 0.6, 0.6]
    level_y_positions = [1.3, 2.0, 2.7, 3.4]
    level_labels = ["P0 (1×)", "P1 (2×)", "P2 (4×)", "P3 (8×)"]

    for i, (y_pos, height, label, color) in enumerate(zip(level_y_positions, level_heights, level_labels, pyramid_colors)):
        rect = mpatches.FancyBboxPatch(
            (2.7, y_pos), 1.8, height, boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="black", linewidth=1
        )
        ax.add_patch(rect)
        ax.text(3.6, y_pos + height/2, label, ha="center", va="center", fontsize=7, fontweight="bold")

    # Categories box
    rect = mpatches.FancyBboxPatch(
        (5.3, 1.5), 1.5, 3, boxstyle="round,pad=0.05", facecolor=cat_color, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(
        6.05,
        3,
        f"Categories\n{data.num_categories}\n({data.group_embedding_dim}D)",
        ha="center",
        va="center",
        fontsize=8,
    )

    # Levels box
    rect = mpatches.FancyBboxPatch(
        (7.2, 2), 1.2, 2, boxstyle="round,pad=0.05", facecolor=level_color, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(
        7.8,
        3,
        f"Levels\n×{data.num_levels}\n(VQ)",
        ha="center",
        va="center",
        fontsize=8,
    )

    # Decoder box
    rect = mpatches.FancyBboxPatch(
        (8.7, 2.5), 1.2, 1, boxstyle="round,pad=0.05", facecolor=decoder_color, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(9.3, 3, f"Decoder\n{data.input_dim}D", ha="center", va="center", fontsize=9)

    # Arrows
    arrow_props = dict(arrowstyle="->", color="black", lw=1.5)
    ax.annotate("", xy=(2.5, 3), xytext=(2, 3), arrowprops=arrow_props)
    ax.annotate("", xy=(5.3, 3), xytext=(4.7, 3), arrowprops=arrow_props)
    ax.annotate("", xy=(7.2, 3), xytext=(6.8, 3), arrowprops=arrow_props)
    ax.annotate("", xy=(8.7, 3), xytext=(8.4, 3), arrowprops=arrow_props)

    # Pyramid info annotation
    ax.text(
        5,
        0.5,
        f"Pyramid: 4 scales (1×, 2×, 4×, 8× downsample) | {data.num_categories} categories | {data.num_levels} VQ levels",
        ha="center",
        fontsize=8,
        style="italic",
    )


def plot_training_curves(ax: Axes, data: VQVAECheckpointData) -> None:
    """Plot training and validation loss curves with quality on secondary axis."""
    epochs = range(1, len(data.train_loss) + 1)

    # Primary axis: Loss
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color="tab:blue")
    ax.plot(epochs, data.train_loss, "b-", alpha=0.7, label="Train Loss")
    ax.plot(epochs, data.val_loss, "b--", alpha=0.9, label="Val Loss")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_yscale("log")

    # Secondary axis: Quality
    ax2 = ax.twinx()
    quality = [m.get("quality", 0) for m in data.metrics_history]
    if quality and any(q > 0 for q in quality):
        ax2.set_ylabel("Quality", color="tab:green")
        ax2.plot(epochs, quality, "g-", alpha=0.8, label="Quality")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        ax2.set_ylim(0, 1.05)

    # Mark best epoch
    best_epoch = data.epoch
    ax.axvline(x=best_epoch, color="red", linestyle=":", alpha=0.5, label=f"Best (epoch {best_epoch})")

    ax.set_title("Training Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)


def plot_loss_component_evolution(ax: Axes, data: VQVAECheckpointData) -> None:
    """Plot evolution of all loss components over training.

    Shows how different loss components (reconstruction, VQ, orthogonality,
    informativeness, topographic, entropy) evolved during training.
    """
    if not data.loss_components_history:
        ax.text(0.5, 0.5, "Loss component history not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Loss Component Evolution", fontsize=12, fontweight="bold")
        return

    # Extract loss components over epochs
    epochs = list(range(1, len(data.loss_components_history) + 1))

    # Define components to plot with colors and labels
    components = {
        'reconstruction_global': {'label': 'Recon (global)', 'color': '#2E86AB', 'linestyle': '-', 'linewidth': 2.5},
        'reconstruction_category': {'label': 'Recon (category)', 'color': '#A23B72', 'linestyle': '--', 'linewidth': 2},
        'vq': {'label': 'VQ', 'color': '#F18F01', 'linestyle': '-', 'linewidth': 1.5},
        'orthogonality': {'label': 'Orthogonality', 'color': '#C73E1D', 'linestyle': '-', 'linewidth': 1.5},
        'informativeness': {'label': 'Informativeness', 'color': '#6A4C93', 'linestyle': '-', 'linewidth': 1.5},
        'topographic': {'label': 'Topographic', 'color': '#1B998B', 'linestyle': '-', 'linewidth': 1.5},
        'entropy': {'label': 'Entropy', 'color': '#E63946', 'linestyle': ':', 'linewidth': 1.5},
    }

    # Extract time series for each component
    component_series = {key: [] for key in components.keys()}
    for loss_dict in data.loss_components_history:
        for key in components.keys():
            value = loss_dict.get(key, np.nan)
            component_series[key].append(value)

    # Plot each component
    for key, config in components.items():
        series = component_series[key]
        # Only plot if we have data
        if series and not all(np.isnan(series)):
            ax.plot(epochs, series,
                   label=config['label'],
                   color=config['color'],
                   linestyle=config['linestyle'],
                   linewidth=config['linewidth'],
                   alpha=0.85)

    # Styling
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss Value", fontsize=10)
    ax.set_title("Loss Component Evolution", fontsize=12, fontweight="bold")
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, len(epochs))

    # Use log scale if dynamic range is large
    try:
        valid_values = [v for series in component_series.values()
                       for v in series if not np.isnan(v) and v > 0]
        if valid_values:
            max_val = max(valid_values)
            min_val = min(valid_values)
            if max_val / min_val > 50:  # Large dynamic range
                ax.set_yscale('log')
                ax.set_ylabel("Loss Value (log scale)", fontsize=10)
    except:
        pass  # Keep linear scale if anything goes wrong


def plot_utilization_heatmap(ax: Axes, data: VQVAECheckpointData) -> None:
    """Plot utilization heatmap: categories × levels."""
    matrix, cat_labels, level_labels = extract_utilization_matrix(data)

    # Create heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1.5)

    # Axis labels
    ax.set_xticks(range(len(level_labels)))
    ax.set_xticklabels(level_labels, fontsize=9)
    ax.set_yticks(range(len(cat_labels)))
    # Abbreviate category names
    short_labels = [c.replace("cluster_", "C") for c in cat_labels]
    ax.set_yticklabels(short_labels, fontsize=8)

    ax.set_xlabel("Level")
    ax.set_ylabel("Category")
    ax.set_title("Codebook Utilization", fontsize=12, fontweight="bold")

    # Annotate cells with values
    for i in range(len(cat_labels)):
        for j in range(len(level_labels)):
            val = matrix[i, j]
            color = "white" if val < 0.5 or val > 1.2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Utilization", fontsize=9)


def plot_reconstruction_bars(ax: Axes, data: VQVAECheckpointData) -> None:
    """Plot per-category reconstruction MSE as horizontal bar chart."""
    mse = extract_reconstruction_mse(data)

    if not mse:
        ax.text(0.5, 0.5, "No MSE data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Reconstruction MSE", fontsize=12, fontweight="bold")
        return

    # Use data.category_names to match VQ layer order (NOT sorted())
    cats = [c for c in data.category_names if c in mse]
    values = [mse[c] for c in cats]
    short_labels = [c.replace("cluster_", "C").replace("architecture_isolated", "arch") for c in cats]

    # Color by performance (green=good, red=bad)
    cmap = plt.get_cmap("RdYlGn_r")
    colors = cmap(np.array(values) / max(values))

    y_pos = range(len(cats))
    bars = ax.barh(y_pos, values, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("MSE")
    ax.set_title("Per-Category Reconstruction", fontsize=12, fontweight="bold")

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=7)


def plot_pyramid_reconstruction_mse(ax: Axes, data: VQVAECheckpointData) -> None:
    """Plot per-pyramid-level reconstruction MSE as horizontal bar chart."""
    if not data.is_pyramid or not data.pyramid_metrics:
        ax.text(0.5, 0.5, "Pyramid metrics not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-Pyramid-Level Quality", fontsize=12, fontweight="bold")
        return

    # Extract pyramid levels and average MSE
    levels = []
    mse_values = []
    colors_pyramid = ["#4a9c3f", "#6ece5a", "#8fd97e", "#b0e49c"]  # P0-P3

    for i, level_key in enumerate(['p0', 'p1', 'p2', 'p3']):
        if level_key in data.pyramid_metrics:
            metrics = data.pyramid_metrics[level_key]
            if metrics['avg_mse'] is not None:
                levels.append(level_key.upper())
                mse_values.append(metrics['avg_mse'])

    if not levels:
        ax.text(0.5, 0.5, "No pyramid MSE data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-Pyramid-Level Quality", fontsize=12, fontweight="bold")
        return

    y_pos = range(len(levels))
    bars = ax.barh(y_pos, mse_values, color=colors_pyramid[:len(levels)])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(levels, fontsize=10)
    ax.set_xlabel("Reconstruction MSE")
    ax.set_title("Per-Pyramid-Level Quality", fontsize=12, fontweight="bold")

    # Annotate bars with values
    for bar, val in zip(bars, mse_values):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)

    # Add scale annotations
    scale_labels = ["(1× fast)", "(2× med)", "(4× slow)", "(8× slowest)"]
    for i, (bar, scale) in enumerate(zip(bars, scale_labels[:len(levels)])):
        ax.text(0.05, bar.get_y() + bar.get_height() / 2,
                scale, va="center", fontsize=7, color="white", fontweight="bold")


def plot_variable_length_stats(ax: Axes, data: VQVAECheckpointData) -> None:
    """Plot variable-length trajectory statistics."""
    if not data.vl_enabled or not data.vl_metrics:
        ax.text(0.5, 0.5, "Variable-length metrics not available", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title("Variable-Length Statistics", fontsize=12, fontweight="bold")
        ax.axis("off")
        return

    ax.set_title("Variable-Length Statistics", fontsize=12, fontweight="bold")

    # Create 2×2 subplot grid within this ax
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    inner_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec(),
                                       hspace=0.4, wspace=0.4)
    ax.axis("off")

    # Get parent figure
    fig = ax.figure

    # Subplot 1: Length distribution
    ax1 = fig.add_subplot(inner_gs[0, 0])
    if 'length_distribution' in data.vl_metrics:
        lengths = sorted(data.vl_metrics['length_distribution'].keys())
        counts = [data.vl_metrics['length_distribution'][l] for l in lengths]
        ax1.bar(range(len(lengths)), counts, color='#4472C4')
        ax1.set_xticks(range(len(lengths)))
        ax1.set_xticklabels(lengths, fontsize=8)
        ax1.set_xlabel("Trajectory Length", fontsize=8)
        ax1.set_ylabel("Count", fontsize=8)
        ax1.set_title("Length Distribution", fontsize=9)

    # Subplot 2: Per-length quality
    ax2 = fig.add_subplot(inner_gs[0, 1])
    if 'per_length_quality' in data.vl_metrics:
        lengths = sorted(data.vl_metrics['per_length_quality'].keys())
        quality = [data.vl_metrics['per_length_quality'][l] for l in lengths]
        ax2.plot(lengths, quality, marker='o', color='#70AD47', linewidth=2)
        ax2.set_xlabel("Trajectory Length", fontsize=8)
        ax2.set_ylabel("Quality", fontsize=8)
        ax2.set_title("Quality vs Length", fontsize=9)
        ax2.grid(True, alpha=0.3)

    # Subplot 3: Active levels by length
    ax3 = fig.add_subplot(inner_gs[1, 0])
    if 'active_levels_by_length' in data.vl_metrics:
        lengths = sorted(data.vl_metrics['active_levels_by_length'].keys())
        active = [data.vl_metrics['active_levels_by_length'][l] for l in lengths]
        ax3.plot(lengths, active, marker='s', color='#FFC000', linewidth=2)
        ax3.set_xlabel("Trajectory Length", fontsize=8)
        ax3.set_ylabel("Avg Active Levels", fontsize=8)
        ax3.set_title("Pyramid Levels Used", fontsize=9)
        ax3.set_ylim(0, 4.5)
        ax3.grid(True, alpha=0.3)

    # Subplot 4: Summary metrics
    ax4 = fig.add_subplot(inner_gs[1, 1])
    ax4.axis("off")
    summary_text = "VL Summary\n\n"
    if 'masking_efficiency' in data.vl_metrics:
        summary_text += f"Masking Efficiency:\n{data.vl_metrics['masking_efficiency']:.1%}\n\n"
    if 'length_distribution' in data.vl_metrics:
        total_samples = sum(data.vl_metrics['length_distribution'].values())
        summary_text += f"Total Samples: {total_samples}\n\n"
    if 'per_length_quality' in data.vl_metrics:
        avg_quality = np.mean(list(data.vl_metrics['per_length_quality'].values()))
        summary_text += f"Avg Quality: {avg_quality:.3f}"

    ax4.text(0.5, 0.5, summary_text, ha="center", va="center",
             fontsize=9, transform=ax4.transAxes, family="monospace")


def create_metrics_table(ax: Axes, data: VQVAECheckpointData) -> None:
    """Create summary metrics table."""
    ax.axis("off")
    ax.set_title("Summary Metrics", fontsize=12, fontweight="bold")

    # Compute aggregate utilization from per-quantizer metrics
    util_keys = [k for k in data.final_metrics.keys() if k.endswith('/utilization')]
    if util_keys:
        avg_utilization = sum(data.final_metrics[k] for k in util_keys) / len(util_keys)
    else:
        avg_utilization = data.final_metrics.get('utilization', 0)

    # Collect metrics
    metrics = [
        ("Quality", f"{data.final_metrics.get('quality', 0):.4f}"),
        ("Utilization", f"{avg_utilization:.1%}"),
        ("Recon. Error", f"{data.final_metrics.get('reconstruction_error', 0):.4f}"),
        ("Best Val Loss", f"{data.best_val_loss:.4f}"),
        ("Categories", str(data.num_categories)),
        ("Levels", str(data.num_levels)),
        ("Input Dim", str(data.input_dim)),
        ("Epochs", str(len(data.train_loss))),
    ]

    # Create table
    table = ax.table(
        cellText=[[m[1]] for m in metrics],
        rowLabels=[m[0] for m in metrics],
        colLabels=["Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif key[1] == -1:
            cell.set_facecolor("#D9E2F3")


def create_engineering_dashboard(
    checkpoint_path: str | Path,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (16, 12),
    dpi: int = 150,
) -> Figure:
    """Create comprehensive engineering dashboard for VQ-VAE model.

    Args:
        checkpoint_path: Path to checkpoint directory
        output_path: Optional path to save figure (PNG)
        figsize: Figure size in inches (default: 16×12, auto-adjusted for pyramid)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    # Load data
    data = load_vqvae_checkpoint(checkpoint_path)

    # Adjust figure size for pyramid models
    if data.is_pyramid or data.vl_enabled:
        figsize = (18, 14)  # Larger for pyramid + VL panels

    # Create figure with grid layout
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Adjust layout based on model type
    if data.is_pyramid:
        # Pyramid layout: 4 rows to accommodate pyramid-specific panels
        gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1.2, 0.8, 1], hspace=0.35, wspace=0.25)

        # Panel A: Architecture (top-left)
        ax_arch = fig.add_subplot(gs[0, 0])
        create_architecture_diagram(ax_arch, data)

        # Panel B: Training curves (top-right)
        ax_curves = fig.add_subplot(gs[0, 1])
        plot_training_curves(ax_curves, data)

        # Panel C: Loss component evolution (row 2, full width)
        ax_loss = fig.add_subplot(gs[1, :])
        plot_loss_component_evolution(ax_loss, data)

        # Panel D: Pyramid-level MSE (row 3, left)
        ax_pyramid_mse = fig.add_subplot(gs[2, 0])
        plot_pyramid_reconstruction_mse(ax_pyramid_mse, data)

        # Panel E: Per-category MSE (row 3, right)
        ax_recon = fig.add_subplot(gs[2, 1])
        plot_reconstruction_bars(ax_recon, data)

        # Panel F: Summary metrics (row 4, left)
        ax_metrics = fig.add_subplot(gs[3, 0])
        create_metrics_table(ax_metrics, data)

        # Panel G: Variable-length stats (row 4, right) - if VL enabled
        ax_vl = fig.add_subplot(gs[3, 1])
        if data.vl_enabled:
            plot_variable_length_stats(ax_vl, data)
        else:
            ax_vl.axis("off")

        title_y = 0.98
        subtitle_y = 0.96
    else:
        # Standard layout (backward compatible)
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.2, 1], hspace=0.3, wspace=0.25)

        # Panel A: Architecture (top-left)
        ax_arch = fig.add_subplot(gs[0, 0])
        create_architecture_diagram(ax_arch, data)

        # Panel B: Training curves (top-right)
        ax_curves = fig.add_subplot(gs[0, 1])
        plot_training_curves(ax_curves, data)

        # Panel C: Loss component evolution (middle, full width)
        ax_loss = fig.add_subplot(gs[1, :])
        plot_loss_component_evolution(ax_loss, data)

        # Panel D: Reconstruction MSE (bottom-left)
        ax_recon = fig.add_subplot(gs[2, 0])
        plot_reconstruction_bars(ax_recon, data)

        # Panel E: Summary metrics (bottom-right)
        ax_metrics = fig.add_subplot(gs[2, 1])
        create_metrics_table(ax_metrics, data)

        title_y = 0.98
        subtitle_y = 0.95

    # Title
    title_text = "VQ-VAE Engineering Dashboard"
    if data.is_pyramid:
        title_text += " (Pyramid Temporal Encoding)"
    fig.suptitle(title_text, fontsize=16, fontweight="bold", y=title_y)

    # Add checkpoint path as subtitle
    checkpoint_name = Path(checkpoint_path).name
    fig.text(0.5, subtitle_y, f"Checkpoint: {checkpoint_name}", ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved engineering dashboard to: {output_path}")

    return fig
