"""Engineering dashboard for VQ-VAE model visualization.

Creates a multi-panel figure showing:
- Model architecture schematic
- Training curves (loss, quality)
- Utilization heatmap (categories × levels)
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
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Model Architecture", fontsize=12, fontweight="bold")

    # Colors
    input_color = "#e1f5e1"
    encoder_color = "#fff4e1"
    cat_color = "#e1e8f5"
    level_color = "#f5e1e8"
    decoder_color = "#e8f5e1"

    # Input box
    rect = mpatches.FancyBboxPatch(
        (0.5, 2.5), 1.5, 1, boxstyle="round,pad=0.05", facecolor=input_color, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(1.25, 3, f"Input\n{data.input_dim}D", ha="center", va="center", fontsize=9)

    # Encoder box
    families = list(data.feature_families.keys())
    encoder_text = "Encoders\n" + "\n".join(families[:3])
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


def create_metrics_table(ax: Axes, data: VQVAECheckpointData) -> None:
    """Create summary metrics table."""
    ax.axis("off")
    ax.set_title("Summary Metrics", fontsize=12, fontweight="bold")

    # Collect metrics
    metrics = [
        ("Quality", f"{data.final_metrics.get('quality', 0):.4f}"),
        ("Utilization", f"{data.final_metrics.get('utilization', 0):.1%}"),
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
        figsize: Figure size in inches
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    # Load data
    data = load_vqvae_checkpoint(checkpoint_path)

    # Create figure with grid layout
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.2, 1], hspace=0.3, wspace=0.25)

    # Panel A: Architecture (top-left)
    ax_arch = fig.add_subplot(gs[0, 0])
    create_architecture_diagram(ax_arch, data)

    # Panel B: Training curves (top-right)
    ax_curves = fig.add_subplot(gs[0, 1])
    plot_training_curves(ax_curves, data)

    # Panel C: Utilization heatmap (middle, full width)
    ax_util = fig.add_subplot(gs[1, :])
    plot_utilization_heatmap(ax_util, data)

    # Panel D: Reconstruction MSE (bottom-left)
    ax_recon = fig.add_subplot(gs[2, 0])
    plot_reconstruction_bars(ax_recon, data)

    # Panel E: Summary metrics (bottom-right)
    ax_metrics = fig.add_subplot(gs[2, 1])
    create_metrics_table(ax_metrics, data)

    # Title
    fig.suptitle(
        "VQ-VAE Engineering Dashboard",
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
        print(f"Saved engineering dashboard to: {output_path}")

    return fig
