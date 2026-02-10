"""Roundtrip consistency and semantic structure dashboard for VQ-VAE models.

This dashboard visualizes:
1. Roundtrip training dynamics (encode(decode(tokens)) == tokens)
2. Semantic structure (combinatorial space size, token frequencies)
3. Compositional properties (level-wise composition, co-occurrence)
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

from .utils import (
    load_vqvae_checkpoint,
    extract_roundtrip_metrics,
    compute_combinatorial_space_size,
    analyze_token_frequencies,
    compute_token_cooccurrence,
    extract_level_composition,
    VQVAECheckpointData,
)
from .components import (
    plot_metric_heatmap,
    plot_dual_axis_curves,
    plot_metric_bars,
    plot_metric_card,
)


def plot_roundtrip_training_curves(ax, data: VQVAECheckpointData) -> None:
    """Plot roundtrip loss evolution during training."""
    metrics = extract_roundtrip_metrics(data)

    if not metrics["training_curve"]:
        # No roundtrip data available
        ax.text(
            0.5,
            0.5,
            "Roundtrip training curves\nnot available.\n\n"
            "Train with roundtrip_weight > 0\nto enable tracking.",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return

    epochs = list(range(len(metrics["training_curve"])))
    primary_data = {"Roundtrip Loss": metrics["training_curve"]}

    plot_dual_axis_curves(
        ax,
        epochs=epochs,
        primary_data=primary_data,
        primary_ylabel="Roundtrip Loss (MSE)",
        title="Roundtrip Loss Training Curves",
        log_scale_primary=True,
        mark_best_epoch=data.epoch if hasattr(data, "epoch") else None,
    )


def plot_token_match_evolution(ax, data: VQVAECheckpointData) -> None:
    """Plot token match percentage over training.

    Token match rate = % of samples where encode(decode(tokens)) == tokens
    """
    # Check if token match rate is available in metrics history
    has_match_rate = any(
        "token_match_rate" in metrics for metrics in data.metrics_history
    )

    if not has_match_rate:
        ax.text(
            0.5,
            0.5,
            "Token match rate tracking\nnot available in checkpoint.\n\n"
            "Add to trainer validation\nfor future runs.",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return

    # Extract match rates
    epochs = []
    match_rates = []
    for idx, metrics in enumerate(data.metrics_history):
        if "token_match_rate" in metrics:
            epochs.append(idx)
            match_rates.append(metrics["token_match_rate"] * 100)  # Convert to percentage

    primary_data = {"Token Match Rate (%)": match_rates}

    plot_dual_axis_curves(
        ax,
        epochs=epochs,
        primary_data=primary_data,
        primary_ylabel="Match Rate (%)",
        title="Token Match Rate Evolution",
        log_scale_primary=False,
    )

    # Add target line at 100%
    ax.axhline(y=100, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Perfect")
    ax.legend(loc="best", fontsize=8)


def plot_roundtrip_heatmap(ax, data: VQVAECheckpointData) -> None:
    """Heatmap of per-quantizer roundtrip loss."""
    metrics = extract_roundtrip_metrics(data)

    if not metrics["per_quantizer"]:
        ax.text(
            0.5,
            0.5,
            "Per-quantizer roundtrip metrics\nnot available.",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        ax.axis("off")
        return

    # Build matrix [categories × levels]
    num_cats = data.num_categories
    num_levels = data.num_levels
    matrix = np.zeros((num_cats, num_levels))
    annotations = [[" " for _ in range(num_levels)] for _ in range(num_cats)]

    for i, cat in enumerate(data.category_names):
        for level in range(num_levels):
            key = f"{cat}_L{level}"
            if key in metrics["per_quantizer"]:
                loss = metrics["per_quantizer"][key]
                matrix[i, level] = loss
                annotations[i][level] = f"{loss:.4f}"

    # Use components.plot_metric_heatmap() for DRY
    plot_metric_heatmap(
        ax,
        matrix=matrix,
        row_labels=[c.replace("_group_", "") for c in data.category_names],
        col_labels=[f"L{i}" for i in range(num_levels)],
        title="Per-Quantizer Roundtrip Loss",
        cmap="RdYlGn_r",  # Red (bad) → Yellow → Green (good)
        vmin=0.0,
        vmax=0.01,  # Target threshold
        annotations=annotations,
        colorbar_label="Roundtrip MSE",
    )


def plot_combinatorial_space(ax, data: VQVAECheckpointData) -> None:
    """Display combinatorial space size metrics."""
    space_metrics = compute_combinatorial_space_size(data)

    # Large number display
    total = space_metrics["total"]
    log_size = space_metrics["log_size"]

    # Format large number (e.g., "3.2M" or "4.5B")
    if total >= 1e9:
        display = f"{total / 1e9:.1f}B"
    elif total >= 1e6:
        display = f"{total / 1e6:.1f}M"
    elif total >= 1e3:
        display = f"{total / 1e3:.1f}K"
    else:
        display = str(total)

    metrics = {
        "Unique Patterns": (display, ""),
        "Log₁₀ Size": (f"{log_size:.1f}", ""),
        "L0 Product": (f"{space_metrics['per_level'][0]:,}", "codes"),
        "L1 Product": (f"{space_metrics['per_level'][1]:,}", "codes"),
        "L2 Product": (f"{space_metrics['per_level'][2]:,}", "codes"),
    }

    plot_metric_card(ax, metrics, title="Combinatorial Token Space")


def plot_token_frequency_distribution(
    ax,
    data: VQVAECheckpointData,
    tokenized_dataset_path: Optional[Path],
) -> None:
    """Histogram of token pattern frequencies."""
    if tokenized_dataset_path is None or not tokenized_dataset_path.exists():
        ax.text(
            0.5,
            0.5,
            "Token frequency analysis requires\npretokenized dataset path.\n\n"
            "Pass --tokenized-dataset to enable.",
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        ax.axis("off")
        return

    try:
        freq_data = analyze_token_frequencies(tokenized_dataset_path, data)

        # Plot histogram (log-log scale to show Zipf distribution)
        ax.hist(
            freq_data["frequency_distribution"],
            bins=50,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Token Pattern Frequency (log)", fontsize=10)
        ax.set_ylabel("Count (log)", fontsize=10)
        ax.set_title(
            f"Token Frequency Distribution\n"
            f"({freq_data['unique_patterns']:,} unique patterns)",
            fontsize=11,
            fontweight="bold",
        )

        # Annotate singletons
        ax.text(
            0.95,
            0.95,
            f"Singletons: {freq_data['singleton_patterns']:,}\n"
            f"({freq_data['singleton_patterns'] / max(freq_data['unique_patterns'], 1):.1%})",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Error analyzing token frequencies:\n{str(e)}",
            ha="center",
            va="center",
            fontsize=10,
            color="red",
        )
        ax.axis("off")


def plot_category_diversity(
    ax,
    data: VQVAECheckpointData,
    tokenized_dataset_path: Optional[Path],
) -> None:
    """Bar chart of Shannon entropy per category."""
    if tokenized_dataset_path is None or not tokenized_dataset_path.exists():
        ax.axis("off")
        return

    try:
        freq_data = analyze_token_frequencies(tokenized_dataset_path, data)
        entropy_dict = freq_data["entropy_per_category"]

        labels = list(entropy_dict.keys())
        values = [entropy_dict[k] for k in labels]

        plot_metric_bars(
            ax,
            labels=[l.replace("_group_", "") for l in labels],
            values=values,
            title="Per-Category Token Diversity (Entropy)",
            xlabel="Shannon Entropy",
            ylabel="",
            colormap="RdYlGn",
            horizontal=True,
            annotate=True,
        )
    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Error computing diversity:\n{str(e)}",
            ha="center",
            va="center",
            fontsize=10,
            color="red",
        )
        ax.axis("off")


def plot_level_composition(ax, data: VQVAECheckpointData) -> None:
    """Three pie charts showing level composition."""
    composition = extract_level_composition(data)

    # Get the current subplot's position
    pos = ax.get_position()

    # Hide the original axis
    ax.axis("off")

    # Create 3 subplots within this panel using figure coordinates
    fig = ax.get_figure()
    width = pos.width / 3.2
    height = pos.height * 0.8

    for level in range(data.num_levels):
        # Calculate position for each pie chart
        left = pos.x0 + (level * pos.width / 3) + 0.02
        bottom = pos.y0 + 0.1

        ax_sub = fig.add_axes([left, bottom, width, height])

        comp = composition[level]
        labels = [
            f"Active\n({comp['active_categories']})",
            f"Inactive\n({data.num_categories - comp['active_categories']})",
        ]
        sizes = [
            comp["active_categories"],
            data.num_categories - comp["active_categories"],
        ]
        colors = ["#4a9c3f", "#cccccc"]

        # Only plot if there's data
        if sum(sizes) > 0:
            ax_sub.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.0f%%",
                startangle=90,
                textprops={"fontsize": 8},
            )
            ax_sub.set_title(
                f"L{level}\n({comp['total_tokens']} tokens)",
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax_sub.text(
                0.5,
                0.5,
                f"L{level}\nNo data",
                ha="center",
                va="center",
                fontsize=9,
            )
            ax_sub.set_xlim(-1, 1)
            ax_sub.set_ylim(-1, 1)


def plot_token_cooccurrence(
    ax,
    data: VQVAECheckpointData,
    tokenized_dataset_path: Optional[Path],
) -> None:
    """Heatmap showing token co-occurrence patterns."""
    if tokenized_dataset_path is None or not tokenized_dataset_path.exists():
        ax.axis("off")
        return

    try:
        # Select representative category pairs
        # Try to find theta vs initial, or temporal vs theta, or first two categories
        category_pairs = []
        cat_names = data.category_names

        # Try to find meaningful pairs
        theta_cats = [c for c in cat_names if "theta" in c.lower()]
        initial_cats = [c for c in cat_names if "initial" in c.lower()]
        temporal_cats = [c for c in cat_names if "temporal" in c.lower()]

        if theta_cats and initial_cats:
            category_pairs.append((theta_cats[0], initial_cats[0]))
        elif temporal_cats and theta_cats:
            category_pairs.append((temporal_cats[0], theta_cats[0]))
        elif len(cat_names) >= 2:
            category_pairs.append((cat_names[0], cat_names[1]))

        if not category_pairs:
            ax.text(
                0.5,
                0.5,
                "Insufficient categories\nfor co-occurrence analysis.",
                ha="center",
                va="center",
                fontsize=11,
                color="gray",
            )
            ax.axis("off")
            return

        cooccur_data = compute_token_cooccurrence(tokenized_dataset_path, category_pairs)

        if len(cooccur_data) > 0:
            pair = list(cooccur_data.keys())[0]
            matrix = cooccur_data[pair]

            # Normalize matrix for better visualization
            matrix_norm = matrix / (matrix.max() + 1e-10)

            plot_metric_heatmap(
                ax,
                matrix=matrix_norm,
                row_labels=[f"Token {i}" for i in range(min(matrix.shape[0], 20))],
                col_labels=[f"T{j}" for j in range(min(matrix.shape[1], 20))],
                title=f"Co-occurrence: {pair[0].replace('_group_', '')} × {pair[1].replace('_group_', '')}",
                cmap="Blues",
                vmin=0,
                vmax=1,
                colorbar_label="Normalized Frequency",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No co-occurrence data available.",
                ha="center",
                va="center",
                fontsize=11,
                color="gray",
            )
            ax.axis("off")

    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Error computing co-occurrence:\n{str(e)}",
            ha="center",
            va="center",
            fontsize=10,
            color="red",
        )
        ax.axis("off")


def create_roundtrip_dashboard(
    checkpoint_path: Path | str,
    output_path: Optional[Path | str] = None,
    tokenized_dataset_path: Optional[Path | str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Create comprehensive roundtrip consistency dashboard.

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint directory or .pt file
        output_path: Output PNG path (if None, don't save)
        tokenized_dataset_path: Optional pretokenized dataset for token analysis
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    # Convert to Path objects
    checkpoint_path = Path(checkpoint_path)
    if output_path:
        output_path = Path(output_path)
    if tokenized_dataset_path:
        tokenized_dataset_path = Path(tokenized_dataset_path)

    # Load checkpoint data
    data = load_vqvae_checkpoint(checkpoint_path)

    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(16, 18))
    gs = GridSpec(
        4, 3, figure=fig, height_ratios=[1.2, 1.5, 1.2, 1.3], hspace=0.35, wspace=0.3
    )

    # Row 1: Training metrics
    ax_rt_curves = fig.add_subplot(gs[0, 0])
    ax_token_match = fig.add_subplot(gs[0, 1:])

    plot_roundtrip_training_curves(ax_rt_curves, data)
    plot_token_match_evolution(ax_token_match, data)

    # Row 2: Per-quantizer heatmap
    ax_heatmap = fig.add_subplot(gs[1, :])
    plot_roundtrip_heatmap(ax_heatmap, data)

    # Row 3: Semantic structure
    ax_comb_space = fig.add_subplot(gs[2, 0])
    ax_freq_dist = fig.add_subplot(gs[2, 1])
    ax_diversity = fig.add_subplot(gs[2, 2])

    plot_combinatorial_space(ax_comb_space, data)
    plot_token_frequency_distribution(ax_freq_dist, data, tokenized_dataset_path)
    plot_category_diversity(ax_diversity, data, tokenized_dataset_path)

    # Row 4: Compositional analysis
    ax_level_comp = fig.add_subplot(gs[3, 0])
    ax_cooccur = fig.add_subplot(gs[3, 1:])

    plot_level_composition(ax_level_comp, data)
    plot_token_cooccurrence(ax_cooccur, data, tokenized_dataset_path)

    # Overall title
    fig.suptitle(
        "Roundtrip Consistency & Semantic Structure Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved roundtrip dashboard to {output_path}")

    return fig
