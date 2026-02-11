"""Roundtrip consistency and semantic structure dashboard for VQ-VAE models.

This dashboard visualizes:
1. Roundtrip training dynamics (encode(decode(tokens)) == tokens)
2. Semantic structure (combinatorial space size, token frequencies)
3. Compositional properties (level-wise composition, co-occurrence)
"""

from pathlib import Path
from typing import Optional, List
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
    extract_rollout_token_sets,
    flatten_rollout_tokens,
    compute_rollout_similarity,
    hierarchical_cluster_rollouts,
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


# ============================================================================
# Hierarchical Token Pattern Analysis Visualizations
# ============================================================================


def plot_rollout_similarity_matrix(
    similarity_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    output_path: str | Path,
    title: str = "Rollout Token Similarity (Jaccard)",
    cmap: str = "viridis",
    save_vector: bool = True,
    show_left_dendrogram: bool = False,
    max_dendrogram_size: int = 5000,
) -> None:
    """Plot similarity heatmap with hierarchical clustering dendrogram.

    Args:
        similarity_matrix: [N, N] similarity matrix
        linkage_matrix: Linkage matrix from hierarchical_cluster_rollouts()
        output_path: Path to save figure (will save both .png and .pdf if save_vector=True)
        title: Plot title
        cmap: Colormap name
        save_vector: If True, also save as PDF for zooming detail
        show_left_dendrogram: If True, show dendrogram on left axis (redundant for symmetric matrices)
        max_dendrogram_size: Skip dendrogram rendering if N > this (prevents recursion errors)

    Layout:
        ┌──────────┬─────────────────┐
        │          │  Dendrogram     │
        │  Empty   │  (top)          │
        ├──────────┼─────────────────┤
        │ Dendro-  │                 │
        │ gram     │   Heatmap       │
        │ (left)   │                 │
        └──────────┴─────────────────┘
    """
    from scipy.cluster.hierarchy import dendrogram
    from matplotlib.gridspec import GridSpec
    import matplotlib as mpl
    import sys

    # Increase recursion limit for large dendrograms
    n_rollouts = similarity_matrix.shape[0]
    original_recursion_limit = sys.getrecursionlimit()

    # Set recursion limit high enough for dendrogram rendering
    # Dendrogram needs ~2-3x the number of leaves in recursion depth
    required_limit = max(original_recursion_limit, n_rollouts * 3 + 1000)
    sys.setrecursionlimit(required_limit)

    if n_rollouts > max_dendrogram_size:
        print(f"    (Large dataset: {n_rollouts} rollouts, increased recursion limit to {required_limit})")

    # Set line properties globally for dendrograms (ensures PDF backend respects them)
    original_linewidth = mpl.rcParams['lines.linewidth']
    mpl.rcParams['lines.linewidth'] = 0.2

    # Create figure with layout depending on whether left dendrogram is shown
    if show_left_dendrogram:
        fig = plt.figure(figsize=(13, 10))
        gs = GridSpec(
            2, 3,
            figure=fig,
            width_ratios=[0.12, 1, 0.05],  # [dendrogram, heatmap, colorbar space]
            height_ratios=[0.12, 1],
            hspace=0.01,
            wspace=0.01,
        )
    else:
        # Simpler layout without left dendrogram
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(
            2, 2,
            figure=fig,
            width_ratios=[1, 0.05],  # [heatmap, colorbar space]
            height_ratios=[0.12, 1],
            hspace=0.01,
            wspace=0.01,
        )

    # Top dendrogram (horizontal orientation) - compute first to get consistent leaf order
    if show_left_dendrogram:
        ax_top = fig.add_subplot(gs[0, 1])
    else:
        ax_top = fig.add_subplot(gs[0, 0])

    # Use gray color to simulate transparency (more reliable in PDF than alpha)
    dendro_top = dendrogram(linkage_matrix, ax=ax_top, orientation="top", no_labels=True,
                            color_threshold=0, link_color_func=lambda k: '#666666')
    # Get leaf order from top dendrogram
    leaf_order = dendro_top["leaves"]

    # Make dendrogram lines very thin
    for line in ax_top.get_lines():
        line.set_linewidth(0.2)
        line.set_color('#666666')  # Gray color
        line.set_alpha(0.66)  # Also set alpha for PNG
    # Remove all axis decorations
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.axis('off')

    # Reorder similarity matrix using the same leaf order
    reordered_matrix = similarity_matrix[leaf_order, :][:, leaf_order]

    # Left dendrogram (only if requested - redundant for symmetric matrices)
    if show_left_dendrogram:
        ax_left = fig.add_subplot(gs[1, 0])
        # Use gray color to simulate transparency (more reliable in PDF than alpha)
        dendrogram(linkage_matrix, ax=ax_left, orientation="left", no_labels=True,
                   color_threshold=0, link_color_func=lambda k: '#666666')
        # Make dendrogram lines very thin
        for line in ax_left.get_lines():
            line.set_linewidth(0.2)
            line.set_color('#666666')  # Gray color
            line.set_alpha(0.66)  # Also set alpha for PNG
        # Invert y-axis to match heatmap row order (top to bottom)
        ax_left.invert_yaxis()
        # Remove all axis decorations
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        ax_left.axis('off')

    # Main heatmap
    if show_left_dendrogram:
        ax_heatmap = fig.add_subplot(gs[1, 1])
    else:
        ax_heatmap = fig.add_subplot(gs[1, 0])
    im = ax_heatmap.imshow(reordered_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Labels with adjusted positioning to avoid dendrogram overlap
    ax_heatmap.set_xlabel("Rollout Index (clustered)", fontsize=10)
    ax_heatmap.set_ylabel("Rollout Index (clustered)", fontsize=10, labelpad=10)

    # Move y-axis to avoid dendrogram
    ax_heatmap.yaxis.set_label_coords(-0.15, 0.5)

    # Colorbar in dedicated column to maintain alignment
    if show_left_dendrogram:
        ax_cbar = fig.add_subplot(gs[1, 2])
        # Hide top-left corner and top-right corner (colorbar space)
        ax_corner = fig.add_subplot(gs[0, 0])
        ax_corner.axis("off")
        ax_corner_right = fig.add_subplot(gs[0, 2])
        ax_corner_right.axis("off")
    else:
        ax_cbar = fig.add_subplot(gs[1, 1])
        # Hide top-right corner (colorbar space)
        ax_corner_right = fig.add_subplot(gs[0, 1])
        ax_corner_right.axis("off")

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label("Similarity", rotation=270, labelpad=15, fontsize=10)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved rollout similarity matrix to {output_path}")

    # Also save as PDF for vector zooming
    if save_vector:
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, format='pdf', bbox_inches="tight")
        print(f"Saved vector version to {pdf_path}")

    plt.close(fig)

    # Restore original settings
    mpl.rcParams['lines.linewidth'] = original_linewidth
    sys.setrecursionlimit(original_recursion_limit)


def plot_token_commonality(
    rollout_tokens: dict,
    output_dir: str | Path,
    category: str = "temporal_group_0_L0",
    n_top: int = 20,
) -> None:
    """Analyze token commonality: shared vs rare tokens.

    For a given category, compute:
    - Token frequency across rollouts (how many rollouts use each token?)
    - Classify tokens as "common" (>50% of rollouts) vs "rare" (<10%)

    Args:
        rollout_tokens: Nested dict from extract_rollout_token_sets()
        output_dir: Directory to save plots
        category: Category to analyze
        n_top: Number of top tokens to show

    Plots:
        1. Bar chart: Token → Frequency (count of rollouts using this token)
        2. Histogram: Distribution of token frequencies
        3. Pie chart: Common vs Rare vs Singleton tokens
    """
    from collections import Counter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract tokens for specified category across all rollouts
    token_counter = Counter()
    n_rollouts = len(rollout_tokens)

    for rollout_idx, categories in rollout_tokens.items():
        if category in categories:
            for token_id in categories[category]:
                token_counter[token_id] += 1

    if not token_counter:
        print(f"No tokens found for category {category}")
        return

    # Sort by frequency
    token_freq = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    tokens, frequencies = zip(*token_freq)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Bar chart of top tokens
    ax = axes[0]
    top_n = min(n_top, len(tokens))
    ax.barh(range(top_n), frequencies[:top_n], color="steelblue", alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"Token {t}" for t in tokens[:top_n]], fontsize=8)
    ax.set_xlabel("Frequency (# rollouts)", fontsize=10)
    ax.set_title(f"Top {top_n} Tokens by Frequency", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    # 2. Histogram of frequencies
    ax = axes[1]
    ax.hist(frequencies, bins=30, color="forestgreen", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Frequency (# rollouts)", fontsize=10)
    ax.set_ylabel("Count (# tokens)", fontsize=10)
    ax.set_title("Token Frequency Distribution", fontsize=11, fontweight="bold")

    # 3. Pie chart: Common vs Rare vs Singleton
    ax = axes[2]
    common_threshold = int(0.5 * n_rollouts)
    rare_threshold = int(0.1 * n_rollouts)

    common_count = sum(1 for freq in frequencies if freq >= common_threshold)
    rare_count = sum(1 for freq in frequencies if freq < rare_threshold and freq > 1)
    singleton_count = sum(1 for freq in frequencies if freq == 1)
    medium_count = len(frequencies) - common_count - rare_count - singleton_count

    labels = []
    sizes = []
    colors = []

    if common_count > 0:
        labels.append(f"Common (≥50%)\n({common_count})")
        sizes.append(common_count)
        colors.append("#4a9c3f")

    if medium_count > 0:
        labels.append(f"Medium\n({medium_count})")
        sizes.append(medium_count)
        colors.append("#ffa500")

    if rare_count > 0:
        labels.append(f"Rare (<10%)\n({rare_count})")
        sizes.append(rare_count)
        colors.append("#ff6b6b")

    if singleton_count > 0:
        labels.append(f"Singleton\n({singleton_count})")
        sizes.append(singleton_count)
        colors.append("#cccccc")

    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Token Commonality", fontsize=11, fontweight="bold")

    # Overall title
    category_clean = category.replace("_group_", "")
    fig.suptitle(f"Token Commonality Analysis: {category_clean}", fontsize=13, fontweight="bold")

    plt.tight_layout()

    # Save
    output_path = output_dir / f"token_commonality_{category}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved token commonality plot to {output_path}")


def plot_upset_token_intersections(
    rollout_tokens: dict,
    output_path: str | Path,
    categories: Optional[List[str]] = None,
    min_size: int = 5,
) -> None:
    """Plot UpSet-style visualization of token set intersections.

    Shows which combinations of categories share tokens across rollouts.

    Args:
        rollout_tokens: Nested dict from extract_rollout_token_sets()
        output_path: Path to save figure
        categories: List of categories to analyze (default: all)
        min_size: Minimum intersection size to display

    Notes:
        - Uses upsetplot library if available, else implements simplified version
        - Focus on showing which token combinations co-occur
    """
    from collections import Counter
    import itertools

    output_path = Path(output_path)

    # Get categories
    if categories is None:
        # Extract all unique categories from rollout_tokens
        all_cats = set()
        for rollout_data in rollout_tokens.values():
            all_cats.update(rollout_data.keys())
        categories = sorted(all_cats)[:5]  # Limit to 5 for readability

    # Try to use upsetplot library
    try:
        import upsetplot

        # Build data structure for upsetplot
        # Need to create a multi-index with boolean columns for each category
        data_dict = {cat: [] for cat in categories}

        for rollout_idx, cat_tokens in rollout_tokens.items():
            for cat in categories:
                has_tokens = cat in cat_tokens and len(cat_tokens[cat]) > 0
                data_dict[cat].append(has_tokens)

        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(data_dict)

        # Set multi-index
        df_indexed = df.set_index(categories)

        # Count intersections
        intersection_counts = df_indexed.groupby(categories).size()

        # Filter by min_size
        intersection_counts = intersection_counts[intersection_counts >= min_size]

        # Plot
        fig = plt.figure(figsize=(12, 6))
        upsetplot.plot(intersection_counts, fig=fig, show_counts=True)
        plt.suptitle("Token Set Intersections (UpSet Plot)", fontsize=14, fontweight="bold", y=0.98)

    except ImportError:
        # Fallback: Simplified matplotlib implementation
        print("upsetplot not available, using simplified visualization")

        # Count intersection sizes for all subsets
        intersection_data = []

        for r in range(1, min(len(categories) + 1, 4)):  # Limit to 3-way intersections
            for combo in itertools.combinations(categories, r):
                # Count rollouts that have tokens in all categories in combo
                count = 0
                for rollout_data in rollout_tokens.values():
                    has_all = all(
                        cat in rollout_data and len(rollout_data[cat]) > 0
                        for cat in combo
                    )
                    if has_all:
                        count += 1

                if count >= min_size:
                    intersection_data.append((combo, count))

        # Sort by count
        intersection_data.sort(key=lambda x: x[1], reverse=True)

        # Take top 20
        intersection_data = intersection_data[:20]

        # Plot as bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        labels = [" ∩ ".join(c.replace("_group_", "") for c in combo) for combo, _ in intersection_data]
        counts = [count for _, count in intersection_data]

        ax.barh(range(len(labels)), counts, color="steelblue", alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Intersection Size (# rollouts)", fontsize=10)
        ax.set_title("Token Set Intersections (Simplified)", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UpSet intersection plot to {output_path}")


def plot_cluster_summary(
    similarity_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    output_path: str | Path,
    n_clusters: int = None,
    distance_threshold: float = 0.7,
    max_dendrogram_size: int = 5000,
) -> None:
    """Create alternative cluster-level visualization.

    Shows cluster structure instead of individual rollouts:
    - Cluster sizes (how many rollouts in each cluster)
    - Inter-cluster similarities
    - Cluster dendrogram with labels (skipped for large datasets)

    Args:
        similarity_matrix: [N, N] similarity matrix
        linkage_matrix: Linkage matrix from hierarchical clustering
        output_path: Path to save figure
        n_clusters: Number of clusters to extract (if None, use distance_threshold)
        distance_threshold: Cut dendrogram at this distance to form clusters
        max_dendrogram_size: Maximum number of rollouts for dendrogram rendering
    """
    from scipy.cluster.hierarchy import fcluster, dendrogram

    import sys

    output_path = Path(output_path)
    n_rollouts = similarity_matrix.shape[0]

    # Increase recursion limit for large dendrograms
    original_recursion_limit = sys.getrecursionlimit()
    required_limit = max(original_recursion_limit, n_rollouts * 3 + 1000)
    sys.setrecursionlimit(required_limit)

    # Extract clusters from dendrogram
    if n_clusters is not None:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    else:
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    unique_clusters = np.unique(cluster_labels)
    n_found_clusters = len(unique_clusters)

    # Compute cluster properties
    cluster_sizes = []
    cluster_avg_similarity = []

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_sizes.append(mask.sum())

        # Average intra-cluster similarity
        cluster_sim_matrix = similarity_matrix[mask][:, mask]
        # Exclude diagonal
        if cluster_sim_matrix.shape[0] > 1:
            intra_sim = (cluster_sim_matrix.sum() - cluster_sim_matrix.trace()) / (cluster_sim_matrix.shape[0] * (cluster_sim_matrix.shape[0] - 1))
        else:
            intra_sim = 1.0
        cluster_avg_similarity.append(intra_sim)

    # Compute inter-cluster similarity matrix
    inter_cluster_sim = np.zeros((n_found_clusters, n_found_clusters))
    for i, cluster_i in enumerate(unique_clusters):
        for j, cluster_j in enumerate(unique_clusters):
            mask_i = cluster_labels == cluster_i
            mask_j = cluster_labels == cluster_j

            if i == j:
                inter_cluster_sim[i, j] = cluster_avg_similarity[i]
            else:
                # Average similarity between clusters
                cross_sim = similarity_matrix[mask_i][:, mask_j]
                inter_cluster_sim[i, j] = cross_sim.mean()

    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # Panel 1: Cluster sizes
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_names = [f"C{i}\n(n={size})" for i, size in enumerate(cluster_sizes)]
    colors = plt.cm.tab20(np.linspace(0, 1, n_found_clusters))

    bars = ax1.barh(range(n_found_clusters), cluster_sizes, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(n_found_clusters))
    ax1.set_yticklabels(cluster_names, fontsize=9)
    ax1.set_xlabel("Number of Rollouts", fontsize=10)
    ax1.set_title(f"Cluster Sizes ({n_found_clusters} clusters, {n_rollouts} rollouts)", fontsize=11, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Panel 2: Inter-cluster similarity heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(inter_cluster_sim, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(n_found_clusters))
    ax2.set_yticks(range(n_found_clusters))
    ax2.set_xticklabels([f"C{i}" for i in range(n_found_clusters)], fontsize=9)
    ax2.set_yticklabels([f"C{i}" for i in range(n_found_clusters)], fontsize=9)
    ax2.set_xlabel("Cluster", fontsize=10)
    ax2.set_ylabel("Cluster", fontsize=10)
    ax2.set_title("Inter-Cluster Similarity", fontsize=11, fontweight='bold')

    # Add text annotations
    for i in range(n_found_clusters):
        for j in range(n_found_clusters):
            text_color = 'white' if inter_cluster_sim[i, j] < 0.5 else 'black'
            ax2.text(j, i, f'{inter_cluster_sim[i, j]:.2f}',
                    ha='center', va='center', color=text_color, fontsize=8)

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity", rotation=270, labelpad=15, fontsize=10)

    # Panel 3: Cluster dendrogram with labels
    ax3 = fig.add_subplot(gs[0, 2])
    dendro = dendrogram(linkage_matrix, ax=ax3, orientation='top', no_labels=True,
                       color_threshold=distance_threshold if n_clusters is None else None,
                       above_threshold_color='#666666')

    # Set line widths and colors for dendrogram (PNG version)
    for line in ax3.get_lines():
        line.set_linewidth(0.5)
        line.set_color('#666666')

    if distance_threshold and n_clusters is None:
        ax3.axhline(y=distance_threshold, color='red', linestyle='--', linewidth=1.5,
                   label=f'Cut threshold={distance_threshold}')
        ax3.legend(loc='upper right', fontsize=8)

    ax3.set_xlabel("Rollout Index", fontsize=10)
    ax3.set_ylabel("Distance", fontsize=10)
    ax3.set_title("Hierarchical Clustering Dendrogram", fontsize=11, fontweight='bold')

    # Overall title
    fig.suptitle(f"Cluster Summary Analysis", fontsize=14, fontweight='bold', y=0.98)

    # Save PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    # Save PDF with larger size and thinner lines for better zoom quality
    pdf_path = output_path.with_suffix('.pdf')

    # Temporarily set thinner lines for PDF
    for line in ax3.get_lines():
        line.set_linewidth(0.2)

    # Save with larger figure size for PDF
    original_size = fig.get_size_inches()
    fig.set_size_inches(original_size[0] * 1.5, original_size[1] * 1.5)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

    # Restore original size and line widths
    fig.set_size_inches(original_size)
    for line in ax3.get_lines():
        line.set_linewidth(0.5)

    plt.close(fig)
    print(f"Saved cluster summary to {output_path}")
    print(f"Saved vector version to {pdf_path} (1.5x larger with thinner lines for zoom)")
    print(f"  → Found {n_found_clusters} clusters")
    print(f"  → Cluster sizes: {cluster_sizes}")
    print(f"  → Avg intra-cluster similarity: {np.mean(cluster_avg_similarity):.3f}")

    # Restore recursion limit
    sys.setrecursionlimit(original_recursion_limit)


def plot_token_cooccurrence_network(
    rollout_tokens_flat: dict,
    output_path: str | Path,
    min_cooccurrence: int = 10,
    layout: str = "spring",
    max_nodes: int = 50,
    save_vector: bool = True,
) -> None:
    """Plot network graph of token co-occurrences.

    Nodes: Tokens
    Edges: Co-occurrence frequency across rollouts

    Args:
        rollout_tokens_flat: Flattened tokens from flatten_rollout_tokens()
        output_path: Path to save figure
        min_cooccurrence: Minimum co-occurrence count to draw edge
        layout: Graph layout algorithm ("spring", "kamada_kawai", "circular")
        max_nodes: Maximum number of nodes to display (limit for performance)

    Notes:
        - Use networkx for graph construction
        - Node size ∝ token frequency
        - Edge width ∝ co-occurrence frequency
        - Color nodes by category family
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx not available, skipping co-occurrence network plot")
        return

    from collections import Counter
    import itertools

    output_path = Path(output_path)

    # Count token frequencies
    token_counter = Counter()
    for token_set in rollout_tokens_flat.values():
        token_counter.update(token_set)

    # Take top max_nodes tokens by frequency
    top_tokens = [token for token, _ in token_counter.most_common(max_nodes)]

    # Build co-occurrence matrix
    cooccurrence = Counter()

    for token_set in rollout_tokens_flat.values():
        # Only consider top tokens
        tokens_in_rollout = [t for t in token_set if t in top_tokens]

        # Count all pairs
        for t1, t2 in itertools.combinations(sorted(tokens_in_rollout), 2):
            cooccurrence[(t1, t2)] += 1

    # Create graph
    G = nx.Graph()

    # Add nodes with frequency as attribute
    for token in top_tokens:
        G.add_node(token, frequency=token_counter[token])

    # Add edges with co-occurrence as weight
    for (t1, t2), count in cooccurrence.items():
        if count >= min_cooccurrence:
            G.add_edge(t1, t2, weight=count)

    # Compute layout with better spacing
    if layout == "spring":
        pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)  # Increased k and iterations for better spacing
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=3.0, seed=42)

    # Draw graph with larger figure
    fig, ax = plt.subplots(figsize=(20, 16))

    # Node sizes proportional to frequency
    node_sizes = [G.nodes[node]["frequency"] * 5 for node in G.nodes()]

    # Random colors for nodes
    import random
    random.seed(42)  # For reproducibility
    node_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                   for _ in range(len(G.nodes()))]

    # Edge widths proportional to co-occurrence
    edges = G.edges()
    edge_weights = [G.edges[edge]["weight"] for edge in edges]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_weight * 3 for w in edge_weights]

    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, ax=ax)

    # Labels (only for high-frequency nodes)
    high_freq_threshold = sorted(token_counter.values(), reverse=True)[min(10, len(token_counter) - 1)]
    labels = {node: node.split("_token")[0].replace("_group_", "") for node in G.nodes() if G.nodes[node]["frequency"] >= high_freq_threshold}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    ax.set_title(
        f"Token Co-occurrence Network\n({len(G.nodes())} tokens, {len(G.edges())} edges, min co-occurrence={min_cooccurrence})",
        fontsize=13,
        fontweight="bold",
    )
    ax.axis("off")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved co-occurrence network to {output_path}")

    # Also save as PDF for vector zooming
    if save_vector:
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, format='pdf', bbox_inches="tight")
        print(f"Saved vector version to {pdf_path}")

    plt.close(fig)


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


def generate_hierarchical_pattern_analysis(
    checkpoint_path: Path | str,
    tokenized_dataset_path: Path | str,
    output_dir: Path | str,
    family: str = "temporal",
    max_rollouts: Optional[int] = None,
    similarity_metric: str = "jaccard",
) -> None:
    """Generate hierarchical token pattern analysis visualizations.

    Creates:
    1. Rollout similarity matrix with dendrogram
    2. Token commonality analysis (for key categories)
    3. UpSet intersection plot
    4. Token co-occurrence network

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint directory
        tokenized_dataset_path: Path to pretokenized HDF5 dataset
        output_dir: Directory to save visualization outputs
        family: Feature family to analyze ("temporal", "initial", "theta", "all")
                Use "all" to analyze all families together.
        max_rollouts: Optional limit for large datasets (sampling for performance)
        similarity_metric: "jaccard" or "cosine" for similarity computation
    """
    from pathlib import Path

    # Convert to Path objects
    checkpoint_path = Path(checkpoint_path)
    tokenized_dataset_path = Path(tokenized_dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Hierarchical Token Pattern Analysis")
    print("=" * 80)

    # Load checkpoint data for category structure
    print("\n[1/7] Loading checkpoint data...")
    data = load_vqvae_checkpoint(checkpoint_path)

    # Extract rollout token sets
    print(f"[2/7] Extracting rollout token sets for family '{family}'...")
    rollout_tokens = extract_rollout_token_sets(tokenized_dataset_path, family=family)
    n_rollouts = len(rollout_tokens)
    print(f"  → Loaded {n_rollouts} rollouts")

    # Apply sampling if needed
    if max_rollouts and n_rollouts > max_rollouts:
        print(f"  → Sampling {max_rollouts} rollouts (dataset has {n_rollouts})")
        import random
        sampled_indices = random.sample(sorted(rollout_tokens.keys()), max_rollouts)
        rollout_tokens = {idx: rollout_tokens[idx] for idx in sampled_indices}
        n_rollouts = len(rollout_tokens)

    # Flatten tokens
    print("[3/7] Flattening token sets...")
    rollout_tokens_flat = flatten_rollout_tokens(rollout_tokens)

    # Compute similarity matrix
    print(f"[4/7] Computing {similarity_metric} similarity matrix...")
    similarity_matrix = compute_rollout_similarity(rollout_tokens_flat, metric=similarity_metric)
    avg_similarity = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean()
    print(f"  → Average pairwise similarity: {avg_similarity:.3f}")

    # Hierarchical clustering
    print("[5/7] Performing hierarchical clustering...")
    linkage_matrix = hierarchical_cluster_rollouts(similarity_matrix, method="average")

    # Generate visualizations
    print("[6/7] Generating visualizations...")

    # 1. Similarity matrix with dendrogram
    print("  → Rollout similarity matrix...")
    family_label = "All Families" if family == "all" else family.capitalize()
    plot_rollout_similarity_matrix(
        similarity_matrix,
        linkage_matrix,
        output_dir / "rollout_similarity_matrix.png",
        title=f"Rollout Token Similarity ({similarity_metric.capitalize()}, {family_label})",
    )

    # 1b. Alternative cluster-level visualization
    print("  → Cluster summary (alternative view)...")
    plot_cluster_summary(
        similarity_matrix,
        linkage_matrix,
        output_dir / "cluster_summary.png",
        n_clusters=None,  # Auto-detect based on distance threshold
        distance_threshold=0.7,  # Cut dendrogram at 70% similarity
    )

    # 2. Token commonality for key categories
    print("  → Token commonality analysis...")
    # Get categories directly from the dataset (not checkpoint)
    # Extract unique category names from rollout_tokens
    all_categories = set()
    for rollout_data in rollout_tokens.values():
        all_categories.update(rollout_data.keys())

    # Filter for specific family and sort
    family_categories = sorted([cat for cat in all_categories if cat.startswith(f"{family}_group_")])

    # Analyze first few unique categories (across all levels)
    unique_category_bases = set()
    for cat in family_categories:
        # Remove _L0/_L1/_L2 suffix to get base category name
        base = cat.rsplit('_L', 1)[0] if '_L' in cat else cat
        unique_category_bases.add(base)

    # Take first 3 base categories and analyze all their levels
    categories_to_analyze = sorted(unique_category_bases)[:3]

    for category_base in categories_to_analyze:
        for level in range(data.num_levels):
            cat_key = f"{category_base}_L{level}"
            if cat_key in all_categories:
                plot_token_commonality(
                    rollout_tokens,
                    output_dir,
                    category=cat_key,
                    n_top=20,
                )

    # 3. UpSet intersection plot
    print("  → Token set intersections (UpSet plot)...")
    # Select representative L0 categories for UpSet plot
    upset_categories = [f"{cat}_L0" for cat in categories_to_analyze if f"{cat}_L0" in all_categories]

    if upset_categories:
        plot_upset_token_intersections(
            rollout_tokens,
            output_dir / "token_intersections_upset.png",
            categories=upset_categories,
            min_size=5,
        )
    else:
        print("    (Skipped - no matching L0 categories found)")

    # 4. Co-occurrence network
    print("  → Token co-occurrence network...")
    min_cooccurrence = max(5, n_rollouts // 100)  # Adaptive threshold
    plot_token_cooccurrence_network(
        rollout_tokens_flat,
        output_dir / "token_cooccurrence_network.png",
        min_cooccurrence=min_cooccurrence,
        layout="spring",
        max_nodes=100,
    )

    print(f"\n[7/7] Complete! Visualizations saved to {output_dir}")
    print("=" * 80 + "\n")
