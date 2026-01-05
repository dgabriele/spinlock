"""Topological dashboard for VQ-VAE codebook visualization.

Creates a multi-panel figure showing:
- t-SNE embedding of all codebook vectors (large)
- Codebook usage heatmap (categories × levels)
- Inter-codebook similarity matrix
- Embedding space statistics
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import torch

from .utils import VQVAECheckpointData, load_vqvae_checkpoint


def extract_codebook_embeddings(
    checkpoint_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Extract codebook embeddings and usage from checkpoint.

    Supports both standard VQ-VAE models and hybrid models (VQVAEWithInitial)
    which have a `vqvae.` prefix on their state dict keys.

    Returns:
        Tuple of:
        - embeddings: Dict[codebook_key] -> embedding matrix
        - usage: Dict[codebook_key] -> usage counts (EMA cluster sizes)
        - codebook_order: List of codebook keys in order
    """
    checkpoint = torch.load(checkpoint_path / "final_model.pt", map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]

    embeddings = {}
    usage = {}
    codebook_keys = []

    # Detect if this is a hybrid model (VQVAEWithInitial) by checking for vqvae. prefix
    is_hybrid = any(key.startswith("vqvae.") for key in state.keys())

    # Extract VQ layer embeddings
    # Standard model: vq_layers.{idx}.embedding.weight
    # Hybrid model: vqvae.vq_layers.{idx}.embedding.weight
    for key in sorted(state.keys()):
        if "vq_layers" in key and "embedding.weight" in key:
            # Parse index based on model type
            parts = key.split(".")
            if is_hybrid:
                # vqvae.vq_layers.{idx}.embedding.weight -> parts[2] is idx
                idx = int(parts[2])
            else:
                # vq_layers.{idx}.embedding.weight -> parts[1] is idx
                idx = int(parts[1])

            codebook_keys.append(f"cb_{idx}")
            embeddings[f"cb_{idx}"] = state[key].numpy()

        if "vq_layers" in key and "ema_cluster_size" in key:
            parts = key.split(".")
            if is_hybrid:
                idx = int(parts[2])
            else:
                idx = int(parts[1])
            usage[f"cb_{idx}"] = state[key].numpy()

    return embeddings, usage, codebook_keys


def compute_tsne_embedding(
    embeddings: Dict[str, np.ndarray],
    perplexity: int = 15,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute t-SNE embedding of all codebook vectors.

    Uses L2 normalization before padding to prevent artificial clustering
    of smaller-dimensional codebooks near the origin.

    Returns:
        Tuple of:
        - coords: (N, 2) array of t-SNE coordinates
        - labels: (N,) array of codebook indices
        - codebook_ids: List of codebook IDs for each point
    """
    from sklearn.manifold import TSNE

    # Concatenate all embeddings, normalizing then padding to same dimension
    all_embeddings = []
    all_labels = []
    codebook_ids = []

    max_dim = max(emb.shape[1] for emb in embeddings.values())

    for cb_key, emb in embeddings.items():
        n_codes = emb.shape[0]

        # L2 normalize each code vector BEFORE padding
        # This prevents smaller-dim codebooks from clustering at origin
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb_normalized = emb / norms

        # Pad to max dimension
        if emb_normalized.shape[1] < max_dim:
            padded = np.zeros((n_codes, max_dim))
            padded[:, : emb_normalized.shape[1]] = emb_normalized
            emb_normalized = padded

        all_embeddings.append(emb_normalized)
        all_labels.extend([int(cb_key.split("_")[1])] * n_codes)
        codebook_ids.extend([cb_key] * n_codes)

    X = np.vstack(all_embeddings)
    labels = np.array(all_labels)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(X) - 1),
        max_iter=max_iter,
        random_state=42,
        init="pca",
    )
    coords = tsne.fit_transform(X)

    return coords, labels, codebook_ids


def plot_tsne_codebooks(
    ax: Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    codebook_ids: List[str],
    num_categories: int = 11,
    num_levels: int = 3,
) -> None:
    """Plot t-SNE visualization of codebook embeddings.

    Points are colored by category, with marker style indicating level.
    """
    # Create color map for categories
    cmap = plt.get_cmap("tab20")
    level_markers = ["o", "s", "^"]  # circle, square, triangle for levels 0, 1, 2

    # Map codebook index to (category, level)
    def get_cat_level(cb_idx: int) -> Tuple[int, int]:
        category = cb_idx // num_levels
        level = cb_idx % num_levels
        return category, level

    # Plot each codebook
    for cb_idx in range(num_categories * num_levels):
        mask = labels == cb_idx
        if not np.any(mask):
            continue

        cat, level = get_cat_level(cb_idx)
        color = cmap(cat / num_categories)
        marker = level_markers[level % len(level_markers)]

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            marker=marker,
            s=35,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
            label=f"C{cat+1}L{level}" if level == 0 else None,
        )

    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title("Codebook Embedding Space (t-SNE)", fontsize=14, fontweight="bold")

    # Create compact legend inside plot
    # Category legend (2 rows)
    cat_handles = []
    for cat in range(num_categories):
        color = cmap(cat / num_categories)
        cat_handles.append(Patch(facecolor=color, label=f"C{cat+1}"))

    # Level markers
    level_handles = []
    for level, marker in enumerate(level_markers[:num_levels]):
        level_handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                color="gray",
                linestyle="",
                markersize=6,
                label=f"L{level}",
            )
        )

    # Place category legend at bottom-left inside plot
    leg1 = ax.legend(
        handles=cat_handles,
        loc="lower left",
        fontsize=7,
        ncol=4,
        framealpha=0.9,
        title="Categories",
        title_fontsize=8,
    )
    ax.add_artist(leg1)

    # Place level legend at bottom-right inside plot
    ax.legend(
        handles=level_handles,
        loc="lower right",
        fontsize=7,
        ncol=3,
        framealpha=0.9,
        title="Levels",
        title_fontsize=8,
    )

    ax.grid(True, alpha=0.3)


def plot_codebook_usage_heatmap(
    ax: Axes,
    usage: Dict[str, np.ndarray],
    num_categories: int = 11,
    num_levels: int = 3,
) -> None:
    """Plot heatmap of codebook usage (normalized).

    Shows: categories (rows) × levels (columns), with cell color showing
    average utilization and cell size showing codebook size.
    """
    # Compute utilization per codebook
    utilization_matrix = np.zeros((num_categories, num_levels))
    size_matrix = np.zeros((num_categories, num_levels))

    for cb_idx in range(num_categories * num_levels):
        cb_key = f"cb_{cb_idx}"
        if cb_key in usage:
            cat = cb_idx // num_levels
            level = cb_idx % num_levels
            counts = usage[cb_key]
            total = counts.sum()
            if total > 0:
                # Utilization = fraction of codes that are used (count > threshold)
                n_used = np.sum(counts > 0.01 * total / len(counts))
                utilization_matrix[cat, level] = n_used / len(counts)
                size_matrix[cat, level] = len(counts)

    # Plot heatmap
    im = ax.imshow(utilization_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(num_levels))
    ax.set_xticklabels([f"L{i}" for i in range(num_levels)], fontsize=10)
    ax.set_yticks(range(num_categories))
    ax.set_yticklabels([f"Cat {i+1}" for i in range(num_categories)], fontsize=9)

    ax.set_xlabel("Level", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.set_title("Codebook Utilization", fontsize=12, fontweight="bold")

    # Annotate with utilization and size
    for i in range(num_categories):
        for j in range(num_levels):
            util = utilization_matrix[i, j]
            size = int(size_matrix[i, j])
            color = "white" if util < 0.5 else "black"
            ax.text(
                j, i,
                f"{util:.0%}\n({size})",
                ha="center", va="center",
                fontsize=7, color=color,
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Utilization", fontsize=9)


def plot_codebook_similarity(
    ax: Axes,
    embeddings: Dict[str, np.ndarray],
    num_categories: int = 11,
    num_levels: int = 3,
) -> None:
    """Plot inter-codebook similarity matrix.

    Shows cosine similarity between codebook centroids.
    """
    n_codebooks = num_categories * num_levels

    # Compute centroids for each codebook
    centroids = []
    max_dim = max(emb.shape[1] for emb in embeddings.values())

    for cb_idx in range(n_codebooks):
        cb_key = f"cb_{cb_idx}"
        if cb_key in embeddings:
            emb = embeddings[cb_key]
            # Pad to max dimension
            if emb.shape[1] < max_dim:
                padded = np.zeros((emb.shape[0], max_dim))
                padded[:, : emb.shape[1]] = emb
                emb = padded
            centroid = emb.mean(axis=0)
            centroids.append(centroid)
        else:
            centroids.append(np.zeros(max_dim))

    centroids = np.array(centroids)

    # Compute cosine similarity
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = centroids / norms
    similarity = normalized @ normalized.T

    # Plot
    im = ax.imshow(similarity, cmap="coolwarm", vmin=-1, vmax=1)

    # Add grid lines between categories
    for i in range(1, num_categories):
        ax.axhline(y=i * num_levels - 0.5, color="black", linewidth=1)
        ax.axvline(x=i * num_levels - 0.5, color="black", linewidth=1)

    # Labels
    tick_labels = [f"C{i//num_levels + 1}L{i % num_levels}" for i in range(n_codebooks)]
    ax.set_xticks(range(n_codebooks))
    ax.set_xticklabels(tick_labels, fontsize=5, rotation=90)
    ax.set_yticks(range(n_codebooks))
    ax.set_yticklabels(tick_labels, fontsize=5)

    ax.set_title("Codebook Similarity (Cosine)", fontsize=12, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", fontsize=9)


def plot_embedding_statistics(
    ax: Axes,
    embeddings: Dict[str, np.ndarray],
    usage: Dict[str, np.ndarray],
    data: VQVAECheckpointData,
) -> None:
    """Create summary statistics panel."""
    ax.axis("off")

    # Compute statistics
    total_codes = sum(emb.shape[0] for emb in embeddings.values())
    total_dims = sum(emb.shape[0] * emb.shape[1] for emb in embeddings.values())

    # Usage statistics
    total_usage = sum(u.sum() for u in usage.values())
    active_codes = sum(np.sum(u > 0.01 * u.sum() / len(u)) for u in usage.values())

    # Dimension statistics
    dims = [emb.shape[1] for emb in embeddings.values()]
    sizes = [emb.shape[0] for emb in embeddings.values()]

    stats_text = f"""Codebook Statistics

Total Codebooks: {len(embeddings)}
Total Codes: {total_codes}
Active Codes: {active_codes} ({100*active_codes/total_codes:.1f}%)

Embedding Dimensions:
  Min: {min(dims)}D
  Max: {max(dims)}D
  Mean: {np.mean(dims):.1f}D

Codebook Sizes:
  Min: {min(sizes)} codes
  Max: {max(sizes)} codes
  Mean: {np.mean(sizes):.1f} codes

Model Quality: {data.final_metrics.get('quality', 0):.4f}
Utilization: {data.final_metrics.get('utilization', 0):.1%}
"""

    ax.text(
        0.1, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
    )

    ax.set_title("Codebook Statistics", fontsize=12, fontweight="bold")


def create_topological_dashboard(
    checkpoint_path: str | Path,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (18, 14),
    dpi: int = 150,
) -> Figure:
    """Create topological dashboard for VQ-VAE codebook visualization.

    Args:
        checkpoint_path: Path to checkpoint directory
        output_path: Optional path to save figure (PNG)
        figsize: Figure size in inches
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    checkpoint_path = Path(checkpoint_path)

    # Load data
    data = load_vqvae_checkpoint(checkpoint_path)
    embeddings, usage, codebook_keys = extract_codebook_embeddings(checkpoint_path)

    # Compute t-SNE
    print("Computing t-SNE embedding...")
    coords, labels, codebook_ids = compute_tsne_embedding(embeddings)

    # Create figure with grid layout
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Layout: t-SNE takes left 2/3, similarity matrix and stats on right
    gs = GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[1.2, 1],
        height_ratios=[2, 1],
        hspace=0.3,
        wspace=0.25,
    )

    # Panel A: t-SNE (large, left side, spans both rows)
    ax_tsne = fig.add_subplot(gs[:, 0])
    plot_tsne_codebooks(
        ax_tsne, coords, labels, codebook_ids,
        num_categories=data.num_categories,
        num_levels=data.num_levels,
    )

    # Panel B: Similarity matrix (top-right, larger now)
    ax_sim = fig.add_subplot(gs[0, 1])
    plot_codebook_similarity(
        ax_sim, embeddings,
        num_categories=data.num_categories,
        num_levels=data.num_levels,
    )

    # Panel C: Statistics (bottom-right)
    ax_stats = fig.add_subplot(gs[1, 1])
    plot_embedding_statistics(ax_stats, embeddings, usage, data)

    # Title
    fig.suptitle(
        "VQ-VAE Codebook Topology",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Subtitle
    checkpoint_name = checkpoint_path.name
    fig.text(0.5, 0.95, f"Checkpoint: {checkpoint_name}", ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved topological dashboard to: {output_path}")

    return fig
