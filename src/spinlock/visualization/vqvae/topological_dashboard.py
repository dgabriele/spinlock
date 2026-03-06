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
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from .utils import VQVAECheckpointData, load_vqvae_checkpoint, get_utilization_cmap


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
    from .utils import find_checkpoint_file

    checkpoint = torch.load(find_checkpoint_file(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]

    embeddings = {}
    usage = {}
    codebook_keys = []

    # Strip _orig_mod. prefix from torch.compile() models
    normalized_keys = {k.replace("_orig_mod.", ""): k for k in state.keys()}

    # Detect if this is a hybrid model (VQVAEWithInitial) by checking for vqvae. prefix
    is_hybrid = any(key.startswith("vqvae.") for key in normalized_keys.keys())

    # Detect newer quantizers structure (quantizers.{group}_{level}.codebook.weight)
    # vs older vq_layers structure (vq_layers.{idx}.embedding.weight)
    has_quantizers = any(key.startswith("quantizers.") for key in normalized_keys.keys())

    if has_quantizers:
        # Newer structure: quantizers.{group}_{level}.embedding.weight and ema_cluster_size
        for norm_key in sorted(normalized_keys.keys()):
            orig_key = normalized_keys[norm_key]
            if norm_key.startswith("quantizers.") and ".embedding.weight" in norm_key:
                # Extract group_level from quantizers.{group}_{level}.embedding.weight
                # e.g., quantizers.temporal_group_1_L0.embedding.weight
                parts = norm_key.split(".")
                group_level = parts[1]  # e.g., "temporal_group_1_L0"
                codebook_keys.append(group_level)
                embeddings[group_level] = state[orig_key].numpy()

            if norm_key.startswith("quantizers.") and ".ema_cluster_size" in norm_key:
                parts = norm_key.split(".")
                group_level = parts[1]
                usage[group_level] = state[orig_key].numpy()
    else:
        # Older structure: vq_layers.{idx}.embedding.weight
        # Standard model: vq_layers.{idx}.embedding.weight
        # Hybrid model: vqvae.vq_layers.{idx}.embedding.weight
        for norm_key in sorted(normalized_keys.keys()):
            orig_key = normalized_keys[norm_key]
            if "vq_layers" in norm_key and "embedding.weight" in norm_key:
                # Parse index based on model type
                parts = norm_key.split(".")
                if is_hybrid:
                    # vqvae.vq_layers.{idx}.embedding.weight -> parts[2] is idx
                    idx = int(parts[2])
                else:
                    # vq_layers.{idx}.embedding.weight -> parts[1] is idx
                    idx = int(parts[1])

                codebook_keys.append(f"cb_{idx}")
                embeddings[f"cb_{idx}"] = state[orig_key].numpy()

            if "vq_layers" in norm_key and "ema_cluster_size" in norm_key:
                parts = norm_key.split(".")
                if is_hybrid:
                    idx = int(parts[2])
                else:
                    idx = int(parts[1])
                usage[f"cb_{idx}"] = state[orig_key].numpy()

    return embeddings, usage, codebook_keys


def compute_tsne_quality_metrics(
    embeddings: Dict[str, np.ndarray],
    coords: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute quality metrics for t-SNE embedding and codebook structure.

    Args:
        embeddings: Dict of codebook embeddings (high-dimensional)
        coords: t-SNE coordinates (2D)
        labels: Category labels for each point

    Returns:
        Dict with quality metrics
    """
    # Concatenate all embeddings (high-D space), normalizing and padding like in t-SNE
    all_embeddings = []
    max_dim = max(emb.shape[1] for emb in embeddings.values())

    for emb in embeddings.values():
        # L2 normalize before padding
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb_normalized = emb / norms

        # Pad to max dimension
        if emb_normalized.shape[1] < max_dim:
            padded = np.zeros((emb.shape[0], max_dim))
            padded[:, :emb_normalized.shape[1]] = emb_normalized
            emb_normalized = padded

        all_embeddings.append(emb_normalized)

    X_high = np.vstack(all_embeddings)

    metrics = {}

    # 1. Cluster Quality Metrics
    try:
        # Silhouette score: measures how similar points are to their own cluster vs other clusters
        # Range: [-1, 1], higher is better
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(X_high, labels)
        else:
            metrics['silhouette_score'] = 0.0
    except:
        metrics['silhouette_score'] = 0.0

    try:
        # Davies-Bouldin index: ratio of within-cluster to between-cluster distances
        # Lower is better, <1.0 indicates well-separated clusters
        if len(np.unique(labels)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(X_high, labels)
        else:
            metrics['davies_bouldin'] = 0.0
    except:
        metrics['davies_bouldin'] = 0.0

    # 2. t-SNE Faithfulness Metrics
    try:
        # Trustworthiness: measures if nearby points in 2D are also nearby in high-D
        from sklearn.manifold import trustworthiness
        metrics['trustworthiness'] = trustworthiness(X_high, coords, n_neighbors=15)
    except:
        metrics['trustworthiness'] = 0.0

    # 3. Codebook Diversity Metrics
    try:
        # Mean pairwise distance in high-D space
        from scipy.spatial.distance import pdist
        pairwise_dists = pdist(X_high, metric='euclidean')
        metrics['mean_pairwise_dist'] = np.mean(pairwise_dists)
        metrics['min_pairwise_dist'] = np.min(pairwise_dists)
        metrics['std_pairwise_dist'] = np.std(pairwise_dists)
    except:
        metrics['mean_pairwise_dist'] = 0.0
        metrics['min_pairwise_dist'] = 0.0
        metrics['std_pairwise_dist'] = 0.0

    # 4. Dimensionality Assessment
    try:
        # PCA to see how much variance is captured by top 2 components
        pca = PCA(n_components=min(2, X_high.shape[1]))
        pca.fit(X_high)
        metrics['variance_2d'] = np.sum(pca.explained_variance_ratio_)
    except:
        metrics['variance_2d'] = 0.0

    return metrics


def filter_active_codes(
    embeddings: Dict[str, np.ndarray],
    usage: Dict[str, np.ndarray],
    threshold: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Filter codebook embeddings to only include active codes.

    Args:
        embeddings: Dict[codebook_key] -> embedding matrix [num_codes, dim]
        usage: Dict[codebook_key] -> usage counts [num_codes]
        threshold: Minimum usage count to consider a code active

    Returns:
        Filtered embeddings dict with only active codes
    """
    filtered = {}

    for cb_key in embeddings.keys():
        if cb_key not in usage:
            # No usage info, keep all codes
            filtered[cb_key] = embeddings[cb_key]
            continue

        # Get active code mask
        usage_counts = usage[cb_key]
        active_mask = usage_counts > threshold

        # Filter embeddings
        if active_mask.any():
            filtered[cb_key] = embeddings[cb_key][active_mask]
        # If no codes are active, skip this codebook entirely

    return filtered


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

    for cb_idx, (cb_key, emb) in enumerate(embeddings.items()):
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

        # Parse label based on key format
        # Old format: "cb_0", "cb_1", etc.
        # New format: "temporal_group_1_L0", "theta_group_1_L0", etc.
        if cb_key.startswith("cb_"):
            label = int(cb_key.split("_")[1])
        else:
            # For new format, use enumeration index to ensure all points are plotted
            # (Using hash % 1000 causes most points to be skipped during plotting)
            label = cb_idx

        all_labels.extend([label] * n_codes)
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

    # Plot heatmap (dark gray → green, neutral for low util)
    im = ax.imshow(utilization_matrix, cmap=get_utilization_cmap(), aspect="auto", vmin=0, vmax=1)

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


def plot_utilization_heatmap(
    checkpoint_path: str | Path,
    output_path: Optional[str | Path] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 200,
    sort_by: str = "avg",
) -> Figure:
    """Plot compact VQ codebook utilization heatmap from a checkpoint.

    Each cell shows ``n/total  pct  dimD`` on one line.  Rows are feature
    groups, columns are hierarchy levels.  Color encodes utilization fraction.

    Works with both new-format keys (``temporal_group_N_LM``) and legacy
    ``cb_N`` keys.

    Args:
        checkpoint_path: Path to checkpoint file or directory.
        output_path: Optional path to save PNG.  None = return figure only.
        figsize: Figure size ``(w, h)``.  None = auto-sized to row count.
        dpi: Resolution.
        sort_by: Row ordering — ``"avg"`` (mean utilization, descending),
            ``"l0"`` (L0 utilization, descending), ``"numeric"`` (group
            index ascending), or ``"none"`` (original key order).

    Returns:
        matplotlib Figure.
    """
    checkpoint_path = Path(checkpoint_path)

    embeddings, usage, codebook_keys = extract_codebook_embeddings(checkpoint_path)

    # Parse group/level structure from keys
    groups: Dict[str, Dict[int, str]] = {}
    for k in codebook_keys:
        if "_L" in k:
            base, lev = k.rsplit("_L", 1)
            groups.setdefault(base, {})[int(lev)] = k
        elif k.startswith("cb_"):
            idx = int(k.split("_")[1])
            # Legacy: infer group/level from flat index (3 levels per group)
            g, l = divmod(idx, 3)
            gname = f"group_{g}"
            groups.setdefault(gname, {})[l] = k

    group_names = sorted(groups.keys())
    nc = len(group_names)
    nl = max(max(lvls.keys()) for lvls in groups.values()) + 1

    # Build matrices
    util_matrix = np.zeros((nc, nl))
    size_matrix = np.zeros((nc, nl), dtype=int)
    used_matrix = np.zeros((nc, nl), dtype=int)
    dim_matrix = np.zeros((nc, nl), dtype=int)

    for gi, gname in enumerate(group_names):
        for lev, key in groups[gname].items():
            if key in usage:
                counts = usage[key]
                total = counts.sum()
                cs = len(counts)
                n_used = int(np.sum(counts > 0.01 * total / cs)) if total > 0 else 0
                util_matrix[gi, lev] = n_used / cs if cs > 0 else 0
                size_matrix[gi, lev] = cs
                used_matrix[gi, lev] = n_used
            if key in embeddings:
                dim_matrix[gi, lev] = embeddings[key].shape[1]

    # Sort rows
    if sort_by == "avg":
        order = np.argsort(-util_matrix.mean(axis=1))  # descending
    elif sort_by == "l0":
        order = np.argsort(-util_matrix[:, 0])
    elif sort_by == "numeric":
        import re
        order = np.argsort([
            int(re.search(r"(\d+)", g).group(1)) if re.search(r"\d+", g) else 0
            for g in group_names
        ])
    else:
        order = np.arange(nc)
    group_names = [group_names[i] for i in order]
    util_matrix = util_matrix[order]
    size_matrix = size_matrix[order]
    used_matrix = used_matrix[order]
    dim_matrix = dim_matrix[order]

    # Auto-size figure
    if figsize is None:
        cell_h = 0.16
        figsize = (4.5, nc * cell_h + 1.2)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Per-column normalized color matrix: each level (L0, L1, L2) gets its
    # own color range based on the active codeword counts in that column.
    # This highlights which groups are over/under-utilized *within* a level,
    # which is what matters for the combinatorial token vocabulary.
    color_matrix = np.zeros_like(used_matrix, dtype=float)
    for j in range(nl):
        col = used_matrix[:, j].astype(float)
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            color_matrix[:, j] = (col - col_min) / (col_max - col_min)
        else:
            color_matrix[:, j] = 1.0  # all equal → full color

    im = ax.imshow(color_matrix, cmap=get_utilization_cmap(), aspect="auto", vmin=0, vmax=1)

    # Column headers: level + capacity + range
    level_headers = []
    for j in range(nl):
        capacity = int(size_matrix[0, j]) if nc > 0 else 0
        col = used_matrix[:, j]
        level_headers.append(f"L{j}  (/{capacity}, {int(col.min())}–{int(col.max())} active)")
    ax.set_xticks(range(nl))
    ax.set_xticklabels(level_headers, fontsize=6)
    ax.set_yticks(range(nc))
    short_labels = [
        g.replace("temporal_group_", "G").replace("group_", "G")
        for g in group_names
    ]
    ax.set_yticklabels(short_labels, fontsize=5)

    ax.set_xlabel("Level", fontsize=7)
    ax.set_ylabel("Group", fontsize=7)
    ax.set_title("Active Codewords per Group", fontsize=8, fontweight="bold", pad=18)
    ax.text(
        0.5, 1.01,
        f"{used_matrix.sum()}/{size_matrix.sum()} active overall, "
        f"{util_matrix.mean():.0%} avg utilization  ·  color normalized per column",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=5, color="0.45",
    )

    # Annotate cells: just the active count
    for i in range(nc):
        for j in range(nl):
            n = used_matrix[i, j]
            c = color_matrix[i, j]
            text_color = "white" if c > 0.75 else "black"
            ax.text(
                j, i, f"{n}",
                ha="center", va="center", fontsize=5.5,
                color=text_color, fontweight="bold", fontfamily="monospace",
            )

    cbar = plt.colorbar(im, ax=ax, shrink=0.3, pad=0.03, aspect=25)
    cbar.set_label("within-level", fontsize=5)
    cbar.ax.tick_params(labelsize=5)

    fig.subplots_adjust(left=0.1, right=0.88, top=0.92, bottom=0.07)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_codebook_similarity(
    ax: Axes,
    embeddings: Dict[str, np.ndarray],
    num_categories: int = 11,
    num_levels: int = 3,
) -> None:
    """Plot inter-codebook similarity matrix.

    Shows cosine similarity between codebook centroids.
    """
    # Compute centroids for each codebook (iterate over actual keys, not assumed format)
    centroids = []
    max_dim = max(emb.shape[1] for emb in embeddings.values())

    for cb_key in sorted(embeddings.keys()):
        emb = embeddings[cb_key]
        # Pad to max dimension
        if emb.shape[1] < max_dim:
            padded = np.zeros((emb.shape[0], max_dim))
            padded[:, : emb.shape[1]] = emb
            emb = padded
        centroid = emb.mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    n_codebooks = len(centroids)  # Actual number of codebooks

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

    # Labels - show only every Nth label to avoid overlap
    # For 96 codebooks, show every 6th (16 labels total)
    tick_interval = max(1, n_codebooks // 16)
    tick_positions = list(range(0, n_codebooks, tick_interval))
    tick_labels = [f"C{i//num_levels + 1}L{i % num_levels}" for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=90)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_title("Codebook Similarity (Cosine)", fontsize=12, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", fontsize=9)


def plot_tsne_quality_panel(
    ax: Axes,
    metrics: Dict[str, float],
) -> None:
    """Plot t-SNE quality metrics panel."""
    ax.axis("off")

    # Format metrics with interpretation
    def format_metric(value: float, good_threshold: float, reverse: bool = False) -> str:
        """Format metric with color indicator."""
        if reverse:
            indicator = "✓" if value < good_threshold else "✗"
        else:
            indicator = "✓" if value >= good_threshold else "✗"
        return f"{indicator} {value:.3f}"

    # Interpret silhouette score
    silhouette = metrics.get('silhouette_score', 0.0)
    if silhouette > 0.7:
        sil_quality = "excellent"
    elif silhouette > 0.5:
        sil_quality = "good"
    elif silhouette > 0.3:
        sil_quality = "moderate"
    else:
        sil_quality = "poor"

    # Interpret Davies-Bouldin
    db = metrics.get('davies_bouldin', 0.0)
    if db < 1.0:
        db_quality = "well-separated"
    elif db < 1.5:
        db_quality = "moderate"
    else:
        db_quality = "overlapping"

    metrics_text = f"""t-SNE Quality Metrics

Cluster Quality:
  Silhouette:     {silhouette:.3f} ({sil_quality})
  Davies-Bouldin: {db:.3f} ({db_quality})

t-SNE Faithfulness:
  Trustworthiness: {metrics.get('trustworthiness', 0.0):.3f}
  (neighbors preserved)

Codebook Diversity:
  Mean distance: {metrics.get('mean_pairwise_dist', 0.0):.2f}
  Min distance:  {metrics.get('min_pairwise_dist', 0.0):.2f}
  Std distance:  {metrics.get('std_pairwise_dist', 0.0):.2f}

Dimensionality:
  Variance (2D): {metrics.get('variance_2d', 0.0)*100:.1f}%
"""

    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    )

    ax.set_title("t-SNE Quality", fontsize=11, fontweight="bold")


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

    # Compute aggregate utilization from per-quantizer metrics
    util_keys = [k for k in data.final_metrics.keys() if k.endswith('/utilization')]
    if util_keys:
        avg_utilization = sum(data.final_metrics[k] for k in util_keys) / len(util_keys)
    else:
        avg_utilization = data.final_metrics.get('utilization', 0)

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
Utilization: {avg_utilization:.1%}
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

    # Filter to only active codes (usage > 0.1)
    print("Filtering to active codes only...")
    embeddings_active = filter_active_codes(embeddings, usage, threshold=0.1)

    # Compute t-SNE (using only active codes)
    print("Computing t-SNE embedding...")
    coords, labels, codebook_ids = compute_tsne_embedding(embeddings_active)

    # Compute t-SNE quality metrics (using only active codes)
    print("Computing quality metrics...")
    quality_metrics = compute_tsne_quality_metrics(embeddings_active, coords, labels)

    # Create figure with grid layout
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Layout: t-SNE takes left 2/3, similarity matrix and stats on right
    # Add 3rd row for t-SNE quality metrics
    gs = GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[1.2, 1],
        height_ratios=[2, 1, 0.8],
        hspace=0.35,
        wspace=0.25,
    )

    # Panel A: t-SNE (large, left side, spans rows 0-1)
    ax_tsne = fig.add_subplot(gs[0:2, 0])
    plot_tsne_codebooks(
        ax_tsne, coords, labels, codebook_ids,
        num_categories=data.num_categories,
        num_levels=data.num_levels,
    )

    # Panel B: Similarity matrix (top-right)
    ax_sim = fig.add_subplot(gs[0, 1])
    plot_codebook_similarity(
        ax_sim, embeddings,
        num_categories=data.num_categories,
        num_levels=data.num_levels,
    )

    # Panel C: Statistics (middle-right)
    ax_stats = fig.add_subplot(gs[1, 1])
    plot_embedding_statistics(ax_stats, embeddings, usage, data)

    # Panel D: t-SNE quality metrics (bottom-left)
    ax_quality = fig.add_subplot(gs[2, 0])
    plot_tsne_quality_panel(ax_quality, quality_metrics)

    # Panel E: Empty (bottom-right, for balance)
    ax_empty = fig.add_subplot(gs[2, 1])
    ax_empty.axis('off')

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
