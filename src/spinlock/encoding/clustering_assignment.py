"""Clustering-based category assignment for dynamic grouping.

Implements hierarchical clustering on feature correlation matrices to discover
optimal feature groupings. Uses Ward linkage with correlation distance (1 - |corr|).

Auto-determines optimal number of clusters via silhouette score maximization.

Automatically uses GPU acceleration (CUDA) when available for faster correlation
matrix computation.
"""

from typing import List, Dict, Optional, Any
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

# Check for CUDA availability
try:
    from .clustering_cuda import compute_correlation_matrix_cuda, CUPY_AVAILABLE
    USE_CUDA = CUPY_AVAILABLE
    if USE_CUDA:
        logger.debug("CUDA acceleration enabled for clustering")
except ImportError:
    USE_CUDA = False
    logger.debug("CUDA acceleration not available for clustering")


def compute_correlation_matrix_cpu(
    features: np.ndarray, subsample_size: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute correlation and distance matrices using CPU.

    Uses MAD (Median Absolute Deviation) normalization before computing
    Pearson correlation for robustness to outliers.

    Args:
        features: [N_samples, N_features] data
        subsample_size: Optional subsampling for large datasets

    Returns:
        corr_matrix: [N_features, N_features] correlation matrix
        condensed_dist: Condensed distance matrix for linkage
    """
    N_samples, N_features = features.shape

    # Subsample if requested
    if subsample_size is not None and subsample_size < N_samples:
        indices = np.random.choice(N_samples, subsample_size, replace=False)
        features = features[indices]
        N_samples = subsample_size

    # MAD-normalize each feature for robustness to outliers
    normalized_features = np.zeros_like(features, dtype=np.float64)
    for j in range(N_features):
        col = features[:, j]
        median = np.median(col)
        mad = np.median(np.abs(col - median)) * 1.4826
        mad = max(mad, 1e-8)  # Avoid division by zero
        normalized_features[:, j] = (col - median) / mad

    # Compute Pearson correlation matrix on MAD-normalized features
    corr_matrix = np.corrcoef(normalized_features.T)  # [N_features, N_features]

    # Convert to distance: d = 1 - |correlation|
    distance_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0.0)

    # Convert to condensed format for scipy linkage
    condensed_dist = squareform(distance_matrix, checks=False)

    return corr_matrix, condensed_dist


def compute_distance_matrix(
    features: np.ndarray,
    metric: str = 'correlation',
    corr_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute pairwise distance matrix with configurable metric.

    Args:
        features: [N, D] feature array
        metric: 'correlation', 'euclidean', or 'cosine'
        corr_matrix: Pre-computed correlation matrix (for correlation metric)

    Returns:
        distance_matrix: [D, D] pairwise distance matrix
    """
    if metric == 'correlation':
        if corr_matrix is None:
            corr_matrix = np.corrcoef(features.T)
        distance_matrix = 1.0 - np.abs(corr_matrix)
        np.fill_diagonal(distance_matrix, 0.0)
        return distance_matrix

    elif metric == 'euclidean':
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(features.T, metric='euclidean'))

    elif metric == 'cosine':
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(features.T, metric='cosine'))

    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def auto_determine_num_clusters(
    features: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 12,
    method: str = "silhouette",
    random_seed: int = 42,
    subsample_size: Optional[int] = None,
) -> int:
    """Automatically determine optimal number of clusters.

    Methods:
    - 'silhouette': Maximize silhouette score (recommended)
    - 'elbow': Elbow method on within-cluster variance

    Args:
        features: [N_samples, N_features] data
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        method: Auto-determination method
        random_seed: Random seed for reproducibility
        subsample_size: Optional subsampling for large datasets

    Returns:
        Optimal number of clusters
    """
    np.random.seed(random_seed)

    N_samples, N_features = features.shape

    # Diagnostic output
    logger.info(f"\n=== Clustering Diagnostics ===")
    logger.info(f"Feature matrix shape: {features.shape}")
    if subsample_size is not None and subsample_size < N_samples:
        logger.info(f"Subsampling: {subsample_size:,} / {N_samples:,} samples")
    logger.info(f"Feature scale check:")
    logger.info(f"  Min (avg): {features.min(axis=0).mean():.4f}")
    logger.info(f"  Max (avg): {features.max(axis=0).mean():.4f}")
    logger.info(f"  Mean (avg): {features.mean(axis=0).mean():.4f}")
    logger.info(f"  Std (avg): {features.std(axis=0).mean():.4f}")

    # Compute correlation and distance matrix (use CUDA if available)
    if USE_CUDA:
        logger.info("Computing correlation matrix (CUDA)")
        corr_matrix = compute_correlation_matrix_cuda(features, subsample_size=subsample_size)
        distance_matrix = 1.0 - np.abs(corr_matrix)
        np.fill_diagonal(distance_matrix, 0.0)
        condensed_dist = squareform(distance_matrix, checks=False)
    else:
        logger.info("Computing correlation matrix (CPU)")
        corr_matrix, condensed_dist = compute_correlation_matrix_cpu(
            features, subsample_size=subsample_size
        )
        distance_matrix = 1.0 - np.abs(corr_matrix)
        np.fill_diagonal(distance_matrix, 0.0)

    off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    logger.info(f"\nCorrelation distribution:")
    logger.info(f"  Mean: {off_diag.mean():.3f}")
    logger.info(f"  Median: {np.median(off_diag):.3f}")
    logger.info(f"  Std: {off_diag.std():.3f}")
    logger.info(f"  Min: {off_diag.min():.3f}")
    logger.info(f"  Max: {off_diag.max():.3f}")
    logger.info("===\n")

    if method == "silhouette":
        scores = []
        for k in range(min_clusters, max_clusters + 1):
            # Use correlation distance with Ward linkage
            linkage_matrix = sch.linkage(condensed_dist, method="ward")
            labels = sch.fcluster(linkage_matrix, k, criterion="maxclust")

            # Skip if any cluster has < 2 samples (silhouette undefined)
            cluster_sizes = np.bincount(labels)[1:]  # Exclude label 0
            if np.any(cluster_sizes < 2):
                scores.append(-1.0)
                logger.info(
                    f"  K={k}: SKIPPED (cluster sizes={cluster_sizes}, has singleton)"
                )
                continue

            # Compute silhouette score using correlation distance
            score = silhouette_score(
                distance_matrix, labels, metric="precomputed", random_state=random_seed
            )
            scores.append(score)
            logger.info(
                f"  K={k}: silhouette={score:.3f}, cluster sizes={np.bincount(labels)[1:]}"
            )

        # Return k with highest silhouette score
        best_k = min_clusters + np.argmax(scores)
        logger.info(
            f"Auto-determined K={best_k} clusters (silhouette={max(scores):.3f})"
        )
        return best_k

    elif method == "elbow":
        # Within-cluster sum of squares
        wcss = []
        for k in range(min_clusters, max_clusters + 1):
            linkage_matrix = sch.linkage(condensed_dist, method="ward")
            labels = sch.fcluster(linkage_matrix, k, criterion="maxclust")

            # Compute WCSS using correlation distance
            ss = 0.0
            for cluster_id in range(1, k + 1):
                cluster_mask = labels == cluster_id
                if cluster_mask.sum() == 0:
                    continue

                cluster_indices = np.where(cluster_mask)[0]
                # Average distance within cluster
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        ss += distance_matrix[cluster_indices[i], cluster_indices[j]]

            wcss.append(ss)

        # Find elbow (largest drop in WCSS)
        drops = np.abs(np.diff(wcss))
        elbow_k = min_clusters + np.argmax(drops)
        logger.info(
            f"Auto-determined K={elbow_k} clusters (elbow method, max drop={max(drops):.2f})"
        )
        return elbow_k

    else:
        raise ValueError(f"Unknown auto-determination method: {method}")


def _compute_within_cluster_dispersion(features: np.ndarray, labels: np.ndarray) -> float:
    """Sum of pairwise distances within clusters.

    Args:
        features: [N_samples, N_features] data
        labels: Cluster labels for each feature

    Returns:
        Total within-cluster dispersion
    """
    from scipy.spatial.distance import pdist

    total_dispersion = 0.0
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_features = features[:, cluster_mask]
        if cluster_features.shape[1] > 1:
            total_dispersion += np.sum(pdist(cluster_features.T, metric='euclidean'))

    return total_dispersion


def _compute_wcss(features: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster sum of squares.

    Args:
        features: [N_samples, N_features] data
        labels: Cluster labels for each feature

    Returns:
        Total within-cluster sum of squares
    """
    wcss = 0.0
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_features = features[:, cluster_mask]
        if cluster_features.shape[1] > 1:
            centroid = cluster_features.mean(axis=1, keepdims=True)
            wcss += np.sum((cluster_features - centroid) ** 2)
    return wcss


def _gap_statistic_k_selection(
    features: np.ndarray,
    min_clusters: int,
    max_clusters: int,
    n_refs: int = 10,
    linkage_method: str = 'ward',
    distance_metric: str = 'correlation',
    subsample_size: Optional[int] = None,
) -> int:
    """Gap statistic: Compare log(W_k) to reference distribution.

    Gap(K) = E[log(W_k)] - log(W_k)
    Choose K where Gap(K) - Gap(K+1) + s_{K+1} >= 0

    Args:
        features: [N_samples, N_features] data
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters
        n_refs: Number of reference distributions
        linkage_method: Linkage method for hierarchical clustering
        distance_metric: Distance metric
        subsample_size: Optional subsampling for large datasets

    Returns:
        Optimal number of clusters
    """
    # Subsample if requested
    if subsample_size is not None and subsample_size < features.shape[0]:
        indices = np.random.choice(features.shape[0], subsample_size, replace=False)
        features = features[indices]

    # Compute distance matrix for real data
    distance_matrix = compute_distance_matrix(features, distance_metric)
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = sch.linkage(condensed_dist, method=linkage_method)

    gaps = []
    s_k = []  # Standard errors

    for k in range(min_clusters, max_clusters + 1):
        # Real clustering
        labels = sch.fcluster(Z, k, criterion='maxclust')
        W_k = _compute_within_cluster_dispersion(features, labels)

        # Reference distributions (uniform over feature range)
        W_k_refs = []
        for _ in range(n_refs):
            ref_data = np.random.uniform(
                low=features.min(axis=0),
                high=features.max(axis=0),
                size=features.shape
            )
            ref_dist = compute_distance_matrix(ref_data, distance_metric)
            ref_condensed = squareform(ref_dist, checks=False)
            ref_Z = sch.linkage(ref_condensed, method=linkage_method)
            ref_labels = sch.fcluster(ref_Z, k, criterion='maxclust')
            W_k_refs.append(_compute_within_cluster_dispersion(ref_data, ref_labels))

        # Gap statistic
        gap = np.mean(np.log(W_k_refs)) - np.log(W_k)
        gaps.append(gap)
        s_k.append(np.std(np.log(W_k_refs)) * np.sqrt(1 + 1/n_refs))

        logger.info(f"  K={k}: gap={gap:.3f}, s_k={s_k[-1]:.3f}")

    # Find first K where Gap(K) >= Gap(K+1) - s_{K+1}
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i+1] - s_k[i+1]:
            best_k = min_clusters + i
            logger.info(f"Auto-determined K={best_k} clusters (gap statistic={gaps[i]:.3f})")
            return best_k

    # If no elbow found, return max
    logger.info(f"Auto-determined K={max_clusters} clusters (gap statistic, no clear elbow)")
    return max_clusters


def _elbow_k_selection(
    features: np.ndarray,
    min_clusters: int,
    max_clusters: int,
    linkage_method: str = 'ward',
    distance_metric: str = 'correlation',
    subsample_size: Optional[int] = None,
) -> int:
    """Elbow method: Find 'elbow' in WCSS curve using second derivative.

    Args:
        features: [N_samples, N_features] data
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters
        linkage_method: Linkage method for hierarchical clustering
        distance_metric: Distance metric
        subsample_size: Optional subsampling for large datasets

    Returns:
        Optimal number of clusters
    """
    # Subsample if requested
    if subsample_size is not None and subsample_size < features.shape[0]:
        indices = np.random.choice(features.shape[0], subsample_size, replace=False)
        features = features[indices]

    distance_matrix = compute_distance_matrix(features, distance_metric)
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = sch.linkage(condensed_dist, method=linkage_method)

    wcss_values = []
    k_values = list(range(min_clusters, max_clusters + 1))

    for k in k_values:
        labels = sch.fcluster(Z, k, criterion='maxclust')
        wcss = _compute_wcss(features, labels)
        wcss_values.append(wcss)
        logger.info(f"  K={k}: WCSS={wcss:.2f}")

    if len(k_values) < 3:
        return min_clusters

    # Compute second differences (discrete second derivative)
    first_diff = np.diff(wcss_values)
    second_diff = np.diff(first_diff)

    # Elbow is where second derivative is maximum
    elbow_idx = np.argmax(np.abs(second_diff))
    best_k = k_values[elbow_idx + 1]
    logger.info(f"Auto-determined K={best_k} clusters (elbow method)")
    return best_k


def _export_dendrogram(
    Z: np.ndarray,
    feature_names: List[str],
    output_path: str,
    family_name: str
):
    """Export dendrogram as PNG and linkage matrix as NPZ.

    Args:
        Z: Linkage matrix from scipy.cluster.hierarchy.linkage
        feature_names: List of feature names (for labels)
        output_path: Directory path for outputs
        family_name: Name for this dendrogram (used in filename)
    """
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dendrogram plot
    plt.figure(figsize=(20, 10))
    sch.dendrogram(
        Z,
        labels=feature_names,
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title(f'Feature Dendrogram: {family_name}')
    plt.xlabel('Feature')
    plt.ylabel('Distance')
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f'{family_name}_dendrogram.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Save linkage matrix for later analysis
    linkage_path = output_dir / f'{family_name}_linkage.npz'
    np.savez(
        linkage_path,
        linkage_matrix=Z,
        feature_names=np.array(feature_names)
    )

    logger.info(f"  Dendrogram saved: {plot_path}")


def hierarchical_clustering_assignment(
    features: np.ndarray,
    feature_names: List[str],
    num_clusters: Optional[int] = None,
    min_features_per_cluster: int = 3,
    orthogonality_target: float = 0.3,
    random_seed: int = 42,
    max_samples_for_clustering: int = 50000,
    min_clusters: int = 2,
    max_clusters: int = 50,
    isolated_families: Optional[List[str]] = None,
    reassign_orphans: bool = False,
    # NEW: Clustering configuration
    linkage_method: str = "ward",
    distance_metric: str = "correlation",
    k_selection_method: str = "silhouette",
    manual_k: Optional[int] = None,
    gap_statistic_refs: int = 10,
    distance_threshold: Optional[float] = None,
    export_dendrogram: bool = False,
    dendrogram_path: str = "diagnostics/dendrograms",
    # NEW: Mega-category splitting
    split_mega_categories: bool = False,
    max_category_size: int = 40,
    max_split_recursion_depth: int = 3,
    max_clusters_for_split: int = 8,
) -> Dict[str, List[int]]:
    """Assign features to clusters using hierarchical clustering.

    Uses configurable distance metrics and linkage methods to cluster features
    based on their statistical similarity.

    For large datasets (>50K samples), subsamples for clustering to avoid
    prohibitive computational cost. Category discovery is based on feature
    correlations, which are stable with representative subsampling.

    Args:
        features: [N_samples, N_features] data
        feature_names: List of feature names (length N_features)
        num_clusters: Number of clusters (None = auto-determine)
        min_features_per_cluster: Minimum features per cluster (prevents singletons)
        orthogonality_target: Target max correlation (used for validation warning)
        random_seed: Random seed for reproducibility
        max_samples_for_clustering: Maximum samples to use for clustering
        min_clusters: Minimum clusters to explore for auto-determination (default: 2)
        max_clusters: Maximum clusters to explore for auto-determination
        isolated_families: List of feature family names (e.g., ["architecture"]) that
            should be placed in their own dedicated categories, separate from clustering.
            Feature names must have format "family::name" for family detection.
        reassign_orphans: If True, features in too-small clusters are reassigned to
            nearest valid cluster by correlation distance. If False (default), small
            clusters are skipped. Set to True to guarantee 100% feature assignment.
        linkage_method: Linkage method: 'ward', 'average', 'complete', 'single'
        distance_metric: Distance metric: 'correlation', 'euclidean', 'cosine'
        k_selection_method: K selection: 'silhouette', 'gap_statistic', 'elbow', 'manual'
        manual_k: If k_selection_method='manual', use this K value
        gap_statistic_refs: Number of reference distributions for gap statistic
        distance_threshold: If set, cut dendrogram at this distance (overrides K selection)
        export_dendrogram: If True, export dendrogram visualizations
        dendrogram_path: Directory path for dendrogram exports
        split_mega_categories: If True, recursively split categories > max_category_size
        max_category_size: Maximum features per category (default: 40)
        max_split_recursion_depth: Maximum recursion depth for splitting (default: 3)
        max_clusters_for_split: Maximum clusters to try when splitting (default: 8)

    Returns:
        Dict mapping category_name -> list of feature indices
        Example: {'cluster_1': [0, 2, 5], 'cluster_2': [1, 3, 4, 6]}
    """
    N_samples, N_features = features.shape
    np.random.seed(random_seed)

    # Handle isolated families - separate them before clustering
    isolated_assignments = {}
    clustering_indices = list(range(N_features))  # Indices to cluster

    if isolated_families:
        logger.info(f"Isolating feature families: {isolated_families}")

        for family in isolated_families:
            family_indices = []
            for idx, name in enumerate(feature_names):
                # Match "family::*" pattern
                if "::" in name:
                    feat_family = name.split("::")[0]
                    if feat_family.lower() == family.lower():
                        family_indices.append(idx)

            if family_indices:
                # Create dedicated category for this family
                category_name = f"{family}_isolated"
                isolated_assignments[category_name] = family_indices
                logger.info(
                    f"✓ {category_name}: {len(family_indices)} features isolated"
                )

                # Remove from clustering pool
                for idx in family_indices:
                    if idx in clustering_indices:
                        clustering_indices.remove(idx)
            else:
                logger.warning(
                    f"No features found for isolated family '{family}'. "
                    f"Feature names should have format 'family::name'."
                )

        logger.info(
            f"Remaining features for clustering: {len(clustering_indices)}"
        )

    # If all features are isolated, return early
    if not clustering_indices:
        logger.info("All features isolated - skipping clustering")
        return isolated_assignments

    # Subset features for clustering (only non-isolated)
    clustering_feature_indices = np.array(clustering_indices)
    clustering_features = features[:, clustering_feature_indices]
    clustering_feature_names = [feature_names[i] for i in clustering_indices]

    # Determine subsample size for clustering
    subsample_size = None
    if N_samples > max_samples_for_clustering:
        subsample_size = max_samples_for_clustering
        logger.info(
            f"Will subsample {max_samples_for_clustering:,} of {N_samples:,} samples for clustering"
        )
    else:
        logger.info(f"Using all {N_samples:,} samples for clustering")

    # === HIERARCHICAL CLUSTERING ===
    # Compute correlation matrix if needed (for correlation distance or for CUDA optimization)
    corr_matrix = None
    if USE_CUDA and distance_metric == 'correlation':
        logger.info("Computing correlation matrix (CUDA)")
        corr_matrix = compute_correlation_matrix_cuda(
            clustering_features, subsample_size=subsample_size
        )
    elif distance_metric == 'correlation':
        logger.info("Computing correlation matrix (CPU)")
        corr_matrix, _ = compute_correlation_matrix_cpu(
            clustering_features, subsample_size=subsample_size
        )

    # Compute distance matrix with configurable metric
    logger.info(f"Computing distance matrix (metric: {distance_metric})")
    distance_matrix = compute_distance_matrix(clustering_features, distance_metric, corr_matrix)
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering with configurable linkage
    logger.info(f"Performing hierarchical clustering (linkage: {linkage_method})")
    linkage_matrix = sch.linkage(condensed_dist, method=linkage_method)

    # Export dendrogram if requested
    if export_dendrogram:
        family_name = "global" if not clustering_feature_names else "clustering"
        _export_dendrogram(
            linkage_matrix,
            clustering_feature_names,
            dendrogram_path,
            family_name
        )

    # Determine number of clusters
    if distance_threshold is not None:
        # Cut by distance threshold (no K constraint)
        labels = sch.fcluster(linkage_matrix, distance_threshold, criterion='distance')
        num_clusters = len(np.unique(labels))
        logger.info(f"Distance threshold {distance_threshold:.3f} → {num_clusters} clusters")
    else:
        # K-based approach with configurable selection method
        if num_clusters is None:
            # Auto-determine K
            if k_selection_method == 'manual':
                if manual_k is None:
                    raise ValueError("manual_k must be specified when k_selection_method='manual'")
                num_clusters = manual_k
                logger.info(f"Using manual K={num_clusters}")

            elif k_selection_method == 'gap_statistic':
                logger.info("Auto-determining K via gap statistic")
                num_clusters = _gap_statistic_k_selection(
                    clustering_features,
                    min_clusters,
                    max_clusters,
                    n_refs=gap_statistic_refs,
                    linkage_method=linkage_method,
                    distance_metric=distance_metric,
                    subsample_size=subsample_size,
                )

            elif k_selection_method == 'elbow':
                logger.info("Auto-determining K via elbow method")
                num_clusters = _elbow_k_selection(
                    clustering_features,
                    min_clusters,
                    max_clusters,
                    linkage_method=linkage_method,
                    distance_metric=distance_metric,
                    subsample_size=subsample_size,
                )

            elif k_selection_method == 'silhouette':
                logger.info("Auto-determining K via silhouette score")
                num_clusters = auto_determine_num_clusters(
                    clustering_features,
                    method="silhouette",
                    min_clusters=min_clusters,
                    max_clusters=max_clusters,
                    random_seed=random_seed,
                    subsample_size=subsample_size,
                )

            else:
                raise ValueError(f"Unknown K selection method: {k_selection_method}")

        # Cut dendrogram to get cluster labels
        labels = sch.fcluster(linkage_matrix, num_clusters, criterion="maxclust")

    # Build category assignments
    # Note: cluster_indices are indices into clustering_features, need to map back
    assignments = {}
    orphaned_features = []  # Track features in too-small clusters (if reassigning)

    for cluster_id in range(1, num_clusters + 1):
        # These are indices into the clustering subset
        subset_indices = np.where(labels == cluster_id)[0].tolist()

        # Handle small clusters
        if len(subset_indices) < min_features_per_cluster:
            if reassign_orphans:
                logger.info(
                    f"cluster_{cluster_id}: only {len(subset_indices)} features (will reassign)"
                )
                # Track orphaned features for reassignment
                orphaned_features.extend(subset_indices)
            else:
                logger.warning(
                    f"Skipping cluster_{cluster_id}: only {len(subset_indices)} features "
                    f"(min={min_features_per_cluster})"
                )
            continue

        # Map back to original feature indices
        original_indices = [clustering_indices[i] for i in subset_indices]

        category_name = f"cluster_{cluster_id}"
        assignments[category_name] = original_indices

        # Print cluster membership for inspection
        cluster_feat_names = [feature_names[i] for i in original_indices]
        preview = (
            cluster_feat_names[:3]
            if len(cluster_feat_names) <= 3
            else cluster_feat_names[:3] + ["..."]
        )
        logger.info(
            f"✓ {category_name}: {len(original_indices)} features - {preview}"
        )

    # Reassign orphaned features to nearest valid cluster
    if orphaned_features and len(assignments) > 0:
        logger.info(f"\nReassigning {len(orphaned_features)} orphaned features to nearest clusters...")

        for orphan_idx in orphaned_features:
            # Find nearest cluster by correlation distance
            orphan_feature = clustering_features[:, orphan_idx]
            min_dist = float('inf')
            best_cluster = None

            for cluster_name, cluster_orig_indices in assignments.items():
                # Get cluster features from original feature space
                cluster_subset_indices = [clustering_indices.index(idx) for idx in cluster_orig_indices]
                cluster_features = clustering_features[:, cluster_subset_indices]
                cluster_centroid = cluster_features.mean(axis=1)

                # Correlation distance
                corr_matrix = np.corrcoef(orphan_feature, cluster_centroid)
                dist = 1.0 - abs(corr_matrix[0, 1])

                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_name

            if best_cluster is not None:
                # Reassign orphan to best cluster
                orphan_original_idx = clustering_indices[orphan_idx]
                assignments[best_cluster].append(orphan_original_idx)
                logger.debug(
                    f"  ↳ Reassigned feature {feature_names[orphan_original_idx]} to {best_cluster} (dist={min_dist:.3f})"
                )

        logger.info(f"✓ All {len(orphaned_features)} orphaned features reassigned")

    # NEW: Split mega-categories if requested
    if split_mega_categories:
        # Use module-level function to avoid shadowing
        split_fn = globals()['split_mega_categories']
        assignments = split_fn(
            assignments=assignments,
            features=clustering_features,
            feature_names=clustering_feature_names,
            max_category_size=max_category_size,
            max_recursion_depth=max_split_recursion_depth,
            min_features_per_cluster=min_features_per_cluster,
            linkage_method=linkage_method,
            distance_metric=distance_metric,
            k_selection_method=k_selection_method,
            min_clusters=2,
            max_clusters_for_split=max_clusters_for_split,
            random_seed=random_seed,
            subsample_size=subsample_size,
            export_dendrogram=export_dendrogram,
            dendrogram_path=dendrogram_path,
        )

        # Remap to original indices (assignments use clustering subset indices)
        assignments = {
            name: [clustering_indices[i] for i in local_indices]
            for name, local_indices in assignments.items()
        }

    # Merge isolated assignments with clustered assignments
    all_assignments = {**isolated_assignments, **assignments}

    # Validate orthogonality on full feature set with all assignments
    max_corr = validate_cluster_orthogonality(features, all_assignments)
    logger.info(f"\nOrthogonality validation:")
    logger.info(f"  Max inter-cluster correlation: {max_corr:.3f}")
    logger.info(f"  Target: {orthogonality_target:.3f}")

    if max_corr > orthogonality_target:
        logger.warning(
            f"  Exceeded orthogonality target by {max_corr - orthogonality_target:.3f}"
        )
        logger.warning("     Consider: Increase num_clusters or use gradient refinement")
    else:
        logger.info(f"  ✓ Within target (margin: {orthogonality_target - max_corr:.3f})")

    if len(all_assignments) == 0:
        raise ValueError(
            f"All {num_clusters} clusters were too small (min={min_features_per_cluster}). "
            f"Try: Decrease min_features_per_cluster or decrease num_clusters"
        )

    return all_assignments


def extract_family_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    """Extract feature indices grouped by family from names.

    Args:
        feature_names: List of feature names (e.g., ["initial::initial_0", "temporal::temporal_5"])

    Returns:
        Dict mapping family name -> list of feature indices
        Example: {"initial": [0, 1, ..., 13], "temporal": [14, 15, ..., 141]}

    Note: For single-family configs without "::" prefix, returns {"default": [all_indices]}
    """
    family_groups = {}

    for idx, name in enumerate(feature_names):
        if "::" in name:
            family = name.split("::")[0]
        else:
            family = "default"  # Fallback for single-family configs

        if family not in family_groups:
            family_groups[family] = []
        family_groups[family].append(idx)

    return family_groups


def per_family_clustering_assignment(
    features: np.ndarray,
    feature_names: List[str],
    per_family_params: Dict[str, Dict[str, Any]],
    min_features_per_cluster: int = 3,
    orthogonality_target: float = 0.3,
    random_seed: int = 42,
    max_samples_for_clustering: int = 50000,
    isolated_families: Optional[List[str]] = None,
    reassign_orphans: bool = False,
    # NEW: Default clustering configuration (can be overridden per-family)
    linkage_method: str = "ward",
    distance_metric: str = "correlation",
    k_selection_method: str = "silhouette",
    manual_k: Optional[int] = None,
    gap_statistic_refs: int = 10,
    distance_threshold: Optional[float] = None,
    export_dendrogram: bool = False,
    dendrogram_path: str = "diagnostics/dendrograms",
    # NEW: Mega-category splitting
    split_mega_categories: bool = False,
    max_category_size: int = 40,
    max_split_recursion_depth: int = 3,
    max_clusters_for_split: int = 8,
) -> Dict[str, List[int]]:
    """Cluster features per-family independently.

    Args:
        features: [N_samples, N_features] data
        feature_names: List of feature names (with "family::" prefix for multi-family)
        per_family_params: Dict mapping family name -> clustering params
            Example: {
                "initial": {"min_clusters": 2, "max_clusters": 5},
                "temporal": {"min_clusters": 8, "max_clusters": 20}
            }
        min_features_per_cluster: Minimum features per cluster (applied per-family)
        orthogonality_target: Target max correlation (used for validation)
        random_seed: Random seed for reproducibility
        max_samples_for_clustering: Maximum samples for clustering
        isolated_families: Families to isolate (place in single category, skip clustering)
        reassign_orphans: Reassign small clusters within families
        linkage_method: Hierarchical linkage method (can be overridden per-family)
        distance_metric: Distance metric (can be overridden per-family)
        k_selection_method: K selection method (can be overridden per-family)
        manual_k: Manual K value if k_selection_method='manual'
        gap_statistic_refs: Number of reference distributions for gap statistic
        distance_threshold: Distance threshold for cutting dendrogram
        export_dendrogram: Whether to export dendrograms
        dendrogram_path: Path for dendrogram exports
        split_mega_categories: If True, recursively split categories > max_category_size
        max_category_size: Maximum features per category (default: 40)
        max_split_recursion_depth: Maximum recursion depth for splitting (default: 3)
        max_clusters_for_split: Maximum clusters to try when splitting (default: 8)

    Returns:
        Dict mapping category_name -> list of feature indices
        Example: {
            "initial_cluster_1": [0, 2, 5],
            "initial_cluster_2": [1, 3, 4],
            "temporal_cluster_1": [14, 18, 22],
            "temporal_cluster_2": [15, 16, 17, ...],
            ...
        }
    """
    # 1. Extract family groups from feature names
    family_groups = extract_family_groups(feature_names)
    logger.info(f"\nDetected {len(family_groups)} feature families:")
    for family, indices in family_groups.items():
        logger.info(f"  {family}: {len(indices)} features")

    # 2. Handle isolated families (skip clustering)
    assignments = {}
    families_to_cluster = []

    if isolated_families:
        for family in isolated_families:
            if family in family_groups:
                family_indices = family_groups[family]
                category_name = f"{family}_isolated"
                assignments[category_name] = family_indices
                logger.info(f"✓ {category_name}: {len(family_indices)} features (isolated)")
            else:
                logger.warning(f"Isolated family '{family}' not found in feature names")

        # Remaining families to cluster
        families_to_cluster = [f for f in family_groups if f not in (isolated_families or [])]
    else:
        families_to_cluster = list(family_groups.keys())

    # 3. Cluster each family independently
    for family in families_to_cluster:
        family_indices = family_groups[family]
        family_features = features[:, family_indices]
        family_feature_names = [feature_names[i] for i in family_indices]

        # Get family-specific clustering params (or use defaults)
        family_params = per_family_params.get(family, {})
        family_min_clusters = family_params.get("min_clusters", 2)
        family_max_clusters = family_params.get("max_clusters", min(12, len(family_indices) // 2))
        family_num_clusters = family_params.get("num_clusters", None)  # Explicit K

        # Allow per-family override of clustering parameters
        family_linkage = family_params.get("linkage_method", linkage_method)
        family_distance = family_params.get("distance_metric", distance_metric)
        family_k_selection = family_params.get("k_selection_method", k_selection_method)
        family_manual_k = family_params.get("manual_k", manual_k)
        family_gap_refs = family_params.get("gap_statistic_refs", gap_statistic_refs)
        family_dist_threshold = family_params.get("distance_threshold", distance_threshold)

        logger.info(f"\nClustering family '{family}' ({len(family_indices)} features):")
        logger.info(f"  min_clusters: {family_min_clusters}")
        logger.info(f"  max_clusters: {family_max_clusters}")
        logger.info(f"  num_clusters: {family_num_clusters or 'auto'}")
        logger.info(f"  linkage_method: {family_linkage}")
        logger.info(f"  distance_metric: {family_distance}")
        logger.info(f"  k_selection_method: {family_k_selection}")

        # Cluster this family (reuse existing function)
        family_assignments = hierarchical_clustering_assignment(
            features=family_features,
            feature_names=family_feature_names,
            num_clusters=family_num_clusters,
            min_features_per_cluster=min_features_per_cluster,
            orthogonality_target=orthogonality_target,
            random_seed=random_seed,
            max_samples_for_clustering=max_samples_for_clustering,
            min_clusters=family_min_clusters,
            max_clusters=family_max_clusters,
            isolated_families=None,  # Already handled above
            reassign_orphans=reassign_orphans,
            # Pass through clustering configuration (with per-family overrides)
            linkage_method=family_linkage,
            distance_metric=family_distance,
            k_selection_method=family_k_selection,
            manual_k=family_manual_k,
            gap_statistic_refs=family_gap_refs,
            distance_threshold=family_dist_threshold,
            export_dendrogram=export_dendrogram,
            dendrogram_path=dendrogram_path,
            # Pass through mega-category splitting
            split_mega_categories=split_mega_categories,
            max_category_size=max_category_size,
            max_split_recursion_depth=max_split_recursion_depth,
            max_clusters_for_split=max_clusters_for_split,
        )

        # 4. Map local indices → global indices and prefix category names
        for local_cat_name, local_indices in family_assignments.items():
            # Map from family subset indices to original feature indices
            global_indices = [family_indices[i] for i in local_indices]

            # Prefix category name with family
            global_cat_name = f"{family}_{local_cat_name}"
            assignments[global_cat_name] = global_indices

            logger.info(f"  ✓ {global_cat_name}: {len(global_indices)} features")

    # 5. Validate merged assignments
    logger.info(f"\nPer-family clustering complete:")
    logger.info(f"  Total categories: {len(assignments)}")
    logger.info(f"  Total features assigned: {sum(len(indices) for indices in assignments.values())}")

    return assignments


def _recursive_split_category(
    category_name: str,
    category_indices: List[int],
    features: np.ndarray,
    feature_names: List[str],
    max_category_size: int,
    current_depth: int,
    max_depth: int,
    min_features_per_cluster: int,
    linkage_method: str,
    distance_metric: str,
    k_selection_method: str,
    min_clusters: int,
    max_clusters_for_split: int,
    random_seed: int,
    subsample_size: Optional[int],
    export_dendrogram: bool,
    dendrogram_path: str,
) -> Dict[str, List[int]]:
    """Recursively split a single category if it exceeds max_category_size.

    Args:
        category_name: Name of the category being split
        category_indices: Global feature indices in this category
        features: Full feature matrix [N_samples, N_features]
        feature_names: Full list of feature names
        max_category_size: Maximum allowed category size
        current_depth: Current recursion depth
        max_depth: Maximum recursion depth
        min_features_per_cluster: Minimum features per cluster
        linkage_method: Hierarchical linkage method
        distance_metric: Distance metric for clustering
        k_selection_method: K selection method
        min_clusters: Minimum clusters for splitting
        max_clusters_for_split: Maximum clusters to try when splitting
        random_seed: Random seed
        subsample_size: Subsample size for clustering
        export_dendrogram: Whether to export dendrograms
        dendrogram_path: Path for dendrogram exports

    Returns:
        Dict mapping category_name -> list of global feature indices
        If no split occurs, returns {category_name: category_indices}
        If split occurs, returns {"name_split_0": [...], "name_split_1": [...], ...}
    """
    # Base case 1: Category is small enough
    if len(category_indices) <= max_category_size:
        return {category_name: category_indices}

    # Base case 2: Max recursion depth reached
    if current_depth >= max_depth:
        logger.warning(
            f"  Max recursion depth ({max_depth}) reached for {category_name} "
            f"({len(category_indices)} features). Accepting mega-category."
        )
        return {category_name: category_indices}

    # Base case 3: Too few features to split
    if len(category_indices) < min_features_per_cluster * 2:
        logger.warning(
            f"  Cannot split {category_name}: {len(category_indices)} features < "
            f"{min_features_per_cluster * 2} (2 * min_features_per_cluster)"
        )
        return {category_name: category_indices}

    # Extract features for this category
    category_features = features[:, category_indices]
    category_feature_names = [feature_names[i] for i in category_indices]

    logger.info(
        f"  Splitting {category_name} ({len(category_indices)} features) at depth {current_depth}..."
    )

    # Run hierarchical clustering on this subset
    try:
        # Determine max clusters for this split (scale with category size)
        effective_max_clusters = min(
            max_clusters_for_split,
            len(category_indices) // min_features_per_cluster
        )

        sub_assignments = hierarchical_clustering_assignment(
            features=category_features,
            feature_names=category_feature_names,
            num_clusters=None,  # Auto-determine
            min_features_per_cluster=min_features_per_cluster,
            orthogonality_target=0.3,  # Not critical for splitting
            random_seed=random_seed,
            max_samples_for_clustering=50000,
            min_clusters=min_clusters,
            max_clusters=effective_max_clusters,
            isolated_families=None,
            reassign_orphans=True,  # Ensure all features assigned
            linkage_method=linkage_method,
            distance_metric=distance_metric,
            k_selection_method=k_selection_method,
            manual_k=None,
            gap_statistic_refs=10,
            distance_threshold=None,
            export_dendrogram=export_dendrogram,
            dendrogram_path=dendrogram_path,
            # Disable recursive splitting within the recursive call (prevent infinite recursion)
            split_mega_categories=False,
        )

        # Check if splitting occurred (more than 1 sub-category)
        if len(sub_assignments) <= 1:
            logger.warning(
                f"  Silhouette chose K=1 for {category_name}. Cannot split (uniform data)."
            )
            return {category_name: category_indices}

        # Map local indices back to global indices
        result = {}
        for split_idx, (sub_name, local_indices) in enumerate(sub_assignments.items()):
            # Map from local (category subset) indices to global indices
            global_indices = [category_indices[i] for i in local_indices]

            # Generate split name
            split_name = f"{category_name}_split_{split_idx}"

            # Recursively split if sub-category is still too large
            sub_result = _recursive_split_category(
                category_name=split_name,
                category_indices=global_indices,
                features=features,
                feature_names=feature_names,
                max_category_size=max_category_size,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                min_features_per_cluster=min_features_per_cluster,
                linkage_method=linkage_method,
                distance_metric=distance_metric,
                k_selection_method=k_selection_method,
                min_clusters=min_clusters,
                max_clusters_for_split=max_clusters_for_split,
                random_seed=random_seed,
                subsample_size=subsample_size,
                export_dendrogram=export_dendrogram,
                dendrogram_path=dendrogram_path,
            )
            result.update(sub_result)

        logger.info(
            f"  ✓ Split {category_name} into {len(result)} sub-categories: "
            f"{list(result.keys())}"
        )
        return result

    except Exception as e:
        logger.error(
            f"  Error splitting {category_name}: {e}. Accepting mega-category."
        )
        return {category_name: category_indices}


def split_mega_categories(
    assignments: Dict[str, List[int]],
    features: np.ndarray,
    feature_names: List[str],
    max_category_size: int = 40,
    max_recursion_depth: int = 3,
    min_features_per_cluster: int = 3,
    linkage_method: str = "ward",
    distance_metric: str = "correlation",
    k_selection_method: str = "silhouette",
    min_clusters: int = 2,
    max_clusters_for_split: int = 8,
    random_seed: int = 42,
    subsample_size: Optional[int] = None,
    export_dendrogram: bool = False,
    dendrogram_path: str = "diagnostics/dendrograms",
) -> Dict[str, List[int]]:
    """Recursively split categories exceeding max_category_size.

    Identifies mega-categories and splits them using the same hierarchical
    clustering method, recursively until all categories are ≤ max_category_size
    or max_recursion_depth is reached.

    Args:
        assignments: Dict mapping category_name -> list of feature indices
        features: [N_samples, N_features] full feature matrix
        feature_names: Full list of feature names
        max_category_size: Maximum features per category (default: 40)
        max_recursion_depth: Maximum recursion depth (default: 3)
        min_features_per_cluster: Minimum features per cluster
        linkage_method: Hierarchical linkage method
        distance_metric: Distance metric for clustering
        k_selection_method: K selection method
        min_clusters: Minimum clusters for splitting (default: 2)
        max_clusters_for_split: Maximum clusters to try when splitting (default: 8)
        random_seed: Random seed for reproducibility
        subsample_size: Optional subsampling for large datasets
        export_dendrogram: Whether to export dendrograms for splits
        dendrogram_path: Path for dendrogram exports

    Returns:
        Updated assignments dict with mega-categories split
        Example: {
            "cluster_1": [0, 2, 5],  # Small category, unchanged
            "cluster_2_split_0": [1, 3, 4],  # Split from cluster_2
            "cluster_2_split_1": [6, 7, 8],  # Split from cluster_2
        }
    """
    # Identify mega-categories and non-mega categories
    mega_categories = {}
    normal_categories = {}

    for cat_name, indices in assignments.items():
        if len(indices) > max_category_size:
            mega_categories[cat_name] = indices
        else:
            normal_categories[cat_name] = indices

    # If no mega-categories, return early
    if not mega_categories:
        logger.info("No mega-categories found. Skipping split.")
        return assignments

    logger.info(
        f"\nSplitting {len(mega_categories)} mega-categories "
        f"(size > {max_category_size}):"
    )
    for cat_name, indices in mega_categories.items():
        logger.info(f"  {cat_name}: {len(indices)} features")

    # Split each mega-category recursively
    split_assignments = {}
    for cat_name, indices in mega_categories.items():
        result = _recursive_split_category(
            category_name=cat_name,
            category_indices=indices,
            features=features,
            feature_names=feature_names,
            max_category_size=max_category_size,
            current_depth=0,
            max_depth=max_recursion_depth,
            min_features_per_cluster=min_features_per_cluster,
            linkage_method=linkage_method,
            distance_metric=distance_metric,
            k_selection_method=k_selection_method,
            min_clusters=min_clusters,
            max_clusters_for_split=max_clusters_for_split,
            random_seed=random_seed,
            subsample_size=subsample_size,
            export_dendrogram=export_dendrogram,
            dendrogram_path=dendrogram_path,
        )
        split_assignments.update(result)

    # Merge normal + split categories
    final_assignments = {**normal_categories, **split_assignments}

    # Log summary
    logger.info(f"\nMega-category splitting complete:")
    logger.info(f"  Original categories: {len(assignments)}")
    logger.info(f"  Final categories: {len(final_assignments)}")
    logger.info(f"  Categories added: {len(final_assignments) - len(assignments)}")

    # Log size distribution
    sizes = [len(indices) for indices in final_assignments.values()]
    logger.info(f"  Category sizes: min={min(sizes)}, max={max(sizes)}, "
                f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")

    # Check if any mega-categories remain (couldn't be split)
    remaining_mega = [
        (name, len(indices))
        for name, indices in final_assignments.items()
        if len(indices) > max_category_size
    ]
    if remaining_mega:
        logger.warning(f"  Warning: {len(remaining_mega)} unsplittable mega-categories remain:")
        for name, size in remaining_mega:
            logger.warning(f"    {name}: {size} features (could not split)")

    return final_assignments


def validate_cluster_orthogonality(
    features: np.ndarray, assignments: Dict[str, List[int]]
) -> float:
    """Compute max absolute correlation between cluster centroids.

    This measures orthogonality: low correlation = independent categories.

    Args:
        features: [N_samples, N_features] data
        assignments: Dict mapping category_name -> list of feature indices

    Returns:
        Max absolute correlation between any two cluster centroids
    """
    category_names = list(assignments.keys())
    N_categories = len(category_names)

    if N_categories < 2:
        return 0.0  # Single category → no off-diagonal correlations

    # Compute cluster centroids (mean across features in each cluster)
    centroids = []
    for cat_name in category_names:
        indices = assignments[cat_name]
        centroid = features[:, indices].mean(axis=1)  # [N_samples]
        centroids.append(centroid)

    # Compute pairwise correlations between centroids
    max_corr = 0.0
    for i in range(N_categories):
        for j in range(i + 1, N_categories):
            corr, _ = pearsonr(centroids[i], centroids[j])
            max_corr = max(max_corr, abs(corr))

    return max_corr


def get_cluster_statistics(
    features: np.ndarray, assignments: Dict[str, List[int]]
) -> Dict[str, Dict]:
    """Compute detailed statistics for each cluster.

    Useful for debugging and understanding cluster quality.

    Args:
        features: [N_samples, N_features] data
        assignments: Dict mapping category_name -> list of feature indices

    Returns:
        Dict mapping category_name -> statistics dict with keys:
            - num_features: Number of features in cluster
            - within_cluster_corr: Mean |correlation| within cluster
            - between_cluster_corr: Mean |correlation| to other clusters
            - separation: within - between (negative is good)
    """
    stats = {}

    for cat_name, indices in assignments.items():
        # Within-cluster correlation
        if len(indices) > 1:
            within_corrs = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    corr, _ = pearsonr(
                        features[:, indices[i]], features[:, indices[j]]
                    )
                    within_corrs.append(abs(corr))
            within_cluster_corr = np.mean(within_corrs) if within_corrs else 0.0
        else:
            within_cluster_corr = 0.0

        # Between-cluster correlation
        other_indices = []
        for other_cat, other_idx in assignments.items():
            if other_cat != cat_name:
                other_indices.extend(other_idx)

        if len(other_indices) > 0:
            between_corrs = []
            for i in indices:
                for j in other_indices:
                    corr, _ = pearsonr(features[:, i], features[:, j])
                    between_corrs.append(abs(corr))
            between_cluster_corr = np.mean(between_corrs) if between_corrs else 0.0
        else:
            between_cluster_corr = 0.0

        stats[cat_name] = {
            "num_features": len(indices),
            "within_cluster_corr": within_cluster_corr,
            "between_cluster_corr": between_cluster_corr,
            "separation": within_cluster_corr
            - between_cluster_corr,  # Negative is good
        }

    return stats
