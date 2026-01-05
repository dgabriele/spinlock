"""Clustering-based category assignment for dynamic grouping.

Implements hierarchical clustering on feature correlation matrices to discover
optimal feature groupings. Uses Ward linkage with correlation distance (1 - |corr|).

Auto-determines optimal number of clusters via silhouette score maximization.

Automatically uses GPU acceleration (CUDA) when available for faster correlation
matrix computation.
"""

from typing import List, Dict, Optional
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


def hierarchical_clustering_assignment(
    features: np.ndarray,
    feature_names: List[str],
    num_clusters: Optional[int] = None,
    min_features_per_cluster: int = 3,
    orthogonality_target: float = 0.3,
    random_seed: int = 42,
    max_samples_for_clustering: int = 50000,
    max_clusters: int = 50,
    isolated_families: Optional[List[str]] = None,
) -> Dict[str, List[int]]:
    """Assign features to clusters using hierarchical clustering.

    Uses correlation distance (1 - |correlation|) with Ward linkage to cluster
    features based on their statistical similarity.

    **IMPORTANT**: For large datasets (>50K samples), subsamples for clustering
    to avoid prohibitive computational cost. Category discovery is based on
    feature correlations, which are stable with representative subsampling.

    Args:
        features: [N_samples, N_features] data
        feature_names: List of feature names (length N_features)
        num_clusters: Number of clusters (None = auto-determine via silhouette)
        min_features_per_cluster: Minimum features per cluster (prevents singletons)
        orthogonality_target: Target max correlation (used for validation warning)
        random_seed: Random seed for reproducibility
        max_samples_for_clustering: Maximum samples to use for clustering
        max_clusters: Maximum clusters to explore for auto-determination
        isolated_families: List of feature family names (e.g., ["architecture"]) that
            should be placed in their own dedicated categories, separate from clustering.
            Feature names must have format "family::name" for family detection.

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

    # Auto-determine num_clusters if not specified
    # Use clustering subset (excluding isolated families)
    if num_clusters is None:
        num_clusters = auto_determine_num_clusters(
            clustering_features,
            method="silhouette",
            max_clusters=max_clusters,
            random_seed=random_seed,
            subsample_size=subsample_size,
        )

    # Compute correlation matrix on clustering subset (use CUDA if available)
    if USE_CUDA:
        logger.info("Computing correlation matrix (CUDA)")
        corr_matrix_only = compute_correlation_matrix_cuda(
            clustering_features, subsample_size=subsample_size
        )
        # Convert to distance and condensed form for scipy
        distance_matrix = 1.0 - np.abs(corr_matrix_only)
        np.fill_diagonal(distance_matrix, 0.0)
        condensed_dist = squareform(distance_matrix, checks=False)
        corr_matrix = corr_matrix_only
    else:
        logger.info("Computing correlation matrix (CPU)")
        corr_matrix, condensed_dist = compute_correlation_matrix_cpu(
            clustering_features, subsample_size=subsample_size
        )

    linkage_matrix = sch.linkage(condensed_dist, method="ward")

    # Cut dendrogram to get cluster labels
    labels = sch.fcluster(linkage_matrix, num_clusters, criterion="maxclust")

    # Build category assignments
    # Note: cluster_indices are indices into clustering_features, need to map back
    assignments = {}
    skipped_clusters = []

    for cluster_id in range(1, num_clusters + 1):
        # These are indices into the clustering subset
        subset_indices = np.where(labels == cluster_id)[0].tolist()

        # Skip clusters that are too small
        if len(subset_indices) < min_features_per_cluster:
            skipped_clusters.append((cluster_id, len(subset_indices)))
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
