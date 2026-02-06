"""GPU-powered hierarchical clustering engine."""

from typing import Dict, List, Optional
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from .models import ClusteringParams, LinkageMethod, DistanceMetric, KSelectionMethod


class ClusteringEngine:
    """
    Hierarchical clustering with GPU acceleration.

    Provides:
    - Distance matrix computation (GPU via CuPy)
    - Hierarchical clustering (Ward, average, complete, single)
    - Automatic K selection (silhouette, gap statistic, elbow)
    """

    def __init__(self, config: ClusteringParams):
        self.config = config
        self._gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if CuPy GPU acceleration available."""
        if not self.config.use_gpu:
            return False
        try:
            import cupy as cp
            # Test GPU availability
            _ = cp.array([1.0])
            return True
        except (ImportError, Exception):
            raise ImportError(
                "CuPy is required for GPU clustering but not installed or GPU not available. "
                "Install with: pip install cupy-cuda12x"
            )

    def cluster(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, List[int]]:
        """
        Perform hierarchical clustering.

        Args:
            features: Normalized features [N, D]
            feature_names: Feature names

        Returns:
            Dict mapping group names to feature indices
        """
        # Compute distance matrix
        distance_matrix = self._compute_distances(features)

        # Perform hierarchical clustering
        linkage_matrix = self._compute_linkage(distance_matrix)

        # Determine K
        if self.config.num_groups is not None:
            num_clusters = self.config.num_groups
        else:
            num_clusters = self.select_k(features, linkage_matrix)

        # Cut dendrogram
        labels = hierarchy.fcluster(
            linkage_matrix,
            num_clusters,
            criterion='maxclust'
        )

        # Convert to dict
        groups = {}
        for cluster_id in range(1, num_clusters + 1):
            indices = np.where(labels == cluster_id)[0].tolist()
            if indices:
                groups[f"group_{cluster_id}"] = indices

        return groups

    def _compute_distances(self, features: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix.

        Args:
            features: Features [N, D]

        Returns:
            Distance matrix in condensed form
        """
        if self.config.distance_metric == DistanceMetric.CORRELATION:
            return self._correlation_distance(features)
        elif self.config.distance_metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_distance(features)
        elif self.config.distance_metric == DistanceMetric.COSINE:
            return self._cosine_distance(features)
        else:
            raise ValueError(f"Unknown metric: {self.config.distance_metric}")

    def _correlation_distance(self, features: np.ndarray) -> np.ndarray:
        """
        Compute correlation-based distance: 1 - |correlation|

        GPU-only (CuPy required).
        """
        if not self._gpu_available:
            raise RuntimeError("GPU clustering required but CuPy not available")
        return self._correlation_distance_gpu(features)

    def _correlation_distance_gpu(self, features: np.ndarray) -> np.ndarray:
        """GPU-accelerated correlation distance via CuPy."""
        import cupy as cp

        # Transfer to GPU
        features_gpu = cp.asarray(features, dtype=cp.float32)

        # Subsample if needed
        if self.config.subsample_size and features.shape[0] > self.config.subsample_size:
            indices = cp.random.choice(
                features_gpu.shape[0],
                self.config.subsample_size,
                replace=False
            )
            features_gpu = features_gpu[indices]

        # Standardize features
        mean = cp.mean(features_gpu, axis=0, keepdims=True)
        std = cp.std(features_gpu, axis=0, keepdims=True)
        std = cp.where(std > 1e-10, std, 1.0)
        features_normalized = (features_gpu - mean) / std

        # Correlation matrix: (1/N) * X^T @ X
        N = features_normalized.shape[0]
        corr_matrix = (1.0 / (N - 1)) * cp.dot(features_normalized.T, features_normalized)

        # Clip to valid correlation range
        corr_matrix = cp.clip(corr_matrix, -1.0, 1.0)

        # Distance: 1 - |corr|
        distance_matrix = 1.0 - cp.abs(corr_matrix)

        # Set diagonal to zero
        cp.fill_diagonal(distance_matrix, 0.0)

        # Convert to condensed form
        distance_np = cp.asnumpy(distance_matrix)

        # Cleanup GPU memory
        del features_gpu, features_normalized, corr_matrix, distance_matrix
        cp.get_default_memory_pool().free_all_blocks()

        # Convert to condensed form (upper triangular)
        return squareform(distance_np, checks=False)

    def _euclidean_distance(self, features: np.ndarray) -> np.ndarray:
        """Euclidean distance between feature columns."""
        from scipy.spatial.distance import pdist
        return pdist(features.T, metric='euclidean')

    def _cosine_distance(self, features: np.ndarray) -> np.ndarray:
        """Cosine distance between feature columns."""
        from scipy.spatial.distance import pdist
        return pdist(features.T, metric='cosine')

    def _compute_linkage(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Compute linkage matrix."""
        method_map = {
            LinkageMethod.WARD: "ward",
            LinkageMethod.AVERAGE: "average",
            LinkageMethod.COMPLETE: "complete",
            LinkageMethod.SINGLE: "single",
        }
        method = method_map[self.config.linkage_method]
        return hierarchy.linkage(distance_matrix, method=method)

    def select_k(
        self,
        features: np.ndarray,
        linkage_matrix: np.ndarray,
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
    ) -> int:
        """
        Automatically select number of clusters.

        Args:
            features: Features [N, D]
            linkage_matrix: Precomputed linkage matrix
            min_k: Minimum K to try (default: config.min_groups)
            max_k: Maximum K to try (default: config.max_groups)

        Returns:
            Optimal number of clusters
        """
        min_k = min_k or self.config.min_groups
        max_k = max_k or self.config.max_groups

        # Ensure valid range
        max_k = min(max_k, features.shape[1] - 1)
        if min_k > max_k:
            min_k = max_k

        if self.config.k_selection_method == KSelectionMethod.SILHOUETTE:
            return self._silhouette_selection(features, linkage_matrix, min_k, max_k)
        elif self.config.k_selection_method == KSelectionMethod.GAP_STATISTIC:
            return self._gap_statistic_selection(features, linkage_matrix, min_k, max_k)
        elif self.config.k_selection_method == KSelectionMethod.ELBOW:
            return self._elbow_selection(features, linkage_matrix, min_k, max_k)
        else:
            raise ValueError(f"Unknown K selection: {self.config.k_selection_method}")

    def _silhouette_selection(
        self,
        features: np.ndarray,
        linkage_matrix: np.ndarray,
        min_k: int,
        max_k: int,
    ) -> int:
        """Select K via silhouette score maximization."""
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import pdist, squareform

        # Compute pairwise distances between features (transpose to get feature-wise)
        distance_matrix = pdist(features.T, metric='euclidean')
        distance_matrix_square = squareform(distance_matrix)

        best_k = min_k
        best_score = -1.0

        for k in range(min_k, max_k + 1):
            labels = hierarchy.fcluster(linkage_matrix, k, criterion='maxclust')
            if len(np.unique(labels)) < 2:
                continue
            try:
                score = silhouette_score(distance_matrix_square, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_k = k
            except ValueError:
                # Can fail if clusters are invalid
                continue

        return best_k

    def _gap_statistic_selection(
        self,
        features: np.ndarray,
        linkage_matrix: np.ndarray,
        min_k: int,
        max_k: int,
    ) -> int:
        """Select K via gap statistic."""
        # TODO: Implement gap statistic
        # For now, fall back to silhouette
        print("Warning: Gap statistic not implemented, using silhouette instead")
        return self._silhouette_selection(features, linkage_matrix, min_k, max_k)

    def _elbow_selection(
        self,
        features: np.ndarray,
        linkage_matrix: np.ndarray,
        min_k: int,
        max_k: int,
    ) -> int:
        """Select K via elbow method."""
        # TODO: Implement elbow method
        # For now, fall back to silhouette
        print("Warning: Elbow method not implemented, using silhouette instead")
        return self._silhouette_selection(features, linkage_matrix, min_k, max_k)
