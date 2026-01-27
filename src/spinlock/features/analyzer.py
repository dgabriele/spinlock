"""Feature quality analysis for pruning low-value features.

Analyzes extracted features to identify candidates for pruning:
- Near-constant features (low variance)
- Redundant features (high correlation)
- High-NaN features (poor quality)

Generates actionable recommendations for feature selection before VQ-VAE training.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


@dataclass
class AnalysisMetadata:
    """Metadata for feature analysis."""
    dataset_path: str
    feature_family: str
    analysis_timestamp: str
    total_samples: int
    total_timesteps: Optional[int]
    total_features: int
    thresholds: Dict[str, float]


class FeatureAnalyzer:
    """Analyze feature quality and identify low-value features for pruning.

    Performs comprehensive analysis across three dimensions:
    1. Variance: Identify near-constant features
    2. Correlation: Find redundant feature groups
    3. NaN Frequency: Flag unreliable features

    Attributes:
        variance_threshold: Minimum variance for features to keep
        correlation_threshold: Max correlation for features to be considered independent
        nan_threshold: Max allowed fraction of NaN values
    """

    def __init__(
        self,
        variance_threshold: float = 1e-8,
        correlation_threshold: float = 0.99,
        nan_threshold: float = 0.01,
    ):
        """Initialize feature analyzer.

        Args:
            variance_threshold: Variance threshold for near-zero features (default: 1e-8)
            correlation_threshold: Correlation threshold for redundancy (default: 0.99)
            nan_threshold: Max NaN fraction (default: 0.01 = 1%)
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.nan_threshold = nan_threshold

    def analyze(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[AnalysisMetadata] = None,
        max_samples_for_correlation: int = 100000
    ) -> Dict[str, Any]:
        """Run complete analysis pipeline.

        Args:
            features: Feature array [N, D] where N is samples (potentially N*T if temporal)
            feature_names: Optional feature names for reporting
            metadata: Optional analysis metadata
            max_samples_for_correlation: Max samples to use for correlation analysis (default: 100k)

        Returns:
            Comprehensive analysis results dictionary
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        # Build metadata if not provided
        if metadata is None:
            metadata = AnalysisMetadata(
                dataset_path="unknown",
                feature_family="unknown",
                analysis_timestamp=datetime.now().isoformat(),
                total_samples=features.shape[0],
                total_timesteps=None,
                total_features=features.shape[1],
                thresholds={
                    "variance": self.variance_threshold,
                    "correlation": self.correlation_threshold,
                    "nan": self.nan_threshold,
                }
            )

        # Run individual analyses
        variance_results = self._analyze_variance(features)
        correlation_results = self._analyze_correlation(
            features,
            feature_names,
            max_samples_for_correlation
        )
        nan_results = self._analyze_nans(features)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            variance_results,
            correlation_results,
            nan_results,
            feature_names
        )

        # Compile results
        results = {
            "metadata": {
                "dataset_path": metadata.dataset_path,
                "feature_family": metadata.feature_family,
                "analysis_timestamp": metadata.analysis_timestamp,
                "total_samples": metadata.total_samples,
                "total_timesteps": metadata.total_timesteps,
                "total_features": metadata.total_features,
                "thresholds": metadata.thresholds,
            },
            "summary": {
                "features_analyzed": metadata.total_features,
                "zero_variance_features": len(variance_results["zero_variance_indices"]),
                "high_nan_features": len(nan_results["high_nan_indices"]),
                "redundant_features": len(recommendations["prune_indices"]) -
                                      len(variance_results["zero_variance_indices"]) -
                                      len(nan_results["high_nan_indices"]),
                "recommended_keep": len(recommendations["keep_indices"]),
                "recommended_prune": len(recommendations["prune_indices"]),
                "compression_ratio": len(recommendations["keep_indices"]) / metadata.total_features,
            },
            "analysis": {
                "variance": variance_results,
                "correlation": correlation_results,
                "nan_analysis": nan_results,
            },
            "recommendations": recommendations,
        }

        return results

    def _analyze_variance(self, features: np.ndarray) -> Dict[str, Any]:
        """Compute variance metrics.

        Args:
            features: Feature array [N, D]

        Returns:
            Dict containing variance analysis results
        """
        # Handle NaN values by computing nanvar/nanstd
        with np.errstate(invalid='ignore'):
            per_feature_variance = np.nanvar(features, axis=0)
            per_feature_std = np.nanstd(features, axis=0)
            per_feature_mean = np.nanmean(features, axis=0)

            # Coefficient of variation (CV = std/mean), handle div by zero
            cv = np.zeros_like(per_feature_std)
            nonzero_mean = np.abs(per_feature_mean) > 1e-10
            cv[nonzero_mean] = per_feature_std[nonzero_mean] / np.abs(per_feature_mean[nonzero_mean])

        # Identify zero/low variance features
        zero_variance_mask = per_feature_variance < self.variance_threshold
        zero_variance_indices = np.where(zero_variance_mask)[0].tolist()

        # Low variance (above threshold but still small)
        low_variance_threshold = self.variance_threshold * 100
        low_variance_mask = (per_feature_variance >= self.variance_threshold) & \
                           (per_feature_variance < low_variance_threshold)
        low_variance_indices = np.where(low_variance_mask)[0].tolist()

        return {
            "per_feature_variance": per_feature_variance.tolist(),
            "per_feature_std": per_feature_std.tolist(),
            "per_feature_mean": per_feature_mean.tolist(),
            "per_feature_cv": cv.tolist(),
            "zero_variance_indices": zero_variance_indices,
            "low_variance_indices": low_variance_indices,
        }

    def _analyze_correlation(
        self,
        features: np.ndarray,
        feature_names: List[str],
        max_samples: int = 100000
    ) -> Dict[str, Any]:
        """Find redundant features via correlation analysis.

        Uses hierarchical clustering on correlation matrix to group
        highly correlated features, then applies informativeness scoring
        to select the best feature from each group.

        Args:
            features: Feature array [N, D]
            feature_names: List of feature names
            max_samples: Max samples to use for correlation (default: 100k)

        Returns:
            Dict containing correlation analysis results
        """
        D = features.shape[1]
        N = features.shape[0]

        # Subsample if too many samples for efficient correlation computation
        if N > max_samples:
            indices = np.random.choice(N, max_samples, replace=False)
            features_for_corr = features[indices]
        else:
            features_for_corr = features

        # Compute correlation matrix (handle NaNs)
        with np.errstate(invalid='ignore'):
            # Use masked arrays to handle NaN
            masked_features = np.ma.masked_invalid(features_for_corr)
            corr_matrix = np.ma.corrcoef(masked_features, rowvar=False).filled(0.0)

        # Find highly correlated pairs
        highly_correlated_pairs = []
        for i in range(D):
            for j in range(i + 1, D):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    highly_correlated_pairs.append({
                        "feature_i": int(i),
                        "feature_j": int(j),
                        "feature_i_name": feature_names[i],
                        "feature_j_name": feature_names[j],
                        "correlation": float(corr_matrix[i, j])
                    })

        # Identify redundant groups using hierarchical clustering
        redundant_groups = self._cluster_correlated_features(
            corr_matrix,
            features,
            feature_names
        )

        return {
            "correlation_matrix": corr_matrix.tolist(),
            "highly_correlated_pairs": highly_correlated_pairs,
            "redundant_groups": redundant_groups,
        }

    def _cluster_correlated_features(
        self,
        corr_matrix: np.ndarray,
        features: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Cluster correlated features and select best from each group.

        Args:
            corr_matrix: Correlation matrix [D, D]
            features: Feature array [N, D] for informativeness scoring
            feature_names: List of feature names

        Returns:
            List of redundant feature groups with keep/prune decisions
        """
        D = corr_matrix.shape[0]

        # Convert correlation to distance (1 - |corr|)
        distance_matrix = 1.0 - np.abs(corr_matrix)
        np.fill_diagonal(distance_matrix, 0.0)

        # Convert to condensed distance matrix for scipy
        try:
            condensed_dist = squareform(distance_matrix, checks=False)
        except ValueError:
            # If there's an issue, return empty groups
            return []

        # Hierarchical clustering
        try:
            linkage_matrix = hierarchy.linkage(condensed_dist, method='average')

            # Cut tree at correlation threshold
            distance_threshold = 1.0 - self.correlation_threshold
            cluster_labels = hierarchy.fcluster(linkage_matrix, distance_threshold, criterion='distance')
        except Exception:
            # If clustering fails, return empty groups
            return []

        # Group features by cluster
        redundant_groups = []
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Only consider groups with 2+ features
            if len(cluster_indices) < 2:
                continue

            # Compute informativeness scores for group members
            informativeness_scores = self._compute_informativeness(
                features[:, cluster_indices]
            )

            # Keep feature with highest informativeness
            best_idx_in_group = np.argmax(informativeness_scores)
            keep_feature_idx = int(cluster_indices[best_idx_in_group])
            prune_indices = [int(idx) for i, idx in enumerate(cluster_indices)
                           if i != best_idx_in_group]

            redundant_groups.append({
                "features": cluster_indices.tolist(),
                "feature_names": [feature_names[i] for i in cluster_indices],
                "keep": keep_feature_idx,
                "keep_name": feature_names[keep_feature_idx],
                "prune": prune_indices,
                "prune_names": [feature_names[i] for i in prune_indices],
                "informativeness_scores": informativeness_scores.tolist(),
                "reason": f"Redundant with {feature_names[keep_feature_idx]} (correlation > {self.correlation_threshold})"
            })

        return redundant_groups

    def _compute_informativeness(self, features: np.ndarray) -> np.ndarray:
        """Compute informativeness score for features.

        Informativeness = variance * (1 - NaN_fraction)
        Higher score means more informative feature.

        Args:
            features: Feature array [N, D]

        Returns:
            Informativeness scores [D]
        """
        with np.errstate(invalid='ignore'):
            variance = np.nanvar(features, axis=0)
            nan_fraction = np.isnan(features).mean(axis=0)
            informativeness = variance * (1.0 - nan_fraction)

        # Handle edge cases
        informativeness = np.nan_to_num(informativeness, nan=0.0, posinf=0.0, neginf=0.0)

        return informativeness

    def _analyze_nans(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze NaN patterns across features.

        Args:
            features: Feature array [N, D]

        Returns:
            Dict containing NaN analysis results
        """
        # Count NaNs per feature
        nan_count_per_feature = np.isnan(features).sum(axis=0)
        total_values = features.shape[0]
        nan_fraction_per_feature = nan_count_per_feature / total_values

        # Identify high-NaN features
        high_nan_mask = nan_fraction_per_feature > self.nan_threshold
        high_nan_indices = np.where(high_nan_mask)[0].tolist()

        return {
            "per_feature_nan_count": nan_count_per_feature.tolist(),
            "per_feature_nan_fraction": nan_fraction_per_feature.tolist(),
            "high_nan_indices": high_nan_indices,
            "total_values_per_feature": int(total_values),
        }

    def _generate_recommendations(
        self,
        variance_results: Dict[str, Any],
        correlation_results: Dict[str, Any],
        nan_results: Dict[str, Any],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate pruning recommendations from all analyses.

        Args:
            variance_results: Results from variance analysis
            correlation_results: Results from correlation analysis
            nan_results: Results from NaN analysis
            feature_names: List of feature names

        Returns:
            Dict with keep/prune indices and reasons
        """
        total_features = len(feature_names)
        prune_indices = set()
        prune_reasons = {}

        # Add zero variance features
        for idx in variance_results["zero_variance_indices"]:
            prune_indices.add(idx)
            prune_reasons[idx] = "zero_variance"

        # Add high NaN features
        for idx in nan_results["high_nan_indices"]:
            prune_indices.add(idx)
            prune_reasons[idx] = "high_nan_frequency"

        # Add redundant features from correlation analysis
        for group in correlation_results["redundant_groups"]:
            keep_idx = group["keep"]
            for prune_idx in group["prune"]:
                prune_indices.add(prune_idx)
                prune_reasons[prune_idx] = f"redundant_with_{keep_idx}_{feature_names[keep_idx]}"

        # Create keep indices (all indices not in prune set)
        keep_indices = sorted([i for i in range(total_features) if i not in prune_indices])
        prune_indices = sorted(list(prune_indices))

        return {
            "keep_indices": keep_indices,
            "prune_indices": prune_indices,
            "prune_reasons": {str(k): v for k, v in prune_reasons.items()},
        }
