"""Abstract base class for feature grouping."""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from .models import GroupingConfig, GroupingResult, FeatureGroup
from .clustering import ClusteringEngine
from .gradient import GradientRefiner
from .splitter import RecursiveSplitter


class FeatureGrouper(ABC):
    """
    Abstract base class for feature grouping.

    Provides shared infrastructure for clustering, gradient refinement,
    and recursive splitting. Subclasses implement family-specific logic.
    """

    def __init__(self, config: GroupingConfig):
        """
        Initialize grouper.

        Args:
            config: Grouping configuration (validated Pydantic model)
        """
        self.config = config

        # Shared components (composition)
        self.clustering_engine = ClusteringEngine(config.clustering)
        self.gradient_refiner = GradientRefiner(config.gradient)
        self.recursive_splitter = RecursiveSplitter(config.splitting)

    @abstractmethod
    def get_default_config(self) -> GroupingConfig:
        """
        Get default configuration for this feature family.

        Returns:
            Default GroupingConfig
        """
        pass

    @abstractmethod
    def validate_features(self, features: np.ndarray, feature_names: List[str]) -> None:
        """
        Validate that features are appropriate for this family.

        Args:
            features: Feature array [N, D]
            feature_names: Feature names

        Raises:
            ValueError: If features invalid for this family
        """
        pass

    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features before grouping.

        Uses configurable preprocessing method from config.

        Args:
            features: Raw features [N, D]

        Returns:
            Preprocessed features [N, D]
        """
        method = self.config.preprocessing.method

        if method == "mad":
            return self._mad_normalize(features)
        elif method == "zscore":
            return self._zscore_normalize(features)
        elif method == "minmax":
            return self._minmax_normalize(features)
        elif method == "none":
            return features
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    def _mad_normalize(self, features: np.ndarray) -> np.ndarray:
        """MAD (Median Absolute Deviation) normalization."""
        normalized = np.zeros_like(features)
        mad_constant = self.config.preprocessing.mad_constant

        for j in range(features.shape[1]):
            col = features[:, j]
            median = np.median(col)
            mad = np.median(np.abs(col - median)) * mad_constant
            if mad > 1e-10:
                normalized[:, j] = (col - median) / mad
            else:
                normalized[:, j] = col - median

        # Optional clipping
        if self.config.preprocessing.clip_outliers:
            threshold = self.config.preprocessing.clip_std_threshold
            std = np.std(normalized, axis=0)
            normalized = np.clip(normalized, -threshold * std, threshold * std)

        return normalized

    def _zscore_normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        normalized = np.where(std > 1e-10, (features - mean) / std, features - mean)
        return normalized

    def _minmax_normalize(self, features: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val
        normalized = np.where(range_val > 1e-10, (features - min_val) / range_val, 0.0)
        return normalized

    def group_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> GroupingResult:
        """
        Main entry point for feature grouping.

        Sequential pipeline: clustering → gradient refinement

        Args:
            features: Feature array [N, D]
            feature_names: Feature names (length D)

        Returns:
            GroupingResult with discovered groups
        """
        # Validate
        self.validate_features(features, feature_names)

        # Preprocess
        normalized = self.preprocess_features(features)

        # Stage 1: Clustering initialization
        group_dict = self.clustering_engine.cluster(normalized, feature_names)

        # Recursive splitting if enabled (before gradient)
        if self.config.splitting.enabled:
            group_dict = self.recursive_splitter.split(
                group_dict, normalized, feature_names
            )

        # Stage 2: Gradient refinement (optional)
        if not self.config.skip_gradient_refinement:
            num_groups = len(group_dict)
            group_dict = self.gradient_refiner.refine(
                normalized, feature_names, num_groups, init_groups=group_dict
            )

        # Convert to result object
        return self._to_result(group_dict, feature_names)

    def _to_result(
        self,
        group_dict: Dict[str, List[int]],
        feature_names: List[str],
    ) -> GroupingResult:
        """Convert dict to GroupingResult."""
        groups = {}
        for name, indices in group_dict.items():
            groups[name] = FeatureGroup(
                name=name,
                feature_indices=indices,
                feature_names=[feature_names[i] for i in indices],
                size=len(indices),
            )

        return GroupingResult(
            groups=groups,
            num_groups=len(groups),
            total_features=len(feature_names),
            config=self.config,
        )
