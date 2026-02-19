"""Theta (parameter) feature grouper."""

from typing import List
import numpy as np

from .base import FeatureGrouper
from .models import ThetaGroupingConfig, GroupingConfig


class ThetaFeatureGrouper(FeatureGrouper):
    """
    Grouper for theta (operator parameter) features.

    Uses Ward hierarchical clustering with silhouette-based K selection
    (when num_groups is not set) to discover the natural grouping of the
    low-dimensional parameter space at runtime.
    """

    def get_default_config(self) -> GroupingConfig:
        """Get default config for theta features."""
        return ThetaGroupingConfig()

    def validate_features(self, features: np.ndarray, feature_names: List[str]) -> None:
        """
        Validate theta features.

        Args:
            features: Feature array [N, D] where D is typically 14
            feature_names: Feature names

        Raises:
            ValueError: If features invalid
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {features.shape}")

        N, D = features.shape

        if N < self.config.min_samples_required:
            raise ValueError(
                f"Theta features need at least {self.config.min_samples_required} samples, "
                f"got {N}"
            )

        # No upper bound on D, but typically 14
        if D < 1:
            raise ValueError(f"Theta features must have at least 1 dimension, got {D}")

    pass
