"""Temporal feature grouper."""

from typing import List
import numpy as np

from .base import FeatureGrouper
from .models import TemporalGroupingConfig, GroupingConfig


class TemporalFeatureGrouper(FeatureGrouper):
    """
    Grouper for temporal features.

    Temporal features typically:
    - Have strong temporal autocorrelation
    - Require more groups (8-20) due to diversity
    - Benefit from Ward linkage (variance minimization)
    """

    def get_default_config(self) -> GroupingConfig:
        """Get default config for temporal features."""
        return TemporalGroupingConfig()

    def validate_features(self, features: np.ndarray, feature_names: List[str]) -> None:
        """
        Validate temporal features.

        Args:
            features: Feature array [N, D]
            feature_names: Feature names

        Raises:
            ValueError: If features invalid
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {features.shape}")

        N, D = features.shape

        if N < self.config.min_samples_required:
            raise ValueError(
                f"Temporal features need at least {self.config.min_samples_required} samples "
                f"for robust clustering, got {N}"
            )

        if D < self.config.clustering.min_groups:
            raise ValueError(
                f"Temporal features has {D} dimensions but config requires "
                f"min_groups={self.config.clustering.min_groups}"
            )

    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess temporal features.

        Temporal-specific preprocessing:
        - MAD normalization (inherited from base)
        - Optional: temporal smoothing for noisy sequences
        """
        # Use base MAD normalization
        normalized = super().preprocess_features(features)

        # Could add temporal-specific preprocessing here
        # e.g., smoothing, detrending, etc.

        return normalized
