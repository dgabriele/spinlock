"""Initial feature grouper."""

from typing import List
import numpy as np

from .base import FeatureGrouper
from .models import InitialGroupingConfig, GroupingConfig


class InitialFeatureGrouper(FeatureGrouper):
    """
    Grouper for initial condition features.

    Initial features typically:
    - Capture spatial structure (not temporal)
    - Require fewer groups (2-5) due to simpler structure
    - Ward linkage suitable for spatial patterns
    """

    def get_default_config(self) -> GroupingConfig:
        """Get default config for initial features."""
        return InitialGroupingConfig()

    def validate_features(self, features: np.ndarray, feature_names: List[str]) -> None:
        """
        Validate initial features.

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
                f"Initial features need at least {self.config.min_samples_required} samples, "
                f"got {N}"
            )

        if self.config.clustering.min_groups is not None and D < self.config.clustering.min_groups:
            raise ValueError(
                f"Initial features has {D} dimensions but config requires "
                f"min_groups={self.config.clustering.min_groups}"
            )

    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess initial features.

        Initial features may have different scale properties than temporal.
        """
        # Use base MAD normalization
        normalized = super().preprocess_features(features)

        # Could add initial-specific preprocessing
        # e.g., spatial smoothing, edge detection, etc.

        return normalized
