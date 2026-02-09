"""Theta (parameter) feature grouper."""

from typing import List
import numpy as np

from .base import FeatureGrouper
from .models import ThetaGroupingConfig, GroupingConfig, GroupingResult, FeatureGroup


class ThetaFeatureGrouper(FeatureGrouper):
    """
    Grouper for theta (operator parameter) features.

    Theta features are low-dimensional (14D) continuous parameters
    in [0,1] range. We put all parameters in a single group since:
    - 14D is small enough for single VQ codebook
    - Hierarchical quantization (L0, L1, L2) provides granularity
    - Parameters are semantically related (operator specification)
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

    def group_features(
        self, features: np.ndarray, feature_names: List[str]
    ) -> GroupingResult:
        """
        Group theta features into a single group.

        For theta (parameters), we simply put all features in one group
        since they're semantically related and low-dimensional.

        Args:
            features: Feature array [N, D]
            feature_names: Feature names (length D)

        Returns:
            GroupingResult with single group containing all features

        Example:
            >>> grouper = ThetaFeatureGrouper()
            >>> features = np.random.rand(1000, 14)  # 1000 samples, 14 params
            >>> names = [f"theta_{i}" for i in range(14)]
            >>> result = grouper.group_features(features, names)
            >>> print(result.num_groups)  # 1
            >>> print(result.groups["group_1"].size)  # 14
        """
        # Validate inputs
        self.validate_features(features, feature_names)

        N, D = features.shape

        # Create single group with all features
        group = FeatureGroup(
            name="group_1",
            feature_indices=list(range(D)),
            feature_names=feature_names,
            size=D,
        )

        return GroupingResult(
            groups={"group_1": group},
            num_groups=1,
            total_features=D,
            config=self.config,
        )
