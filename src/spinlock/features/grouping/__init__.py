"""V2 Feature Grouping System.

This package provides a clean, inheritance-based architecture for feature grouping
(formerly "category discovery"). It abstracts clustering and gradient-based refinement
from VQ-VAE training, making it reusable across pipelines.

Key Features:
- Family-specific groupers (temporal, initial)
- Sequential pipeline: clustering → gradient refinement
- GPU-first (CuPy/PyTorch required)
- Type-safe Pydantic configuration
- Separation of concerns from training

Example:
    >>> from spinlock.v2.grouping import create_grouper
    >>> grouper = create_grouper("temporal")
    >>> result = grouper.group_features(features, feature_names)
    >>> print(f"Discovered {result.num_groups} groups")
"""

from .factory import create_grouper, FeatureFamily
from .base import FeatureGrouper
from .temporal import TemporalFeatureGrouper
from .initial import InitialFeatureGrouper
from .theta import ThetaFeatureGrouper
from .pca_grouper import PCAGrouper
from .opq import OPQGrouper
from .models import (
    GroupingConfig,
    TemporalGroupingConfig,
    InitialGroupingConfig,
    ThetaGroupingConfig,
    ClusteringParams,
    GradientParams,
    PreprocessingParams,
    SplittingParams,
    GroupingResult,
    FeatureGroup,
    LinkageMethod,
    DistanceMetric,
    KSelectionMethod,
)

__all__ = [
    # Factory
    "create_grouper",
    "FeatureFamily",
    # Base classes
    "FeatureGrouper",
    "TemporalFeatureGrouper",
    "InitialFeatureGrouper",
    "ThetaFeatureGrouper",
    "PCAGrouper",
    "OPQGrouper",
    # Configuration models
    "GroupingConfig",
    "TemporalGroupingConfig",
    "InitialGroupingConfig",
    "ThetaGroupingConfig",
    "ClusteringParams",
    "GradientParams",
    "PreprocessingParams",
    "SplittingParams",
    # Result models
    "GroupingResult",
    "FeatureGroup",
    # Enums
    "LinkageMethod",
    "DistanceMetric",
    "KSelectionMethod",
]
