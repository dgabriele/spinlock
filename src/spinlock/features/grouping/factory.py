"""Factory for creating family-specific groupers."""

from typing import Literal, Optional

from .base import FeatureGrouper
from .temporal import TemporalFeatureGrouper
from .initial import InitialFeatureGrouper
from .theta import ThetaFeatureGrouper
from .pca_grouper import PCAGrouper
from .models import GroupingConfig

FeatureFamily = Literal["temporal", "initial", "theta"]


def create_grouper(
    family: FeatureFamily,
    config: Optional[GroupingConfig] = None,
) -> FeatureGrouper:
    """
    Factory function to create family-specific grouper.

    For temporal features, the grouping method in config determines which
    grouper is used:
    - "correlation" (default): Ward hierarchical clustering (legacy path)
    - "pca_striped": PCA rotation + round-robin assignment (recommended)
    - "opq": FAISS OPQ — optimal product quantization (requires faiss-cpu)

    Args:
        family: Feature family name
        config: Optional config (uses family default if None).
                config.method controls the grouping algorithm for temporal.

    Returns:
        FeatureGrouper instance for the family

    Example:
        >>> grouper = create_grouper("temporal")
        >>> result = grouper.group_features(features, feature_names)
    """
    from .models import TemporalGroupingConfig, InitialGroupingConfig, ThetaGroupingConfig

    if family not in ("temporal", "initial", "theta"):
        raise ValueError(f"Unknown family: {family}. Must be one of ['temporal', 'initial', 'theta']")

    if config is None:
        config_map = {
            "temporal": TemporalGroupingConfig,
            "initial": InitialGroupingConfig,
            "theta": ThetaGroupingConfig,
        }
        config = config_map[family]()

    # For temporal, dispatch by method
    if family == "temporal":
        method = getattr(config, "method", "correlation")
        if method in ("pca_striped", "pca_raw"):
            return PCAGrouper(config)
        elif method == "opq":
            from .opq import OPQGrouper
            return OPQGrouper(config)
        # else: fall through to correlation grouper

    grouper_map = {
        "temporal": TemporalFeatureGrouper,
        "initial": InitialFeatureGrouper,
        "theta": ThetaFeatureGrouper,
    }
    return grouper_map[family](config)
