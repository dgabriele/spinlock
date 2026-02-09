"""Factory for creating family-specific groupers."""

from typing import Literal, Optional

from .base import FeatureGrouper
from .temporal import TemporalFeatureGrouper
from .initial import InitialFeatureGrouper
from .theta import ThetaFeatureGrouper
from .models import GroupingConfig

FeatureFamily = Literal["temporal", "initial", "theta"]


def create_grouper(
    family: FeatureFamily,
    config: Optional[GroupingConfig] = None,
) -> FeatureGrouper:
    """
    Factory function to create family-specific grouper.

    Args:
        family: Feature family name
        config: Optional config (uses family default if None)

    Returns:
        FeatureGrouper instance for the family

    Example:
        >>> grouper = create_grouper("temporal")
        >>> result = grouper.group_features(features, feature_names)
    """
    from .models import TemporalGroupingConfig, InitialGroupingConfig, ThetaGroupingConfig

    grouper_map = {
        "temporal": TemporalFeatureGrouper,
        "initial": InitialFeatureGrouper,
        "theta": ThetaFeatureGrouper,
    }

    config_map = {
        "temporal": TemporalGroupingConfig,
        "initial": InitialGroupingConfig,
        "theta": ThetaGroupingConfig,
    }

    if family not in grouper_map:
        raise ValueError(f"Unknown family: {family}. Must be one of {list(grouper_map.keys())}")

    grouper_class = grouper_map[family]

    if config is None:
        # Use family-specific default config
        config = config_map[family]()

    return grouper_class(config)
