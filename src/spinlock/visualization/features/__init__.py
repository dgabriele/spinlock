"""
Feature visualization module for Spinlock.

Provides tools for visualizing extracted features from neural operators:
- Diversity-based operator sampling
- Time series plotting with uncertainty envelopes
- Category-based feature organization
- SVG export for documentation

Example:
    >>> from spinlock.visualization.features import select_diverse_operators, FeatureLinePlotter
    >>> # Sample diverse operators
    >>> indices = select_diverse_operators(parameters, features, n_select=2)
    >>> # Create visualization
    >>> plotter = FeatureLinePlotter()
    >>> plotter.create_tall_svg(features_by_category, timesteps, output_path)
"""

from .sampling import select_diverse_operators
from .plotter import FeatureLinePlotter
from .layout import FeatureLayoutManager
from .data_loader import (
    load_operator_features,
    load_parameters,
    load_per_timestep_features,
    check_dataset_compatibility,
)

__all__ = [
    "select_diverse_operators",
    "FeatureLinePlotter",
    "FeatureLayoutManager",
    "load_operator_features",
    "load_parameters",
    "load_per_timestep_features",
    "check_dataset_compatibility",
]
