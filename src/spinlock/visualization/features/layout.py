"""
Category-based feature layout and organization.

Manages canonical ordering of feature categories and grouping
of features for visualization.
"""

import numpy as np
from typing import Dict
from spinlock.features.registry import FeatureRegistry


class FeatureLayoutManager:
    """
    Manages category ordering and feature grouping for visualization.

    Provides canonical ordering of categories (spatial first, then spectral, etc.)
    and organizes features by category for clean visualization layouts.

    Example:
        >>> manager = FeatureLayoutManager()
        >>> features = np.random.rand(250, 96)  # [T, D]
        >>> organized = manager.organize_features(features, registry)
        >>> print(organized.keys())
        dict_keys(['spatial', 'spectral', 'temporal', ...])
    """

    # Canonical category ordering for visualization
    # Per-timestep categories first (v1.0), then trajectory-level (v2.0)
    CATEGORY_ORDER = [
        # Per-timestep features (v1.0) - evolve over T timesteps
        "spatial",
        "spectral",
        "cross_channel",
        "distributional",
        "structural",
        "physics",
        "morphological",
        "multiscale",

        # Trajectory-level features (v2.0) - aggregated over T
        "temporal",
        "causality",
        "invariant_drift",
        "operator_sensitivity",
        "nonlinear",
    ]

    def __init__(self):
        """Initialize feature layout manager."""
        pass

    def organize_features(
        self,
        features: np.ndarray,
        registry: FeatureRegistry
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Organize features by category in canonical order.

        Args:
            features: Feature array [T, D] for single operator
                T = number of timesteps
                D = feature dimension
            registry: FeatureRegistry mapping names to indices

        Returns:
            Nested dictionary {category: {feature_name: values}}
            where values are [T] arrays

        Note:
            Only includes features that fit within the provided feature array.
            If registry has more features than the array (e.g., includes
            trajectory-level features but only per-timestep features loaded),
            out-of-bounds features are silently skipped.

        Example:
            >>> features = np.random.rand(250, 96)  # 250 timesteps, 96 features
            >>> organized = manager.organize_features(features, registry)
            >>> print(organized['spatial'].keys())
            dict_keys(['spatial_mean', 'spatial_std', 'spatial_skewness', ...])
            >>> print(organized['spatial']['spatial_mean'].shape)
            (250,)
        """
        organized = {}
        D = features.shape[1]  # Actual feature dimension

        for category in self.CATEGORY_ORDER:
            # Get all features in this category
            category_features = registry.get_features_by_category(category)

            if not category_features:
                # Category not present in this dataset
                continue

            # Extract features for this category (skip out-of-bounds)
            category_dict = {}
            for feat_meta in category_features:
                feat_idx = feat_meta.index
                feat_name = feat_meta.name

                # Skip if index out of bounds (e.g., trajectory features when loading per-timestep)
                if feat_idx >= D:
                    continue

                category_dict[feat_name] = features[:, feat_idx]  # [T]

            if category_dict:
                organized[category] = category_dict

        return organized

    def organize_multi_realization_features(
        self,
        features: np.ndarray,
        registry: FeatureRegistry
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Organize features with multiple realizations.

        Args:
            features: Feature array [M, T, D] for single operator
                M = number of realizations
                T = number of timesteps
                D = feature dimension
            registry: FeatureRegistry

        Returns:
            Nested dictionary {category: {feature_name: values}}
            where values are [M, T] arrays

        Note:
            This preserves realization dimension for plotting mean ± std envelopes.
        """
        M, T, D = features.shape
        organized = {}

        for category in self.CATEGORY_ORDER:
            category_features = registry.get_features_by_category(category)

            if not category_features:
                continue

            category_dict = {}
            for feat_meta in category_features:
                feat_idx = feat_meta.index
                feat_name = feat_meta.name
                category_dict[feat_name] = features[:, :, feat_idx]  # [M, T]

            if category_dict:
                organized[category] = category_dict

        return organized

    def filter_categories(
        self,
        organized_features: Dict[str, Dict[str, np.ndarray]],
        include_categories: list[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Filter features to include only specified categories.

        Args:
            organized_features: Organized feature dict
            include_categories: List of category names to include

        Returns:
            Filtered dict with only specified categories (in canonical order)

        Example:
            >>> filtered = manager.filter_categories(organized, ['spatial', 'spectral'])
            >>> print(filtered.keys())
            dict_keys(['spatial', 'spectral'])
        """
        filtered = {}

        for category in self.CATEGORY_ORDER:
            if category in include_categories and category in organized_features:
                filtered[category] = organized_features[category]

        return filtered

    def get_feature_count_by_category(
        self,
        registry: FeatureRegistry
    ) -> Dict[str, int]:
        """
        Get count of features in each category.

        Args:
            registry: FeatureRegistry

        Returns:
            Dictionary {category: count}

        Example:
            >>> counts = manager.get_feature_count_by_category(registry)
            >>> print(counts)
            {'spatial': 12, 'spectral': 15, 'temporal': 18, ...}
        """
        counts = {}

        for category in self.CATEGORY_ORDER:
            features = registry.get_features_by_category(category)
            if features:
                counts[category] = len(features)

        return counts

    def get_category_summary(
        self,
        organized_features: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, any]]:
        """
        Get summary statistics for each category.

        Args:
            organized_features: Organized feature dict

        Returns:
            Dict with category statistics:
                - num_features: int
                - feature_names: List[str]
                - value_range: Tuple[float, float] (min, max across all features)

        Example:
            >>> summary = manager.get_category_summary(organized)
            >>> print(summary['spatial'])
            {
                'num_features': 12,
                'feature_names': ['spatial_mean', 'spatial_std', ...],
                'value_range': (-2.5, 8.3)
            }
        """
        summary = {}

        for category, features in organized_features.items():
            feature_names = sorted(features.keys())
            all_values = np.concatenate([features[name].flatten()
                                        for name in feature_names])

            summary[category] = {
                'num_features': len(feature_names),
                'feature_names': feature_names,
                'value_range': (float(np.min(all_values)), float(np.max(all_values))),
                'mean_value': float(np.mean(all_values)),
                'std_value': float(np.std(all_values)),
            }

        return summary
