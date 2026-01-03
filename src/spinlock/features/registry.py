"""
Feature registry for dynamic feature name-to-index mapping.

The registry system handles:
- Dynamic feature names (e.g., multiscale features like "haar_scale1_energy")
- Consistent feature ordering across extraction runs
- Category-based organization
- Serialization to/from JSON for HDF5 storage

Example:
    >>> registry = FeatureRegistry("summary")
    >>> idx = registry.register("spatial_mean", "spatial", "Mean field value")
    >>> idx = registry.register("fft_power_scale_0", "spectral", "FFT power in band 0")
    >>>
    >>> # Export to JSON for HDF5 storage
    >>> json_str = registry.to_json()
    >>>
    >>> # Load from HDF5
    >>> loaded_registry = FeatureRegistry.from_json(json_str, "summary")
    >>> assert loaded_registry.get_index("spatial_mean") == 0
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class FeatureMetadata:
    """
    Metadata for a single feature.

    Attributes:
        name: Unique feature name (e.g., "spatial_mean", "fft_power_scale_0_mean")
        index: Integer index for array indexing
        category: Feature category (e.g., "spatial", "spectral", "temporal")
        dtype: Data type (default "float32")
        description: Human-readable description
        multiscale_index: Optional index if part of multiscale family (e.g., scale number)
    """
    name: str
    index: int
    category: str
    dtype: str = "float32"
    description: str = ""
    multiscale_index: Optional[int] = None


@dataclass
class CategoryInfo:
    """
    Information about a feature category.

    Attributes:
        name: Category name (e.g., "spatial", "spectral")
        count: Number of features in this category
        start_idx: Starting index for this category's features
        end_idx: Ending index (exclusive) for this category's features
    """
    name: str
    count: int = 0
    start_idx: int = 0
    end_idx: int = 0


class FeatureRegistry:
    """
    Registry mapping feature names to indices.

    Supports dynamic feature generation (e.g., multiscale with variable scales).
    Ensures consistent ordering across extraction runs via explicit registration.

    The registry organizes features by category for easier analysis and selection.

    Attributes:
        family_name: Feature family identifier (e.g., "summary")
        features: Dict mapping feature names to metadata
        categories: Dict mapping category names to CategoryInfo
    """

    def __init__(self, family_name: str = "summary"):
        """
        Initialize feature registry.

        Args:
            family_name: Feature family identifier (default: "summary")
        """
        self.family_name = family_name
        self._features: Dict[str, FeatureMetadata] = {}
        self._index_counter = 0
        self._categories: Dict[str, CategoryInfo] = {}

    def register(
        self,
        name: str,
        category: str,
        description: str = "",
        multiscale_index: Optional[int] = None
    ) -> int:
        """
        Register a feature and get its index.

        Features are registered in order of first call. If a feature is already
        registered, returns its existing index without modification.

        Args:
            name: Feature name (must be unique within registry)
            category: Feature category for organization
            description: Human-readable description
            multiscale_index: Optional scale index for multiscale features

        Returns:
            Feature index (integer)

        Raises:
            ValueError: If name is empty or contains invalid characters

        Example:
            >>> reg = FeatureRegistry("summary")
            >>> reg.register("spatial_mean", "spatial", "Mean field value")
            0
            >>> reg.register("spatial_std", "spatial", "Std dev field value")
            1
            >>> reg.register("fft_power_scale_0", "spectral", "FFT power band 0", multiscale_index=0)
            2
        """
        if not name or not name.strip():
            raise ValueError("Feature name cannot be empty")

        # If already registered, return existing index
        if name in self._features:
            return self._features[name].index

        # Create metadata
        metadata = FeatureMetadata(
            name=name,
            index=self._index_counter,
            category=category,
            description=description,
            multiscale_index=multiscale_index
        )

        # Register feature
        self._features[name] = metadata
        self._index_counter += 1

        # Update category info
        if category not in self._categories:
            self._categories[category] = CategoryInfo(name=category)

        self._categories[category].count += 1

        return metadata.index

    def register_batch(
        self,
        names: List[str],
        category: str,
        description_template: str = ""
    ) -> List[int]:
        """
        Register multiple features in the same category.

        Convenience method for registering many similar features.

        Args:
            names: List of feature names to register
            category: Category for all features
            description_template: Optional description template (can include {name})

        Returns:
            List of feature indices

        Example:
            >>> reg = FeatureRegistry("summary")
            >>> names = ["mean", "std", "skewness", "kurtosis"]
            >>> indices = reg.register_batch(
            ...     [f"spatial_{n}" for n in names],
            ...     "spatial",
            ...     "Spatial {name} statistic"
            ... )
        """
        indices = []
        for name in names:
            desc = description_template.format(name=name) if description_template else ""
            idx = self.register(name, category, desc)
            indices.append(idx)
        return indices

    def get_index(self, name: str) -> Optional[int]:
        """
        Get index for a feature name.

        Args:
            name: Feature name

        Returns:
            Feature index if found, None otherwise

        Example:
            >>> reg = FeatureRegistry("summary")
            >>> reg.register("spatial_mean", "spatial")
            0
            >>> reg.get_index("spatial_mean")
            0
            >>> reg.get_index("nonexistent")
            None
        """
        if name in self._features:
            return self._features[name].index
        return None

    def get_feature(self, name: str) -> Optional[FeatureMetadata]:
        """
        Get metadata for a feature.

        Args:
            name: Feature name

        Returns:
            FeatureMetadata if found, None otherwise
        """
        return self._features.get(name)

    def get_features_by_category(self, category: str) -> List[FeatureMetadata]:
        """
        Get all features in a category.

        Args:
            category: Category name

        Returns:
            List of FeatureMetadata in the category, sorted by index

        Example:
            >>> reg = FeatureRegistry("summary")
            >>> reg.register("spatial_mean", "spatial")
            >>> reg.register("spatial_std", "spatial")
            >>> reg.register("spectral_centroid", "spectral")
            >>> spatial = reg.get_features_by_category("spatial")
            >>> len(spatial)
            2
        """
        features = [
            f for f in self._features.values()
            if f.category == category
        ]
        return sorted(features, key=lambda f: f.index)

    def get_feature_names(self, category: Optional[str] = None) -> List[str]:
        """
        Get feature names, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of feature names in index order

        Example:
            >>> reg = FeatureRegistry("summary")
            >>> reg.register("spatial_mean", "spatial")
            >>> reg.register("spectral_centroid", "spectral")
            >>> reg.get_feature_names()
            ['spatial_mean', 'spectral_centroid']
            >>> reg.get_feature_names("spatial")
            ['spatial_mean']
        """
        if category is not None:
            features = self.get_features_by_category(category)
        else:
            features = sorted(self._features.values(), key=lambda f: f.index)

        return [f.name for f in features]

    @property
    def num_features(self) -> int:
        """
        Total number of registered features.

        Returns:
            Feature count
        """
        return len(self._features)

    @property
    def categories(self) -> List[str]:
        """
        List of all categories.

        Returns:
            Category names (unsorted)
        """
        return list(self._categories.keys())

    def get_category_info(self, category: str) -> Optional[CategoryInfo]:
        """
        Get information about a category.

        Args:
            category: Category name

        Returns:
            CategoryInfo if category exists, None otherwise
        """
        return self._categories.get(category)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """
        Export registry as nested dict for HDF5 storage.

        Returns:
            Nested dict with structure: {category: {name: index}}

        Example:
            >>> reg = FeatureRegistry("summary")
            >>> reg.register("spatial_mean", "spatial")
            >>> reg.register("spectral_centroid", "spectral")
            >>> reg.to_dict()
            {
                "spatial": {"spatial_mean": 0},
                "spectral": {"spectral_centroid": 1}
            }
        """
        result: Dict[str, Dict[str, int]] = {}

        for feature in self._features.values():
            if feature.category not in result:
                result[feature.category] = {}
            result[feature.category][feature.name] = feature.index

        return result

    def to_json(self, indent: int = 2) -> str:
        """
        Export registry as JSON string.

        Args:
            indent: JSON indentation level (default: 2)

        Returns:
            JSON string representation

        Example:
            >>> reg = FeatureRegistry("summary")
            >>> reg.register("spatial_mean", "spatial")
            >>> print(reg.to_json())
            {
              "spatial": {
                "spatial_mean": 0
              }
            }
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Dict[str, int]],
        family_name: str = "summary"
    ) -> 'FeatureRegistry':
        """
        Load registry from nested dict.

        Reconstructs the registry with exact same indices as the serialized version.
        This ensures consistency when loading features from HDF5.

        Args:
            data: Nested dict with structure {category: {name: index}}
            family_name: Feature family identifier

        Returns:
            FeatureRegistry instance

        Raises:
            AssertionError: If reconstructed indices don't match original

        Example:
            >>> data = {
            ...     "spatial": {"spatial_mean": 0, "spatial_std": 1},
            ...     "spectral": {"spectral_centroid": 2}
            ... }
            >>> reg = FeatureRegistry.from_dict(data, "summary")
            >>> reg.num_features
            3
        """
        registry = cls(family_name=family_name)

        # Flatten and sort by index to maintain order
        all_features: List[Tuple[str, str, int]] = []
        for category, features in data.items():
            for name, index in features.items():
                all_features.append((name, category, index))

        all_features.sort(key=lambda x: x[2])  # Sort by index

        # Register in order
        for name, category, expected_index in all_features:
            actual_index = registry.register(name, category)
            assert actual_index == expected_index, \
                f"Index mismatch for {name}: expected {expected_index}, got {actual_index}"

        return registry

    @classmethod
    def from_json(cls, json_str: str, family_name: str = "summary") -> 'FeatureRegistry':
        """
        Load registry from JSON string.

        Args:
            json_str: JSON string representation
            family_name: Feature family identifier

        Returns:
            FeatureRegistry instance

        Example:
            >>> json_str = '{"spatial": {"spatial_mean": 0}}'
            >>> reg = FeatureRegistry.from_json(json_str, "summary")
            >>> reg.get_index("spatial_mean")
            0
        """
        data = json.loads(json_str)
        return cls.from_dict(data, family_name)

    def __repr__(self) -> str:
        """String representation of registry."""
        return (
            f"FeatureRegistry(family='{self.family_name}', "
            f"features={self.num_features}, "
            f"categories={len(self._categories)})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"FeatureRegistry: {self.family_name}"]
        lines.append(f"Total features: {self.num_features}")
        lines.append(f"Categories: {len(self._categories)}")

        for category in sorted(self._categories.keys()):
            info = self._categories[category]
            lines.append(f"  - {category}: {info.count} features")

        return "\n".join(lines)
