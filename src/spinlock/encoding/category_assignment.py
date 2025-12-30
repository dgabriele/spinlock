"""Category assignment strategies for categorical VQ-VAE.

This module provides abstract and concrete implementations for assigning features
to categories in the hierarchical VQ-VAE system. Supports data-driven optimal
groupings via hierarchical clustering.

Design philosophy:
- Clean OOP abstraction (no if/else branching in client code)
- DRY principles (assignment logic encapsulated)
- Extensible (easy to add new strategies)

Adapted from unisim.system.models.category_assignment (manual assignment removed,
gradient optimization can be added later if needed).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np


class CategoryAssignment(ABC):
    """Abstract base class for feature-to-category assignment strategies.

    All assignment strategies must implement this interface, enabling polymorphic
    usage in training code without if/else branching.
    """

    @abstractmethod
    def assign_categories(
        self, feature_names: List[str], features: Optional[np.ndarray] = None
    ) -> Dict[str, List[int]]:
        """Assign features to categories.

        Args:
            feature_names: List of feature names (length N_features)
            features: Optional feature data [N_samples, N_features] for data-driven assignment

        Returns:
            Dict mapping category_name -> list of feature indices
            Example: {'cluster_1': [0, 2, 5], 'cluster_2': [1, 3, 4, 6]}
        """
        pass

    @abstractmethod
    def get_category_names(self) -> List[str]:
        """Get ordered list of category names.

        Returns:
            List of category names in consistent order
        """
        pass

    @abstractmethod
    def get_num_categories(self) -> int:
        """Get number of categories.

        Returns:
            Total number of categories
        """
        pass


class DynamicCategoryAssignment(CategoryAssignment):
    """Data-driven optimal groupings via hierarchical clustering.

    Discovers optimal feature groupings through hierarchical clustering:
    - Uses correlation distance (1 - |corr|) with Ward linkage
    - Auto-determines optimal K via silhouette score if num_categories=None
    - Validates orthogonality (target inter-category correlation <0.3)
    """

    def __init__(
        self,
        num_categories: Optional[int] = None,
        orthogonality_target: float = 0.3,
        min_features_per_category: int = 3,
        random_seed: int = 42,
        max_samples_for_clustering: int = 50000,
        max_clusters: int = 50,
    ):
        """Initialize dynamic category assignment.

        Args:
            num_categories: Number of categories (None = auto-determine via silhouette)
            orthogonality_target: Target max correlation between categories
            min_features_per_category: Minimum features per category (prevents empty categories)
            random_seed: Random seed for reproducibility
            max_samples_for_clustering: Maximum samples for clustering
            max_clusters: Maximum clusters to explore for auto-determination
        """
        self.num_categories = num_categories
        self.orthogonality_target = orthogonality_target
        self.min_features_per_category = min_features_per_category
        self.random_seed = random_seed
        self.max_samples_for_clustering = max_samples_for_clustering
        self.max_clusters = max_clusters

        # Cached assignments (computed on first call to assign_categories)
        self._assignments = None
        self._category_names = None

    def assign_categories(
        self, feature_names: List[str], features: Optional[np.ndarray] = None
    ) -> Dict[str, List[int]]:
        """Compute optimal groupings from feature data via clustering.

        Args:
            feature_names: List of feature names
            features: Required [N_samples, N_features] data for clustering

        Returns:
            Dict mapping category_name -> list of feature indices

        Raises:
            ValueError: If features is None (dynamic assignment requires data)
        """
        if features is None:
            raise ValueError("DynamicCategoryAssignment requires feature data")

        # Use cached assignments if available
        if self._assignments is not None:
            return self._assignments

        # Compute clustering-based assignments
        assignments = self._clustering_assignment(features, feature_names)

        # Cache results
        self._assignments = assignments
        self._category_names = list(assignments.keys())

        return assignments

    def get_category_names(self) -> List[str]:
        """Get ordered list of category names.

        Returns:
            List of category names (sorted)

        Raises:
            ValueError: If assign_categories() hasn't been called yet
        """
        if self._category_names is None:
            raise ValueError("Must call assign_categories() first")
        return sorted(self._category_names)

    def get_num_categories(self) -> int:
        """Get number of categories.

        Returns:
            Total number of categories

        Raises:
            ValueError: If assign_categories() hasn't been called yet
        """
        if self._category_names is None:
            raise ValueError("Must call assign_categories() first")
        return len(self._category_names)

    def _clustering_assignment(
        self, features: np.ndarray, feature_names: List[str]
    ) -> Dict[str, List[int]]:
        """Hierarchical clustering assignment.

        Uses correlation distance (1 - |corr|) with Ward linkage.
        Auto-determines optimal K via silhouette score if num_categories=None.

        Args:
            features: [N_samples, N_features] data
            feature_names: List of feature names

        Returns:
            Dict mapping category_name -> list of feature indices
        """
        from .clustering_assignment import hierarchical_clustering_assignment

        assignments = hierarchical_clustering_assignment(
            features=features,
            feature_names=feature_names,
            num_clusters=self.num_categories,
            min_features_per_cluster=self.min_features_per_category,
            orthogonality_target=self.orthogonality_target,
            random_seed=self.random_seed,
            max_samples_for_clustering=self.max_samples_for_clustering,
            max_clusters=self.max_clusters,
        )

        return assignments
