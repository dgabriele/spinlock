"""Category assignment strategies for categorical VQ-VAE.

This module provides abstract and concrete implementations for assigning features
to categories in the hierarchical VQ-VAE system. Supports data-driven optimal
groupings via hierarchical clustering and gradient-based refinement.

Design philosophy:
- Clean OOP abstraction (no if/else branching in client code)
- DRY principles (assignment logic encapsulated)
- Extensible (easy to add new strategies)

Assignment methods:
- 'clustering': Hierarchical clustering only (fast, approximate)
- 'gradient': Gumbel-Softmax gradient optimization only
- 'hybrid': Clustering init + gradient refinement (recommended)

Adapted from unisim.system.models.category_assignment.
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
    """Data-driven optimal groupings via clustering and/or gradient optimization.

    Discovers optimal feature groupings through:
    - 'clustering': Hierarchical clustering with correlation distance
    - 'gradient': Gumbel-Softmax gradient-based optimization
    - 'hybrid': Clustering init + gradient refinement (recommended)

    Optimization objectives:
    - Orthogonality: Minimize inter-category correlation (target <0.3)
    - Informativeness: Maximize per-category reconstruction quality
    """

    def __init__(
        self,
        num_categories: Optional[int] = None,
        method: str = "clustering",
        orthogonality_target: float = 0.3,
        min_features_per_category: int = 3,
        random_seed: int = 42,
        max_samples_for_clustering: int = 50000,
        max_clusters: int = 50,
        gradient_epochs: int = 500,
        gradient_lr: float = 0.01,
        subsample_excess_fraction: float = 0.1,
        device: str = "cuda",
    ):
        """Initialize dynamic category assignment.

        Args:
            num_categories: Number of categories (None = auto-determine via silhouette)
            method: Assignment method: 'clustering', 'gradient', or 'hybrid'
            orthogonality_target: Target max correlation between categories
            min_features_per_category: Minimum features per category (prevents empty categories)
            random_seed: Random seed for reproducibility
            max_samples_for_clustering: Maximum samples for clustering
            max_clusters: Maximum clusters to explore for auto-determination
            gradient_epochs: Number of epochs for gradient optimization (default: 500)
            gradient_lr: Learning rate for gradient optimization (default: 0.01)
            subsample_excess_fraction: For datasets > 10K, use 10K + this fraction of excess
            device: Device for gradient optimization ('cuda' or 'cpu'), defaults to 'cuda'
        """
        if method not in ("clustering", "gradient", "hybrid"):
            raise ValueError(f"Unknown method: {method}. Use 'clustering', 'gradient', or 'hybrid'")

        self.num_categories = num_categories
        self.method = method
        self.orthogonality_target = orthogonality_target
        self.min_features_per_category = min_features_per_category
        self.random_seed = random_seed
        self.max_samples_for_clustering = max_samples_for_clustering
        self.max_clusters = max_clusters
        self.gradient_epochs = gradient_epochs
        self.gradient_lr = gradient_lr
        self.subsample_excess_fraction = subsample_excess_fraction
        self.device = device

        # Cached assignments (computed on first call to assign_categories)
        self._assignments = None
        self._category_names = None

    def assign_categories(
        self, feature_names: List[str], features: Optional[np.ndarray] = None
    ) -> Dict[str, List[int]]:
        """Compute optimal groupings from feature data.

        Args:
            feature_names: List of feature names
            features: Required [N_samples, N_features] data for optimization

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

        # Dispatch based on method
        if self.method == "clustering":
            assignments = self._clustering_assignment(features, feature_names)
        elif self.method == "gradient":
            assignments = self._gradient_assignment(features, feature_names)
        elif self.method == "hybrid":
            assignments = self._hybrid_assignment(features, feature_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")

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

    def _gradient_assignment(
        self,
        features: np.ndarray,
        feature_names: List[str],
        init: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, List[int]]:
        """Gumbel-Softmax gradient-based optimization.

        Optimizes soft category assignments via:
        - Orthogonality loss: Mean absolute off-diagonal correlation
        - Informativeness loss: Per-category reconstruction MSE

        Args:
            features: [N_samples, N_features] data
            feature_names: List of feature names
            init: Optional initialization from clustering

        Returns:
            Dict mapping category_name -> list of feature indices
        """
        from .gradient_assignment import optimize_category_assignment

        # Use auto-determined K from init if available
        num_categories = self.num_categories
        if init is not None and num_categories is None:
            num_categories = len(init)

        # Default to reasonable number if still None
        if num_categories is None:
            num_categories = 10

        assignments = optimize_category_assignment(
            features=features,
            num_categories=num_categories,
            init_assignments=init,
            num_epochs=self.gradient_epochs,
            learning_rate=self.gradient_lr,
            orthogonality_target=self.orthogonality_target,
            min_features_per_category=self.min_features_per_category,
            random_seed=self.random_seed,
            subsample_excess_fraction=self.subsample_excess_fraction,
            device=self.device,
        )

        return assignments

    def _hybrid_assignment(
        self, features: np.ndarray, feature_names: List[str]
    ) -> Dict[str, List[int]]:
        """Clustering initialization + gradient refinement (recommended).

        Two-stage process:
        1. Initialize with hierarchical clustering (fast, approximate)
        2. Refine with gradient descent (slow, optimal)

        Best of both worlds: clustering provides good initialization,
        gradient descent finds local optimum.

        Args:
            features: [N_samples, N_features] data
            feature_names: List of feature names

        Returns:
            Dict mapping category_name -> list of feature indices
        """
        # Stage 1: Clustering initialization
        print("  Stage 1: Hierarchical clustering initialization")
        init_assignments = self._clustering_assignment(features, feature_names)
        print(f"    Initialized {len(init_assignments)} categories")

        # Stage 2: Gradient refinement
        print("  Stage 2: Gradient-based refinement (Gumbel-Softmax)")
        refined_assignments = self._gradient_assignment(
            features, feature_names, init=init_assignments
        )

        return refined_assignments
