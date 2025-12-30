"""Categorical projector for hierarchical VQ-VAE.

Projects N category embeddings to N×L independent latent vectors
(N categories × L hierarchical levels).

Supports both uniform levels (all categories use same config) and
per-category level configurations.

Ported from unisim.system.models.categorical_projector (100% generic).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class CategoricalProjector(nn.Module):
    """Projects category embeddings to hierarchical latent vectors.

    Architecture:
        For each of N categories (e.g., cluster_1, cluster_2, ...):
            - Create L projection heads (hierarchical levels)
            - Each head maps group_embedding_dim -> level_latent_dim

        Output: N×L independent latent vectors organized by category order

        Example (5 categories, 3 levels):
            [cluster_1_L0, cluster_1_L1, cluster_1_L2,
             cluster_2_L0, cluster_2_L1, cluster_2_L2,
             cluster_3_L0, cluster_3_L1, cluster_3_L2,
             cluster_4_L0, cluster_4_L1, cluster_4_L2,
             cluster_5_L0, cluster_5_L1, cluster_5_L2]

    This enables:
        - N×L independent VectorQuantizers (one per category-level pair)
        - Categorical tokenization: each category gets its own token sequence
        - Hierarchical refinement: coarse -> medium -> fine per category
    """

    def __init__(
        self,
        group_embedding_dim: int,
        levels: List[Dict] = None,
        category_levels: Dict[str, List[Dict]] = None,
        categories: List[str] = None,
        dropout: float = 0.1,
    ):
        """Initialize categorical projector.

        Args:
            group_embedding_dim: Dimension of each category embedding from feature extractor
            levels: List of level configs (uniform across categories), each with 'latent_dim' key
                    e.g., [{'latent_dim': 64}, {'latent_dim': 32}, {'latent_dim': 16}]
            category_levels: Dict mapping category -> list of levels (overrides 'levels')
                    e.g., {'cluster_1': [{'latent_dim': 96}, ...], 'cluster_2': [...], ...}
            categories: List of category names (only used with 'levels')
                    e.g., ['cluster_1', 'cluster_2'] for dynamic grouping
            dropout: Dropout rate for projection heads
        """
        super().__init__()

        self.group_embedding_dim = group_embedding_dim
        self.dropout = dropout

        # Determine which levels to use and derive categories
        if category_levels is not None:
            self.category_levels = category_levels
            self.levels = None  # Not used when category_levels provided
            # Derive category order from keys (preserves insertion order in Python 3.7+)
            self.categories = list(category_levels.keys())
            # Verify all categories have same number of levels
            level_counts = [len(levels) for levels in category_levels.values()]
            if len(set(level_counts)) > 1:
                raise ValueError(
                    f"All categories must have same number of levels. Got: {level_counts}"
                )
            self.num_levels = level_counts[0]
        elif levels is not None:
            # Use uniform levels with provided categories
            self.levels = levels
            self.category_levels = None
            if categories is None:
                raise ValueError("Must provide 'categories' when using uniform 'levels'")
            self.categories = categories
            self.num_levels = len(levels)
        else:
            raise ValueError("Must provide either 'levels' or 'category_levels'")

        # Create projection heads: N categories × L levels
        self.projection_heads = nn.ModuleDict()

        for category in self.categories:
            category_heads = nn.ModuleList()

            # Get levels for this category
            if self.category_levels is not None:
                cat_levels = self.category_levels[category]
            else:
                cat_levels = self.levels

            for level_idx, level_config in enumerate(cat_levels):
                latent_dim = level_config["latent_dim"]

                # Simple projection: Linear + LayerNorm
                head = nn.Sequential(
                    nn.Linear(group_embedding_dim, latent_dim),
                    nn.Dropout(dropout),
                    nn.LayerNorm(latent_dim),
                )

                category_heads.append(head)

            self.projection_heads[category] = category_heads

    def forward(self, group_embeddings: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Project category embeddings to N×L hierarchical latent vectors.

        Args:
            group_embeddings: Dict of category embeddings from feature extractor
                Keys match self.categories (e.g., 'cluster_1', 'cluster_2', etc.)
                Values are tensors of shape [batch, group_embedding_dim]

        Returns:
            List of N×L latent vectors [batch, latent_dim] organized by category order.

            Example (5 categories, 3 levels = 15 vectors):
                [cluster_1_L0, cluster_1_L1, cluster_1_L2,
                 cluster_2_L0, cluster_2_L1, cluster_2_L2,
                 cluster_3_L0, cluster_3_L1, cluster_3_L2,
                 cluster_4_L0, cluster_4_L1, cluster_4_L2,
                 cluster_5_L0, cluster_5_L1, cluster_5_L2]
        """
        latent_vectors = []

        # Process each category
        for category in self.categories:
            group_emb = group_embeddings[category]  # [batch, group_embedding_dim]

            # Project to each hierarchical level
            for level_idx in range(self.num_levels):
                head = self.projection_heads[category][level_idx]
                latent = head(group_emb)  # [batch, latent_dim]
                latent_vectors.append(latent)

        return latent_vectors

    def get_latent_dims(self) -> List[int]:
        """Get latent dimensions for all projections.

        Returns:
            List of latent dimensions in order:
                [dim_cluster_1_L0, dim_cluster_1_L1, dim_cluster_1_L2,
                 dim_cluster_2_L0, ..., dim_cluster_N_L2]
        """
        dims = []
        for category in self.categories:
            # Get levels for this category
            if self.category_levels is not None:
                cat_levels = self.category_levels[category]
            else:
                cat_levels = self.levels

            for level_config in cat_levels:
                dims.append(level_config["latent_dim"])
        return dims

    def get_category_level_index(self, category: str, level: int) -> int:
        """Get flat index for a category-level pair.

        Args:
            category: Category key (must be in self.categories)
            level: Hierarchical level (0=coarse, 1=medium, 2=fine)

        Returns:
            Flat index in range [0, N×L - 1]
            Example: For 5 categories × 3 levels, range is [0, 14]
        """
        category_idx = self.categories.index(category)
        return category_idx * self.num_levels + level

    def get_category_and_level(self, flat_index: int) -> Tuple[str, int]:
        """Get category and level from flat index.

        Args:
            flat_index: Index in range [0, N×L - 1]

        Returns:
            Tuple of (category, level)
        """
        category_idx = flat_index // self.num_levels
        level = flat_index % self.num_levels
        return self.categories[category_idx], level
