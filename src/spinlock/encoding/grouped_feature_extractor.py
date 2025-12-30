"""Grouped feature extractor for categorical hierarchical VQ-VAE.

Extracts per-category embeddings WITHOUT fusion - returns dictionary of
independent category representations for categorical tokenization.

Each discovered category gets its own MLP encoder that maps variable-dimension
category features to fixed-dimension embeddings.

Ported from unisim.system.models.grouped_feature_extractor (100% generic).
"""

import torch
import torch.nn as nn
from typing import Dict, List


class GroupMLP(nn.Module):
    """MLP for single category with LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        """Initialize category MLP.

        Args:
            input_dim: Number of features in this category
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (category embedding size)
            dropout: Dropout rate
        """
        super().__init__()

        if input_dim == 0:
            # Empty category - create identity that outputs zeros
            self.net = None
            self.output_dim = output_dim
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),  # Normalize output for stability
            )
            self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Category features [batch, input_dim]

        Returns:
            Category embedding [batch, output_dim]
        """
        if self.net is None:
            # Empty category - return zeros
            return torch.zeros(
                x.size(0), self.output_dim, device=x.device, dtype=x.dtype
            )
        return self.net(x)


class GroupedFeatureExtractor(nn.Module):
    """Feature extractor with parallel processing of categories.

    Unlike typical encoders, this does NOT fuse categories into a single vector.
    Instead, it returns a dictionary of independent category embeddings for
    categorical hierarchical tokenization.

    Architecture:
    1. Split input by discovered categories (cluster_1, cluster_2, ...)
    2. Process each category with dedicated MLP + LayerNorm
    3. Return dictionary of category embeddings (no fusion)

    Handles dynamic feature dimensions:
    - Some categories may have 0 features after filtering
    - Feature indices computed from category assignment
    - Categories with 0 features produce zero embeddings
    """

    def __init__(
        self,
        input_dim: int,
        group_indices: Dict[str, List[int]],
        group_embedding_dim: int = 64,
        group_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize grouped feature extractor.

        Args:
            input_dim: Total input feature dimension
            group_indices: Dict mapping category -> feature indices
                           e.g., {'cluster_1': [0, 5, 12], 'cluster_2': [1, 3], ...}
            group_embedding_dim: Output dimension for each category (fixed)
            group_hidden_dim: Hidden dimension for each category MLP
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.group_embedding_dim = group_embedding_dim
        self.group_indices = group_indices

        # Validate indices
        all_indices = []
        for indices in group_indices.values():
            all_indices.extend(indices)

        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Duplicate indices in group_indices")

        if len(all_indices) > 0 and max(all_indices) >= input_dim:
            raise ValueError(f"Max index {max(all_indices)} >= input_dim {input_dim}")

        # Create category MLPs (all output same dimension for uniformity)
        self.group_mlps = nn.ModuleDict()
        for key, indices in group_indices.items():
            self.group_mlps[key] = GroupMLP(
                input_dim=len(indices),
                hidden_dim=group_hidden_dim,
                output_dim=group_embedding_dim,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with grouped processing.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Dictionary of category embeddings:
            {
                'cluster_1': [batch, group_embedding_dim],
                'cluster_2': [batch, group_embedding_dim],
                ...
            }
        """
        group_embeddings = {}

        for key in self.group_indices.keys():
            indices = self.group_indices[key]

            if len(indices) > 0:
                # Extract category features
                group_features = x[:, indices]

                # Process with category MLP
                group_embeddings[key] = self.group_mlps[key](group_features)
            else:
                # Empty category - create zero embedding
                batch_size = x.size(0)
                group_embeddings[key] = torch.zeros(
                    batch_size,
                    self.group_embedding_dim,
                    device=x.device,
                    dtype=x.dtype,
                )

        return group_embeddings
