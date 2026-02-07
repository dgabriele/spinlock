"""Simple hierarchical projector for tokens.

Projects category features to hierarchical latent representations.
"""

import torch
import torch.nn as nn
from typing import List, Dict


class HierarchicalProjector(nn.Module):
    """Simple hierarchical projector for single category.

    Projects a single category's features through L hierarchical levels,
    producing L latent vectors of potentially different dimensions.

    Args:
        input_dim: Input feature dimension for this category
        levels: List of dicts with 'latent_dim' keys
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        levels: List[Dict],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_levels = len(levels)
        self.dropout = dropout

        # Create projection heads for each level
        self.heads = nn.ModuleList()

        for level_config in levels:
            latent_dim = level_config["latent_dim"]

            head = nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(latent_dim),
            )

            self.heads.append(head)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Project features to hierarchical latents.

        Args:
            x: Category features [B, input_dim]

        Returns:
            List of L latent vectors [B, latent_dim_l]
        """
        latents = []
        for head in self.heads:
            latent = head(x)  # [B, latent_dim]
            latents.append(latent)

        return latents
