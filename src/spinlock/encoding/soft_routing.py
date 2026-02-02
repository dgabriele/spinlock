"""Soft routing for learnable category assignment."""

import torch
import torch.nn as nn
from typing import List


class SoftGroupedFeatureExtractor(nn.Module):
    """Soft-routed feature extraction using weighted combinations."""

    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        embedding_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Per-category MLPs
        self.category_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            for _ in range(num_categories)
        ])

    def forward(self, x: torch.Tensor, soft_assignments: torch.Tensor) -> torch.Tensor:
        """Soft routing with weighted feature extraction.

        Args:
            x: [B, D] input features
            soft_assignments: [D, K] soft assignment probabilities

        Returns:
            category_embeddings: [B, K, E] category embeddings
        """
        B = x.shape[0]
        K = self.num_categories
        E = self.embedding_dim

        category_embeddings = torch.zeros(B, K, E, device=x.device)

        # For each category, compute weighted features
        for k in range(K):
            # Get assignment weights for this category
            weights = soft_assignments[:, k]  # [D]

            # Weight the input features
            weighted_x = x * weights.unsqueeze(0)  # [B, D]

            # Pass through category-specific MLP
            category_embeddings[:, k] = self.category_mlps[k](weighted_x)

        return category_embeddings
