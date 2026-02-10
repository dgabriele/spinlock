"""Contrastive similarity for parameter-rollout alignment.

This module projects rollouts and parameters into a shared embedding space
where similar pairs (same params → same rollout) have high similarity and
dissimilar pairs (different params → different rollouts) have low similarity.

Used in contrastive learning (InfoNCE) to enforce that parameter changes
produce distinguishable rollout changes.

Design principles:
- DRY: Reuses RolloutFeatureExtractor for rollout embedding
- Modular: Can be used with InfoNCE, triplet loss, or other contrastive losses
- Symmetric: Both rollouts and params projected to same embedding dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from spinlock.mno.features import RolloutFeatureExtractor


class ContrastiveSimilarity(nn.Module):
    """Project rollouts and parameters into shared embedding space.

    Architecture:
        Rollout branch:
            rollout [B, T, C, H, W]
              ↓
            RolloutFeatureExtractor
              ↓
            features [B, 32]
              ↓
            MLP (32 → hidden → embed_dim)
              ↓
            rollout_embed [B, embed_dim]

        Parameter branch:
            params [B, 14]
              ↓
            MLP (14 → hidden → embed_dim)
              ↓
            param_embed [B, embed_dim]

        Similarity:
            sim = cosine_similarity(rollout_embed, param_embed)

    The embeddings are L2-normalized to use cosine similarity for
    contrastive learning (InfoNCE, triplet loss, etc.).

    Attributes:
        feature_extractor: Reusable RolloutFeatureExtractor (DRY)
        rollout_projector: MLP that projects rollout features to embedding space
        param_projector: MLP that projects parameters to embedding space
        embed_dim: Shared embedding dimensionality (default: 128)
    """

    def __init__(
        self,
        param_dim: int = 14,
        embed_dim: int = 128,
        hidden_dim: int = 256,
    ):
        """Initialize contrastive similarity module.

        Args:
            param_dim: Dimensionality of parameter vector (default: 14)
            embed_dim: Shared embedding dimensionality (default: 128)
            hidden_dim: Hidden layer size for projectors (default: 256)
        """
        super().__init__()
        self.param_dim = param_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Reusable feature extractor (DRY principle)
        self.feature_extractor = RolloutFeatureExtractor()
        feature_dim = self.feature_extractor.feature_dim  # 32

        # Rollout projector: features → embedding
        self.rollout_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Parameter projector: params → embedding
        self.param_projector = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def embed_rollout(self, rollout: torch.Tensor) -> torch.Tensor:
        """Embed rollout into shared space.

        Args:
            rollout: Tensor of shape [B, T, C, H, W] or [B, M, T, C, H, W]

        Returns:
            rollout_embed: L2-normalized embedding [B, embed_dim]
        """
        # Extract statistical features (reuses shared extractor)
        features = self.feature_extractor(rollout)  # [B, 32]

        # Project to embedding space
        embed = self.rollout_projector(features)  # [B, embed_dim]

        # L2 normalize for cosine similarity
        embed = F.normalize(embed, p=2, dim=1)

        return embed

    def embed_params(self, params: torch.Tensor) -> torch.Tensor:
        """Embed parameters into shared space.

        Args:
            params: Parameter tensor [B, param_dim]

        Returns:
            param_embed: L2-normalized embedding [B, embed_dim]
        """
        # Project to embedding space
        embed = self.param_projector(params)  # [B, embed_dim]

        # L2 normalize for cosine similarity
        embed = F.normalize(embed, p=2, dim=1)

        return embed

    def forward(
        self,
        rollout: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed both rollouts and parameters.

        Args:
            rollout: Tensor of shape [B, T, C, H, W] or [B, M, T, C, H, W]
            params: Parameter tensor [B, param_dim]

        Returns:
            rollout_embed: L2-normalized embedding [B, embed_dim]
            param_embed: L2-normalized embedding [B, embed_dim]
        """
        rollout_embed = self.embed_rollout(rollout)
        param_embed = self.embed_params(params)
        return rollout_embed, param_embed

    def compute_similarity(
        self,
        rollout: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between rollout and params.

        Args:
            rollout: Tensor of shape [B, T, C, H, W] or [B, M, T, C, H, W]
            params: Parameter tensor [B, param_dim]

        Returns:
            similarity: Cosine similarity [B] (range: [-1, 1])
        """
        rollout_embed, param_embed = self.forward(rollout, params)

        # Cosine similarity (already normalized)
        similarity = (rollout_embed * param_embed).sum(dim=1)  # [B]

        return similarity

    def compute_similarity_matrix(
        self,
        rollout_embeds: torch.Tensor,
        param_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise similarity matrix.

        Used for contrastive learning (InfoNCE) where we need similarities
        between all rollout-param pairs in a batch.

        Args:
            rollout_embeds: Rollout embeddings [B, embed_dim] (L2-normalized)
            param_embeds: Parameter embeddings [B, embed_dim] (L2-normalized)

        Returns:
            similarity_matrix: [B, B] where entry [i, j] is similarity
                             between rollout i and params j
        """
        # Matrix multiply gives all pairwise cosine similarities
        similarity_matrix = rollout_embeds @ param_embeds.T  # [B, B]

        return similarity_matrix

    def __repr__(self) -> str:
        return (
            f"ContrastiveSimilarity("
            f"param_dim={self.param_dim}, "
            f"embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )
