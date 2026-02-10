"""Contrastive loss component (InfoNCE).

This loss enforces that different parameters produce distinguishable rollouts.
Uses InfoNCE (Information Noise Contrastive Estimation) to maximize similarity
between matched (rollout, params) pairs while minimizing similarity to
mismatched pairs.

For each rollout i in batch:
- Positive: params i (same params that generated rollout i)
- Negatives: params j≠i (different params from other samples)

Goal: Rollout i should be more similar to params i than to any params j≠i

Design:
- Modular: Can be used standalone or composed in ParameterSensitiveLoss
- Scalable: Batch-level contrastive learning (all pairs in batch)
- Temperature-controlled: Adjustable sharpness of similarity distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from spinlock.mno.modules import ContrastiveSimilarity


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for parameter-rollout alignment.

    Architecture:
        rollout, params → ContrastiveSimilarity → embeddings
        similarity_matrix = rollout_embeds @ param_embeds.T  # [B, B]
        loss = CrossEntropy(similarity_matrix / temperature, identity)

    The loss encourages:
    - High similarity on diagonal (rollout i ↔ params i)
    - Low similarity off-diagonal (rollout i ↔ params j, j≠i)

    Attributes:
        similarity: Module that embeds rollouts and params
        temperature: Softmax temperature (lower = sharper distribution)
    """

    def __init__(
        self,
        param_dim: int = 14,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.1,
    ):
        """Initialize contrastive loss.

        Args:
            param_dim: Dimensionality of parameter vector (default: 14)
            embed_dim: Shared embedding dimensionality (default: 128)
            hidden_dim: Hidden layer size for projectors (default: 256)
            temperature: Softmax temperature (default: 0.1, range: [0.01, 1.0])
        """
        super().__init__()
        self.temperature = temperature

        # Similarity module (embeds rollouts and params)
        self.similarity = ContrastiveSimilarity(
            param_dim=param_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        rollout: torch.Tensor,
        params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE contrastive loss.

        Args:
            rollout: Predicted or generated rollout [B, T, C, H, W]
            params: Parameters that generated the rollout [B, param_dim]

        Returns:
            Dictionary containing:
            - loss: InfoNCE contrastive loss
            - accuracy: Fraction of samples where correct param is top-1 (detached)
            - mean_positive_sim: Average similarity for matched pairs (detached)
            - mean_negative_sim: Average similarity for mismatched pairs (detached)
        """
        B = rollout.shape[0]
        device = rollout.device

        # Embed rollouts and parameters into shared space
        rollout_embeds, param_embeds = self.similarity(rollout, params)  # [B, D], [B, D]

        # Compute similarity matrix [B, B]
        # Entry [i, j] = similarity between rollout i and params j
        logits = self.similarity.compute_similarity_matrix(
            rollout_embeds, param_embeds
        ) / self.temperature  # [B, B]

        # Positive pairs are on the diagonal (rollout i ↔ params i)
        labels = torch.arange(B, device=device)  # [0, 1, 2, ..., B-1]

        # InfoNCE loss (cross-entropy with diagonal as targets)
        loss = F.cross_entropy(logits, labels)

        # Compute validation metrics
        with torch.no_grad():
            # Accuracy: fraction where correct param is top-1
            pred_indices = logits.argmax(dim=1)  # [B]
            accuracy = (pred_indices == labels).float().mean()

            # Similarity statistics
            diagonal_mask = torch.eye(B, device=device, dtype=torch.bool)
            positive_sims = logits[diagonal_mask]  # [B]
            negative_sims = logits[~diagonal_mask]  # [B*(B-1)]

            mean_positive_sim = positive_sims.mean()
            mean_negative_sim = negative_sims.mean()

        return {
            'loss': loss,
            'accuracy': accuracy,
            'mean_positive_sim': mean_positive_sim,
            'mean_negative_sim': mean_negative_sim,
        }

    def __repr__(self) -> str:
        return f"ContrastiveLoss(temperature={self.temperature})"
