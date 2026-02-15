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
- Optional MoCo-style queue for small-batch training (queue_size > 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

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
        queue_size: int = 0,
        feature_dim: Optional[int] = None,
    ):
        """Initialize contrastive loss.

        Args:
            param_dim: Dimensionality of parameter vector (default: 14)
            embed_dim: Shared embedding dimensionality (default: 128)
            hidden_dim: Hidden layer size for projectors (default: 256)
            temperature: Softmax temperature (default: 0.1, range: [0.01, 1.0])
            queue_size: MoCo-style memory bank size (default: 0 = no queue).
                        When > 0, stores recent embeddings as additional negatives.
                        Essential for small-batch contrastive learning (B=2-4).
            feature_dim: When provided, enables forward_from_features() for
                pre-computed VQ tokenizer features. Creates a separate projector
                (feature_dim → embed_dim) in ContrastiveSimilarity.
        """
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size

        # Similarity module (embeds rollouts and params)
        self.similarity = ContrastiveSimilarity(
            param_dim=param_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
        )

        # MoCo-style FIFO queue for small-batch training
        if queue_size > 0:
            self.register_buffer('rollout_queue', torch.zeros(queue_size, embed_dim))
            self.register_buffer('param_queue', torch.zeros(queue_size, embed_dim))
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
            self.register_buffer('queue_len', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, rollout_embeds: torch.Tensor, param_embeds: torch.Tensor):
        """Add current batch embeddings to FIFO queue (detached, no gradients)."""
        B = rollout_embeds.shape[0]
        ptr = int(self.queue_ptr.item())

        if ptr + B <= self.queue_size:
            self.rollout_queue[ptr:ptr + B] = rollout_embeds
            self.param_queue[ptr:ptr + B] = param_embeds
        else:
            # Wrap around: fill end of buffer, then start
            remaining = self.queue_size - ptr
            self.rollout_queue[ptr:] = rollout_embeds[:remaining]
            self.param_queue[ptr:] = param_embeds[:remaining]
            self.rollout_queue[:B - remaining] = rollout_embeds[remaining:]
            self.param_queue[:B - remaining] = param_embeds[remaining:]

        self.queue_ptr[0] = (ptr + B) % self.queue_size
        self.queue_len[0] = min(int(self.queue_len.item()) + B, self.queue_size)

    def forward(
        self,
        rollout: torch.Tensor,
        params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE contrastive loss.

        When queue_size > 0, recent embeddings from past batches are used as
        additional negatives (MoCo pattern). This is essential for small-batch
        training where B=2-4 provides insufficient negatives for InfoNCE.

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

        # Build logits matrix — optionally include queue as extra negatives
        if self.queue_size > 0 and hasattr(self, 'queue_len') and self.queue_len.item() > 0:
            Q = int(self.queue_len.item())
            # Queue entries are detached buffers — no gradient flows through them
            all_params = torch.cat([param_embeds, self.param_queue[:Q]], dim=0)  # [B+Q, D]
            logits = self.similarity.compute_similarity_matrix(
                rollout_embeds, all_params
            ) / self.temperature  # [B, B+Q]
        else:
            logits = self.similarity.compute_similarity_matrix(
                rollout_embeds, param_embeds
            ) / self.temperature  # [B, B]

        # Positive pairs are always in the first B columns (diagonal)
        labels = torch.arange(B, device=device)  # [0, 1, 2, ..., B-1]

        # InfoNCE loss (cross-entropy with diagonal as targets)
        loss = F.cross_entropy(logits, labels)

        # Enqueue current batch embeddings for future batches
        if self.queue_size > 0:
            self._enqueue(rollout_embeds.detach(), param_embeds.detach())

        # Compute validation metrics
        with torch.no_grad():
            pred_indices = logits.argmax(dim=1)  # [B]
            accuracy = (pred_indices == labels).float().mean()

            # Positive sims: diagonal of first B columns
            positive_sims = torch.stack([logits[i, i] for i in range(B)])
            # Negative sims: everything else
            pos_mask = torch.zeros_like(logits, dtype=torch.bool)
            for i in range(B):
                pos_mask[i, i] = True
            negative_sims = logits[~pos_mask]

            mean_positive_sim = positive_sims.mean()
            mean_negative_sim = negative_sims.mean() if negative_sims.numel() > 0 else torch.tensor(0.0, device=device)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'mean_positive_sim': mean_positive_sim,
            'mean_negative_sim': mean_negative_sim,
        }

    def forward_from_features(
        self,
        features: torch.Tensor,
        params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """InfoNCE on pre-computed features (e.g., VQ tokenizer temporal features).

        Uses the feature_projector (separate from rollout_projector) to embed
        pre-computed features into the shared contrastive space. This measures
        distinguishability in the same feature space the tokenizer uses.

        If features are 3D [B, T, D], aggregates over T (mean) → [B, D].

        Args:
            features: Pre-computed temporal features [B, T, D_clean] or [B, D_clean]
            params: Parameters that generated the rollout [B, param_dim]

        Returns:
            Dictionary with loss, accuracy, mean_positive_sim, mean_negative_sim
        """
        B = features.shape[0]
        device = features.device

        # Aggregate temporal dimension if present
        if features.ndim == 3:
            features = features.mean(dim=1)  # [B, T, D] → [B, D]

        # Embed features and params
        feature_embeds = self.similarity.embed_features(features)  # [B, embed_dim]
        param_embeds = self.similarity.embed_params(params)        # [B, embed_dim]

        # Build logits — optionally include queue negatives
        if self.queue_size > 0 and hasattr(self, 'queue_len') and self.queue_len.item() > 0:
            Q = int(self.queue_len.item())
            all_params = torch.cat([param_embeds, self.param_queue[:Q]], dim=0)
            logits = self.similarity.compute_similarity_matrix(
                feature_embeds, all_params
            ) / self.temperature
        else:
            logits = self.similarity.compute_similarity_matrix(
                feature_embeds, param_embeds
            ) / self.temperature

        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(logits, labels)

        # Enqueue for future batches
        if self.queue_size > 0:
            self._enqueue(feature_embeds.detach(), param_embeds.detach())

        # Metrics
        with torch.no_grad():
            pred_indices = logits.argmax(dim=1)
            accuracy = (pred_indices == labels).float().mean()

            positive_sims = logits.diagonal()
            pos_mask = torch.eye(B, logits.shape[1], dtype=torch.bool, device=device)
            negative_sims = logits[~pos_mask]

            mean_positive_sim = positive_sims.mean()
            mean_negative_sim = negative_sims.mean() if negative_sims.numel() > 0 else torch.tensor(0.0, device=device)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'mean_positive_sim': mean_positive_sim,
            'mean_negative_sim': mean_negative_sim,
        }

    def __repr__(self) -> str:
        queue_str = f", queue_size={self.queue_size}" if self.queue_size > 0 else ""
        return f"ContrastiveLoss(temperature={self.temperature}{queue_str})"
