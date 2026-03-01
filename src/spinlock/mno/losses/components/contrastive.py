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
from torch import Tensor
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


class SoftTokenContrastiveLoss(nn.Module):
    """Soft contrastive loss with GT token-set Jaccard similarity as targets.

    Instead of identity targets (diagonal=1, off-diagonal=0), uses the
    continuous Jaccard similarity between GT token indicator vectors as
    soft targets. Loss = KL(P_pred || Q_target) where both are softmax
    distributions over the similarity matrix rows.

    Architecture:
        features [B, feature_dim] -> MLP -> embeds [B, embed_dim]
        cosine_sim(embeds, embeds+queue) / tau_pred -> P (softmax)
        jaccard(indicators, indicators+queue) / tau_target -> Q (softmax)
        loss = KL(P || Q)
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        tau_pred: float = 0.07,
        tau_target: float = 0.1,
        queue_size: int = 0,
    ):
        """Initialize soft token contrastive loss.

        Args:
            feature_dim: Input dimension from VQ encoder features.
            embed_dim: Contrastive embedding space dimension.
            hidden_dim: Hidden layer size in projector MLP.
            tau_pred: Temperature for predicted similarity softmax.
            tau_target: Temperature for GT Jaccard similarity softmax.
            queue_size: MoCo-style queue size (0 = no queue).
        """
        super().__init__()
        self.tau_pred = tau_pred
        self.tau_target = tau_target
        self.queue_size = queue_size

        # Feature projector: VQ features -> contrastive embedding space
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # MoCo queue: embeddings + indicators
        if queue_size > 0:
            self.register_buffer('embed_queue', torch.zeros(queue_size, embed_dim))
            self.register_buffer(
                'indicator_queue', torch.zeros(queue_size, 1),
            )  # resized at first enqueue
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
            self.register_buffer('queue_len', torch.zeros(1, dtype=torch.long))
            self._indicator_dim_set = False

    @staticmethod
    def _jaccard_matrix(A: Tensor, B: Tensor) -> Tensor:
        """Compute pairwise Jaccard similarity between binary indicator sets.

        Jaccard(a, b) = |a & b| / |a | b| = dot(a, b) / (|a| + |b| - dot(a, b))

        Args:
            A: [M, D] binary indicator vectors.
            B: [N, D] binary indicator vectors.

        Returns:
            [M, N] Jaccard similarity matrix in [0, 1].
        """
        A_f = A.float()
        B_f = B.float()
        dot = A_f @ B_f.T                            # [M, N]
        sizes_a = A_f.sum(dim=1)                      # [M]
        sizes_b = B_f.sum(dim=1)                      # [N]
        union = sizes_a[:, None] + sizes_b[None, :] - dot
        return torch.where(union > 0, dot / union, torch.zeros_like(dot))

    def forward(
        self,
        features: Tensor,
        gt_indicators: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute soft contrastive loss.

        Args:
            features: VQ encoder features [B, feature_dim] (from pyramid encoder).
            gt_indicators: Binary token indicators [B, indicator_dim] (bool/float).

        Returns:
            Dict with: loss, mean_jaccard, rank_correlation.
        """
        B = features.shape[0]
        device = features.device

        # Project features -> embedding space
        embeds = F.normalize(self.projector(features), dim=1)  # [B, embed_dim]

        # Build similarity matrices with optional queue negatives
        if (
            self.queue_size > 0
            and hasattr(self, 'queue_len')
            and self.queue_len.item() > 0
        ):
            Q = int(self.queue_len.item())
            all_embeds = torch.cat(
                [embeds, self.embed_queue[:Q]], dim=0,
            )
            all_indicators = torch.cat(
                [gt_indicators.float(), self.indicator_queue[:Q]], dim=0,
            )
        else:
            all_embeds = embeds
            all_indicators = gt_indicators.float()

        # Predicted similarity: cosine sim / tau_pred
        pred_sim = (embeds @ all_embeds.T) / self.tau_pred  # [B, B+Q]

        # GT similarity: Jaccard / tau_target
        gt_jaccard = self._jaccard_matrix(
            gt_indicators, all_indicators,
        )  # [B, B+Q]
        target_dist = F.softmax(gt_jaccard / self.tau_target, dim=1)

        # KL divergence: P=softmax(pred_sim), Q=target_dist
        pred_log_dist = F.log_softmax(pred_sim, dim=1)
        loss = F.kl_div(pred_log_dist, target_dist, reduction='batchmean')

        # Enqueue current batch
        if self.queue_size > 0:
            self._enqueue(embeds.detach(), gt_indicators.float().detach())

        # Diagnostic metrics (no gradient)
        with torch.no_grad():
            # Mean off-diagonal Jaccard: average behavioral similarity
            # between distinct samples in the batch
            if B > 1:
                off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
                mean_jaccard = gt_jaccard[:B, :B][off_diag_mask].mean()
            else:
                mean_jaccard = torch.tensor(0.0, device=device)
            # Rank correlation: does predicted ordering match GT ordering?
            if B > 1:
                off_diag = ~torch.eye(B, dtype=torch.bool, device=device)
                pred_flat = pred_sim[:B, :B][off_diag]
                gt_flat = gt_jaccard[:B, :B][off_diag]
                if pred_flat.numel() > 1:
                    vx = pred_flat - pred_flat.mean()
                    vy = gt_flat - gt_flat.mean()
                    corr = (vx * vy).sum() / (
                        vx.norm() * vy.norm() + 1e-8
                    )
                else:
                    corr = torch.tensor(0.0, device=device)
            else:
                corr = torch.tensor(0.0, device=device)

        return {
            'loss': loss,
            'mean_jaccard': mean_jaccard,
            'rank_correlation': corr,
        }

    @torch.no_grad()
    def _enqueue(self, embeds: Tensor, indicators: Tensor) -> None:
        """Add batch embeddings and indicators to FIFO queue."""
        B = embeds.shape[0]

        # Resize indicator queue on first use (indicator_dim unknown at init)
        if not self._indicator_dim_set:
            D = indicators.shape[1]
            self.indicator_queue = torch.zeros(
                self.queue_size, D, device=embeds.device,
            )
            self._indicator_dim_set = True

        ptr = int(self.queue_ptr.item())
        if ptr + B <= self.queue_size:
            self.embed_queue[ptr:ptr + B] = embeds
            self.indicator_queue[ptr:ptr + B] = indicators
        else:
            remaining = self.queue_size - ptr
            self.embed_queue[ptr:] = embeds[:remaining]
            self.indicator_queue[ptr:] = indicators[:remaining]
            self.embed_queue[:B - remaining] = embeds[remaining:]
            self.indicator_queue[:B - remaining] = indicators[remaining:]

        self.queue_ptr[0] = (ptr + B) % self.queue_size
        self.queue_len[0] = min(
            int(self.queue_len.item()) + B, self.queue_size,
        )

    def __repr__(self) -> str:
        queue_str = f", queue_size={self.queue_size}" if self.queue_size > 0 else ""
        return (
            f"SoftTokenContrastiveLoss("
            f"tau_pred={self.tau_pred}, tau_target={self.tau_target}"
            f"{queue_str})"
        )
