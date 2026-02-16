"""Token-space contrastive loss (InfoNCE on VQ soft-probability embeddings).

Ensures different (theta, IC) produce different token distributions.
Complements roundtrip loss (accuracy) with a diversity signal:
  - Roundtrip: "match GT tokens for this input"
  - Token contrastive: "different inputs produce different tokens"

Architecture:
    soft_logits {key: [B, K_q]} per quantizer
        -> softmax(logits / tau_token) per quantizer
        -> concat all -> [B, sum(K_q)]
        -> token_projector MLP -> [B, embed_dim] L2-norm

    params [B, param_dim]
        -> param_projector MLP -> [B, embed_dim] L2-norm

    logits = token_embed @ param_embed.T / tau_nce
    loss = CrossEntropy(logits, arange(B))

Two temperatures serve different roles:
    tau_token: Sharpness of per-quantizer softmax (lower = more peaked)
    tau_nce:   Standard InfoNCE temperature (contrastive discrimination)

Gradient flow:
    MNO -> trajectory -> features -> VQ encoder (frozen) -> soft_logits
        -> softmax -> concat -> token_projector (learnable) -> InfoNCE
    All differentiable. Frozen VQ encoder passes gradient via distances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TokenContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss in token probability space.

    Ensures different (theta, IC) produce different token distributions.

    The token_projector MLP is lazily initialized on first forward() call
    because the total soft-probability dimension sum(K_q) depends on the
    VQ checkpoint and is unknown at __init__ time.

    Args:
        param_dim: Dimensionality of parameter vector
        embed_dim: Shared embedding dimensionality for contrastive space
        hidden_dim: Hidden layer size for projector MLPs
        nce_temperature: InfoNCE temperature (contrastive sharpness)
        token_temperature: Per-quantizer softmax temperature
        queue_size: MoCo-style FIFO queue size (0 = no queue)
    """

    def __init__(
        self,
        param_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        nce_temperature: float = 0.1,
        token_temperature: float = 0.5,
        queue_size: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.nce_temperature = nce_temperature
        self.token_temperature = token_temperature
        self.queue_size = queue_size

        # Param projector: param_dim -> embed_dim (separate from ContrastiveLoss)
        self.param_projector = nn.Sequential(
            nn.LayerNorm(param_dim),
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Token projector: lazily initialized (sum(K_q) unknown at init)
        self.token_projector: nn.Module | None = None
        self._total_prob_dim: int | None = None

        # MoCo-style FIFO queue for small-batch training
        if queue_size > 0:
            self.register_buffer('token_queue', torch.zeros(queue_size, embed_dim))
            self.register_buffer('param_queue', torch.zeros(queue_size, embed_dim))
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
            self.register_buffer('queue_len', torch.zeros(1, dtype=torch.long))

    def _ensure_projector(self, total_prob_dim: int, device: torch.device) -> None:
        """Lazy init token_projector on first forward() call."""
        if self.token_projector is not None:
            return
        self._total_prob_dim = total_prob_dim
        self.token_projector = nn.Sequential(
            nn.LayerNorm(total_prob_dim),
            nn.Linear(total_prob_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        ).to(device)

    def _build_token_embedding(
        self, soft_logits: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """soft_logits -> softmax -> concat -> MLP -> L2-norm -> [B, embed_dim]"""
        prob_vectors = []
        for key in sorted(soft_logits.keys()):
            logits = soft_logits[key]  # [B, K_q]
            probs = F.softmax(logits / self.token_temperature, dim=-1)
            prob_vectors.append(probs)

        # Concat all quantizer probabilities: [B, sum(K_q)]
        prob_concat = torch.cat(prob_vectors, dim=-1)

        # Lazy init projector
        self._ensure_projector(prob_concat.shape[-1], prob_concat.device)

        # Project and L2-normalize
        token_embed = self.token_projector(prob_concat)
        token_embed = F.normalize(token_embed, p=2, dim=-1)
        return token_embed, prob_concat

    @torch.no_grad()
    def _enqueue(
        self, token_embeds: torch.Tensor, param_embeds: torch.Tensor,
    ) -> None:
        """Add current batch embeddings to FIFO queue (detached)."""
        B = token_embeds.shape[0]
        ptr = int(self.queue_ptr.item())

        if ptr + B <= self.queue_size:
            self.token_queue[ptr:ptr + B] = token_embeds
            self.param_queue[ptr:ptr + B] = param_embeds
        else:
            remaining = self.queue_size - ptr
            self.token_queue[ptr:] = token_embeds[:remaining]
            self.param_queue[ptr:] = param_embeds[:remaining]
            self.token_queue[:B - remaining] = token_embeds[remaining:]
            self.param_queue[:B - remaining] = param_embeds[remaining:]

        self.queue_ptr[0] = (ptr + B) % self.queue_size
        self.queue_len[0] = min(int(self.queue_len.item()) + B, self.queue_size)

    def forward(
        self,
        soft_logits: Dict[str, torch.Tensor],
        params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE loss in token probability space.

        Args:
            soft_logits: {quantizer_key: [B, K_q]} differentiable logits
            params: [B, param_dim] QBM/physics parameters

        Returns:
            Dict with: loss, accuracy, mean_positive_sim, mean_negative_sim,
            token_prob_entropy
        """
        B = params.shape[0]
        device = params.device

        # Build token embedding from soft probability vectors
        token_embed, prob_concat = self._build_token_embedding(soft_logits)

        # Build param embedding
        param_embed = self.param_projector(params)
        param_embed = F.normalize(param_embed, p=2, dim=-1)

        # Compute similarity logits (optionally with queue negatives)
        if self.queue_size > 0 and hasattr(self, 'queue_len') and self.queue_len.item() > 0:
            Q = int(self.queue_len.item())
            all_params = torch.cat([param_embed, self.param_queue[:Q]], dim=0)
            logits = (token_embed @ all_params.T) / self.nce_temperature  # [B, B+Q]
        else:
            logits = (token_embed @ param_embed.T) / self.nce_temperature  # [B, B]

        # Positive pairs are diagonal (first B columns)
        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(logits, labels)

        # Enqueue current batch for future negatives
        if self.queue_size > 0:
            self._enqueue(token_embed.detach(), param_embed.detach())

        # Monitoring metrics (detached)
        with torch.no_grad():
            pred_indices = logits.argmax(dim=1)
            accuracy = (pred_indices == labels).float().mean()

            positive_sims = logits.diagonal()
            pos_mask = torch.eye(B, logits.shape[1], dtype=torch.bool, device=device)
            negative_sims = logits[~pos_mask]

            mean_positive_sim = positive_sims.mean()
            mean_negative_sim = (
                negative_sims.mean()
                if negative_sims.numel() > 0
                else torch.tensor(0.0, device=device)
            )

            # Average entropy of probability vectors (diversity indicator)
            # High entropy = soft/uniform probs; low = peaked/committed
            token_prob_entropy = -(
                prob_concat * (prob_concat + 1e-10).log()
            ).sum(dim=-1).mean()

        return {
            'loss': loss,
            'accuracy': accuracy,
            'mean_positive_sim': mean_positive_sim,
            'mean_negative_sim': mean_negative_sim,
            'token_prob_entropy': token_prob_entropy,
        }

    def __repr__(self) -> str:
        queue_str = f", queue={self.queue_size}" if self.queue_size > 0 else ""
        proj_str = f", prob_dim={self._total_prob_dim}" if self._total_prob_dim else ", proj=lazy"
        return (
            f"TokenContrastiveLoss("
            f"tau_nce={self.nce_temperature}, "
            f"tau_token={self.token_temperature}"
            f"{queue_str}{proj_str})"
        )
