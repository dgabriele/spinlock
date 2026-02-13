"""Token embedding extraction from the DenoisingNetwork.

Uses the denoiser's transformer hidden states as the "token language"
embedding — the internal representation the model has learned about
token co-occurrence patterns. These embeddings are projected into a
shared space for contrastive alignment with English descriptions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class AttentionPooling(nn.Module):
    """Pool a variable-length sequence into a fixed-size vector via learned attention.

    A single learnable query vector attends to all positions in the sequence.
    More expressive than mean-pooling because different token positions
    (e.g., L0 coarse codes vs L2 fine codes) can contribute differently
    to the summary embedding.

    Args:
        hidden_dim: Dimension of input features and query vector.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over sequence positions.

        Args:
            x: [B, N, D] — sequence of hidden states

        Returns:
            [B, D] — pooled vector
        """
        # query: [1, 1, D] broadcast to [B, 1, D]
        attn_weights = torch.bmm(
            self.query.expand(x.size(0), -1, -1), x.transpose(1, 2)
        )  # [B, 1, N]
        attn_weights = F.softmax(attn_weights * self.scale, dim=-1)
        pooled = torch.bmm(attn_weights, x)  # [B, 1, D]
        return pooled.squeeze(1)  # [B, D]


class TokenEmbedder(nn.Module):
    """Extract token-sequence embeddings from the DenoisingNetwork.

    Uses the denoiser's transformer hidden states (the internal
    representation learned about token co-occurrence) as the
    "token language" embedding.

    Architecture:
        tokens_dict → DenoisingNetwork(return_hidden=True) → [B, N, H]
                    → AttentionPooling → [B, H]
                    → Linear(H, D_shared) → [B, D_shared]

    The denoiser is always frozen (no gradients flow through it).
    Only the attention pooling query and projection layer are trainable.

    Args:
        denoiser: Frozen DenoisingNetwork instance.
        shared_dim: Output embedding dimension for contrastive alignment.
    """

    def __init__(self, denoiser: nn.Module, shared_dim: int = 256):
        super().__init__()
        self.denoiser = denoiser
        self.hidden_dim = denoiser.hidden_dim
        self.num_tokens = denoiser.num_tokens
        self.pool = AttentionPooling(self.hidden_dim)
        self.project = nn.Linear(self.hidden_dim, shared_dim)

    @torch.no_grad()
    def extract_hidden(self, tokens_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get denoiser hidden states at t=0 (clean tokens).

        At t=0 there's no noise — the transformer sees the actual token
        pattern and its hidden states encode the learned co-occurrence
        structure.

        Args:
            tokens_dict: Dict mapping key → token indices [B]

        Returns:
            [B, N_total, hidden_dim] — transformer encoded hidden states
        """
        B = next(iter(tokens_dict.values())).shape[0]
        device = next(self.denoiser.parameters()).device
        t_zero = torch.zeros(B, dtype=torch.long, device=device)
        _, hidden = self.denoiser(tokens_dict, t_zero, return_hidden=True)
        return hidden

    def forward(self, tokens_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract, pool, and project token embeddings.

        Args:
            tokens_dict: Dict mapping key → token indices [B]

        Returns:
            [B, D_shared] — projected embedding for contrastive alignment
        """
        hidden = self.extract_hidden(tokens_dict)  # [B, N, H]
        pooled = self.pool(hidden)                  # [B, H]
        return self.project(pooled)                 # [B, D_shared]
