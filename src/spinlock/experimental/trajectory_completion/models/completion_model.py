"""Model for predicting missing tokens given partial observation."""

import torch
import torch.nn as nn
from typing import Dict, List


class TrajectoryCompletionModel(nn.Module):
    """
    Model for predicting missing tokens given partial observation.

    Architecture:
    1. Embed observed tokens (separate embeddings per level)
    2. Hierarchical guidance: coarse (L0) embeddings modulate fine predictions
    3. Transformer encoder processes full sequence with attention masking
    4. Project to target token logits (separate projection per level)

    Hierarchical Structure:
    - L0 (coarse): Captures global trajectory structure
    - L1 (medium): Refines coarse structure
    - L2 (fine): Adds high-frequency details
    - Coarse embeddings provide residual guidance to finer levels
    """

    def __init__(
        self,
        num_tokens_per_level: List[int],  # Codebook sizes per level [L0, L1, L2, ...]
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        hierarchical_guidance_weight: float = 0.1
    ):
        super().__init__()

        self.num_tokens_per_level = num_tokens_per_level
        self.hidden_dim = hidden_dim
        self.hierarchical_guidance_weight = hierarchical_guidance_weight
        self.num_levels = 3  # Assuming 3 levels per category

        # Token embeddings (separate per level)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(num_tokens, hidden_dim, padding_idx=0)
            for num_tokens in num_tokens_per_level
        ])

        # Position embeddings (accommodate up to 1000 token positions)
        max_seq_len = 1000
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projections (separate per level)
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_dim, num_tokens)
            for num_tokens in num_tokens_per_level
        ])

        # Hierarchical guidance projection (coarse → fine)
        self.guidance_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        tokens_observed: torch.Tensor,  # [batch, N×L]
        mask_observed: torch.Tensor,    # [batch, N×L] bool
        mask_target: torch.Tensor       # [batch, N×L] bool
    ) -> Dict[str, torch.Tensor]:
        """
        Predict missing tokens.

        Returns:
            {
                'logits': [batch, N×L, num_tokens_level] - per-token logits
                'predictions': [batch, N×L] - predicted token indices
                'tokens_completed': [batch, N×L] - observed + predicted tokens
            }
        """
        batch_size, seq_len = tokens_observed.shape

        # Embed observed tokens
        # tokens_observed: [batch, N×L] where each position has its own codebook
        embeddings = []
        for i in range(seq_len):
            level_idx = i % self.num_levels  # Assuming 3 levels per category
            token_ids = tokens_observed[:, i]  # [batch]
            emb = self.token_embeddings[level_idx](token_ids)  # [batch, hidden_dim]
            embeddings.append(emb)
        embeddings = torch.stack(embeddings, dim=1)  # [batch, seq_len, hidden_dim]

        # Add position embeddings
        positions = torch.arange(seq_len, device=tokens_observed.device).unsqueeze(0)
        embeddings = embeddings + self.position_embedding(positions)

        # Hierarchical guidance: coarse (L0) embeddings modulate fine predictions
        # Extract coarse (L0) embeddings (every 3rd token assuming 3 levels per category)
        coarse_indices = torch.arange(0, seq_len, self.num_levels, device=tokens_observed.device)
        coarse_embeddings = embeddings[:, coarse_indices, :]  # [batch, N, hidden_dim]

        # Project and broadcast coarse guidance to all levels
        coarse_guidance = self.guidance_proj(coarse_embeddings)  # [batch, N, hidden_dim]
        coarse_guidance = coarse_guidance.repeat_interleave(self.num_levels, dim=1)  # [batch, N×3, hidden_dim]

        # Trim if necessary (in case seq_len is not exactly divisible)
        if coarse_guidance.shape[1] > seq_len:
            coarse_guidance = coarse_guidance[:, :seq_len, :]

        # Add residual coarse influence to all token embeddings
        embeddings = embeddings + self.hierarchical_guidance_weight * coarse_guidance

        # Transformer with bidirectional attention over observed tokens
        # src_key_padding_mask: True = ignore this position
        src_key_padding_mask = ~mask_observed
        encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Project to token logits (list of tensors with varying vocab sizes)
        logits_list = []
        max_vocab_size = max(self.num_tokens_per_level)

        for i in range(seq_len):
            level_idx = i % self.num_levels
            logits_i = self.output_projections[level_idx](encoded[:, i, :])  # [batch, num_tokens_level]

            # Pad to max vocab size for stacking
            if logits_i.shape[1] < max_vocab_size:
                padding = torch.full(
                    (batch_size, max_vocab_size - logits_i.shape[1]),
                    float('-inf'),
                    device=logits_i.device
                )
                logits_i = torch.cat([logits_i, padding], dim=1)

            logits_list.append(logits_i)

        logits = torch.stack(logits_list, dim=1)  # [batch, seq_len, max_vocab_size]

        # Predictions (only consider valid vocab range per position)
        predictions = []
        for i in range(seq_len):
            level_idx = i % self.num_levels
            vocab_size = self.num_tokens_per_level[level_idx]
            pred_i = torch.argmax(logits[:, i, :vocab_size], dim=-1)  # [batch]
            predictions.append(pred_i)
        predictions = torch.stack(predictions, dim=1)  # [batch, seq_len]

        # Combine observed + predicted
        tokens_completed = torch.where(mask_observed, tokens_observed, predictions)

        return {
            'logits': logits,
            'predictions': predictions,
            'tokens_completed': tokens_completed
        }
