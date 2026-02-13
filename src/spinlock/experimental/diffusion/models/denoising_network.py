"""Denoising network for discrete diffusion on hierarchical dict tokens.

Transformer-based architecture that:
- Handles Dict[str, Tensor] format with variable vocab sizes
- Uses flatten-process-unflatten pattern for transformer
- Implements hierarchical guidance (L0 coarse → all levels)
- Produces per-category-level output heads for predictions
"""

import logging
import math
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SinusoidalTimeEmbedding(nn.Module):
    """DDPM-style sinusoidal time embeddings.

    Maps timestep t to continuous embedding vector using
    sinusoidal positional encoding.

    Args:
        embedding_dim: Dimension of time embedding
        max_period: Maximum period for sinusoids (default: 10000)
    """

    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings for timesteps.

        Args:
            timesteps: [B] integer timesteps

        Returns:
            Time embeddings [B, embedding_dim]
        """
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, embedding_dim]
        return emb


class DenoisingNetwork(nn.Module):
    """Transformer-based denoising network for hierarchical dict tokens.

    Architecture:
        1. Embed tokens: Per-category-level embeddings with variable vocab sizes
        2. Flatten dict: Convert to sequence [B, N_total, hidden_dim]
        3. Add time embedding: Broadcast sinusoidal t embedding
        4. Add positional encoding: Learnable position embeddings
        5. Hierarchical guidance: Extract L0 embeddings, add as residual
        6. Transformer encoder: Multi-head self-attention with masking
        7. Output heads: Per-category-level projections to vocab logits

    Args:
        vocab_sizes: Dict mapping "family_category_Ll" → vocab_size
        category_level_info: Dict mapping key → {family, category, level}
        hidden_dim: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_hierarchical_guidance: Enable L0 → all levels guidance

    Example:
        >>> vocab_sizes = {"temporal_group_1_L0": 28, "temporal_group_1_L1": 14}
        >>> denoiser = DenoisingNetwork(vocab_sizes, category_level_info)
        >>> logits_dict = denoiser(noisy_tokens_dict, t=torch.tensor([25]))
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        category_level_info: Dict[str, Dict[str, any]],
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_hierarchical_guidance: bool = True,
        hierarchical_guidance_weight: float = 0.1,
        guidance_mode: str = "global",
        transition_type: str = "uniform",
    ):
        super().__init__()

        self.vocab_sizes = vocab_sizes
        self.category_level_info = category_level_info
        self.hidden_dim = hidden_dim
        self.use_hierarchical_guidance = use_hierarchical_guidance
        self.hierarchical_guidance_weight = hierarchical_guidance_weight
        self.guidance_mode = guidance_mode

        # Sort keys for deterministic ordering
        self.sorted_keys = sorted(vocab_sizes.keys())
        self.num_tokens = len(self.sorted_keys)

        # Parse level info for each key
        self.key_levels = {}
        for key in self.sorted_keys:
            if key in category_level_info:
                self.key_levels[key] = category_level_info[key]['level']
            else:
                # Parse from key string (e.g., "temporal_group_1_L0" → 0)
                level = int(key.split('_L')[-1])
                self.key_levels[key] = level

        # Determine effective vocab size for embeddings (absorbing adds mask token)
        self.effective_vocab_sizes = {}
        for key, v in vocab_sizes.items():
            self.effective_vocab_sizes[key] = v + 1 if transition_type == "absorbing" else v

        # Token embeddings (per-category-level with variable vocab sizes)
        self.token_embeddings = nn.ModuleDict({
            key: nn.Embedding(self.effective_vocab_sizes[key], hidden_dim)
            for key in vocab_sizes
        })

        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Position embeddings (learnable)
        self.position_embedding = nn.Embedding(self.num_tokens, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # Disable nested tensor (incompatible with norm_first=True)
        )

        # Hierarchical guidance projection
        if use_hierarchical_guidance:
            self.guidance_proj = nn.Linear(hidden_dim, hidden_dim)
            # Build L0 indices for global mode
            self._l0_indices = [
                i for i, key in enumerate(self.sorted_keys)
                if self.key_levels[key] == 0
            ]
            # Build parent mask for per-category mode
            if guidance_mode == "per_category":
                self._build_hierarchy_map()

        # Output projections predict clean tokens only (no mask token in output)
        self.output_heads = nn.ModuleDict({
            key: nn.Linear(hidden_dim, vocab_size)
            for key, vocab_size in vocab_sizes.items()
        })

        logger.info(
            f"DenoisingNetwork initialized: "
            f"hidden_dim={hidden_dim}, layers={num_layers}, heads={num_heads}, "
            f"num_tokens={self.num_tokens}, hierarchical_guidance={use_hierarchical_guidance}, "
            f"guidance_mode={guidance_mode}"
        )

    def _build_hierarchy_map(self):
        """Build [N, N] parent mask for per-category hierarchical guidance.

        Each position attends to the L0 parent(s) of its own (family, category)
        group. The mask is row-normalized so guidance is a weighted average.
        """
        N = self.num_tokens
        parent_mask = torch.zeros(N, N)

        # Group positions by (family, category)
        cat_groups = defaultdict(list)
        for idx, key in enumerate(self.sorted_keys):
            if key in self.category_level_info:
                info = self.category_level_info[key]
                cat_groups[(info['family'], info['category'])].append(
                    (idx, info['level'])
                )
            else:
                # Fallback: parse from key name
                parts = key.rsplit('_L', 1)
                level = int(parts[1])
                family_cat = parts[0]
                cat_groups[family_cat].append((idx, level))

        # For each group, connect all positions to their L0 parent(s)
        for positions in cat_groups.values():
            l0_indices = [idx for idx, level in positions if level == 0]
            if l0_indices:
                for idx, _ in positions:
                    for l0_idx in l0_indices:
                        parent_mask[idx, l0_idx] = 1.0

        # Row-normalize (positions with no L0 parent get zero guidance)
        row_sums = parent_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer('parent_mask', parent_mask / row_sums)

    def _flatten_dict_to_sequence(
        self, tokens_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Convert dict tokens to flat sequence for transformer.

        Args:
            tokens_dict: Dict mapping key → token indices [B]

        Returns:
            Embedded sequence [B, N_total, hidden_dim]
        """
        embeddings = []
        for key in self.sorted_keys:
            tokens = tokens_dict[key]  # [B]
            emb = self.token_embeddings[key](tokens)  # [B, hidden_dim]
            embeddings.append(emb)

        # Stack into sequence
        embeddings = torch.stack(embeddings, dim=1)  # [B, N_total, hidden_dim]
        return embeddings

    def _unflatten_sequence_to_dict(
        self, sequence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Convert flat sequence back to dict logits.

        Args:
            sequence: Encoded sequence [B, N_total, hidden_dim]

        Returns:
            Dict mapping key → logits [B, vocab_size]
        """
        logits_dict = {}
        for i, key in enumerate(self.sorted_keys):
            hidden = sequence[:, i, :]  # [B, hidden_dim]
            logits = self.output_heads[key](hidden)  # [B, vocab_size]
            logits_dict[key] = logits

        return logits_dict

    def _add_hierarchical_guidance(
        self, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Add hierarchical guidance from L0 (coarse) tokens.

        Two modes:
        - "global": All L0 embeddings mean-pooled → broadcast to all positions
        - "per_category": Each position receives guidance only from its own
          family/category L0 parent(s) via a sparse parent_mask

        Args:
            embeddings: Token embeddings [B, N_total, hidden_dim]

        Returns:
            Embeddings with guidance added [B, N_total, hidden_dim]
        """
        if not self.use_hierarchical_guidance:
            return embeddings

        if self.guidance_mode == "per_category" and hasattr(self, 'parent_mask'):
            # Per-category: each position attends to its own L0 parent(s)
            # parent_mask: [N, N], embeddings: [B, N, H] → guidance: [B, N, H]
            guidance = torch.einsum('ij,bjd->bid', self.parent_mask, embeddings)
        else:
            # Global: all L0 embeddings mean-pooled and broadcast
            if len(self._l0_indices) == 0:
                return embeddings
            l0_embeddings = embeddings[:, self._l0_indices, :]  # [B, N_L0, H]
            guidance = l0_embeddings.mean(dim=1, keepdim=True)  # [B, 1, H]

        # Project and add as residual
        guidance = self.guidance_proj(guidance)
        embeddings = embeddings + self.hierarchical_guidance_weight * guidance

        return embeddings

    def forward(
        self,
        tokens_dict: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        observed_dict: Optional[Dict[str, torch.BoolTensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict clean tokens from noisy tokens and timestep.

        Args:
            tokens_dict: Dict mapping key → noisy token indices [B]
            timesteps: Timestep values [B]
            observed_dict: Optional dict of observed masks [B] (for conditioning)

        Returns:
            Dict mapping key → predicted logits [B, vocab_size]
        """
        B = timesteps.shape[0]

        # Flatten dict to sequence
        embeddings = self._flatten_dict_to_sequence(tokens_dict)  # [B, N_total, hidden_dim]

        # Add time embedding (broadcast to all tokens)
        t_emb = self.time_embedding(timesteps)  # [B, hidden_dim]
        t_emb = self.time_mlp(t_emb)  # [B, hidden_dim]
        embeddings = embeddings + t_emb.unsqueeze(1)  # [B, N_total, hidden_dim]

        # Add position embeddings
        positions = torch.arange(self.num_tokens, device=embeddings.device)
        pos_emb = self.position_embedding(positions)  # [N_total, hidden_dim]
        embeddings = embeddings + pos_emb.unsqueeze(0)  # [B, N_total, hidden_dim]

        # Add hierarchical guidance from L0
        embeddings = self._add_hierarchical_guidance(embeddings)

        # Create attention mask for observed tokens (training only)
        # During training: mask unobserved positions so predictions come from
        # observed context only. During eval/inference: allow full attention
        # since unobserved tokens need to coordinate, and observed values are
        # re-injected by RePaint at each reverse step anyway.
        src_key_padding_mask = None
        if observed_dict is not None and self.training:
            mask = []
            for key in self.sorted_keys:
                if key in observed_dict:
                    mask.append(~observed_dict[key])  # True = ignore unobserved
                else:
                    mask.append(torch.zeros(B, dtype=torch.bool, device=embeddings.device))
            src_key_padding_mask = torch.stack(mask, dim=1)  # [B, N_total]

        # Transformer encoding
        encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Unflatten to dict logits
        logits_dict = self._unflatten_sequence_to_dict(encoded)

        return logits_dict
