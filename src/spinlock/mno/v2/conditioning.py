"""Conditioning adapters for V2 MNO.

The ConditioningAdapter protocol decouples how conditioning inputs (parameters,
initial conditions, latent codes, etc.) are mapped to backbone-compatible tensors.
This enables future adapter variants (latent-conditioned, token-conditioned) without
modifying the V2MNO model or training loop.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, FrozenSet, List, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@runtime_checkable
class ConditioningAdapter(Protocol):
    """Maps arbitrary conditioning inputs to backbone-compatible format."""

    def prepare(self, raw: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Transform raw conditioning dict into backbone-ready format.

        Must return at least ``{"ic": ...}``. May also return ``{"params": ...}``
        for parameter-conditioned backbones (FiLM / concat).
        """
        ...

    @property
    def required_keys(self) -> FrozenSet[str]:
        """Keys that must be present in the raw conditioning dict."""
        ...


class ThetaICAdapter:
    """Default adapter: theta + IC -> backbone params + initial condition.

    Expects raw dict with keys ``"theta"`` ([B, P]) and ``"ic"`` ([B, C, H, W]).
    Passes through as ``{"params": theta, "ic": ic}`` for the MNOBackbone.
    """

    @property
    def required_keys(self) -> FrozenSet[str]:
        return frozenset({"theta", "ic"})

    def prepare(self, raw: Dict[str, Tensor]) -> Dict[str, Tensor]:
        missing = self.required_keys - raw.keys()
        if missing:
            raise ValueError(f"Missing conditioning keys: {missing}")
        return {"ic": raw["ic"], "params": raw["theta"]}


class TokenEmbeddingProjector(nn.Module):
    """Frozen VQ codebook lookup + per-group shared MLP for token conditioning.

    Loads quantizer embeddings from a VQ checkpoint, freezes them, and processes
    each group's concatenated L0+L1+L2 embeddings through a shared MLP (DeepSets
    principle — groups are symmetric by construction). Mean pooling aggregates
    across groups, followed by a refinement MLP to token_embed_dim.

    Architecture::

        Per group g: cat(L0[g], L1[g], L2[g]) → [B, per_group_dim]
        Shared MLP:  Linear → LayerNorm → GELU → Linear → [B, hidden_dim]
        Mean pool:   mean over groups → [B, hidden_dim]
        Refine MLP:  Linear → LayerNorm → GELU → [B, token_embed_dim]

    Args:
        vq_checkpoint_path: Path to VQTokenizer checkpoint.
        token_embed_dim: Output dimension of the refinement layer.
    """

    # Pattern: "temporal_group_5_L2" → group="temporal_group_5", level=2
    _QKEY_PATTERN = re.compile(r"^(.+)_L(\d+)$")

    def __init__(self, vq_checkpoint_path: str, token_embed_dim: int = 128) -> None:
        super().__init__()
        self._token_embed_dim = token_embed_dim

        # Load VQ tokenizer, extract quantizer embeddings, discard the rest
        from spinlock.tokens.tokenizer import VQTokenizer

        tokenizer = VQTokenizer.from_checkpoint(vq_checkpoint_path)

        self._quantizer_keys: List[str] = sorted(
            tokenizer.model.quantizers.keys(),
        )
        embeddings = nn.ModuleDict()
        total_dim = 0

        for qkey in self._quantizer_keys:
            q = tokenizer.model.quantizers[qkey]
            weight = q.embedding.weight.data.clone()  # [K, D]
            emb = nn.Embedding(weight.shape[0], weight.shape[1])
            emb.weight.data.copy_(weight)
            emb.weight.requires_grad = False
            embeddings[qkey] = emb
            total_dim += weight.shape[1]

        del tokenizer  # Free memory — only embeddings are needed

        self.embeddings = embeddings
        self._total_codebook_dim = total_dim

        # --- Auto-detect group structure from quantizer keys ---
        # Parse "temporal_group_5_L2" → group="temporal_group_5", level=2
        group_levels: Dict[str, Dict[int, str]] = defaultdict(dict)
        for qkey in self._quantizer_keys:
            m = self._QKEY_PATTERN.match(qkey)
            if m:
                group_name, level_str = m.group(1), m.group(2)
                group_levels[group_name][int(level_str)] = qkey
            else:
                raise ValueError(
                    f"Quantizer key {qkey!r} does not match expected "
                    f"pattern '<group>_L<level>'"
                )

        # Sort groups for deterministic ordering
        self._group_names: List[str] = sorted(group_levels.keys())
        self._num_groups = len(self._group_names)

        # Build ordered level keys per group and verify uniform structure
        ref_levels = sorted(group_levels[self._group_names[0]].keys())
        self._group_keys: Dict[str, List[str]] = {}
        for gname in self._group_names:
            levels = sorted(group_levels[gname].keys())
            if levels != ref_levels:
                raise ValueError(
                    f"Group {gname!r} has levels {levels}, expected {ref_levels}"
                )
            self._group_keys[gname] = [group_levels[gname][lv] for lv in ref_levels]

        # Compute per-group dimension (sum of embedding dims across levels)
        ref_group = self._group_names[0]
        per_group_dim = sum(
            embeddings[qk].embedding_dim for qk in self._group_keys[ref_group]
        )
        # Verify all groups have the same per_group_dim
        for gname in self._group_names[1:]:
            gd = sum(
                embeddings[qk].embedding_dim for qk in self._group_keys[gname]
            )
            if gd != per_group_dim:
                raise ValueError(
                    f"Group {gname!r} has per_group_dim={gd}, "
                    f"expected {per_group_dim} (from {ref_group!r})"
                )
        self._per_group_dim = per_group_dim
        self._num_levels = len(ref_levels)

        # --- Shared group MLP (weight-tied across all groups) ---
        group_hidden = 64
        self.group_mlp = nn.Sequential(
            nn.Linear(per_group_dim, group_hidden),
            nn.LayerNorm(group_hidden),
            nn.GELU(),
            nn.Linear(group_hidden, group_hidden),
        )

        # --- Refinement MLP: pooled → token_embed_dim ---
        self.refine_mlp = nn.Sequential(
            nn.Linear(group_hidden, token_embed_dim),
            nn.LayerNorm(token_embed_dim),
            nn.GELU(),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "TokenEmbeddingProjector: %d groups x %d levels, "
            "per_group_dim=%d, group_mlp: %d→%d, refine: %d→%d (~%dK params)",
            self._num_groups,
            self._num_levels,
            per_group_dim,
            per_group_dim,
            group_hidden,
            group_hidden,
            token_embed_dim,
            n_params // 1000,
        )

    @property
    def token_embed_dim(self) -> int:
        return self._token_embed_dim

    @property
    def total_codebook_dim(self) -> int:
        return self._total_codebook_dim

    @property
    def quantizer_keys(self) -> List[str]:
        return self._quantizer_keys

    def forward(self, tokens: Dict[str, Tensor]) -> Tensor:
        """Look up frozen codebook embeddings, apply per-group MLP, mean-pool.

        Args:
            tokens: Dict mapping quantizer key → [B] token indices.

        Returns:
            Projected embeddings [B, token_embed_dim].
        """
        group_embeddings = []
        for gname in self._group_names:
            # Concatenate L0+L1+L2 embeddings for this group
            level_parts = []
            for qkey in self._group_keys[gname]:
                idx = tokens[qkey]  # [B]
                emb = self.embeddings[qkey](idx)  # [B, D_level]
                level_parts.append(emb)
            group_emb = torch.cat(level_parts, dim=-1)  # [B, per_group_dim]
            group_embeddings.append(group_emb)

        # Stack groups: [B, num_groups, per_group_dim]
        stacked = torch.stack(group_embeddings, dim=1)

        # Shared MLP across groups: [B, num_groups, group_hidden]
        h = self.group_mlp(stacked)

        # Mean pool across groups: [B, group_hidden]
        pooled = h.mean(dim=1)

        # Refinement: [B, token_embed_dim]
        return self.refine_mlp(pooled)


class TokenThetaICAdapter(nn.Module):
    """Token-conditioned adapter: theta + token embeddings + IC → backbone.

    Concatenates a learnable token embedding projection with the physics
    parameter vector theta, producing an augmented conditioning vector
    [B, param_dim + token_embed_dim] for the backbone's param_embedding MLP.

    Args:
        projector: Frozen codebook lookup + learnable projection module.
    """

    def __init__(self, projector: TokenEmbeddingProjector) -> None:
        super().__init__()
        self.projector = projector

    @property
    def required_keys(self) -> FrozenSet[str]:
        return frozenset({"theta", "ic", "token_indices"})

    def prepare(self, raw: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Concatenate theta with projected token embeddings.

        Args:
            raw: Must contain "theta" [B, P], "ic" [B, C, H, W],
                and "token_indices" Dict[str, [B]].

        Returns:
            {"ic": ..., "params": [B, P + token_embed_dim]}
        """
        token_emb = self.projector(raw["token_indices"])  # [B, token_embed_dim]
        params = torch.cat([raw["theta"], token_emb], dim=1)
        return {"ic": raw["ic"], "params": params}
