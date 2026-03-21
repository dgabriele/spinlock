"""Denoising roundtrip consistency loss for D3PM training.

Soft-decodes D3PM logits through the frozen VQ decode → re-encode pipeline
and compares the resulting quantizer-distance logits against ground-truth
tokens at a truncation level matching the current noise level.

At high noise, the D3PM's uncertain logits produce a coarse soft-decode
that matches short-truncation (T=32) tokens. At low noise, sharp logits
produce precise embeddings matching long-truncation (T=512) tokens.
This gives the denoiser a structural consistency signal at every
denoising step — not just at full resolution.

Architecture:
    D3PM logits [B, V_k] per position
        → soft-decode: softmax(logits/τ) @ codebook.weight → [B, D_k]
        → frozen shared decoder → reconstructed [B, total_encoded_dim]
        → split temporal slice → frozen TemporalInverseMLP
        → [B, T_rt, D_cnn]
        → per group: slice → frozen PyramidTemporalEncoder → frozen rt_projection
        → [B, D_group] per group
        → frozen projector → frozen quantizer distance logits [B, V_k]
        → CE(roundtrip_logits, GT tokens at truncation T_k)

Gradient flows: loss → frozen VQ chain → soft-decode softmax → D3PM logits → denoiser.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from spinlock.experimental.diffusion.config import DenoisingRoundtripLossConfig
from spinlock.experimental.diffusion.training.physics_loss import (
    PhysicsDecodeHead,
    compute_timestep_gate,
)

logger = logging.getLogger(__name__)


class DenoisingRoundtripHead(nn.Module):
    """Frozen VQ decode + re-encode pipeline for roundtrip consistency.

    Composes a PhysicsDecodeHead (soft-decode path) with the temporal
    re-encode components extracted from a VQTokenizer checkpoint:
    - TemporalInverseMLP: reconstructed temporal → [B, T_rt, D_cnn]
    - Per-group PyramidTemporalEncoders: [B, T_rt, G_k] → [B, pyramid_out]
    - Per-group rt_projections: [B, pyramid_out] → [B, D_group]
    - Per-group HierarchicalProjectors: [B, D_group] → List[[B, latent_dim_l]]
    - Per-level quantizer codebooks: for distance-logit computation

    Note on hierarchy: L0/L1/L2 are independent multi-scale projections from
    the same encoded features (NOT residual VQ). Each level is a parallel view
    at a different latent dimensionality, with its own projector head and
    codebook. See projector.py — all heads apply to the same input.

    All parameters are frozen. Gradients flow only through the soft-decode
    path (softmax @ codebook.weight → D3PM logits).
    """

    def __init__(
        self,
        decode_head: PhysicsDecodeHead,
        temporal_inverse: nn.Module,
        per_group_pyramid_encoders: nn.ModuleDict,
        rt_projections: Optional[nn.ModuleDict],
        projectors: nn.ModuleDict,
        quantizers: nn.ModuleDict,
        temporal_group_keys: List[str],
        temporal_group_dim: int,
        family_dims: Dict[str, int],
        sorted_families: List[str],
    ):
        super().__init__()
        self.decode_head = decode_head
        self.temporal_inverse = temporal_inverse
        self.per_group_pyramid_encoders = per_group_pyramid_encoders
        self.rt_projections = rt_projections
        self.projectors = projectors
        self.quantizers = quantizers
        self.temporal_group_keys = temporal_group_keys
        self.temporal_group_dim = temporal_group_dim
        self.family_dims = family_dims
        self.sorted_families = sorted_families

        # Freeze everything except what's already frozen in decode_head
        self.temporal_inverse.requires_grad_(False)
        self.per_group_pyramid_encoders.requires_grad_(False)
        if self.rt_projections is not None:
            self.rt_projections.requires_grad_(False)
        self.projectors.requires_grad_(False)
        self.quantizers.requires_grad_(False)

    @classmethod
    def from_tokenizer_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        decode_head: PhysicsDecodeHead,
        device: str = "cpu",
    ) -> "DenoisingRoundtripHead":
        """Load VQ checkpoint, extract re-encode components.

        Args:
            checkpoint_path: Path to VQTokenizer checkpoint.
            decode_head: Already-loaded PhysicsDecodeHead (shared with PhysicsAwareLoss).
            device: Target device.

        Returns:
            DenoisingRoundtripHead with all components frozen.
        """
        from spinlock.tokens.tokenizer import VQTokenizer

        logger.info(f"Loading VQ checkpoint for roundtrip loss: {checkpoint_path}")
        tokenizer = VQTokenizer.from_checkpoint(checkpoint_path)
        model = tokenizer.model

        if model.temporal_inverse is None:
            raise ValueError(
                "VQ checkpoint has no TemporalInverseMLP — cannot build "
                "DenoisingRoundtripHead. Ensure inverse_heads.temporal config "
                "was set during VQ training."
            )
        if model.per_group_pyramid_encoders is None:
            raise ValueError(
                "VQ checkpoint has no per_group_pyramid_encoders — cannot "
                "build DenoisingRoundtripHead."
            )

        # Temporal group keys (sorted order, matching forward pass)
        temporal_group_keys = sorted(
            k for k in model.group_indices if k.startswith("temporal_")
        )

        # Extract per-group projectors (temporal only)
        projectors = nn.ModuleDict({
            k: model.projectors[k]
            for k in temporal_group_keys
            if k in model.projectors
        })

        # Extract per-group quantizers (temporal only, all levels)
        quantizers = nn.ModuleDict()
        for k in temporal_group_keys:
            for level_idx in range(model.config.hierarchy.num_levels):
                qkey = f"{k}_L{level_idx}"
                if qkey in model.quantizers:
                    quantizers[qkey] = model.quantizers[qkey]

        # rt_projections may not exist (only in pyramid_first mode)
        rt_projections = getattr(model, '_temporal_rt_projections', None)
        temporal_group_dim = getattr(
            model, '_temporal_group_dim',
            model.config.encoder.temporal.learned.embedding_dim
            // model.config.encoder.temporal.learned.num_groups
            if model.config.encoder.temporal.learned is not None
            else model.config.encoder.embedding_dim,
        )

        head = cls(
            decode_head=decode_head,
            temporal_inverse=model.temporal_inverse,
            per_group_pyramid_encoders=model.per_group_pyramid_encoders,
            rt_projections=rt_projections,
            projectors=projectors,
            quantizers=quantizers,
            temporal_group_keys=temporal_group_keys,
            temporal_group_dim=temporal_group_dim,
            family_dims=decode_head.family_dims,
            sorted_families=decode_head.sorted_families,
        )
        head = head.to(device)

        n_params = sum(p.numel() for p in head.parameters())
        logger.info(
            f"DenoisingRoundtripHead: {len(temporal_group_keys)} temporal groups, "
            f"{len(quantizers)} quantizers, params={n_params:,} (all frozen)"
        )

        return head

    def soft_roundtrip(
        self,
        logits_dict: Dict[str, torch.Tensor],
        target_tokens: Dict[str, torch.Tensor],
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Differentiable roundtrip: logits → decode → re-encode → distance logits.

        Args:
            logits_dict: D3PM predicted logits per quantizer key [B, V_k].
            target_tokens: Ground-truth token indices per key [B].
            temperature: Soft-decode temperature.

        Returns:
            Dict mapping temporal quantizer key (e.g. "temporal_group_0_L0")
            → [B, V_k] distance-based logits for CE computation.
        """
        # Remap temporal-resolution keys if needed
        from spinlock.experimental.diffusion.training.physics_loss import (
            PhysicsAwareLoss,
        )
        logits_dict, target_tokens = PhysicsAwareLoss._remap_temporal_trunc(
            logits_dict, target_tokens,
        )

        # 1. Soft-decode → reconstructed [B, total_encoded_dim]
        reconstructed = self.decode_head._soft_decode_to_reconstructed(
            logits_dict, temperature, target_tokens
        )

        # 2. Split temporal slice from reconstructed
        offset = 0
        temporal_slice = None
        for family in self.sorted_families:
            dim = self.family_dims[family]
            if family == "temporal":
                temporal_slice = reconstructed[:, offset:offset + dim]
            offset += dim

        if temporal_slice is None:
            raise ValueError("No temporal family found in reconstructed output")

        # 3. Temporal inverse → [B, T_rt, D_cnn]
        temporal_cnn = self.temporal_inverse(temporal_slice)

        # 4. Per-group: slice → pyramid encoder → rt_projection → projector → distance logits
        roundtrip_logits = {}
        group_dim = self.temporal_group_dim
        for group_idx, group_key in enumerate(self.temporal_group_keys):
            start = group_idx * group_dim
            end = start + group_dim
            group_features = temporal_cnn[:, :, start:end]  # [B, T_rt, group_dim]

            # Pyramid encode
            encoder = self.per_group_pyramid_encoders[group_key]
            enc = encoder(group_features)  # [B, pyramid_out_dim]

            # RT projection (pyramid_first mode: sum(level_dims) → D_group)
            if self.rt_projections is not None and group_key in self.rt_projections:
                enc = self.rt_projections[group_key](enc)

            # Hierarchical projector → per-level latents
            if group_key not in self.projectors:
                continue
            projector = self.projectors[group_key]
            latents = projector(enc)  # List of [B, latent_dim_l]

            # Distance-based logits for each hierarchy level
            for level_idx, latent in enumerate(latents):
                qkey = f"{group_key}_L{level_idx}"
                if qkey not in self.quantizers:
                    continue
                quantizer = self.quantizers[qkey]
                codebook = quantizer.embedding.weight  # [K, D]
                # Negative squared distance → logits (higher = closer)
                dists = torch.cdist(
                    latent.unsqueeze(0), codebook.unsqueeze(0)
                ).squeeze(0).pow(2)
                roundtrip_logits[qkey] = -dists  # [B, K]

        return roundtrip_logits


class DenoisingRoundtripLoss(nn.Module):
    """Multi-resolution roundtrip CE against truncation-matched ground truth.

    Maps each sample's denoising timestep to a truncation level, then
    compares roundtrip logits against ground-truth tokens at that level.

    Args:
        roundtrip_head: Frozen DenoisingRoundtripHead.
        config: DenoisingRoundtripLossConfig.
        truncation_levels: Sorted list of available truncation lengths (e.g. [32, 64, 128, 256, 512]).
    """

    def __init__(
        self,
        roundtrip_head: DenoisingRoundtripHead,
        config: DenoisingRoundtripLossConfig,
        truncation_levels: List[int],
    ):
        super().__init__()
        self.roundtrip_head = roundtrip_head
        self.config = config
        self.truncation_levels = sorted(truncation_levels)
        self.num_levels = len(self.truncation_levels)

        # Pre-compute noise fraction boundaries for truncation mapping.
        # noise_frac in [0, boundaries[0]) → level 0 (shortest truncation),
        # noise_frac in [boundaries[-1], 1] → last level (longest truncation).
        if config.noise_boundaries is not None:
            self._boundaries = config.noise_boundaries
        else:
            # Uniform spacing: n levels → n-1 boundaries
            self._boundaries = [
                (i + 1) / self.num_levels for i in range(self.num_levels - 1)
            ]

        # Collect total code slots for set-coherence Jaccard
        # This maps each temporal quantizer key to its codebook size
        self._temporal_vocab_sizes: Dict[str, int] = {}
        if roundtrip_head is not None:
            for qkey in roundtrip_head.quantizers:
                q = roundtrip_head.quantizers[qkey]
                self._temporal_vocab_sizes[qkey] = q.embedding.weight.shape[0]

        logger.info(
            f"DenoisingRoundtripLoss: {self.num_levels} truncation levels "
            f"{self.truncation_levels}, boundaries={self._boundaries}"
            f", set_coherence_weight={config.set_coherence_weight}"
        )

    def _map_timestep_to_truncation(
        self, effective_t: torch.Tensor, T: int
    ) -> torch.Tensor:
        """Map effective timesteps to truncation level indices.

        Args:
            effective_t: [B] effective timestep values.
            T: Total diffusion timesteps.

        Returns:
            [B] truncation level indices (0 = shortest/coarsest, N-1 = longest/finest).
        """
        noise_frac = effective_t.float() / T  # [B] in [0, 1)

        # High noise (frac near 1) → coarse/short truncation (index 0)
        # Low noise (frac near 0) → fine/long truncation (last index)
        # Invert: use (1 - frac) so low noise maps to high index
        inv_frac = 1.0 - noise_frac

        trunc_idx = torch.zeros_like(effective_t, dtype=torch.long)
        for i, boundary in enumerate(self._boundaries):
            trunc_idx = trunc_idx + (inv_frac > boundary).long()

        return trunc_idx.clamp(0, self.num_levels - 1)

    def _compute_set_coherence(
        self,
        rt_logits: Dict[str, torch.Tensor],
        gt_tokens_per_key: Dict[str, torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        """Differentiable soft Jaccard between predicted code usage and GT.

        For each temporal quantizer key:
        - Soft code usage: softmax(logits / tau) → p_k [B, V_k]
        - Hard GT indicator: one_hot(gt_k) → q_k [B, V_k]
        Aggregate across keys, then compute differentiable Jaccard:
            J = sum(min(p, q)) / sum(max(p, q))

        Args:
            rt_logits: Roundtrip logits per temporal key [B, V_k].
            gt_tokens_per_key: GT tokens per key [B] (already truncation-matched).
            temperature: Softmax sharpness for soft code usage.

        Returns:
            Scalar loss: mean(1 - J) over batch.
        """
        device = next(iter(rt_logits.values())).device
        B = next(iter(rt_logits.values())).shape[0]

        # Accumulate soft and hard code usage across all temporal keys
        p_parts = []  # soft usage vectors
        q_parts = []  # hard GT indicators

        for qkey, logits in rt_logits.items():
            if qkey not in gt_tokens_per_key:
                continue
            V_k = logits.shape[1]
            gt = gt_tokens_per_key[qkey]  # [B]

            # Soft code usage from roundtrip logits
            p_k = F.softmax(logits / temperature, dim=-1)  # [B, V_k]

            # Hard GT indicator
            q_k = F.one_hot(gt.clamp(0, V_k - 1), num_classes=V_k).float()  # [B, V_k]

            p_parts.append(p_k)
            q_parts.append(q_k)

        if not p_parts:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Concatenate across all keys → [B, total_codes]
        p = torch.cat(p_parts, dim=-1)
        q = torch.cat(q_parts, dim=-1)

        # Differentiable Jaccard: J = sum(min(p,q)) / sum(max(p,q))
        intersection = torch.sum(torch.min(p, q), dim=-1)  # [B]
        union = torch.sum(torch.max(p, q), dim=-1)  # [B]
        jaccard = intersection / union.clamp(min=1e-8)  # [B]

        return (1.0 - jaccard).mean()

    def forward(
        self,
        predicted_logits: Dict[str, torch.Tensor],
        target_tokens: Dict[str, torch.Tensor],
        aux_trunc_tokens: Dict[int, Dict[str, torch.Tensor]],
        target_mask: Dict[str, torch.BoolTensor],
        timesteps: torch.Tensor,
        effective_timesteps_dict: Optional[Dict[str, torch.Tensor]],
        T: int,
    ) -> torch.Tensor:
        """Compute roundtrip consistency loss.

        Args:
            predicted_logits: D3PM logits per key [B, V_k].
            target_tokens: Ground-truth tokens per key [B].
            aux_trunc_tokens: {trunc_len: {key: [B]}} auxiliary GT tokens
                at each truncation level.
            target_mask: Boolean mask per key [B] (True = target position).
            timesteps: Global diffusion timestep [B].
            effective_timesteps_dict: Per-key effective timesteps from graded
                schedule. If None, uses global timesteps for all keys.
            T: Total number of diffusion timesteps.

        Returns:
            Scalar roundtrip loss (gated by timestep).
        """
        B = next(iter(predicted_logits.values())).shape[0]
        device = next(iter(predicted_logits.values())).device

        # 1. Timestep gate: weight heavier at low noise (more resolved)
        gate = compute_timestep_gate(timesteps, T, self.config.timestep_gate, device)

        # 2. Soft roundtrip → Dict[temporal_key → [B, V_k] logits]
        rt_logits = self.roundtrip_head.soft_roundtrip(
            predicted_logits, target_tokens, self.config.temperature
        )

        if not rt_logits:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 3. Per-key CE against truncation-matched GT
        per_sample_loss = torch.zeros(B, device=device)
        n_keys = 0

        for qkey, logits in rt_logits.items():
            # Determine effective timestep for this key
            if effective_timesteps_dict is not None and qkey in effective_timesteps_dict:
                eff_t = effective_timesteps_dict[qkey]
            else:
                eff_t = timesteps

            # Map to truncation level index per sample
            trunc_idx = self._map_timestep_to_truncation(eff_t, T)  # [B]

            # Gather GT tokens: stack all truncation levels → [num_trunc, B]
            # then select per-sample based on trunc_idx
            gt_per_level = []
            for tl in self.truncation_levels:
                if tl in aux_trunc_tokens and qkey in aux_trunc_tokens[tl]:
                    gt_per_level.append(aux_trunc_tokens[tl][qkey])
                else:
                    # Fallback: use primary tokens (longest truncation)
                    if qkey in target_tokens:
                        gt_per_level.append(target_tokens[qkey])
                    else:
                        gt_per_level.append(torch.zeros(B, dtype=torch.long, device=device))

            gt_stack = torch.stack(gt_per_level)  # [num_trunc, B]
            gt_selected = gt_stack[trunc_idx, torch.arange(B, device=device)]  # [B]

            # Clamp target to valid range for logits
            V = logits.shape[1]
            gt_selected = gt_selected.clamp(0, V - 1)

            # Per-sample loss: CE or soft weighted Hamming
            if self.config.roundtrip_metric == "weighted_hamming":
                # Soft weighted Hamming: expected embedding distance under
                # predicted distribution vs GT embedding.
                # Gradients flow through softmax → logits → denoiser, and
                # nearby codes get gentler gradients than distant ones.
                codebook = self.roundtrip_head.quantizers[qkey].embedding.weight  # [V, D]
                probs = F.softmax(logits / self.config.temperature, dim=-1)  # [B, V]
                pred_embed = probs @ codebook  # [B, D] — expected embedding
                gt_embed = codebook[gt_selected]  # [B, D] — GT embedding
                loss = (pred_embed - gt_embed).pow(2).sum(dim=-1)  # [B]
            else:
                # Standard CE: all wrong predictions penalized equally
                loss = F.cross_entropy(logits, gt_selected, reduction='none')  # [B]

            per_sample_loss = per_sample_loss + loss
            n_keys += 1

        if n_keys == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Mean across keys, gate by timestep, mean across batch
        per_sample_loss = per_sample_loss / n_keys
        ce_loss = (per_sample_loss * gate).mean()

        # Optional soft set-level coherence (differentiable Jaccard)
        if self.config.set_coherence_weight > 0:
            # Gather truncation-matched GT tokens for each roundtrip key
            gt_matched: Dict[str, torch.Tensor] = {}
            for qkey in rt_logits:
                if effective_timesteps_dict is not None and qkey in effective_timesteps_dict:
                    eff_t = effective_timesteps_dict[qkey]
                else:
                    eff_t = timesteps
                trunc_idx = self._map_timestep_to_truncation(eff_t, T)

                gt_per_level = []
                for tl in self.truncation_levels:
                    if tl in aux_trunc_tokens and qkey in aux_trunc_tokens[tl]:
                        gt_per_level.append(aux_trunc_tokens[tl][qkey])
                    elif qkey in target_tokens:
                        gt_per_level.append(target_tokens[qkey])
                    else:
                        gt_per_level.append(torch.zeros(B, dtype=torch.long, device=device))
                gt_stack = torch.stack(gt_per_level)
                gt_matched[qkey] = gt_stack[trunc_idx, torch.arange(B, device=device)]

            set_loss = self._compute_set_coherence(
                rt_logits, gt_matched, self.config.set_coherence_temperature,
            )
            ce_loss = ce_loss + self.config.set_coherence_weight * set_loss

        return ce_loss
