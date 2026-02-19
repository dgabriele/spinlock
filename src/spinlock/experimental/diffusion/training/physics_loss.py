"""Physics-aware auxiliary loss for D3PM training.

Soft-decodes denoiser logits through frozen VQTokenizer components
(codebooks → shared decoder → inverse heads) and computes MSE against
ground-truth decoded physics parameters (theta, initial conditions).

This encourages the denoiser to produce token distributions that decode
to physically consistent values, reducing reliance on post-hoc correction.

Architecture:
    denoiser logits [B, V_k] per key k
        → softmax(logits / τ) @ codebook_k.weight  [B, D_k]
        → concat all soft_embeddings → [B, total_latent_dim]
        → frozen decoder MLP → [B, total_encoded_dim]
        → split by family → theta_encoded, initial_encoded
        → frozen inverse heads → theta_pred, u0_pred
        → MSE(pred, ground_truth)

Gradients flow: loss → frozen inverse → frozen decoder → soft_probs → logits → denoiser.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from spinlock.experimental.diffusion.config import PhysicsLossConfig

logger = logging.getLogger(__name__)


class PhysicsDecodeHead(nn.Module):
    """Frozen decode pipeline extracted from a VQTokenizer checkpoint.

    Contains only the components needed for soft-decoding:
    - Codebook embeddings per quantizer key (nn.Embedding, frozen)
    - Shared decoder MLP (MultiLayerResidualDecoder, frozen)
    - Inverse heads (ThetaInverseMLP / InitialInverseCNN, frozen)
    - Family dimension metadata and sorted key ordering

    All parameters are frozen (requires_grad=False) but still propagate
    input gradients for the soft-decode path.
    """

    def __init__(
        self,
        codebooks: nn.ModuleDict,
        decoder: nn.Module,
        theta_inverse: Optional[nn.Module],
        initial_inverse: Optional[nn.Module],
        sorted_keys: list,
        sorted_families: list,
        family_dims: Dict[str, int],
    ):
        super().__init__()
        self.codebooks = codebooks
        self.decoder = decoder
        self.theta_inverse = theta_inverse
        self.initial_inverse = initial_inverse
        self.sorted_keys = sorted_keys
        self.sorted_families = sorted_families
        self.family_dims = family_dims

        # Freeze all parameters (gradients still flow through inputs)
        self.requires_grad_(False)

    @classmethod
    def from_tokenizer_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        families: Optional[list] = None,
        device: str = "cpu",
    ) -> "PhysicsDecodeHead":
        """Load VQTokenizer checkpoint, extract decode components, discard rest.

        Args:
            checkpoint_path: Path to VQTokenizer checkpoint (.pt file).
            families: Optional list of families to include inverse heads for.
                      None means include all available.
            device: Device to place the extracted modules on.

        Returns:
            PhysicsDecodeHead with frozen components.
        """
        from spinlock.tokens.tokenizer import VQTokenizer

        logger.info(f"Loading tokenizer checkpoint for physics loss: {checkpoint_path}")
        tokenizer = VQTokenizer.from_checkpoint(checkpoint_path)
        model = tokenizer.model

        # Extract codebook embeddings (sorted key order, matching _extract_embeddings)
        sorted_keys = sorted(model.quantizers.keys())
        codebooks = nn.ModuleDict()
        for key in sorted_keys:
            codebooks[key] = model.quantizers[key].embedding

        # Extract shared decoder
        decoder = model.decoder

        # Extract family dimensions (sorted order, matching _split_reconstructed)
        sorted_families = sorted(model.families)
        family_dims = {}
        for family in sorted_families:
            if family == "temporal":
                family_dims[family] = model.temporal_dim
            elif family == "initial":
                family_dims[family] = model.initial_dim
            elif family == "theta":
                family_dims[family] = model.theta_dim

        # Extract inverse heads (only for requested families)
        theta_inverse = None
        initial_inverse = None

        if families is None or "theta" in families:
            if hasattr(model, "theta_inverse") and model.theta_inverse is not None:
                theta_inverse = model.theta_inverse

        if families is None or "initial" in families:
            if hasattr(model, "initial_inverse") and model.initial_inverse is not None:
                initial_inverse = model.initial_inverse

        head = cls(
            codebooks=codebooks,
            decoder=decoder,
            theta_inverse=theta_inverse,
            initial_inverse=initial_inverse,
            sorted_keys=sorted_keys,
            sorted_families=sorted_families,
            family_dims=family_dims,
        )
        head = head.to(device)

        # Log what was extracted
        total_params = sum(p.numel() for p in head.parameters())
        logger.info(
            f"PhysicsDecodeHead: {len(sorted_keys)} codebooks, "
            f"families={sorted_families}, "
            f"theta_inverse={'yes' if theta_inverse else 'no'}, "
            f"initial_inverse={'yes' if initial_inverse else 'no'}, "
            f"params={total_params:,} (all frozen)"
        )

        return head

    def soft_decode(
        self,
        logits_dict: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        target_tokens: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Differentiable soft-decode: softmax(logits/τ) @ codebook → decoder → inverse heads.

        For masked positions (keys in logits_dict), computes a soft weighted embedding
        that carries gradients into the denoiser.  For unmasked positions (keys absent
        from logits_dict but present in target_tokens), uses the ground-truth hard
        embedding as a constant filler so the decoder always receives the full
        concatenated latent, regardless of masking pattern.

        Args:
            logits_dict: Dict mapping quantizer key → logits [B, V_k].
                         Contains only masked-position keys.
            temperature: Softmax temperature (lower = sharper).
            target_tokens: Dict mapping quantizer key → token indices [B].
                           Used to fill in unmasked positions. If None, unmasked
                           positions are skipped (original behaviour, unsafe for
                           partial masking).

        Returns:
            Dict with decoded physics parameters:
                'theta': [B, param_dim] if theta_inverse available
                'u0': [B, C, H, W] if initial_inverse available
        """
        soft_embeddings = []
        for key in self.sorted_keys:
            if key in logits_dict:
                # Masked position: differentiable soft embedding — gradients flow here.
                logits = logits_dict[key]  # [B, V_k]
                # Schema vocab may be smaller than codebook capacity (unused codes).
                # Pad with -inf so unseen codes get zero probability after softmax.
                V_cb = self.codebooks[key].weight.shape[0]
                if logits.shape[1] < V_cb:
                    logits = F.pad(logits, (0, V_cb - logits.shape[1]), value=float('-inf'))
                probs = F.softmax(logits / temperature, dim=-1)  # [B, V_cb]
                soft_emb = probs @ self.codebooks[key].weight  # [B, D_k]
            elif target_tokens is not None and key in target_tokens:
                # Unmasked position: ground-truth hard embedding as a constant filler.
                # Codebooks are frozen (requires_grad=False) and token indices are
                # integers, so no gradient escapes through this path — correct.
                with torch.no_grad():
                    soft_emb = self.codebooks[key](target_tokens[key])  # [B, D_k]
            else:
                # Key absent from both logits and targets — skip (degenerate case).
                continue
            soft_embeddings.append(soft_emb)

        if not soft_embeddings:
            raise ValueError(
                "soft_decode: no embeddings collected — logits_dict and target_tokens "
                "are both empty or share no keys with sorted_keys."
            )

        latent = torch.cat(soft_embeddings, dim=-1)  # [B, total_latent_dim]
        reconstructed = self.decoder(latent)  # [B, total_encoded_dim]
        return self._split_and_apply_inverse(reconstructed)

    @torch.no_grad()
    def hard_decode(
        self,
        tokens_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Non-differentiable hard-decode: token lookup → decoder → inverse heads.

        Used to compute ground-truth targets from the true token indices.

        Args:
            tokens_dict: Dict mapping quantizer key → token indices [B].

        Returns:
            Dict with decoded physics parameters (detached).
        """
        embeddings = []
        for key in self.sorted_keys:
            if key not in tokens_dict:
                continue
            emb = self.codebooks[key](tokens_dict[key])  # [B, D_k]
            embeddings.append(emb)

        latent = torch.cat(embeddings, dim=-1)  # [B, total_latent_dim]
        reconstructed = self.decoder(latent)  # [B, total_encoded_dim]
        return self._split_and_apply_inverse(reconstructed)

    def _split_and_apply_inverse(
        self, reconstructed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Split reconstructed tensor by family (sorted order) and apply inverse heads.

        Follows the same ordering as JointHierarchicalVQVAE._split_reconstructed():
        sorted(families) → offset accumulation → slice per family.

        Args:
            reconstructed: [B, total_encoded_dim] from shared decoder.

        Returns:
            Dict with 'theta' and/or 'u0' decoded values.
        """
        result = {}
        offset = 0

        for family in self.sorted_families:
            dim = self.family_dims[family]
            family_slice = reconstructed[:, offset:offset + dim]

            if family == "theta" and self.theta_inverse is not None:
                result["theta"] = self.theta_inverse(family_slice)
            elif family == "initial" and self.initial_inverse is not None:
                result["u0"] = self.initial_inverse(family_slice)

            offset += dim

        return result


class PhysicsAwareLoss(nn.Module):
    """Physics-aware auxiliary loss using soft-decoded denoiser logits.

    Computes MSE between soft-decoded predictions (from denoiser logits) and
    hard-decoded ground truth (from true tokens), weighted by a timestep gate
    that downweights high-noise timesteps where logits are near-random.

    Args:
        decode_head: Frozen PhysicsDecodeHead for soft/hard decoding.
        config: PhysicsLossConfig with temperature, weights, gating.
    """

    def __init__(self, decode_head: PhysicsDecodeHead, config: PhysicsLossConfig):
        super().__init__()
        self.decode_head = decode_head
        self.config = config

    @staticmethod
    def _remap_temporal_trunc(
        predicted_logits: Dict[str, torch.Tensor],
        target_tokens: Dict[str, torch.Tensor],
    ):
        """Handle temporal-resolution pretokenized datasets.

        In temporal-resolution mode, temporal quantizer keys are stored with a
        truncation suffix: ``temporal_group_N_trunc_T{TTT}_LM``.  The frozen
        PhysicsDecodeHead expects the base key ``temporal_group_N_LM``.

        This method adds base-key aliases pointing to the max-T (longest rollout)
        truncated version, so soft_decode can collect all embeddings the decoder needs.
        Non-temporal keys and already-present base keys are left unchanged.
        """
        from spinlock.tokens.schema import strip_trunc_suffix, _TRUNC_RE

        # Collect max-T trunc key for each base key
        trunc_map: Dict[str, tuple] = {}  # base_key → (T_int, trunc_key)
        for k in target_tokens:
            m = _TRUNC_RE.match(k)
            if not m:
                continue
            base = strip_trunc_suffix(k)
            T_val = int(m.group(2))
            if base not in trunc_map or T_val > trunc_map[base][0]:
                trunc_map[base] = (T_val, k)

        if not trunc_map:
            return predicted_logits, target_tokens  # not temporal-resolution mode

        new_logits = dict(predicted_logits)
        new_targets = dict(target_tokens)
        for base, (_, trunc_k) in trunc_map.items():
            if base not in new_logits and trunc_k in predicted_logits:
                new_logits[base] = predicted_logits[trunc_k]
            if base not in new_targets and trunc_k in target_tokens:
                new_targets[base] = target_tokens[trunc_k]
        return new_logits, new_targets

    def forward(
        self,
        predicted_logits: Dict[str, torch.Tensor],
        target_tokens: Dict[str, torch.Tensor],
        target_mask: Dict[str, torch.BoolTensor],
        timesteps: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Compute physics-aware loss.

        Args:
            predicted_logits: Dict mapping key → logits [B, V_k] from denoiser.
            target_tokens: Dict mapping key → true token indices [B].
            target_mask: Dict mapping key → bool mask [B] (True = target position).
                         Used only for filtering — physics loss is computed on all
                         samples but the denoiser only generates logits for targets.
            timesteps: Per-sample diffusion timestep [B].
            T: Total number of diffusion timesteps.

        Returns:
            Scalar physics loss (already gated by timestep).
        """
        B = next(iter(predicted_logits.values())).shape[0]
        device = next(iter(predicted_logits.values())).device

        # 1. Timestep gate: downweight at high noise
        gate = self._compute_gate(timesteps, T, device)  # [B]

        # 2. Remap temporal-resolution keys before soft-decoding.
        # In temporal-resolution mode, temporal quantizers are stored as
        # temporal_group_N_trunc_T{TTT}_LM rather than temporal_group_N_LM.
        # Alias the max-T version back to the base key expected by the decoder.
        predicted_logits, target_tokens = self._remap_temporal_trunc(
            predicted_logits, target_tokens
        )

        # 3. Soft-decode predicted logits (differentiable path).
        # Pass target_tokens so unmasked positions are filled with hard embeddings,
        # giving the decoder its expected full-rank concatenated latent regardless
        # of how many tokens were actually masked this batch.
        pred = self.decode_head.soft_decode(
            predicted_logits, self.config.temperature, target_tokens=target_tokens
        )

        # 4. Hard-decode ground-truth tokens (no gradient needed)
        gt = self.decode_head.hard_decode(target_tokens)

        # 5. Per-family MSE, reduced per sample
        per_sample_loss = torch.zeros(B, device=device)

        if "theta" in pred and "theta" in gt:
            theta_mse = F.mse_loss(
                pred["theta"], gt["theta"], reduction="none"
            ).mean(dim=-1)  # [B]
            per_sample_loss = per_sample_loss + self.config.theta_weight * theta_mse

        if "u0" in pred and "u0" in gt:
            u0_mse = F.mse_loss(
                pred["u0"], gt["u0"], reduction="none"
            ).flatten(1).mean(dim=-1)  # [B]
            per_sample_loss = per_sample_loss + self.config.initial_weight * u0_mse

        # 5. Gate by timestep and average
        return (per_sample_loss * gate).mean()

    def _compute_gate(
        self, timesteps: torch.Tensor, T: int, device: torch.device
    ) -> torch.Tensor:
        """Compute timestep gate values.

        Args:
            timesteps: [B] integer timesteps.
            T: Total timesteps.
            device: Tensor device.

        Returns:
            [B] gate values in [0, 1].
        """
        t_float = timesteps.float()

        if self.config.timestep_gate == "cosine":
            # cos(πt/2T): ≈1 at t=0, ≈0 at t=T
            return torch.cos(math.pi * t_float / (2 * T))
        elif self.config.timestep_gate == "linear":
            # 1 - t/T: 1 at t=0, 0 at t=T
            return 1.0 - t_float / T
        else:
            # No gating
            return torch.ones(timesteps.shape[0], device=device)
