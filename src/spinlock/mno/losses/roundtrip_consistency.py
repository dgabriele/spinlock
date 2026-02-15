"""Roundtrip token self-consistency loss for MNO.

Trains MNO to produce trajectories that tokenize to self-consistent tokens,
treating the MNO as a neural operator with discrete (token vocabulary) output.

Two GT target modes:

Mode A (pretokenized, fast):
    MNO → trajectory → features → soft_logits
    GT tokens loaded from pretokenized store (same tokens VQTokenizer was trained on)
    Loss: CrossEntropy(soft_logits, GT_hard_tokens)

Mode B (full roundtrip, slow):
    MNO → trajectory → features → soft_logits + hard_tokens
    decode(hard_tokens) → (θ', IC') → QBM(θ', IC') → GT rollout → tokenize → GT_hard_tokens
    Loss: CrossEntropy(soft_logits, GT_hard_tokens)

Gradient flows through soft logits (differentiable) back to MNO parameters.
GT targets are always non-differentiable (hard token indices).
"""

import logging
import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from spinlock.mno.vq_coherence import VQCoherenceAdapter

logger = logging.getLogger(__name__)


class RoundtripConsistencyLoss(nn.Module):
    """Roundtrip token self-consistency loss.

    Computes cross-entropy between MNO's soft token logits and ground-truth
    hard token indices. The soft logits are differentiable through the frozen
    VQ encoder, allowing gradient to flow back to MNO parameters.

    Args:
        vq_adapter: VQCoherenceAdapter with frozen VQ model
        replayer: QBM replayer for Mode B (not needed for Mode A)
        temperature: Softmax temperature for soft logits (default: 1.0)
        families_to_compare: Filter to specific families (None = all)
        num_gt_timesteps: Timesteps for Mode B QBM rollout (default: 256)
    """

    def __init__(
        self,
        vq_adapter: "VQCoherenceAdapter",
        replayer=None,
        temperature: float = 1.0,
        families_to_compare: Optional[List[str]] = None,
        num_gt_timesteps: int = 256,
    ):
        super().__init__()
        self.vq_adapter = vq_adapter
        self.replayer = replayer
        self.temperature = temperature
        self.families_to_compare = families_to_compare
        self.num_gt_timesteps = num_gt_timesteps

    def compute(
        self,
        cleaned_features: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        gt_tokens: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute roundtrip consistency loss.

        Args:
            cleaned_features: [B, T, D_clean] pre-extracted temporal features
                (with gradient to MNO trajectory)
            params: [B, param_dim] QBM Sobol parameters
            gt_tokens: Mode A — pretokenized GT tokens {key: [B] long}.
                If None, falls back to Mode B (full roundtrip).

        Returns:
            Dict with:
                'loss': Scalar cross-entropy loss (differentiable)
                'token_agreement_rate': Fraction of quantizers where
                    argmax(soft_logits) == gt_token (monitoring metric)
        """
        device = cleaned_features.device

        # Step 1: MNO side — soft logits WITH gradient
        mno_out = self.vq_adapter.extract_soft_logits_and_hard_tokens(
            cleaned_features, params=params, temperature=self.temperature,
        )

        if gt_tokens is not None:
            # Mode A: pretokenized GT targets (fast path)
            gt_hard_tokens = gt_tokens
        else:
            # Mode B: full roundtrip — decode → simulate → retokenize
            gt_hard_tokens = self._mode_b_roundtrip(mno_out['hard_tokens'], device)

        # Step 2: Cross-entropy per quantizer
        loss, agreement = self._compute_xent(
            mno_out['soft_logits'], gt_hard_tokens, mno_out['hard_tokens'],
        )

        return {
            'loss': loss,
            'token_agreement_rate': agreement,
        }

    def _compute_xent(
        self,
        soft_logits: Dict[str, torch.Tensor],
        gt_tokens: Dict[str, torch.Tensor],
        pred_tokens: Dict[str, torch.Tensor],
    ) -> tuple:
        """Per-quantizer cross-entropy, averaged over all quantizers.

        Args:
            soft_logits: {key: [B, K]} differentiable logits from MNO path
            gt_tokens: {key: [B]} hard GT token indices
            pred_tokens: {key: [B]} hard predicted indices (for agreement metric)

        Returns:
            (loss, agreement_rate) tuple
        """
        losses = []
        agreements = []
        num_compared = 0

        for key in sorted(soft_logits.keys()):
            if key not in gt_tokens:
                continue

            # Family filter
            if self.families_to_compare is not None:
                family = key.split('_')[0]
                if family not in self.families_to_compare:
                    continue

            logits = soft_logits[key]    # [B, K]
            target = gt_tokens[key].to(logits.device)  # [B]

            # Validate target indices are in range
            K = logits.shape[1]
            valid_mask = target < K
            if not valid_mask.all():
                # Skip out-of-range targets (can happen with vocab mismatches)
                logits = logits[valid_mask]
                target = target[valid_mask]
                if target.numel() == 0:
                    continue

            losses.append(F.cross_entropy(logits, target))

            # Agreement: does argmax(soft_logits) match GT?
            if key in pred_tokens:
                pred = pred_tokens[key].to(target.device)
                if self.families_to_compare is not None:
                    pred = pred[valid_mask] if not valid_mask.all() else pred
                agreements.append((pred == target).float().mean().item())

            num_compared += 1

        if not losses:
            device = next(iter(soft_logits.values())).device
            return torch.tensor(0.0, device=device), 0.0

        loss = torch.stack(losses).mean()
        agreement = sum(agreements) / len(agreements) if agreements else 0.0

        return loss, agreement

    @torch.no_grad()
    def _mode_b_roundtrip(
        self,
        hard_tokens: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Mode B: decode tokens → simulate → retokenize to get GT targets.

        All operations are non-differentiable (GT targets don't need gradient).

        Args:
            hard_tokens: MNO's predicted hard token indices
            device: Target device

        Returns:
            GT hard tokens from the roundtrip path
        """
        if self.replayer is None:
            raise RuntimeError(
                "Mode B roundtrip requires a replayer. Either provide "
                "gt_tokens (Mode A) or set replayer in constructor."
            )

        # Decode tokens → physics parameters
        decoded = self.vq_adapter.decode_tokens_to_params(hard_tokens)

        if 'theta' not in decoded:
            raise RuntimeError(
                "Mode B requires theta_inverse to decode tokens to parameters. "
                "Load inverse heads via vq_adapter.load_inverse_heads()."
            )

        theta = decoded['theta']  # [B, param_dim]
        u0 = decoded.get('u0')    # [B, C, H, W] or None

        # Simulate with QBM
        if hasattr(self.replayer, 'rollout_batch_with_explicit_ic') and u0 is not None:
            gt_trajectory = self.replayer.rollout_batch_with_explicit_ic(
                params_batch=theta.cpu().numpy(),
                psi_0=u0,
                num_timesteps=self.num_gt_timesteps,
            )
        else:
            gt_trajectory = self.replayer.rollout_batch(
                params_batch=theta.cpu().numpy(),
                num_realizations=1,
                num_timesteps=self.num_gt_timesteps,
            ).squeeze(1)  # [B, T, C, H, W]

        # Extract features and tokenize
        gt_cleaned = self.vq_adapter.extract_features(gt_trajectory.to(device))
        gt_out = self.vq_adapter.encode_and_quantize(
            gt_cleaned.detach(), params=theta.to(device),
        )

        return gt_out['token_indices']
