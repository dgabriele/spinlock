"""Multi-scale roundtrip VQ tokenisation loss for MNO.

Trains the MNO to produce trajectories that tokenise to diverse, semantically
distinct tokens at *every* temporal resolution defined by the tokeniser's pyramid
encoder — not just at the full trajectory length.

For each truncation length T_i auto-detected from the tokeniser:
    1. Slice MNO predicted trajectory to first T_i timesteps
    2. Extract + clean temporal features via VQCoherenceAdapter.encode_trajectory()
    3. Compute cross-entropy against GT tokens at truncation T_i
    4. Accumulate weighted sum

Truncation levels are auto-detected from:
    tokeniser.config.encoder.temporal.downsample_factors + max_timesteps
Formula (mirrors pretokenize_dataset.py):
    T_i = max_timesteps // factor  for factor in downsample_factors

Falls back to [max_timesteps] (single scale) if encoder is not pyramid variant.

GT token sources:
    Fast path  — gt_tokens dict provided (loaded from pretokenized HDF5):
                 no feature re-extraction, no tokeniser forward pass for GT
    Slow path  — gt_trajectory provided:
                 tokenise GT trajectory on-the-fly via frozen VQCoherenceAdapter
"""

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from spinlock.mno.base_loss import BaseNOALoss, LossOutput
from spinlock.mno.config import MultiScaleRoundtripConfig

if TYPE_CHECKING:
    from spinlock.mno.vq_coherence import VQCoherenceAdapter

logger = logging.getLogger(__name__)


class MultiScaleRoundtripLoss(BaseNOALoss):
    """Roundtrip VQ tokenisation loss at every temporal resolution defined by the
    tokeniser's pyramid encoder.

    Args:
        vq_adapter: VQCoherenceAdapter wrapping a frozen VQTokenizer
        config: MultiScaleRoundtripConfig with scale weights and family filter
    """

    def __init__(
        self,
        vq_adapter: "VQCoherenceAdapter",
        config: MultiScaleRoundtripConfig,
    ):
        super().__init__()
        self._adapter = vq_adapter
        self._config = config
        self._truncation_lengths = self._resolve_truncation_lengths()
        self._scale_weights = self._resolve_scale_weights()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_truncation_lengths(self) -> List[int]:
        """Detect truncation lengths from tokeniser pyramid config or explicit list."""
        if self._config.truncation_lengths is not None:
            lengths = sorted(self._config.truncation_lengths)
            logger.info(f"MultiScaleRoundtripLoss: using explicit truncation lengths {lengths}")
            return lengths

        enc_cfg = self._adapter._config.encoder.temporal

        if getattr(enc_cfg, 'variant', 'mean') != 'pyramid':
            max_T = getattr(enc_cfg, 'max_timesteps', 256)
            logger.info(
                f"MultiScaleRoundtripLoss: non-pyramid encoder ({enc_cfg.variant!r}), "
                f"using single scale [{max_T}]"
            )
            return [max_T]

        factors = enc_cfg.downsample_factors   # e.g. [1, 2, 4, 8]
        max_T = enc_cfg.max_timesteps          # e.g. 256
        lengths = sorted(set(max_T // f for f in factors))
        logger.info(
            f"MultiScaleRoundtripLoss: auto-detected truncation lengths {lengths} "
            f"from downsample_factors={factors}, max_timesteps={max_T}"
        )
        return lengths

    def _resolve_scale_weights(self) -> Dict[int, float]:
        """Return per-scale weights, defaulting to uniform 1/N."""
        if self._config.scale_weights:
            return {int(k): float(v) for k, v in self._config.scale_weights.items()}
        w = 1.0 / len(self._truncation_lengths)
        return {T: w for T in self._truncation_lengths}

    # ------------------------------------------------------------------
    # Primary compute method
    # ------------------------------------------------------------------

    def compute(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: Optional[torch.Tensor] = None,
        ic: Optional[torch.Tensor] = None,
        noa=None,
        *,
        gt_tokens: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        gt_raw_features: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """Compute multi-scale roundtrip loss.

        Args:
            pred_trajectory:  [B, T_full, C, H, W]  MNO predicted output
            target_trajectory:[B, T_full, C, H, W]  GT trajectory (slow path)
            ic:               Unused (kept for BaseNOALoss interface compatibility)
            noa:              Unused (kept for BaseNOALoss interface compatibility)
            gt_tokens:        Fast path — pretokenized GT tokens loaded from HDF5.
                              {T_trunc: {base_key: [B] long tensor}}
                              When provided, target_trajectory is not needed.

        Returns:
            LossOutput with per-scale components keyed as "roundtrip_T{TTT}"
        """
        compute_roundtrip = gt_tokens is not None or target_trajectory is not None
        compute_feat_mse = gt_raw_features is not None and self._config.lambda_feat_mse > 0

        assert compute_roundtrip or compute_feat_mse, (
            "MultiScaleRoundtripLoss.compute: either gt_tokens/target_trajectory (roundtrip CE) "
            "or gt_raw_features + lambda_feat_mse > 0 (feature MSE) must be provided."
        )

        device = pred_trajectory.device
        total = torch.zeros(1, device=device)
        components: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, float] = {}

        # --- Extract features ONCE over the available prediction window ---------------
        # With truncated BPTT, pred_trajectory has bptt_window timesteps (e.g. 32),
        # while truncation lengths T ≥ 32. We extract features a single time here and
        # reuse them for both the feature MSE and all per-scale roundtrip CE calls —
        # avoiding redundant feature extraction (each call is ~1-2s of compute).
        #
        # All scales slice pred_trajectory[:, :T] which clips to the same BPTT window,
        # so they share identical features regardless of T.
        pred_feat = None
        if compute_roundtrip or compute_feat_mse:
            pred_feat = self._adapter.extract_features(pred_trajectory)  # [B, actual_T, D]

        if compute_feat_mse:
            actual_T = pred_feat.shape[1]
            gt_slice = gt_raw_features[:, :actual_T].to(device)
            gt_clean_full = self._adapter.clean_raw_features(gt_slice).detach()
            feat_mse_val = F.mse_loss(pred_feat, gt_clean_full)
            total = total + feat_mse_val * self._config.lambda_feat_mse
            # Keep non-detached so grad diagnostic can backprop through it
            components['feat_mse'] = feat_mse_val
            metrics['feat_mse'] = feat_mse_val.item()

        # --- Per-scale roundtrip cross-entropy (token CE against GT hard tokens) ----
        # Compute soft logits ONCE — with BPTT, pred_feat covers the same window
        # regardless of scale T, so all 4 CE calls share identical logits.
        # GT tokens differ per scale (different temporal pooling targets).
        pred_logits_shared = None
        if compute_roundtrip and pred_feat is not None:
            pred_logits_shared = self._adapter.extract_soft_logits_and_hard_tokens(
                pred_feat
            )['soft_logits']

        for T in self._truncation_lengths:
            if not compute_roundtrip:
                break

            pred_logits_T = pred_logits_shared

            # Determine GT token source (fast path or slow path)
            if gt_tokens is not None and T in gt_tokens:
                gt_tokens_T = {
                    k: v.to(device)
                    for k, v in gt_tokens[T].items()
                    if self._is_target_family(k)
                }
            else:
                assert target_trajectory is not None, (
                    f"gt_tokens missing entry for T={T} and no target_trajectory provided"
                )
                with torch.no_grad():
                    gt_T = target_trajectory[:, :T]
                    _, gt_logits_T = self._adapter.encode_trajectory(gt_T)
                    gt_tokens_T = {
                        k: logits.argmax(-1)
                        for k, logits in gt_logits_T.items()
                        if self._is_target_family(k)
                    }

            pred_logits_filtered = {
                k: v for k, v in pred_logits_T.items()
                if self._is_target_family(k)
            }

            scale_ce = self._cross_entropy(pred_logits_filtered, gt_tokens_T)
            weight = self._scale_weights.get(T, 1.0)
            total = total + weight * scale_ce * self._config.lambda_roundtrip
            key = f"roundtrip_T{T:03d}"
            components[key] = scale_ce.detach()
            metrics[key] = scale_ce.item()

        metrics['total'] = total.item()
        return LossOutput(total=total.squeeze(), components=components, metrics=metrics)

    # ------------------------------------------------------------------
    # BaseNOALoss abstract properties
    # ------------------------------------------------------------------

    @property
    def leading_loss_name(self) -> str:
        return "roundtrip"

    @property
    def auxiliary_loss_names(self) -> List[str]:
        return [f"roundtrip_T{T:03d}" for T in self._truncation_lengths]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_target_family(self, key: str) -> bool:
        """Return True if the quantizer key belongs to one of the target families."""
        family = key.split('_')[0]
        return family in self._config.families

    @staticmethod
    def _cross_entropy(
        logits: Dict[str, torch.Tensor],   # base_key → [B, V]  (differentiable)
        targets: Dict[str, torch.Tensor],  # base_key → [B]     (long, non-diff)
    ) -> torch.Tensor:
        """Per-quantizer cross-entropy averaged over all matching keys."""
        losses = []
        for key in sorted(targets.keys()):
            if key not in logits:
                continue
            logit = logits[key]              # [B, V]
            target = targets[key].long()     # [B]
            # Guard against stale vocab mismatches (e.g. checkpoint vs dataset)
            V = logit.shape[1]
            valid = target < V
            if not valid.all():
                logit = logit[valid]
                target = target[valid]
                if target.numel() == 0:
                    continue
            losses.append(F.cross_entropy(logit, target))

        if not losses:
            device = next(iter(logits.values())).device if logits else torch.device('cpu')
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()
