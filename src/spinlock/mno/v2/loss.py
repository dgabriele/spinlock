"""TrajectoryLoss: trajectory MSE + IC MSE + contrastive + token CE for v2 MNO training.

Gradient hierarchy:
  1. Trajectory MSE — direct pixel-level supervision on BPTT-aligned window
  2. IC MSE — ensures warmup endpoint is close to GT
  3. InfoNCE contrastive — different parameters must produce distinguishable rollouts
  4. Feature MSE (optional) — matches VQ tokenizer feature space (disabled by default)
  5. Token CE (optional) — soft VQ cross-entropy against pretokenized GT tokens
     Enabled via lambda_token_ce > 0 in QA mode. Becomes primary loss when active.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spinlock.mno.base_loss import BaseNOALoss, LossOutput
from spinlock.mno.losses.components.contrastive import ContrastiveLoss

logger = logging.getLogger(__name__)


class TrajectoryLoss(BaseNOALoss):
    """Trajectory-first loss for v2 MNO training.

    L_total = λ_traj × MSE(pred, gt)
            + λ_ic × MSE(pred[:,0], gt[:,0])
            + λ_contrastive × InfoNCE(rollout, params)
            + λ_feat_mse × MSE(pred_features, gt_features)  [optional]
            + λ_token_ce × CE(temporal_soft_logits, gt_tokens) [optional, QA mode]
    """

    # Loss component names that support EMA normalization.
    _EMA_KEYS = ("traj_mse", "ic_mse", "contrastive", "feat_mse", "token_ce")

    def __init__(
        self,
        contrastive_loss: ContrastiveLoss,
        lambda_traj: float = 1.0,
        lambda_ic: float = 0.3,
        lambda_contrastive: float = 0.3,
        lambda_feat_mse: float = 0.0,
        lambda_token_ce: float = 0.0,
        token_ce_temperature: float = 1.0,
        normalize_loss_scales: bool = False,
        loss_scale_ema_momentum: float = 0.99,
        vq_adapter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self._contrastive = contrastive_loss
        self._lambda_traj = lambda_traj
        self._lambda_ic = lambda_ic
        self._lambda_contrastive = lambda_contrastive
        self._lambda_feat_mse = lambda_feat_mse
        self._lambda_token_ce = lambda_token_ce
        self._token_ce_temperature = token_ce_temperature
        self._normalize = normalize_loss_scales
        self._ema_momentum = loss_scale_ema_momentum
        self._vq_adapter = vq_adapter

        # EMA buffers for loss-scale normalization (persisted in checkpoint)
        for key in self._EMA_KEYS:
            self.register_buffer(f"ema_{key}", torch.tensor(1.0))

        logger.info(
            "TrajectoryLoss: λ_traj=%.2f, λ_ic=%.2f, λ_contrastive=%.2f, "
            "λ_feat_mse=%.2f, λ_token_ce=%.2f, normalize=%s, vq_adapter=%s",
            lambda_traj, lambda_ic, lambda_contrastive, lambda_feat_mse,
            lambda_token_ce, normalize_loss_scales,
            "yes" if vq_adapter is not None else "no",
        )

    @property
    def leading_loss_name(self) -> str:
        if self._lambda_token_ce > 0:
            return "token_ce"
        return "traj_mse"

    @property
    def auxiliary_loss_names(self) -> List[str]:
        names = ["ic_mse", "contrastive"]
        if self._lambda_feat_mse > 0:
            names.append("feat_mse")
        if self._lambda_token_ce > 0:
            names.append("traj_mse")
            names.append("token_match_acc")
        else:
            pass  # traj_mse is the leading loss
        return names

    def _scale_loss(self, name: str, raw_loss: Tensor) -> Tensor:
        """Normalize a loss component by its running EMA magnitude.

        When normalize_loss_scales is enabled, each loss L_i is replaced by
        L_i / EMA(L_i), so that λ weights reflect *actual* gradient ratios
        regardless of raw loss scale differences.

        Args:
            name: Loss component name (must be in _EMA_KEYS).
            raw_loss: Raw loss tensor (scalar).

        Returns:
            raw_loss / EMA if normalizing and loss is non-zero, else raw_loss.
        """
        if not self._normalize:
            return raw_loss
        raw_val = raw_loss.item()
        if raw_val == 0.0:
            return raw_loss
        ema: Tensor = getattr(self, f"ema_{name}")
        # Update EMA (no grad — this is bookkeeping, not part of the loss graph)
        with torch.no_grad():
            ema.lerp_(torch.tensor(raw_val, device=ema.device), 1.0 - self._ema_momentum)
        return raw_loss / ema.clamp(min=1e-6)

    def compute(
        self,
        pred_trajectory: Tensor,
        target_trajectory: Optional[Tensor] = None,
        ic: Optional[Tensor] = None,
        noa: Optional[nn.Module] = None,
        *,
        params: Optional[Tensor] = None,
        gt_raw_features: Optional[Tensor] = None,
        gt_tokens: Optional[Dict[str, Tensor]] = None,
    ) -> LossOutput:
        """Compute trajectory-first loss.

        Args:
            pred_trajectory: BPTT-aligned predicted states [B, W, C, H, W].
            target_trajectory: BPTT-aligned GT states [B, W, C, H, W].
            params: Parameter vectors [B, P] for contrastive loss.
            gt_raw_features: Optional GT temporal features [B, T, D_raw].
            gt_tokens: Optional pretokenized GT tokens {quantizer_key: [B]}.
                Used with vq_adapter for soft VQ cross-entropy loss.

        Returns:
            LossOutput with total loss and per-component metrics.
        """
        device = pred_trajectory.device
        components: Dict[str, Tensor] = {}
        metrics: Dict[str, float] = {}

        # --- Trajectory MSE ---
        traj_mse = F.mse_loss(pred_trajectory, target_trajectory)
        components["traj_mse"] = traj_mse
        metrics["traj_mse"] = traj_mse.item()

        # --- IC MSE (warmup endpoint accuracy) ---
        # pred[:,0] and gt[:,0] are the first states in the supervised window
        ic_mse = F.mse_loss(pred_trajectory[:, 0], target_trajectory[:, 0])
        components["ic_mse"] = ic_mse
        metrics["ic_mse"] = ic_mse.item()

        # --- Contrastive (diversity) ---
        contrastive_loss = torch.tensor(0.0, device=device)
        contrastive_acc = 0.0
        if params is not None and self._lambda_contrastive > 0:
            # ContrastiveLoss.forward expects [B, T, C, H, W] rollout
            contrastive_out = self._contrastive(pred_trajectory, params)
            contrastive_loss = contrastive_out["loss"]
            contrastive_acc = contrastive_out["accuracy"].item()
        components["contrastive"] = contrastive_loss
        metrics["contrastive"] = contrastive_loss.item()
        metrics["contrastive_acc"] = contrastive_acc

        # --- Feature MSE (optional, disabled by default) ---
        feat_mse = torch.tensor(0.0, device=device)
        if (
            self._vq_adapter is not None
            and gt_raw_features is not None
            and self._lambda_feat_mse > 0
        ):
            pred_feat = self._vq_adapter.extract_features(pred_trajectory)
            gt_clean = self._vq_adapter.clean_raw_features(
                gt_raw_features.to(device),
            ).detach()
            # Match temporal dims
            T_match = min(pred_feat.shape[1], gt_clean.shape[1])
            feat_mse = F.mse_loss(
                pred_feat[:, :T_match], gt_clean[:, :T_match],
            )
        components["feat_mse"] = feat_mse
        metrics["feat_mse"] = feat_mse.item()

        # --- Token CE (optional, QA mode soft VQ cross-entropy) ---
        token_ce = torch.tensor(0.0, device=device)
        token_match_acc = 0.0
        if (
            self._vq_adapter is not None
            and gt_tokens is not None
            and self._lambda_token_ce > 0
        ):
            # Differentiable: pred_trajectory → features → soft logits
            cleaned = self._vq_adapter.extract_features(pred_trajectory)
            result = self._vq_adapter.extract_soft_logits_and_hard_tokens(
                cleaned, temperature=self._token_ce_temperature,
            )
            token_ce = self._temporal_token_ce(
                result["soft_logits"], gt_tokens,
            )
            token_match_acc = self._temporal_token_accuracy(
                result["hard_tokens"], gt_tokens,
            )
        components["token_ce"] = token_ce
        metrics["token_ce"] = token_ce.item()
        metrics["token_match_acc"] = token_match_acc

        # --- Normalize loss scales (optional) ---
        # When enabled, each L_i is divided by EMA(L_i) so lambdas reflect
        # true gradient importance rather than accidental scale differences.
        traj_scaled = self._scale_loss("traj_mse", traj_mse)
        ic_scaled = self._scale_loss("ic_mse", ic_mse)
        contrastive_scaled = self._scale_loss("contrastive", contrastive_loss)
        feat_scaled = self._scale_loss("feat_mse", feat_mse)
        token_ce_scaled = self._scale_loss("token_ce", token_ce)

        # --- Combined ---
        total = (
            self._lambda_traj * traj_scaled
            + self._lambda_ic * ic_scaled
            + self._lambda_contrastive * contrastive_scaled
            + self._lambda_feat_mse * feat_scaled
            + self._lambda_token_ce * token_ce_scaled
        )
        metrics["total"] = total.item()

        # Log effective contributions when normalizing
        if self._normalize:
            metrics["eff_traj"] = (self._lambda_traj * traj_scaled).item()
            metrics["eff_token_ce"] = (self._lambda_token_ce * token_ce_scaled).item()

        return LossOutput(total=total, components=components, metrics=metrics)

    @staticmethod
    def _temporal_token_ce(
        soft_logits: Dict[str, Tensor],
        gt_tokens: Dict[str, Tensor],
    ) -> Tensor:
        """Per-quantizer cross-entropy averaged over temporal families only.

        Args:
            soft_logits: {quantizer_key: [B, K]} negative L2 distances / temp
            gt_tokens: {quantizer_key: [B]} hard GT token indices

        Returns:
            Scalar mean CE loss across all temporal quantizers.
        """
        losses = []
        for key in sorted(gt_tokens.keys()):
            if not key.startswith("temporal_"):
                continue
            if key not in soft_logits:
                continue
            logit = soft_logits[key]        # [B, K]
            target = gt_tokens[key].long()  # [B]
            V = logit.shape[1]
            valid = target < V              # guard against vocab mismatch
            if valid.any():
                losses.append(F.cross_entropy(logit[valid], target[valid]))
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=next(iter(soft_logits.values())).device)

    @staticmethod
    def _temporal_token_accuracy(
        hard_tokens: Dict[str, Tensor],
        gt_tokens: Dict[str, Tensor],
    ) -> float:
        """Mean token match accuracy across temporal quantizers.

        Args:
            hard_tokens: {quantizer_key: [B]} predicted hard indices
            gt_tokens: {quantizer_key: [B]} GT hard indices

        Returns:
            Fraction of (quantizer, sample) pairs where pred == gt.
        """
        correct, total = 0, 0
        for key in sorted(gt_tokens.keys()):
            if not key.startswith("temporal_"):
                continue
            if key not in hard_tokens:
                continue
            pred = hard_tokens[key]
            target = gt_tokens[key].long()
            correct += (pred == target).sum().item()
            total += target.numel()
        return correct / max(total, 1)
