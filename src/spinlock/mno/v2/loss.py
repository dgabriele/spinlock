"""TrajectoryLoss: trajectory MSE + IC MSE + contrastive + token CE for v2 MNO training.

Gradient hierarchy:
  1. Trajectory MSE — direct pixel-level supervision on BPTT-aligned window
  2. IC MSE — ensures warmup endpoint is close to GT
  3. Soft token contrastive — KL(pred_sim || GT_jaccard) teaches behavioral topology
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
from spinlock.mno.losses.components.contrastive import SoftTokenContrastiveLoss

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
    _EMA_KEYS = ("traj_mse", "ic_mse", "contrastive", "feat_mse", "token_ce", "centroid_mse", "token_head")

    def __init__(
        self,
        contrastive_loss: SoftTokenContrastiveLoss,
        lambda_traj: float = 1.0,
        lambda_ic: float = 0.3,
        lambda_contrastive: float = 0.3,
        lambda_feat_mse: float = 0.0,
        lambda_token_ce: float = 0.0,
        lambda_centroid_mse: float = 0.0,
        lambda_token_head: float = 0.0,
        token_ce_temperature: float = 1.0,
        gate_weight_token_ce: bool = False,
        normalize_loss_scales: bool = False,
        loss_scale_ema_momentum: float = 0.99,
        vq_adapter: Optional[nn.Module] = None,
        token_pred_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self._contrastive = contrastive_loss
        self._lambda_traj = lambda_traj
        self._lambda_ic = lambda_ic
        self._lambda_contrastive = lambda_contrastive
        self._lambda_feat_mse = lambda_feat_mse
        self._lambda_token_ce = lambda_token_ce
        self._lambda_centroid_mse = lambda_centroid_mse
        self._lambda_token_head = lambda_token_head
        self._token_ce_temperature = token_ce_temperature
        self._gate_weight_token_ce = gate_weight_token_ce
        self._normalize = normalize_loss_scales
        self._ema_momentum = loss_scale_ema_momentum
        self._vq_adapter = vq_adapter

        # Learned token prediction head (nn.Module attribute assignment
        # auto-registers as submodule for optimizer and checkpoint save/load)
        self._token_pred_head: Optional[nn.Module] = None
        if token_pred_head is not None:
            self._token_pred_head = token_pred_head

        # EMA buffers for loss-scale normalization (persisted in checkpoint)
        for key in self._EMA_KEYS:
            self.register_buffer(f"ema_{key}", torch.tensor(1.0))

        logger.info(
            "TrajectoryLoss: λ_traj=%.2f, λ_ic=%.2f, λ_contrastive=%.2f, "
            "λ_feat_mse=%.2f, λ_token_ce=%.2f, λ_centroid_mse=%.2f, "
            "λ_token_head=%.2f, normalize=%s, vq_adapter=%s",
            lambda_traj, lambda_ic, lambda_contrastive, lambda_feat_mse,
            lambda_token_ce, lambda_centroid_mse, lambda_token_head,
            normalize_loss_scales,
            "yes" if vq_adapter is not None else "no",
        )

    @property
    def leading_loss_name(self) -> str:
        if self._lambda_token_head > 0:
            return "token_head"
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
        if self._lambda_centroid_mse > 0:
            names.append("centroid_mse")
        if self._lambda_token_head > 0:
            names.append("token_head_acc")
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
        gt_indicators: Optional[Tensor] = None,
    ) -> LossOutput:
        """Compute trajectory-first loss.

        Args:
            pred_trajectory: BPTT-aligned predicted states [B, W, C, H, W].
            target_trajectory: BPTT-aligned GT states [B, W, C, H, W].
            params: Parameter vectors [B, P] (unused by soft contrastive,
                kept for backward compat with old ContrastiveLoss).
            gt_raw_features: Optional GT temporal features [B, T, D_raw].
            gt_tokens: Optional pretokenized GT tokens {quantizer_key: [B]}.
                Used with vq_adapter for soft VQ cross-entropy loss.
            gt_indicators: Optional binary token indicators [B, indicator_dim]
                for soft contrastive loss (Jaccard similarity targets).

        Returns:
            LossOutput with total loss and per-component metrics.
        """
        device = pred_trajectory.device
        components: Dict[str, Tensor] = {}
        metrics: Dict[str, float] = {}

        # --- Trajectory MSE ---
        traj_mse = torch.tensor(0.0, device=device)
        if target_trajectory is not None and self._lambda_traj > 0:
            traj_mse = F.mse_loss(pred_trajectory, target_trajectory)
        components["traj_mse"] = traj_mse
        if self._lambda_traj > 0:
            metrics["traj_mse"] = traj_mse.item()

        # --- IC MSE (warmup endpoint accuracy) ---
        ic_mse = torch.tensor(0.0, device=device)
        if target_trajectory is not None and self._lambda_ic > 0:
            ic_mse = F.mse_loss(pred_trajectory[:, 0], target_trajectory[:, 0])
        components["ic_mse"] = ic_mse
        if self._lambda_ic > 0:
            metrics["ic_mse"] = ic_mse.item()

        # --- Contrastive (soft token Jaccard targets) ---
        contrastive_loss = torch.tensor(0.0, device=device)
        contrastive_metrics: Dict[str, float] = {}
        if self._lambda_contrastive > 0 and gt_indicators is not None:
            if self._vq_adapter is not None:
                pred_features = self._vq_adapter.extract_features(pred_trajectory)
                # Aggregate temporal dim: [B, T, D] -> [B, D]
                if pred_features.ndim == 3:
                    pred_features = pred_features.mean(dim=1)
                contrastive_out = self._contrastive(
                    pred_features, gt_indicators.to(device),
                )
                contrastive_loss = contrastive_out["loss"]
                contrastive_metrics = {
                    "mean_jaccard": contrastive_out["mean_jaccard"].item(),
                    "rank_corr": contrastive_out["rank_correlation"].item(),
                }
        components["contrastive"] = contrastive_loss
        if self._lambda_contrastive > 0:
            metrics["contrastive"] = contrastive_loss.item()
            metrics.update(contrastive_metrics)

        # --- Feature MSE (optional, disabled by default) ---
        # Two paths for GT features:
        #   1. Pyramid-first / learned mode: extract from GT trajectory directly
        #      (no pre-computed gt_raw_features needed)
        #   2. Manual mode: use pre-computed gt_raw_features from dataset
        feat_mse = torch.tensor(0.0, device=device)
        if self._vq_adapter is not None and self._lambda_feat_mse > 0:
            can_extract_from_gt_traj = (
                self._vq_adapter._pyramid_first_mode
                or self._vq_adapter._learned_mode
            )
            if can_extract_from_gt_traj and target_trajectory is not None:
                # Extract features from both pred and GT trajectories
                pred_feat = self._vq_adapter.extract_features(pred_trajectory)
                with torch.no_grad():
                    gt_feat = self._vq_adapter.extract_features(
                        target_trajectory,
                    )
                feat_mse = F.mse_loss(pred_feat, gt_feat)
            elif gt_raw_features is not None:
                # Manual mode: use pre-computed raw features from dataset
                pred_feat = self._vq_adapter.extract_features(pred_trajectory)
                gt_clean = self._vq_adapter.clean_raw_features(
                    gt_raw_features.to(device),
                ).detach()
                T_match = min(pred_feat.shape[1], gt_clean.shape[1])
                feat_mse = F.mse_loss(
                    pred_feat[:, :T_match], gt_clean[:, :T_match],
                )
        components["feat_mse"] = feat_mse
        if self._lambda_feat_mse > 0:
            metrics["feat_mse"] = feat_mse.item()

        # --- Token CE (optional, QA mode soft VQ cross-entropy) ---
        token_ce = torch.tensor(0.0, device=device)
        token_match_acc = 0.0
        result = None  # shared with centroid MSE below
        if (
            self._vq_adapter is not None
            and gt_tokens is not None
            and self._lambda_token_ce > 0
        ):
            # Unified pipeline: handles pyramid-first, learned CNN, and manual modes
            result = self._vq_adapter.extract_soft_logits_from_trajectory(
                pred_trajectory, temperature=self._token_ce_temperature,
            )
            # Gate-weighted CE: weight each group's CE by its frozen gate value
            gate_values = (
                self._vq_adapter.get_gate_values()
                if self._gate_weight_token_ce else None
            )
            token_ce = self._temporal_token_ce(
                result["soft_logits"], gt_tokens, gate_values=gate_values,
            )
            token_match_acc = self._temporal_token_accuracy(
                result["hard_tokens"], gt_tokens,
            )
        components["token_ce"] = token_ce
        if self._lambda_token_ce > 0:
            metrics["token_ce"] = token_ce.item()
            metrics["token_match_acc"] = token_match_acc

        # --- Centroid MSE (optional, VQ centroid supervision) ---
        centroid_mse = torch.tensor(0.0, device=device)
        if (
            self._vq_adapter is not None
            and gt_tokens is not None
            and self._lambda_centroid_mse > 0
        ):
            # Pred latents: reuse from token CE if available, else run encoder+projectors
            if result is not None and result.get('projected_latents') is not None:
                pred_latents = result['projected_latents']
            else:
                pred_latents = self._vq_adapter.extract_projected_latents(pred_trajectory)

            # GT centroids: cheap codebook lookup, no encoder pass needed
            gt_centroids = self._vq_adapter.lookup_gt_centroids(gt_tokens)

            # Per-quantizer MSE, uniform average across all temporal quantizers
            mse_terms = []
            for qkey in sorted(gt_centroids.keys()):
                if qkey not in pred_latents:
                    continue
                mse_terms.append(F.mse_loss(pred_latents[qkey], gt_centroids[qkey].detach()))
            if mse_terms:
                centroid_mse = torch.stack(mse_terms).mean()

        components["centroid_mse"] = centroid_mse
        if self._lambda_centroid_mse > 0:
            metrics["centroid_mse"] = centroid_mse.item()

        # --- Token head CE (optional, learned bypass of frozen VQ encoder) ---
        token_head_loss = torch.tensor(0.0, device=device)
        token_head_acc = 0.0
        if (
            self._token_pred_head is not None
            and self._lambda_token_head > 0
            and gt_tokens is not None
        ):
            # Detach: head learns the backbone's representation without
            # corrupting it.  Backbone gradients come from token_ce (soft
            # logits through frozen VQ encoder) which is already differentiable.
            head_logits = self._token_pred_head(pred_trajectory.detach())
            token_head_loss = self._temporal_token_ce(head_logits, gt_tokens)
            with torch.no_grad():
                head_hard = {k: v.argmax(dim=1) for k, v in head_logits.items()}
                token_head_acc = self._temporal_token_accuracy(head_hard, gt_tokens)
        components["token_head"] = token_head_loss
        if self._lambda_token_head > 0:
            metrics["token_head"] = token_head_loss.item()
            metrics["token_head_acc"] = token_head_acc

        # --- Normalize loss scales (optional) ---
        # When enabled, each L_i is divided by EMA(L_i) so lambdas reflect
        # true gradient importance rather than accidental scale differences.
        traj_scaled = self._scale_loss("traj_mse", traj_mse)
        ic_scaled = self._scale_loss("ic_mse", ic_mse)
        contrastive_scaled = self._scale_loss("contrastive", contrastive_loss)
        feat_scaled = self._scale_loss("feat_mse", feat_mse)
        token_ce_scaled = self._scale_loss("token_ce", token_ce)
        centroid_mse_scaled = self._scale_loss("centroid_mse", centroid_mse)
        token_head_scaled = self._scale_loss("token_head", token_head_loss)

        # --- Combined ---
        total = (
            self._lambda_traj * traj_scaled
            + self._lambda_ic * ic_scaled
            + self._lambda_contrastive * contrastive_scaled
            + self._lambda_feat_mse * feat_scaled
            + self._lambda_token_ce * token_ce_scaled
            + self._lambda_centroid_mse * centroid_mse_scaled
            + self._lambda_token_head * token_head_scaled
        )
        metrics["total"] = total.item()

        # Log effective contributions when normalizing (only enabled losses)
        if self._normalize:
            if self._lambda_traj > 0:
                metrics["eff_traj"] = (self._lambda_traj * traj_scaled).item()
            if self._lambda_token_ce > 0:
                metrics["eff_token_ce"] = (self._lambda_token_ce * token_ce_scaled).item()
            if self._lambda_centroid_mse > 0:
                metrics["eff_centroid_mse"] = (self._lambda_centroid_mse * centroid_mse_scaled).item()
            if self._lambda_token_head > 0:
                metrics["eff_token_head"] = (self._lambda_token_head * token_head_scaled).item()

        return LossOutput(total=total, components=components, metrics=metrics)

    @staticmethod
    def _temporal_token_ce(
        soft_logits: Dict[str, Tensor],
        gt_tokens: Dict[str, Tensor],
        gate_values: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-quantizer cross-entropy averaged over temporal families only.

        When ``gate_values`` is provided, each group's CE is weighted by its
        frozen gate activation (groups the tokenizer deemed unimportant
        contribute less to the MNO's token matching objective).

        Args:
            soft_logits: {quantizer_key: [B, K]} negative L2 distances / temp
            gt_tokens: {quantizer_key: [B]} hard GT token indices
            gate_values: Optional [num_groups] gate activations in [0,1].
                Parsed from quantizer key "temporal_group_X_LY" → group X.

        Returns:
            Scalar (weighted) mean CE loss across all temporal quantizers.
        """
        losses = []
        weights = []
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
                ce = F.cross_entropy(logit[valid], target[valid])
                losses.append(ce)
                if gate_values is not None:
                    # "temporal_group_X_LY" → X is the group index
                    group_idx = int(key.split('_')[2])
                    weights.append(gate_values[group_idx].detach())
        if not losses:
            return torch.tensor(0.0, device=next(iter(soft_logits.values())).device)
        loss_stack = torch.stack(losses)
        if weights:
            w = torch.stack(weights)
            return (w * loss_stack).sum() / w.sum().clamp(min=1e-8)
        return loss_stack.mean()

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
