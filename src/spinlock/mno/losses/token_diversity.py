"""Token diversity loss: contrastive behavioral diversity as primary objective.

Trains MNO as a diverse trajectory generator where different parameters produce
distinguishable rollouts, rather than optimizing for exact QBM emulation.

Loss = λ_contrastive * L_contrastive    ← PRIMARY: behavioral diversity
     + λ_recon * L_recon                ← OPTIONAL: VQ feature reconstruction
     + λ_commit * L_commit              ← OPTIONAL: VQ manifold adherence
     + λ_traj * L_traj                  ← OPTIONAL: physics regularizer (0 = skip)
     + λ_ic * L_ic                      ← OPTIONAL: IC regularizer (0 = skip)

When lambda_traj=0 and lambda_ic=0, exposes needs_target_trajectory=False so the
training loop can skip the expensive QBM replayer rollout entirely.

VQ coherence (recon + commit) requires a VQCoherenceAdapter instance, which wraps
the modern VQTokenizer/JointHierarchicalVQVAE architecture. When vq_adapter is
provided, the contrastive loss operates in VQ tokenizer feature space (D_clean
temporal features) instead of the generic 32D RolloutFeatureExtractor space.

Design:
- Contrastive loss is always available (primary training signal)
- VQ coherence is optional (requires VQCoherenceAdapter)
- When VQ adapter is present, contrastive uses VQ feature space (better alignment)
- Inherits BaseNOALoss for training loop compatibility
- Token set diversity metric for monitoring codebook coverage

Documentation:
    - Design rationale: docs/mno-diverse-dreamer-analysis.md
    - Contrastive component: src/spinlock/mno/losses/components/contrastive.py
    - VQ coherence adapter: src/spinlock/mno/vq_coherence.py
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, List, TYPE_CHECKING

logger = logging.getLogger(__name__)

from spinlock.mno.base_loss import BaseNOALoss, LossOutput
from spinlock.mno.losses.components import ContrastiveLoss

if TYPE_CHECKING:
    from spinlock.mno.vq_coherence import VQCoherenceAdapter
    from spinlock.mno.losses.roundtrip_consistency import RoundtripConsistencyLoss
    from spinlock.mno.losses.components.token_contrastive import TokenContrastiveLoss


class TokenDiversityLoss(BaseNOALoss):
    """MNO Diverse Dreamer: contrastive diversity primary, VQ + physics optional.

    The contrastive loss (InfoNCE) ensures different parameters produce
    distinguishable trajectories — the key requirement for downstream token
    synthesis. VQ coherence losses can be added when a VQCoherenceAdapter
    is provided.

    When a VQ adapter is present, the contrastive loss operates in the same
    temporal feature space the tokenizer uses (D_clean features), instead of
    the generic 32D RolloutFeatureExtractor space. This directly optimizes
    for tokenization distinguishability.

    Attributes:
        lambda_contrastive: Weight for InfoNCE contrastive loss (default: 1.0)
        lambda_recon: Weight for VQ reconstruction loss (default: 0.0)
        lambda_commit: Weight for VQ commitment loss (default: 0.0)
        lambda_traj: Weight for trajectory MSE (default: 0.0 = disabled)
        lambda_ic: Weight for IC preservation MSE (default: 0.0 = disabled)
        vq_adapter: Optional VQCoherenceAdapter for frozen VQ pipeline
        contrastive_loss: ContrastiveLoss for InfoNCE
    """

    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_recon: float = 0.0,
        lambda_commit: float = 0.0,
        lambda_traj: float = 0.0,
        lambda_ic: float = 0.0,
        lambda_roundtrip: float = 0.0,
        lambda_token_contrastive: float = 0.0,
        # VQ coherence adapter (replaces old vqvae_alignment)
        vq_adapter: Optional["VQCoherenceAdapter"] = None,
        # Roundtrip consistency loss (optional)
        roundtrip_loss: Optional["RoundtripConsistencyLoss"] = None,
        roundtrip_enable_batch: int = 0,
        roundtrip_warmup_batches: int = 0,
        # Token contrastive loss (optional)
        token_contrastive_loss: Optional["TokenContrastiveLoss"] = None,
        token_contrastive_enable_batch: int = 0,
        token_contrastive_warmup_batches: int = 0,
        # Contrastive config
        param_dim: int = 14,
        contrastive_embed_dim: int = 128,
        contrastive_hidden_dim: int = 256,
        contrastive_temperature: float = 0.1,
        contrastive_queue_size: int = 0,
        contrastive_feature_dim: Optional[int] = None,
    ):
        """Initialize token diversity loss.

        Args:
            lambda_contrastive: Weight for InfoNCE contrastive loss (default: 1.0)
            lambda_recon: Weight for VQ reconstruction loss (default: 0.0)
            lambda_commit: Weight for VQ commitment loss (default: 0.0)
            lambda_traj: Weight for trajectory MSE (default: 0.0 = disabled)
            lambda_ic: Weight for IC preservation MSE (default: 0.0 = disabled)
            lambda_roundtrip: Weight for roundtrip token consistency loss (default: 0.0)
            lambda_token_contrastive: Weight for token-space contrastive loss (default: 0.0)
            vq_adapter: Optional VQCoherenceAdapter. Required when
                       lambda_recon > 0 or lambda_commit > 0.
            roundtrip_loss: Optional RoundtripConsistencyLoss instance
            roundtrip_enable_batch: Batch number to start roundtrip loss (curriculum)
            roundtrip_warmup_batches: Batches for cosine warmup to full lambda
            token_contrastive_loss: Optional TokenContrastiveLoss instance
            token_contrastive_enable_batch: Batch number to start token contrastive (curriculum)
            token_contrastive_warmup_batches: Batches for cosine warmup to full lambda
            param_dim: Dimensionality of parameter vector (auto-detected from dataset)
            contrastive_embed_dim: Embedding dim for contrastive projectors
            contrastive_hidden_dim: Hidden dim for contrastive projectors
            contrastive_temperature: Temperature for InfoNCE softmax
            contrastive_queue_size: MoCo-style memory bank size (default: 0).
                        When > 0, stores recent embeddings as extra negatives.
                        Essential for small-batch contrastive (B=2-4).
            contrastive_feature_dim: VQ tokenizer cleaned feature dim (D_clean).
                        When provided, creates a feature-space projector in
                        ContrastiveLoss. Set to vq_adapter.cleaned_feature_dim.

        Raises:
            ValueError: If VQ lambdas > 0 but no vq_adapter provided
        """
        super().__init__()

        self.vq_adapter = vq_adapter
        self.lambda_contrastive = lambda_contrastive
        self.lambda_recon = lambda_recon
        self.lambda_commit = lambda_commit
        self.lambda_traj = lambda_traj
        self.lambda_ic = lambda_ic
        self.lambda_roundtrip = lambda_roundtrip

        # Roundtrip consistency
        self.roundtrip_loss = roundtrip_loss
        self._roundtrip_enable_batch = roundtrip_enable_batch
        self._roundtrip_warmup_batches = roundtrip_warmup_batches
        self._current_batch = 0

        # Token contrastive
        self.lambda_token_contrastive = lambda_token_contrastive
        self.token_contrastive_loss = token_contrastive_loss
        self._token_contrastive_enable_batch = token_contrastive_enable_batch
        self._token_contrastive_warmup_batches = token_contrastive_warmup_batches

        # Validate: VQ losses require adapter
        if (lambda_recon > 0 or lambda_commit > 0) and vq_adapter is None:
            raise ValueError(
                "lambda_recon > 0 or lambda_commit > 0 requires vq_adapter. "
                "Either provide a VQCoherenceAdapter or set lambda_recon=0, lambda_commit=0."
            )

        if lambda_roundtrip > 0 and roundtrip_loss is None:
            raise ValueError(
                "lambda_roundtrip > 0 requires roundtrip_loss. "
                "Either provide a RoundtripConsistencyLoss or set lambda_roundtrip=0."
            )

        if lambda_token_contrastive > 0 and token_contrastive_loss is None:
            raise ValueError(
                "lambda_token_contrastive > 0 requires token_contrastive_loss. "
                "Either provide a TokenContrastiveLoss or set lambda_token_contrastive=0."
            )

        # Contrastive component (has learnable projectors)
        self.contrastive_loss = ContrastiveLoss(
            param_dim=param_dim,
            embed_dim=contrastive_embed_dim,
            hidden_dim=contrastive_hidden_dim,
            temperature=contrastive_temperature,
            queue_size=contrastive_queue_size,
            feature_dim=contrastive_feature_dim,
        )

        # Track whether to use feature-space contrastive
        self._use_feature_contrastive = (
            vq_adapter is not None and contrastive_feature_dim is not None
        )

    @property
    def needs_target_trajectory(self) -> bool:
        """Whether this loss needs QBM replayer target trajectory.

        When False, the training loop can skip the expensive replayer rollout.
        This is the major speed advantage of pure diversity mode.
        """
        return self.lambda_traj > 0

    def set_batch(self, batch_number: int) -> None:
        """Update current batch counter for curriculum scheduling."""
        self._current_batch = batch_number

    def _get_roundtrip_lambda(self) -> float:
        """Get effective roundtrip lambda with curriculum scheduling.

        Returns 0 before enable_batch, then cosine-ramps from 0 to
        lambda_roundtrip over warmup_batches.
        """
        if self.lambda_roundtrip <= 0 or self.roundtrip_loss is None:
            return 0.0

        if self._current_batch < self._roundtrip_enable_batch:
            return 0.0

        elapsed = self._current_batch - self._roundtrip_enable_batch
        if self._roundtrip_warmup_batches > 0 and elapsed < self._roundtrip_warmup_batches:
            # Cosine warmup: small_start → lambda_roundtrip
            # Use (elapsed+1)/(N+1) so batch 0 gives nonzero lambda
            # (pure cosine with progress=0 gives exactly 0, which breaks
            # roundtrip-only mode where this is the sole gradient source)
            progress = (elapsed + 1) / (self._roundtrip_warmup_batches + 1)
            scale = 0.5 * (1 - math.cos(math.pi * progress))
            return self.lambda_roundtrip * scale

        return self.lambda_roundtrip

    def _get_token_contrastive_lambda(self) -> float:
        """Get effective token contrastive lambda with curriculum scheduling.

        Same cosine warmup pattern as roundtrip lambda.
        """
        if self.lambda_token_contrastive <= 0 or self.token_contrastive_loss is None:
            return 0.0

        if self._current_batch < self._token_contrastive_enable_batch:
            return 0.0

        elapsed = self._current_batch - self._token_contrastive_enable_batch
        if self._token_contrastive_warmup_batches > 0 and elapsed < self._token_contrastive_warmup_batches:
            progress = (elapsed + 1) / (self._token_contrastive_warmup_batches + 1)
            scale = 0.5 * (1 - math.cos(math.pi * progress))
            return self.lambda_token_contrastive * scale

        return self.lambda_token_contrastive

    def compute(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: Optional[torch.Tensor] = None,
        ic: Optional[torch.Tensor] = None,
        mno: Optional[nn.Module] = None,
        params: Optional[torch.Tensor] = None,
        gt_tokens: Optional[Dict[str, torch.Tensor]] = None,
    ) -> LossOutput:
        """Compute token diversity loss components.

        Args:
            pred_trajectory: MNO predicted trajectory [B, T, C, H, W]
            target_trajectory: Ground truth trajectory (optional, only needed when
                             lambda_traj > 0). Can be None when replayer is skipped.
            ic: Initial condition [B, C, H, W] (for feature extraction)
            mno: MNO backbone reference (unused in this loss)
            params: PDE parameters [B, param_dim] (required for contrastive loss)
            gt_tokens: Pretokenized GT tokens {key: [B] long} for roundtrip loss
                      Mode A. If None and roundtrip enabled, falls back to Mode B.

        Returns:
            LossOutput with total, components, and monitoring metrics.
        """
        device = pred_trajectory.device

        # ── 0. Sanitize trajectory ────────────────────────────────────────
        # Untrained/divergent MNOs can produce NaN/Inf. Replace with zeros so
        # all loss components get finite values. nan_to_num is differentiable
        # for non-NaN positions; NaN positions get zero gradient (strictly
        # better than skipping the entire batch).
        nan_mask = torch.isnan(pred_trajectory) | torch.isinf(pred_trajectory)
        if nan_mask.any():
            nan_frac = nan_mask.float().mean().item()
            if nan_frac > 0.5:
                logger.warning(
                    f"TokenDiversityLoss: {nan_frac:.1%} of trajectory values are "
                    f"NaN/Inf — MNO may be diverging"
                )
            pred_trajectory = torch.nan_to_num(
                pred_trajectory, nan=0.0, posinf=1e6, neginf=-1e6
            )

        # ── 1. VQ coherence + feature extraction (if adapter present) ─────
        # Do this FIRST so we can use cleaned_features for contrastive

        recon_loss = torch.tensor(0.0, device=device)
        commit_loss = torch.tensor(0.0, device=device)
        token_diversity = 0.0
        cleaned_features = None

        if self.vq_adapter is not None:
            # extract_features() preserves gradient through feature extractors
            # so contrastive loss can train MNO diversity in VQ feature space.
            # LayerNorm + ±100 clamping stabilize the gradient.
            cleaned_features = self.vq_adapter.extract_features(pred_trajectory)

            if self.lambda_recon > 0 or self.lambda_commit > 0:
                if self.lambda_commit > 0:
                    # Commit loss WITH gradient — pulls features toward VQ
                    # codebook entries for roundtrip tokenization consistency.
                    # VQ encoder path is all frozen MLPs (well-conditioned,
                    # no volatile ops). Recon stays detached (decoder + STE).
                    vq_metrics = self.vq_adapter.encode_and_quantize(
                        cleaned_features, params=params,
                    )
                    commit_loss = vq_metrics['vq_loss']
                    recon_loss = vq_metrics['recon_loss'].detach()

                    with torch.no_grad():
                        token_diversity = self._compute_token_diversity(
                            vq_metrics['token_indices']
                        )
                else:
                    # Recon monitoring only — no commit gradient needed
                    with torch.no_grad():
                        vq_metrics = self.vq_adapter.encode_and_quantize(
                            cleaned_features.detach(), params=params,
                        )
                        recon_loss = vq_metrics['recon_loss']
                        commit_loss = vq_metrics['vq_loss']
                        token_diversity = self._compute_token_diversity(
                            vq_metrics['token_indices']
                        )

        # ── 2. Contrastive diversity (PRIMARY) ──────────────────────────────

        contrastive_loss_val = torch.tensor(0.0, device=device)
        contrastive_accuracy = 0.0
        mean_positive_sim = 0.0
        mean_negative_sim = 0.0

        if self.lambda_contrastive > 0 and params is not None:
            # When VQ adapter provides cleaned features, use feature-space
            # contrastive (152D VQ features) — directly optimizes for
            # tokenization distinguishability. Gradient flows through
            # LayerNorm → feature extractors → trajectory → MNO.
            # Falls back to 32D RolloutFeatureExtractor when no VQ adapter.
            if self._use_feature_contrastive and cleaned_features is not None:
                contrastive_out = self.contrastive_loss.forward_from_features(
                    cleaned_features, params
                )
            else:
                contrastive_out = self.contrastive_loss(pred_trajectory, params)

            contrastive_loss_val = contrastive_out['loss']
            contrastive_accuracy = contrastive_out['accuracy'].item()
            mean_positive_sim = contrastive_out['mean_positive_sim'].item()
            mean_negative_sim = contrastive_out['mean_negative_sim'].item()

        # ── 3. Physics regularizers (skipped when lambda=0) ─────────────────

        traj_loss = torch.tensor(0.0, device=device)
        ic_loss = torch.tensor(0.0, device=device)

        if self.lambda_traj > 0 and target_trajectory is not None:
            traj_loss = F.mse_loss(pred_trajectory, target_trajectory)

        if self.lambda_ic > 0 and ic is not None:
            ic_loss = F.mse_loss(pred_trajectory[:, 0], ic)

        # ── 4. Token-space losses (roundtrip + token contrastive) ─────────
        # Both need soft_logits from the same VQ encoder pass. Extract once.

        roundtrip_loss_val = torch.tensor(0.0, device=device)
        token_agreement_rate = 0.0
        token_contrastive_val = torch.tensor(0.0, device=device)
        token_contrastive_accuracy = 0.0
        token_prob_entropy = 0.0

        effective_roundtrip_lambda = self._get_roundtrip_lambda()
        effective_tc_lambda = self._get_token_contrastive_lambda()

        # Compute soft logits ONCE if any token-space loss needs them
        needs_soft_logits = (
            (effective_roundtrip_lambda > 0 and self.roundtrip_loss is not None)
            or (effective_tc_lambda > 0 and self.token_contrastive_loss is not None)
        )
        soft_logits_out = None
        if needs_soft_logits and cleaned_features is not None and self.vq_adapter is not None:
            soft_logits_out = self.vq_adapter.extract_soft_logits_and_hard_tokens(
                cleaned_features, params=params, temperature=1.0,
            )

        # 4a. Roundtrip: pass precomputed soft logits
        if effective_roundtrip_lambda > 0 and soft_logits_out is not None:
            rt_out = self.roundtrip_loss.compute(
                cleaned_features, params=params, gt_tokens=gt_tokens,
                precomputed=soft_logits_out,
            )
            roundtrip_loss_val = rt_out['loss']
            token_agreement_rate = rt_out['token_agreement_rate']

        # 4b. Token contrastive: use soft_logits for diversity
        if effective_tc_lambda > 0 and soft_logits_out is not None and params is not None:
            tc_out = self.token_contrastive_loss(
                soft_logits_out['soft_logits'], params,
            )
            token_contrastive_val = tc_out['loss']
            token_contrastive_accuracy = tc_out['accuracy'].item()
            token_prob_entropy = tc_out['token_prob_entropy'].item()

        # ── 5. Total weighted loss ──────────────────────────────────────────

        total = (
            self.lambda_contrastive * contrastive_loss_val
            + self.lambda_recon * recon_loss
            + self.lambda_commit * commit_loss
            + self.lambda_traj * traj_loss
            + self.lambda_ic * ic_loss
            + effective_roundtrip_lambda * roundtrip_loss_val
            + effective_tc_lambda * token_contrastive_val
        )

        # ── 6. Monitoring metrics (detached) ────────────────────────────────

        metrics = {
            'feature_contrastive': contrastive_loss_val.item(),
            'recon': recon_loss.item(),
            'commit': commit_loss.item(),
            'traj': traj_loss.item(),
            'ic': ic_loss.item(),
            'roundtrip': roundtrip_loss_val.item(),
            'token_contrastive': token_contrastive_val.item(),
            'total': total.item(),
            'feature_contrastive_accuracy': contrastive_accuracy,
            'mean_positive_sim': mean_positive_sim,
            'mean_negative_sim': mean_negative_sim,
            'token_set_diversity': token_diversity,
            'token_agreement_rate': token_agreement_rate,
            'token_contrastive_accuracy': token_contrastive_accuracy,
            'token_prob_entropy': token_prob_entropy,
            'roundtrip_lambda_effective': effective_roundtrip_lambda,
            'token_contrastive_lambda_effective': effective_tc_lambda,
        }

        # Relative L2 (only when target available)
        if target_trajectory is not None:
            with torch.no_grad():
                target_norm = target_trajectory.norm()
                if target_norm > 1e-8:
                    rel_l2 = (pred_trajectory - target_trajectory).norm() / target_norm
                    metrics['relative_l2'] = rel_l2.item()

        return LossOutput(
            total=total,
            components={
                'feature_contrastive': contrastive_loss_val,
                'recon': recon_loss,
                'commit': commit_loss,
                'traj': traj_loss,
                'ic': ic_loss,
                'roundtrip': roundtrip_loss_val,
                'token_contrastive': token_contrastive_val,
            },
            metrics=metrics,
        )

    def _compute_token_diversity(self, indices) -> float:
        """Count unique token combinations across batch.

        Args:
            indices: Dict of quantizer_key → [B] token indices, or list of tensors

        Returns:
            Diversity ratio: unique_combos / batch_size (1.0 = all different)
        """
        if isinstance(indices, dict):
            token_list = [indices[k] for k in sorted(indices.keys())]
        elif isinstance(indices, (list, tuple)):
            token_list = list(indices)
        else:
            return 0.0

        if not token_list:
            return 0.0

        # Stack all tokens per sample → [B, num_quantizers]
        token_matrix = torch.stack(token_list, dim=1)
        B = token_matrix.shape[0]

        if B == 0:
            return 0.0

        unique = torch.unique(token_matrix, dim=0)
        return unique.shape[0] / B

    @property
    def leading_loss_name(self) -> str:
        """Primary loss is feature-space contrastive diversity."""
        return "feature_contrastive"

    @property
    def auxiliary_loss_names(self) -> List[str]:
        """Auxiliary losses: VQ coherence, physics regularizers, roundtrip, token contrastive."""
        return ["recon", "commit", "traj", "ic", "roundtrip", "token_contrastive"]

    def __repr__(self) -> str:
        vq_status = "with VQ adapter" if self.vq_adapter is not None else "no VQ"
        contrastive_space = "feature-space" if self._use_feature_contrastive else "raw-trajectory"
        rt_status = f", λ_roundtrip={self.lambda_roundtrip}" if self.lambda_roundtrip > 0 else ""
        tc_status = f", λ_token_contrastive={self.lambda_token_contrastive}" if self.lambda_token_contrastive > 0 else ""
        return (
            f"TokenDiversityLoss("
            f"λ_contrastive={self.lambda_contrastive}, "
            f"λ_recon={self.lambda_recon}, "
            f"λ_commit={self.lambda_commit}, "
            f"λ_traj={self.lambda_traj}, "
            f"λ_ic={self.lambda_ic}"
            f"{rt_status}{tc_status}, "
            f"{vq_status}, "
            f"contrastive={contrastive_space})"
        )
