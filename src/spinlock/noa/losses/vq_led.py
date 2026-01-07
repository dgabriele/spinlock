"""VQ-led loss: VQ reconstruction/commitment as primary objective.

This loss function implements the "creative observer" training paradigm where
symbolic coherence (expressibility in VQ-VAE vocabulary) is the primary
objective, and trajectory matching serves as an auxiliary regularizer.

Loss = λ_recon * L_recon + λ_commit * L_commit + λ_traj * L_traj
       ════════════════
          PRIMARY

Philosophy:
    The Neural Operator Agent (NOA) is fundamentally a creative observer of
    dynamical systems. Rather than merely simulating trajectories with rigid
    fidelity to ground-truth rollouts, NOA generates its own "ideas" of
    evolution—pathways that may diverge from observed reality but remain
    coherent and meaningful when translated into VQ-VAE behavioral tokens.

    A "wrong" rollout by traditional MSE metrics may represent a novel
    perspective on dynamics—much like how different neural activation patterns
    in human brains converge on shared concepts despite varying implementations.

When to use VQ-led:
- After physics grounding is established (not for cold start)
- When symbolic interpretation matters more than exact matching
- Training for downstream reasoning with tokens
- Encouraging "creative" exploration of dynamics
- When VQ token quality is the primary metric

Example:
    >>> loss_fn = VQLedLoss(
    ...     lambda_recon=1.0,
    ...     lambda_commit=0.5,
    ...     lambda_traj=0.3,
    ...     vqvae_alignment=alignment,
    ... )
    >>> output = loss_fn.compute(pred_trajectory, target_trajectory, ic, noa)
    >>> output.total.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

from spinlock.noa.base_loss import BaseNOALoss, LossOutput

if TYPE_CHECKING:
    from spinlock.noa.vqvae_alignment import VQVAEAlignmentLoss


class VQLedLoss(BaseNOALoss):
    """VQ-led training: VQ quality primary, L_traj auxiliary.

    This is the "creative observer" training paradigm where NOA learns to
    generate outputs that are meaningful in the VQ-VAE behavioral vocabulary,
    even if they diverge from ground truth. Physics matching (L_traj) serves
    as a regularizer preventing complete drift from reality.

    The Surprise Principle:
        In VQ-led training, deviations from ground truth become
        opportunities for discovery:

        | Traditional View | Creative Observer View |
        |-----------------|----------------------|
        | High MSE = bad model | High MSE = novel perspective |
        | Match CNO exactly | Generate valid symbolic sequences |
        | Penalize deviation | Embrace meaningful divergence |

    Attributes:
        lambda_recon: Weight for VQ reconstruction loss (primary)
        lambda_commit: Weight for VQ commitment loss (primary)
        lambda_traj: Weight for trajectory MSE loss (auxiliary regularizer)
        alignment: VQVAEAlignmentLoss for VQ encoding/decoding
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_commit: float = 0.5,
        lambda_traj: float = 0.3,
        vqvae_alignment: Optional["VQVAEAlignmentLoss"] = None,
    ):
        """Initialize VQ-led loss function.

        Args:
            lambda_recon: Weight for VQ reconstruction loss (default: 1.0)
            lambda_commit: Weight for VQ commitment loss (default: 0.5)
            lambda_traj: Weight for trajectory MSE loss (default: 0.3)
                        Note: Lower than MSE-led to allow creative deviation
            vqvae_alignment: VQVAEAlignmentLoss for VQ encoding/decoding.
                           Required for VQ-led mode.

        Raises:
            ValueError: If vqvae_alignment is None
        """
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_commit = lambda_commit
        self.lambda_traj = lambda_traj
        self.alignment = vqvae_alignment

        if vqvae_alignment is None:
            raise ValueError(
                "VQLedLoss requires vqvae_alignment. "
                "The creative observer paradigm needs VQ-VAE for symbolic grounding."
            )

    def compute(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
        noa: Optional[nn.Module] = None,
    ) -> LossOutput:
        """Compute VQ-led loss components.

        The key insight is that we evaluate NOA on whether its outputs are
        "expressible" in the VQ vocabulary (L_recon), not just accurate to
        physics (L_traj). A high-quality reconstruction means the NOA trajectory
        is symbolically coherent.

        Args:
            pred_trajectory: NOA predicted trajectory [B, T, C, H, W]
            target_trajectory: CNO ground truth trajectory [B, T, C, H, W]
            ic: Initial condition [B, C, H, W] (for feature extraction)
            noa: NOA backbone reference (unused in this loss)

        Returns:
            LossOutput with:
            - total: λ_recon * L_recon + λ_commit * L_commit + λ_traj * L_traj
            - components: {'recon': L_recon, 'commit': L_commit, 'traj': L_traj}
            - metrics: Detached float values for logging
        """
        device = pred_trajectory.device

        # Type narrowing: alignment is guaranteed non-None by __init__
        assert self.alignment is not None

        # Extract features from predicted trajectory
        pred_result = self.alignment.feature_extractor(pred_trajectory, ic=ic)

        if isinstance(pred_result, tuple):
            pred_features, pred_raw_ics = pred_result
        else:
            pred_features = pred_result
            pred_raw_ics = ic

        # Normalize features using alignment's normalization
        features_norm = self.alignment._normalize_features(pred_features)

        # Encode to pre-quantization latents
        if self.alignment._is_hybrid_model and pred_raw_ics is not None:
            z_list = self.alignment.vqvae.encode(features_norm, raw_ics=pred_raw_ics)
        else:
            z_list = self.alignment.vqvae.encode(features_norm)

        z = torch.cat(z_list, dim=1)  # [B, total_latent_dim]

        # Quantize to discrete codes
        z_q_list, indices, _ = self.alignment.vqvae.quantize(z_list)
        z_q = torch.cat(z_q_list, dim=1)

        # L_recon: VQ reconstruction quality (PRIMARY)
        # Decode quantized latents back to feature space
        # This measures how well the trajectory is "expressible" in VQ vocabulary
        recon_features = self.alignment.vqvae.decode(z_q_list)
        recon_loss = F.mse_loss(recon_features, features_norm)

        # L_commit: Commitment loss (embedding sharpness)
        # Forces pre-quantization latents close to quantized codes
        commit_loss = F.mse_loss(z, z_q.detach())

        # L_traj: Physics regularizer (AUXILIARY)
        # Prevents complete drift from physical reality
        # Lower weight (0.3) allows creative deviation while maintaining grounding
        traj_loss = F.mse_loss(pred_trajectory, target_trajectory)

        # Weighted sum with L_recon and L_commit as primary
        total = (
            self.lambda_recon * recon_loss
            + self.lambda_commit * commit_loss
            + self.lambda_traj * traj_loss
        )

        return LossOutput(
            total=total,
            components={
                'recon': recon_loss,
                'commit': commit_loss,
                'traj': traj_loss,
            },
            metrics={
                'recon': recon_loss.item(),
                'commit': commit_loss.item(),
                'traj': traj_loss.item(),
                'total': total.item(),
            },
        )

    @property
    def leading_loss_name(self) -> str:
        """Primary loss is VQ reconstruction quality."""
        return "recon"

    @property
    def auxiliary_loss_names(self) -> list[str]:
        """Auxiliary losses are commitment and trajectory matching."""
        return ["commit", "traj"]

    def __repr__(self) -> str:
        return (
            f"VQLedLoss("
            f"λ_recon={self.lambda_recon}, "
            f"λ_commit={self.lambda_commit}, "
            f"λ_traj={self.lambda_traj})"
        )
