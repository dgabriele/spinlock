"""MSE-led loss: Physics fidelity as primary objective.

This loss function implements the traditional physics-first training paradigm
where trajectory matching (L_traj) is the primary objective, and VQ-VAE
alignment losses serve as auxiliary regularizers.

Loss = λ_traj * L_traj + λ_commit * L_commit + λ_latent * L_latent
       ═══════════════
          PRIMARY

When to use MSE-led:
- Early-stage training to establish physics grounding
- When exact trajectory matching is critical
- Benchmarking against CNO baselines
- Physics fidelity is the primary metric

Example:
    >>> loss_fn = MSELedLoss(
    ...     lambda_traj=1.0,
    ...     lambda_commit=0.5,
    ...     lambda_latent=0.1,
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


class MSELedLoss(BaseNOALoss):
    """MSE-led training: L_traj primary, VQ alignment auxiliary.

    This is the physics-first training paradigm where NOA learns to
    match CNO ground-truth rollouts exactly. VQ-VAE commitment and
    latent losses serve as regularizers to ensure outputs are still
    expressible in the symbolic vocabulary.

    Attributes:
        lambda_traj: Weight for trajectory MSE loss (primary)
        lambda_commit: Weight for VQ commitment loss (auxiliary)
        lambda_latent: Weight for latent alignment loss (auxiliary)
        alignment: VQVAEAlignmentLoss for commitment/latent computation
    """

    def __init__(
        self,
        lambda_traj: float = 1.0,
        lambda_commit: float = 0.5,
        lambda_latent: float = 0.1,
        vqvae_alignment: Optional["VQVAEAlignmentLoss"] = None,
    ):
        """Initialize MSE-led loss function.

        Args:
            lambda_traj: Weight for trajectory MSE loss (default: 1.0)
            lambda_commit: Weight for VQ commitment loss (default: 0.5)
            lambda_latent: Weight for latent alignment loss (default: 0.1)
            vqvae_alignment: Optional VQVAEAlignmentLoss for commitment/latent
                           losses. If None, only trajectory loss is computed.
        """
        super().__init__()
        self.lambda_traj = lambda_traj
        self.lambda_commit = lambda_commit
        self.lambda_latent = lambda_latent
        self.alignment = vqvae_alignment

    def compute(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
        noa: Optional[nn.Module] = None,
    ) -> LossOutput:
        """Compute MSE-led loss components.

        Args:
            pred_trajectory: NOA predicted trajectory [B, T, C, H, W]
            target_trajectory: CNO ground truth trajectory [B, T, C, H, W]
            ic: Initial condition [B, C, H, W] (for feature extraction)
            noa: NOA backbone reference (for L_latent computation)

        Returns:
            LossOutput with:
            - total: λ_traj * L_traj + λ_commit * L_commit + λ_latent * L_latent
            - components: {'traj': L_traj, 'commit': L_commit, 'latent': L_latent}
            - metrics: Detached float values for logging
        """
        device = pred_trajectory.device

        # L_traj: Primary physics loss (trajectory matching)
        traj_loss = F.mse_loss(pred_trajectory, target_trajectory)

        # Initialize auxiliary losses to zero
        commit_loss = torch.tensor(0.0, device=device)
        latent_loss = torch.tensor(0.0, device=device)

        # Compute VQ alignment losses if alignment module available
        if self.alignment is not None:
            align_out = self.alignment.compute_losses(
                pred_trajectory=pred_trajectory,
                target_trajectory=target_trajectory,
                ic=ic,
            )
            commit_loss = align_out.get('commit', torch.tensor(0.0, device=device))
            latent_loss = align_out.get('latent', torch.tensor(0.0, device=device))

        # Weighted sum with L_traj as primary
        total = (
            self.lambda_traj * traj_loss
            + self.lambda_commit * commit_loss
            + self.lambda_latent * latent_loss
        )

        return LossOutput(
            total=total,
            components={
                'traj': traj_loss,
                'commit': commit_loss,
                'latent': latent_loss,
            },
            metrics={
                'traj': traj_loss.item(),
                'commit': commit_loss.item() if isinstance(commit_loss, torch.Tensor) else commit_loss,
                'latent': latent_loss.item() if isinstance(latent_loss, torch.Tensor) else latent_loss,
                'total': total.item(),
            },
        )

    @property
    def leading_loss_name(self) -> str:
        """Primary loss is trajectory MSE."""
        return "traj"

    @property
    def auxiliary_loss_names(self) -> list[str]:
        """Auxiliary losses are commitment and latent alignment."""
        return ["commit", "latent"]

    def __repr__(self) -> str:
        return (
            f"MSELedLoss("
            f"λ_traj={self.lambda_traj}, "
            f"λ_commit={self.lambda_commit}, "
            f"λ_latent={self.lambda_latent})"
        )
