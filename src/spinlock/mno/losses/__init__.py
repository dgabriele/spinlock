"""MNO loss functions for training.

This module provides loss functions for training MNO (Meta Neural Operator)
with different training paradigms:

MSE-led (MSELedLoss):
    Physics fidelity first. L_traj is primary, VQ alignment is auxiliary.
    Use when exact trajectory matching is critical.

VQ-led (VQLedLoss):
    Symbolic coherence first. L_recon/L_latent is primary, L_traj is auxiliary.
    Use for "creative observer" training where meaningful deviation is allowed.

Parameter-sensitive (ParameterSensitiveLoss):
    MSE + contrastive + param reconstruction + sensitivity regularization.
    Use to fix weak parameter conditioning.

Token diversity (TokenDiversityLoss):
    Contrastive diversity + VQ coherence primary, physics optional.
    Use when downstream token synthesis needs diverse vocabulary coverage.
    Can skip QBM replayer when physics regularizers are disabled (faster).

All losses inherit from BaseNOALoss and return standardized LossOutput.

Example:
    >>> from spinlock.mno.losses import MSELedLoss, VQLedLoss, TokenDiversityLoss
    >>>
    >>> # Physics-first training
    >>> mse_loss = MSELedLoss(lambda_traj=1.0, lambda_commit=0.5)
    >>>
    >>> # Creative observer training
    >>> vq_loss = VQLedLoss(
    ...     lambda_recon=1.0, lambda_commit=0.5, lambda_traj=0.3,
    ...     vqvae_alignment=alignment,
    ... )
    >>>
    >>> # Diverse dreamer training (replayer-skip when traj=0)
    >>> diversity_loss = TokenDiversityLoss(
    ...     vqvae_alignment=alignment,
    ...     lambda_contrastive=1.0, lambda_recon=1.0, lambda_commit=0.5,
    ... )
"""

from spinlock.mno.losses.mse_led import MSELedLoss
from spinlock.mno.losses.vq_led import VQLedLoss
from spinlock.mno.losses.parameter_sensitive import ParameterSensitiveLoss
from spinlock.mno.losses.token_diversity import TokenDiversityLoss
from spinlock.mno.losses.roundtrip_consistency import RoundtripConsistencyLoss

__all__ = [
    "MSELedLoss",
    "VQLedLoss",
    "ParameterSensitiveLoss",
    "TokenDiversityLoss",
    "RoundtripConsistencyLoss",
]
