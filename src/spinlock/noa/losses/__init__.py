"""NOA loss functions for training.

This module provides loss functions for training Neural Operator Agents (NOA)
with different training paradigms:

MSE-led (MSELedLoss):
    Physics fidelity first. L_traj is primary, VQ alignment is auxiliary.
    Use when exact trajectory matching is critical.

VQ-led (VQLedLoss):
    Symbolic coherence first. L_recon is primary, L_traj is auxiliary.
    Use for "creative observer" training where meaningful deviation is allowed.

Both losses inherit from BaseNOALoss and return standardized LossOutput.

Example:
    >>> from spinlock.noa.losses import MSELedLoss, VQLedLoss
    >>>
    >>> # Physics-first training
    >>> mse_loss = MSELedLoss(lambda_traj=1.0, lambda_commit=0.5)
    >>>
    >>> # Creative observer training
    >>> vq_loss = VQLedLoss(
    ...     lambda_recon=1.0, lambda_commit=0.5, lambda_traj=0.3,
    ...     vqvae_alignment=alignment,
    ... )
"""

from spinlock.noa.losses.mse_led import MSELedLoss
from spinlock.noa.losses.vq_led import VQLedLoss

__all__ = [
    "MSELedLoss",
    "VQLedLoss",
]
