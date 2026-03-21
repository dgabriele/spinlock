"""Clipped Gaussian perturbation in [0,1]^D Sobol space.

Generates nearby parameter vectors around a D3PM-decoded theta proposal.
Categorical dimensions (kernel_type, growth_type) are frozen by default
since small perturbations can flip category entirely.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spinlock.experimental.diffusion.config import PerturbationConfig


class LocalParameterPerturber:
    """Generate clipped Gaussian perturbations around center points in [0,1]^D."""

    def __init__(self, config: PerturbationConfig, sobol_dim: int = 34) -> None:
        self.config = config
        self.sobol_dim = sobol_dim

        # Build mask for mutable dimensions (True = perturb, False = freeze)
        self.mutable_mask = torch.ones(sobol_dim, dtype=torch.bool)
        if config.freeze_categorical_dims:
            for idx in config.categorical_dim_indices:
                if idx < sobol_dim:
                    self.mutable_mask[idx] = False

    def perturb(
        self,
        centers: Tensor,       # [B, D] decoded theta in [0,1]^D
        sigma: float,          # current perturbation radius
        n_per_center: int,     # perturbations per center
    ) -> Tensor:               # [B * n_per_center, D] clamped to [0,1]
        """Generate clipped Gaussian perturbations around each center.

        Args:
            centers: Center points in [0,1]^D (decoded from D3PM's best tokens).
            sigma: Gaussian std for perturbation.
            n_per_center: Number of perturbations per center point.

        Returns:
            Perturbed vectors [B * n_per_center, D] clamped to [0,1].
            Categorical dimensions are copied unchanged from centers.
        """
        B, D = centers.shape
        device = centers.device

        # Expand centers: [B, D] -> [B*N, D]
        expanded = centers.unsqueeze(1).expand(B, n_per_center, D).reshape(B * n_per_center, D)

        # Gaussian noise on mutable dimensions only
        noise = torch.zeros_like(expanded)
        mask = self.mutable_mask.to(device)
        noise[:, mask] = torch.randn(B * n_per_center, mask.sum().item(), device=device) * sigma

        # Add noise and clamp to [0,1]
        perturbed = (expanded + noise).clamp(0.0, 1.0)

        return perturbed
