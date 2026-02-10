"""Parameter sensitivity regularization.

This loss penalizes if parameter changes don't produce sufficient output changes.
Uses finite difference approximation to measure gradient of rollout with respect
to parameters, then regularizes sensitivity to match a target level.

Target: Parameter changes should cause ~10% of the variation caused by time

Design:
- Functional: Uses metrics.sensitivity.measure_parameter_sensitivity
- Regularization: Penalizes deviation from target sensitivity ratio
- Efficient: Single forward pass with perturbed parameters
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from spinlock.mno.metrics.sensitivity import measure_parameter_sensitivity


class SensitivityRegularization(nn.Module):
    """Regularize parameter sensitivity to target level.

    Measures how much rollouts change when parameters are perturbed, then
    penalizes if this sensitivity is too low (MNO ignoring parameters) or
    too high (MNO overly sensitive to parameter noise).

    Target: Parameter changes should cause variation similar to temporal dynamics
    (typically 10% of temporal variance = diversity ratio of 0.1)

    Attributes:
        epsilon: Finite difference step size for gradient estimation
        target_ratio: Target diversity ratio (param_var / temporal_var)
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        target_ratio: float = 0.1,
    ):
        """Initialize sensitivity regularization.

        Args:
            epsilon: Finite difference step size (default: 0.01 = 1% perturbation)
            target_ratio: Target diversity ratio (default: 0.1 = 10%)
        """
        super().__init__()
        self.epsilon = epsilon
        self.target_ratio = target_ratio

    def forward(
        self,
        mno: nn.Module,
        ic: torch.Tensor,
        params: torch.Tensor,
        rollout_steps: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """Compute sensitivity regularization loss.

        Args:
            mno: Meta Neural Operator (must have .rollout() method)
            ic: Initial condition [B, C, H, W]
            params: Parameters [B, param_dim]
            rollout_steps: Number of timesteps to generate (default: 256)

        Returns:
            Dictionary containing:
            - loss: Regularization loss (MSE between actual and target sensitivity)
            - diversity_ratio: Measured diversity ratio (detached)
            - param_variance: Variance across parameter perturbations (detached)
            - temporal_variance: Variance across time (detached)
        """
        # Measure parameter sensitivity using finite differences
        sensitivity_metrics = measure_parameter_sensitivity(
            mno=mno,
            ic=ic,
            params=params,
            epsilon=self.epsilon,
            rollout_steps=rollout_steps,
        )

        diversity_ratio = sensitivity_metrics['diversity_ratio']
        param_variance = sensitivity_metrics['param_variance']
        temporal_variance = sensitivity_metrics['temporal_variance']

        # Regularization loss: penalize deviation from target diversity ratio
        # Normalize by target to make loss scale-invariant
        target = torch.tensor(self.target_ratio, device=diversity_ratio.device)
        relative_error = torch.abs(diversity_ratio - target) / (target + 1e-8)
        loss = relative_error  # Relative error (unitless, naturally normalized)

        return {
            'loss': loss,
            'diversity_ratio': diversity_ratio.detach(),
            'param_variance': param_variance.detach(),
            'temporal_variance': temporal_variance.detach(),
        }

    def __repr__(self) -> str:
        return (
            f"SensitivityRegularization("
            f"epsilon={self.epsilon}, "
            f"target_ratio={self.target_ratio})"
        )
