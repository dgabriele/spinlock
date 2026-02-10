"""Parameter sensitivity metrics.

Functional utilities for measuring how much MNO rollouts change when
parameters are varied. Used by both SensitivityRegularization loss
and validation diagnostics.

Key metric: diversity_ratio = param_variance / temporal_variance
- Ratio < 0.05: Weak parameter conditioning (parameters mostly ignored)
- Ratio > 0.1: Strong parameter conditioning (target for training)
- Ratio > 0.2: Very strong conditioning (stretch goal)
"""

import torch
import torch.nn as nn
from typing import Dict


def measure_parameter_sensitivity(
    mno: nn.Module,
    ic: torch.Tensor,
    params: torch.Tensor,
    epsilon: float = 0.01,
    rollout_steps: int = 256,
) -> Dict[str, torch.Tensor]:
    """Measure rollout sensitivity to parameter changes.

    Uses finite differences to estimate how much rollouts change when
    parameters are perturbed:

    1. Generate rollout_0 = MNO(ic, params)
    2. Perturb params: params' = params + ε * noise
    3. Generate rollout_1 = MNO(ic, params')
    4. Measure variance across parameter dimension
    5. Compare to variance across time dimension

    Args:
        mno: Meta Neural Operator with .rollout() method
        ic: Initial condition [B, C, H, W]
        params: Parameters [B, param_dim]
        epsilon: Perturbation magnitude (default: 0.01 = 1%)
        rollout_steps: Number of timesteps (default: 256)

    Returns:
        Dictionary containing:
        - diversity_ratio: param_variance / temporal_variance
        - param_variance: Variance across parameter perturbations
        - temporal_variance: Variance across time
        - sensitivity: Mean squared difference from parameter perturbation
    """
    B, C, H, W = ic.shape
    device = ic.device

    # Generate baseline rollout
    with torch.no_grad():
        rollout_0 = mno.rollout(ic, params=params, steps=rollout_steps)  # [B, T, C, H, W]
        T = rollout_0.shape[1]

    # Perturb parameters (small random noise)
    params_perturbed = params + epsilon * torch.randn_like(params)

    # Generate perturbed rollout
    with torch.no_grad():
        rollout_1 = mno.rollout(ic, params=params_perturbed, steps=rollout_steps)  # [B, T, C, H, W]

    # Measure sensitivity: mean squared difference per parameter change
    diff = rollout_1 - rollout_0  # [B, T, C, H, W]
    sensitivity = (diff ** 2).mean()

    # Measure variance across parameters vs time
    diversity_ratio, param_var, temporal_var = compute_diversity_ratio(
        rollout_0, rollout_1
    )

    return {
        'diversity_ratio': diversity_ratio,
        'param_variance': param_var,
        'temporal_variance': temporal_var,
        'sensitivity': sensitivity,
    }


def compute_diversity_ratio(
    rollout_0: torch.Tensor,
    rollout_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute diversity ratio from two rollouts.

    diversity_ratio = variance_across_params / variance_across_time

    This ratio measures whether parameter changes cause as much variation
    as temporal evolution:
    - Ratio << 1: Parameters have little effect (weak conditioning)
    - Ratio ~ 0.1: Parameters cause 10% of temporal variation (target)
    - Ratio ~ 1: Parameters and time equally important

    Args:
        rollout_0: Baseline rollout [B, T, C, H, W]
        rollout_1: Perturbed rollout [B, T, C, H, W]

    Returns:
        diversity_ratio: param_variance / temporal_variance
        param_variance: Variance across the two rollouts (parameter axis)
        temporal_variance: Variance across time (temporal axis)
    """
    # Stack rollouts along new dimension for variance computation
    rollouts_stacked = torch.stack([rollout_0, rollout_1], dim=0)  # [2, B, T, C, H, W]

    # Variance across parameter perturbations (dim=0)
    param_variance = rollouts_stacked.var(dim=0).mean()  # Scalar

    # Variance across time (dim=2, which is T after accounting for the stacked dim)
    # Average over batch, channels, spatial dims
    temporal_variance = rollout_0.var(dim=1).mean()  # Scalar

    # Diversity ratio
    diversity_ratio = param_variance / (temporal_variance + 1e-8)

    return diversity_ratio, param_variance, temporal_variance
