"""Parameter-sensitive loss for MNO training.

This loss addresses weak parameter conditioning by combining:
1. Standard trajectory MSE (physics fidelity)
2. Parameter reconstruction (forces rollouts to preserve param info)
3. Contrastive learning (different params → different outputs)
4. Sensitivity regularization (penalizes if params have no effect)

Loss = λ_traj * L_traj + λ_ic * L_ic
       + λ_param_recon * L_param_recon
       + λ_contrastive * L_contrastive
       + λ_sensitivity * L_sensitivity

Design:
- Modular: Composes reusable loss components
- Compatible: Extends BaseNOALoss interface
- Configurable: All weights adjustable via config

Usage:
    >>> loss_fn = ParameterSensitiveLoss(
    ...     lambda_traj=1.0,
    ...     lambda_param_recon=0.5,
    ...     lambda_contrastive=0.3,
    ...     lambda_sensitivity=0.2,
    ... )
    >>> output = loss_fn.compute(
    ...     pred_trajectory, target_trajectory,
    ...     ic=ic, noa=mno, params=params
    ... )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from spinlock.mno.base_loss import BaseNOALoss, LossOutput
from spinlock.mno.losses.components import (
    ParameterReconstructionLoss,
    ContrastiveLoss,
    SensitivityRegularization,
)


class ParameterSensitiveLoss(BaseNOALoss):
    """MSE + parameter sensitivity losses for fixing weak conditioning.

    Combines trajectory matching with three parameter sensitivity components:

    1. Parameter Reconstruction: Ensures rollouts preserve param information
       - Predicts params from rollout features
       - High reconstruction accuracy (>80%) = strong conditioning

    2. Contrastive Learning: Enforces different params → different outputs
       - InfoNCE loss on (rollout, params) pairs
       - High discrimination (>90%) = params have strong effect

    3. Sensitivity Regularization: Penalizes weak parameter gradients
       - Measures output change from parameter perturbation
       - Target: param changes cause ~10% of temporal variation

    Attributes:
        lambda_traj: Weight for trajectory MSE (primary, default: 1.0)
        lambda_ic: Weight for IC reconstruction (auxiliary, default: 0.3)
        lambda_param_recon: Weight for parameter reconstruction (default: 0.5)
        lambda_contrastive: Weight for contrastive loss (default: 0.3)
        lambda_sensitivity: Weight for sensitivity regularization (default: 0.2)
        param_recon_loss: ParameterReconstructionLoss component
        contrastive_loss: ContrastiveLoss component
        sensitivity_reg: SensitivityRegularization component
    """

    def __init__(
        self,
        lambda_traj: float = 1.0,
        lambda_ic: float = 0.3,
        lambda_param_recon: float = 0.5,
        lambda_contrastive: float = 0.3,
        lambda_sensitivity: float = 0.2,
        # Parameter reconstruction config
        param_dim: int = 14,
        param_recon_hidden_dim: int = 128,
        param_recon_num_layers: int = 2,
        # Contrastive config
        contrastive_embed_dim: int = 128,
        contrastive_hidden_dim: int = 256,
        contrastive_temperature: float = 0.1,
        # Sensitivity config
        sensitivity_epsilon: float = 0.01,
        sensitivity_target_ratio: float = 0.1,
    ):
        """Initialize parameter-sensitive loss.

        Args:
            lambda_traj: Weight for trajectory MSE (default: 1.0)
            lambda_ic: Weight for IC reconstruction (default: 0.3)
            lambda_param_recon: Weight for parameter reconstruction (default: 0.5)
            lambda_contrastive: Weight for contrastive loss (default: 0.3)
            lambda_sensitivity: Weight for sensitivity regularization (default: 0.2)
            param_dim: Dimensionality of parameter vector (default: 14)
            param_recon_hidden_dim: Hidden dim for param reconstructor (default: 128)
            param_recon_num_layers: Num layers for param reconstructor (default: 2)
            contrastive_embed_dim: Embedding dim for contrastive (default: 128)
            contrastive_hidden_dim: Hidden dim for contrastive (default: 256)
            contrastive_temperature: Temperature for contrastive (default: 0.1)
            sensitivity_epsilon: Finite difference epsilon (default: 0.01)
            sensitivity_target_ratio: Target diversity ratio (default: 0.1)
        """
        super().__init__()
        self.lambda_traj = lambda_traj
        self.lambda_ic = lambda_ic
        self.lambda_param_recon = lambda_param_recon
        self.lambda_contrastive = lambda_contrastive
        self.lambda_sensitivity = lambda_sensitivity

        # Initialize loss components (modular design)
        self.param_recon_loss = ParameterReconstructionLoss(
            param_dim=param_dim,
            hidden_dim=param_recon_hidden_dim,
            num_layers=param_recon_num_layers,
        )

        self.contrastive_loss = ContrastiveLoss(
            param_dim=param_dim,
            embed_dim=contrastive_embed_dim,
            hidden_dim=contrastive_hidden_dim,
            temperature=contrastive_temperature,
        )

        self.sensitivity_reg = SensitivityRegularization(
            epsilon=sensitivity_epsilon,
            target_ratio=sensitivity_target_ratio,
        )

    def compute(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
        noa: Optional[nn.Module] = None,
        params: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """Compute parameter-sensitive loss components.

        Args:
            pred_trajectory: MNO predicted trajectory [B, T, C, H, W]
            target_trajectory: CNO ground truth trajectory [B, T, C, H, W]
            ic: Initial condition [B, C, H, W] (required for sensitivity)
            noa: MNO reference (required for sensitivity)
            params: PDE parameters [B, param_dim] (required for param sensitivity)

        Returns:
            LossOutput with:
            - total: Weighted sum of all losses
            - components: {'traj', 'ic', 'param_recon', 'contrastive', 'sensitivity'}
            - metrics: Detached values + accuracy/diversity metrics
        """
        device = pred_trajectory.device

        # Standard trajectory loss (primary physics objective)
        # Use raw MSE - already at reasonable scale (~1-1000)
        traj_loss = F.mse_loss(pred_trajectory, target_trajectory)

        # IC reconstruction loss (auxiliary)
        # Use raw MSE - already at reasonable scale
        ic_loss = torch.tensor(0.0, device=device)
        if self.lambda_ic > 0.0 and ic is not None:
            ic_loss = F.mse_loss(pred_trajectory[:, 0], ic)

        # Initialize parameter sensitivity losses
        param_recon_loss = torch.tensor(0.0, device=device)
        contrastive_loss = torch.tensor(0.0, device=device)
        sensitivity_loss = torch.tensor(0.0, device=device)

        # Metrics for logging
        param_recon_accuracy = 0.0
        param_recon_mae = 0.0
        contrastive_accuracy = 0.0
        mean_positive_sim = 0.0
        mean_negative_sim = 0.0
        diversity_ratio = 0.0
        param_variance = 0.0
        temporal_variance = 0.0

        # Compute parameter sensitivity losses if params provided
        if params is not None:
            # 1. Parameter reconstruction
            if self.lambda_param_recon > 0.0:
                param_recon_out = self.param_recon_loss(pred_trajectory, params)
                param_recon_loss = param_recon_out['loss']
                param_recon_accuracy = param_recon_out['accuracy'].item()
                param_recon_mae = param_recon_out['mae'].item()

            # 2. Contrastive learning
            if self.lambda_contrastive > 0.0:
                contrastive_out = self.contrastive_loss(pred_trajectory, params)
                contrastive_loss = contrastive_out['loss']
                contrastive_accuracy = contrastive_out['accuracy'].item()
                mean_positive_sim = contrastive_out['mean_positive_sim'].item()
                mean_negative_sim = contrastive_out['mean_negative_sim'].item()

            # 3. Sensitivity regularization (requires MNO and IC)
            if self.lambda_sensitivity > 0.0 and noa is not None and ic is not None:
                T = pred_trajectory.shape[1]
                sensitivity_out = self.sensitivity_reg(
                    mno=noa,
                    ic=ic,
                    params=params,
                    rollout_steps=T,
                )
                sensitivity_loss = sensitivity_out['loss']
                diversity_ratio = sensitivity_out['diversity_ratio'].item()
                param_variance = sensitivity_out['param_variance'].item()
                temporal_variance = sensitivity_out['temporal_variance'].item()

        # Weighted sum
        total = (
            self.lambda_traj * traj_loss
            + self.lambda_ic * ic_loss
            + self.lambda_param_recon * param_recon_loss
            + self.lambda_contrastive * contrastive_loss
            + self.lambda_sensitivity * sensitivity_loss
        )

        # Compute normalized metrics (like MSELedLoss)
        with torch.no_grad():
            target_var = target_trajectory.var()
            target_norm = target_trajectory.norm()
            target_energy = (target_trajectory ** 2).mean()

            nrmse = (traj_loss / target_var).item() if target_var > 1e-8 else float('inf')
            pred_target_diff = pred_trajectory - target_trajectory
            relative_l2 = (pred_target_diff.norm() / target_norm).item() if target_norm > 1e-8 else float('inf')
            energy_norm_mse = (traj_loss / target_energy).item() if target_energy > 1e-8 else float('inf')

        return LossOutput(
            total=total,
            components={
                'traj': traj_loss,
                'ic': ic_loss,
                'param_recon': param_recon_loss,
                'contrastive': contrastive_loss,
                'sensitivity': sensitivity_loss,
            },
            metrics={
                # Standard losses
                'traj': traj_loss.item(),
                'ic': ic_loss.item(),
                'param_recon': param_recon_loss.item(),
                'contrastive': contrastive_loss.item(),
                'sensitivity': sensitivity_loss.item(),
                'total': total.item(),
                # Normalized metrics
                'nrmse': nrmse,
                'relative_l2': relative_l2,
                'energy_norm_mse': energy_norm_mse,
                # Parameter sensitivity metrics
                'param_recon_accuracy': param_recon_accuracy,
                'param_recon_mae': param_recon_mae,
                'contrastive_accuracy': contrastive_accuracy,
                'mean_positive_sim': mean_positive_sim,
                'mean_negative_sim': mean_negative_sim,
                'diversity_ratio': diversity_ratio,
                'param_variance': param_variance,
                'temporal_variance': temporal_variance,
            },
        )

    @property
    def leading_loss_name(self) -> str:
        """Primary loss is trajectory MSE (physics fidelity)."""
        return "traj"

    @property
    def auxiliary_loss_names(self) -> list[str]:
        """Auxiliary losses for parameter sensitivity."""
        return ["ic", "param_recon", "contrastive", "sensitivity"]

    def __repr__(self) -> str:
        return (
            f"ParameterSensitiveLoss("
            f"λ_traj={self.lambda_traj}, "
            f"λ_ic={self.lambda_ic}, "
            f"λ_param_recon={self.lambda_param_recon}, "
            f"λ_contrastive={self.lambda_contrastive}, "
            f"λ_sensitivity={self.lambda_sensitivity})"
        )
