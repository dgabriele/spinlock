"""Parameter reconstruction loss component.

This loss ensures that rollouts preserve enough information to recover
the input parameters. Prevents MNO from ignoring parameters and producing
generic "average" dynamics.

Loss = MSE(predicted_params, true_params)

where predicted_params = ParameterReconstructor(rollout)

Design:
- Modular: Can be used standalone or composed in ParameterSensitiveLoss
- Lightweight: Small MLP prevents overfitting
- Interpretable: Can measure reconstruction accuracy as validation metric
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from spinlock.mno.modules import ParameterReconstructor


class ParameterReconstructionLoss(nn.Module):
    """Force rollouts to preserve parameter information.

    Architecture:
        rollout → ParameterReconstructor → predicted_params
        loss = MSE(predicted_params, true_params)

    This auxiliary loss is added to the main training objective to ensure
    MNO doesn't take the shortcut of ignoring parameters. If reconstruction
    accuracy is high (>80%), rollouts contain sufficient parameter info.

    Attributes:
        reconstructor: Neural network that predicts params from rollouts
        param_dim: Dimensionality of parameter vector (14)
    """

    def __init__(
        self,
        param_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize parameter reconstruction loss.

        Args:
            param_dim: Dimensionality of parameter vector (default: 14)
            hidden_dim: Hidden layer size (default: 128)
            num_layers: Number of MLP layers (default: 2)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.param_dim = param_dim

        # Reconstructor network (reuses ParameterReconstructor module)
        self.reconstructor = ParameterReconstructor(
            param_dim=param_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        rollout: torch.Tensor,
        params_true: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute parameter reconstruction loss.

        Args:
            rollout: Predicted or generated rollout [B, T, C, H, W]
            params_true: True parameters that generated the rollout [B, param_dim]

        Returns:
            Dictionary containing:
            - loss: Parameter reconstruction MSE
            - accuracy: Fraction of parameters predicted within 10% error (detached)
            - params_pred: Predicted parameters (detached, for logging)
        """
        # Predict parameters from rollout
        params_pred = self.reconstructor(rollout)  # [B, param_dim]

        # Simple MSE with constant normalization
        # Use a fixed scale to keep loss at reasonable magnitude (~1-100 range)
        raw_mse = nn.functional.mse_loss(params_pred, params_true)

        # Fixed normalization constant (empirically tuned to match traj/ic scale)
        # Initial raw MSE is ~4e9, divide by 1e8 to get ~40 range
        loss = raw_mse / 1e8  # Scale down to match other loss components

        # Compute reconstruction accuracy (validation metric)
        with torch.no_grad():
            # Relative error per parameter
            rel_error = torch.abs(params_pred - params_true) / (torch.abs(params_true) + 1e-8)

            # Accuracy: fraction within 10% relative error
            accuracy = (rel_error < 0.1).float().mean()

            # Mean absolute error (interpretable metric)
            mae = torch.abs(params_pred - params_true).mean()

        return {
            'loss': loss,
            'accuracy': accuracy,
            'mae': mae,
            'params_pred': params_pred.detach(),
        }

    def __repr__(self) -> str:
        return f"ParameterReconstructionLoss(param_dim={self.param_dim})"
