"""Parameter reconstruction from rollout features.

This module provides a neural network that predicts PDE parameters from
rollout statistical features. Used in the parameter reconstruction loss
to ensure MNO preserves parameter information in its outputs.

The reconstructor follows a simple architecture:
    rollout → RolloutFeatureExtractor → MLP → predicted params

Design principles:
- DRY: Reuses RolloutFeatureExtractor (shared across components)
- Modular: Can be used standalone or composed in loss functions
- Simple: Small MLP (2-3 layers) prevents overfitting on features
"""

import torch
import torch.nn as nn
from typing import Optional

from spinlock.mno.features import RolloutFeatureExtractor


class ParameterReconstructor(nn.Module):
    """Predict PDE parameters from rollout features.

    Architecture:
        rollout [B, T, C, H, W]
          ↓
        RolloutFeatureExtractor
          ↓
        features [B, 32]
          ↓
        MLP (32 → hidden → ... → 14)
          ↓
        params [B, 14]

    The reconstructor is trained jointly with MNO to ensure that rollouts
    preserve enough information to recover the input parameters. This
    prevents MNO from ignoring parameters and producing generic outputs.

    Attributes:
        feature_extractor: Reusable RolloutFeatureExtractor (DRY)
        mlp: Learnable MLP that maps features to parameters
        param_dim: Dimensionality of parameter vector (14)
        hidden_dim: Hidden layer size (default: 128)
        num_layers: Number of MLP layers (default: 2)
    """

    def __init__(
        self,
        param_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize parameter reconstructor.

        Args:
            param_dim: Dimensionality of parameter vector (default: 14)
            hidden_dim: Hidden layer size (default: 128)
            num_layers: Number of MLP layers (default: 2, minimum: 1)
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super().__init__()
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, num_layers)

        # Reusable feature extractor (DRY principle)
        self.feature_extractor = RolloutFeatureExtractor()
        feature_dim = self.feature_extractor.feature_dim  # 32

        # Build MLP: features → hidden layers → params
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        # Output layer (no activation - parameters can be any real value)
        layers.append(nn.Linear(hidden_dim, param_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, rollout: torch.Tensor) -> torch.Tensor:
        """Predict parameters from rollout.

        Args:
            rollout: Tensor of shape [B, T, C, H, W] or [B, M, T, C, H, W]

        Returns:
            params_pred: Predicted parameters [B, param_dim]
        """
        # Extract statistical features (reuses shared extractor)
        features = self.feature_extractor(rollout)  # [B, 32]

        # Predict parameters from features
        params_pred = self.mlp(features)  # [B, param_dim]

        return params_pred

    def reconstruction_loss(
        self,
        rollout: torch.Tensor,
        params_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute parameter reconstruction loss.

        Args:
            rollout: Rollout tensor [B, T, C, H, W] or [B, M, T, C, H, W]
            params_true: True parameters [B, param_dim]

        Returns:
            loss: MSE between predicted and true parameters
        """
        params_pred = self.forward(rollout)
        return nn.functional.mse_loss(params_pred, params_true)

    def __repr__(self) -> str:
        return (
            f"ParameterReconstructor("
            f"param_dim={self.param_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers})"
        )
