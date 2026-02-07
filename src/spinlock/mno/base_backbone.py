"""Abstract base class for NOA backbones.

Provides the interface for autoregressive neural operator backbones
that generate trajectories and support feature extraction for alignment losses.

Design Pattern:
    All NOA backbones inherit from BaseNOABackbone to ensure consistent APIs.
    This enables:
    - Swappable architectures (U-AFNO, FNO, etc.)
    - Unified training loops via abstract interface
    - Feature extraction for VQ-VAE alignment

Example:
    >>> class MyBackbone(BaseNOABackbone):
    ...     def single_step(self, x): return self.operator(x)
    ...     def get_intermediate_features(self, x, extract_from="bottleneck"):
    ...         return {"bottleneck": self.encoder(x)}
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import torch
import torch.nn as nn


class BaseNOABackbone(nn.Module, ABC):
    """Abstract base for autoregressive neural operator backbones.

    NOA backbones generate trajectories by autoregressively applying
    a neural operator: u₀ → u₁ → u₂ → ... → uₜ

    Subclasses must implement:
    - single_step(): One-step prediction u_t → u_{t+1}
    - get_intermediate_features(): Extract features for alignment losses
    - in_channels, out_channels: Channel dimensions

    The base class provides:
    - rollout(): Autoregressive trajectory generation
    - forward(): Alias for rollout()
    """

    @abstractmethod
    def single_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single timestep prediction.

        Advances the state by one timestep: u_t → u_{t+1}

        Args:
            x: Current state [B, C, H, W]

        Returns:
            Next state [B, C, H, W]
        """
        pass

    @abstractmethod
    def get_intermediate_features(
        self,
        x: torch.Tensor,
        extract_from: str = "bottleneck",
    ) -> Dict[str, torch.Tensor]:
        """Extract intermediate features for alignment losses.

        Used by VQ-VAE alignment to compute L_latent loss, which aligns
        NOA's internal representations with VQ-VAE's learned behavioral space.

        Args:
            x: Input state [B, C, H, W]
            extract_from: Which features to extract
                - "bottleneck": Deepest encoding (most compressed)
                - "skips": Skip connection features (multi-scale)
                - "all": All available features

        Returns:
            Dictionary mapping feature names to tensors
            Example: {"bottleneck": [B, 256, 8, 8], "skip_0": [B, 64, 64, 64]}
        """
        pass

    def rollout(
        self,
        u0: torch.Tensor,
        steps: int,
        return_all_steps: bool = True,
        num_realizations: int = 1,
    ) -> torch.Tensor:
        """Generate autoregressive trajectory from initial condition.

        Applies single_step() repeatedly to generate a trajectory.
        This is the core forward pass of NOA.

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps to generate
            return_all_steps: If True, return full trajectory including u0
                            If False, return only final state
            num_realizations: Number of independent realizations to generate
                            (for stochastic operators with noise)

        Returns:
            If return_all_steps and num_realizations == 1:
                Trajectory [B, T+1, C, H, W]
            If return_all_steps and num_realizations > 1:
                Trajectories [B, M, T+1, C, H, W]
            If not return_all_steps and num_realizations == 1:
                Final state [B, C, H, W]
            If not return_all_steps and num_realizations > 1:
                Final states [B, M, C, H, W]
        """
        if num_realizations > 1:
            return self._rollout_multi_realization(
                u0, steps, return_all_steps, num_realizations
            )

        trajectory: List[torch.Tensor] = [u0] if return_all_steps else []
        x = u0

        for _ in range(steps):
            x = self.single_step(x)
            if return_all_steps:
                trajectory.append(x)

        if return_all_steps:
            return torch.stack(trajectory, dim=1)  # [B, T+1, C, H, W]
        return x

    def _rollout_multi_realization(
        self,
        u0: torch.Tensor,
        steps: int,
        return_all_steps: bool,
        num_realizations: int,
    ) -> torch.Tensor:
        """Generate multiple independent realizations from same IC.

        Each realization runs through the operator independently.
        With stochastic noise enabled, each will follow a different path.

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps
            return_all_steps: If True, return full trajectories
            num_realizations: Number of realizations (M)

        Returns:
            If return_all_steps: [B, M, T+1, C, H, W]
            Else: [B, M, C, H, W]
        """
        realizations = []

        for _ in range(num_realizations):
            traj = self.rollout(u0, steps, return_all_steps, num_realizations=1)
            realizations.append(traj)

        if return_all_steps:
            # Each traj is [B, T+1, C, H, W] → stack to [B, M, T+1, C, H, W]
            return torch.stack(realizations, dim=1)
        else:
            # Each traj is [B, C, H, W] → stack to [B, M, C, H, W]
            return torch.stack(realizations, dim=1)

    def forward(
        self,
        u0: torch.Tensor,
        steps: int = 64,
        return_all_steps: bool = True,
        num_realizations: int = 1,
    ) -> torch.Tensor:
        """Forward pass - alias for rollout().

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps to generate
            return_all_steps: If True, return full trajectory
            num_realizations: Number of independent realizations

        Returns:
            Trajectory tensor (shape depends on arguments)
        """
        return self.rollout(u0, steps, return_all_steps, num_realizations)

    @property
    @abstractmethod
    def in_channels(self) -> int:
        """Number of input channels expected by the backbone."""
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of output channels produced by the backbone."""
        pass

    @property
    def num_parameters(self) -> int:
        """Total number of parameters in the backbone."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters in the backbone."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
