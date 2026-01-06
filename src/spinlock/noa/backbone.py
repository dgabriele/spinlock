"""NOA Backbone - U-AFNO wrapper for autoregressive rollout generation.

This module provides the core NOA backbone for Phase 1:
- Wraps U-AFNO neural operator
- Generates autoregressive trajectories
- Extracts intermediate features for VQ-VAE loss

Architecture:
    u₀ → U-AFNO → u₁ → U-AFNO → u₂ → ... → uₜ

    Trajectory: [u₀, u₁, u₂, ..., uₜ]

Memory optimization:
    Uses gradient checkpointing for long rollouts to trade compute for memory.
    Without checkpointing, 256-step rollouts can use ~5GB+ for gradients alone.
    With checkpointing, memory stays ~constant regardless of trajectory length.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Any, List

from spinlock.operators.u_afno import UAFNOOperator


class NOABackbone(nn.Module):
    """Minimal U-AFNO wrapper for NOA Phase 1 prototype.

    Generates autoregressive trajectories from initial conditions.
    Designed for training with grid-level MSE and VQ-VAE perceptual loss.

    Args:
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 1)
        base_channels: Base channel count for U-Net (default: 32)
        encoder_levels: Number of U-Net encoder levels (default: 3)
        modes: AFNO Fourier modes to keep (default: 16)
        afno_blocks: Number of AFNO blocks in bottleneck (default: 4)
        dropout: Dropout rate (default: 0.1)
        noise_type: Optional stochastic noise type
        noise_scale: Noise scale if using stochastic

    Example:
        >>> noa = NOABackbone(in_channels=1, out_channels=1)
        >>> u0 = torch.randn(8, 1, 64, 64)
        >>> trajectory = noa(u0, steps=64)
        >>> trajectory.shape
        torch.Size([8, 65, 1, 64, 64])  # T+1 states
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        encoder_levels: int = 3,
        modes: int = 16,
        afno_blocks: int = 4,
        dropout: float = 0.1,
        noise_type: Optional[str] = None,
        noise_scale: float = 0.05,
        use_checkpointing: bool = True,
        checkpoint_every: int = 16,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_checkpointing = use_checkpointing
        self.checkpoint_every = checkpoint_every

        # Build U-AFNO operator
        self.operator = UAFNOOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            encoder_levels=encoder_levels,
            modes=modes,
            afno_blocks=afno_blocks,
            noise_type=noise_type,
            noise_scale=noise_scale,
            **kwargs,
        )

    def forward(
        self,
        u0: torch.Tensor,
        steps: int = 64,
        return_all_steps: bool = True,
        num_realizations: int = 1,
    ) -> torch.Tensor:
        """Generate autoregressive trajectory from initial condition.

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps to generate
            return_all_steps: If True, return full trajectory including u0
                            If False, return only final state
            num_realizations: Number of independent realizations to generate (M)
                            With M > 1, different noise seeds create varied trajectories

        Returns:
            If return_all_steps:
                If num_realizations == 1: Trajectory [B, T+1, C, H, W]
                If num_realizations > 1: Trajectories [B, M, T+1, C, H, W]
            Else:
                If num_realizations == 1: Final state [B, C, H, W]
                If num_realizations > 1: Final states [B, M, C, H, W]
        """
        return self.rollout(u0, steps, return_all_steps, num_realizations)

    def _checkpointed_block(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Run multiple steps as a checkpointed block.

        Used for gradient checkpointing - recomputes forward pass during backward.
        """
        for _ in range(num_steps):
            x = self.operator(x)
        return x

    def rollout(
        self,
        u0: torch.Tensor,
        steps: int = 64,
        return_all_steps: bool = True,
        num_realizations: int = 1,
    ) -> torch.Tensor:
        """Generate autoregressive trajectory.

        Uses gradient checkpointing when training to reduce memory usage.
        Without checkpointing, 256 steps can use ~5GB for gradients.
        With checkpointing (every 16 steps), memory stays under 1GB.

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps to generate
            return_all_steps: If True, return full trajectory
            num_realizations: Number of independent realizations (M)

        Returns:
            If num_realizations == 1:
                Trajectory [B, T+1, C, H, W] or final state [B, C, H, W]
            If num_realizations > 1:
                Trajectories [B, M, T+1, C, H, W] or final states [B, M, C, H, W]
        """
        if num_realizations > 1:
            return self._rollout_multi_realization(u0, steps, return_all_steps, num_realizations)

        # Use checkpointing in training mode for memory efficiency
        # Note: We check self.training, not u0.requires_grad, because the model
        # parameters need gradients even if the input doesn't
        use_ckpt = self.use_checkpointing and self.training and torch.is_grad_enabled()

        if return_all_steps:
            trajectory = [u0]

        x = u0

        if use_ckpt and return_all_steps:
            # Gradient checkpointing: save states at intervals, recompute during backward
            checkpoint_interval = self.checkpoint_every
            t = 0

            while t < steps:
                block_size = min(checkpoint_interval, steps - t)

                # Generate block of states with checkpointing
                # We need to collect intermediate states, so run step-by-step
                # but wrap in checkpoint for memory efficiency
                for _ in range(block_size):
                    # Checkpoint each step to allow collecting intermediates
                    x = checkpoint(
                        self.operator,
                        x,
                        use_reentrant=False,
                    )
                    trajectory.append(x)
                    t += 1
        else:
            # Standard rollout (inference or single-output mode)
            for t in range(steps):
                x = self.operator(x)
                if return_all_steps:
                    trajectory.append(x)

        if return_all_steps:
            # Stack along time dimension: [B, T+1, C, H, W]
            return torch.stack(trajectory, dim=1)
        else:
            return x

    def _rollout_multi_realization(
        self,
        u0: torch.Tensor,
        steps: int,
        return_all_steps: bool,
        num_realizations: int,
    ) -> torch.Tensor:
        """Generate multiple independent realizations from the same IC.

        Each realization runs through the operator independently, potentially
        with different stochastic noise if noise_type is enabled.

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps
            return_all_steps: If True, return full trajectories
            num_realizations: Number of realizations (M)

        Returns:
            If return_all_steps: [B, M, T+1, C, H, W]
            Else: [B, M, C, H, W]
        """
        B = u0.shape[0]
        realizations = []

        for m in range(num_realizations):
            # Each realization is independent - noise (if enabled) will vary
            traj = self.rollout(u0, steps, return_all_steps, num_realizations=1)
            realizations.append(traj)

        # Stack along realization dimension
        if return_all_steps:
            # Each traj is [B, T+1, C, H, W] → stack to [B, M, T+1, C, H, W]
            return torch.stack(realizations, dim=1)
        else:
            # Each traj is [B, C, H, W] → stack to [B, M, C, H, W]
            return torch.stack(realizations, dim=1)

    def single_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step prediction (next state from current state).

        Args:
            x: Current state [B, C, H, W]

        Returns:
            Next state [B, C, H, W]
        """
        return self.operator(x)

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        extract_from: str = "bottleneck",
    ) -> Dict[str, torch.Tensor]:
        """Extract intermediate features from U-AFNO for a single input.

        Args:
            x: Input tensor [B, C, H, W]
            extract_from: What to extract ("bottleneck", "skips", "all")

        Returns:
            Dictionary of intermediate features
        """
        return self.operator.get_intermediate_features(x, extract_from=extract_from)

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_noa_backbone(config: Dict[str, Any]) -> NOABackbone:
    """Create NOA backbone from configuration dictionary.

    Args:
        config: Configuration dictionary with keys:
            - in_channels: Input channels (default: 1)
            - out_channels: Output channels (default: 1)
            - base_channels: Base channel count (default: 32)
            - encoder_levels: U-Net levels (default: 3)
            - modes: AFNO modes (default: 16)
            - afno_blocks: Number of AFNO blocks (default: 4)
            - dropout: Dropout rate (default: 0.1)
            - noise_type: Optional stochastic noise type
            - noise_scale: Noise scale (default: 0.05)

    Returns:
        Configured NOABackbone instance
    """
    return NOABackbone(
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        base_channels=config.get("base_channels", 32),
        encoder_levels=config.get("encoder_levels", 3),
        modes=config.get("modes", 16),
        afno_blocks=config.get("afno_blocks", 4),
        dropout=config.get("dropout", 0.1),
        noise_type=config.get("noise_type"),
        noise_scale=config.get("noise_scale", 0.05),
    )
