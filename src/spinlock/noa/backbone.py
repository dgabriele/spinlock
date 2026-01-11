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

Documentation:
    - Architecture overview: docs/noa-architecture.md
    - Complete architecture spec: docs/MNO_ARCHITECTURE.md
    - Training guide: docs/noa-training-guide.md
    - Two-stage curriculum: docs/two-stage-curriculum-architecture.md
    - Truncated BPTT integration: docs/truncated-bptt-integration.md
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Any, List

from spinlock.operators.u_afno import UAFNOOperator
from spinlock.noa.base_backbone import BaseNOABackbone


class NOABackbone(BaseNOABackbone):
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
        update_mode: str = "residual",  # "residual" or "autoregressive"
        # NEW: Token conditioning parameters
        token_conditioning: bool = False,
        token_embed_dim: int = 64,
        num_tokens: int = 21,
        codebook_sizes: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self.use_checkpointing = use_checkpointing
        self.checkpoint_every = checkpoint_every
        self.update_mode = update_mode
        self.residual_scale = 0.1  # Scale down residuals initially
        self.token_conditioning = token_conditioning

        # Initialize token embedding if enabled
        if token_conditioning:
            if codebook_sizes is None:
                raise ValueError("codebook_sizes required when token_conditioning=True")

            from .token_embedding import TokenEmbedding
            self.token_embedding = TokenEmbedding(
                num_tokens=num_tokens,
                codebook_sizes=codebook_sizes,
                embed_dim=32,  # Per-token embedding dimension
                projection_dim=token_embed_dim,  # Final projected dimension
            )
            self.token_embed_dim = token_embed_dim  # Store for use in rollout

            # Adjust operator input channels to account for token embeddings
            operator_input_channels = in_channels + token_embed_dim
        else:
            self.token_embedding = None
            self.token_embed_dim = None
            operator_input_channels = in_channels

        # Build U-AFNO operator
        self.operator = UAFNOOperator(
            in_channels=operator_input_channels,
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
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate autoregressive trajectory from initial condition.

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps to generate
            return_all_steps: If True, return full trajectory including u0
                            If False, return only final state
            num_realizations: Number of independent realizations to generate (M)
                            With M > 1, different noise seeds create varied trajectories
            tokens: Optional VQ token indices [B, num_tokens] for conditioning
                   (required if token_conditioning=True)

        Returns:
            If return_all_steps:
                If num_realizations == 1: Trajectory [B, T+1, C, H, W]
                If num_realizations > 1: Trajectories [B, M, T+1, C, H, W]
            Else:
                If num_realizations == 1: Final state [B, C, H, W]
                If num_realizations > 1: Final states [B, M, C, H, W]
        """
        return self.rollout(u0, steps, return_all_steps, num_realizations, tokens)

    def _single_step_for_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper for single_step that works with torch.utils.checkpoint."""
        return self.single_step(x)

    def _checkpointed_block(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Run multiple steps as a checkpointed block.

        Used for gradient checkpointing - recomputes forward pass during backward.
        """
        for _ in range(num_steps):
            x = self.single_step(x)
        return x

    def rollout(
        self,
        u0: torch.Tensor,
        steps: int = 64,
        return_all_steps: bool = True,
        num_realizations: int = 1,
        tokens: Optional[torch.Tensor] = None,
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
            tokens: Optional VQ token indices [B, num_tokens] for conditioning

        Returns:
            If num_realizations == 1:
                Trajectory [B, T+1, C, H, W] or final state [B, C, H, W]
            If num_realizations > 1:
                Trajectories [B, M, T+1, C, H, W] or final states [B, M, C, H, W]
        """
        # Validate token conditioning
        # Prepare token embeddings if conditioning is enabled
        if self.token_conditioning:
            B, C, H, W = u0.shape

            if tokens is None:
                # Stage 2: No tokens provided, use zero embeddings (model must self-regulate)
                # This allows loading token-conditioned checkpoints for VQ-led training
                token_spatial = torch.zeros(
                    B, self.token_embed_dim, H, W,
                    device=u0.device, dtype=u0.dtype
                )
            else:
                # Stage 1: Tokens provided, use them for conditioning
                # Embed tokens: [B, num_tokens] -> [B, token_embed_dim]
                token_embed = self.token_embedding(tokens)
                # Broadcast to spatial dimensions
                token_spatial = token_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)
                # token_spatial is now [B, token_embed_dim, H, W]
        else:
            token_spatial = None

        if num_realizations > 1:
            return self._rollout_multi_realization(u0, steps, return_all_steps, num_realizations, token_spatial)

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
                    # Concatenate token embeddings if conditioning
                    if token_spatial is not None:
                        x_augmented = torch.cat([x, token_spatial], dim=1)
                    else:
                        x_augmented = x

                    # Checkpoint each step to allow collecting intermediates
                    x = checkpoint(
                        self._single_step_for_checkpoint,
                        x_augmented,
                        use_reentrant=False,
                    )
                    trajectory.append(x)
                    t += 1
        else:
            # Standard rollout (inference or single-output mode)
            for t in range(steps):
                # Concatenate token embeddings if conditioning
                if token_spatial is not None:
                    x_augmented = torch.cat([x, token_spatial], dim=1)
                else:
                    x_augmented = x

                x = self.single_step(x_augmented)
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
        token_spatial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate multiple independent realizations from the same IC.

        Each realization runs through the operator independently, potentially
        with different stochastic noise if noise_type is enabled.

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps
            return_all_steps: If True, return full trajectories
            num_realizations: Number of realizations (M)
            token_spatial: Optional pre-computed token embeddings [B, token_embed_dim, H, W]

        Returns:
            If return_all_steps: [B, M, T+1, C, H, W]
            Else: [B, M, C, H, W]
        """
        B = u0.shape[0]
        realizations = []

        for m in range(num_realizations):
            # Run rollout for single realization
            if return_all_steps:
                trajectory = [u0]

            x = u0

            # Standard rollout (no checkpointing for multi-realization)
            for t in range(steps):
                # Concatenate token embeddings if conditioning
                if token_spatial is not None:
                    x_augmented = torch.cat([x, token_spatial], dim=1)
                else:
                    x_augmented = x

                x = self.single_step(x_augmented)
                if return_all_steps:
                    trajectory.append(x)

            if return_all_steps:
                realizations.append(torch.stack(trajectory, dim=1))  # [B, T+1, C, H, W]
            else:
                realizations.append(x)  # [B, C, H, W]

        # Stack along realization dimension
        if return_all_steps:
            # Each traj is [B, T+1, C, H, W] → stack to [B, M, T+1, C, H, W]
            return torch.stack(realizations, dim=1)
        else:
            # Each x is [B, C, H, W] → stack to [B, M, C, H, W]
            return torch.stack(realizations, dim=1)

    def single_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step prediction (next state from current state).

        Args:
            x: Current state [B, C, H, W]
               When token_conditioning=True, this is augmented:
               [B, C + token_embed_dim, H, W]

        Returns:
            Next state [B, out_channels, H, W]
        """
        if self.update_mode == "residual":
            # Extract base state for residual connection
            # When token conditioning is enabled, x has extra channels
            if self.token_conditioning:
                # Split: base_state [B, C, H, W] and tokens [B, token_embed_dim, H, W]
                base_state = x[:, :self._in_channels, :, :]
                # Operator receives full augmented input
                delta = self.operator(x)
                # Residual update only on base state
                return base_state + self.residual_scale * delta
            else:
                # u_{t+1} = u_t + scale * NOA(u_t) - Euler-style, better gradient flow
                return x + self.residual_scale * self.operator(x)
        else:
            # u_{t+1} = NOA(u_t) - pure autoregressive
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
    def in_channels(self) -> int:
        """Number of input channels expected by the backbone."""
        return self._in_channels

    @property
    def out_channels(self) -> int:
        """Number of output channels produced by the backbone."""
        return self._out_channels


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
            - update_mode: "residual" or "autoregressive" (default: "residual")

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
        update_mode=config.get("update_mode", "residual"),
    )
