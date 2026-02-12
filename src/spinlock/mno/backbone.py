"""MNO Backbone - U-AFNO wrapper for autoregressive rollout generation.

This module provides the core MNO backbone for Phase 1:
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
from typing import Optional, Dict, Any, List, Literal

from spinlock.operators.u_afno import UAFNOOperator, FiLMUAFNOOperator
from spinlock.operators.film import FiLMConfig
from spinlock.mno.base_backbone import BaseMNOBackbone


class MNOBackbone(BaseMNOBackbone):
    """Minimal U-AFNO wrapper for MNO Phase 1 prototype.

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
        >>> mno = MNOBackbone(in_channels=1, out_channels=1)
        >>> u0 = torch.randn(8, 1, 64, 64)
        >>> trajectory = mno(u0, steps=64)
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
        # Parameter conditioning (operator θ)
        param_conditioning: bool = False,
        param_dim: int = 14,
        param_embed_dim: int = 64,
        # FiLM conditioning mode: "concat" (default), "film", or "both"
        conditioning_mode: Literal["concat", "film", "both"] = "concat",
        # FiLM configuration (only used when conditioning_mode includes "film")
        film_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self.use_checkpointing = use_checkpointing
        self.checkpoint_every = checkpoint_every
        self.update_mode = update_mode
        self.residual_scale = 0.1  # Scale down residuals initially
        self.param_conditioning = param_conditioning
        self.conditioning_mode = conditioning_mode

        # Validate conditioning_mode
        if conditioning_mode not in ("concat", "film", "both"):
            raise ValueError(
                f"conditioning_mode must be 'concat', 'film', or 'both', got '{conditioning_mode}'"
            )

        # FiLM mode requires param_conditioning
        if conditioning_mode in ("film", "both") and not param_conditioning:
            raise ValueError(
                "FiLM conditioning requires param_conditioning=True"
            )

        # Parse FiLM config
        self.film_config = None
        if conditioning_mode in ("film", "both"):
            if film_config is None:
                # Default FiLM config: spatial-only modulation
                film_config = {
                    "enabled": True,
                    "embed_dim": param_embed_dim,
                    "hidden_dim": 128,
                    "post_norm": True,
                    "modulate_encoder": True,
                    "modulate_decoder": True,
                    "modulate_afno_post": False,
                }
            else:
                # Ensure embed_dim matches param_embed_dim
                film_config = dict(film_config)  # Copy to avoid mutation
                film_config["enabled"] = True
                film_config.setdefault("embed_dim", param_embed_dim)

            self.film_config = film_config

        # Initialize parameter embedding if enabled
        if param_conditioning:
            # MLP to embed parameter vector θ: [B, param_dim] → [B, param_embed_dim]
            self.param_embedding = nn.Sequential(
                nn.Linear(param_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, param_embed_dim),
                nn.LayerNorm(param_embed_dim),
            )
            self.param_embed_dim = param_embed_dim
        else:
            self.param_embedding = None
            self.param_embed_dim = None

        # Calculate operator input channels based on conditioning mode
        # - "concat": add param_embed_dim to input channels
        # - "film": no extra input channels (FiLM modulates internally)
        # - "both": add concat channels AND use FiLM modulation
        operator_input_channels = in_channels

        if conditioning_mode in ("concat", "both"):
            # Add conditioning channels for concatenation
            if param_conditioning:
                operator_input_channels += param_embed_dim

        # Build operator based on conditioning mode
        if conditioning_mode in ("film", "both"):
            # Use FiLMUAFNOOperator for FiLM-based conditioning
            self.operator = FiLMUAFNOOperator(
                in_channels=operator_input_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                encoder_levels=encoder_levels,
                modes=modes,
                afno_blocks=afno_blocks,
                film_config=self.film_config,
                noise_type=noise_type,
                noise_scale=noise_scale,
                **kwargs,
            )
        else:
            # Use standard UAFNOOperator for concat-only conditioning
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
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate autoregressive trajectory from initial condition.

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps to generate
            return_all_steps: If True, return full trajectory including u0
                            If False, return only final state
            num_realizations: Number of independent realizations to generate (M)
                            With M > 1, different noise seeds create varied trajectories
            params: Optional operator parameter vector θ [B, param_dim] for conditioning
                   (required if param_conditioning=True)

        Returns:
            If return_all_steps:
                If num_realizations == 1: Trajectory [B, T+1, C, H, W]
                If num_realizations > 1: Trajectories [B, M, T+1, C, H, W]
            Else:
                If num_realizations == 1: Final state [B, C, H, W]
                If num_realizations > 1: Final states [B, M, C, H, W]
        """
        return self.rollout(u0, steps, return_all_steps, num_realizations, params)

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
        params: Optional[torch.Tensor] = None,
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
            params: Optional operator parameter vector θ [B, param_dim] for conditioning
                   (required if param_conditioning=True)
            tokens: Optional token sequence for token conditioning (currently unused, for future compatibility)

        Returns:
            If num_realizations == 1:
                Trajectory [B, T+1, C, H, W] or final state [B, C, H, W]
            If num_realizations > 1:
                Trajectories [B, M, T+1, C, H, W] or final states [B, M, C, H, W]
        """
        B, C, H, W = u0.shape

        # Prepare parameter embeddings if conditioning is enabled
        param_embed = None  # Vector embedding for FiLM
        param_spatial = None  # Spatial broadcast for concat
        if self.param_conditioning:
            if params is None:
                raise ValueError("params required when param_conditioning=True")
            # Embed parameter vector: [B, param_dim] -> [B, param_embed_dim]
            param_embed = self.param_embedding(params)

            # Broadcast to spatial dimensions for concat mode
            if self.conditioning_mode in ("concat", "both"):
                param_spatial = param_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)

        if num_realizations > 1:
            return self._rollout_multi_realization(
                u0, steps, return_all_steps, num_realizations,
                param_embed, param_spatial
            )

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
                    # Augment input based on conditioning mode
                    x_augmented = x
                    if self.conditioning_mode in ("concat", "both"):
                        if param_spatial is not None:
                            x_augmented = torch.cat([x_augmented, param_spatial], dim=1)

                    # Checkpoint each step to allow collecting intermediates
                    # For FiLM mode, pass param_embed to single_step
                    if self.conditioning_mode in ("film", "both"):
                        x = checkpoint(
                            lambda x_aug: self.single_step(x_aug, conditioning=param_embed),
                            x_augmented,
                            use_reentrant=False,
                        )
                    else:
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
                # Augment input based on conditioning mode
                x_augmented = x
                if self.conditioning_mode in ("concat", "both"):
                    if param_spatial is not None:
                        x_augmented = torch.cat([x_augmented, param_spatial], dim=1)

                # Pass param_embed for FiLM conditioning
                if self.conditioning_mode in ("film", "both"):
                    x = self.single_step(x_augmented, conditioning=param_embed)
                else:
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
        param_embed: Optional[torch.Tensor] = None,
        param_spatial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate multiple independent realizations from the same IC.

        Each realization runs through the operator independently, potentially
        with different stochastic noise if noise_type is enabled.

        Args:
            u0: Initial condition [B, C, H, W]
            steps: Number of timesteps
            return_all_steps: If True, return full trajectories
            num_realizations: Number of realizations (M)
            param_embed: Optional parameter embedding [B, param_embed_dim] for FiLM
            param_spatial: Optional pre-computed parameter embeddings [B, param_embed_dim, H, W]

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
                # Augment input based on conditioning mode
                x_augmented = x
                if self.conditioning_mode in ("concat", "both"):
                    if param_spatial is not None:
                        x_augmented = torch.cat([x_augmented, param_spatial], dim=1)

                # Pass param_embed for FiLM conditioning
                if self.conditioning_mode in ("film", "both"):
                    x = self.single_step(x_augmented, conditioning=param_embed)
                else:
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

    def single_step(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Single-step prediction (next state from current state).

        Args:
            x: Current state [B, C, H, W]
               When conditioning_mode="concat" or "both", this is augmented:
               [B, C + param_embed_dim, H, W]
            conditioning: Optional conditioning embedding [B, embed_dim] for FiLM

        Returns:
            Next state [B, out_channels, H, W]
        """
        if self.update_mode == "residual":
            # Extract base state for residual connection
            # When concat conditioning is enabled, x has extra channels
            if self.conditioning_mode in ("concat", "both") and self.param_conditioning:
                # Split: base_state [B, C, H, W] and conditioning channels
                base_state = x[:, :self._in_channels, :, :]
                # Operator receives full augmented input
                # For FiLM mode, also pass conditioning embedding
                if isinstance(self.operator, FiLMUAFNOOperator):
                    delta = self.operator(x, conditioning=conditioning)
                else:
                    delta = self.operator(x)
                # Residual update only on base state
                return base_state + self.residual_scale * delta
            elif self.conditioning_mode == "film" and self.param_conditioning:
                # Pure FiLM mode: no concat, just pass embedding
                if isinstance(self.operator, FiLMUAFNOOperator):
                    delta = self.operator(x, conditioning=conditioning)
                else:
                    delta = self.operator(x)
                return x + self.residual_scale * delta
            else:
                # u_{t+1} = u_t + scale * MNO(u_t) - Euler-style, better gradient flow
                return x + self.residual_scale * self.operator(x)
        else:
            # u_{t+1} = MNO(u_t) - pure autoregressive
            if isinstance(self.operator, FiLMUAFNOOperator):
                return self.operator(x, conditioning=conditioning)
            else:
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


def create_mno_backbone(config: Dict[str, Any]) -> MNOBackbone:
    """Create MNO backbone from configuration dictionary.

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
        Configured MNOBackbone instance
    """
    return MNOBackbone(
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
