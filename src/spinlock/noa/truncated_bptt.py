"""Truncated Backpropagation Through Time (TBPTT) for long-horizon rollouts.

For long rollouts (T > 32), gradients can explode through the autoregressive chain.
This module implements truncated BPTT:
- Warmup phase: Roll out T - bptt_window steps WITHOUT gradient tracking
- Supervised phase: Roll out bptt_window steps WITH gradient tracking

This limits gradient flow to the last bptt_window steps while still generating
the full T-step trajectory for supervision.

Example:
    >>> from spinlock.noa import NOABackbone, TruncatedBPTT
    >>>
    >>> noa = NOABackbone(...)
    >>> tbptt = TruncatedBPTT(noa, timesteps=256, bptt_window=32)
    >>>
    >>> # During training:
    >>> pred_trajectory = tbptt.rollout(ic, tokens=tokens)
    >>> # pred_trajectory: [B, 33, C, H, W] (warmup final state + 32 supervised steps)
    >>>
    >>> # For loss computation:
    >>> pred_states, target_states = tbptt.align_for_loss(pred_trajectory, target_trajectory)
    >>> loss = F.mse_loss(pred_states, target_states)

Documentation:
    - TBPTT implementation details: docs/truncated-bptt-integration.md
    - Training guide: docs/noa-training-guide.md
    - NOA architecture: docs/noa-architecture.md
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TruncatedBPTT(nn.Module):
    """Wrapper for truncated backpropagation through time on NOA rollouts.

    Attributes:
        model: The NOA backbone model
        timesteps: Total number of rollout timesteps (e.g., 256)
        bptt_window: Number of steps to backprop through (e.g., 32)
        warmup_steps: Computed as timesteps - bptt_window (e.g., 224)
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        bptt_window: int,
    ):
        """Initialize truncated BPTT wrapper.

        Args:
            model: NOA backbone model with single_step() and rollout() methods
            timesteps: Total rollout length (must be > bptt_window)
            bptt_window: Number of steps to backprop through

        Raises:
            ValueError: If bptt_window >= timesteps
        """
        super().__init__()

        if bptt_window >= timesteps:
            raise ValueError(
                f"bptt_window ({bptt_window}) must be < timesteps ({timesteps})"
            )

        self.model = model
        self.timesteps = timesteps
        self.bptt_window = bptt_window
        self.warmup_steps = timesteps - bptt_window

    @property
    def use_truncation(self) -> bool:
        """Whether truncation is needed (bptt_window < timesteps)."""
        return self.bptt_window < self.timesteps

    def rollout(
        self,
        ic: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate trajectory with truncated BPTT.

        Args:
            ic: Initial condition [B, C, H, W]
            tokens: Optional token conditioning [B, num_tokens]

        Returns:
            Predicted trajectory [B, bptt_window+1, C, H, W]
            - Index 0: Final warmup state (detached from graph)
            - Index 1 to bptt_window: Supervised states (with gradients)
        """
        if not self.use_truncation:
            # No truncation needed, use standard rollout
            return self.model.rollout(
                ic,
                steps=self.timesteps,
                return_all_steps=True,
                tokens=tokens,
            )

        # Prepare token embeddings if model uses token conditioning
        token_spatial = None
        if hasattr(self.model, 'token_conditioning') and self.model.token_conditioning:
            B, C, H, W = ic.shape

            if tokens is None:
                # Stage 2: No tokens provided, use zero embeddings (model must self-regulate)
                token_spatial = torch.zeros(
                    B, self.model.token_embed_dim, H, W,
                    device=ic.device, dtype=ic.dtype
                )
            else:
                # Stage 1: Tokens provided, use them for conditioning
                # Embed tokens: [B, num_tokens] -> [B, token_embed_dim]
                token_embed = self.model.token_embedding(tokens)
                # Broadcast to spatial dimensions
                token_spatial = token_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)
                # token_spatial is now [B, token_embed_dim, H, W]

        # Phase 1: Warmup without gradients
        x = ic.clone()
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                # Augment with tokens if conditioning is enabled
                if token_spatial is not None:
                    x_augmented = torch.cat([x, token_spatial], dim=1)
                else:
                    x_augmented = x

                # Single step (returns [B, out_channels, H, W])
                x = self.model.single_step(x_augmented)

        # Detach warmup state from computation graph
        warmup_state = x.clone()

        # Phase 2: Supervised rollout with gradients
        # Use rollout() to leverage gradient checkpointing
        supervised_traj = self.model.rollout(
            warmup_state,
            steps=self.bptt_window,
            return_all_steps=True,
            tokens=tokens,
        )  # [B, bptt_window+1, C, H, W]

        return supervised_traj

    def align_for_loss(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        skip_ic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align predicted and target trajectories for loss computation.

        Args:
            pred_trajectory: Predicted trajectory from rollout()
                - If truncated: [B, bptt_window+1, C, H, W]
                - If not truncated: [B, timesteps+1, C, H, W]
            target_trajectory: Full target trajectory [B, timesteps+1, C, H, W]
            skip_ic: Whether to skip the initial condition (index 0)

        Returns:
            (pred_states, target_states) both [B, N, C, H, W] where:
                - If truncated: N = bptt_window (last bptt_window states)
                - If not truncated: N = timesteps
        """
        if not self.use_truncation:
            # No truncation: use all states except IC
            if skip_ic:
                pred_states = pred_trajectory[:, 1:, :, :, :]  # [B, timesteps, C, H, W]
                target_states = target_trajectory[:, 1:, :, :, :]
            else:
                pred_states = pred_trajectory
                target_states = target_trajectory
        else:
            # Truncated: pred has bptt_window+1 states (warmup final + supervised)
            # Skip first state (warmup final) to get supervised portion
            if skip_ic:
                pred_states = pred_trajectory[:, 1:, :, :, :]  # [B, bptt_window, C, H, W]
            else:
                pred_states = pred_trajectory  # [B, bptt_window+1, C, H, W]

            # Extract corresponding window from target trajectory
            # target: [B, timesteps+1, C, H, W] -> take last bptt_window states
            if skip_ic:
                target_states = target_trajectory[:, -self.bptt_window:, :, :, :]
            else:
                target_states = target_trajectory[:, -(self.bptt_window+1):, :, :, :]

        return pred_states, target_states

    def __repr__(self) -> str:
        return (
            f"TruncatedBPTT(timesteps={self.timesteps}, "
            f"bptt_window={self.bptt_window}, "
            f"warmup_steps={self.warmup_steps})"
        )
