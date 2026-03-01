"""Truncated Backpropagation Through Time (TBPTT) for long-horizon rollouts.

For long rollouts (T > 32), gradients can explode through the autoregressive chain.
This module implements truncated BPTT with optional multi-window supervision:

Single-window (default, num_windows=1):
- Warmup phase: Roll out T - bptt_window steps WITHOUT gradient tracking
- Supervised phase: Roll out bptt_window steps WITH gradient tracking

Multi-window (num_windows > 1):
- Place N evenly-spaced supervised windows across the full rollout
- Between windows: no-grad gap steps (cheap)
- Within windows: supervised rollout with gradient checkpointing (expensive)
- Per-window backward() frees each window's graph immediately → peak memory = O(W)

The multi-window approach provides gradient signal at multiple temporal positions,
which is critical for chaotic systems (Lenia) where warmup state diverges
exponentially from GT due to positive Lyapunov exponents.

Example:
    >>> from spinlock.mno.truncated_bptt import TruncatedBPTT
    >>>
    >>> tbptt = TruncatedBPTT(model, timesteps=256, bptt_window=32, num_windows=3)
    >>> tbptt._window_starts  # [0, 112, 224]
    >>>
    >>> # Single-window (backward compatible):
    >>> pred = tbptt.rollout(ic)
    >>> pred_states, gt_states = tbptt.align_for_loss(pred, gt)
    >>>
    >>> # Multi-window:
    >>> segments = tbptt.multi_window_rollout(ic, params=params)
    >>> for pred_seg, win_start in segments:
    ...     pred_w, gt_w = tbptt.align_window_for_loss(pred_seg, gt, win_start)
    ...     loss = F.mse_loss(pred_w, gt_w)
    ...     (loss / num_windows).backward()  # frees this window's graph
"""

import logging
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TruncatedBPTT(nn.Module):
    """Wrapper for truncated backpropagation through time on NOA rollouts.

    Attributes:
        model: The NOA backbone model
        timesteps: Total number of rollout timesteps (e.g., 256)
        bptt_window: Number of steps to backprop through (e.g., 32)
        warmup_steps: Computed as timesteps - bptt_window (e.g., 224)
        num_windows: Number of supervised windows (1 = legacy behavior)
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        bptt_window: int,
        num_windows: int = 1,
    ):
        """Initialize truncated BPTT wrapper.

        Args:
            model: NOA backbone model with single_step() and rollout() methods
            timesteps: Total rollout length (must be > bptt_window)
            bptt_window: Number of steps to backprop through per window
            num_windows: Number of evenly-spaced supervised windows.
                1 = single final window (current behavior).
                N > 1 = N windows from step 0 to step T-W.

        Raises:
            ValueError: If bptt_window >= timesteps or num_windows < 1
        """
        super().__init__()

        if bptt_window >= timesteps:
            raise ValueError(
                f"bptt_window ({bptt_window}) must be < timesteps ({timesteps})"
            )
        if num_windows < 1:
            raise ValueError(f"num_windows must be >= 1, got {num_windows}")

        self.model = model
        self.timesteps = timesteps
        self.bptt_window = bptt_window
        self.warmup_steps = timesteps - bptt_window
        self.num_windows = num_windows
        self._window_starts = self._compute_window_starts()

        if num_windows > 1:
            logger.info(
                "Multi-window BPTT: %d windows of %d steps across %d timesteps, "
                "starts=%s",
                num_windows, bptt_window, timesteps, self._window_starts,
            )

    def _compute_window_starts(self) -> List[int]:
        """Compute evenly-spaced window start positions.

        For num_windows=1: returns [T-W] (final window only, legacy behavior).
        For num_windows=N: linspace from 0 to T-W with N points, rounded to int.

        Returns:
            Sorted list of integer step indices where supervised windows begin.
        """
        if self.num_windows == 1:
            return [self.timesteps - self.bptt_window]

        max_start = self.timesteps - self.bptt_window
        starts = [
            round(max_start * i / (self.num_windows - 1))
            for i in range(self.num_windows)
        ]

        # Warn if windows overlap significantly
        total_supervised = self.num_windows * self.bptt_window
        if total_supervised > self.timesteps:
            overlap_pct = (total_supervised - self.timesteps) / self.timesteps * 100
            logger.warning(
                "Multi-window BPTT: %.0f%% overlap (N=%d × W=%d = %d > T=%d). "
                "Some steps will receive gradient from multiple windows.",
                overlap_pct, self.num_windows, self.bptt_window,
                total_supervised, self.timesteps,
            )

        return starts

    @property
    def use_truncation(self) -> bool:
        """Whether truncation is needed (bptt_window < timesteps)."""
        return self.bptt_window < self.timesteps

    def _prepare_conditioning(
        self,
        ic: torch.Tensor,
        params: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """Prepare conditioning embeddings for param-conditioned models.

        Extracts and broadcasts parameter embeddings once so they can be
        reused across all warmup/gap single_step calls.

        Args:
            ic: Initial condition [B, C, H, W] (used for spatial dims).
            params: Parameter vector [B, param_dim] or None.

        Returns:
            (param_embed, param_spatial, conditioning_mode) tuple where:
            - param_embed: [B, embed_dim] for FiLM conditioning, or None
            - param_spatial: [B, embed_dim, H, W] for concat conditioning, or None
            - conditioning_mode: "concat", "film", or "both"
        """
        B, C, H, W = ic.shape
        conditioning_mode = getattr(self.model, 'conditioning_mode', 'concat')
        param_embed = None
        param_spatial = None

        if hasattr(self.model, 'param_conditioning') and self.model.param_conditioning:
            if params is None:
                raise ValueError(
                    "params required when model has param_conditioning=True"
                )
            param_embed = self.model.param_embedding(params)
            if conditioning_mode in ("concat", "both"):
                param_spatial = param_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)

        return param_embed, param_spatial, conditioning_mode

    def _single_step_with_conditioning(
        self,
        x: torch.Tensor,
        param_embed: Optional[torch.Tensor],
        param_spatial: Optional[torch.Tensor],
        conditioning_mode: str,
    ) -> torch.Tensor:
        """Execute one model step with pre-computed conditioning.

        Handles spatial concatenation (for concat/both modes) and FiLM
        conditioning injection. Used by both warmup and inter-window gaps.

        Args:
            x: Current state [B, C, H, W].
            param_embed: Pre-computed FiLM embedding [B, embed_dim] or None.
            param_spatial: Pre-computed spatial broadcast [B, embed_dim, H, W] or None.
            conditioning_mode: One of "concat", "film", "both".

        Returns:
            Next state [B, C, H, W].
        """
        x_augmented = x
        if conditioning_mode in ("concat", "both"):
            if param_spatial is not None:
                x_augmented = torch.cat([x_augmented, param_spatial], dim=1)

        if conditioning_mode in ("film", "both"):
            return self.model.single_step(x_augmented, conditioning=param_embed)
        else:
            return self.model.single_step(x_augmented)

    def rollout(
        self,
        ic: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate trajectory with truncated BPTT (single final window).

        Args:
            ic: Initial condition [B, C, H, W]
            params: Optional parameter vector [B, param_dim] for conditioning
            tokens: Optional token sequence for token conditioning (ignored if None)

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
                params=params,
                tokens=tokens,
            )

        param_embed, param_spatial, cond_mode = self._prepare_conditioning(
            ic, params,
        )

        # Phase 1: Warmup without gradients
        x = ic.clone()
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                x = self._single_step_with_conditioning(
                    x, param_embed, param_spatial, cond_mode,
                )

        # Detach warmup state from computation graph
        warmup_state = x.clone()

        # Phase 2: Supervised rollout with gradients
        # Use rollout() to leverage gradient checkpointing
        supervised_traj = self.model.rollout(
            warmup_state,
            steps=self.bptt_window,
            return_all_steps=True,
            params=params,
            tokens=tokens,
        )  # [B, bptt_window+1, C, H, W]

        return supervised_traj

    def multi_window_rollout(
        self,
        ic: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
    ) -> Iterator[Tuple[torch.Tensor, int]]:
        """Yield supervised windows one at a time across the full trajectory.

        This is a **generator** — it yields one (segment, window_start) pair
        at a time, then pauses. The caller should compute loss and call
        .backward() before resuming the generator for the next window. This
        ensures only one window's computation graph exists at a time, keeping
        peak activation memory = O(W) regardless of num_windows.

        Between windows, the model runs in no_grad mode (cheap gap steps).
        Each window uses model.rollout() with gradient checkpointing.

        Args:
            ic: Initial condition [B, C, H, W]
            params: Optional parameter vector [B, param_dim] for conditioning
            tokens: Optional token sequence for token conditioning

        Yields:
            (supervised_trajectory, window_start_step) tuples.
            Each supervised_trajectory has shape [B, W+1, C, H, W] where
            index 0 is the detached window IC and indices 1..W are the
            supervised states with gradients.
        """
        param_embed, param_spatial, cond_mode = self._prepare_conditioning(
            ic, params,
        )

        x = ic.clone()
        current_step = 0

        for win_start in self._window_starts:
            # Unsupervised gap: advance from current_step to win_start
            gap = win_start - current_step
            if gap > 0:
                with torch.no_grad():
                    for _ in range(gap):
                        x = self._single_step_with_conditioning(
                            x, param_embed, param_spatial, cond_mode,
                        )

            # Detach: new computation graph for this window
            x = x.detach()

            # Supervised window with gradient checkpointing
            supervised_traj = self.model.rollout(
                x,
                steps=self.bptt_window,
                return_all_steps=True,
                params=params,
                tokens=tokens,
            )  # [B, W+1, C, H, W]

            # Yield to caller — they compute loss + backward, freeing
            # this window's graph before we create the next one
            yield (supervised_traj, win_start)

            # Advance past window end (detached for next gap)
            x = supervised_traj[:, -1].detach()
            current_step = win_start + self.bptt_window

    def align_for_loss(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        skip_ic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align predicted and target trajectories for loss computation.

        Used for single-window mode. For multi-window, use align_window_for_loss().

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

    def align_window_for_loss(
        self,
        pred_segment: torch.Tensor,
        gt_trajectory: torch.Tensor,
        window_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align a single window's prediction with the corresponding GT slice.

        Used for multi-window mode. Each window's prediction is aligned to the
        correct temporal position in the full GT trajectory.

        Args:
            pred_segment: [B, W+1, C, H, W] from one supervised window.
                Index 0 is the detached IC, indices 1..W are supervised states.
            gt_trajectory: [B, T+1, C, H, W] full ground-truth trajectory
                where index 0 is the initial condition.
            window_start: Integer step where this window begins (0-indexed).

        Returns:
            (pred_states, gt_states) each [B, W, C, H, W], aligned in time.
        """
        # Skip detached IC (index 0) to get supervised predictions
        pred_states = pred_segment[:, 1:]  # [B, W, C, H, W]

        # GT trajectory: index 0 = IC, index s = state after step s.
        # Window predicts steps [win_start+1, ..., win_start+W].
        gt_start = window_start + 1
        gt_states = gt_trajectory[:, gt_start:gt_start + self.bptt_window]

        return pred_states, gt_states

    def __repr__(self) -> str:
        win_str = ""
        if self.num_windows > 1:
            win_str = f", num_windows={self.num_windows}, starts={self._window_starts}"
        return (
            f"TruncatedBPTT(timesteps={self.timesteps}, "
            f"bptt_window={self.bptt_window}, "
            f"warmup_steps={self.warmup_steps}{win_str})"
        )
