"""Multi-resolution temporal downsampling pyramid.

Produces multiple temporal resolutions from a single input sequence
using adaptive average pooling along the time axis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class TemporalPyramid(nn.Module):
    """Multi-resolution temporal downsampling with adaptive level selection.

    Takes [B, T, D] and produces a list of [B, T_i, D] tensors at
    decreasing temporal resolutions via nn.AdaptiveAvgPool1d.

    Supports variable-length trajectories through:
    - Adaptive pyramid levels: Skip levels that would produce T_i < min_pyramid_length
    - Mask downsampling: Propagate validity masks through pyramid levels

    Args:
        downsample_factors: List of temporal downsampling factors.
            Factor 1 = full resolution, 2 = half, 4 = quarter, etc.
            Default: [1, 2, 4, 8]
        mask_downsample_method: How to downsample masks ("ceil" or "floor").
            "ceil": Conservative - mark valid if any source timestep is valid
            "floor": Aggressive - mark valid only if all source timesteps are valid
            Default: "ceil"
        adaptive: Enable adaptive pyramid level selection based on trajectory length.
            If True, automatically skip levels where T_i would be < min_pyramid_length.
            Default: True
        min_pyramid_length: Minimum allowed length at any pyramid level.
            Levels producing shorter sequences are skipped when adaptive=True.
            Default: 1

    Example:
        >>> pyramid = TemporalPyramid([1, 2, 4, 8], adaptive=True)
        >>> x = torch.randn(32, 256, 64)  # [B, T, D]
        >>> levels, _, valid_factors = pyramid(x)
        >>> [l.shape[1] for l in levels]  # [256, 128, 64, 32]
        >>> valid_factors  # [1, 2, 4, 8] (all levels valid)

        >>> # Short trajectory: some levels skipped
        >>> x_short = torch.randn(32, 4, 64)
        >>> levels, _, valid_factors = pyramid(x_short, lengths=torch.tensor([4]*32))
        >>> valid_factors  # [1, 2, 4] (skip 8x factor)
    """

    def __init__(
        self,
        downsample_factors: List[int] = [1, 2, 4, 8],
        mask_downsample_method: str = "ceil",
        adaptive: bool = True,
        min_pyramid_length: int = 1,
    ):
        super().__init__()
        if not downsample_factors:
            raise ValueError("downsample_factors must be non-empty")
        for f in downsample_factors:
            if f < 1:
                raise ValueError(f"Downsample factor must be >= 1, got {f}")

        self.downsample_factors = downsample_factors
        self.mask_downsample_method = mask_downsample_method
        self.adaptive = adaptive
        self.min_pyramid_length = min_pyramid_length

        if mask_downsample_method not in ("ceil", "floor"):
            raise ValueError(
                f"mask_downsample_method must be 'ceil' or 'floor', "
                f"got '{mask_downsample_method}'"
            )

        if min_pyramid_length < 1:
            raise ValueError(
                f"min_pyramid_length must be >= 1, got {min_pyramid_length}"
            )

    def get_valid_levels(self, T: int) -> List[int]:
        """Determine valid pyramid levels for given trajectory length.

        Args:
            T: Trajectory length

        Returns:
            List of downsample factors that produce T_i >= min_pyramid_length

        Examples:
            >>> pyramid = TemporalPyramid([1, 2, 4, 8], adaptive=True, min_pyramid_length=1)
            >>> pyramid.get_valid_levels(500)  # [1, 2, 4, 8] (all valid)
            >>> pyramid.get_valid_levels(8)    # [1, 2, 4, 8] (all valid, T/8=1)
            >>> pyramid.get_valid_levels(4)    # [1, 2, 4] (skip 8x, would give T/8=0.5)
            >>> pyramid.get_valid_levels(2)    # [1, 2] (skip 4x, 8x)
            >>> pyramid.get_valid_levels(1)    # [1] (only full resolution)
        """
        if not self.adaptive:
            return self.downsample_factors

        valid_factors = [
            f for f in self.downsample_factors
            if T // f >= self.min_pyramid_length
        ]

        # Always keep at least one level (full resolution)
        return valid_factors if valid_factors else [1]

    def _downsample_mask(
        self,
        mask: torch.Tensor,
        target_len: int
    ) -> torch.Tensor:
        """Downsample mask to target length.

        Args:
            mask: Boolean mask [B, T]
            target_len: Target sequence length

        Returns:
            Downsampled mask [B, target_len]

        The downsampling method is controlled by mask_downsample_method:
        - "ceil": Mark position valid if ANY source position is valid (conservative)
        - "floor": Mark position valid only if ALL source positions are valid (aggressive)
        """
        B, T = mask.shape

        if T == target_len:
            return mask

        # Convert mask to float for pooling
        mask_float = mask.float().unsqueeze(1)  # [B, 1, T]

        # Apply pooling
        pooled = F.adaptive_avg_pool1d(mask_float, target_len)  # [B, 1, target_len]
        pooled = pooled.squeeze(1)  # [B, target_len]

        # Convert back to boolean based on method
        if self.mask_downsample_method == "ceil":
            # Valid if any source position was valid (pooled > 0)
            downsampled_mask = pooled > 0
        else:  # "floor"
            # Valid only if all source positions were valid (pooled == 1)
            downsampled_mask = pooled >= (1.0 - 1e-6)  # Account for float precision

        return downsampled_mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], List[int]]:
        """Produce multi-resolution temporal pyramid.

        Args:
            x: Input tensor [B, T, D]
            mask: Optional validity mask [B, T]. If provided, masked positions
                should have corresponding mask values = False.
            lengths: Optional actual lengths [B]. Used for adaptive level selection
                when adaptive=True. If provided, determines which pyramid levels
                are valid based on minimum batch length.

        Returns:
            Tuple of (levels, level_masks, valid_factors):
            - levels: List of [B, T_i, D] tensors at different resolutions
            - level_masks: List of [B, T_i] masks (or None if mask not provided)
            - valid_factors: List of downsample factors actually used

        Examples:
            >>> pyramid = TemporalPyramid([1, 2, 4, 8], adaptive=True)
            >>> x = torch.randn(32, 256, 64)
            >>> levels, _, factors = pyramid(x)
            >>> len(levels)  # 4 (all factors valid)

            >>> # With mask
            >>> mask = torch.ones(32, 256, dtype=torch.bool)
            >>> mask[:, 200:] = False  # Mask last 56 timesteps
            >>> levels, level_masks, factors = pyramid(x, mask)
            >>> len(level_masks)  # 4 (one mask per level)

            >>> # Adaptive with short trajectories
            >>> x_short = torch.randn(32, 4, 64)
            >>> lengths = torch.tensor([4] * 32)
            >>> levels, _, factors = pyramid(x_short, lengths=lengths)
            >>> factors  # [1, 2, 4] (skip 8x factor)
        """
        B, T, D = x.shape

        # Determine valid pyramid levels
        if lengths is not None and self.adaptive:
            # Use minimum length in batch to determine valid levels
            min_T = int(lengths.min().item())
            valid_factors = self.get_valid_levels(min_T)
        else:
            # Use all configured levels
            valid_factors = self.downsample_factors

        # Transpose for pooling
        x_t = x.transpose(1, 2)  # [B, D, T]

        levels = []
        level_masks = [] if mask is not None else None

        for factor in valid_factors:
            target_len = max(self.min_pyramid_length, T // factor)

            # Downsample features
            if target_len == T:
                # No downsampling needed
                pooled = x
            else:
                pool = nn.AdaptiveAvgPool1d(target_len)
                pooled = pool(x_t).transpose(1, 2)  # [B, target_len, D]

            levels.append(pooled)

            # Downsample mask if provided
            if mask is not None:
                downsampled_mask = self._downsample_mask(mask, target_len)
                level_masks.append(downsampled_mask)

        return levels, level_masks, valid_factors
