"""Multi-resolution pyramid encoder for temporal features.

Uses a shared ResNet-1D backbone (reusing ResidualBlock1D from temporal_cnn.py)
with per-level projection heads to produce multi-scale temporal embeddings.
"""

import torch
import torch.nn as nn
from typing import List, Literal, Optional, Dict, Union, Tuple

from .base import BaseEncoder
from .temporal_cnn import ResidualBlock1D
from ..temporal_pyramid import TemporalPyramid


class PyramidTemporalEncoder(BaseEncoder):
    """Multi-resolution temporal encoder with shared backbone.

    Processes temporal input at multiple resolutions (pyramid levels) through
    a shared ResNet-1D backbone, then projects each level to a different
    embedding dimension via per-level heads.

    Input:  [B, T, D_in]
    Output: [B, sum(level_dims)]  (e.g., [B, 320] for [32, 64, 96, 128])

    Architecture:
        TemporalPyramid → shared backbone (per level) → per-level head → concat

    The shared backbone is:
        Stage 1: Conv1d(D_in→32, k=7, s=2) + BN + ReLU + MaxPool
        Stage 2: ResBlock1D(32→64, s=2)
        Stage 3: ResBlock1D(64→128, s=2)
        Stage 4: ResBlock1D(128→256, s=2)
        GAP → [B, 256]

    Each per-level head: Linear(256→level_dim) + BatchNorm1d(level_dim)

    Attributes:
        output_dims_per_level: List[int] of per-level output dimensions.
            Used by train_vqvae.py to detect pyramid encoder and split
            features into per-level families.

    Args:
        input_dim: Per-timestep feature dimension (D_in)
        level_dims: Output dimensions for each pyramid level.
            Default: [32, 64, 96, 128] (total 320D)
        downsample_factors: Temporal downsampling factors for pyramid.
            Default: [1, 2, 4, 8] (full, half, quarter, eighth resolution)
        architecture: Backbone architecture variant.
            Only 'resnet1d_3' supported.
        variable_length_config: Optional config dict for variable-length support.
            If provided, enables adaptive pyramid levels and masking.
            Expected keys:
                - enabled (bool): Enable variable-length mode
                - adaptive_pyramid (bool): Auto-skip invalid pyramid levels
                - min_pyramid_length (int): Minimum T_i at any level
                - mask_downsample_method (str): "ceil" or "floor"

    Example:
        >>> encoder = PyramidTemporalEncoder(input_dim=345, level_dims=[32, 64, 96, 128])
        >>> x = torch.randn(16, 256, 345)  # [B, T, D_in]
        >>> out = encoder(x)               # [B, 320]
        >>> out.shape
        torch.Size([16, 320])

        >>> # Variable-length mode
        >>> vl_config = {"enabled": True, "adaptive_pyramid": True}
        >>> encoder = PyramidTemporalEncoder(input_dim=345, variable_length_config=vl_config)
        >>> mask = torch.ones(16, 256, dtype=torch.bool)
        >>> lengths = torch.tensor([256] * 16)
        >>> out, mask_info = encoder(x, mask=mask, lengths=lengths)
        >>> out.shape  # [16, 320] (padded to full dimension)
    """

    def __init__(
        self,
        input_dim: int,
        level_dims: List[int] = [32, 64, 96, 128],
        downsample_factors: List[int] = [1, 2, 4, 8],
        architecture: Literal["resnet1d_3"] = "resnet1d_3",
        variable_length_config: Optional[Dict] = None,
    ):
        super().__init__()

        if architecture != "resnet1d_3":
            raise ValueError(f"Only 'resnet1d_3' architecture supported, got: {architecture}")

        if len(level_dims) != len(downsample_factors):
            raise ValueError(
                f"level_dims ({len(level_dims)}) must match "
                f"downsample_factors ({len(downsample_factors)})"
            )

        self._input_dim = input_dim
        self.output_dims_per_level = list(level_dims)
        self.downsample_factors = downsample_factors
        self._output_dim = sum(level_dims)

        # Parse variable-length config
        self.vl_config = variable_length_config or {}
        self.vl_enabled = self.vl_config.get("enabled", False)

        # Multi-resolution temporal pyramid
        if self.vl_enabled:
            self.pyramid = TemporalPyramid(
                downsample_factors,
                mask_downsample_method=self.vl_config.get("mask_downsample_method", "ceil"),
                adaptive=self.vl_config.get("adaptive_pyramid", True),
                min_pyramid_length=self.vl_config.get("min_pyramid_length", 1),
            )
        else:
            # Backward compatible: no adaptive features
            self.pyramid = TemporalPyramid(downsample_factors, adaptive=False)

        # Shared backbone (same architecture as TemporalCNNEncoder)
        self.stage1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage2 = ResidualBlock1D(32, 64, stride=2)
        self.stage3 = ResidualBlock1D(64, 128, stride=2)
        self.stage4 = ResidualBlock1D(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Per-level projection heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, dim),
                nn.BatchNorm1d(dim),
            )
            for dim in level_dims
        ])

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run shared backbone on a single pyramid level.

        Args:
            x: [B, D_in, T_i] (already transposed)

        Returns:
            [B, 256] backbone features
        """
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.gap(h).squeeze(-1)  # [B, 256]
        return h

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Encode temporal input at multiple resolutions.

        Args:
            x: Input sequences [B, T, D_in]
            mask: Optional validity mask [B, T]. False indicates invalid positions.
            lengths: Optional actual lengths [B] for adaptive pyramid levels.

        Returns:
            If mask is None (standard mode):
                embeddings [B, sum(level_dims)]

            If mask provided (variable-length mode):
                (embeddings [B, output_dim], mask_info dict)

                embeddings are zero-padded to total_output_dim if some pyramid
                levels are skipped due to short trajectories.

                mask_info contains:
                    - original_mask: Input mask [B, T]
                    - level_masks: List of downsampled masks [B, T_i]
                    - valid_factors: List of factors actually used
                    - num_active_levels: Number of pyramid levels used
                    - output_dim: Actual embedding dimension before padding
                    - num_valid_timesteps: Number of valid timesteps per sample [B]

        Examples:
            >>> # Standard mode (backward compatible)
            >>> encoder = PyramidTemporalEncoder(input_dim=64)
            >>> x = torch.randn(16, 256, 64)
            >>> out = encoder(x)
            >>> out.shape  # torch.Size([16, 320])

            >>> # Variable-length mode
            >>> vl_config = {"enabled": True, "adaptive_pyramid": True}
            >>> encoder = PyramidTemporalEncoder(input_dim=64, variable_length_config=vl_config)
            >>> mask = torch.ones(16, 256, dtype=torch.bool)
            >>> mask[:, 200:] = False  # Last 56 timesteps invalid
            >>> lengths = torch.full((16,), 200)
            >>> out, info = encoder(x, mask=mask, lengths=lengths)
            >>> out.shape  # torch.Size([16, 320]) (padded)
            >>> info["num_active_levels"]  # 4 (all levels valid for T=200)
        """
        # Create pyramid levels
        pyramid_levels, level_masks, valid_factors = self.pyramid(x, mask, lengths)

        # Process each active level
        level_embeddings = []

        for level_idx, level_input in enumerate(pyramid_levels):
            # Find corresponding head (match factor to original index)
            factor = valid_factors[level_idx]
            head_idx = self.downsample_factors.index(factor)
            head = self.heads[head_idx]

            # Apply mask if provided
            if level_masks is not None:
                level_mask = level_masks[level_idx]  # [B, T_i]
                expanded_mask = level_mask.unsqueeze(-1).float()  # [B, T_i, 1]
                level_input = level_input * expanded_mask

            # Process through backbone
            h = level_input.transpose(1, 2)  # [B, D_in, T_i]
            h = self._backbone(h)  # [B, 256]
            h = head(h)  # [B, level_dim]

            level_embeddings.append(h)

        # Concatenate active levels
        embeddings = torch.cat(level_embeddings, dim=1)  # [B, sum(active_level_dims)]

        # Handle variable-dimension output
        if mask is not None:
            # Pad to full dimension if some levels skipped
            embeddings = self._pad_to_full_dim(embeddings, valid_factors)

            # Compute number of valid timesteps per sample
            num_valid = mask.sum(dim=1)  # [B]

            mask_info = {
                "original_mask": mask,
                "level_masks": level_masks,
                "valid_factors": valid_factors,
                "num_active_levels": len(valid_factors),
                "output_dim": embeddings.shape[1],
                "num_valid_timesteps": num_valid,
            }
            return embeddings, mask_info
        else:
            return embeddings

    def forward_per_level(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return per-level embeddings separately.

        Useful for per-level naming, auxiliary losses, or diagnostics.

        Args:
            x: Input sequences [B, T, D_in]
            mask: Optional validity mask [B, T]
            lengths: Optional actual lengths [B]

        Returns:
            List of [B, level_dim] tensors, one per active pyramid level
        """
        pyramid_levels, level_masks, valid_factors = self.pyramid(x, mask, lengths)

        level_embeddings = []
        for level_idx, level_input in enumerate(pyramid_levels):
            factor = valid_factors[level_idx]
            head_idx = self.downsample_factors.index(factor)
            head = self.heads[head_idx]

            # Apply mask if provided
            if level_masks is not None:
                level_mask = level_masks[level_idx]
                expanded_mask = level_mask.unsqueeze(-1).float()
                level_input = level_input * expanded_mask

            h = level_input.transpose(1, 2)
            h = self._backbone(h)
            h = head(h)
            level_embeddings.append(h)

        return level_embeddings

    def _pad_to_full_dim(
        self,
        embeddings: torch.Tensor,
        valid_factors: List[int]
    ) -> torch.Tensor:
        """Pad variable-dimension embeddings to full dimension.

        When some pyramid levels are skipped (e.g., T=4 can't use 8x factor),
        the output dimension is smaller than total_output_dim. This pads with
        zeros to maintain a consistent dimension.

        Args:
            embeddings: [B, active_dim] where active_dim = sum of dims for valid_factors
            valid_factors: List of factors actually used

        Returns:
            Padded embeddings [B, total_output_dim]

        Example:
            >>> # All levels: [32, 64, 96, 128] = 320D
            >>> # Only [1, 2, 4]: [32, 64, 96] = 192D → pad 128 zeros
        """
        B, active_dim = embeddings.shape

        if active_dim == self._output_dim:
            return embeddings  # No padding needed

        # Zero-pad missing levels
        padding = torch.zeros(
            B, self._output_dim - active_dim,
            device=embeddings.device,
            dtype=embeddings.dtype
        )

        return torch.cat([embeddings, padding], dim=1)

    @property
    def total_output_dim(self) -> int:
        """Maximum output dimension when all pyramid levels are active."""
        return self._output_dim

    @property
    def output_dim(self) -> int:
        """Total output dimension (sum of all level dims)."""
        return self._output_dim
