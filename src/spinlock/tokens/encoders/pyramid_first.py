"""Pyramid-first temporal encoder with learned group projection.

Builds the temporal pyramid on the RAW trajectory before the CNN, so the CNN
sees multi-resolution spatial data.  Replaces sequential slicing with a learned
linear projection for group assignment.

Architecture:
    trajectory [B, T, C, H, W]
        │
        ▼  SpatioTemporalPyramid (downsample raw trajectory along T)
        ├── Level 0: [B, T,   C, H, W]
        ├── Level 1: [B, T/2, C, H, W]
        └── ...
        │
        ▼  shared FrameCNNEncoder (gradient-checkpointed)
        ├── Level 0: [B, T,   D_cnn]
        └── ...
        │
        ▼  per-level TemporalAggregator (Conv1d+BN+ReLU+GAP)
        ├── Level 0: [B, D_agg]
        └── ...
        │
        ▼  concat → [B, num_levels × D_agg]
        │
        ▼  LearnedGroupProjection (Linear + reshape + LayerNorm)
        [B, num_groups, D_group]
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spinlock.features.initial.cnn_encoder import FrameCNNEncoder

logger = logging.getLogger(__name__)


class SpatioTemporalPyramid(nn.Module):
    """Downsample raw trajectory [B, T, C, H, W] at multiple temporal resolutions.

    Uses per-pixel temporal average pooling: reshapes to [B, T, C*H*W], applies
    F.adaptive_avg_pool1d along T, then reshapes back.  No learnable parameters.

    Supports adaptive level selection for variable-length trajectories (same logic
    as the existing TemporalPyramid).

    Args:
        downsample_factors: Temporal downsampling factors (e.g. [1, 2, 4, 8]).
        adaptive: Auto-skip levels where T_i < min_pyramid_length.
        min_pyramid_length: Minimum sequence length at any pyramid level.
        mask_downsample_method: "ceil" or "floor" for mask propagation.
    """

    def __init__(
        self,
        downsample_factors: List[int],
        adaptive: bool = True,
        min_pyramid_length: int = 1,
        mask_downsample_method: str = "ceil",
    ):
        super().__init__()
        self.downsample_factors = downsample_factors
        self.adaptive = adaptive
        self.min_pyramid_length = max(1, min_pyramid_length)
        self.mask_downsample_method = mask_downsample_method

    def get_valid_levels(self, T: int) -> List[int]:
        """Return downsample factors that produce T_i >= min_pyramid_length."""
        if not self.adaptive:
            return self.downsample_factors
        valid = [f for f in self.downsample_factors if T // f >= self.min_pyramid_length]
        return valid if valid else [1]

    def _downsample_mask(self, mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Downsample boolean mask [B, T] → [B, target_len]."""
        B, T = mask.shape
        if T == target_len:
            return mask
        pooled = F.adaptive_avg_pool1d(
            mask.float().unsqueeze(1), target_len
        ).squeeze(1)
        if self.mask_downsample_method == "ceil":
            return pooled > 0
        return pooled >= (1.0 - 1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], List[int]]:
        """Produce multi-resolution spatial trajectory pyramid.

        Args:
            x: Raw trajectory [B, T, C, H, W].
            mask: Validity mask [B, T] (optional).
            lengths: Actual sequence lengths [B] (optional).

        Returns:
            (levels, level_masks, valid_factors) where:
            - levels: list of [B, T_i, C, H, W] tensors
            - level_masks: list of [B, T_i] masks (or None)
            - valid_factors: downsample factors actually used
        """
        B, T, C, H, W = x.shape

        if lengths is not None and self.adaptive:
            min_T = int(lengths.min().item())
            valid_factors = self.get_valid_levels(min_T)
        else:
            valid_factors = self.downsample_factors

        # Flatten spatial dims for temporal pooling: [B, T, C*H*W]
        flat = x.reshape(B, T, C * H * W)
        flat_t = flat.transpose(1, 2)  # [B, C*H*W, T]

        levels = []
        level_masks = [] if mask is not None else None

        for factor in valid_factors:
            target_len = max(self.min_pyramid_length, T // factor)

            if target_len == T:
                levels.append(x)
            else:
                pooled = F.adaptive_avg_pool1d(flat_t, target_len)  # [B, C*H*W, T_i]
                pooled = pooled.transpose(1, 2).reshape(B, target_len, C, H, W)
                levels.append(pooled)

            if mask is not None:
                level_masks.append(self._downsample_mask(mask, target_len))

        return levels, level_masks, valid_factors


class TemporalAggregator(nn.Module):
    """Aggregate per-frame CNN embeddings along time: [B, T_i, D_cnn] → [B, D_agg].

    Conv1d(D_cnn, D_agg, k=3, pad=1) + BatchNorm1d + ReLU + AdaptiveAvgPool1d(1).
    One instance per pyramid level (separate weights capture level-specific
    temporal patterns).

    Args:
        d_cnn: Input feature dimension (CNN output dim per frame).
        d_agg: Output aggregated dimension.
    """

    def __init__(self, d_cnn: int, d_agg: int):
        super().__init__()
        self.conv = nn.Conv1d(d_cnn, d_agg, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(d_agg)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate temporal features.

        Args:
            x: [B, T_i, D_cnn] per-frame CNN features at one pyramid level.
            mask: Optional [B, T_i] validity mask.

        Returns:
            [B, D_agg] aggregated features.
        """
        if mask is not None:
            # Zero invalid positions before conv
            x = x * mask.unsqueeze(-1).float()

        x = x.transpose(1, 2)  # [B, D_cnn, T_i]
        x = self.conv(x)       # [B, D_agg, T_i]
        x = self.bn(x)
        x = self.relu(x)
        x = self.gap(x)        # [B, D_agg, 1]
        return x.squeeze(-1)   # [B, D_agg]


class LearnedGroupProjection(nn.Module):
    """Project multi-resolution features to per-group embeddings with optional gating.

    Linear(D_multi_res, num_groups * D_group) + reshape + LayerNorm(D_group).
    LayerNorm normalizes each group's features independently.

    When ``gated=True``, each group has a learnable scalar gate ``σ(g_i)`` ∈ [0,1].
    During tokenizer pretraining all gates stay open; during downstream joint
    fine-tuning (MNO roundtrip, contrastive), an L1 penalty on the gates
    encourages the model to concentrate information into the useful groups,
    letting the effective group count emerge from the optimization.

    Args:
        d_multi_res: Input dimension (num_levels * D_agg).
        num_groups: Number of VQ groups.
        d_group: Output dimension per group.
        gated: Enable per-group learnable gates.
        gate_init_bias: Initial logit bias (higher = gates start more open).
    """

    def __init__(
        self,
        d_multi_res: int,
        num_groups: int,
        d_group: int,
        gated: bool = False,
        gate_init_bias: float = 5.0,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.d_group = d_group
        self.gated = gated
        self.linear = nn.Linear(d_multi_res, num_groups * d_group)
        self.ln = nn.LayerNorm(d_group)

        if gated:
            # Learnable logits; sigmoid(5.0) ≈ 0.993 → gates start nearly open
            self.gate_logits = nn.Parameter(torch.full((num_groups,), gate_init_bias))
        else:
            self.gate_logits = None

    def get_gate_values(self) -> Optional[torch.Tensor]:
        """Return current gate activations [num_groups] in [0,1], or None."""
        if self.gate_logits is None:
            return None
        return torch.sigmoid(self.gate_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to per-group embeddings.

        Args:
            x: [B, D_multi_res] concatenated multi-resolution features.

        Returns:
            [B, num_groups, D_group] per-group features (gate-scaled if enabled).
        """
        B = x.shape[0]
        h = self.linear(x)                             # [B, G * D_group]
        h = h.reshape(B, self.num_groups, self.d_group) # [B, G, D_group]
        h = self.ln(h)                                  # LayerNorm per-group

        if self.gate_logits is not None:
            gates = torch.sigmoid(self.gate_logits)     # [G]
            h = h * gates.unsqueeze(0).unsqueeze(-1)    # [B, G, D_group] * [1, G, 1]

        return h


class PyramidFirstEncoder(nn.Module):
    """Top-level pyramid-first temporal encoder.

    Composes SpatioTemporalPyramid, shared FrameCNNEncoder, per-level
    TemporalAggregators, and LearnedGroupProjection into a single encoder.

    Args:
        in_channels: Number of input channels per frame (auto-detected).
        d_cnn: CNN output embedding dimension per frame.
        d_agg: Per-level temporal aggregation dimension.
        num_groups: Number of VQ groups.
        d_group: Output dimension per group (= encoder.embedding_dim).
        downsample_factors: Temporal pyramid downsample factors.
        adaptive: Enable adaptive pyramid level selection.
        min_pyramid_length: Minimum T at any pyramid level.
        mask_downsample_method: "ceil" or "floor" for mask propagation.
    """

    BASE_CHUNK_SIZE = 128  # Frames per CNN forward pass at 64×64 resolution

    def __init__(
        self,
        in_channels: int,
        d_cnn: int = 256,
        d_agg: int = 128,
        num_groups: int = 20,
        d_group: int = 64,
        downsample_factors: List[int] = [1, 2, 4, 8],
        adaptive: bool = True,
        min_pyramid_length: int = 1,
        mask_downsample_method: str = "ceil",
        gated_groups: bool = False,
        gate_init_bias: float = 5.0,
    ):
        super().__init__()

        self.num_groups = num_groups
        self.d_group = d_group
        self.d_cnn = d_cnn
        self.d_agg = d_agg
        self.downsample_factors = downsample_factors

        # Spatial-temporal pyramid (no learnable params)
        self.pyramid = SpatioTemporalPyramid(
            downsample_factors=downsample_factors,
            adaptive=adaptive,
            min_pyramid_length=min_pyramid_length,
            mask_downsample_method=mask_downsample_method,
        )

        # Shared per-frame CNN encoder
        self.frame_cnn = FrameCNNEncoder(
            embedding_dim=d_cnn,
            in_channels=in_channels,
        )

        # Per-level temporal aggregators (separate weights per level)
        num_levels = len(downsample_factors)
        self.aggregators = nn.ModuleList([
            TemporalAggregator(d_cnn, d_agg) for _ in range(num_levels)
        ])

        # Learned group projection (with optional gating)
        d_multi_res = num_levels * d_agg
        self.group_proj = LearnedGroupProjection(
            d_multi_res, num_groups, d_group,
            gated=gated_groups, gate_init_bias=gate_init_bias,
        )

        gate_str = " (gated)" if gated_groups else ""
        logger.info(
            "PyramidFirstEncoder: %d levels × %dD_agg → %dD → %d groups × %dD_group%s",
            num_levels, d_agg, d_multi_res, num_groups, d_group, gate_str,
        )

    def _get_chunk_size(self, H: int, W: int) -> int:
        """Scale chunk size inversely with spatial area to keep per-chunk memory constant."""
        ref_area = 64 * 64  # reference resolution
        return max(16, self.BASE_CHUNK_SIZE * ref_area // max(1, H * W))

    def _encode_frames_chunked(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode flattened frames [N, C, H, W] → [N, D_cnn] with gradient checkpointing.

        Streams CPU frames to model device in chunks.  Crucially, the CPU tensor
        is passed to ``checkpoint()`` so that backward saves the small CPU tensor
        rather than a GPU copy — this avoids O(num_chunks) GPU memory for saved
        inputs.  Chunk size scales inversely with spatial area so per-chunk GPU
        memory stays constant regardless of grid resolution.
        """
        N = frames.shape[0]
        device = next(self.frame_cnn.parameters()).device
        H, W = frames.shape[-2:]
        chunk_size = self._get_chunk_size(H, W)

        if N <= chunk_size:
            return self.frame_cnn(frames.to(device))

        frame_cnn = self.frame_cnn  # local ref for closure

        def _forward_on_device(x: torch.Tensor) -> torch.Tensor:
            """Transfer chunk to GPU inside checkpoint boundary."""
            return frame_cnn(x.to(device))

        chunks = []
        for i in range(0, N, chunk_size):
            chunk_cpu = frames[i : i + chunk_size]
            if self.training:
                # Pass CPU tensor to checkpoint — backward saves the CPU tensor
                # and re-transfers to GPU only during recomputation.
                out = checkpoint(_forward_on_device, chunk_cpu, use_reentrant=False)
            else:
                out = frame_cnn(chunk_cpu.to(device))
            chunks.append(out)
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode raw trajectory to per-group embeddings.

        Args:
            x: Raw trajectory [B, T, C, H, W] (may reside on CPU).
            mask: Validity mask [B, T] (optional).
            lengths: Actual sequence lengths [B] (optional).

        Returns:
            (per_group, multi_res) where:
            - per_group: [B, num_groups, D_group] per-group features
            - multi_res: [B, num_levels * D_agg] concatenated level features
              (used for balance loss computation)
        """
        B = x.shape[0]
        device = next(self.frame_cnn.parameters()).device

        # 1. Build spatio-temporal pyramid on raw trajectory
        levels, level_masks, valid_factors = self.pyramid(x, mask, lengths)

        # 2. Per-level: CNN encode all frames → temporal aggregation
        level_embeddings = []
        all_num_levels = len(self.downsample_factors)

        for level_idx, (level_data, factor) in enumerate(zip(levels, valid_factors)):
            # level_data: [B, T_i, C, H, W]
            B_l, T_i, C, H, W = level_data.shape

            # Flatten frames for CNN: [B*T_i, C, H, W]
            flat_frames = level_data.reshape(B_l * T_i, C, H, W)

            # CNN encode with gradient checkpointing
            cnn_out = self._encode_frames_chunked(flat_frames)  # [B*T_i, D_cnn]
            cnn_out = cnn_out.reshape(B_l, T_i, self.d_cnn)    # [B, T_i, D_cnn]

            # Find aggregator for this factor's original index
            agg_idx = self.downsample_factors.index(factor)
            aggregator = self.aggregators[agg_idx]

            # Temporal aggregation with mask
            level_mask = level_masks[level_idx] if level_masks is not None else None
            agg_out = aggregator(cnn_out, mask=level_mask)  # [B, D_agg]
            level_embeddings.append(agg_out)

        # Pad skipped levels with zeros (variable-length: some factors may be invalid)
        if len(level_embeddings) < all_num_levels:
            active_dim = sum(e.shape[1] for e in level_embeddings)
            total_dim = all_num_levels * self.d_agg
            padding = torch.zeros(
                B, total_dim - active_dim, device=device, dtype=level_embeddings[0].dtype
            )
            multi_res = torch.cat(level_embeddings + [padding], dim=1)
        else:
            multi_res = torch.cat(level_embeddings, dim=1)  # [B, num_levels * D_agg]

        # 3. Learned group projection
        per_group = self.group_proj(multi_res)  # [B, num_groups, D_group]

        return per_group, multi_res
