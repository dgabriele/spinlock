"""
Learned SUMMARY Feature Extractor.

Extracts features from U-AFNO intermediate representations (bottleneck and skips).

Pipeline:
1. Collect latents from U-AFNO during forward pass
2. Temporal aggregation: pool across T timesteps
3. Spatial aggregation: global average pooling
4. Optional projection: MLP to fixed dimension

Shape conventions:
- Input trajectories: [N, M, T, C, H, W] (N operators, M realizations, T timesteps)
- Latents per timestep: [B, C_latent, H', W']
- Output features: [N, D_learned]

Author: Claude (Anthropic)
Date: January 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spinlock.features.summary.config import LearnedSummaryConfig


class LearnedSummaryExtractor:
    """
    Extract learned features from U-AFNO latent representations.

    Implements the aggregation pipeline:
    1. Per-timestep: Extract latents via operator.get_intermediate_features()
    2. Temporal: Pool across T (mean, max, or mean+max)
    3. Spatial: Global average pooling (GAP)
    4. Realization: Mean across M realizations
    5. Optional: Project via MLP to fixed dimension

    Attributes:
        device: Torch device
        config: LearnedSummaryConfig
        projection_mlp: Optional MLP for projection

    Example:
        >>> from spinlock.operators.u_afno import UAFNOOperator
        >>> from spinlock.features.summary.config import LearnedSummaryConfig
        >>>
        >>> config = LearnedSummaryConfig(enabled=True, extract_from="bottleneck")
        >>> extractor = LearnedSummaryExtractor(device=torch.device('cuda'), config=config)
        >>>
        >>> operator = UAFNOOperator(...)
        >>> trajectories = torch.randn(3, 10, 3, 64, 64)  # [M, T, C, H, W]
        >>> features = extractor.extract_from_operator(operator, trajectories)
        >>> print(features.shape)  # [D_learned]
    """

    def __init__(
        self,
        device: torch.device,
        config: Optional[LearnedSummaryConfig] = None,
    ):
        """
        Initialize learned feature extractor.

        Args:
            device: Torch device for computation
            config: LearnedSummaryConfig (uses defaults if None)
        """
        self.device = device
        self.config = config
        self.projection_mlp: Optional[nn.Module] = None

        # Projection MLP will be initialized lazily once we know latent dimension
        self._projection_initialized = False
        self._latent_dim: Optional[int] = None

    def _init_projection_mlp(self, input_dim: int) -> None:
        """
        Initialize projection MLP once latent dimension is known.

        Args:
            input_dim: Input dimension (total latent features after aggregation)
        """
        if self.config is None or self.config.projection_dim is None:
            return

        self.projection_mlp = nn.Sequential(
            nn.Linear(input_dim, self.config.projection_dim * 2),
            nn.GELU(),
            nn.Linear(self.config.projection_dim * 2, self.config.projection_dim),
            nn.LayerNorm(self.config.projection_dim),
        ).to(self.device)

        self._projection_initialized = True
        self._latent_dim = input_dim

    def extract_from_operator(
        self,
        operator: nn.Module,
        trajectories: torch.Tensor,  # [M, T, C, H, W] for single operator
    ) -> torch.Tensor:
        """
        Extract learned features from operator latents over trajectory.

        Args:
            operator: U-AFNO operator (must have get_intermediate_features method)
            trajectories: Trajectory tensor [M, T, C, H, W]

        Returns:
            Features [D_learned] (aggregated across M realizations and T timesteps)

        Raises:
            ValueError: If operator doesn't support intermediate feature extraction
        """
        if not hasattr(operator, "get_intermediate_features"):
            raise ValueError(
                "Operator must have get_intermediate_features() method. "
                "Only U-AFNO operators are supported for learned feature extraction."
            )

        M, T, C, H, W = trajectories.shape
        config = self.config

        # Determine extraction settings
        extract_from = config.extract_from if config else "bottleneck"
        skip_levels = config.skip_levels if config else [0, 1, 2]

        # Collect latents for all timesteps
        # Process in batches for memory efficiency: [M*T, C, H, W]
        trajectories_flat = trajectories.reshape(M * T, C, H, W)

        with torch.no_grad():
            latents_dict = operator.get_intermediate_features(
                trajectories_flat,
                extract_from=extract_from,
                skip_levels=skip_levels,
            )

        # Apply spatial aggregation (GAP) to each latent type
        aggregated_latents = []
        for key in sorted(latents_dict.keys()):  # Sort for deterministic ordering
            latent = latents_dict[key]
            # latent: [M*T, C_latent, H', W']
            if config is None or config.spatial_agg == "gap":
                # Global average pooling
                pooled = latent.mean(dim=(-2, -1))  # [M*T, C_latent]
            else:
                # Flatten (expensive, use with caution)
                pooled = latent.flatten(start_dim=1)  # [M*T, C_latent*H'*W']

            # Reshape to [M, T, C_latent]
            pooled = pooled.reshape(M, T, -1)
            aggregated_latents.append(pooled)

        # Concatenate all latent types: [M, T, D_total]
        latents_combined = torch.cat(aggregated_latents, dim=-1)

        # Temporal aggregation: [M, T, D] -> [M, D']
        temporal_agg = config.temporal_agg if config else "mean_max"

        if temporal_agg == "mean":
            temporal_pooled = latents_combined.mean(dim=1)  # [M, D]
        elif temporal_agg == "max":
            temporal_pooled = latents_combined.max(dim=1)[0]  # [M, D]
        elif temporal_agg == "mean_max":
            mean_pooled = latents_combined.mean(dim=1)  # [M, D]
            max_pooled = latents_combined.max(dim=1)[0]  # [M, D]
            temporal_pooled = torch.cat([mean_pooled, max_pooled], dim=-1)  # [M, 2*D]
        elif temporal_agg == "std":
            temporal_pooled = latents_combined.std(dim=1)  # [M, D]
        else:
            raise ValueError(f"Unknown temporal aggregation: {temporal_agg}")

        # Realization aggregation: [M, D'] -> [D']
        # Use mean across realizations (consistent with manual features)
        realization_pooled = temporal_pooled.mean(dim=0)  # [D']

        # Optional projection
        if config is not None and config.projection_dim is not None:
            if not self._projection_initialized:
                self._init_projection_mlp(realization_pooled.shape[0])
            realization_pooled = self.projection_mlp(realization_pooled)

        return realization_pooled

    def extract_batch(
        self,
        operators: List[nn.Module],
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
    ) -> torch.Tensor:
        """
        Extract learned features for batch of operators.

        Args:
            operators: List of N U-AFNO operators
            trajectories: Trajectories [N, M, T, C, H, W]

        Returns:
            Features [N, D_learned]

        Raises:
            ValueError: If batch size mismatch
        """
        N = len(operators)
        if trajectories.shape[0] != N:
            raise ValueError(
                f"Number of operators ({N}) must match batch dimension "
                f"of trajectories ({trajectories.shape[0]})"
            )

        features_list = []
        for i, operator in enumerate(operators):
            traj_i = trajectories[i]  # [M, T, C, H, W]
            feat_i = self.extract_from_operator(operator, traj_i)
            features_list.append(feat_i)

        # Handle variable dimensions across operators
        # Different architectures may produce different latent sizes
        dims = [f.shape[0] for f in features_list]
        max_dim = max(dims)

        # Pad features to max dimension if needed
        if len(set(dims)) > 1:
            padded_features = []
            for feat in features_list:
                if feat.shape[0] < max_dim:
                    padding = torch.zeros(
                        max_dim - feat.shape[0],
                        device=feat.device,
                        dtype=feat.dtype
                    )
                    feat = torch.cat([feat, padding], dim=0)
                padded_features.append(feat)
            features_list = padded_features

        return torch.stack(features_list, dim=0)  # [N, D_learned]

    def get_feature_dim(
        self,
        operator: nn.Module,
        input_shape: tuple = (3, 64, 64),
    ) -> int:
        """
        Get output feature dimension (for registry/storage pre-allocation).

        Args:
            operator: Sample U-AFNO operator to probe dimensions
            input_shape: (C, H, W) input shape for probing

        Returns:
            Feature dimension after all aggregation steps
        """
        C, H, W = input_shape
        dummy_input = torch.zeros(1, C, H, W, device=self.device)

        with torch.no_grad():
            latents = operator.get_intermediate_features(
                dummy_input,
                extract_from=self.config.extract_from if self.config else "bottleneck",
                skip_levels=self.config.skip_levels if self.config else [0, 1, 2],
            )

        total_dim = 0
        for key, latent in latents.items():
            if self.config is None or self.config.spatial_agg == "gap":
                total_dim += latent.shape[1]  # C_latent
            else:
                total_dim += latent.numel() // latent.shape[0]  # Flattened

        # Account for temporal aggregation
        if self.config is None or self.config.temporal_agg == "mean_max":
            total_dim *= 2

        # Account for projection
        if self.config is not None and self.config.projection_dim is not None:
            return self.config.projection_dim

        return total_dim

    @property
    def is_enabled(self) -> bool:
        """Check if learned feature extraction is enabled."""
        return self.config is not None and self.config.enabled
