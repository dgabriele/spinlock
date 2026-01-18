"""TD (Temporal Dynamics) feature extractor.

Extracts temporal features from operator trajectories for VQ-VAE tokenization.
Concatenates SDF per-timestep features + derived temporal curves.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..registry import FeatureRegistry
from .config import TemporalConfig


class TemporalExtractor:
    """Extract TD (Temporal Dynamics) features from trajectories.

    Concatenates per-timestep features (from SDF) + derived temporal curves.
    Output: [N, M, T, D_td] full time series for 1D CNN encoding.

    Design:
        - Reuses SDF per-timestep features (DRY principle)
        - Adds minimal derived curves (energy, variance, smoothness)
        - Preserves full realization dimension M for NOA robustness
        - Outputs full temporal resolution T for 1D CNN encoder

    Args:
        config: TD feature configuration
        device: Torch device for computation

    Example:
        >>> config = TemporalConfig(include_per_timestep=True, include_derived_curves=True)
        >>> extractor = TemporalExtractor(config=config, device=torch.device('cuda'))
        >>> trajectories = torch.randn(100, 3, 500, 3, 128, 128)  # [N, M, T, C, H, W]
        >>> per_timestep = torch.randn(100, 3, 500, 46)  # SDF per-timestep features
        >>> result = extractor.extract_all(trajectories, per_timestep)
        >>> result['sequences'].shape
        torch.Size([100, 3, 500, 49])  # 46 per-timestep + 3 derived
    """

    def __init__(self, config: TemporalConfig, device: torch.device = torch.device("cpu")):
        self.config = config
        self.device = device
        self.registry = self._build_registry()

    def _build_registry(self) -> FeatureRegistry:
        """Build feature registry for TD family.

        Registry tracks feature names, categories, and metadata for HDF5 storage.

        Returns:
            FeatureRegistry with TD features
        """
        registry = FeatureRegistry(family="temporal", version=self.config.version)

        # Per-timestep features (from SDF) - dynamically sized
        if self.config.include_per_timestep:
            # Note: Actual features will be determined from SDF registry during extraction
            # Here we just mark the category as enabled
            for cat in self.config.per_timestep_categories:
                # Placeholder - will be filled during extraction
                pass

        # Derived temporal curves
        if self.config.include_derived_curves:
            for feat in self.config.derived_features:
                registry.add(name=feat, category="derived_temporal")

        return registry

    def extract_all(
        self,
        trajectories: torch.Tensor,
        per_timestep_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract TD features from trajectories.

        Args:
            trajectories: Raw trajectories [N, M, T, C, H, W]
            per_timestep_features: Optional pre-extracted SDF per-timestep features [N, M, T, D_timestep]

        Returns:
            Dictionary containing:
                'sequences': [N, M, T, D_td] full temporal features
                'context': [N, D_context] optional global metadata (mean/std across M and T)
                'registry': FeatureRegistry with metadata

        Example:
            >>> trajectories = torch.randn(10, 3, 500, 3, 128, 128)
            >>> result = extractor.extract_all(trajectories)
            >>> result['sequences'].shape
            torch.Size([10, 3, 500, 3])  # Only derived features if per_timestep=None
        """
        N, M, T, C, H, W = trajectories.shape
        features_list = []

        # 1. Per-timestep features (from SDF if available)
        if self.config.include_per_timestep and per_timestep_features is not None:
            features_list.append(per_timestep_features)

        # 2. Derived temporal curves
        if self.config.include_derived_curves:
            derived = self._extract_derived_curves(trajectories)
            features_list.append(derived)

        # Concatenate along feature dimension
        if len(features_list) == 0:
            raise ValueError("No features to extract (both per_timestep and derived_curves disabled)")

        sequences = torch.cat(features_list, dim=-1)  # [N, M, T, D_td]

        # Optional: Compute global context (operator-level metadata)
        context = None
        if self.config.store_context:
            context = self._compute_context(sequences)

        return {"sequences": sequences, "context": context, "registry": self.registry}

    def _extract_derived_curves(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Extract derived temporal curves from trajectories.

        Computes temporal features that capture dynamics evolution:
        - energy_trajectory: L2 norm per timestep
        - variance_trajectory: Spatial variance per timestep
        - smoothness_trajectory: Gradient magnitude (smoothness proxy)
        - channel_correlation_trajectory: Cross-channel correlation

        Args:
            trajectories: [N, M, T, C, H, W] raw trajectories

        Returns:
            [N, M, T, D_derived] derived temporal features

        Example:
            >>> trajectories = torch.randn(5, 3, 100, 3, 128, 128)
            >>> derived = extractor._extract_derived_curves(trajectories)
            >>> derived.shape  # 3 derived features by default
            torch.Size([5, 3, 100, 3])
        """
        N, M, T, C, H, W = trajectories.shape
        curves = []

        # Reshape for efficient computation: [N*M*T, C, H, W]
        traj_flat = trajectories.reshape(N * M * T, C, H, W)

        for feat_name in self.config.derived_features:
            if feat_name == "energy_trajectory":
                # L2 norm per timestep (energy proxy)
                energy = torch.norm(traj_flat, p=2, dim=(1, 2, 3))  # [N*M*T]
                curves.append(energy.reshape(N, M, T, 1))

            elif feat_name == "variance_trajectory":
                # Spatial variance per timestep (amplitude variability)
                # Mean variance across channels
                variance = torch.var(traj_flat, dim=(2, 3)).mean(dim=1)  # [N*M*T]
                curves.append(variance.reshape(N, M, T, 1))

            elif feat_name == "smoothness_trajectory":
                # Smoothness proxy via spatial variance
                # Higher variance → rougher field → lower smoothness
                smoothness = torch.var(traj_flat, dim=(1, 2, 3))  # [N*M*T]
                curves.append(smoothness.reshape(N, M, T, 1))

            elif feat_name == "channel_correlation_trajectory":
                # Cross-channel correlation per timestep
                if C > 1:
                    corr = self._channel_correlation_per_t(traj_flat, C)
                    curves.append(corr.reshape(N, M, T, 1))
                else:
                    # Single channel - correlation is undefined, use zeros
                    zero_corr = torch.zeros(N, M, T, 1, device=trajectories.device)
                    curves.append(zero_corr)

        if len(curves) == 0:
            raise ValueError("No derived features configured")

        return torch.cat(curves, dim=-1)  # [N, M, T, D_derived]

    def _channel_correlation_per_t(self, traj: torch.Tensor, C: int) -> torch.Tensor:
        """Compute cross-channel correlation per timestep.

        Measures how correlated different channels are at each timestep.
        Higher values → channels evolve similarly (coupled dynamics).

        Args:
            traj: [N*M*T, C, H, W] flattened trajectories
            C: Number of channels

        Returns:
            [N*M*T] mean off-diagonal correlation per timestep

        Example:
            >>> traj = torch.randn(100, 3, 128, 128)
            >>> corr = extractor._channel_correlation_per_t(traj, C=3)
            >>> corr.shape
            torch.Size([100])
        """
        # Flatten spatial dimensions: [N*M*T, C, H*W]
        traj_flat = traj.flatten(2)

        # Mean-center each channel
        traj_centered = traj_flat - traj_flat.mean(dim=2, keepdim=True)

        # Compute covariance matrix: [N*M*T, C, C]
        cov = torch.bmm(traj_centered, traj_centered.transpose(1, 2))

        # Extract off-diagonal elements (cross-channel covariances)
        mask = ~torch.eye(C, dtype=torch.bool, device=traj.device)
        mask_expanded = mask.unsqueeze(0).expand(cov.size(0), -1, -1)

        # Mean of off-diagonal elements
        corr_mean = cov[mask_expanded].reshape(cov.size(0), -1).mean(dim=1)  # [N*M*T]

        return corr_mean

    def _compute_context(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute global context features (operator-level summary).

        Aggregates temporal sequences to operator-level metadata.
        Useful for conditioning or auxiliary losses.

        Args:
            sequences: [N, M, T, D_td] temporal features

        Returns:
            [N, D_context] global metadata (D_context = 2 * D_td)

        Example:
            >>> sequences = torch.randn(10, 3, 500, 56)
            >>> context = extractor._compute_context(sequences)
            >>> context.shape
            torch.Size([10, 112])  # 2 * 56 (mean + std)
        """
        # Compute mean and std across time (T) and realizations (M)
        mean = sequences.mean(dim=(1, 2))  # [N, D_td]
        std = sequences.std(dim=(1, 2))  # [N, D_td]

        # Concatenate: [N, 2*D_td]
        context = torch.cat([mean, std], dim=-1)

        return context
