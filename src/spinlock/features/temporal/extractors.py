"""Temporal Feature Orchestrator (v3.1).

Per-timestep-only feature extraction for online perturbation-based NOA.

v3.1 additions:
- Temporal: skewness, kurtosis, multi-lag autocorrelation, sample entropy,
  enstrophy, divergence
- Spectral: Shannon entropy

Use get_feature_dimensions() to query actual dimension counts dynamically.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Set, Tuple
import numpy as np

from spinlock.features.base import FeatureExtractorBase
from spinlock.features.temporal.spatial import SpatialFeatureExtractor
from spinlock.features.temporal.spectral import SpectralFeatureExtractor
from spinlock.features.temporal.cross_channel import CrossChannelFeatureExtractor
from spinlock.features.temporal.temporal import TemporalFeatureExtractor
from spinlock.features.quantum.quantum import QuantumFeatureExtractor


class TemporalFeatureOrchestrator(FeatureExtractorBase):
    """Per-timestep-only feature orchestrator (v3.1).

    Extracts per-timestep features from four components:
    - Spatial (per-channel statistics)
    - Spectral (frequency domain + entropy)
    - Cross-channel (channel interactions)
    - Enhanced temporal (windowed dynamics, skewness, kurtosis,
                        multi-lag autocorr, sample entropy,
                        enstrophy, divergence)

    This replaces the v2.x SummaryExtractor by focusing exclusively on
    per-timestep features that are computable online.

    Use get_feature_dimensions() to query actual dimension counts.

    Args:
        device: Torch device for computation
        config: TemporalFeatureConfig (optional)

    Example:
        >>> orchestrator = TemporalFeatureOrchestrator(device=torch.device('cuda'))
        >>> trajectories = torch.randn(100, 10, 256, 3, 128, 128)  # [N, M, T, C, H, W]
        >>> features = orchestrator.extract_per_timestep(trajectories)
        >>> print(f"Feature dimensions: {features.shape}")  # [100, 256, D]
    """

    def __init__(
        self,
        device: torch.device,
        config: Optional[Any] = None,
        use_batch_mode: bool = True,
        differentiable: bool = False,
    ):
        """Initialize temporal feature orchestrator.

        Args:
            device: Torch device
            config: TemporalFeatureConfig (optional)
            use_batch_mode: If True, use batch-parallel temporal extraction (GPU-optimized)
            differentiable: When True, preserve gradient through temporal
                feature extraction (needed for contrastive training).
                When False (default), detach windows to save memory.
        """
        self.device = device
        self.config = config
        self.use_batch_mode = use_batch_mode
        self.differentiable = differentiable

        # Initialize sub-extractors
        self.spatial_extractor = SpatialFeatureExtractor(device=device)
        self.spectral_extractor = SpectralFeatureExtractor(device=device)
        self.cross_channel_extractor = CrossChannelFeatureExtractor(device=device)

        # Enhanced temporal extractor with configurable windows
        window_size = 5
        short_window = 5
        medium_window = 20
        long_window = 50

        if config is not None and hasattr(config, 'temporal'):
            window_size = getattr(config.temporal, 'window_size', 5)
            short_window = getattr(config.temporal, 'short_window', 5)
            medium_window = getattr(config.temporal, 'medium_window', 20)
            long_window = getattr(config.temporal, 'long_window', 50)

        # Create sequential extractor (for validation/fallback)
        self.temporal_extractor = TemporalFeatureExtractor(
            device=device,
            window_size=window_size,
            short_window=short_window,
            medium_window=medium_window,
            long_window=long_window,
            differentiable=differentiable,
        )

        # Create batch-parallel extractor if requested
        if use_batch_mode:
            from spinlock.features.temporal.temporal_batch import BatchParallelTemporalExtractor
            self.temporal_extractor_batch = BatchParallelTemporalExtractor(
                device=device,
                window_size=window_size,
                short_window=short_window,
                medium_window=medium_window,
                long_window=long_window,
                differentiable=differentiable,
            )

        # Initialize quantum extractor if enabled (subset of temporal family, registered as category="temporal")
        self.quantum_extractor = None
        if config and hasattr(config, 'quantum') and config.quantum.enabled:
            # Get grid size from trajectories (will be validated during extraction)
            # Default to 64 for now, will be updated on first extract call
            self.quantum_extractor = QuantumFeatureExtractor(
                device=device,
                config=config.quantum,
                grid_size=64  # Will be updated based on actual data
            )

    def _extract_spatial(self, fields: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from mean fields.

        Args:
            fields: [N, T, C, H, W] mean fields (realization-averaged)

        Returns:
            [N, T, D_spatial] spatial features
        """
        N, T = fields.shape[:2]
        spatial_result = self.spatial_extractor.extract(fields)
        if isinstance(spatial_result, dict):
            spatial_features = []
            for key in sorted(spatial_result.keys()):
                feat = spatial_result[key]
                if feat.dim() == 3:
                    spatial_features.append(feat)
                elif feat.dim() == 2:
                    spatial_features.append(feat.unsqueeze(-1))
            spatial = torch.cat(spatial_features, dim=-1) if spatial_features else torch.zeros(N, T, 24, device=self.device)
        else:
            spatial = spatial_result
        if spatial.dim() > 2:
            spatial = spatial.flatten(start_dim=2)
        return spatial

    def _extract_spectral(self, fields: torch.Tensor) -> torch.Tensor:
        """Extract spectral features from mean fields.

        Args:
            fields: [N, T, C, H, W] mean fields

        Returns:
            [N, T, D_spectral] spectral features
        """
        N, T = fields.shape[:2]
        spectral_result = self.spectral_extractor.extract(fields)
        if isinstance(spectral_result, dict):
            spectral_features = []
            for key in sorted(spectral_result.keys()):
                feat = spectral_result[key]
                if feat.dim() == 3:
                    spectral_features.append(feat)
                elif feat.dim() == 2:
                    spectral_features.append(feat.unsqueeze(-1))
            spectral = torch.cat(spectral_features, dim=-1) if spectral_features else torch.zeros(N, T, 27, device=self.device)
        else:
            spectral = spectral_result
        if spectral.dim() > 2:
            spectral = spectral.flatten(start_dim=2)
        return spectral

    def _extract_cross_channel(self, fields: torch.Tensor, num_channels: int) -> Optional[torch.Tensor]:
        """Extract cross-channel features from mean fields.

        Args:
            fields: [N, T, C, H, W] mean fields
            num_channels: Number of channels (C)

        Returns:
            [N, T, D_cross] cross-channel features, or zero-dim tensor if C=1
        """
        N, T = fields.shape[:2]
        if num_channels == 1:
            return torch.zeros(N, T, 0, device=self.device)

        cross_result = self.cross_channel_extractor.extract(fields)
        if isinstance(cross_result, dict):
            cross_features = []
            for key in sorted(cross_result.keys()):
                feat = cross_result[key]
                if feat.dim() == 3 and feat.shape[1] == 1:
                    feat = feat.squeeze(1)
                if feat.dim() == 3:
                    cross_features.append(feat)
                elif feat.dim() == 2:
                    cross_features.append(feat.unsqueeze(-1))
            cross = torch.cat(cross_features, dim=-1) if cross_features else torch.zeros(N, T, 12, device=self.device)
        else:
            cross = cross_result
        if cross.dim() > 2:
            cross = cross.flatten(start_dim=2)
        return cross

    def _extract_temporal(self, fields: torch.Tensor) -> torch.Tensor:
        """Extract enhanced temporal features from mean fields.

        Args:
            fields: [N, T, C, H, W] mean fields

        Returns:
            [N, T, D_temporal] temporal features
        """
        T = fields.shape[1]
        if self.use_batch_mode:
            temporal = self.temporal_extractor_batch.extract_batch(fields)
        else:
            temporal_features = []
            self.temporal_extractor.reset()
            for t in range(T):
                u_t = fields[:, t, :, :, :]
                temporal_t = self.temporal_extractor.extract(u_t)
                temporal_features.append(temporal_t)
            temporal = torch.stack(temporal_features, dim=0).transpose(0, 1)
        return temporal

    def _extract_quantum(self, fields: torch.Tensor, num_channels: int) -> Optional[torch.Tensor]:
        """Extract quantum features from mean fields.

        Args:
            fields: [N, T, C, H, W] mean fields
            num_channels: Number of channels (C)

        Returns:
            [N, T, D_quantum] quantum features, or None if not applicable
        """
        if self.quantum_extractor is None:
            return None
        if num_channels != 2:
            return None

        if fields.shape[-1] != self.quantum_extractor.grid_size:
            self.quantum_extractor.grid_size = fields.shape[-1]
            self.quantum_extractor._setup_grids()

        return self.quantum_extractor.extract(fields)

    def extract_per_timestep(
        self,
        trajectories: torch.Tensor,
        skip_extractors: Optional[Set[str]] = None,
    ) -> torch.Tensor:
        """Extract per-timestep features (v3.1 method).

        Args:
            trajectories: [N, M, T, C, H, W] trajectory tensor
                N: number of operators
                M: number of realizations per operator
                T: number of timesteps
                C: number of channels
                H, W: spatial dimensions
            skip_extractors: Optional set of sub-extractor names to skip.
                Valid names: 'spatial', 'spectral', 'cross_channel', 'temporal', 'quantum'.
                When a sub-extractor is skipped, its features are omitted from output,
                reducing the output dimension accordingly.

        Returns:
            features: [N, T, D] per-timestep features (D depends on skipped extractors)

        Note:
            The realization dimension M is averaged out before extraction,
            as we compute features on the mean field across realizations.
        """
        N, M, T, C, H, W = trajectories.shape

        # Apply channel selection if configured (e.g., density-only for MNO/CNO compatibility)
        if self.config is not None and hasattr(self.config, 'channel_indices') and self.config.channel_indices is not None:
            trajectories = trajectories[:, :, :, self.config.channel_indices, :, :]
            C = len(self.config.channel_indices)

        # Average across realizations: [N, M, T, C, H, W] → [N, T, C, H, W]
        fields = trajectories.mean(dim=1)

        skip = skip_extractors or set()
        feature_list = []

        if 'spatial' not in skip:
            feature_list.append(self._extract_spatial(fields))

        if 'spectral' not in skip:
            feature_list.append(self._extract_spectral(fields))

        if 'cross_channel' not in skip:
            feature_list.append(self._extract_cross_channel(fields, C))

        if 'temporal' not in skip:
            feature_list.append(self._extract_temporal(fields))

        if 'quantum' not in skip:
            quantum = self._extract_quantum(fields, C)
            if quantum is not None:
                feature_list.append(quantum)

        features = torch.cat(feature_list, dim=-1)

        return features

    def extract_all(
        self,
        batch_outputs: torch.Tensor,
        operators: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """Legacy compatibility adapter for pipeline code.

        In v3.0, only per-timestep features exist. This method adapts the old
        interface to the new v3.0 architecture by returning a dict.

        Args:
            batch_outputs: [N, M, T, C, H, W] trajectories
            operators: Optional operator metadata (unused in v3.0)

        Returns:
            Dictionary with keys:
                - per_timestep: [N, T, ~328] features
                - per_trajectory: [N, M, 0] empty (removed in v3.0)
                - aggregated_mean: [N, 0] empty (removed in v3.0)
        """
        per_timestep = self.extract_per_timestep(batch_outputs)
        N = batch_outputs.shape[0]
        M = batch_outputs.shape[1]

        # Return empty tensors for removed features (for backward compatibility)
        return {
            'per_timestep': per_timestep,
            'per_trajectory': torch.zeros(N, M, 0, device=self.device),
            'aggregated_mean': torch.zeros(N, 0, device=self.device),
        }

    def get_feature_ranges(self, num_channels: int) -> Dict[str, range]:
        """Return the feature index range per sub-extractor via dummy forward pass.

        This dynamically determines which indices in the concatenated feature
        vector belong to each sub-extractor. Used by the synthesis pipeline
        to identify which sub-extractors produce features that are entirely
        masked out by the cleaning mask (and can therefore be skipped).

        Args:
            num_channels: Number of channels in the data

        Returns:
            Dict mapping sub-extractor name to range of indices
        """
        # Create a small dummy batch: [N=1, M=2, T=5, C=num_channels, H=64, W=64]
        dummy = torch.zeros(1, 2, 5, num_channels, 64, 64, device=self.device)
        fields = dummy.mean(dim=1)  # [1, T, C, H, W]

        ranges = {}
        offset = 0

        with torch.no_grad():
            # Run each extraction block individually, measure output dim
            for name, extract_fn in [
                ('spatial', lambda: self._extract_spatial(fields)),
                ('spectral', lambda: self._extract_spectral(fields)),
                ('cross_channel', lambda: self._extract_cross_channel(fields, num_channels)),
                ('temporal', lambda: self._extract_temporal(fields)),
                ('quantum', lambda: self._extract_quantum(fields, num_channels)),
            ]:
                feat = extract_fn()
                if feat is not None and feat.shape[-1] > 0:
                    dim = feat.shape[-1]
                    ranges[name] = range(offset, offset + dim)
                    offset += dim

        return ranges

    def get_actual_per_timestep_dim(self, num_channels: int = 3) -> int:
        """Get actual per-timestep feature dimension by runtime extraction.

        Args:
            num_channels: Number of channels in the data (dynamically determined)

        Returns:
            Actual feature dimension (auto-detected)
        """
        # Create a small dummy batch to measure actual dimensions
        # Shape: [N=1, M=2, T=5, C=num_channels, H=64, W=64]
        dummy = torch.zeros(1, 2, 5, num_channels, 64, 64, device=self.device)

        # Extract to get actual shape
        with torch.no_grad():
            features = self.extract_per_timestep(dummy)

        return features.shape[-1]

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each component.

        Returns:
            Dictionary mapping component names to ACTUAL dimensions
        """
        total_dim = self.get_actual_per_timestep_dim()

        return {
            'spatial': 105,      # Actual: stats + gradients + histograms
            'spectral': 93,      # Actual: multi-scale FFT features
            'cross_channel': 10, # Actual: pairwise correlations
            'temporal': 120,     # Actual: enhanced temporal dynamics
            'total': total_dim,  # Auto-detected at runtime
        }

    def reset(self):
        """Reset temporal history buffers.

        Call this between operators or when starting a new trajectory.
        """
        self.temporal_extractor.reset()

    # Abstract method implementations

    @property
    def family_name(self) -> str:
        """Feature family name."""
        return "temporal"

    @property
    def version(self) -> str:
        """Feature family version."""
        return "3.0.0"

    def extract_per_trajectory(
        self,
        trajectories: torch.Tensor,
        metadata: Optional[Any] = None
    ) -> Optional[torch.Tensor]:
        """Extract per-trajectory features.

        In v3.0, trajectory-level features are removed (per-timestep only).
        This method returns None for backward compatibility.

        Args:
            trajectories: [N, M, T, C, H, W]
            metadata: Optional metadata

        Returns:
            None (v3.0 has no trajectory-level features)
        """
        return None

    def aggregate_realizations(
        self,
        per_trajectory_features: Optional[torch.Tensor],
        method: str = "mean"
    ) -> Optional[torch.Tensor]:
        """Aggregate realizations.

        In v3.0, trajectory-level features are removed.
        This method returns None for backward compatibility.

        Args:
            per_trajectory_features: [N, M, D_traj] or None
            method: Aggregation method (unused)

        Returns:
            None (v3.0 has no trajectory-level features)
        """
        return None

    def get_feature_registry(self, num_channels: int = 3):
        """Get feature registry for v3.0+.

        Args:
            num_channels: Number of channels in the data (dynamically determined)

        Returns:
            FeatureRegistry with dynamically determined feature mapping
        """
        from spinlock.features.registry import FeatureRegistry

        registry = FeatureRegistry(family_name="temporal")

        # Register features by category (dynamically based on num_channels)
        # Spatial: (8 features x num_channels)
        for feat in ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'min', 'max', 'range']:
            for ch_idx in range(num_channels):
                registry.register(name=f"spatial_{feat}_ch{ch_idx}", category="spatial")

        # Spectral: (9 features x num_channels)
        for feat in ['peak_freq', 'spectral_centroid', 'spectral_spread', 'spectral_flatness',
                     'spectral_rolloff', 'spectral_entropy', 'power_low', 'power_mid', 'power_high']:
            for ch_idx in range(num_channels):
                registry.register(name=f"spectral_{feat}_ch{ch_idx}", category="spectral")

        # Cross-channel: (4 features x num_channel_pairs)
        # For C channels, we have C*(C-1)/2 unique pairs
        if num_channels >= 2:
            for feat in ['correlation', 'covariance', 'mutual_info', 'phase_sync']:
                for i in range(num_channels):
                    for j in range(i+1, num_channels):
                        registry.register(name=f"cross_channel_{feat}_ch{i}_ch{j}", category="cross_channel")

        # Temporal: 130D (enhanced temporal features)
        # Instantaneous (22D)
        for i in range(22):
            registry.register(name=f"temporal_instantaneous_{i}", category="temporal")

        # Local temporal (28D)
        for i in range(28):
            registry.register(name=f"temporal_local_{i}", category="temporal")

        # Local stability (24D)
        for i in range(24):
            registry.register(name=f"temporal_stability_{i}", category="temporal")

        # Phase space (26D)
        for i in range(26):
            registry.register(name=f"temporal_phase_space_{i}", category="temporal")

        # Multi-scale (30D)
        for i in range(30):
            registry.register(name=f"temporal_multiscale_{i}", category="temporal")

        # Quantum features (10D) - part of temporal family, conditionally enabled
        # Only register if quantum extractor is enabled AND we have 2-channel data (wavefunctions)
        if self.quantum_extractor is not None:
            quantum_feature_names = [
                'purity', 'linear_entropy', 'von_neumann_entropy_approx', 'coherence_measure',
                'position_uncertainty_x', 'position_uncertainty_y',
                'momentum_uncertainty_px', 'momentum_uncertainty_py',
                'uncertainty_product_x', 'uncertainty_product_y'
            ]
            for feat_name in quantum_feature_names:
                # Register as part of temporal category (not separate quantum category)
                registry.register(name=f"temporal_quantum_{feat_name}", category="temporal")

        return registry

    def get_all_feature_names(self, num_channels: int) -> List[str]:
        """Get ordered feature names matching extract_per_timestep() output.

        Uses a dummy forward pass through each sub-extractor to discover the
        actual dict keys (for dict-returning extractors) and their per-channel
        expansion. This is self-describing: if extractors add features, the
        names update automatically.

        For the temporal sub-extractor (which returns a raw tensor, not a dict),
        uses positional names with subcategory prefix (temporal_dynamics_0, ...).

        Args:
            num_channels: Number of channels in the data

        Returns:
            Ordered list of feature names, one per element in the last dim of
            extract_per_timestep() output.
        """
        # Use randn to avoid division-by-zero in variance/correlation computations
        dummy = torch.randn(1, 2, 5, num_channels, 64, 64, device=self.device)
        fields = dummy.mean(dim=1)  # [1, T=5, C, H, W]

        all_names: List[str] = []

        with torch.no_grad():
            # --- Spatial (dict-based, per-channel) ---
            spatial_result = self.spatial_extractor.extract(fields)
            if isinstance(spatial_result, dict):
                for key in sorted(spatial_result.keys()):
                    feat = spatial_result[key]
                    # After reshape inside extractor: [N, T, C] or [N, T]
                    if feat.dim() >= 3 and feat.shape[-1] > 1:
                        for ch in range(feat.shape[-1]):
                            all_names.append(f"{key}_ch{ch}")
                    else:
                        all_names.append(key)

            # --- Spectral (dict-based, per-channel) ---
            spectral_result = self.spectral_extractor.extract(fields)
            if isinstance(spectral_result, dict):
                for key in sorted(spectral_result.keys()):
                    feat = spectral_result[key]
                    if feat.dim() >= 3 and feat.shape[-1] > 1:
                        for ch in range(feat.shape[-1]):
                            all_names.append(f"{key}_ch{ch}")
                    else:
                        all_names.append(key)

            # --- Cross-channel (dict-based, scalar per timestep) ---
            if num_channels >= 2:
                cross_result = self.cross_channel_extractor.extract(fields)
                if isinstance(cross_result, dict):
                    for key in sorted(cross_result.keys()):
                        feat = cross_result[key]
                        # Cross-channel features may be [N, T] (scalar) or
                        # [N, T, something]. Handle like the orchestrator does:
                        # squeeze dim=1 if shape[1]==1, then unsqueeze(-1) if 2D.
                        if feat.dim() == 3 and feat.shape[1] == 1:
                            feat = feat.squeeze(1)
                        if feat.dim() >= 3 and feat.shape[-1] > 1:
                            for ch in range(feat.shape[-1]):
                                all_names.append(f"{key}_ch{ch}")
                        else:
                            all_names.append(key)

            # --- Temporal dynamics (tensor-based, no dict) ---
            temporal_feat = self._extract_temporal(fields)
            D_temporal = temporal_feat.shape[-1]
            for i in range(D_temporal):
                all_names.append(f"temporal_dynamics_{i}")

            # --- Quantum (tensor-based, known names) ---
            quantum = self._extract_quantum(fields, num_channels)
            if quantum is not None and quantum.shape[-1] > 0:
                _quantum_names = [
                    'quantum_purity', 'quantum_linear_entropy',
                    'quantum_von_neumann_entropy', 'quantum_coherence',
                    'quantum_position_uncertainty_x',
                    'quantum_position_uncertainty_y',
                    'quantum_momentum_uncertainty_px',
                    'quantum_momentum_uncertainty_py',
                    'quantum_uncertainty_product_x',
                    'quantum_uncertainty_product_y',
                ]
                D_q = quantum.shape[-1]
                for i in range(D_q):
                    name = _quantum_names[i] if i < len(_quantum_names) else f"quantum_{i}"
                    all_names.append(name)

        return all_names


# Legacy alias for backward compatibility
SummaryExtractor = TemporalFeatureOrchestrator
