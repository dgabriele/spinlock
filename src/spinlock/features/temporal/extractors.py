"""
Temporal Feature Extraction - main orchestrator.

Coordinates extraction of per-timestep temporal features:
- Spatial statistics (24D)
- Spectral/frequency features (27D)
- Cross-channel interactions (12D)
- Enhanced temporal dynamics (130D)

Total: 193D per-timestep features for online perturbation-based NOA.

Example:
    >>> from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator
    >>> from spinlock.features.temporal.config import TemporalFeatureConfig
    >>>
    >>> extractor = TemporalFeatureOrchestrator(device='cuda', config=TemporalFeatureConfig())
    >>> trajectories = torch.randn(32, 10, 100, 3, 128, 128, device='cuda')
    >>> features = extractor.extract_per_timestep(trajectories)  # [32, 100, 193]
"""

import torch
from contextlib import nullcontext
from typing import Dict, Optional, TYPE_CHECKING
from spinlock.features.base import FeatureExtractorBase
from spinlock.features.registry import FeatureRegistry
from spinlock.features.temporal.spatial import SpatialFeatureExtractor
from spinlock.features.temporal.spectral import SpectralFeatureExtractor
from spinlock.features.temporal.temporal import TemporalFeatureExtractor
from spinlock.features.temporal.cross_channel import CrossChannelFeatureExtractor

if TYPE_CHECKING:
    from spinlock.features.temporal.config import TemporalFeatureConfig


class TemporalFeatureOrchestrator(FeatureExtractorBase):
    """
    Temporal Feature Orchestrator for per-timestep feature extraction.

    Orchestrates extraction of all per-timestep features for online perturbation-based NOA:
    - Spatial statistics (24D): Statistics, gradients, Laplacian, percentiles
    - Spectral features (27D): FFT power, frequencies, bandwidth
    - Cross-channel features (12D): Correlation, mutual information, eigenvalues
    - Enhanced temporal dynamics (130D): Instantaneous, local temporal, stability,
      phase space, multi-scale features

    Total output: 193D per-timestep features

    Attributes:
        device: Computation device (cuda or cpu)
        config: Temporal feature configuration
        spatial_extractor: Spatial statistics extractor
        spectral_extractor: Spectral/frequency extractor
        cross_channel_extractor: Cross-channel interaction extractor
        temporal_extractor: Enhanced temporal dynamics extractor (new 130D implementation)
        registry: Feature name-to-index registry
    """

    def __init__(
        self,
        device: torch.device = torch.device('cuda'),
        config: Optional['TemporalFeatureConfig'] = None,
        profiling_context: Optional['FeatureProfilingContext'] = None
    ):
        """
        Initialize temporal feature orchestrator.

        Args:
            device: Computation device
            config: Optional TemporalFeatureConfig instance
            profiling_context: Optional profiling context for performance measurement
        """
        self.device = device
        self.config = config
        self.profiling_context = profiling_context

        # Initialize per-timestep component extractors
        self.spatial_extractor = SpatialFeatureExtractor(device=device)
        self.spectral_extractor = SpectralFeatureExtractor(device=device)

        # Cross-channel extractor (optional, based on config)
        self.cross_channel_extractor: Optional[CrossChannelFeatureExtractor] = None
        if config is not None and config.cross_channel is not None and config.cross_channel.enabled:
            self.cross_channel_extractor = CrossChannelFeatureExtractor(device=device)

        # Enhanced temporal extractor (new 130D per-timestep implementation)
        # Extract window parameters from config
        window_size = 5
        short_window = 5
        medium_window = 20
        long_window = 50
        if config is not None and config.temporal is not None:
            window_size = getattr(config.temporal, 'window_size', 5)
            short_window = getattr(config.temporal, 'short_window', 5)
            medium_window = getattr(config.temporal, 'medium_window', 20)
            long_window = getattr(config.temporal, 'long_window', 50)

        self.temporal_extractor = TemporalFeatureExtractor(
            device=device,
            window_size=window_size,
            short_window=short_window,
            medium_window=medium_window,
            long_window=long_window
        )

        # Initialize feature registry
        self._registry: Optional[FeatureRegistry] = None
        self._build_registry()

    @property
    def family_name(self) -> str:
        """Feature family name."""
        return "temporal"

    @property
    def version(self) -> str:
        """Feature family version."""
        return "3.0.0"  # New per-timestep-only architecture

    def _build_registry(self) -> None:
        """
        Build feature registry based on configuration.

        Registers all per-timestep features that will be extracted.
        """
        registry = FeatureRegistry(family_name="temporal")

        # Spatial features (per-timestep features)
        if self.config is None or self.config.spatial.enabled:
            spatial_base_features = [
                'spatial_mean', 'spatial_variance', 'spatial_std',
                'spatial_skewness', 'spatial_kurtosis',
                'spatial_min', 'spatial_max', 'spatial_range',
                'spatial_iqr', 'spatial_mad',
                'gradient_magnitude_mean', 'gradient_magnitude_std', 'gradient_magnitude_max',
                'gradient_x_mean', 'gradient_y_mean', 'gradient_anisotropy',
                'laplacian_mean', 'laplacian_std', 'laplacian_energy'
            ]

            for feat in spatial_base_features:
                registry.register(
                    feat,
                    category="spatial",
                    description=f"{feat} per-timestep spatial feature"
                )

            # Phase 1 extension: Percentiles
            if self.config is None or self.config.spatial.include_percentiles:
                for percentile in [5, 25, 50, 75, 95]:
                    registry.register(
                        f'percentile_{percentile}',
                        category="spatial",
                        description=f"{percentile}th percentile per-timestep"
                    )

            # Phase 2 extension: Histogram/occupancy
            if self.config is not None and self.config.spatial.include_histogram:
                for feat in ['histogram_entropy', 'histogram_peak_fraction', 'histogram_effective_bins']:
                    registry.register(
                        feat,
                        category="spatial",
                        description=f"{feat} (state space coverage)"
                    )

        # Spectral features (per-timestep features, no temporal aggregation)
        if self.config is None or self.config.spectral.enabled:
            # FFT power spectrum (multiscale)
            num_fft_scales = self.config.spectral.num_fft_scales if self.config else 5
            for scale_idx in range(num_fft_scales):
                for stat in ['mean', 'max', 'std']:
                    feat_name = f"fft_power_scale_{scale_idx}_{stat}"
                    registry.register(
                        feat_name,
                        category="spectral",
                        description=f"FFT power scale {scale_idx} {stat} per-timestep",
                        multiscale_index=scale_idx
                    )

            # Other spectral features
            spectral_base_features = [
                'dominant_freq_x', 'dominant_freq_y', 'dominant_freq_magnitude',
                'spectral_centroid_x', 'spectral_centroid_y', 'spectral_bandwidth',
                'low_freq_ratio', 'mid_freq_ratio', 'high_freq_ratio',
                'spectral_flatness', 'spectral_rolloff', 'spectral_anisotropy'
            ]

            for feat in spectral_base_features:
                registry.register(
                    feat,
                    category="spectral",
                    description=f"{feat} per-timestep spectral feature"
                )

        # Cross-channel (per-timestep)
        if self.cross_channel_extractor is not None:
            config_cc = self.config.cross_channel if self.config else None

            # Eigenvalue features
            if config_cc is None or config_cc.include_eigen_values:
                num_eigen_top = config_cc.num_eigen_top if config_cc else 3
                for i in range(1, num_eigen_top + 1):
                    registry.register(
                        f"cross_channel_eigen_top_{i}",
                        category="cross_channel",
                        description=f"Top {i} eigenvalue of correlation matrix"
                    )

            if config_cc is None or config_cc.include_eigen_trace:
                registry.register("cross_channel_eigen_trace", category="cross_channel",
                                 description="Trace of correlation eigenvalues")

            if config_cc is None or config_cc.include_condition_number:
                registry.register("cross_channel_condition_number", category="cross_channel",
                                 description="Condition number of correlation matrix")

            if config_cc is None or config_cc.include_participation_ratio:
                registry.register("cross_channel_participation_ratio", category="cross_channel",
                                 description="Participation ratio (effective dimensionality)")

            # Correlation statistics
            for stat in ['mean', 'max', 'min', 'std']:
                if config_cc is None or getattr(config_cc, f'include_corr_{stat}', True):
                    registry.register(f"cross_channel_corr_{stat}", category="cross_channel",
                                     description=f"Pairwise correlation {stat}")

            # Optional coherence
            if config_cc and config_cc.include_coherence:
                for band in config_cc.coherence_freq_bands:
                    registry.register(f"cross_spectral_coherence_{band}", category="cross_channel",
                                     description=f"Cross-spectral coherence in {band} band")
                registry.register("cross_spectral_coherence_max", category="cross_channel",
                                 description="Maximum cross-spectral coherence")
                registry.register("cross_spectral_coherence_peak_freq", category="cross_channel",
                                 description="Frequency of peak coherence")

            # Optional mutual information
            if config_cc and config_cc.include_mutual_info:
                for stat in ['mean', 'max']:
                    registry.register(f"cross_channel_mi_{stat}", category="cross_channel",
                                     description=f"Mutual information {stat}")

        # Enhanced temporal features (per-timestep, 130D total)
        if self.config is None or (self.config.temporal is not None and self.config.temporal.enabled):
            # Instantaneous dynamics (22D)
            instantaneous_features = [
                'energy_kinetic', 'energy_enstrophy', 'energy_potential', 'energy_total',
                'dissipation_viscous', 'dissipation_reaction', 'dissipation_total',
                'inst_spectral_centroid', 'inst_spectral_bandwidth', 'inst_spectral_rolloff',
                'inst_spectral_high_freq_ratio', 'inst_spectral_entropy',
                'structure_gradient_norm_mean', 'structure_gradient_norm_std',
                'structure_laplacian_mean', 'structure_laplacian_std', 'structure_divergence',
                'stats_skewness', 'stats_kurtosis', 'stats_entropy', 'stats_gini', 'stats_iqr'
            ]
            for feat in instantaneous_features:
                registry.register(feat, category="temporal", description=f"Instantaneous: {feat}")

            # Local temporal (28D)
            local_temporal_features = [
                'autocorr_lag1', 'autocorr_lag2', 'autocorr_lag5',
                'trend_slope', 'trend_r2', 'trend_accel', 'trend_jerk',
                'windowed_mean', 'windowed_std', 'windowed_min', 'windowed_max',
                'windowed_range', 'windowed_cv',
                'change_delta_mean', 'change_delta_std', 'change_L1', 'change_L2', 'change_Linf',
                'oscillation_zero_crossings', 'oscillation_peaks', 'oscillation_troughs',
                'oscillation_amplitude',
                'growth_exponential', 'growth_power_law', 'growth_relative',
                'growth_doubling_time', 'growth_energy', 'growth_enstrophy'
            ]
            for feat in local_temporal_features:
                registry.register(feat, category="temporal", description=f"Local temporal: {feat}")

            # Local stability (24D)
            local_stability_features = [
                'lipschitz_spatial', 'lipschitz_x', 'lipschitz_y', 'lipschitz_anisotropy',
                'jacobian_free_grad_alignment', 'jacobian_free_flow_consistency',
                'jacobian_free_contraction', 'jacobian_free_expansion', 'jacobian_free_stability',
                'linearization_gain', 'linearization_sensitivity', 'linearization_condition',
                'linearization_spectral_radius', 'linearization_damping',
                'divergence_trajectory', 'divergence_prediction', 'divergence_curvature',
                'divergence_phase_distance', 'divergence_lyapunov_proxy',
                'regularity_smoothness', 'regularity_holder', 'regularity_tv',
                'regularity_roughness', 'regularity_irregularity'
            ]
            for feat in local_stability_features:
                registry.register(feat, category="temporal", description=f"Local stability: {feat}")

            # Phase space geometry (26D)
            phase_space_features = [
                'flow_velocity_magnitude', 'flow_velocity_direction', 'flow_coherence',
                'flow_curl', 'flow_divergence', 'flow_alignment',
                'vorticity_mean', 'vorticity_std', 'vorticity_enstrophy_density',
                'vorticity_count', 'vorticity_helicity',
                'strain_shear', 'strain_normal_xx', 'strain_normal_yy', 'strain_magnitude',
                'topology_critical_density', 'topology_saddles', 'topology_attractors',
                'topology_repellers', 'topology_entropy',
                'manifold_dimension', 'manifold_tangent_angle', 'manifold_curvature',
                'manifold_geodesic_deviation', 'manifold_metric_det', 'manifold_volume'
            ]
            for feat in phase_space_features:
                registry.register(feat, category="temporal", description=f"Phase space: {feat}")

            # Multi-scale temporal (30D)
            multiscale_features = [
                'multiscale_energy_instant', 'multiscale_energy_short', 'multiscale_energy_medium',
                'multiscale_energy_long', 'multiscale_energy_trend',
                'multiscale_gradient_instant', 'multiscale_gradient_short', 'multiscale_gradient_medium',
                'multiscale_gradient_long', 'multiscale_gradient_trend',
                'multiscale_spectral_instant', 'multiscale_spectral_short', 'multiscale_spectral_medium',
                'multiscale_spectral_long', 'multiscale_spectral_drift',
                'multiscale_entropy_instant', 'multiscale_entropy_short', 'multiscale_entropy_medium',
                'multiscale_entropy_long', 'multiscale_entropy_production',
                'cross_scale_instant_short', 'cross_scale_short_medium', 'cross_scale_medium_long',
                'cross_scale_coherence', 'cross_scale_separation',
                'persistence_hurst', 'persistence_dfa', 'persistence_memory_depth',
                'persistence_score', 'persistence_anti'
            ]
            for feat in multiscale_features:
                registry.register(feat, category="temporal", description=f"Multi-scale: {feat}")

        self._registry = registry

    def get_feature_registry(self) -> FeatureRegistry:
        """Get feature registry."""
        if self._registry is None:
            self._build_registry()
        assert self._registry is not None, "Failed to build feature registry"
        return self._registry

    def extract_per_timestep(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Extract per-timestep features.

        Combines spatial, spectral, cross-channel, and enhanced temporal features
        to produce 193D per-timestep features for online perturbation-based NOA.

        Args:
            trajectories: Stochastic trajectories [N, M, T, C, H, W]
            metadata: Optional metadata dict

        Returns:
            Per-timestep features [N, T, 193]
            (averaged across realizations and channels)
        """
        N, M, T, C, H, W = trajectories.shape
        prof = self.profiling_context

        # Extract spatial features (24D)
        with prof.time_category('spatial', 0, N) if prof else nullcontext():
            spatial_features = self.spatial_extractor.extract(
                trajectories,
                config=self.config.spatial if self.config else None
            )

        # Extract spectral features (27D)
        with prof.time_category('spectral', 0, N) if prof else nullcontext():
            spectral_features = self.spectral_extractor.extract(
                trajectories,
                config=self.config.spectral if self.config else None,
                num_scales=self.config.spectral.num_fft_scales if self.config else 5
            )

        # Extract cross-channel features (12D)
        cross_channel_features = {}
        if self.cross_channel_extractor is not None:
            with prof.time_category('cross_channel', 0, N) if prof else nullcontext():
                # Reshape for cross-channel extraction: [N*M*T, C, H, W]
                fields_flat = trajectories.reshape(N * M * T, C, H, W)
                cross_channel_features = self.cross_channel_extractor.extract(
                    fields_flat,
                    config=self.config.cross_channel if self.config else None
                )
            # Reshape features back to [N, M, T] structure
            for name in cross_channel_features:
                feat = cross_channel_features[name]
                if feat.ndim == 1:  # [N*M*T]
                    cross_channel_features[name] = feat.reshape(N, M, T)
                else:
                    raise ValueError(f"Unexpected cross-channel feature shape: {feat.shape}")

        # Extract enhanced temporal features (130D)
        # Reshape to [N*M*T, C, H, W] for per-state temporal extraction
        with prof.time_category('temporal', 0, N) if prof else nullcontext():
            fields_flat = trajectories.reshape(N * M * T, C, H, W)
            temporal_features_flat = self.temporal_extractor.extract(fields_flat)  # [N*M*T, 130]
            # Reshape to [N, M, T, 130]
            temporal_features_reshaped = temporal_features_flat.reshape(N, M, T, 130)

        # Combine all per-timestep features
        all_features = {
            **spatial_features,
            **spectral_features,
            **cross_channel_features,
        }

        # Get feature names in registry order
        registry = self.get_feature_registry()
        spatial_names = [f.name for f in registry.get_features_by_category('spatial')]
        spectral_names = [f.name for f in registry.get_features_by_category('spectral')]
        cross_channel_names = [f.name for f in registry.get_features_by_category('cross_channel')]
        temporal_names = [f.name for f in registry.get_features_by_category('temporal')]

        # Feature ordering: [spatial | spectral | cross_channel | temporal]
        feature_list = []

        # Add spatial features
        for name in spatial_names:
            if name in all_features:
                feat = all_features[name]  # [N, M, T, C] or [N, T, C]
                if feat.ndim == 4:
                    feat_avg = feat.mean(dim=(1, 3))  # [N, T]
                else:
                    feat_avg = feat.mean(dim=2)  # [N, T]
                feature_list.append(feat_avg.unsqueeze(-1))

        # Add spectral features
        for name in spectral_names:
            if name in all_features:
                feat = all_features[name]  # [N, M, T, C] or [N, T, C]
                if feat.ndim == 4:
                    feat_avg = feat.mean(dim=(1, 3))  # [N, T]
                else:
                    feat_avg = feat.mean(dim=2)  # [N, T]
                feature_list.append(feat_avg.unsqueeze(-1))

        # Add cross-channel features
        if self.cross_channel_extractor is not None:
            for name in cross_channel_names:
                if name in all_features:
                    feat = all_features[name]  # [N, M, T]
                    feat_avg = feat.mean(dim=1)  # [N, T]
                    feature_list.append(feat_avg.unsqueeze(-1))

        # Add enhanced temporal features (already computed, just need to average across M)
        temporal_features_avg = temporal_features_reshaped.mean(dim=1)  # [N, T, 130]

        # Concatenate all features: [spatial | spectral | cross_channel | temporal]
        if len(feature_list) > 0:
            traditional_features = torch.cat(feature_list, dim=-1)  # [N, T, 24+27+12]
            per_timestep_features = torch.cat([traditional_features, temporal_features_avg], dim=-1)  # [N, T, 193]
        else:
            # Only temporal features (if spatial/spectral disabled)
            per_timestep_features = temporal_features_avg  # [N, T, 130]

        return per_timestep_features
