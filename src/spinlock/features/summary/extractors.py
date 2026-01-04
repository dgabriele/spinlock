"""
Summary Feature Extraction - main orchestrator.

Coordinates extraction of spatial, spectral, and temporal features,
implementing the three-stage extraction pipeline:
1. Per-timestep extraction (spatial, spectral)
2. Per-trajectory extraction (temporal dynamics)
3. Aggregation across realizations

Example:
    >>> from spinlock.features.summary.extractors import SummaryExtractor
    >>> from spinlock.features.summary.config import SummaryConfig
    >>>
    >>> extractor = SummaryExtractor(device='cuda', config=SummaryConfig())
    >>> trajectories = torch.randn(32, 10, 100, 3, 128, 128, device='cuda')
    >>> features = extractor.extract_all(trajectories)
"""

import torch
from contextlib import nullcontext
from typing import Dict, List, Optional, TYPE_CHECKING
from spinlock.features.base import FeatureExtractorBase
from spinlock.features.registry import FeatureRegistry
from spinlock.features.summary.spatial import SpatialFeatureExtractor
from spinlock.features.summary.spectral import SpectralFeatureExtractor
from spinlock.features.summary.temporal import TemporalFeatureExtractor
from spinlock.features.summary.cross_channel import CrossChannelFeatureExtractor
from spinlock.features.summary.causality import CausalityFeatureExtractor
from spinlock.features.summary.invariant_drift import InvariantDriftExtractor
from spinlock.features.summary.operator_sensitivity import OperatorSensitivityExtractor
from spinlock.features.summary.nonlinear import NonlinearFeatureExtractor
# Phase 2 extractors (v2.1)
from spinlock.features.summary.distributional import DistributionalFeatureExtractor
from spinlock.features.summary.structural import StructuralFeatureExtractor
from spinlock.features.summary.physics import PhysicsFeatureExtractor
from spinlock.features.summary.morphological import MorphologicalFeatureExtractor
from spinlock.features.summary.multiscale import MultiscaleFeatureExtractor
# Learned features (U-AFNO latent extraction)
from spinlock.features.summary.learned import LearnedSummaryExtractor

if TYPE_CHECKING:
    from spinlock.features.summary.config import SummaryConfig, LearnedSummaryConfig


class SummaryExtractor(FeatureExtractorBase):
    """
    Summary Descriptor Features extractor (v2.0).

    Orchestrates spatial, spectral, temporal, and v2.0 feature extraction,
    implementing the three-stage pipeline defined in FeatureExtractorBase.

    v1.0 categories: spatial, spectral, temporal
    v2.0 categories: operator_sensitivity, cross_channel, causality, invariant_drift

    Attributes:
        device: Computation device (cuda or cpu)
        config: Summary feature configuration
        spatial_extractor: Spatial statistics extractor
        spectral_extractor: Spectral/frequency extractor
        temporal_extractor: Temporal dynamics extractor
        operator_sensitivity_extractor: Operator sensitivity extractor (v2.0, optional)
        cross_channel_extractor: Cross-channel interaction extractor (v2.0, optional)
        causality_extractor: Causality/directionality extractor (v2.0, optional)
        invariant_drift_extractor: Invariant drift extractor (v2.0, optional)
        learned_extractor: Learned feature extractor from U-AFNO latents (Phase 2, optional)
        registry: Feature name-to-index registry
    """

    def __init__(
        self,
        device: torch.device = torch.device('cuda'),
        config: Optional['SummaryConfig'] = None,
        profiling_context: Optional['FeatureProfilingContext'] = None
    ):
        """
        Initialize SUMMARY extractor.

        Args:
            device: Computation device
            config: Optional SummaryConfig instance
            profiling_context: Optional profiling context for performance measurement
        """
        self.device = device
        self.config = config
        self.profiling_context = profiling_context

        # Initialize v1.0 component extractors (always enabled)
        self.spatial_extractor = SpatialFeatureExtractor(device=device)
        self.spectral_extractor = SpectralFeatureExtractor(device=device)
        self.temporal_extractor = TemporalFeatureExtractor(device=device)

        # Initialize v2.0 component extractors (optional, based on config)
        self.operator_sensitivity_extractor: Optional[OperatorSensitivityExtractor] = None
        self.cross_channel_extractor: Optional[CrossChannelFeatureExtractor] = None
        self.causality_extractor: Optional[CausalityFeatureExtractor] = None
        self.invariant_drift_extractor: Optional[InvariantDriftExtractor] = None
        self.nonlinear_extractor: Optional[NonlinearFeatureExtractor] = None  # Phase 1 extension

        # Initialize v2.1 component extractors (Phase 2)
        self.distributional_extractor: Optional[DistributionalFeatureExtractor] = None
        self.structural_extractor: Optional[StructuralFeatureExtractor] = None
        self.physics_extractor: Optional[PhysicsFeatureExtractor] = None
        self.morphological_extractor: Optional[MorphologicalFeatureExtractor] = None
        self.multiscale_extractor: Optional[MultiscaleFeatureExtractor] = None

        # Initialize learned feature extractor (U-AFNO latent extraction)
        self.learned_extractor: Optional[LearnedSummaryExtractor] = None

        if config is not None:
            if config.operator_sensitivity is not None and config.operator_sensitivity.enabled:
                self.operator_sensitivity_extractor = OperatorSensitivityExtractor(
                    device=device,
                    lipschitz_epsilon_scales=config.operator_sensitivity.lipschitz_epsilon_scales,
                    gain_scale_factors=config.operator_sensitivity.gain_scale_factors
                )

            if config.cross_channel is not None and config.cross_channel.enabled:
                self.cross_channel_extractor = CrossChannelFeatureExtractor(device=device)

            if config.causality is not None and config.causality.enabled:
                self.causality_extractor = CausalityFeatureExtractor(device=device)

            if config.invariant_drift is not None and config.invariant_drift.enabled:
                self.invariant_drift_extractor = InvariantDriftExtractor(device=device)

            if config.nonlinear is not None and config.nonlinear.enabled:
                self.nonlinear_extractor = NonlinearFeatureExtractor(device=device)

            # Initialize v2.1 extractors
            if config.distributional is not None and config.distributional.enabled:
                self.distributional_extractor = DistributionalFeatureExtractor(device=device)

            if config.structural is not None and config.structural.enabled:
                self.structural_extractor = StructuralFeatureExtractor(device=device)

            if config.physics is not None and config.physics.enabled:
                self.physics_extractor = PhysicsFeatureExtractor(device=device)

            if config.morphological is not None and config.morphological.enabled:
                self.morphological_extractor = MorphologicalFeatureExtractor(device=device)

            if config.multiscale is not None and config.multiscale.enabled:
                self.multiscale_extractor = MultiscaleFeatureExtractor(device=device)

            # Initialize learned feature extractor (U-AFNO latent extraction)
            if config.learned is not None and config.learned.enabled:
                self.learned_extractor = LearnedSummaryExtractor(
                    device=device,
                    config=config.learned
                )

        # Initialize feature registry
        self._registry: Optional[FeatureRegistry] = None
        self._build_registry()

    @property
    def family_name(self) -> str:
        """Feature family name."""
        return "summary"

    @property
    def version(self) -> str:
        """Feature family version."""
        return "2.1.0"

    def _build_registry(self) -> None:
        """
        Build feature registry based on configuration.

        Registers all features that will be extracted, accounting for
        multiscale parameters and aggregation methods.
        """
        registry = FeatureRegistry(family_name="summary")

        # Get aggregation methods from config
        if self.config is not None:
            temporal_aggs = self.config.temporal_aggregation
            realization_aggs = self.config.realization_aggregation
        else:
            temporal_aggs = ['mean', 'std']
            realization_aggs = ['mean', 'std', 'cv']

        # Spatial features (per-timestep features, no temporal aggregation)
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

        # Temporal features (trajectory-level, per-realization)
        if self.config is None or self.config.temporal.enabled:
            temporal_base_features = [
                'energy_growth_rate', 'energy_growth_accel', 'variance_growth_rate',
                'temporal_freq_dominant', 'oscillation_amplitude', 'oscillation_period',
                'autocorr_decay_time', 'lyapunov_approx', 'trajectory_smoothness',
                'regime_switches', 'final_to_initial_ratio',
                'trend_strength', 'detrended_variance'
            ]

            for feat in temporal_base_features:
                registry.register(
                    feat,
                    category="temporal",
                    description=f"{feat} per-trajectory temporal feature"
                )

            # Phase 1 extension: Event counts
            if self.config is None or self.config.temporal.include_event_counts:
                for feat in ['num_spikes', 'num_bursts', 'num_zero_crossings']:
                    registry.register(
                        feat,
                        category="temporal",
                        description=f"{feat} event count feature"
                    )

            # Phase 1 extension: Time-to-event
            if self.config is None or self.config.temporal.include_time_to_event:
                for thresh in [0.5, 2.0]:
                    registry.register(
                        f'time_to_{thresh}x',
                        category="temporal",
                        description=f"Time to {thresh}× initial value crossing"
                    )

            # Phase 1 extension: Rolling window statistics
            if self.config is None or self.config.temporal.include_rolling_windows:
                window_fractions = self.config.temporal.rolling_window_fractions if self.config else [0.05, 0.10, 0.20]
                for frac in window_fractions:
                    window_pct = int(frac * 100)
                    for stat in ['mean', 'std', 'max', 'min', 'mean_variability', 'std_variability']:
                        registry.register(
                            f'rolling_w{window_pct}_{stat}',
                            category="temporal",
                            description=f"Rolling window {window_pct}% {stat}"
                        )

            # Phase 2 extension: PACF (Partial Autocorrelation Function)
            if self.config is not None and self.config.temporal.include_pacf:
                max_lag_pacf = self.config.temporal.pacf_max_lag if self.config else 10
                for lag in range(1, max_lag_pacf + 1):
                    registry.register(
                        f'pacf_lag_{lag}',
                        category="temporal",
                        description=f"PACF at lag {lag}"
                    )

        # ========== v2.0 Features ==========

        # Operator sensitivity (trajectory-level, extracted during rollout)
        if self.operator_sensitivity_extractor is not None:
            config_ops = self.config.operator_sensitivity if self.config else None

            # Lipschitz estimates
            if config_ops is None or config_ops.include_lipschitz:
                eps_scales = config_ops.lipschitz_epsilon_scales if config_ops else [1e-4, 1e-3, 1e-2]
                for eps in eps_scales:
                    registry.register(
                        f"lipschitz_eps_{eps:.0e}",
                        category="operator_sensitivity",
                        description=f"Lipschitz constant estimate at epsilon {eps:.0e}"
                    )

            # Gain curves
            if config_ops is None or config_ops.include_gain_curve:
                gain_scales = config_ops.gain_scale_factors if config_ops else [0.5, 0.75, 1.25, 1.5]
                for scale in gain_scales:
                    registry.register(
                        f"gain_scale_{scale:.2f}",
                        category="operator_sensitivity",
                        description=f"Output gain at input scale {scale:.2f}"
                    )

            # Linearity metrics
            if config_ops is None or config_ops.include_linearity_metrics:
                for feat in ['linearity_r2', 'saturation_degree', 'compression_ratio']:
                    registry.register(
                        feat,
                        category="operator_sensitivity",
                        description=f"Operator {feat}"
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

        # Causality (trajectory-level)
        if self.causality_extractor is not None:
            config_caus = self.config.causality if self.config else None

            # Lagged correlation asymmetry
            max_lag_corr = config_caus.max_lag_correlation if config_caus else 3
            for lag in range(1, max_lag_corr + 1):
                for stat in ['mean', 'max']:
                    registry.register(
                        f"causality_lag_corr_asymmetry_{stat}_lag{lag}",
                        category="causality",
                        description=f"Lagged correlation asymmetry {stat} at lag {lag}"
                    )

            # Prediction error ratio
            max_lag_pred = config_caus.max_lag_prediction if config_caus else 2
            for lag in range(1, max_lag_pred + 1):
                for metric in ['ratio', 'diff']:
                    registry.register(
                        f"causality_pred_error_{metric}_lag{lag}",
                        category="causality",
                        description=f"Prediction error {metric} at lag {lag}"
                    )

            # Time irreversibility
            if config_caus is None or config_caus.include_time_irreversibility:
                registry.register("causality_time_irreversibility", category="causality",
                                 description="Third-order time irreversibility")
                registry.register("causality_time_asymmetry_index", category="causality",
                                 description="Normalized time asymmetry index")

            # Spatial flow
            if config_caus is None or config_caus.include_spatial_flow:
                registry.register("causality_spatial_flow_magnitude", category="causality",
                                 description="Spatial information flow magnitude")
                registry.register("causality_spatial_flow_anisotropy", category="causality",
                                 description="Spatial flow directional anisotropy")

            # Optional transfer entropy
            if config_caus and config_caus.include_transfer_entropy:
                for stat in ['mean', 'max', 'asymmetry']:
                    registry.register(f"causality_transfer_entropy_{stat}", category="causality",
                                     description=f"Transfer entropy {stat}")

            # Optional Granger causality
            if config_caus and config_caus.include_granger_causality:
                for stat in ['mean', 'max', 'asymmetry']:
                    registry.register(f"causality_granger_score_{stat}", category="causality",
                                     description=f"Granger causality score {stat}")

        # Invariant drift (trajectory-level)
        if self.invariant_drift_extractor is not None:
            config_drift = self.config.invariant_drift if self.config else None

            # Determine scales
            num_scales = config_drift.num_scales if config_drift else 3
            scales = ['raw']
            if num_scales >= 2:
                scales.append('lowpass')
            if num_scales >= 3:
                scales.append('highpass')

            # Generic norms
            norms = []
            if config_drift is None or config_drift.include_L1_drift:
                norms.append('L1')
            if config_drift is None or config_drift.include_L2_drift:
                norms.append('L2')
            if config_drift is None or config_drift.include_Linf_drift:
                norms.append('Linf')
            if config_drift is None or config_drift.include_entropy_drift:
                norms.append('entropy')
            if config_drift is None or config_drift.include_tv_drift:
                norms.append('tv')

            metrics = ['mean_drift', 'drift_variance', 'final_initial_ratio', 'monotonicity']

            for norm in norms:
                for metric in metrics:
                    for scale in scales:
                        registry.register(
                            f"{norm}_{metric}_{scale}",
                            category="invariant_drift",
                            description=f"{norm} norm {metric} ({scale}-filtered)"
                        )

            # Optional mass drift
            if config_drift and config_drift.include_mass_drift:
                for metric in metrics:
                    for scale in scales:
                        registry.register(
                            f"mass_{metric}_{scale}",
                            category="invariant_drift",
                            description=f"Mass {metric} ({scale}-filtered)"
                        )

            # Optional energy drift
            if config_drift and config_drift.include_energy_drift:
                for energy_type in ['L2_energy', 'gradient_energy']:
                    for metric in metrics:
                        for scale in scales:
                            registry.register(
                                f"{energy_type}_{metric}_{scale}",
                                category="invariant_drift",
                                description=f"{energy_type} {metric} ({scale}-filtered)"
                            )

        # Nonlinear dynamics (Phase 1 extension, trajectory-level, expensive, opt-in)
        if self.nonlinear_extractor is not None:
            config_nonlinear = self.config.nonlinear if self.config else None

            # Recurrence Quantification Analysis (RQA)
            if config_nonlinear is None or config_nonlinear.include_recurrence:
                for feat in ['recurrence_rate', 'determinism', 'laminarity', 'entropy_diag_length']:
                    registry.register(
                        feat,
                        category="nonlinear",
                        description=f"RQA {feat}"
                    )

            # Correlation dimension
            if config_nonlinear is None or config_nonlinear.include_correlation_dim:
                registry.register(
                    'correlation_dimension',
                    category="nonlinear",
                    description="Correlation dimension (attractor complexity)"
                )

            # Phase 2 extension: Permutation entropy
            if config_nonlinear is not None and config_nonlinear.include_permutation_entropy:
                registry.register(
                    'permutation_entropy',
                    category="nonlinear",
                    description="Permutation entropy (ordinal pattern complexity)"
                )

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

        Combines spatial and spectral features at each timestep.

        Args:
            trajectories: Stochastic trajectories [N, M, T, C, H, W]
            metadata: Optional metadata dict

        Returns:
            Per-timestep features [N, T, D]
            (averaged across realizations and channels for simplicity)
        """
        N, M, T, C, H, W = trajectories.shape
        prof = self.profiling_context

        # Extract v1.0 per-timestep features
        with prof.time_category('spatial', 0, N) if prof else nullcontext():
            spatial_features = self.spatial_extractor.extract(
                trajectories,
                config=self.config.spatial if self.config else None
            )

        with prof.time_category('spectral', 0, N) if prof else nullcontext():
            spectral_features = self.spectral_extractor.extract(
                trajectories,
                config=self.config.spectral if self.config else None,
                num_scales=self.config.spectral.num_fft_scales if self.config else 5
            )

        # Extract v2.0 per-timestep features
        cross_channel_features = {}
        if self.cross_channel_extractor is not None:
            with prof.time_category('cross_channel', 0, N) if prof else nullcontext():
                # Reshape for cross-channel extraction: [N*M*T, C, H, W]
                fields_flat = trajectories.reshape(N * M * T, C, H, W)
                cross_channel_features = self.cross_channel_extractor.extract(
                    fields_flat,
                    config=self.config.cross_channel if self.config else None
                )
            # Reshape features back to [N, M, T, C] structure
            # (Each feature has shape [N*M*T] or [N*M*T, ...])
            for name in cross_channel_features:
                feat = cross_channel_features[name]
                if feat.ndim == 1:  # [N*M*T]
                    cross_channel_features[name] = feat.reshape(N, M, T)
                else:
                    raise ValueError(f"Unexpected cross-channel feature shape: {feat.shape}")

        # Extract v2.1 per-timestep features (Phase 2)
        distributional_features = {}
        structural_features = {}
        physics_features = {}
        morphological_features = {}
        multiscale_features = {}

        # Reshape for v2.1 extraction: [N*M, T, C, H, W]
        fields_v21 = trajectories.reshape(N * M, T, C, H, W)

        if self.distributional_extractor is not None:
            with prof.time_category('distributional', 0, N) if prof else nullcontext():
                distributional_features = self.distributional_extractor.extract(
                    fields_v21,
                    config=self.config.distributional if self.config else None
                )

        if self.structural_extractor is not None:
            with prof.time_category('structural', 0, N) if prof else nullcontext():
                structural_features = self.structural_extractor.extract(
                    fields_v21,
                    config=self.config.structural if self.config else None
                )

        if self.physics_extractor is not None:
            with prof.time_category('physics', 0, N) if prof else nullcontext():
                physics_features = self.physics_extractor.extract(
                    fields_v21,
                    config=self.config.physics if self.config else None
                )

        if self.morphological_extractor is not None:
            with prof.time_category('morphological', 0, N) if prof else nullcontext():
                morphological_features = self.morphological_extractor.extract(
                    fields_v21,
                    config=self.config.morphological if self.config else None
                )

        if self.multiscale_extractor is not None:
            with prof.time_category('multiscale', 0, N) if prof else nullcontext():
                multiscale_features = self.multiscale_extractor.extract(
                    fields_v21,
                    config=self.config.multiscale if self.config else None
                )

        # Combine all per-timestep features
        all_features = {
            **spatial_features,
            **spectral_features,
            **cross_channel_features,
            **distributional_features,
            **structural_features,
            **physics_features,
            **morphological_features,
            **multiscale_features
        }

        # Stack features into single tensor following registry order
        # Each feature has shape [N, M, T, C] or [N, T, C]
        # We need to average across M and C dimensions

        # Get feature names in registry order (all per-timestep categories)
        registry = self.get_feature_registry()
        spatial_names = [f.name for f in registry.get_features_by_category('spatial')]
        spectral_names = [f.name for f in registry.get_features_by_category('spectral')]
        cross_channel_names = [f.name for f in registry.get_features_by_category('cross_channel')]
        distributional_names = [f.name for f in registry.get_features_by_category('distributional')]
        structural_names = [f.name for f in registry.get_features_by_category('structural')]
        physics_names = [f.name for f in registry.get_features_by_category('physics')]
        morphological_names = [f.name for f in registry.get_features_by_category('morphological')]
        multiscale_names = [f.name for f in registry.get_features_by_category('multiscale')]

        feature_names_in_order = (
            spatial_names +
            spectral_names +
            cross_channel_names +
            distributional_names +
            structural_names +
            physics_names +
            morphological_names +
            multiscale_names
        )

        feature_list = []
        for name in feature_names_in_order:
            if name not in all_features:
                # Skip features not extracted (e.g., if disabled in config)
                continue

            feat = all_features[name]

            if feat.ndim == 4:  # [N, M, T, C]
                # Average across realizations and channels
                feat_avg = feat.mean(dim=(1, 3))  # [N, T]
            elif feat.ndim == 3:
                # Could be [N, T, C] (v1.0) or [N, M, T] (v2.0 cross-channel)
                if name in cross_channel_names:
                    # Cross-channel features: [N, M, T] → average across realizations
                    feat_avg = feat.mean(dim=1)  # [N, T]
                else:
                    # v1.0 features: [N, T, C] → average across channels
                    feat_avg = feat.mean(dim=2)  # [N, T]
            else:
                raise ValueError(f"Unexpected feature shape for {name}: {feat.shape}")

            feature_list.append(feat_avg.unsqueeze(-1))  # [N, T, 1]

        # Concatenate all features
        per_timestep_features = torch.cat(feature_list, dim=-1)  # [N, T, D]

        return per_timestep_features

    def extract_per_trajectory(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Extract per-trajectory features.

        Computes temporal dynamics features for each realization.

        Args:
            trajectories: Stochastic trajectories [N, M, T, C, H, W]
            metadata: Optional metadata dict (can include 'operator_sensitivity_features' key)

        Returns:
            Per-trajectory features [N, M, D_traj]
            (averaged across channels for simplicity)
        """
        N = trajectories.shape[0]
        prof = self.profiling_context

        # Extract v1.0 temporal features
        with prof.time_category('temporal', 0, N) if prof else nullcontext():
            temporal_features = self.temporal_extractor.extract(
                trajectories,
                config=self.config.temporal if self.config else None
            )

        # Extract v2.0 trajectory-level features
        causality_features = {}
        if self.causality_extractor is not None:
            with prof.time_category('causality', 0, N) if prof else nullcontext():
                causality_features = self.causality_extractor.extract(
                    trajectories,
                    config=self.config.causality if self.config else None
                )

        invariant_drift_features = {}
        if self.invariant_drift_extractor is not None:
            with prof.time_category('invariant_drift', 0, N) if prof else nullcontext():
                invariant_drift_features = self.invariant_drift_extractor.extract(
                    trajectories,
                    config=self.config.invariant_drift if self.config else None
                )

        nonlinear_features = {}
        if self.nonlinear_extractor is not None:
            with prof.time_category('nonlinear', 0, N) if prof else nullcontext():
                nonlinear_features = self.nonlinear_extractor.extract(
                    trajectories,
                    config=self.config.nonlinear if self.config else None
                )

        # Operator sensitivity features (from inline extraction during generation)
        operator_sensitivity_features = {}
        if metadata is not None and 'operator_sensitivity_features' in metadata:
            # Features provided from rollout engine
            # Shape: Dict[str, torch.Tensor] with [N] scalars
            operator_sensitivity_features = metadata['operator_sensitivity_features']

        # Combine all trajectory-level features
        all_trajectory_features = {
            **temporal_features,
            **causality_features,
            **invariant_drift_features,
            **nonlinear_features
        }

        # Stack features into single tensor following registry order
        # Each feature has shape [N, M, C]

        # Get feature names in registry order
        registry = self.get_feature_registry()
        temporal_names = [f.name for f in registry.get_features_by_category('temporal')]
        causality_names = [f.name for f in registry.get_features_by_category('causality')]
        invariant_drift_names = [f.name for f in registry.get_features_by_category('invariant_drift')]
        nonlinear_names = [f.name for f in registry.get_features_by_category('nonlinear')]
        operator_sensitivity_names = [f.name for f in registry.get_features_by_category('operator_sensitivity')]

        # Post-hoc extracted features (temporal, causality, invariant_drift, nonlinear)
        feature_names_in_order = temporal_names + causality_names + invariant_drift_names + nonlinear_names

        feature_list = []
        for name in feature_names_in_order:
            if name not in all_trajectory_features:
                # Skip features not extracted (e.g., if disabled in config)
                continue

            feat = all_trajectory_features[name]

            # Ensure float dtype for aggregation operations
            if not feat.is_floating_point():
                feat = feat.float()

            if feat.ndim == 3:  # [N, M, C]
                # Average across channels
                feat_avg = feat.mean(dim=2)  # [N, M]
            elif feat.ndim == 2:  # [N, M] (already channel-averaged, e.g., invariant drift)
                feat_avg = feat  # [N, M]
            else:
                raise ValueError(f"Unexpected feature shape for {name}: {feat.shape}")

            feature_list.append(feat_avg.unsqueeze(-1))  # [N, M, 1]

        # Append operator sensitivity features (inline-extracted, scalar per sample)
        # Only include if actually provided in metadata - don't fill with NaN placeholders
        # These need to be broadcast across realizations: [N] → [N, M, 1]
        N, M = trajectories.shape[0], trajectories.shape[1]
        if operator_sensitivity_features:
            for name in operator_sensitivity_names:
                if name in operator_sensitivity_features:
                    # Scalar feature [N] → broadcast to [N, M, 1]
                    feat_scalar = operator_sensitivity_features[name]  # [N]
                    feat_broadcast = feat_scalar.unsqueeze(1).unsqueeze(2).expand(N, M, 1)  # [N, M, 1]
                    feature_list.append(feat_broadcast)

        # Concatenate all features
        per_trajectory_features = torch.cat(feature_list, dim=-1)  # [N, M, D_traj]

        return per_trajectory_features

    def aggregate_realizations(
        self,
        per_trajectory_features: torch.Tensor,  # [N, M, D_traj]
        method: str = "mean"
    ) -> torch.Tensor:
        """
        Aggregate per-trajectory features across realizations.

        Args:
            per_trajectory_features: Features [N, M, D_traj]
            method: Aggregation method

        Returns:
            Aggregated features [N, D_final]
        """
        if method == "mean":
            return per_trajectory_features.mean(dim=1)  # [N, D_traj]
        elif method == "std":
            return per_trajectory_features.std(dim=1)
        elif method == "min":
            return per_trajectory_features.amin(dim=1)
        elif method == "max":
            return per_trajectory_features.amax(dim=1)
        elif method == "cv":
            # Coefficient of variation: std / mean
            mean = per_trajectory_features.mean(dim=1)
            std = per_trajectory_features.std(dim=1)
            return std / (mean.abs() + 1e-8)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def extract_learned_features(
        self,
        operators: 'List[torch.nn.Module]',
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
    ) -> torch.Tensor:
        """
        Extract learned features from U-AFNO latent representations.

        This method uses the U-AFNO intermediate features (bottleneck and/or skips)
        to compute aggregated learned features. Only available when:
        1. `learned_extractor` is initialized (config.learned.enabled=True)
        2. Operators are U-AFNO instances with `get_intermediate_features()` method

        Args:
            operators: List of N U-AFNO operators
            trajectories: Stochastic trajectories [N, M, T, C, H, W]

        Returns:
            Learned features [N, D_learned]

        Raises:
            ValueError: If learned_extractor not initialized or operators not provided
        """
        if self.learned_extractor is None:
            raise ValueError(
                "Learned feature extraction not enabled. "
                "Set config.learned.enabled=True to enable."
            )

        if operators is None or len(operators) == 0:
            raise ValueError(
                "Operators required for learned feature extraction. "
                "Ensure pipeline passes operators to feature extraction."
            )

        N = trajectories.shape[0]
        if len(operators) != N:
            raise ValueError(
                f"Number of operators ({len(operators)}) must match batch size ({N})"
            )

        # Extract learned features for each operator
        return self.learned_extractor.extract_batch(operators, trajectories)

    def extract_all(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        metadata: Optional[Dict] = None,
        operators: Optional[List[torch.nn.Module]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all SUMMARY features (convenience method).

        Runs full three-stage pipeline and returns all feature representations.
        Supports three summary_mode options:
        - "manual": Hand-crafted features only (spatial, spectral, temporal, v2.0+)
        - "learned": U-AFNO latent features only (requires operators)
        - "hybrid": Both manual and learned features concatenated

        Args:
            trajectories: Stochastic trajectories [N, M, T, C, H, W]
            metadata: Optional metadata dict
            operators: List of N U-AFNO operators (required for "learned" and "hybrid" modes)

        Returns:
            Dictionary with keys:
            - 'per_timestep': Features [N, T, D] (manual mode only)
            - 'per_trajectory': Features [N, M, D_traj] (manual mode only)
            - 'aggregated_mean': Features [N, D_final] (mean across realizations)
            - 'aggregated_std': Features [N, D_final] (std across realizations)
            - 'aggregated_cv': Features [N, D_final] (cv across realizations)
            - 'learned': Features [N, D_learned] (learned/hybrid modes only)
        """
        # Determine summary mode
        summary_mode = self.config.summary_mode if self.config else "manual"

        result: Dict[str, torch.Tensor] = {}

        # Extract manual features (for "manual" and "hybrid" modes)
        if summary_mode in ("manual", "hybrid"):
            # Stage 1: Per-timestep features (optional - skip if disabled)
            extract_per_timestep = self.config.extract_per_timestep if self.config else True
            if extract_per_timestep:
                result['per_timestep'] = self.extract_per_timestep(trajectories, metadata)

            # Stage 2: Per-trajectory features
            per_trajectory = self.extract_per_trajectory(trajectories, metadata)
            result['per_trajectory'] = per_trajectory

            # Stage 3: Aggregate across realizations
            agg_methods = self.config.realization_aggregation if self.config else ['mean', 'std', 'cv']
            for method in agg_methods:
                result[f'aggregated_{method}'] = self.aggregate_realizations(
                    per_trajectory,
                    method=method
                )

        # Extract learned features (for "learned" and "hybrid" modes)
        if summary_mode in ("learned", "hybrid"):
            if self.learned_extractor is None:
                raise ValueError(
                    f"summary_mode='{summary_mode}' requires learned features, "
                    "but config.learned.enabled=False. Enable learned extraction "
                    "or use summary_mode='manual'."
                )

            if operators is None or len(operators) == 0:
                raise ValueError(
                    f"summary_mode='{summary_mode}' requires operators, "
                    "but none were provided. Ensure pipeline passes operators "
                    "to feature extraction."
                )

            learned_features = self.extract_learned_features(operators, trajectories)
            result['learned'] = learned_features

        return result
