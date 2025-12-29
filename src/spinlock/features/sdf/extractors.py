"""
SDF (Summary Descriptor Features) main orchestrator.

Coordinates extraction of spatial, spectral, and temporal features,
implementing the three-stage extraction pipeline:
1. Per-timestep extraction (spatial, spectral)
2. Per-trajectory extraction (temporal dynamics)
3. Aggregation across realizations

Example:
    >>> from spinlock.features.sdf.extractors import SDFExtractor
    >>> from spinlock.features.sdf.config import SDFConfig
    >>>
    >>> extractor = SDFExtractor(device='cuda', config=SDFConfig())
    >>> trajectories = torch.randn(32, 10, 100, 3, 128, 128, device='cuda')
    >>> features = extractor.extract_all(trajectories)
"""

import torch
from typing import Dict, Optional, TYPE_CHECKING
from spinlock.features.base import FeatureExtractorBase
from spinlock.features.registry import FeatureRegistry
from spinlock.features.sdf.spatial import SpatialFeatureExtractor
from spinlock.features.sdf.spectral import SpectralFeatureExtractor
from spinlock.features.sdf.temporal import TemporalFeatureExtractor
from spinlock.features.sdf.cross_channel import CrossChannelFeatureExtractor
from spinlock.features.sdf.causality import CausalityFeatureExtractor
from spinlock.features.sdf.invariant_drift import InvariantDriftExtractor
from spinlock.features.sdf.operator_sensitivity import OperatorSensitivityExtractor

if TYPE_CHECKING:
    from spinlock.features.sdf.config import SDFConfig


class SDFExtractor(FeatureExtractorBase):
    """
    Summary Descriptor Features extractor (v2.0).

    Orchestrates spatial, spectral, temporal, and v2.0 feature extraction,
    implementing the three-stage pipeline defined in FeatureExtractorBase.

    v1.0 categories: spatial, spectral, temporal
    v2.0 categories: operator_sensitivity, cross_channel, causality, invariant_drift

    Attributes:
        device: Computation device (cuda or cpu)
        config: SDF configuration
        spatial_extractor: Spatial statistics extractor
        spectral_extractor: Spectral/frequency extractor
        temporal_extractor: Temporal dynamics extractor
        operator_sensitivity_extractor: Operator sensitivity extractor (v2.0, optional)
        cross_channel_extractor: Cross-channel interaction extractor (v2.0, optional)
        causality_extractor: Causality/directionality extractor (v2.0, optional)
        invariant_drift_extractor: Invariant drift extractor (v2.0, optional)
        registry: Feature name-to-index registry
    """

    def __init__(
        self,
        device: torch.device = torch.device('cuda'),
        config: Optional['SDFConfig'] = None
    ):
        """
        Initialize SDF extractor.

        Args:
            device: Computation device
            config: Optional SDFConfig instance
        """
        self.device = device
        self.config = config

        # Initialize v1.0 component extractors (always enabled)
        self.spatial_extractor = SpatialFeatureExtractor(device=device)
        self.spectral_extractor = SpectralFeatureExtractor(device=device)
        self.temporal_extractor = TemporalFeatureExtractor(device=device)

        # Initialize v2.0 component extractors (optional, based on config)
        self.operator_sensitivity_extractor: Optional[OperatorSensitivityExtractor] = None
        self.cross_channel_extractor: Optional[CrossChannelFeatureExtractor] = None
        self.causality_extractor: Optional[CausalityFeatureExtractor] = None
        self.invariant_drift_extractor: Optional[InvariantDriftExtractor] = None

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

        # Initialize feature registry
        self._registry: Optional[FeatureRegistry] = None
        self._build_registry()

    @property
    def family_name(self) -> str:
        """Feature family name."""
        return "sdf"

    @property
    def version(self) -> str:
        """Feature family version."""
        return "2.0.0"

    def _build_registry(self) -> None:
        """
        Build feature registry based on configuration.

        Registers all features that will be extracted, accounting for
        multiscale parameters and aggregation methods.
        """
        registry = FeatureRegistry(family_name="sdf")

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

        # Extract v1.0 per-timestep features
        spatial_features = self.spatial_extractor.extract(
            trajectories,
            config=self.config.spatial if self.config else None
        )

        spectral_features = self.spectral_extractor.extract(
            trajectories,
            config=self.config.spectral if self.config else None,
            num_scales=self.config.spectral.num_fft_scales if self.config else 5
        )

        # Extract v2.0 per-timestep features
        cross_channel_features = {}
        if self.cross_channel_extractor is not None:
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

        # Combine all per-timestep features
        all_features = {**spatial_features, **spectral_features, **cross_channel_features}

        # Stack features into single tensor following registry order
        # Each feature has shape [N, M, T, C] or [N, T, C]
        # We need to average across M and C dimensions

        # Get feature names in registry order (spatial + spectral + cross_channel categories)
        registry = self.get_feature_registry()
        spatial_names = [f.name for f in registry.get_features_by_category('spatial')]
        spectral_names = [f.name for f in registry.get_features_by_category('spectral')]
        cross_channel_names = [f.name for f in registry.get_features_by_category('cross_channel')]
        feature_names_in_order = spatial_names + spectral_names + cross_channel_names

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
        # Extract v1.0 temporal features
        temporal_features = self.temporal_extractor.extract(
            trajectories,
            config=self.config.temporal if self.config else None
        )

        # Extract v2.0 trajectory-level features
        causality_features = {}
        if self.causality_extractor is not None:
            causality_features = self.causality_extractor.extract(
                trajectories,
                config=self.config.causality if self.config else None
            )

        invariant_drift_features = {}
        if self.invariant_drift_extractor is not None:
            invariant_drift_features = self.invariant_drift_extractor.extract(
                trajectories,
                config=self.config.invariant_drift if self.config else None
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
            **invariant_drift_features
        }

        # Stack features into single tensor following registry order
        # Each feature has shape [N, M, C]

        # Get feature names in registry order
        registry = self.get_feature_registry()
        temporal_names = [f.name for f in registry.get_features_by_category('temporal')]
        causality_names = [f.name for f in registry.get_features_by_category('causality')]
        invariant_drift_names = [f.name for f in registry.get_features_by_category('invariant_drift')]
        operator_sensitivity_names = [f.name for f in registry.get_features_by_category('operator_sensitivity')]

        # Post-hoc extracted features (temporal, causality, invariant_drift)
        feature_names_in_order = temporal_names + causality_names + invariant_drift_names

        feature_list = []
        for name in feature_names_in_order:
            if name not in all_trajectory_features:
                # Skip features not extracted (e.g., if disabled in config)
                continue

            feat = all_trajectory_features[name]

            if feat.ndim == 3:  # [N, M, C]
                # Average across channels
                feat_avg = feat.mean(dim=2)  # [N, M]
            elif feat.ndim == 2:  # [N, M] (already channel-averaged, e.g., invariant drift)
                feat_avg = feat  # [N, M]
            else:
                raise ValueError(f"Unexpected feature shape for {name}: {feat.shape}")

            feature_list.append(feat_avg.unsqueeze(-1))  # [N, M, 1]

        # Append operator sensitivity features (inline-extracted, scalar per sample)
        # These need to be broadcast across realizations: [N] → [N, M, 1]
        N, M = trajectories.shape[0], trajectories.shape[1]
        for name in operator_sensitivity_names:
            if name in operator_sensitivity_features:
                # Scalar feature [N] → broadcast to [N, M, 1]
                feat_scalar = operator_sensitivity_features[name]  # [N]
                feat_broadcast = feat_scalar.unsqueeze(1).unsqueeze(2).expand(N, M, 1)  # [N, M, 1]
                feature_list.append(feat_broadcast)
            else:
                # Feature not provided, fill with NaN
                feat_nan = torch.full((N, M, 1), float('nan'), device=self.device)
                feature_list.append(feat_nan)

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

    def extract_all(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all SDF features (convenience method).

        Runs full three-stage pipeline and returns all feature representations.

        Args:
            trajectories: Stochastic trajectories [N, M, T, C, H, W]
            metadata: Optional metadata dict

        Returns:
            Dictionary with keys:
            - 'per_timestep': Features [N, T, D]
            - 'per_trajectory': Features [N, M, D_traj]
            - 'aggregated_mean': Features [N, D_final] (mean across realizations)
            - 'aggregated_std': Features [N, D_final] (std across realizations)
            - 'aggregated_cv': Features [N, D_final] (cv across realizations)
        """
        # Stage 1: Per-timestep features
        per_timestep = self.extract_per_timestep(trajectories, metadata)

        # Stage 2: Per-trajectory features
        per_trajectory = self.extract_per_trajectory(trajectories, metadata)

        # Stage 3: Aggregate across realizations
        agg_methods = self.config.realization_aggregation if self.config else ['mean', 'std', 'cv']

        aggregated = {}
        for method in agg_methods:
            aggregated[f'aggregated_{method}'] = self.aggregate_realizations(
                per_trajectory,
                method=method
            )

        return {
            'per_timestep': per_timestep,
            'per_trajectory': per_trajectory,
            **aggregated
        }
