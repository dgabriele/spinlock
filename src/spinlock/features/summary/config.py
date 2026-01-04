"""
SUMMARY feature family configuration.

Configuration schemas for all SUMMARY feature categories:
- Spatial statistics
- Spectral/frequency features
- Distributional features
- Temporal dynamics
- Structural features
- Statistical physics
- Morphological features
- Multiscale analysis

Example:
    >>> from spinlock.features.summary.config import SummaryConfig
    >>> config = SummaryConfig()  # All features enabled with defaults
    >>> config.spatial.include_mean
    True
"""

from typing import Literal, List, Optional
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# SUMMARY Feature Category Configurations
# =============================================================================

class SummarySpatialConfig(BaseModel):
    """
    Spatial statistics feature configuration.

    Features: moments, gradients, curvature
    """

    enabled: bool = True

    # Basic moments
    include_mean: bool = True
    include_variance: bool = True
    include_std: bool = True
    include_skewness: bool = True
    include_kurtosis: bool = True
    include_min: bool = True
    include_max: bool = True
    include_range: bool = True  # max - min

    # Robust statistics
    include_iqr: bool = True  # Interquartile range
    include_mad: bool = True  # Median absolute deviation

    # Distribution percentiles (Phase 1 extension)
    include_percentiles: bool = True  # 5%, 25%, 50%, 75%, 95%

    # Histogram/occupancy features (Phase 2 extension)
    include_histogram: bool = False  # State space coverage (opt-in, moderate cost)
    histogram_num_bins: int = Field(default=16, ge=8, le=64)  # Number of histogram bins

    # Gradients
    include_gradient_magnitude: bool = True
    include_gradient_x_mean: bool = True
    include_gradient_y_mean: bool = True
    include_gradient_anisotropy: bool = True

    # Curvature (second derivatives)
    include_laplacian: bool = True
    include_hessian_trace: bool = False  # More expensive
    include_hessian_det: bool = False  # More expensive


class SummarySpectralConfig(BaseModel):
    """
    Spectral/frequency feature configuration.

    Features: FFT power spectrum, dominant frequencies, spectral ratios
    """

    enabled: bool = True

    # FFT power spectrum (multiscale)
    num_fft_scales: int = Field(default=5, ge=1, le=10)
    include_fft_power: bool = True

    # Dominant frequencies
    include_dominant_freq: bool = True
    include_dominant_freq_magnitude: bool = True

    # Spectral centroids (power-weighted frequency center)
    include_spectral_centroid_x: bool = True
    include_spectral_centroid_y: bool = True
    include_spectral_bandwidth: bool = True

    # Spectral ratios (energy distribution across frequency bands)
    include_low_freq_ratio: bool = True
    include_mid_freq_ratio: bool = True
    include_high_freq_ratio: bool = True
    include_spectral_flatness: bool = True  # Tonality measure
    include_spectral_rolloff: bool = True  # 85th percentile frequency

    # Anisotropy
    include_spectral_anisotropy: bool = True
    include_spectral_orientation: bool = False

    @field_validator('num_fft_scales')
    @classmethod
    def validate_fft_scales(cls, v: int) -> int:
        """Ensure FFT scales is reasonable."""
        if v < 1 or v > 10:
            raise ValueError("num_fft_scales must be between 1 and 10")
        return v


class SummaryDistributionalConfig(BaseModel):
    """
    Distribution-based feature configuration.

    Features: entropy, complexity, compression metrics
    """

    enabled: bool = False  # v2.1 Phase 2: Disabled by default (3.2s overhead) - opt-in via config

    # Entropy (multiscale via coarse-graining)
    num_entropy_scales: int = Field(default=3, ge=1, le=5)
    include_entropy: bool = True

    # Complexity measures
    include_sample_entropy: bool = True  # Regularity measure
    include_approximate_entropy: bool = True  # Pattern predictability
    include_lempel_ziv_complexity: bool = False  # Expensive

    # Compression-based features
    include_svd_entropy: bool = True  # Entropy of singular value spectrum
    include_participation_ratio: bool = True  # Effective dimensionality
    include_compression_ratio_pca: bool = True  # 90% variance capture

    # Quantiles
    include_quantiles: bool = True  # p10, p25, p50, p75, p90

    @field_validator('num_entropy_scales')
    @classmethod
    def validate_entropy_scales(cls, v: int) -> int:
        """Ensure entropy scales is reasonable."""
        if v < 1 or v > 5:
            raise ValueError("num_entropy_scales must be between 1 and 5")
        return v


class SummaryTemporalConfig(BaseModel):
    """
    Temporal dynamics feature configuration.

    Features: growth rates, oscillations, stability, stationarity

    Note: These are trajectory-level features computed once per realization.
    """

    enabled: bool = True

    # Growth & decay
    include_energy_growth_rate: bool = True
    include_energy_growth_accel: bool = True  # Second derivative
    include_variance_growth_rate: bool = True
    include_mean_growth_rate: bool = False

    # Oscillations
    include_temporal_freq_dominant: bool = True  # FFT of energy time series
    include_oscillation_amplitude: bool = True
    include_oscillation_period: bool = True
    include_autocorr_decay_time: bool = True  # Exponential decay timescale

    # Stability metrics
    include_lyapunov_approx: bool = True  # Approximate Lyapunov exponent
    include_trajectory_smoothness: bool = True
    include_regime_switches: bool = True  # Number of sign changes in growth
    include_final_to_initial_ratio: bool = True

    # Stationarity
    include_stationarity_test: bool = False  # ADF test (expensive)
    include_trend_strength: bool = True  # R² of linear trend fit
    include_detrended_variance: bool = True

    # Event detection features (Phase 1 extension)
    include_event_counts: bool = True  # Spikes, bursts, zero-crossings
    include_time_to_event: bool = True  # Time to threshold crossings

    # Rolling window statistics (Phase 1 CRITICAL: multi-timescale analysis)
    include_rolling_windows: bool = True
    rolling_window_fractions: List[float] = Field(
        default_factory=lambda: [0.05, 0.10, 0.20]
    )

    # Autocorrelation settings
    autocorr_max_lag: int = Field(default=20, ge=1, le=100)

    # Phase 2 extension: PACF (Partial Autocorrelation Function)
    include_pacf: bool = False  # Partial autocorrelation (opt-in, moderate cost)
    pacf_max_lag: int = Field(default=10, ge=1, le=50)  # PACF lag count

    @field_validator('autocorr_max_lag')
    @classmethod
    def validate_max_lag(cls, v: int) -> int:
        """Ensure max lag is reasonable."""
        if v < 1 or v > 100:
            raise ValueError("autocorr_max_lag must be between 1 and 100")
        return v

    @field_validator('pacf_max_lag')
    @classmethod
    def validate_pacf_lag(cls, v: int) -> int:
        """Ensure PACF lag is reasonable."""
        if v < 1 or v > 50:
            raise ValueError("pacf_max_lag must be between 1 and 50")
        return v


class SummaryStructuralConfig(BaseModel):
    """
    Structural feature configuration.

    Features: connectivity, topology, edges, texture
    """

    enabled: bool = False  # v2.1 Phase 2: Disabled by default (1.0s overhead) - opt-in via config

    # Connectivity & topology
    include_num_connected_components: bool = True
    include_largest_component_size: bool = True
    include_component_size_mean: bool = True
    include_component_size_std: bool = True
    include_euler_characteristic: bool = False  # Topological invariant (expensive)

    # Edge & boundary
    include_edge_density: bool = True
    include_edge_length_total: bool = True
    include_edge_curvature_mean: bool = False  # Expensive
    include_boundary_smoothness: bool = False  # Expensive

    # Texture (GLCM features)
    include_glcm_contrast: bool = True
    include_glcm_homogeneity: bool = True
    include_glcm_energy: bool = True
    include_glcm_correlation: bool = True

    # Thresholds
    component_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    edge_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator('component_threshold', 'edge_threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure thresholds are in valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v


class SummaryPhysicsConfig(BaseModel):
    """
    Statistical physics feature configuration.

    Features: correlation functions, structure factor, fluctuations
    """

    enabled: bool = False  # v2.1 Phase 2: Disabled by default (0.55s overhead) - opt-in via config

    # Correlation functions (multiscale via binning)
    num_correlation_scales: int = Field(default=3, ge=1, le=5)
    include_correlation_length: bool = True
    include_correlation_peak: bool = True

    # Structure factor S(k)
    include_structure_factor_peak: bool = True  # S(k) maximum location
    include_structure_factor_width: bool = True
    include_structure_factor_integral: bool = True

    # Fluctuations
    include_density_fluctuation: bool = True
    include_compressibility_proxy: bool = True
    include_clustering_coefficient: bool = True

    @field_validator('num_correlation_scales')
    @classmethod
    def validate_correlation_scales(cls, v: int) -> int:
        """Ensure correlation scales is reasonable."""
        if v < 1 or v > 5:
            raise ValueError("num_correlation_scales must be between 1 and 5")
        return v


class SummaryMorphologicalConfig(BaseModel):
    """
    Morphological feature configuration.

    Features: shape descriptors, image moments, granulometry
    """

    enabled: bool = False  # v2.1 Phase 2: Disabled by default (0.47s overhead) - opt-in via config

    # Shape descriptors
    include_area_fraction: bool = True  # Fraction above threshold
    include_perimeter_total: bool = True
    include_shape_circularity: bool = True  # 4π × Area / Perimeter²
    include_shape_eccentricity: bool = True  # Major / minor axis ratio
    include_shape_solidity: bool = True  # Area / convex hull area
    include_shape_extent: bool = True  # Area / bounding box area

    # Image moments (Hu invariants)
    include_moment_hu_1: bool = True
    include_moment_hu_2: bool = True
    include_centroid_x: bool = True
    include_centroid_y: bool = True
    include_centroid_displacement: bool = True  # Distance from grid center

    # Granulometry (size distribution)
    include_granulometry_mean: bool = True
    include_granulometry_std: bool = True

    # Threshold for shape analysis
    shape_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator('shape_threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure threshold is in valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("shape_threshold must be between 0.0 and 1.0")
        return v


class SummaryMultiscaleConfig(BaseModel):
    """
    Multiscale analysis feature configuration.

    Features: wavelet coefficients, Laplacian pyramid, scale-space extrema
    """

    enabled: bool = True

    # Wavelet decomposition
    wavelet: Literal["haar", "db4", "coif1"] = "haar"
    num_wavelet_levels: int = Field(default=4, ge=1, le=6)
    include_wavelet_energy: bool = True
    include_wavelet_mean: bool = True
    include_wavelet_std: bool = True

    # Laplacian pyramid
    num_pyramid_levels: int = Field(default=4, ge=1, le=6)
    include_pyramid_energy: bool = True
    include_pyramid_contrast: bool = True

    # Scale-space extrema (blob detection)
    include_scale_space_extrema: bool = False  # Expensive
    include_extrema_scale_mean: bool = False
    include_extrema_scale_std: bool = False

    @field_validator('num_wavelet_levels', 'num_pyramid_levels')
    @classmethod
    def validate_levels(cls, v: int) -> int:
        """Ensure levels is reasonable."""
        if v < 1 or v > 6:
            raise ValueError("Decomposition levels must be between 1 and 6")
        return v


class SummaryCrossChannelConfig(BaseModel):
    """
    Cross-channel interaction feature configuration.

    Features: correlation spectra, coherence, mutual information

    Measures channel coupling structure at each timestep. Optimized for
    Mid-C operators (5-16 channels), degrades gracefully for High-C (32+).

    Note: These are per-timestep features extracted at each time point.
    """

    enabled: bool = True

    # Correlation matrix eigendecomposition (always included)
    num_eigen_top: int = Field(default=3, ge=1, le=10)
    include_eigen_values: bool = True
    include_eigen_trace: bool = True
    include_condition_number: bool = True
    include_participation_ratio: bool = True

    # Pairwise correlation statistics (fallback summary)
    include_corr_mean: bool = True
    include_corr_max: bool = True
    include_corr_min: bool = True
    include_corr_std: bool = True

    # Cross-spectral coherence (temporal dynamics, expensive)
    include_coherence: bool = False  # Default off (expensive)
    coherence_freq_bands: List[str] = Field(
        default_factory=lambda: ["low", "mid", "high"]
    )

    # Mutual information (nonlinear coupling, expensive)
    include_mutual_info: bool = True  # Enabled by default (information-theoretic coupling)
    mi_num_bins: int = Field(default=16, ge=8, le=32)

    # Edge case handling
    max_channels_for_full_corr: int = Field(default=16, ge=4, le=128)

    @field_validator('num_eigen_top')
    @classmethod
    def validate_eigen_top(cls, v: int) -> int:
        """Ensure num_eigen_top is reasonable."""
        if v < 1 or v > 10:
            raise ValueError("num_eigen_top must be between 1 and 10")
        return v

    @field_validator('mi_num_bins')
    @classmethod
    def validate_mi_bins(cls, v: int) -> int:
        """Ensure MI bin count is reasonable."""
        if v < 8 or v > 32:
            raise ValueError("mi_num_bins must be between 8 and 32")
        return v

    @field_validator('max_channels_for_full_corr')
    @classmethod
    def validate_max_channels(cls, v: int) -> int:
        """Ensure max_channels is reasonable."""
        if v < 4 or v > 128:
            raise ValueError("max_channels_for_full_corr must be between 4 and 128")
        return v


class SummaryOperatorSensitivityConfig(BaseModel):
    """
    Operator sensitivity feature configuration.

    Features: Lipschitz estimates, gain curves, linearity metrics

    Characterizes how neural operators respond to input perturbations by
    re-executing operators with perturbed inputs. Measures local sensitivity,
    amplitude response, and nonlinearity.

    CRITICAL: This extractor requires access to the operator during extraction.
    It must be called during dataset generation when operators are in memory.

    Note: These are trajectory-level features extracted during rollout.
    Expensive (requires multiple forward passes per operator).
    """

    enabled: bool = True

    # Lipschitz constant estimation (local sensitivity to noise)
    include_lipschitz: bool = True
    lipschitz_epsilon_scales: List[float] = Field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2]
    )

    # Gain curves (response to amplitude scaling)
    include_gain_curve: bool = True
    gain_scale_factors: List[float] = Field(
        default_factory=lambda: [0.5, 0.75, 1.25, 1.5]
    )

    # Linearity metrics (R², saturation, compression)
    include_linearity_metrics: bool = True

    @field_validator('lipschitz_epsilon_scales')
    @classmethod
    def validate_lipschitz_scales(cls, v: List[float]) -> List[float]:
        """Ensure Lipschitz scales are positive and reasonable."""
        if not v:
            raise ValueError("lipschitz_epsilon_scales must be non-empty")
        if any(eps <= 0 or eps > 1.0 for eps in v):
            raise ValueError("Lipschitz epsilon scales must be in (0, 1.0]")
        return v

    @field_validator('gain_scale_factors')
    @classmethod
    def validate_gain_scales(cls, v: List[float]) -> List[float]:
        """Ensure gain scales are positive and reasonable."""
        if not v:
            raise ValueError("gain_scale_factors must be non-empty")
        if any(scale <= 0 or scale > 10.0 for scale in v):
            raise ValueError("Gain scale factors must be in (0, 10.0]")
        return v


class SummaryCausalityConfig(BaseModel):
    """
    Causality/directionality feature configuration.

    Features: temporal information flow, lagged correlations, transfer entropy

    Detects directional asymmetry and information flow in temporal dynamics
    using time-lagged correlations, prediction error asymmetry, and optional
    information-theoretic measures.

    Note: These are trajectory-level features computed once per realization.
    Requires T > 1 for meaningful results.
    """

    enabled: bool = True

    # Complexity level controls which features to extract
    complexity_level: Literal["fast", "medium", "full"] = "fast"

    # Level 1: Fast (lagged correlation, prediction error, irreversibility)
    max_lag_correlation: int = Field(default=3, ge=1, le=10)
    max_lag_prediction: int = Field(default=2, ge=1, le=5)
    include_time_irreversibility: bool = True
    include_spatial_flow: bool = True

    # Level 2: Medium (transfer entropy, Granger causality)
    include_transfer_entropy: bool = False  # Expensive, default off
    include_granger_causality: bool = False  # Expensive, default off
    transfer_entropy_num_bins: int = Field(default=8, ge=4, le=32)
    granger_ar_order: int = Field(default=2, ge=1, le=5)

    @field_validator('max_lag_correlation')
    @classmethod
    def validate_max_lag_corr(cls, v: int) -> int:
        """Ensure max_lag_correlation is reasonable."""
        if v < 1 or v > 10:
            raise ValueError("max_lag_correlation must be between 1 and 10")
        return v

    @field_validator('max_lag_prediction')
    @classmethod
    def validate_max_lag_pred(cls, v: int) -> int:
        """Ensure max_lag_prediction is reasonable."""
        if v < 1 or v > 5:
            raise ValueError("max_lag_prediction must be between 1 and 5")
        return v

    @field_validator('transfer_entropy_num_bins')
    @classmethod
    def validate_te_bins(cls, v: int) -> int:
        """Ensure transfer entropy bin count is reasonable."""
        if v < 4 or v > 32:
            raise ValueError("transfer_entropy_num_bins must be between 4 and 32")
        return v

    @field_validator('granger_ar_order')
    @classmethod
    def validate_granger_order(cls, v: int) -> int:
        """Ensure Granger AR order is reasonable."""
        if v < 1 or v > 5:
            raise ValueError("granger_ar_order must be between 1 and 5")
        return v


class SummaryNonlinearConfig(BaseModel):
    """
    Nonlinear dynamics feature configuration (Phase 1 extension).

    Features: Recurrence Quantification Analysis (RQA), correlation dimension

    These features are computationally expensive (O(T²)) and use temporal
    subsampling for efficiency. Default: disabled (opt-in).

    Note: These are trajectory-level features computed once per realization.
    """

    enabled: bool = False  # Expensive, opt-in by default

    # Recurrence Quantification Analysis
    include_recurrence: bool = True  # RQA metrics (if enabled)
    rqa_epsilon: float = Field(default=0.1, ge=0.01, le=1.0)  # Recurrence threshold
    rqa_embedding_dim: int = Field(default=3, ge=2, le=10)  # Phase space dimension
    rqa_tau: int = Field(default=1, ge=1, le=10)  # Time delay
    rqa_subsample_factor: int = Field(default=10, ge=1, le=50)  # Temporal subsampling

    # Correlation dimension
    include_correlation_dim: bool = True  # Attractor dimension (if enabled)
    corr_dim_embedding_dim: int = Field(default=5, ge=2, le=10)
    corr_dim_tau: int = Field(default=1, ge=1, le=10)
    corr_dim_subsample_factor: int = Field(default=10, ge=1, le=50)

    # Phase 2 extension: Permutation entropy
    include_permutation_entropy: bool = False  # Ordinal pattern complexity (opt-in)
    perm_entropy_embedding_dim: int = Field(default=3, ge=2, le=7)
    perm_entropy_tau: int = Field(default=1, ge=1, le=10)
    perm_entropy_subsample_factor: int = Field(default=10, ge=1, le=50)

    @field_validator('rqa_epsilon')
    @classmethod
    def validate_rqa_epsilon(cls, v: float) -> float:
        """Ensure RQA epsilon is reasonable."""
        if v < 0.01 or v > 1.0:
            raise ValueError("rqa_epsilon must be between 0.01 and 1.0")
        return v

    @field_validator('rqa_subsample_factor', 'corr_dim_subsample_factor', 'perm_entropy_subsample_factor')
    @classmethod
    def validate_subsample(cls, v: int) -> int:
        """Ensure subsampling factor is reasonable."""
        if v < 1 or v > 50:
            raise ValueError("Subsample factor must be between 1 and 50")
        return v


class SummaryInvariantDriftConfig(BaseModel):
    """
    Invariant drift feature configuration.

    Features: norm-based drift tracking with multiscale filtering

    Tracks generic norms (L1, L2, L∞, entropy, total variation) across
    raw, low-pass, and high-pass filtered fields to characterize operator
    stability, dissipation, and scale-specific dynamics.

    Note: These are trajectory-level features computed once per realization.
    """

    enabled: bool = True

    # Mandatory generic norms (always computed by default)
    include_L1_drift: bool = True
    include_L2_drift: bool = True
    include_Linf_drift: bool = True
    include_entropy_drift: bool = True
    include_tv_drift: bool = True  # Total variation

    # Multi-scale filtering
    num_scales: int = Field(default=3, ge=1, le=3)  # raw, low-pass, high-pass
    gaussian_sigma: float = Field(default=2.0, ge=0.5, le=5.0)

    # Entropy settings
    entropy_num_bins: int = Field(default=32, ge=8, le=128)

    # Optional physical invariants (conditional, config-gated)
    include_mass_drift: bool = False  # Scalar fields only
    include_energy_drift: bool = False  # L2 and gradient energy
    include_divergence_drift: bool = False  # Vector fields only (not implemented yet)

    @field_validator('num_scales')
    @classmethod
    def validate_scales(cls, v: int) -> int:
        """Ensure num_scales is valid."""
        if v not in [1, 2, 3]:
            raise ValueError("num_scales must be 1 (raw only), 2 (raw+low), or 3 (raw+low+high)")
        return v

    @field_validator('gaussian_sigma')
    @classmethod
    def validate_sigma(cls, v: float) -> float:
        """Ensure Gaussian sigma is reasonable."""
        if v < 0.5 or v > 5.0:
            raise ValueError("gaussian_sigma must be between 0.5 and 5.0")
        return v

    @field_validator('entropy_num_bins')
    @classmethod
    def validate_bins(cls, v: int) -> int:
        """Ensure bin count is reasonable."""
        if v < 8 or v > 128:
            raise ValueError("entropy_num_bins must be between 8 and 128")
        return v


# =============================================================================
# Learned SUMMARY Features (Phase 2)
# =============================================================================


class LearnedSummaryConfig(BaseModel):
    """
    Learned SUMMARY feature configuration.

    Extracts features from U-AFNO intermediate representations:
    - Bottleneck latents (default): Global spectral features after AFNO
    - Skip connections (optional): Multi-scale encoder features

    Aggregation pipeline:
    1. Temporal: Pool across T timesteps (mean, max, or concatenated)
    2. Spatial: Global average pooling across H, W
    3. Optional: Project to fixed dimension via MLP

    Note: Only available for U-AFNO operators. CNN operators do not support
    learned feature extraction.

    Example:
        >>> config = LearnedSummaryConfig(
        ...     enabled=True,
        ...     extract_from="all",
        ...     temporal_agg="mean_max",
        ... )
    """

    enabled: bool = Field(
        default=False,
        description="Enable learned feature extraction from U-AFNO latents"
    )

    extract_from: Literal["bottleneck", "skips", "all"] = Field(
        default="bottleneck",
        description="Which latents to extract: bottleneck only, skips only, or all"
    )

    skip_levels: List[int] = Field(
        default_factory=lambda: [0, 1, 2],
        description="Which encoder levels to extract (0=shallowest, used when extract_from='skips' or 'all')"
    )

    temporal_agg: Literal["mean", "max", "mean_max", "std"] = Field(
        default="mean_max",
        description="Temporal aggregation: mean, max, mean+max concatenated, or std"
    )

    spatial_agg: Literal["gap", "flatten"] = Field(
        default="gap",
        description="Spatial aggregation: global average pooling (gap) or flatten"
    )

    projection_dim: Optional[int] = Field(
        default=None,
        ge=8,
        le=512,
        description="Optional projection to fixed dimension via MLP (None = raw latents)"
    )

    # Training config for learned features
    training_epochs: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of epochs to train each operator on next-step prediction"
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0,
        description="Learning rate for operator training (Adam optimizer)"
    )
    lr_scheduler: Literal["constant", "cosine"] = Field(
        default="cosine",
        description="Learning rate schedule: constant or cosine annealing"
    )
    early_stopping_patience: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Stop training if no improvement for this many epochs"
    )

    @field_validator('skip_levels')
    @classmethod
    def validate_skip_levels(cls, v: List[int]) -> List[int]:
        """Ensure skip levels are valid."""
        if not v:
            raise ValueError("skip_levels must be non-empty")
        for level in v:
            if level < 0 or level > 5:
                raise ValueError("skip_levels must be in range [0, 5]")
        return v


# =============================================================================
# SUMMARY Top-Level Configuration
# =============================================================================

class SummaryConfig(BaseModel):
    """
    SUMMARY feature family configuration.

    Controls which SUMMARY features to extract, multiscale parameters,
    and aggregation settings.

    Attributes:
        spatial: Spatial statistics configuration
        spectral: Spectral/frequency features configuration
        distributional: Distribution-based features configuration
        temporal: Temporal dynamics configuration
        structural: Structural features configuration
        physics: Statistical physics configuration
        morphological: Morphological features configuration
        multiscale: Multiscale analysis configuration
        per_channel: Extract features per-channel or aggregate across channels
        temporal_aggregation: Temporal aggregation methods for per-timestep features
    """

    # Feature category configs (v1.0)
    spatial: SummarySpatialConfig = Field(default_factory=SummarySpatialConfig)
    spectral: SummarySpectralConfig = Field(default_factory=SummarySpectralConfig)
    distributional: SummaryDistributionalConfig = Field(default_factory=SummaryDistributionalConfig)
    temporal: SummaryTemporalConfig = Field(default_factory=SummaryTemporalConfig)
    structural: SummaryStructuralConfig = Field(default_factory=SummaryStructuralConfig)
    physics: SummaryPhysicsConfig = Field(default_factory=SummaryPhysicsConfig)
    morphological: SummaryMorphologicalConfig = Field(default_factory=SummaryMorphologicalConfig)
    multiscale: SummaryMultiscaleConfig = Field(default_factory=SummaryMultiscaleConfig)

    # v2.0 feature categories (new)
    operator_sensitivity: SummaryOperatorSensitivityConfig = Field(default_factory=SummaryOperatorSensitivityConfig)
    cross_channel: SummaryCrossChannelConfig = Field(default_factory=SummaryCrossChannelConfig)
    causality: SummaryCausalityConfig = Field(default_factory=SummaryCausalityConfig)
    invariant_drift: SummaryInvariantDriftConfig = Field(default_factory=SummaryInvariantDriftConfig)
    nonlinear: SummaryNonlinearConfig = Field(default_factory=SummaryNonlinearConfig)  # Phase 1 extension

    # Learned features (Phase 2) - extract from U-AFNO latents
    learned: LearnedSummaryConfig = Field(
        default_factory=LearnedSummaryConfig,
        description="Learned feature extraction from U-AFNO intermediate representations"
    )

    # Summary mode toggle
    summary_mode: Literal["manual", "learned", "hybrid"] = Field(
        default="manual",
        description="Feature mode: manual (hand-crafted), learned (U-AFNO latents), hybrid (both concatenated)"
    )

    # Per-timestep extraction toggle
    extract_per_timestep: bool = Field(
        default=True,
        description="Extract per-timestep (TEMPORAL) features. Disable for SUMMARY-only mode."
    )

    # Aggregation settings
    per_channel: bool = True  # Extract features per-channel or aggregate across channels
    temporal_aggregation: List[Literal["mean", "std", "min", "max", "final"]] = Field(
        default_factory=lambda: ["mean", "std"]
    )
    realization_aggregation: List[Literal["mean", "std", "min", "max", "cv"]] = Field(
        default_factory=lambda: ["mean", "std", "cv"]
    )

    @field_validator('temporal_aggregation', 'realization_aggregation')
    @classmethod
    def validate_aggregations(cls, v: List[str]) -> List[str]:
        """Ensure at least one aggregation method is specified."""
        if not v:
            raise ValueError("Must specify at least one aggregation method")
        return v

    def estimate_feature_count(self) -> int:
        """
        Estimate total number of features that will be extracted.

        This is an approximation since exact count depends on multiscale
        parameters and which features are enabled.

        Returns:
            Estimated feature count
        """
        count = 0

        # Spatial (roughly 10 base features × aggregations)
        if self.spatial.enabled:
            base_count = sum([
                self.spatial.include_mean,
                self.spatial.include_variance,
                self.spatial.include_std,
                self.spatial.include_skewness,
                self.spatial.include_kurtosis,
                self.spatial.include_min,
                self.spatial.include_max,
                self.spatial.include_range,
                self.spatial.include_iqr,
                self.spatial.include_mad,
                self.spatial.include_gradient_magnitude,
                self.spatial.include_gradient_x_mean,
                self.spatial.include_gradient_y_mean,
                self.spatial.include_gradient_anisotropy,
                self.spatial.include_laplacian,
                self.spatial.include_hessian_trace,
                self.spatial.include_hessian_det,
            ])
            # Phase 1 extension: percentiles
            if self.spatial.include_percentiles:
                base_count += 5  # 5th, 25th, 50th, 75th, 95th percentiles
            count += base_count * len(self.temporal_aggregation)

        # Spectral (multiscale FFT + other features)
        if self.spectral.enabled:
            # FFT power spectrum (3 features per scale)
            if self.spectral.include_fft_power:
                count += 3 * self.spectral.num_fft_scales * len(self.temporal_aggregation)
            # Other spectral features
            base_count = sum([
                self.spectral.include_dominant_freq,
                self.spectral.include_dominant_freq_magnitude,
                self.spectral.include_spectral_centroid_x,
                self.spectral.include_spectral_centroid_y,
                self.spectral.include_spectral_bandwidth,
                self.spectral.include_low_freq_ratio,
                self.spectral.include_mid_freq_ratio,
                self.spectral.include_high_freq_ratio,
                self.spectral.include_spectral_flatness,
                self.spectral.include_spectral_rolloff,
                self.spectral.include_spectral_anisotropy,
                self.spectral.include_spectral_orientation,
            ])
            count += base_count * len(self.temporal_aggregation)

        # Distributional
        if self.distributional.enabled:
            # Entropy (multiscale)
            if self.distributional.include_entropy:
                count += self.distributional.num_entropy_scales * len(self.temporal_aggregation)
            # Other features
            base_count = sum([
                self.distributional.include_sample_entropy,
                self.distributional.include_approximate_entropy,
                self.distributional.include_lempel_ziv_complexity,
                self.distributional.include_svd_entropy,
                self.distributional.include_participation_ratio,
                self.distributional.include_compression_ratio_pca,
            ])
            count += base_count * len(self.temporal_aggregation)
            if self.distributional.include_quantiles:
                count += 5 * len(self.temporal_aggregation)  # 5 quantiles

        # Temporal (trajectory-level, no temporal aggregation)
        if self.temporal.enabled:
            base_count = sum([
                self.temporal.include_energy_growth_rate,
                self.temporal.include_energy_growth_accel,
                self.temporal.include_variance_growth_rate,
                self.temporal.include_mean_growth_rate,
                self.temporal.include_temporal_freq_dominant,
                self.temporal.include_oscillation_amplitude,
                self.temporal.include_oscillation_period,
                self.temporal.include_autocorr_decay_time,
                self.temporal.include_lyapunov_approx,
                self.temporal.include_trajectory_smoothness,
                self.temporal.include_regime_switches,
                self.temporal.include_final_to_initial_ratio,
                self.temporal.include_stationarity_test,
                self.temporal.include_trend_strength,
                self.temporal.include_detrended_variance,
            ])
            # Phase 1 extensions
            if self.temporal.include_event_counts:
                base_count += 3  # num_spikes, num_bursts, num_zero_crossings
            if self.temporal.include_time_to_event:
                base_count += 2  # time_to_0.5x, time_to_2.0x
            if self.temporal.include_rolling_windows:
                # 6 features per window × 3 windows = 18 features
                base_count += 6 * len(self.temporal.rolling_window_fractions)
            count += base_count * len(self.realization_aggregation)

        # Structural, Physics, Morphological (similar pattern)
        # Adding rough estimates for brevity
        if self.structural.enabled:
            count += 10 * len(self.temporal_aggregation)
        if self.physics.enabled:
            count += 8 * len(self.temporal_aggregation)
        if self.morphological.enabled:
            count += 12 * len(self.temporal_aggregation)

        # Multiscale
        if self.multiscale.enabled:
            # Wavelets (3 features per level)
            count += 3 * self.multiscale.num_wavelet_levels * len(self.temporal_aggregation)
            # Pyramids (2 features per level)
            count += 2 * self.multiscale.num_pyramid_levels * len(self.temporal_aggregation)

        # Operator sensitivity (v2.0, trajectory-level, extracted during rollout)
        if self.operator_sensitivity is not None and self.operator_sensitivity.enabled:
            ops_count = 0

            # Lipschitz estimates (one per epsilon scale)
            if self.operator_sensitivity.include_lipschitz:
                ops_count += len(self.operator_sensitivity.lipschitz_epsilon_scales)

            # Gain curves (one per scale factor)
            if self.operator_sensitivity.include_gain_curve:
                ops_count += len(self.operator_sensitivity.gain_scale_factors)

            # Linearity metrics (3 features)
            if self.operator_sensitivity.include_linearity_metrics:
                ops_count += 3  # linearity_r2, saturation_degree, compression_ratio

            # Multiply by realization aggregation (trajectory-level features)
            count += ops_count * len(self.realization_aggregation)

        # Cross-channel (v2.0, per-timestep)
        if self.cross_channel is not None and self.cross_channel.enabled:
            # Eigenvalue features
            cross_ch_count = (
                self.cross_channel.num_eigen_top +  # Top eigenvalues
                3  # trace, condition number, participation ratio
            )
            # Correlation statistics
            cross_ch_count += 4  # mean, max, min, std

            # Optional coherence features
            if self.cross_channel.include_coherence:
                cross_ch_count += len(self.cross_channel.coherence_freq_bands)  # Per-band coherence
                cross_ch_count += 2  # max coherence, peak freq

            # Optional MI features
            if self.cross_channel.include_mutual_info:
                cross_ch_count += 2  # mean MI, max MI

            # Multiply by temporal aggregation (per-timestep features)
            count += cross_ch_count * len(self.temporal_aggregation)

        # Causality (v2.0, trajectory-level)
        if self.causality is not None and self.causality.enabled:
            # Level 1 (Fast): lagged correlation asymmetry + prediction error + irreversibility + spatial flow
            causality_count = 0

            # Lagged correlation asymmetry: 2 features × max_lag
            causality_count += 2 * self.causality.max_lag_correlation

            # Prediction error ratio: 2 features × max_lag_prediction
            causality_count += 2 * self.causality.max_lag_prediction

            # Time irreversibility: 2 features (if enabled)
            if self.causality.include_time_irreversibility:
                causality_count += 2

            # Spatial information flow: 2 features (if enabled)
            if self.causality.include_spatial_flow:
                causality_count += 2

            # Level 2 (Medium): optional transfer entropy and Granger causality
            if self.causality.include_transfer_entropy:
                causality_count += 3  # mean, max, asymmetry

            if self.causality.include_granger_causality:
                causality_count += 3  # mean, max, asymmetry

            # Multiply by realization aggregation (trajectory-level features)
            count += causality_count * len(self.realization_aggregation)

        # Invariant drift (v2.0, trajectory-level)
        if self.invariant_drift is not None and self.invariant_drift.enabled:
            # Generic norms: 5 norms × 4 metrics × num_scales
            num_norms = sum([
                self.invariant_drift.include_L1_drift,
                self.invariant_drift.include_L2_drift,
                self.invariant_drift.include_Linf_drift,
                self.invariant_drift.include_entropy_drift,
                self.invariant_drift.include_tv_drift,
            ])
            num_metrics = 4  # mean_drift, variance, ratio, monotonicity
            num_scales = self.invariant_drift.num_scales
            drift_base_count = num_norms * num_metrics * num_scales

            # Optional physical invariants
            if self.invariant_drift.include_mass_drift:
                drift_base_count += num_metrics * num_scales  # mass × 4 metrics × scales
            if self.invariant_drift.include_energy_drift:
                drift_base_count += 2 * num_metrics * num_scales  # (L2 + gradient) × 4 × scales

            # Multiply by realization aggregation (trajectory-level features)
            count += drift_base_count * len(self.realization_aggregation)

        # Nonlinear (Phase 1 extension, trajectory-level, expensive)
        if self.nonlinear is not None and self.nonlinear.enabled:
            nonlinear_count = 0

            # RQA metrics: 4 features (recurrence_rate, determinism, laminarity, entropy)
            if self.nonlinear.include_recurrence:
                nonlinear_count += 4

            # Correlation dimension: 1 feature
            if self.nonlinear.include_correlation_dim:
                nonlinear_count += 1

            # Multiply by realization aggregation (trajectory-level features)
            count += nonlinear_count * len(self.realization_aggregation)

        return count

    @classmethod
    def from_schema_config(cls, schema_config: "SummaryFeaturesConfig") -> "SummaryConfig":
        """
        Create a SummaryConfig from a schema SummaryFeaturesConfig.

        This factory method bridges the gap between the schema config
        (used by SpinlockConfig for YAML parsing) and the extractor config
        (used by SummaryExtractor for feature extraction).

        Args:
            schema_config: SummaryFeaturesConfig from spinlock.config.schema

        Returns:
            SummaryConfig with summary_mode and learned settings from schema
        """
        # Import here to avoid circular imports
        from spinlock.config.schema import SummaryFeaturesConfig as SchemaConfig

        if not isinstance(schema_config, SchemaConfig):
            raise TypeError(
                f"Expected SummaryFeaturesConfig, got {type(schema_config).__name__}"
            )

        # Convert learned config if present
        learned_config = LearnedSummaryConfig()
        if schema_config.learned is not None:
            learned_config = LearnedSummaryConfig(
                enabled=schema_config.learned.enabled,
                extract_from=schema_config.learned.extract_from,
                skip_levels=list(schema_config.learned.skip_levels),
                temporal_agg=schema_config.learned.temporal_agg,
                spatial_agg=schema_config.learned.spatial_agg,
                projection_dim=schema_config.learned.projection_dim,
                # Training config
                training_epochs=schema_config.learned.training_epochs,
                learning_rate=schema_config.learned.learning_rate,
                lr_scheduler=schema_config.learned.lr_scheduler,
                early_stopping_patience=schema_config.learned.early_stopping_patience,
            )

        return cls(
            summary_mode=schema_config.summary_mode,
            learned=learned_config,
        )

