"""
TEMPORAL feature family configuration.

Configuration schemas for per-timestep temporal feature categories:
- Spatial statistics (24D)
- Spectral/frequency features (27D)
- Cross-channel features (12D)
- Enhanced temporal dynamics (130D): instantaneous, local temporal, stability, phase space, multi-scale

Per-timestep-only architecture for online perturbation-based NOA.

Example:
    >>> from spinlock.features.temporal.config import TemporalFeatureConfig
    >>> config = TemporalFeatureConfig()  # All features enabled with defaults
    >>> config.spatial.include_mean
    True
"""

from typing import Literal, List, Optional
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Per-Timestep Feature Category Configurations
# =============================================================================

class SpatialConfig(BaseModel):
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


class SpectralConfig(BaseModel):
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


class TemporalConfig(BaseModel):
    """
    Enhanced temporal dynamics feature configuration.

    Per-timestep temporal features (130D total):
    - Instantaneous dynamics (22D): energy, dissipation, spectral, structure, stats
    - Local temporal (28D): autocorr, trends, windowed stats, oscillations, growth
    - Local stability (24D): Lipschitz, stability, divergence, regularity
    - Phase space geometry (26D): flow, vorticity, strain, topology, manifold
    - Multi-scale temporal (30D): hierarchical averaging, cross-scale, persistence

    Uses circular buffers for windowed features.
    """

    enabled: bool = True

    # Window sizes for local temporal context
    window_size: int = Field(default=5, ge=1, le=20)  # For local temporal features
    short_window: int = Field(default=5, ge=1, le=20)  # For multi-scale (instant to short)
    medium_window: int = Field(default=20, ge=5, le=100)  # For multi-scale (short to medium)
    long_window: int = Field(default=50, ge=10, le=200)  # For multi-scale (medium to long)

    @field_validator('window_size')
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        """Ensure window size is reasonable."""
        if v < 1 or v > 20:
            raise ValueError("window_size must be between 1 and 20")
        return v

    @field_validator('short_window')
    @classmethod
    def validate_short_window(cls, v: int) -> int:
        """Ensure short window is reasonable."""
        if v < 1 or v > 20:
            raise ValueError("short_window must be between 1 and 20")
        return v

    @field_validator('medium_window')
    @classmethod
    def validate_medium_window(cls, v: int) -> int:
        """Ensure medium window is reasonable."""
        if v < 5 or v > 100:
            raise ValueError("medium_window must be between 5 and 100")
        return v

    @field_validator('long_window')
    @classmethod
    def validate_long_window(cls, v: int) -> int:
        """Ensure long window is reasonable."""
        if v < 10 or v > 200:
            raise ValueError("long_window must be between 10 and 200")
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


class CrossChannelConfig(BaseModel):
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

    Extracts features from neural operator intermediate representations:

    For U-AFNO operators:
    - Bottleneck latents: Global spectral features after AFNO
    - Skip connections: Multi-scale encoder features

    For CNN operators:
    - Early: First conv block activations (local edges/gradients)
    - Mid: Middle block activations (mid-level patterns)
    - Pre-output: Final hidden state before output layer

    Aggregation pipeline:
    1. Temporal: Pool across T timesteps (mean, max, or concatenated)
    2. Spatial: Global average pooling across H, W
    3. Optional: Project to fixed dimension via MLP

    Example (U-AFNO):
        >>> config = LearnedSummaryConfig(
        ...     enabled=True,
        ...     extract_from="bottleneck",  # U-AFNO specific
        ...     temporal_agg="mean_max",
        ... )

    Example (CNN):
        >>> config = LearnedSummaryConfig(
        ...     enabled=True,
        ...     extract_from="all",  # Works for both
        ...     temporal_agg="mean_max",
        ... )
    """

    enabled: bool = Field(
        default=False,
        description="Enable learned feature extraction from operator latents"
    )

    extract_from: Literal["bottleneck", "skips", "all", "early", "mid", "pre_output"] = Field(
        default="all",
        description=(
            "Which latents to extract. "
            "U-AFNO: 'bottleneck', 'skips', or 'all'. "
            "CNN: 'early', 'mid', 'pre_output', or 'all'."
        )
    )

    # U-AFNO specific
    skip_levels: List[int] = Field(
        default_factory=lambda: [0, 1, 2],
        description="(U-AFNO) Which encoder levels to extract (0=shallowest)"
    )

    # CNN specific
    layer_indices: Optional[List[int]] = Field(
        default=None,
        description="(CNN) Which mid layer indices to extract. None = all mid layers."
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
# TEMPORAL Top-Level Configuration
# =============================================================================

class TemporalFeatureConfig(BaseModel):
    """
    TEMPORAL feature family configuration.

    Per-timestep-only architecture for online perturbation-based NOA.
    Total: 193D per-timestep features.

    Attributes:
        spatial: Spatial statistics configuration (24D)
        spectral: Spectral/frequency features configuration (27D)
        cross_channel: Cross-channel interaction configuration (12D)
        temporal: Enhanced temporal dynamics configuration (130D)
        per_channel: Extract features per-channel or aggregate across channels
    """

    # Per-timestep feature category configs
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    spectral: SpectralConfig = Field(default_factory=SpectralConfig)
    cross_channel: CrossChannelConfig = Field(default_factory=CrossChannelConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    structural: SummaryStructuralConfig = Field(default_factory=SummaryStructuralConfig)
    physics: SummaryPhysicsConfig = Field(default_factory=SummaryPhysicsConfig)
    morphological: SummaryMorphologicalConfig = Field(default_factory=SummaryMorphologicalConfig)
    multiscale: SummaryMultiscaleConfig = Field(default_factory=SummaryMultiscaleConfig)

    # Settings
    per_channel: bool = True  # Extract features per-channel or aggregate across channels

    def estimate_feature_count(self) -> int:
        """
        Estimate total number of per-timestep features.

        Returns:
            Estimated feature count (193D for standard config)
        """
        count = 0

        # Spatial features (~24D)
        if self.spatial.enabled:
            count += 24

        # Spectral features (~27D)
        if self.spectral.enabled:
            count += 27

        # Cross-channel features (~12D)
        if self.cross_channel is not None and self.cross_channel.enabled:
            count += 12

        # Enhanced temporal features (130D)
        if self.temporal is not None and self.temporal.enabled:
            count += 130

        return count

    @classmethod
    def from_schema_config(cls, schema_config: "TemporalFeaturesConfig") -> "TemporalFeatureConfig":
        """
        Create a TemporalFeatureConfig from a schema TemporalFeaturesConfig.

        Args:
            schema_config: TemporalFeaturesConfig from spinlock.config.schema

        Returns:
            TemporalFeatureConfig instance
        """
        # Import here to avoid circular imports
        from spinlock.config.schema import TemporalFeaturesConfig as SchemaConfig

        if not isinstance(schema_config, SchemaConfig):
            raise TypeError(
                f"Expected TemporalFeaturesConfig, got {type(schema_config).__name__}"
            )

        return cls()


# Legacy alias for backward compatibility
SummaryConfig = TemporalFeatureConfig
