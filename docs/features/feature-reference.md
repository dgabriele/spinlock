# SDF Feature Reference v2.0 + Phase 1/2 Extensions

Complete reference for all Summary Descriptor Features (SDF) computed by Spinlock.

**Total Features**: 221 (baseline 174 + Phase 1: 33 + Phase 2: 14)

## Table of Contents

### v1.0 Categories (Baseline)
1. [Spatial Statistics](#spatial-statistics) (26 baseline + 8 extensions = 34 features)
2. [Spectral Features](#spectral-features) (31 features)
3. [Temporal Dynamics](#temporal-dynamics) (13 baseline + 33 extensions = 44 features)

### v2.0 Categories (Operator-Aware Features)
4. [Operator Sensitivity](#operator-sensitivity) (12 features, trajectory-level)
5. [Cross-Channel Interactions](#cross-channel-interactions) (12 features, per-timestep)
6. [Causality/Directionality](#causalitydirectionality) (15 features, trajectory-level)
7. [Invariant Drift](#invariant-drift) (64 features, trajectory-level)
8. [Nonlinear Dynamics](#nonlinear-dynamics) (8 features, trajectory-level)

### Extensions
- **Phase 1** (High-Impact, 33 features): Percentiles, event counts, time-to-event, rolling windows, RQA, correlation dimension
- **Phase 2** (Research, 14 features): PACF, permutation entropy, histogram/occupancy

---

## Spatial Statistics

Computed per timestep from 2D spatial fields. All features averaged across channels and realizations.

### Distributional Moments

**`spatial_mean`**
- **Formula**: `μ = (1/HW) Σᵢⱼ u[i,j]`
- **Range**: Depends on input data scale
- **Interpretation**: Average field value; baseline intensity
- **Use**: Distinguishes high vs low activity patterns

**`spatial_variance`**
- **Formula**: `σ² = (1/HW) Σᵢⱼ (u[i,j] - μ)²`
- **Range**: 0 to ∞ (typically 0-10)
- **Interpretation**: Spread of field values
- **Use**: Identifies homogeneous vs heterogeneous patterns

**`spatial_std`**
- **Formula**: `σ = √variance`
- **Range**: 0 to ∞ (typically 0-3)
- **Interpretation**: Standard deviation; typical deviation from mean
- **Use**: Same scale as input data; easier to interpret than variance

**`spatial_skewness`**
- **Formula**: `γ₁ = E[(u - μ)³] / σ³`
- **Range**: -∞ to +∞ (typically -3 to +3)
- **Interpretation**: Asymmetry of distribution
  - γ₁ < 0: Left-skewed (tail on left)
  - γ₁ = 0: Symmetric
  - γ₁ > 0: Right-skewed (tail on right)
- **Use**: Detects asymmetric patterns (e.g., rare hot spots)

**`spatial_kurtosis`**
- **Formula**: `γ₂ = E[(u - μ)⁴] / σ⁴ - 3`
- **Range**: -3 to +∞ (typically -2 to +10)
- **Interpretation**: Tail heaviness
  - γ₂ < 0: Light tails (uniform-like)
  - γ₂ = 0: Normal tails (Gaussian)
  - γ₂ > 0: Heavy tails (outliers present)
- **Use**: Identifies extreme events, localized features

### Extrema and Spread

**`spatial_min`**
- **Formula**: `min(u)`
- **Range**: Depends on input data
- **Interpretation**: Minimum field value
- **Use**: Lower bound of pattern activity

**`spatial_max`**
- **Formula**: `max(u)`
- **Range**: Depends on input data
- **Interpretation**: Maximum field value
- **Use**: Peak intensity; identifies hot spots

**`spatial_range`**
- **Formula**: `range = max - min`
- **Range**: 0 to ∞
- **Interpretation**: Total dynamic range
- **Use**: Pattern contrast; distinguishes flat vs varied fields

**`spatial_iqr`**
- **Formula**: `IQR = Q₃ - Q₁` (75th - 25th percentile)
- **Range**: 0 to ∞
- **Interpretation**: Spread of middle 50% of values
- **Use**: Robust measure of spread (insensitive to outliers)

**`spatial_mad`**
- **Formula**: `MAD = median(|u - median(u)|)`
- **Range**: 0 to ∞
- **Interpretation**: Median absolute deviation
- **Use**: Robust measure of variability (alternative to std)

### Spatial Gradients

Computed using Sobel filters for numerical stability.

**`gradient_magnitude_mean`**
- **Formula**: `⟨|∇u|⟩` where `|∇u| = √(u_x² + u_y²)`
- **Range**: 0 to ∞ (typically 0-5)
- **Interpretation**: Average edge strength
- **Use**: Measures spatial smoothness; high = sharp features

**`gradient_magnitude_std`**
- **Formula**: `std(|∇u|)`
- **Range**: 0 to ∞
- **Interpretation**: Variability of edge strength
- **Use**: Distinguishes uniform vs localized gradients

**`gradient_magnitude_max`**
- **Formula**: `max(|∇u|)`
- **Range**: 0 to ∞
- **Interpretation**: Strongest edge in field
- **Use**: Detects sharpest transitions (e.g., shocks, interfaces)

**`gradient_x_mean`**, **`gradient_y_mean`**
- **Formula**: `⟨∂u/∂x⟩`, `⟨∂u/∂y⟩`
- **Range**: -∞ to +∞ (typically -1 to +1)
- **Interpretation**: Average directional derivatives
- **Use**: Detects directional bias (e.g., traveling waves)

**`gradient_anisotropy`**
- **Formula**: `|⟨u_x⟩ - ⟨u_y⟩| / (|⟨u_x⟩| + |⟨u_y⟩| + ε)`
- **Range**: 0 to 1
- **Interpretation**: Directional preference of gradients
  - 0: Isotropic (no preferred direction)
  - 1: Highly anisotropic (strong directional bias)
- **Use**: Identifies oriented patterns (stripes, waves)

### Curvature

Computed using Laplacian operator (∇²u).

**`laplacian_mean`**
- **Formula**: `⟨∇²u⟩` where `∇²u = ∂²u/∂x² + ∂²u/∂y²`
- **Range**: -∞ to +∞ (typically -1 to +1)
- **Interpretation**: Average curvature
  - > 0: Convex regions dominate
  - < 0: Concave regions dominate
- **Use**: Characterizes overall pattern curvature

**`laplacian_std`**
- **Formula**: `std(∇²u)`
- **Range**: 0 to ∞
- **Interpretation**: Variability of curvature
- **Use**: Detects localized curvature features

**`laplacian_energy`**
- **Formula**: `(1/HW) Σᵢⱼ (∇²u[i,j])²`
- **Range**: 0 to ∞ (typically 0-20)
- **Interpretation**: Energy in curvature field (normalized per pixel)
- **Use**: Measures pattern complexity; high = many small-scale features

### Phase 1 Extension: Percentiles (5 features)

Capture distribution shape beyond mean/variance:

**`percentile_5`**, **`percentile_25`**, **`percentile_50`**, **`percentile_75`**, **`percentile_95`**
- **Formula**: `Q_p = value at pth percentile`
- **Range**: Depends on input data (same scale as field values)
- **Interpretation**: Distribution quantiles
  - Q₅, Q₉₅: Extreme tails (outlier boundaries)
  - Q₂₅, Q₇₅: Interquartile boundaries
  - Q₅₀: Median (robust central tendency)
- **Use**:
  - Distribution shape characterization (skewness via Q₂₅-Q₅₀-Q₇₅ spacing)
  - Robust outlier detection (Q₅/Q₉₅ vs min/max)
  - Heavy-tailed vs light-tailed patterns
- **Advantage**: More comprehensive than IQR alone; captures tail behavior

### Phase 2 Extension: Histogram/Occupancy (3 features)

Characterize state space coverage and occupancy patterns:

**`histogram_entropy`**
- **Formula**: `H = -Σᵢ pᵢ log(pᵢ)` where `pᵢ = count in bin i / total count`
- **Range**: 0 to log(num_bins) (default: 0 to log(16) ≈ 2.77)
- **Interpretation**: Uniformity of state space coverage
  - 0: All values in single bin (delta distribution)
  - log(num_bins): Perfectly uniform across bins
- **Use**: Distinguishes unimodal vs multimodal distributions

**`histogram_peak_fraction`**
- **Formula**: `max(pᵢ)` - fraction in most populated bin
- **Range**: 0 to 1
- **Interpretation**: Concentration in dominant mode
  - High (>0.5): Strongly concentrated (narrow distribution)
  - Low (<0.1): Dispersed (broad distribution)
- **Use**: Detects dominant states vs diffuse patterns

**`histogram_effective_bins`**
- **Formula**: Count of bins with > 1% of mass
- **Range**: 1 to num_bins (default: 1 to 16)
- **Interpretation**: Effective support size
  - Low (1-3): Few distinct states
  - High (10-16): Many distinct states
- **Use**: Quantifies state space utilization

**Note**: Histogram features use 16 bins by default (configurable)

---

## Spectral Features

Computed from 2D Fast Fourier Transform (FFT) of spatial fields. Uses orthonormal FFT for grid-size independence.

### FFT Power Spectrum (Multiscale)

Frequency space divided into 5 radial bands (sqrt-spaced for balanced coverage):

**Frequency Bands**:
- **Scale 0**: Low frequencies (0.5 - 10.6 cycles)
- **Scale 1**: Low-mid frequencies (10.6 - 33.7 cycles)
- **Scale 2**: Mid frequencies (33.7 - 69.8 cycles)
- **Scale 3**: Mid-high frequencies (69.8 - 118.9 cycles)
- **Scale 4**: High frequencies (118.9 - 181 cycles)

For each scale, three statistics:

**`fft_power_scale_X_mean`**
- **Formula**: `(1/N_pixels) Σ_{k∈band} |FFT(u)[k]|²`
- **Range**: 0 to ∞ (low freq: 0-500, high freq: 0-1)
- **Interpretation**: Average power in frequency band
- **Use**: Energy distribution across scales
  - High scale 0: DC/low-frequency dominance
  - High scale 4: High-frequency noise

**`fft_power_scale_X_max`**
- **Formula**: `max_{k∈band} |FFT(u)[k]|²`
- **Range**: 0 to ∞
- **Interpretation**: Peak power in band
- **Use**: Detects strong periodic components

**`fft_power_scale_X_std`**
- **Formula**: `std(|FFT(u)[k]|²)` for k in band
- **Range**: 0 to ∞
- **Interpretation**: Variability of power within band
- **Use**: Distinguishes broadband vs narrowband signals

### Dominant Frequency

**`dominant_freq_x`**, **`dominant_freq_y`**
- **Formula**: Frequency coordinates (kₓ, kᵧ) of max power
- **Range**: 0 to 0.5 (normalized by grid size)
- **Interpretation**: Peak frequency location
- **Use**: Detects dominant spatial wavelength and orientation

**`dominant_freq_magnitude`**
- **Formula**: `|FFT(u)[k_max]|²` at peak
- **Range**: 10 to 10,000+ (can be large for periodic data)
- **Interpretation**: Strength of dominant frequency
- **Use**: Measures periodicity strength
  - High: Strong periodic pattern
  - Low: Noisy/aperiodic pattern

### Spectral Centroids

**`spectral_centroid_x`**, **`spectral_centroid_y`**
- **Formula**: `Σ_k k * P(k) / Σ_k P(k)` where P = power
- **Range**: 0 to 0.5
- **Interpretation**: Center of mass of power spectrum
- **Use**: Average frequency content
  - Low: Smooth patterns
  - High: Fine-scale features

**`spectral_bandwidth`**
- **Formula**: `√(Σ_k (k - centroid)² * P(k) / Σ_k P(k))`
- **Range**: 0 to 0.5
- **Interpretation**: Spread of frequencies
- **Use**: Spectral diversity
  - Low: Narrowband (few frequencies)
  - High: Broadband (many frequencies)

### Frequency Ratios

Energy distribution across frequency ranges:

**`low_freq_ratio`**
- **Formula**: `E_low / E_total` (frequencies < 33% Nyquist)
- **Range**: 0 to 1
- **Interpretation**: Fraction of energy in low frequencies
- **Use**: Smooth vs detailed patterns

**`mid_freq_ratio`**
- **Formula**: `E_mid / E_total` (33% - 66% Nyquist)
- **Range**: 0 to 1
- **Interpretation**: Fraction of energy in mid frequencies

**`high_freq_ratio`**
- **Formula**: `E_high / E_total` (> 66% Nyquist)
- **Range**: 0 to 1
- **Interpretation**: Fraction of energy in high frequencies
- **Use**: Noise content; fine-scale features

**Note**: Ratios sum to 1.

### Spectral Shape

**`spectral_flatness`**
- **Formula**: `(geometric_mean(P) / arithmetic_mean(P))`
- **Range**: 0 to 1
- **Interpretation**: Flatness of power spectrum
  - 0: Tonal (few strong peaks)
  - 1: Noise-like (flat spectrum)
- **Use**: Distinguishes structured vs random patterns

**`spectral_rolloff`**
- **Formula**: Frequency below which 85% of energy lies
- **Range**: 0 to 0.5
- **Interpretation**: High-frequency cutoff
- **Use**: Effective bandwidth of signal

**`spectral_anisotropy`**
- **Formula**: Ratio of power along principal axes
- **Range**: 0 to 1
- **Interpretation**: Directional preference in frequency space
  - 0: Isotropic (radially symmetric)
  - 1: Anisotropic (directional features)
- **Use**: Detects oriented patterns (waves, stripes)

---

## Temporal Dynamics

Computed from temporal evolution of trajectories. **Requires T > 1 timesteps.**

### Growth and Decay

**`energy_growth_rate`**
- **Formula**: `d(log E)/dt` where `E = ⟨u²⟩`
- **Range**: -10 to +10
- **Interpretation**: Exponential growth/decay rate
  - > 0: Growing energy
  - < 0: Decaying energy
  - ≈ 0: Stationary
- **Use**: Identifies unstable/stable dynamics

**`energy_growth_accel`**
- **Formula**: `d²(log E)/dt²`
- **Range**: -10 to +10
- **Interpretation**: Acceleration of growth
- **Use**: Detects transitions (e.g., onset of instability)

**`variance_growth_rate`**
- **Formula**: `d(log σ²)/dt`
- **Range**: -10 to +10
- **Interpretation**: Growth of spatial variability
- **Use**: Pattern formation dynamics

### Oscillations

**`temporal_freq_dominant`**
- **Formula**: Peak frequency in temporal power spectrum
- **Range**: 0 to Nyquist frequency
- **Interpretation**: Dominant oscillation frequency
- **Use**: Identifies periodic behavior

**`oscillation_amplitude`**
- **Formula**: Amplitude of dominant oscillation
- **Range**: 0 to ∞
- **Interpretation**: Strength of periodic component
- **Use**: Distinguishes oscillatory vs non-oscillatory dynamics

**`oscillation_period`**
- **Formula**: `1 / temporal_freq_dominant`
- **Range**: 2Δt to ∞
- **Interpretation**: Period of oscillation
- **Use**: Characteristic timescale

**`autocorr_decay_time`**
- **Formula**: Time to autocorrelation < 1/e
- **Range**: Δt to T
- **Interpretation**: Memory timescale
- **Use**: Decorrelation time; mixing rate

### Stability

**`lyapunov_approx`**
- **Formula**: `log(|δu(t)| / |δu(0)|) / t` (finite-time approximation)
- **Range**: -∞ to +∞ (typically -5 to +5)
- **Interpretation**: Exponential divergence rate
  - > 0: Chaotic/unstable
  - < 0: Stable
- **Use**: Measures sensitivity to perturbations

**`trajectory_smoothness`**
- **Formula**: `1 / ⟨|du/dt|⟩`
- **Range**: 0 to ∞
- **Interpretation**: Inverse of temporal variation
- **Use**: Smooth vs jerky dynamics

**`regime_switches`**
- **Formula**: Number of detected regime changes
- **Range**: 0 to T-1
- **Interpretation**: Count of qualitative transitions
- **Use**: Multistable or switching dynamics

### Stationarity

**`final_to_initial_ratio`**
- **Formula**: `E(T) / E(0)` (energy ratio)
- **Range**: 0 to ∞
- **Interpretation**: Net energy change
  - > 1: Energy increased
  - < 1: Energy decreased
- **Use**: Transient vs stationary behavior

**`trend_strength`**
- **Formula**: Coefficient of linear trend fit
- **Range**: -∞ to +∞
- **Interpretation**: Strength of monotonic trend
- **Use**: Distinguishes trending vs stationary

**`detrended_variance`**
- **Formula**: Variance after removing linear trend
- **Range**: 0 to ∞
- **Interpretation**: Fluctuations around trend
- **Use**: Variability after accounting for drift

### Phase 1 Extension: Event Counts (3 features)

Quantify extreme events and threshold crossings:

**`event_count_spikes`**
- **Formula**: Count of timesteps where `|u(t) - mean| > k * std` (default: k=2.0)
- **Range**: 0 to T (number of timesteps)
- **Interpretation**: Number of outlier events
- **Use**: Detects rare large-amplitude fluctuations

**`event_count_bursts`**
- **Formula**: Count of sustained periods (≥3 consecutive timesteps) above threshold
- **Range**: 0 to T/3
- **Interpretation**: Number of sustained extreme events
- **Use**: Distinguishes transient spikes from prolonged excursions

**`event_count_zero_crossings`**
- **Formula**: Count of sign changes in `u(t) - mean`
- **Range**: 0 to T-1
- **Interpretation**: Number of oscillation cycles (× 2)
- **Use**: Detects oscillatory behavior; low = monotonic, high = high-frequency oscillations

### Phase 1 Extension: Time-to-Event (2 features)

Measure time until critical transitions:

**`time_to_event_0.5x`**, **`time_to_event_2.0x`**
- **Formula**: First timestep where `|u(t)| crosses threshold * u(0)|`
  - 0.5x: Time to 50% decrease (decay)
  - 2.0x: Time to 200% increase (growth)
- **Range**: 1 to T (or T if never crossed)
- **Interpretation**: Characteristic timescale for transitions
  - Early crossing: Fast dynamics
  - Late/no crossing: Slow or stationary dynamics
- **Use**:
  - Classify decay vs growth operators
  - Measure transient vs asymptotic timescales
  - Detect bifurcations (threshold never crossed)

### Phase 1 Extension: Rolling Windows (18 features)

Multi-timescale analysis across different window sizes:

**Window Sizes**: 5%, 10%, 20% of total trajectory length T
**Statistics per Window**: mean, std, max, min, range, peak_time

**`rolling_mean_w5`**, **`rolling_mean_w10`**, **`rolling_mean_w20`**
- **Formula**: Mean of trajectory within sliding window
- **Range**: Same as field values
- **Interpretation**: Average behavior at different timescales
- **Use**: Detects transient vs sustained dynamics

**`rolling_std_w5`**, **`rolling_std_w10`**, **`rolling_std_w20`**
- **Formula**: Standard deviation within sliding window
- **Range**: 0 to ∞
- **Interpretation**: Variability at different timescales
- **Use**: Identifies bursty vs smooth dynamics

**`rolling_max_w5`**, **`rolling_max_w10`**, **`rolling_max_w20`**
- **Formula**: Maximum value within sliding window
- **Range**: Depends on field values
- **Interpretation**: Peak intensity at different timescales
- **Use**: Detects extreme events at multiple scales

**`rolling_min_w5`**, **`rolling_min_w10`**, **`rolling_min_w20`**
- **Formula**: Minimum value within sliding window
- **Range**: Depends on field values
- **Interpretation**: Minimum intensity at different timescales

**`rolling_range_w5`**, **`rolling_range_w10`**, **`rolling_range_w20`**
- **Formula**: `rolling_max - rolling_min` within window
- **Range**: 0 to ∞
- **Interpretation**: Dynamic range at different timescales
- **Use**: Measures volatility/variability at multiple scales

**`rolling_peak_time_w5`**, **`rolling_peak_time_w10`**, **`rolling_peak_time_w20`**
- **Formula**: Timestep of maximum `rolling_mean` value
- **Range**: 0 to T-window_size
- **Interpretation**: When peak activity occurs
- **Use**: Detects early vs late vs sustained peaks

**Critical Use Case**: Distinguishes transient (early peak in w5) from sustained (late peak in w20) dynamics

### Phase 2 Extension: PACF (10 features)

Partial autocorrelation function isolates direct temporal correlations:

**`pacf_lag_1`** through **`pacf_lag_10`**
- **Formula**: PACF(k) via Yule-Walker approximation (Levinson-Durbin recursion)
- **Range**: -1 to +1 (like correlation)
- **Interpretation**: Direct correlation at lag k (removing intermediate effects)
  - PACF(1): Same as ACF(1) (direct lag-1 correlation)
  - PACF(k): Correlation at lag k after removing lags 1..(k-1)
- **Use**:
  - Model order selection (PACF cutoff indicates AR order)
  - Distinguish AR vs MA processes
  - Detect periodic patterns vs exponential decay
- **Example**:
  - AR(1): PACF(1) ≠ 0, PACF(k>1) ≈ 0 (exponential decay)
  - AR(2): PACF(1), PACF(2) ≠ 0, PACF(k>2) ≈ 0 (oscillations)
  - White noise: All PACF ≈ 0

**Note**: PACF complements autocorrelation decay time for richer temporal characterization

---

## Nonlinear Dynamics

**Category:** Trajectory-level (requires T>1)
**Feature Count:** 8 (Phase 1: 5, Phase 2: 1, reserved: 2)
**Purpose:** Detect complex dynamics, chaos, and hidden temporal patterns

**Note**: All nonlinear features use temporal subsampling (default: factor of 10) to reduce O(T²) computational cost.

### Phase 1: Recurrence Quantification Analysis (4 features)

Analyze recurrence plots to detect hidden periodicities and structures:

**`rqa_recurrence_rate`**
- **Formula**: `RR = (1/N²) Σᵢⱼ R[i,j]` where R is recurrence matrix
- **Range**: 0 to 1
- **Interpretation**: Fraction of recurrent points in phase space
  - Low (<0.01): Non-recurrent, chaotic
  - Medium (0.01-0.1): Weakly recurrent
  - High (>0.1): Strongly recurrent, periodic
- **Use**: Measures phase space recurrence density

**`rqa_determinism`**
- **Formula**: `DET = Σ(l≥l_min) l*P(l) / Σ(l≥1) l*P(l)` where P(l) = diagonal line length distribution
- **Range**: 0 to 1
- **Interpretation**: Fraction of recurrence points in diagonal structures
  - High: Deterministic dynamics
  - Low: Stochastic/chaotic dynamics
- **Use**: Distinguishes deterministic vs random processes

**`rqa_laminarity`**
- **Formula**: `LAM = Σ(v≥v_min) v*P(v) / Σ(v≥1) v*P(v)` where P(v) = vertical line length distribution
- **Range**: 0 to 1
- **Interpretation**: Fraction of recurrence points in vertical structures
  - High: Laminar states (trapping regions)
  - Low: No trapping
- **Use**: Detects intermittency, trapping in attractors

**`rqa_entropy`**
- **Formula**: `ENTR = -Σ p(l) log(p(l))` where p(l) = normalized diagonal line distribution
- **Range**: 0 to ∞ (typically 0-5)
- **Interpretation**: Shannon entropy of diagonal line lengths
  - High: Complex recurrence structure
  - Low: Simple recurrence structure
- **Use**: Quantifies recurrence pattern complexity

### Phase 1: Correlation Dimension (1 feature)

Estimate attractor complexity via Grassberger-Procaccia algorithm:

**`correlation_dimension`**
- **Formula**: `D₂ ≈ ∂log(C(r))/∂log(r)` where C(r) = correlation integral
- **Range**: 0 to embedding_dim (typically 0.5-5)
- **Interpretation**: Fractal dimension of attractor
  - D₂ ≈ 0: Fixed point
  - D₂ ≈ 1: Limit cycle
  - D₂ ≈ 2-3: Torus, strange attractor
  - D₂ ≈ embedding_dim: High-dimensional chaos or noise
- **Use**: Characterizes attractor complexity
- **Examples**:
  - Lorenz attractor: D₂ ≈ 2.05
  - Rössler attractor: D₂ ≈ 1.99
  - White noise: D₂ → embedding_dim

**Computational Note**: Uses phase space embedding with τ=1, dim=5, subsampled by factor of 10

### Phase 2: Permutation Entropy (1 feature)

Measure ordinal pattern complexity:

**`permutation_entropy`**
- **Formula**: `H_p = -Σ p(π) log(p(π))` normalized by log(d!) where π = ordinal patterns of length d
- **Range**: 0 to 1 (normalized)
- **Interpretation**: Complexity of ordinal patterns
  - 0: Perfectly regular (constant or monotonic)
  - ~0.5: Partially predictable
  - ~1: Random/chaotic (all patterns equally likely)
- **Use**: Robust entropy measure for noisy data
- **Advantages**:
  - Robust to amplitude scaling
  - Captures temporal ordering structure
  - Less sensitive to noise than Shannon entropy
- **Example**:
  - Sine wave: H_p ≈ 0.2-0.4 (regular pattern)
  - Chaotic time series: H_p ≈ 0.8-1.0
  - White noise: H_p ≈ 1.0

**Parameters**: embedding_dim=3 (default), τ=1, subsample_factor=10

---

## Operator Sensitivity

**Category:** Trajectory-level (requires operator access during extraction)
**Feature Count:** 12 features
**Purpose:** Characterize neural operator input-output response

### Realization Aggregation

Temporal features (13) aggregated across M realizations with 3 methods:

**`mean`**: Average value
- **Interpretation**: Expected behavior
- **Use**: Typical trajectory

**`std`**: Standard deviation
- **Interpretation**: Stochastic variability
- **Use**: Uncertainty quantification

**`cv`**: Coefficient of variation (std / |mean|)
- **Interpretation**: Relative variability
- **Use**: Normalized uncertainty (dimensionless)

**Result**: 13 × 3 = 39 aggregated features

---

## Feature Selection Guide

### For VQ-VAE Training

**Recommended features** (compact, informative):

**Spatial** (10 features):
- `spatial_mean`, `spatial_std`, `spatial_skewness`
- `gradient_magnitude_mean`, `gradient_magnitude_std`
- `laplacian_energy`
- `spatial_min`, `spatial_max`
- `gradient_anisotropy`
- `spatial_iqr`

**Spectral** (8 features):
- `fft_power_scale_0_mean` (DC/low freq)
- `fft_power_scale_2_mean` (mid freq)
- `fft_power_scale_4_mean` (high freq)
- `dominant_freq_magnitude`
- `spectral_centroid_x`, `spectral_centroid_y`
- `low_freq_ratio`, `high_freq_ratio`

**Temporal** (6 features, if T>1):
- `energy_growth_rate_mean`, `energy_growth_rate_std`
- `temporal_freq_dominant_mean`
- `lyapunov_approx_mean`
- `autocorr_decay_time_mean`
- `final_to_initial_ratio_mean`

**Total**: 24 features (or 18 if T=1)

### For Pattern Classification

Focus on **spatial and spectral shape**:
- Moments: mean, std, skewness, kurtosis
- FFT power distribution across scales
- Spectral centroids and bandwidth
- Gradient anisotropy
- Spectral anisotropy

### For Dynamics Analysis

Focus on **temporal features** (requires T>1):
- All growth rate features
- Oscillation features
- Stability metrics
- Stationarity measures

---

## Operator Sensitivity

**Category:** Trajectory-level (requires operator access during extraction)
**Feature Count:** 10 (default config)
**Purpose:** Characterize neural operator input-output response

### Lipschitz Estimates (3 features)

Measure local sensitivity to input perturbations:

**`lipschitz_eps_1e-4`, `lipschitz_eps_1e-3`, `lipschitz_eps_1e-2`**
- **Formula**: `L ≈ ||O(x + δ) - O(x)|| / ||δ||` where `δ ~ N(0, ε²)`
- **Range**: 0 to ∞ (typically 0.1-10)
- **Interpretation**: Output sensitivity to small noise
- **Use**: Identifies stable vs chaotic operators

### Gain Curves (4 features)

Response to input amplitude scaling:

**`gain_scale_0.50`, `gain_scale_0.75`, `gain_scale_1.25`, `gain_scale_1.50`**
- **Formula**: `Gain(α) = ||O(α·x)|| / ||O(x)||`
- **Range**: 0 to ∞ (typically 0.3-2.0)
- **Interpretation**: Output energy ratio at scaled inputs
- **Use**: Detects linearity, saturation, amplification

### Linearity Metrics (3 features)

**`linearity_r2`**
- **Formula**: R² of linear fit to gain curve
- **Range**: 0 to 1
- **Interpretation**: 1.0 = perfectly linear operator
- **Use**: Quantifies deviation from linearity

**`saturation_degree`**
- **Formula**: `expected_gain - actual_gain` at max scale
- **Range**: -∞ to ∞ (typically -0.5 to +0.5)
- **Interpretation**: Positive = compression, Negative = amplification
- **Use**: Detects compressive/expansive nonlinearity

**`compression_ratio`**
- **Formula**: `gain(0.5) / gain(1.5)`
- **Range**: 0 to ∞ (typically 0.5-2.0)
- **Interpretation**: >1 = compressive, <1 = expansive
- **Use**: Summarizes nonlinearity type

---

## Cross-Channel Interactions

**Category:** Per-timestep
**Feature Count:** 10 (default config)
**Purpose:** Measure channel coupling structure

### Correlation Eigendecomposition (6 features)

**`cross_channel_eigen_top_1`, `cross_channel_eigen_top_2`, `cross_channel_eigen_top_3`**
- **Formula**: Top 3 eigenvalues of correlation matrix `R[i,j] = corr(channel_i, channel_j)`
- **Range**: 0 to C (number of channels)
- **Interpretation**: Magnitude of principal coupling modes
- **Use**: Effective dimensionality, dominant interactions

**`cross_channel_eigen_trace`**
- **Formula**: `Σλᵢ = C` (sum of eigenvalues)
- **Range**: Equals number of channels
- **Interpretation**: Total correlation structure
- **Use**: Normalization check

**`cross_channel_condition_number`**
- **Formula**: `λ_max / λ_min`
- **Range**: 1 to ∞
- **Interpretation**: Matrix conditioning; >1000 = ill-conditioned
- **Use**: Detects redundant vs independent channels

**`cross_channel_participation_ratio`**
- **Formula**: `(Σλᵢ)² / Σλᵢ²` (effective dimensionality)
- **Range**: 1 to C
- **Interpretation**: Number of "active" coupling modes
- **Use**: Dimensionality reduction indicator

### Pairwise Correlation Statistics (4 features)

**`cross_channel_corr_mean`, `cross_channel_corr_max`, `cross_channel_corr_min`, `cross_channel_corr_std`**
- **Formula**: Statistics over all pairwise correlations
- **Range**: -1 to +1 (correlations)
- **Interpretation**: Summary of channel coupling strength
- **Use**: Quick coupling assessment without eigenanalysis

---

## Causality/Directionality

**Category:** Trajectory-level (requires T > 1)
**Feature Count:** 14 (default fast mode)
**Purpose:** Detect temporal information flow and directional asymmetry

### Lagged Correlation Asymmetry (6 features)

**`causality_lag_corr_asymmetry_mean_lag1/2/3`, `causality_lag_corr_asymmetry_max_lag1/2/3`**
- **Formula**: `asymmetry(τ) = corr(x(t), y(t+τ)) - corr(x(t+τ), y(t))`
- **Range**: -2 to +2 (correlation difference)
- **Interpretation**: Directional preference in coupling
  - Positive: x → y causality dominant
  - Negative: y → x causality dominant
  - Zero: Symmetric coupling
- **Use**: Fast causality screening

### Prediction Error Ratios (4 features)

**`causality_pred_error_ratio_lag1/2`, `causality_pred_error_diff_lag1/2`**
- **Formula**: `ratio = error_forward / error_backward`
- **Range**: 0 to ∞ (typically 0.5-2.0)
- **Interpretation**: Asymmetry in predictive power
- **Use**: Identifies predictable vs unpredictable directions

### Time Irreversibility (2 features)

**`causality_time_irreversibility`**
- **Formula**: `<(u(t+1)-u(t))³> - <(u(t)-u(t-1))³>` (third-order moment asymmetry)
- **Range**: -∞ to ∞ (typically -0.5 to +0.5)
- **Interpretation**: Time-forward vs time-backward asymmetry
- **Use**: Detects arrow of time, non-equilibrium dynamics

**`causality_time_asymmetry_index`**
- **Formula**: Normalized irreversibility measure
- **Range**: 0 to 1
- **Interpretation**: 0 = reversible, 1 = strongly irreversible
- **Use**: Comparable across operators

### Spatial Information Flow (2 features)

**`causality_spatial_flow_magnitude`**
- **Formula**: Information propagation strength from gradient correlations
- **Range**: 0 to ∞
- **Interpretation**: Spatial information transport rate
- **Use**: Detects diffusion-like vs wave-like propagation

**`causality_spatial_flow_anisotropy`**
- **Formula**: Directional preference in spatial flow
- **Range**: 0 to 1 (0 = isotropic, 1 = highly anisotropic)
- **Interpretation**: Preferred propagation directions
- **Use**: Identifies axial symmetry breaking

---

## Invariant Drift

**Category:** Trajectory-level
**Feature Count:** 60 (default: 5 norms × 4 metrics × 3 scales)
**Purpose:** Track generic norm-based stability without assuming physical laws

### Design Philosophy

**Critical Principle:** Do NOT assume physical conservation laws (mass, energy) unless operator explicitly has them. Instead, measure **generic norm drift** as a latent operator property.

**Multi-scale Analysis:** Compute on 3 field versions:
- **Raw**: Original field
- **Low-pass**: Gaussian-filtered (σ=2 pixels) - smooth modes
- **High-pass**: `raw - low_pass` - fine-scale features

### Generic Norms (5 × 4 × 3 = 60 features)

**Norms Tracked:**
1. **L1**: `∫|u| dx` (total absolute magnitude)
2. **L2**: `√(∫u² dx)` (energy-like)
3. **L∞**: `max|u|` (peak magnitude)
4. **Entropy**: `-∫p log p dx` (distributional complexity)
5. **Total Variation (TV)**: `∫|∇u| dx` (edge content)

**Metrics per Norm:**

**`{norm}_mean_drift_{scale}`** (e.g., `L2_mean_drift_raw`)
- **Formula**: `(1/T) Σ(I(t+1) - I(t))` where I is the invariant
- **Range**: -∞ to ∞
- **Interpretation**: Average rate of change
  - Positive: Growth/amplification
  - Negative: Decay/dissipation
  - Zero: Stationary/conserved
- **Use**: Classifies operator stability

**`{norm}_drift_variance_{scale}`**
- **Formula**: `var(I(t+1) - I(t))`
- **Range**: 0 to ∞
- **Interpretation**: Consistency of drift
  - Low: Monotonic behavior
  - High: Oscillatory/chaotic drift
- **Use**: Distinguishes smooth vs noisy evolution

**`{norm}_final_initial_ratio_{scale}`**
- **Formula**: `I(T) / I(0)`
- **Range**: 0 to ∞ (typically 0.1-10)
- **Interpretation**: Total change factor
  - >1: Net growth
  - <1: Net decay
  - =1: Conserved
- **Use**: Long-term stability assessment

**`{norm}_monotonicity_{scale}`**
- **Formula**: Sign consistency of `I(t+1) - I(t)`
- **Range**: 0 to 1
- **Interpretation**: Fraction of timesteps with consistent sign
  - 1.0: Perfectly monotonic
  - 0.5: Random fluctuations
  - 0.0: Perfect oscillation
- **Use**: Detects oscillatory vs monotonic dynamics

### Example Features

- `L1_mean_drift_raw`: Average L1 norm change (raw field)
- `L2_drift_variance_lowpass`: L2 energy variance (smooth modes only)
- `entropy_final_initial_ratio_highpass`: Entropy change (fine-scale features)
- `tv_monotonicity_raw`: Edge content monotonicity

### Interpretation Guide

**Stable operators:** Low drift variance, monotonicity near 1.0, ratio near 1.0
**Dissipative operators:** Negative mean drift, ratio < 1.0
**Unstable operators:** Positive mean drift, high variance, ratio > 1.0
**Oscillatory operators:** High variance, low monotonicity, ratio near 1.0
**Scale-selective:** Different drift patterns across raw/lowpass/highpass

---

## Mathematical Notation

- `u`: Spatial field [H, W]
- `H, W`: Grid dimensions
- `⟨·⟩`: Spatial average
- `E[·]`: Expected value
- `σ`: Standard deviation
- `∇u`: Gradient (∂u/∂x, ∂u/∂y)
- `∇²u`: Laplacian
- `FFT(u)`: 2D Fast Fourier Transform (orthonormal)
- `P(k)`: Power spectrum at frequency k
- `E`: Energy = ⟨u²⟩
- `Δt`: Timestep
- `T`: Total time
- `M`: Number of realizations
