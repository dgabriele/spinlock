# Feature Reference: TEMPORAL Features (v3.0)

**Last Updated:** 2026-01-18

Complete reference for TEMPORAL feature family computed by Spinlock v3.0.

**TEMPORAL Features**: ~328D per-timestep behavioral features (online-computable working memory)

## v3.0 Breaking Changes

**SUMMARY features removed** - Aggregated trajectory-level features (causality, invariant drift, operator sensitivity, nonlinear dynamics) were removed in v3.0.0 because they require complete trajectories and are incompatible with per-timestep online prediction in the Neural Operator Apprentice (NOA).

**Archived:** v2.x SUMMARY feature documentation and implementation available at `src/spinlock/features/temporal_old_v2/summary/`

**Migration:** Users needing trajectory-level aggregates can compute them post-hoc from TEMPORAL features or use v2.x datasets.

## Cognitive Interpretation (v3.0)

TEMPORAL features provide per-timestep behavioral representations:
- **Per-timestep computable**: All features can be computed at each timestep without seeing future trajectory (online operation)
- **Working memory**: Sequential representations preserving full temporal resolution for studying capacity constraints and attention mechanisms
- **Multi-modal**: ~328D feature space captures spatial, spectral, cross-channel, and temporal dynamics simultaneously

The enhanced v3.0 TEMPORAL features enable studying online prediction, real-time decision-making, and how systems maintain rich behavioral representations without trajectory lookahead.

## Table of Contents

### v3.0 TEMPORAL Features
1. [TEMPORAL Features](#temporal-features-time-series-family) (~328D per-timestep)
   - Spatial Features (~105D)
   - Spectral Features (~93D)
   - Cross-Channel Features (~10D)
   - Enhanced Temporal Dynamics (~120D)

### Legacy Documentation (v2.x)
- **SUMMARY features removed in v3.0** - See archived documentation in v2.x codebase
- Categories removed: Spatial Statistics, Spectral Features, Temporal Dynamics (aggregated), Operator Sensitivity, Causality/Directionality, Invariant Drift, Nonlinear Dynamics
- Archived implementation: `src/spinlock/features/temporal_old_v2/summary/`

---


## Legacy SUMMARY Features (v2.x) - REMOVED

**Note:** All content below this section in v2.x documentation described SUMMARY features that are **no longer present in v3.0+**.

### Why SUMMARY Features Were Removed

SUMMARY features required aggregating complete trajectories to compute features like:
- **Spatial Statistics**: Trajectory-level mean, variance, gradients, etc.
- **Spectral Features**: Aggregated FFT power, dominant frequencies
- **Temporal Dynamics**: Autocorrelation, stationarity, regime transitions
- **Operator Sensitivity**: Response to parameter perturbations (requires full trajectory)
- **Cross-Channel Interactions**: Trajectory-averaged correlations
- **Causality/Directionality**: Granger causality, transfer entropy (requires full trajectory)
- **Invariant Drift**: Long-term conservation law violations (requires start-to-end comparison)
- **Nonlinear Dynamics**: Lyapunov exponents, attractor dimensions (requires full trajectory)

These features are incompatible with **per-timestep online prediction** in the Neural Operator Apprentice (NOA), which must make predictions at each timestep without seeing the future.

### Migration Path

1. **Use v3.0 TEMPORAL features** - Enhanced from 63D to ~328D per-timestep, capturing most behavioral information in an online-computable form
2. **Compute post-hoc aggregates** - If you need trajectory-level statistics, compute them from TEMPORAL features after rollout completion
3. **Use v2.x datasets** - If your research requires SUMMARY features, use v2.x datasets (not compatible with v3.0 models)

### Archived Implementation

v2.x SUMMARY feature extraction code and documentation preserved at:
- Code: `src/spinlock/features/temporal_old_v2/summary/`
- Documentation: v2.x branch of repository

---

## TEMPORAL Features (Time Series Family)

**Category:** Per-timestep time series (v3.0: online-computable)
**Feature Count:** ~328D per timestep (enhanced from 63D in v2.x)
**Shape:** `[N, T, D_temporal]` - Full temporal resolution preserved
**Purpose:** Online per-timestep features for VQ-VAE tokenization and NOA prediction

**v3.0 Enhancement:** TEMPORAL features expanded from 63D to ~328D per-timestep by adding:
- Multi-scale spatial features (histogram-based distributions)
- Enhanced spectral analysis (frequency band decomposition)
- Windowed temporal dynamics (velocity, acceleration, stability)
- Phase space reconstruction features

All features are **per-timestep computable** (no trajectory lookahead required), enabling online NOA operation.

### Feature Categories

#### 1. Spatial Features (~105D)

**Per-Channel Statistics:**
- Moments: `mean`, `variance`, `std`, `skewness`, `kurtosis` (per channel)
- Extrema: `min`, `max`, `range`, `median`, `quartiles`
- Spread: `MAD`, `IQR`, percentiles (`p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99`)
- Entropy: histogram-based Shannon entropy

**Spatial Gradients:**
- Gradient magnitude statistics: `gradient_mean`, `gradient_std`, `gradient_max`
- Laplacian statistics: `laplacian_mean`, `laplacian_std`, `laplacian_max`
- Directional gradients: `gradient_x_mean`, `gradient_y_mean`
- Anisotropy: `gradient_anisotropy`

**Histogram Features (v3.0 NEW):**
- Multi-bin histogram distributions (10-20 bins)
- Distribution shape metrics
- Multi-scale histogram features (2-4 scales)
- Adaptive binning statistics

**Pattern Metrics:**
- Spatial autocorrelation (Moran's I)
- Isotropy/anisotropy measures
- Spatial coherence length
- Clustering metrics

#### 2. Spectral Features (~93D)

**Power Spectrum (v3.0 ENHANCED):**
- FFT power statistics: `fft_power_mean`, `fft_power_std`, `fft_power_max`
- Multi-scale FFT features (2-4 scales): `fft_power_scale_{i}_{stat}`
- Frequency band energy: `low_freq_power`, `mid_freq_power`, `high_freq_power`
- Band ratios: `low_to_high_ratio`, `mid_to_total_ratio`

**Frequency Characteristics:**
- Dominant frequency: `dominant_freq_x`, `dominant_freq_y`, `dominant_freq_magnitude`
- Spectral centroid: `spectral_centroid_x`, `spectral_centroid_y`
- Spectral bandwidth, rolloff frequency
- Peak frequency locations

**Spectral Shape:**
- Spectral flatness (Wiener entropy)
- Spectral entropy (Shannon entropy)
- Power law exponent
- Spectral slope
- Spectral anisotropy

**Multi-Scale Analysis (v3.0 NEW):**
- Wavelet energy per scale (4-8 scales)
- Scale-specific spectral features
- Cross-scale coupling metrics

#### 3. Cross-Channel Features (~10D)

**Pairwise Correlations:**
- Instantaneous correlation: `cross_channel_corr_mean`, `cross_channel_corr_std`
- Covariance statistics

**Covariance Matrix Analysis:**
- Eigenvalue statistics: `cross_channel_eigen_top_1`, `cross_channel_eigen_top_2`, `cross_channel_eigen_top_3`
- Matrix properties: `cross_channel_eigen_trace`, `cross_channel_condition_number`
- Effective rank: `cross_channel_participation_ratio`

#### 4. Enhanced Temporal Dynamics (~120D, v3.0 NEW)

**Windowed Statistics:**
- Velocity (first derivative over short windows): `velocity_mean`, `velocity_std`
- Acceleration (second derivative): `acceleration_mean`, `acceleration_std`
- Local trend estimates (short-window linear fit)
- Recent volatility (windowed variance)

**Stability Metrics (Lyapunov-inspired):**
- Local divergence rate estimates
- Trajectory stability indicators
- Online perturbation sensitivity estimates
- Convergence/divergence indicators

**Phase Space Reconstruction:**
- Embedding dimension estimates (using windowed data)
- Attractor reconstruction metrics
- Online recurrence features (windowed RQA)
- Phase space occupancy measures

**Temporal Autocorrelation (windowed):**
- Short-lag autocorrelation: `autocorr_lag_1` through `autocorr_lag_5`
- Decorrelation timescale estimates
- Memory depth indicators
- Persistence metrics

**Rate of Change Features:**
- Instantaneous rate of change (derivative estimate)
- Smoothed derivatives (Savitzky-Golay filter)
- Cumulative change tracking
- Change detection metrics (thresholded derivatives)

### Storage Format

- **Sequences**: `[N, T, D_temporal]` where D_temporal ≈ 328 (v3.0)
- **HDF5 path**: `/features/temporal/features`
- **Attributes**: `@version` (3.0.0), `@feature_registry` (JSON mapping)

### v3.0 vs v2.x Comparison

| Aspect | v3.0 TEMPORAL | v2.x TEMPORAL | v2.x SUMMARY |
|--------|---------------|---------------|--------------|
| Granularity | Per-timestep `[N, T, ~328]` | Per-timestep `[N, T, 63]` | Aggregated `[N, 360]` |
| Temporal info | Full time series | Full time series | Collapsed to statistics |
| Lookahead | None (online) | None (online) | Full trajectory required |
| Use case | Online VQ-VAE encoding, NOA prediction | Sequence modeling | Episodic compression |
| Feature count | ~328 per timestep | 63 per timestep | 360 aggregated |

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
