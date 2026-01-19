# Feature Catalog: Current Configuration (v3.0)

**Date**: 2026-01-18
**Purpose**: Enumerate features computed in the current system configuration

This document lists the multi-modal features extracted from operator rollouts, organized by family and analysis category. Feature dimensions reflect the current configuration and are adjustable based on enabled categories and embedding sizes.

**v3.0 Update**: SUMMARY features removed. All features are now per-timestep computable for online NOA predictions.

---

## Overview

| Family | Dimensions (Current Config) | Temporal Resolution | Purpose |
|--------|-----------|---------------------|---------|
| **INITIAL** | Manual (14) + CNN (configurable) | Single snapshot (IC) | Encode initial condition characteristics |
| **TEMPORAL** | ~328D per-timestep (v3.0) | Per-timestep sequence | Capture behavioral evolution with online-computable features |

**Configuration-Dependent**: Exact dimensions vary based on which feature categories are enabled and CNN embedding size.

**v3.0 Changes**:
- Removed SUMMARY features (incompatible with online prediction)
- Enhanced TEMPORAL features from ~63D to ~328D per-timestep
- All features are now per-timestep computable (no trajectory lookahead)

---

## INITIAL Features

**Purpose**: Characterize the initial condition's spatial structure, spectral content, information content, and morphological properties.

**Resolution**: Single snapshot per IC (computed once at t=0)

### Manual Features (14D)

Hand-crafted features providing interpretable, domain-driven characterization.

#### Spatial Structure (4 features)

1. **`spatial_cluster_count`**: Number of connected components (approximate clustering)
2. **`spatial_largest_cluster_frac`**: Fraction of domain occupied by largest cluster
3. **`spatial_autocorr`**: Moran's I spatial autocorrelation (locality measure)
4. **`spatial_centroid_dist`**: Distance of mass centroid from domain center (localization)

#### Spectral Analysis (3 features)

5. **`spectral_dominant_freq`**: Frequency with maximum power (characteristic scale)
6. **`spectral_centroid`**: Center of spectral mass (weighted average frequency)
7. **`spectral_power_law_exp`**: Power law exponent from log-log spectrum fit

#### Information Theory (4 features)

8. **`info_entropy`**: Shannon entropy of histogram-binned field values
9. **`info_local_entropy_var`**: Variance of local entropy (spatial heterogeneity)
10. **`info_lz_complexity`**: Lempel-Ziv complexity (predictability measure)
11. **`info_predictability`**: 1 - normalized LZ complexity

#### Morphological Properties (3 features)

12. **`morph_density`**: Fraction of domain above threshold (activity density)
13. **`morph_radial_gradient`**: Average radial gradient magnitude (edge content)
14. **`morph_symmetry`**: Rotational symmetry measure (anisotropy)

### CNN Learned Features (configurable)

**`cnn_embed_0` through `cnn_embed_N-1`**: Learned embeddings from convolutional encoder (N = embedding_dim config parameter)
   - Captures high-level latent structure not readily expressed in manual features
   - Optional VAE mode enables generative bidirectionality (embed ↔ IC reconstruction)
   - Frozen or trainable depending on configuration

---

## ~~SUMMARY Features~~ [REMOVED in v3.0]

**SUMMARY features were removed in v3.0.0** to enable online per-timestep predictions in the Neural Operator Apprentice (NOA).

**Why removed:** Aggregated trajectory-level features (causality, invariant drift, operator sensitivity, nonlinear dynamics) require seeing the complete trajectory, making them incompatible with real-time online operation where the model must make predictions at each timestep without lookahead.

**Archived implementation:** `src/spinlock/features/temporal_old_v2/summary/`

**Migration:** Users needing trajectory-level aggregates can compute them post-hoc from TEMPORAL features or use v2.x datasets.

---

### Legacy SUMMARY Categories (v2.x only - for reference)

<details>
<summary>Click to expand legacy SUMMARY feature documentation (v2.x)</summary>

**Note:** This documentation is preserved for reference but describes features that are **not present in v3.0+ datasets**.

**Purpose**: Aggregate behavioral signatures across full trajectory (v2.x only)

**Resolution**: Trajectory-level (single vector per rollout)

### 1. Spatial Statistics

**Distributional Moments**:
- `spatial_mean`, `spatial_variance`, `spatial_std`
- `spatial_skewness`, `spatial_kurtosis`

**Extrema and Spread**:
- `spatial_min`, `spatial_max`, `spatial_range`
- `spatial_median`, `spatial_iqr`
- `spatial_q25`, `spatial_q75` (quartiles)

**Central Tendency**:
- `spatial_mean_absolute_deviation` (MAD)
- `spatial_robust_std` (using IQR)

**Distribution Shape**:
- `spatial_entropy` (histogram-based)
- `spatial_effective_dimension` (participation ratio)

**Spatial Structure**:
- `spatial_autocorr_lag1` (Moran's I at lag 1)
- `spatial_isotropy` (directional uniformity)

**Gradient Properties**:
- `spatial_gradient_mean`, `spatial_gradient_std`
- `spatial_gradient_max`
- `spatial_laplacian_mean`, `spatial_laplacian_std`

**Pattern Complexity**:
- `spatial_fractal_dim` (box-counting dimension)
- `spatial_perimeter_area_ratio`
- `spatial_circularity`

**Energy-Based**:
- `spatial_total_energy` (L2 norm)
- `spatial_energy_concentration` (top-k contribution)

**Extensions** (Phase 1: +8 features):
- Percentiles: `p10`, `p90`, `p95`, `p99`
- Event statistics: `above_threshold_count`, `time_above_threshold`
- Rolling windows: `rolling_mean_std`, `rolling_max_range`

### 2. Spectral Features

**Power Spectrum**:
- `spectral_total_power`
- `spectral_mean_power`, `spectral_std_power`

**Frequency Characteristics**:
- `spectral_dominant_freq` (peak frequency)
- `spectral_centroid` (weighted center of mass)
- `spectral_bandwidth` (frequency spread)
- `spectral_rolloff` (95% energy frequency)

**Multi-Scale Analysis**:
- `spectral_power_low`, `spectral_power_mid`, `spectral_power_high`
- `spectral_low_to_high_ratio`

**Spectral Shape**:
- `spectral_flatness` (Wiener entropy)
- `spectral_entropy` (Shannon entropy of normalized spectrum)
- `spectral_slope` (log-log fit)
- `spectral_power_law_exponent`

**Phase Information**:
- `spectral_phase_coherence` (cross-channel phase locking)
- `spectral_phase_entropy`

**Anisotropy**:
- `spectral_directional_anisotropy` (angular power distribution)

**Wavelet Features** (if enabled):
- `wavelet_energy_scale1` through `wavelet_energy_scale4`
- `wavelet_entropy`

### 3. Temporal Dynamics

**Trend Analysis**:
- `temporal_mean_trend` (linear trend slope)
- `temporal_std_trend`
- `temporal_acceleration` (second derivative)

**Variation Metrics**:
- `temporal_total_variation` (TV semi-norm)
- `temporal_rate_of_change_mean`, `temporal_rate_of_change_std`

**Stationarity**:
- `temporal_stationarity` (variance of chunk means)
- `temporal_drift` (end - start)

**Autocorrelation**:
- `temporal_autocorr_lag1`, `temporal_autocorr_lag5`, `temporal_autocorr_lag10`
- `temporal_decorrelation_time` (ACF crossing zero)

**Periodicity**:
- `temporal_dominant_period` (from ACF)
- `temporal_periodicity_strength` (ACF peak height)

**Complexity**:
- `temporal_sample_entropy` (ApEn analog)
- `temporal_permutation_entropy`
- `temporal_lz_complexity`

**Stability**:
- `temporal_lyapunov_estimate` (maximal Lyapunov exponent approx)
- `temporal_trajectory_divergence`

**Predictability**:
- `temporal_forecast_error` (AR model residual)
- `temporal_next_step_correlation`

**Phase Space**:
- `temporal_recurrence_rate` (RQA)
- `temporal_determinism` (RQA diagonal lines)
- `temporal_laminarity` (RQA vertical lines)

**Transitions**:
- `temporal_num_transitions` (significant regime changes)
- `temporal_transition_frequency`
- `temporal_mean_dwell_time` (time in regimes)

**Event Statistics**:
- `temporal_peak_count`, `temporal_peak_prominence`
- `temporal_zero_crossing_rate`

**Extensions** (Phase 1: +33 features):
- Advanced RQA: `recurrence_determinism`, `recurrence_entropy`, `recurrence_laminarity`
- Correlation dimension: `correlation_dim` (phase space dimensionality)
- Event analysis: `peak_intervals_mean`, `peak_intervals_std`
- Rolling statistics: `rolling_autocorr_mean`, `rolling_entropy_std`

### 4. Operator Sensitivity

**Parameter Gradient Analysis**:
- `param_gradient_l2` (sensitivity to parameter perturbations)
- `param_gradient_mean`, `param_gradient_std`

**Jacobian Properties**:
- `jacobian_spectral_radius` (largest eigenvalue)
- `jacobian_condition_number` (stability indicator)
- `jacobian_rank_deficit` (dimensionality)

**Perturbation Response**:
- `ic_sensitivity` (response to IC perturbations)
- `ic_lyapunov_stability` (exponential divergence rate)

**Nonlinearity**:
- `operator_nonlinearity_index` (deviation from linear response)
- `interaction_strength` (cross-term magnitudes)

**Regime Sensitivity**:
- `regime_transition_sensitivity` (ease of changing regimes)
- `bifurcation_proximity` (distance to instability)

### 5. Cross-Channel Interactions

**Per-Timestep Statistics** (averaged across trajectory):
- `channel_correlation_mean`, `channel_correlation_std`
- `channel_covariance_mean`, `channel_covariance_std`

**Information Flow**:
- `channel_mutual_info` (shared information)
- `channel_transfer_entropy` (directional information transfer)

**Synchronization**:
- `channel_phase_sync` (phase locking value)
- `channel_amplitude_coupling` (correlation of envelopes)

**Coupling Strength**:
- `channel_coupling_coefficient`
- `channel_effective_connectivity` (Granger causality)

**Nonlinear Coupling**:
- `channel_coherence` (frequency-domain coupling)
- `channel_bispectrum` (phase coupling, three-wave interactions)

### 6. Causality & Directionality

**Granger Causality**:
- `granger_causality_forward` (X → Y predictability)
- `granger_causality_reverse` (Y → X predictability)
- `granger_net_directionality` (forward - reverse)

**Transfer Entropy**:
- `transfer_entropy_xy`, `transfer_entropy_yx`
- `transfer_entropy_asymmetry`

**Phase-Based Directionality**:
- `phase_slope_index` (frequency-domain causality)
- `directed_phase_lag_index`

**Lag Structure**:
- `optimal_lag_forward`, `optimal_lag_reverse`
- `lag_asymmetry`

**Feedback Detection**:
- `feedback_strength` (bidirectional coupling)
- `feedback_delay` (characteristic timescale)

**Nonlinear Directionality**:
- `convergent_cross_mapping` (state-space causality)
- `symbolic_transfer_entropy`

### 7. Invariant Drift

**Conservation Laws** (per conserved quantity: energy, mass, momentum, angular momentum):
- `{quantity}_initial`, `{quantity}_final`
- `{quantity}_drift` (total change)
- `{quantity}_drift_rate` (per timestep)
- `{quantity}_drift_acceleration`
- `{quantity}_fluctuation_amplitude`
- `{quantity}_fluctuation_period`
- `{quantity}_restoration_rate` (return to equilibrium)

**Multi-Scale Invariants** (per scale: full, highpass, lowpass):
- `energy_{scale}_initial`, `energy_{scale}_final`
- `energy_{scale}_drift`, `energy_{scale}_drift_rate`
- `entropy_{scale}_initial`, `entropy_{scale}_final`
- `entropy_{scale}_drift`, `entropy_{scale}_drift_rate`

**Derived Ratios**:
- `energy_final_initial_ratio_{scale}` (growth/decay)
- `entropy_final_initial_ratio_{scale}` (information change)

### 8. Nonlinear Dynamics

**Attractor Properties**:
- `attractor_dimension` (correlation dimension)
- `attractor_volume` (phase space occupancy)

**Chaos Indicators**:
- `lyapunov_exponent_max` (maximal Lyapunov exponent)
- `lyapunov_exponent_sum` (KS entropy)

**Bifurcation Proximity**:
- `bifurcation_distance` (nearest instability)
- `fold_bifurcation_indicator`

**Stability**:
- `fixed_point_stability` (convergence to equilibrium)
- `limit_cycle_stability` (periodic attractor robustness)

</details>

---

## TEMPORAL Features (per timestep)

**Purpose**: Per-timestep behavioral features for sequential reasoning, online prediction, and real-time NOA operation.

**Resolution**: One D_temporal-dimensional vector per timestep (typical: ~328D in v3.0)

**v3.0 Enhancement**: Expanded from ~63D to ~328D per-timestep by adding multi-scale spatial features, enhanced spectral analysis, and advanced temporal dynamics.

### Per-Timestep Feature Categories

#### 1. Spatial Features (~105D)

**Per-Channel Statistics**:
- Moments: mean, variance, std, skewness, kurtosis (per channel)
- Extrema: min, max, range, median, quartiles (per channel)
- Spread: MAD, IQR, percentiles (p10, p90, p95, p99)
- Entropy measures (histogram-based)

**Spatial Gradients**:
- Gradient magnitude statistics (mean, std, max)
- Laplacian statistics (mean, std, max)
- Directional gradients (x, y components)
- Edge density metrics

**Histogram Features (v3.0 NEW)**:
- Multi-bin histogram distributions (10-20 bins)
- Distribution shape metrics
- Multi-scale histogram features
- Adaptive binning statistics

**Pattern Metrics**:
- Autocorrelation (Moran's I)
- Isotropy/anisotropy measures
- Spatial coherence length
- Clustering metrics

#### 2. Spectral Features (~93D)

**Power Spectrum (v3.0 ENHANCED)**:
- Total power, mean power, std power
- Multi-scale FFT features (2-4 scales)
- Frequency band decomposition (low, mid, high, ultra-high)
- Per-band energy distributions

**Frequency Characteristics**:
- Dominant frequency, spectral centroid
- Spectral bandwidth, rolloff frequency
- Frequency spread metrics
- Peak frequency locations

**Spectral Shape**:
- Spectral flatness (Wiener entropy)
- Spectral entropy (Shannon entropy)
- Power law exponent
- Spectral slope

**Multi-Scale Analysis (v3.0 NEW)**:
- Wavelet energy per scale (4-8 scales)
- Scale-specific spectral features
- Cross-scale coupling metrics

#### 3. Cross-Channel Features (~10D)

**Pairwise Correlations**:
- Instantaneous correlation coefficients
- Covariance statistics
- Correlation matrix properties

**Covariance Matrix Analysis**:
- Eigenvalue statistics (max, min, sum)
- Condition number
- Effective rank
- Trace and determinant

**Synchronization**:
- Phase synchronization metrics
- Amplitude coupling

#### 4. Enhanced Temporal Dynamics (~120D, v3.0 NEW)

**Windowed Statistics**:
- Velocity (first derivative) statistics over short windows
- Acceleration (second derivative) statistics
- Local trend estimates
- Recent volatility measures

**Stability Metrics (Lyapunov-inspired)**:
- Local divergence rates
- Trajectory stability indicators
- Perturbation sensitivity (online estimate)
- Convergence/divergence indicators

**Phase Space Reconstruction**:
- Embedding dimension estimates
- Attractor reconstruction metrics
- Recurrence features (online RQA)
- Phase space occupancy

**Temporal Autocorrelation (windowed)**:
- Short-lag autocorrelation (lags 1-5)
- Decorrelation timescale estimates
- Memory depth indicators
- Persistence metrics

**Rate of Change Features**:
- Instantaneous rate of change
- Smoothed derivatives
- Cumulative change tracking
- Change detection metrics

---

## Usage in NOA Pipeline (v3.0)

### Stage 1: MNO Training (Pure MSE)
- **INITIAL features**: Not used (MNO trains on raw IC grids)
- **TEMPORAL features**: Not used

### Stage 2: Feature Generation
- Extract **INITIAL** and **TEMPORAL** from 100K+ MNO rollouts
- Inline GPU-optimized extraction (no trajectory storage)
- **v3.0**: Enhanced TEMPORAL features (~328D per-timestep)
- Output: ~2 GB compressed HDF5 dataset (larger due to enhanced features)

### Stage 3: VQ-VAE Training
- Train hierarchical VQ-VAE on MNO's feature distribution
- **INITIAL** + **TEMPORAL** jointly encoded
- **ARCHITECTURE** features excluded (NOA already conditions on θ)
- Discover behavioral categories through compression
- **v3.0**: Per-timestep encoding for online operation

### Phase 2+: NOA Agent
- **INITIAL**: Encode IC characteristics for state representation
- **TEMPORAL**: Per-timestep working memory for online prediction
- Discrete VQ tokens enable online symbolic reasoning
- **v3.0**: All features computable without trajectory lookahead

---

## References

- [Feature Reference](feature-reference.md) - Detailed formulas and interpretations for TEMPORAL features
- [Feature Families README](README.md) - Overview of 3-family architecture
- [HDF5 Layout](hdf5-layout.md) - Dataset schema and storage format
- [NOA Architecture](../noa-architecture.md) - How features integrate with NOA backbone
- [VQ-VAE Training Guide](../vqvae/training-guide.md) - Feature-to-token encoding pipeline

**Implementation (v3.0)**:
- INITIAL: `src/spinlock/features/initial/`
- TEMPORAL: `src/spinlock/features/temporal/`
- ARCHITECTURE: `src/spinlock/features/architecture/` (parameter extraction only)

**Archived (v2.x)**:
- SUMMARY (removed): `src/spinlock/features/temporal_old_v2/summary/`
