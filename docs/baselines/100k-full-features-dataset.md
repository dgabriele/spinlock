# Dataset: 100K Full Features (v3.0)

**Date:** January 2026 (Updated: 2026-01-18)
**File:** `datasets/100k_full_features_v3.h5`
**Size:** ~12 GB
**Status:** PRODUCTION READY (v3.0)

---

## Executive Summary

Production dataset of 100,000 neural operators with comprehensive feature extraction. **v3.0 architecture** with enhanced per-timestep TEMPORAL features (~328D) and expanded 14D parameter space. Designed for VQ-VAE tokenization and Neural Operator Agent training.

| Metric | Value |
|--------|-------|
| **Total Operators** | 100,000 |
| **Realizations per Operator** | 5 |
| **Grid Size** | 64×64 |
| **Timesteps** | 256 |
| **Channels** | 3 |
| **Parameter Dimensions** | 14 (v3.0+) |

**v3.0 Changes:**
- Removed SUMMARY features (incompatible with online prediction)
- Enhanced TEMPORAL features: 63D → ~328D per-timestep
- Parameter space: 12D → 14D (added dt and alpha variation)
- Evolution dynamics: 2 policies → ~19 unique configurations

---

## Dataset Structure

```
datasets/100k_full_features_v3.h5
│
├── inputs/
│   └── fields                           [100000, 3, 64, 64]    float32
│
├── parameters/
│   └── params                           [100000, 14]           float32  (Sobol unit cube)
│
├── features/
│   └── temporal/
│       ├── @version                                                     (3.0.0)
│       ├── @feature_registry                                            (JSON)
│       └── features                     [100000, 256, 328]     float32  (per-timestep)
│
└── metadata/
    ├── ic_types                         [100000]               object
    ├── evolution_policies               [100000]               object
    ├── grid_sizes                       [100000]               int32
    └── noise_regimes                    [100000]               object
```

**v3.0 Structure Changes:**
- `/features/summary/` → REMOVED (aggregated, per_trajectory, learned)
- `/features/architecture/` → REMOVED (now in `/parameters/params`)
- `/features/temporal/features`: [N, T, 63] → [N, T, 328]
- `/parameters/params`: [N, 12] → [N, 14]

---

## Feature Families

### TEMPORAL (~328 features × 256 timesteps)

**v3.0 Enhanced:** Per-timestep behavioral features preserving full temporal resolution. All features are online-computable (no lookahead required) for NOA predictions.

| Category | Features | Description |
|----------|----------|-------------|
| **Spatial** | ~105 | Per-channel statistics, gradients, Laplacian, histograms |
| **Spectral** | ~93 | Multi-scale FFT, power spectrum, frequency bands, spectral entropy |
| **Cross-channel** | ~10 | Pairwise correlations, covariance eigenvalues |
| **Temporal dynamics** | ~120 | Windowed statistics, stability metrics, phase space, autocorrelation |

**Storage:** `features/temporal/features` [N, 256, 328]

**Key Change (v3.0):** Enhanced from 63D to ~328D by adding:
- Multi-scale spatial features (histogram-based distributions)
- Enhanced spectral analysis (frequency band decomposition)
- Phase space reconstruction and stability metrics
- Windowed temporal dynamics (velocity, acceleration)

### ~~SUMMARY~~ [REMOVED in v3.0]

Aggregated trajectory-level features (360D: causality, invariant drift, operator sensitivity) were removed because they require complete trajectories and are incompatible with per-timestep online prediction.

**Archived code:** `src/spinlock/features/temporal_old_v2/summary/`

### ARCHITECTURE (14D parameter space → ~20D features)

Operator parameters stored in `/parameters/params [N, 14]` as normalized Sobol unit cube values.

**14D Parameter Space:**

| Category | Parameters | Description |
|----------|------------|-------------|
| Architecture (5D) | num_layers, base_channels, kernel_size, activation, dropout_rate | Network architecture |
| Stochastic (4D) | noise_type, noise_scale, noise_schedule, spatial_correlation | Stochastic forcing |
| Operator (2D) | normalization, grid_size | Operator configuration |
| Evolution (3D) | update_policy, dt, alpha | Integration dynamics |

**v3.0 Enhancement:** Evolution parameters expanded from 1D to 3D:
- `dt`: 10 discrete choices [0.005, 0.01, ..., 0.05] for residual policy
- `alpha`: 9 discrete choices [0.1, 0.2, ..., 0.9] for convex policy
- Results in **~19 unique evolution dynamics** (was 2 in v2.x)

**Storage:** `parameters/params` [N, 14] (Sobol unit cube, not in `/features/`)

**Note:** ARCHITECTURE features are excluded from VQ-VAE training (NOA already knows operator parameters θ).

---

## Initial Condition Distribution

Balanced 4-family IC design minimizing semantic bias:

| Family | Variants | Total % |
|--------|----------|---------|
| **Gaussian Noise** | 5 variance levels | 25.0% |
| **Band-limited** | 3 frequency bands | 25.0% |
| **Sinusoids** | structured | 25.0% |
| **Localized Blobs** | localized | 25.0% |

### Detailed Distribution

| IC Type | Count | Percentage |
|---------|-------|------------|
| localized | 24,993 | 25.0% |
| structured | 24,984 | 25.0% |
| multiscale_grf_mid | 8,494 | 8.5% |
| multiscale_grf_high | 8,487 | 8.5% |
| multiscale_grf_low | 8,269 | 8.3% |
| gaussian_random_field_v0 | 5,005 | 5.0% |
| gaussian_random_field_v4 | 4,982 | 5.0% |
| gaussian_random_field_v1 | 4,957 | 5.0% |
| gaussian_random_field_v2 | 4,952 | 5.0% |
| gaussian_random_field_v3 | 4,877 | 4.9% |

### Evolution Policy Distribution

**v3.0 Enhanced:** Evolution dynamics now vary dt and alpha parameters, resulting in **~19 unique configurations** instead of 2.

| Base Policy | Parameter Variation | Unique Configs |
|-------------|---------------------|----------------|
| **residual** | dt ∈ [0.005, 0.01, ..., 0.05] (10 choices) | ~10 |
| **convex** | alpha ∈ [0.1, 0.2, ..., 0.9] (9 choices) | ~9 |

**Total:** ~19 unique evolution dynamics configurations across the dataset

**Distribution (approximate):**
- ~75% residual policy with varying dt (10 different timestep sizes)
- ~25% convex policy with varying alpha (9 different convex weights)

---

## Generation Configuration

```yaml
version: "1.0"

metadata:
  name: "100k_full_features_v3"
  description: |
    Production 100K dataset with v3.0 enhanced TEMPORAL features (~328D per-timestep).
    64×64 grid optimal for VQ-VAE compression + NOA training.
    T=256 captures transient dynamics, M=5 for statistics.
    14D parameter space with varying evolution dynamics (dt, alpha).

# 14-dimensional parameter space (normalized to [0, 1] Sobol cube)
parameter_space:
  architecture:
    num_layers:
      type: integer
      bounds: [2, 5]

    base_channels:
      type: integer
      bounds: [16, 64]

    kernel_size:
      type: choice
      choices: [3, 5, 7]

    activation:
      type: choice
      choices: ["gelu"]

    dropout_rate:
      type: continuous
      bounds: [0.0, 0.3]

  stochastic:
    noise_type:
      type: choice
      choices: ["gaussian"]

    noise_scale:
      type: continuous
      bounds: [0.00001, 1.0]
      log_scale: true

    noise_schedule:
      type: choice
      choices: ["constant"]

    spatial_correlation:
      type: continuous
      bounds: [0.0, 0.3]

  operator:
    normalization:
      type: choice
      choices: ["instance"]

    grid_size:
      type: choice
      choices: [64]

  evolution:
    update_policy:
      type: choice
      choices: ["residual", "convex"]
      weights: [0.75, 0.25]

    dt:  # v3.0: Timestep size for residual policy
      type: choice
      choices: [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

    alpha:  # v3.0: Convex weight for convex policy
      type: choice
      choices: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Sampling configuration
sampling:
  strategy: "sobol_stratified"

  sobol:
    scramble: true
    seed: 42

  stratification:
    method: "adaptive"
    num_strata_per_dim: 5
    min_samples_per_stratum: 20

  validation:
    check_discrepancy: true
    check_correlation: true

  total_samples: 100000
  batch_size: 8

# Simulation configuration
simulation:
  device: "cuda"

  input_generation:
    method: "sampled"

    # Equal weighting: 25% per IC family
    ic_type_weights:
      # Gaussian noise family (25% total, 5 variance levels)
      gaussian_random_field_v0: 0.05   # variance=0.25
      gaussian_random_field_v1: 0.05   # variance=0.5
      gaussian_random_field_v2: 0.05   # variance=1.0
      gaussian_random_field_v3: 0.05   # variance=2.0
      gaussian_random_field_v4: 0.05   # variance=4.0

      # Band-limited noise family (25% total, 3 bands)
      multiscale_grf_low: 0.0833
      multiscale_grf_mid: 0.0833
      multiscale_grf_high: 0.0834

      # Sinusoid family (25%)
      structured: 0.25

      # Localized blob family (25%)
      localized: 0.25

    # Gaussian noise configurations (5 variance levels)
    gaussian_random_field_v0:
      length_scale: 0.05
      variance: 0.25

    gaussian_random_field_v1:
      length_scale: 0.05
      variance: 0.5

    gaussian_random_field_v2:
      length_scale: 0.05
      variance: 1.0

    gaussian_random_field_v3:
      length_scale: 0.05
      variance: 2.0

    gaussian_random_field_v4:
      length_scale: 0.05
      variance: 4.0

    # Band-limited noise configurations (3 frequency bands)
    multiscale_grf_low:
      scales: [0.30, 0.35, 0.40]
      variance: 1.0

    multiscale_grf_mid:
      scales: [0.08, 0.10, 0.12]
      variance: 1.0

    multiscale_grf_high:
      scales: [0.02, 0.025, 0.03]
      variance: 1.0

    # Sinusoid configurations
    structured:
      num_modes: 1
      wavelength_range: [8.0, 64.0]
      amplitude_range: [0.5, 2.0]

    # Localized blob configurations
    localized:
      num_blobs: 5
      min_width: 5.0
      max_width: 15.0

  num_realizations: 5
  num_timesteps: 256

# Dataset output
dataset:
  output_path: "datasets/100k_full_features.h5"

  storage:
    compression: "gzip"
    compression_level: 4
    chunk_size: 32

# Feature extraction (v3.0)
features:
  temporal:
    enabled: true   # TEMPORAL family (~328D per-timestep features)
    version: "3.0.0"  # Enhanced feature set

  # v3.0: SUMMARY features removed (incompatible with online prediction)
```

---

## Usage

### Load Dataset (v3.0)

```python
import h5py
import numpy as np

with h5py.File("datasets/100k_full_features_v3.h5", "r") as f:
    # Load TEMPORAL features (primary VQ-VAE input)
    temporal = f["features/temporal/features"][:]               # [100000, 256, 328]

    # Load parameters (14D Sobol unit cube)
    params = f["parameters/params"][:]                          # [100000, 14]

    # Load metadata
    ic_types = f["metadata/ic_types"][:].astype(str)
    evolution_policies = f["metadata/evolution_policies"][:].astype(str)

    # Load inputs (initial conditions)
    inputs = f["inputs/fields"][:]  # [100000, 3, 64, 64]

    # Check feature registry
    import json
    registry = json.loads(f["features/temporal"].attrs["feature_registry"])
    print(f"Feature categories: {list(registry.keys())}")
```

### Filter by IC Type

```python
# Get indices for specific IC types
structured_idx = np.where(ic_types == "structured")[0]
localized_idx = np.where(ic_types == "localized")[0]

# Load subset of temporal features
structured_temporal = temporal[structured_idx]  # [N_structured, 256, 328]
```

### Access Per-Timestep Features

```python
# v3.0: All features are per-timestep [N, T, D]
# Extract features at specific timesteps
t0_features = temporal[:, 0, :]     # [100000, 328] - Initial timestep
t_final_features = temporal[:, -1, :]  # [100000, 328] - Final timestep

# Compute temporal aggregates if needed (post-hoc)
mean_over_time = temporal.mean(axis=1)  # [100000, 328]
std_over_time = temporal.std(axis=1)    # [100000, 328]
```

---

## Known Issues

### None (v3.0)

v3.0 datasets have no known NaN issues. All features are validated during extraction:
- TEMPORAL features use robust statistics (NaN-safe operations)
- Feature cleaning pipeline removes any NaN-containing features
- All per-timestep computations handle edge cases (t=0, uniform fields, etc.)

### Legacy Issues (v2.x only)

**v2.x datasets had:**
- SUMMARY operator sensitivity features with NaN (when `extract_operator_features: false`)
- TEMPORAL skewness/kurtosis NaN at t=0 for symmetric ICs

**v3.0 fixes:**
- Removed SUMMARY features entirely
- Enhanced TEMPORAL features with robust statistics

---

## Related Documents

- [**100K VQ-VAE Baseline**](100k-full-features-vqvae.md) - VQ-VAE trained on this dataset
- [**Feature Reference**](../features/README.md) - Feature family definitions
- [**HDF5 Layout**](../features/hdf5-layout.md) - Dataset schema reference

---

**Generated:** January 2026
**Status:** PRODUCTION READY
