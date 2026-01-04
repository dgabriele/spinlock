# Dataset: 100K Full Features

**Date:** January 2026
**File:** `datasets/100k_full_features.h5`
**Size:** ~10 GB
**Status:** PRODUCTION READY

---

## Executive Summary

Production dataset of 100,000 neural operators with comprehensive feature extraction across three families (SUMMARY, TEMPORAL, ARCHITECTURE). Designed for VQ-VAE tokenization and Neural Operator Agent training.

| Metric | Value |
|--------|-------|
| **Total Operators** | 100,000 |
| **Realizations per Operator** | 5 |
| **Grid Size** | 64×64 |
| **Timesteps** | 256 |
| **Channels** | 3 |
| **Parameter Dimensions** | 12 |

---

## Dataset Structure

```
datasets/100k_full_features.h5
│
├── inputs/
│   └── fields                           [100000, 3, 64, 64]    float32
│
├── parameters/
│   └── params                           [100000, 12]           float32
│
├── features/
│   ├── summary/
│   │   ├── aggregated/
│   │   │   ├── features                 [100000, 360]          float32
│   │   │   └── metadata/extraction_time [100000]               float64
│   │   └── per_trajectory/
│   │       └── features                 [100000, 5, 120]       float32
│   │
│   ├── temporal/
│   │   └── features                     [100000, 256, 63]      float32
│   │
│   └── architecture/
│       └── aggregated/
│           └── features                 [100000, 12]           float32
│
└── metadata/
    ├── ic_types                         [100000]               object
    ├── evolution_policies               [100000]               object
    ├── grid_sizes                       [100000]               int32
    └── noise_regimes                    [100000]               object
```

---

## Feature Families

### SUMMARY (360 features)

Aggregated behavioral statistics per operator, computed across all timesteps and realizations.

| Category | Features | Description |
|----------|----------|-------------|
| Spatial | 36 | Mean, std, gradients, Laplacian, skewness, kurtosis |
| Spectral | 36 | FFT power, dominant frequencies, spectral entropy |
| Temporal | 36 | Autocorrelation, trend, stability metrics |
| Cross-channel | 36 | Channel correlations, coherence |
| Causality | 42 | Granger causality, transfer entropy |
| Invariant drift | 180 | Multiscale drift statistics |
| Operator sensitivity | 30 | Lipschitz estimates (NaN when disabled) |

**Storage:** `features/summary/aggregated/features` [N, 360]

**Per-trajectory:** `features/summary/per_trajectory/features` [N, 5, 120]

### TEMPORAL (63 features × 256 timesteps)

Full temporal resolution feature trajectories preserving time-series structure.

| Category | Features | Description |
|----------|----------|-------------|
| Spatial statistics | 19 | Per-timestep spatial moments |
| Spectral properties | 27 | Per-timestep FFT features |
| Cross-channel | 17 | Per-timestep channel correlations |

**Storage:** `features/temporal/features` [N, 256, 63]

### ARCHITECTURE (12 features)

Normalized operator parameters in [0, 1] range.

| Parameter | Description |
|-----------|-------------|
| num_layers | Number of convolutional layers (2-5) |
| base_channels | Base channel count (16-64) |
| kernel_size | Convolution kernel size (3, 5, 7) |
| activation | Activation function (gelu) |
| dropout_rate | Dropout probability (0.0-0.3) |
| noise_scale | Stochastic noise scale (log-scaled) |
| spatial_correlation | Noise spatial correlation (0.0-0.3) |
| update_policy | Evolution policy (residual, convex) |
| ... | Additional architecture/evolution params |

**Storage:** `features/architecture/aggregated/features` [N, 12]

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

| Policy | Count | Percentage |
|--------|-------|------------|
| residual | 75,000 | 75.0% |
| convex | 25,000 | 25.0% |

---

## Generation Configuration

```yaml
version: "1.0"

metadata:
  name: "100k_full_features"
  description: |
    Production 100K dataset with SUMMARY + TEMPORAL feature extraction.
    64×64 grid optimal for VQ-VAE compression + NOA training.
    T=256 captures transient dynamics, M=5 for statistics.

# 12-dimensional parameter space (normalized to [0, 1])
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

# Feature extraction
features:
  temporal:
    enabled: true   # TEMPORAL family (per-timestep time series)

  summary:
    enabled: true   # SUMMARY family (aggregated scalars)
    extract_operator_features: false  # Disable operator sensitivity (expensive)
```

---

## Usage

### Load Dataset

```python
import h5py
import numpy as np

with h5py.File("datasets/100k_full_features.h5", "r") as f:
    # Load features
    summary = f["features/summary/aggregated/features"][:]      # [100000, 360]
    temporal = f["features/temporal/features"][:]               # [100000, 256, 63]
    architecture = f["features/architecture/aggregated/features"][:]  # [100000, 12]

    # Load metadata
    ic_types = f["metadata/ic_types"][:].astype(str)
    evolution_policies = f["metadata/evolution_policies"][:].astype(str)

    # Load inputs (initial conditions)
    inputs = f["inputs/fields"][:]  # [100000, 3, 64, 64]
```

### Filter by IC Type

```python
# Get indices for specific IC types
structured_idx = np.where(ic_types == "structured")[0]
localized_idx = np.where(ic_types == "localized")[0]

# Load subset
structured_features = summary[structured_idx]
```

### Access Per-Trajectory Features

```python
# Per-trajectory features before aggregation
per_traj = f["features/summary/per_trajectory/features"][:]  # [100000, 5, 120]

# Aggregate manually if needed
mean_features = per_traj.mean(axis=1)  # [100000, 120]
std_features = per_traj.std(axis=1)    # [100000, 120]
```

---

## Known Issues

### SUMMARY NaN (30 features)

Operator sensitivity features (indices 110-119, 230-239, 350-359) are NaN because `extract_operator_features: false` in generation config. This is expected and handled by VQ-VAE training (replaced with 0).

### TEMPORAL NaN (Fixed)

Skewness/kurtosis at t=0 were NaN for structured ICs (symmetric distributions). Fixed in dataset by replacing NaN with 0. Source code fix applied to `src/spinlock/features/summary/spatial.py` for future extractions.

---

## Related Documents

- [**100K VQ-VAE Baseline**](100k-full-features-vqvae.md) - VQ-VAE trained on this dataset
- [**Feature Reference**](../features/README.md) - Feature family definitions
- [**HDF5 Layout**](../features/hdf5-layout.md) - Dataset schema reference

---

**Generated:** January 2026
**Status:** PRODUCTION READY
