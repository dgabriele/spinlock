# HDF5 Dataset Layout Reference

This document describes the complete HDF5 schema for Spinlock datasets, including the feature storage structure used by the VQ-VAE tokenization pipeline.

**Last Updated:** 2026-01-30 (v3.2 - per-channel independent IC support)

## Overview

Spinlock datasets use HDF5 format with two main sections:

1. **Core Dataset** (`/metadata/`, `/parameters/`, `/inputs/`, `/outputs/`) - Operator parameters and rollout data
2. **Features** (`/features/`) - Extracted behavioral features (INITIAL, ARCHITECTURE, TEMPORAL)

## Complete Schema

```
dataset.h5
├── metadata/
│   ├── evolution_policies [N]  # object - Evolution policy per sample
│   ├── grid_sizes [N]          # int32 - Grid size per sample
│   ├── ic_types [N]            # object - IC type per sample (see IC Type Format below)
│   └── noise_regimes [N]       # object - Noise regime per sample
│
├── parameters/
│   └── params [N, P]           # float32 - Sobol parameter vectors (P=14)
│
├── inputs/
│   └── fields [N, M, C, H, W]  # float32 - Initial conditions
│                               # N: samples, M: realizations, C: channels
│                               # H, W: grid height/width
│
├── outputs/                    # (Only if store_trajectories=true)
│   └── (empty or trajectories) # Rollout data (typically not stored)
│
└── features/
    ├── architecture/
    │   └── aggregated/
    │       └── features [N, D_arch]  # float32 - Per-operator architectural features
    │
    ├── initial/
    │   └── aggregated/
    │       └── features [N, D_init]  # float32 - Initial condition features
    │                                 # (aggregated over realizations)
    │
    ├── summary/                # (Present but typically empty in v3.1)
    │   └── (empty)
    │
    └── temporal/
        └── features [N, T, D_temporal]  # float32 - Per-timestep features
```

**v3.1 Changes (3-Channel Support):**
- **inputs/fields** now has shape `[N, M, C, H, W]` with explicit realization dimension
- Support for multi-channel inputs (C=3 for RGB-like data)
- ARCHITECTURE features stored in `/features/architecture/aggregated/features`
- INITIAL features stored in `/features/initial/aggregated/features`
- TEMPORAL features enhanced to ~345D per-timestep

## Dimensions

| Symbol | Description | Typical Value | Example (50K Dataset) |
|--------|-------------|---------------|----------------------|
| N | Number of samples (operators) | 1,000 - 100,000 | 50,000 |
| M | Number of realizations | 3 - 10 | 3 |
| T | Number of timesteps | 100 - 500 | 256 |
| C | Number of channels | 1 - 3 | 3 |
| H, W | Grid height/width | 64 - 128 | 64 |
| P | Parameter dimension | 14 | 14 |
| D_arch | ARCHITECTURE feature dim | ~23 | 23 |
| D_init | INITIAL feature dim | ~38 | 38 |
| D_temporal | TEMPORAL feature dim | ~345 | 345 |

## Feature Families

### ARCHITECTURE Family (`/features/architecture/`)

Per-operator architectural features describing the neural operator structure.

**Shape:** `[N, D_arch]` where D_arch ≈ 23

**Storage:** `/features/architecture/aggregated/features`

**Contents:**
- Operator architecture parameters
- Network topology features
- Structural characteristics

**Use Case:** Understanding how operator architecture affects behavioral regimes.

### INITIAL Family (`/features/initial/`)

Initial condition features aggregated over stochastic realizations.

**Shape:** `[N, D_init]` where D_init ≈ 38

**Storage:** `/features/initial/aggregated/features`

**Contents:**
- Spatial statistics of initial conditions
- Spectral characteristics
- Information-theoretic measures
- Morphological features
- **Aggregated over M realizations** to provide representative IC features

**Use Case:** Understanding how initial conditions influence operator dynamics.

### TEMPORAL Family (`/features/temporal/`)

Per-timestep time series preserving full temporal resolution.

**Shape:** `[N, T, D_temporal]` where D_temporal ≈ 345

**Storage:** `/features/temporal/features`

**Contents (v3.1 Enhanced):**
- **Spatial features:** Per-channel statistics, gradients, Laplacian, histogram features
- **Spectral features:** Multi-scale FFT, power spectrum, frequency bands, spectral entropy
- **Local dynamics:** Windowed statistics, stability metrics, phase space features
- **Wavelet analysis:** Multi-resolution temporal-frequency decomposition
- **Cross-channel features:** Pairwise correlations, covariance eigenvalues (for multi-channel data)

**Use Case:** Working memory analysis, temporal pattern detection, trajectory classification, online NOA predictions.

**Key Property:** All features are **per-timestep computable** (no lookahead required), enabling online operation.

### SUMMARY Family (`/features/summary/`)

**Status:** Present in schema but typically empty in v3.1 datasets. Reserved for future trajectory-level aggregations.

## IC Type Format

The `ic_types` metadata field stores the initial condition type(s) used for each sample.

### Single IC Type (Legacy)

When all channels use the same IC type:
```python
ic_types[0] = "gaussian_random_field"
ic_types[1] = "localized"
ic_types[2] = "multiscale_grf"
```

### Per-Channel IC Types (v3.2+)

When using per-channel independent ICs (`method: "per_channel"`), the format includes channel-specific types:
```python
ic_types[0] = "ch0:grf|ch1:struct|ch2:mgrf"
ic_types[1] = "ch0:local|ch1:grf|ch2:grf"
ic_types[2] = "ch0:mgrf|ch1:struct|ch2:local"
```

**Format:** `ch{i}:{type}|ch{j}:{type}|...`

**Abbreviations:**
- `grf` = gaussian_random_field
- `local` = localized
- `mgrf` = multiscale_grf
- `struct` = structured
- `comp` = composite
- `heavy` = heavy_tailed

**Benefits:**
- Each channel can have a different IC type with different parameters
- Creates richer behavioral diversity for VQ-VAE category discovery
- Enables cross-channel interaction pattern analysis
- Supports compositional reasoning in NOA training

**Example Configuration:**
```yaml
simulation:
  input_generation:
    method: "per_channel"
    channel_configs:
      channel_0:  # Fine-grained features
        ic_type_weights:
          gaussian_random_field: 0.4
          localized: 0.3
          multiscale_grf: 0.3
      channel_1:  # Structured patterns
        ic_type_weights:
          structured: 0.5
          gaussian_random_field: 0.5
      channel_2:  # Coarse features
        ic_type_weights:
          gaussian_random_field: 1.0
```

## Data Types and Ranges

### Inputs (`/inputs/fields`)

**Shape:** `[N, M, C, H, W]` - 5D tensor

**Interpretation:**
- **N:** Sample index (operator)
- **M:** Realization index (stochastic runs)
- **C:** Channel index (e.g., RGB components)
- **H, W:** Spatial grid coordinates

**Example:** `(50000, 3, 3, 64, 64)` = 50,000 operators × 3 realizations × 3 channels × 64×64 grid

**Data Range:** Typically normalized, e.g., `[-12.1, 12.3]` with mean ≈ 0

### Parameters (`/parameters/params`)

**Shape:** `[N, P]` where P=14

**Type:** float32

**Range:** `[0, 1]` (Sobol unit cube)

**Interpretation:** Normalized operator parameters sampled via Sobol sequences for optimal space-filling coverage.

### Features

All feature datasets use **float32** dtype for efficiency.

**Feature ranges vary by type:**
- Spatial statistics: normalized or standardized
- Spectral features: log-scale or normalized power
- Information measures: bits or nats

## Reading Examples

### Python (h5py)

```python
import h5py
import numpy as np

with h5py.File("datasets/cno_50k_3channel_dev.h5", "r") as f:
    # Read inputs with realizations
    inputs = f["/inputs/fields"][:]
    print(f"Inputs shape: {inputs.shape}")  # [N, M, C, H, W]
    print(f"Example: {inputs.shape} = {inputs.shape[0]} samples × "
          f"{inputs.shape[1]} realizations × {inputs.shape[2]} channels × "
          f"{inputs.shape[3]}×{inputs.shape[4]} grid")

    # Read feature families
    arch_features = f["/features/architecture/aggregated/features"][:]
    print(f"Architecture features: {arch_features.shape}")  # [N, 23]

    init_features = f["/features/initial/aggregated/features"][:]
    print(f"Initial features: {init_features.shape}")  # [N, 38]

    temporal_features = f["/features/temporal/features"][:]
    print(f"Temporal features: {temporal_features.shape}")  # [N, T, 345]

    # Read parameters
    params = f["/parameters/params"][:]
    print(f"Parameters: {params.shape}")  # [N, 14]

    # Read metadata
    ic_types = f["/metadata/ic_types"][:]
    evolution_policies = f["/metadata/evolution_policies"][:]
    print(f"IC types: {ic_types[:5]}")
    print(f"Evolution policies: {evolution_policies[:5]}")
```

### VQ-VAE Feature Loading (v3.1)

```python
import h5py
import torch

def load_features_for_vqvae(h5_path):
    """Load features for VQ-VAE training.

    VQ-VAE trains on INITIAL + TEMPORAL features.
    ARCHITECTURE features are excluded (MNO already knows operator parameters θ).
    """
    with h5py.File(h5_path, "r") as f:
        # Load INITIAL features (aggregated over realizations)
        initial = f["/features/initial/aggregated/features"][:]

        # Load TEMPORAL features (per-timestep)
        temporal = f["/features/temporal/features"][:]

        # Convert to torch tensors
        initial_tensor = torch.from_numpy(initial)
        temporal_tensor = torch.from_numpy(temporal)

    return {
        "initial": initial_tensor,  # [N, D_init]
        "temporal": temporal_tensor  # [N, T, D_temporal]
    }
```

### Inspecting Dataset Structure

```python
import h5py

def print_h5_structure(h5_path):
    """Print complete HDF5 structure."""
    with h5py.File(h5_path, "r") as f:
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  {name}/ (group)")

        f.visititems(print_item)

# Example usage
print_h5_structure("datasets/cno_50k_3channel_dev.h5")
```

## Compression

Default HDF5 settings:

| Setting | Value | Notes |
|---------|-------|-------|
| Compression | gzip | Level 4 |
| Chunk size | 100 | Samples per chunk |
| Dtype | float32 | All features |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.2.0 | 2026-01-30 | Per-channel independent IC generation; new ic_types format: `ch0:type\|ch1:type\|ch2:type` |
| 3.1.0 | 2026-01-29 | 3-channel support; explicit realization dimension in inputs; ARCHITECTURE and INITIAL stored in /features/ |
| 3.0.0 | 2026-01-18 | Removed SUMMARY features; enhanced TEMPORAL to ~328D; 14D parameter space |
| 2.0.0 | 2026-01-12 | Two-family structure (TEMPORAL, SUMMARY) with enhanced features |
| 1.0.0 | 2025-12 | Initial implementation |

## Migration Notes

### v3.2 Changes (2026-01-30)

**Key Changes:**
- **Per-channel independent IC generation** now supported via `method: "per_channel"`
- **ic_types format** changed for per-channel datasets:
  - Old: `"gaussian_random_field"` (single type for all channels)
  - New: `"ch0:grf|ch1:struct|ch2:mgrf"` (different type per channel)
- **Backward compatible:** Single IC type format still supported for `method: "sampled"` or other methods
- **No schema changes:** HDF5 structure remains the same, only ic_types string format differs

**Migration:**
- **Existing datasets:** No migration needed - they use single IC type format and work as-is
- **VQ-VAE training:** No changes needed - VQ-VAE only uses field tensors, not ic_types metadata
- **Analysis scripts:** May need updates if parsing ic_types strings

**Configuration example:**
```yaml
simulation:
  input_generation:
    method: "per_channel"  # Enable per-channel ICs
    channel_configs:
      channel_0:
        ic_type_weights: {gaussian_random_field: 0.5, localized: 0.5}
        gaussian_random_field: {length_scale: 0.05}
      channel_1:
        ic_type_weights: {structured: 1.0}
      channel_2:
        ic_type_weights: {gaussian_random_field: 1.0}
```

### v3.1 Changes (2026-01-29)

**Key Changes:**
- **inputs/fields shape** changed from `[N, C, H, W]` to `[N, M, C, H, W]`
  - Explicit realization dimension M (typically 3)
  - Enables proper stochastic sampling and uncertainty quantification
- **Multi-channel support:** C can now be 1-3 (grayscale or RGB-like)
- **ARCHITECTURE features** now stored in `/features/architecture/aggregated/features [N, D_arch]`
- **INITIAL features** now stored in `/features/initial/aggregated/features [N, D_init]`
- **TEMPORAL features** expanded to ~345D per-timestep (from ~328D in v3.0)

**Migration from v3.0:**
1. **inputs/fields:** Add realization dimension if missing (expand [N, C, H, W] → [N, 1, C, H, W])
2. **features/architecture:** Extract from parameters if needed
3. **features/initial:** Compute from inputs/fields with realization aggregation

### From v2.x to v3.1

If you have v2.x datasets:
1. **Regenerate datasets** using v3.1 feature extraction (recommended)
2. **Convert features** by extracting all three families (ARCHITECTURE, INITIAL, TEMPORAL)
3. **Note:** SUMMARY features removed - compute trajectory-level aggregates post-hoc if needed

## Example: Full Dataset Inspection

```python
import h5py
import numpy as np

with h5py.File("datasets/cno_50k_3channel_dev.h5", "r") as f:
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    # Core dimensions
    N = f["/inputs/fields"].shape[0]
    M = f["/inputs/fields"].shape[1]
    C = f["/inputs/fields"].shape[2]
    H, W = f["/inputs/fields"].shape[3:5]
    T = f["/features/temporal/features"].shape[1]

    print(f"\nSamples: {N}")
    print(f"Realizations per sample: {M}")
    print(f"Channels: {C}")
    print(f"Grid size: {H}×{W}")
    print(f"Timesteps: {T}")

    # Feature dimensions
    D_arch = f["/features/architecture/aggregated/features"].shape[1]
    D_init = f["/features/initial/aggregated/features"].shape[1]
    D_temporal = f["/features/temporal/features"].shape[2]

    print(f"\nFeature dimensions:")
    print(f"  ARCHITECTURE: {D_arch}D")
    print(f"  INITIAL: {D_init}D")
    print(f"  TEMPORAL: {D_temporal}D per timestep")

    # Memory footprint
    inputs_size = np.prod(f["/inputs/fields"].shape) * 4 / 1e9  # GB
    temporal_size = np.prod(f["/features/temporal/features"].shape) * 4 / 1e9

    print(f"\nStorage (uncompressed):")
    print(f"  Inputs: {inputs_size:.2f} GB")
    print(f"  Temporal features: {temporal_size:.2f} GB")
    print(f"  Total: {inputs_size + temporal_size:.2f} GB")
```

## Notes

- **Realization dimension:** The explicit M dimension in inputs allows proper handling of stochastic dynamics
- **Feature aggregation:** INITIAL features aggregate over M realizations, while TEMPORAL features are per-realization or averaged
- **ARCHITECTURE vs parameters:** `/parameters/params` stores raw Sobol samples; `/features/architecture/` stores derived architectural features
- **Empty outputs:** The `/outputs/` group is present but typically empty (trajectories not stored to save space)
- **Metadata fields:** Evolution policies, IC types, noise regimes, and grid sizes provide per-sample metadata for analysis
