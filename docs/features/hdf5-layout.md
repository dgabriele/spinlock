# HDF5 Dataset Layout Reference

This document describes the complete HDF5 schema for Spinlock datasets, including the feature storage structure used by the VQ-VAE tokenization pipeline.

## Overview

Spinlock datasets use HDF5 format with two main sections:

1. **Core Dataset** (`/metadata/`, `/parameters/`, `/inputs/`, `/outputs/`) - Operator parameters and rollout data
2. **Features** (`/features/`) - Extracted behavioral features (TEMPORAL and SUMMARY families)

## Complete Schema

```
dataset.h5
├── metadata/
│   ├── config              # JSON - full generation config
│   ├── timestamp           # ISO timestamp
│   └── version             # Schema version
│
├── parameters/
│   ├── vectors [N, P]      # float32 - Sobol parameter vectors
│   └── @dimension_names    # Attribute: parameter dimension names
│
├── inputs/
│   └── fields [N, C, H, W] # float32 - Initial conditions
│
├── outputs/                # (Only if store_trajectories=true)
│   └── trajectories [N, M, T, C, H, W]  # float32 - Rollout data
│
└── features/
    ├── @family_versions    # {"temporal": "1.0.0", "summary": "1.0.0"}
    ├── @extraction_timestamp
    ├── @extraction_config
    │
    ├── temporal/           # TEMPORAL family (per-timestep)
    │   ├── @version
    │   └── features [N, T, D_temporal]  # float32
    │
    └── summary/            # SUMMARY family (aggregated)
        ├── @version
        ├── @feature_registry   # JSON {category: {name: index}}
        ├── @num_features
        │
        ├── per_trajectory/
        │   └── features [N, M, D_traj]  # float32
        │
        ├── aggregated/
        │   ├── features [N, D_final]    # float32
        │   └── metadata/
        │       └── extraction_time [N]  # float64
        │
        ├── learned/        # Optional (U-AFNO latents)
        │   ├── @version
        │   ├── @description
        │   └── features [N, D_learned]  # float32
        │
        └── operator_sensitivity_inline/  # Sparse inline features
            └── {feature_name} [N]       # float32
```

## Dimensions

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| N | Number of samples (operators) | 1,000 - 100,000 |
| M | Number of realizations | 3 - 10 |
| T | Number of timesteps | 100 - 500 |
| C | Number of channels | 1 |
| H, W | Grid height/width | 128 |
| P | Parameter dimension | 14 |
| D_temporal | TEMPORAL feature dim | ~120 |
| D_traj | Per-trajectory feature dim | ~120 |
| D_final | Aggregated feature dim | ~360 |
| D_learned | Learned feature dim | 64 - 256 |

## Feature Families

### TEMPORAL Family (`/features/temporal/`)

Per-timestep time series preserving full temporal resolution.

**Shape:** `[N, T, D_temporal]`

**Contents:**
- Spatial statistics per timestep (gradients, anisotropy)
- Spectral features per timestep (FFT power, dominant frequencies)
- Cross-channel correlations per timestep

**Use Case:** Working memory analysis, temporal pattern detection, trajectory classification.

### SUMMARY Family (`/features/summary/`)

Aggregated scalar statistics collapsed across time and realizations.

**Structure:**

| Dataset | Shape | Description |
|---------|-------|-------------|
| `per_trajectory/features` | [N, M, D_traj] | Per-realization aggregates |
| `aggregated/features` | [N, D_final] | Final cross-realization aggregates |
| `learned/features` | [N, D_learned] | U-AFNO latent embeddings (optional) |

**Contents:**
- Temporal dynamics (autocorrelation, stationarity, regimes)
- Causality metrics (information flow, Granger causality)
- Invariant drift (long-term behavioral evolution)

**Use Case:** VQ-VAE tokenization, operator clustering, behavioral classification.

## Reading Examples

### Python (h5py)

```python
import h5py
import numpy as np

with h5py.File("dataset.h5", "r") as f:
    # Check available feature families
    families = list(f["/features"].keys())
    print(f"Available families: {families}")

    # Read SUMMARY aggregated features (primary VQ-VAE input)
    if "summary" in families:
        features = f["/features/summary/aggregated/features"][:]
        print(f"SUMMARY shape: {features.shape}")  # [N, D_final]

    # Read TEMPORAL per-timestep features
    if "temporal" in families:
        temporal = f["/features/temporal/features"][:]
        print(f"TEMPORAL shape: {temporal.shape}")  # [N, T, D]

    # Read feature registry for interpretability
    registry_json = f["/features/summary"].attrs["feature_registry"]
    import json
    registry = json.loads(registry_json)
    # registry = {category: {feature_name: index}}
```

### Multi-Family Loading (VQ-VAE)

```python
# VQ-VAE loads features using this path pattern:
# /features/{family}/{type}/features

# For SUMMARY aggregated:
features_path = "/features/summary/aggregated/features"

# For TEMPORAL:
features_path = "/features/temporal/features"
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
| 1.0.0 | 2026-01 | Initial two-family structure (TEMPORAL, SUMMARY) |

## Migration Notes

### From Legacy 4-Family Structure

Earlier versions conceptually used 4 families (INITIAL, ARCHITECTURE, SUMMARY, TEMPORAL). The current implementation consolidates these into 2 HDF5 families:

- **INITIAL** features → Now part of SUMMARY (spatial/spectral characteristics of initial conditions)
- **ARCHITECTURE** features → Stored in `/parameters/vectors` (not in `/features/`)
- **SUMMARY** features → `/features/summary/aggregated/features`
- **TEMPORAL** features → `/features/temporal/features`

The conceptual 4-family framework remains valid for understanding feature semantics; see [Feature Families README](README.md) for details.
