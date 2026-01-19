# HDF5 Dataset Layout Reference

This document describes the complete HDF5 schema for Spinlock datasets, including the feature storage structure used by the VQ-VAE tokenization pipeline.

**Last Updated:** 2026-01-18 (v3.0 - SUMMARY features removed)

## Overview

Spinlock datasets use HDF5 format with two main sections:

1. **Core Dataset** (`/metadata/`, `/parameters/`, `/inputs/`, `/outputs/`) - Operator parameters and rollout data
2. **Features** (`/features/`) - Extracted behavioral features (TEMPORAL family only in v3.0+)

## Complete Schema

```
dataset.h5
├── metadata/
│   ├── config              # JSON - full generation config
│   ├── timestamp           # ISO timestamp
│   └── version             # Schema version
│
├── parameters/
│   ├── params [N, P]       # float32 - Sobol parameter vectors (P=14 in v3.0)
│   └── @dimension_names    # Attribute: parameter dimension names
│
├── inputs/
│   └── fields [N, C, H, W] # float32 - Initial conditions
│
├── outputs/                # (Only if store_trajectories=true)
│   └── trajectories [N, M, T, C, H, W]  # float32 - Rollout data
│
└── features/
    ├── @family_versions    # {"temporal": "3.0.0"}
    ├── @extraction_timestamp
    ├── @extraction_config
    │
    └── temporal/           # TEMPORAL family (per-timestep only)
        ├── @version
        ├── @feature_registry   # JSON {category: {name: index}}
        └── features [N, T, D_temporal]  # float32 - D_temporal ≈ 328
```

**v3.0 Changes:**
- Removed `/features/summary/` entirely (incompatible with online prediction)
- TEMPORAL features enhanced from ~63D to ~328D per-timestep
- All features now per-timestep computable for NOA online operation

## Dimensions

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| N | Number of samples (operators) | 1,000 - 100,000 |
| M | Number of realizations | 3 - 10 |
| T | Number of timesteps | 100 - 500 |
| C | Number of channels | 1 |
| H, W | Grid height/width | 128 |
| P | Parameter dimension (v3.0+) | 14 |
| D_temporal | TEMPORAL feature dim (v3.0+) | ~328 |

## Feature Families

### TEMPORAL Family (`/features/temporal/`)

Per-timestep time series preserving full temporal resolution. **This is the only feature family stored in `/features/` as of v3.0.**

**Shape:** `[N, T, D_temporal]` where D_temporal ≈ 328

**Contents (v3.0 Enhanced):**
- **Spatial features (~105D):** Per-channel statistics, gradients, Laplacian, histogram features
- **Spectral features (~93D):** Multi-scale FFT, power spectrum, frequency bands, spectral entropy
- **Cross-channel features (~10D):** Pairwise correlations, covariance eigenvalues
- **Enhanced temporal dynamics (~120D):** Windowed statistics, stability metrics, phase space features, autocorrelation

**Use Case:** Working memory analysis, temporal pattern detection, trajectory classification, online NOA predictions.

**Key Property:** All features are **per-timestep computable** (no lookahead required), enabling online operation.

### ~~SUMMARY Family~~ [REMOVED in v3.0]

Aggregated trajectory-level features (causality, invariant drift, operator sensitivity) were removed in v3.0 because they require complete trajectories and are incompatible with online prediction.

**Archived code:** `src/spinlock/features/temporal_old_v2/summary/`

**Migration:** If you need trajectory-level aggregates, compute them post-hoc from TEMPORAL features or use v2.x datasets.

## Reading Examples

### Python (h5py)

```python
import h5py
import numpy as np

with h5py.File("dataset.h5", "r") as f:
    # Check available feature families (v3.0: only 'temporal')
    families = list(f["/features"].keys())
    print(f"Available families: {families}")  # ['temporal']

    # Read TEMPORAL per-timestep features
    temporal = f["/features/temporal/features"][:]
    print(f"TEMPORAL shape: {temporal.shape}")  # [N, T, ~328]

    # Read parameter vectors (14D Sobol space)
    params = f["/parameters/params"][:]
    print(f"Parameters shape: {params.shape}")  # [N, 14]

    # Read feature registry for interpretability
    registry_json = f["/features/temporal"].attrs["feature_registry"]
    import json
    registry = json.loads(registry_json)
    # registry = {category: {feature_name: index}}

    # Check metadata
    version = f["/metadata/version"][()]
    print(f"Dataset version: {version}")
```

### VQ-VAE Feature Loading (v3.0)

```python
# VQ-VAE loads TEMPORAL features directly
features_path = "/features/temporal/features"

# Load INITIAL features (computed inline from /inputs/fields)
# Load TEMPORAL features from HDF5
# ARCHITECTURE features excluded from VQ-VAE training
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
| 3.0.0 | 2026-01-18 | Removed SUMMARY features; enhanced TEMPORAL to ~328D; 14D parameter space |
| 2.0.0 | 2026-01-12 | Two-family structure (TEMPORAL, SUMMARY) with enhanced features |
| 1.0.0 | 2025-12 | Initial implementation |

## Migration Notes

### v3.0 Changes (2026-01-18)

**Breaking Changes:**
- **Removed `/features/summary/`** entirely (per_trajectory, aggregated, learned, operator_sensitivity_inline)
- **Enhanced TEMPORAL** from ~63D to ~328D per-timestep
- **Parameter space** expanded from 12D to 14D (added dt and alpha)
- **Feature families** reduced from 4 conceptual families to 3

**Rationale:**
SUMMARY features (causality, invariant drift, operator sensitivity, nonlinear dynamics) required complete trajectories for aggregation, making them incompatible with online NOA predictions. All features must now be computable per-timestep.

**Current Structure (v3.0):**
- **INITIAL** features → Computed inline from `/inputs/fields` (not stored in `/features/`)
- **ARCHITECTURE** features → Stored in `/parameters/params [N, 14]` (Sobol unit cube)
- **TEMPORAL** features → `/features/temporal/features [N, T, ~328]` (per-timestep only)

**Archived Code:**
Old SUMMARY implementation available at `src/spinlock/features/temporal_old_v2/summary/` for reference.

### From v2.x to v3.0

If you have v2.x datasets with SUMMARY features:
1. **Regenerate datasets** using v3.0 feature extraction (recommended)
2. **Convert features** by extracting TEMPORAL features and discarding SUMMARY
3. **Use v2.x datasets** if you need trajectory-level aggregates (not compatible with v3.0 models)

The conceptual 3-family framework (INITIAL, ARCHITECTURE, TEMPORAL) is described in [Feature Families README](README.md).
