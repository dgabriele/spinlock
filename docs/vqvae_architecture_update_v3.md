# VQ-VAE Architecture Update for v3.0.0

## Overview

This document describes the VQ-VAE architecture updates required for the enhanced temporal feature extraction (v3.0.0), which transitions from a mixed trajectory-level/per-timestep architecture to a per-timestep-only architecture.

## Architecture Comparison

### Old Architecture (v2.x)

**Feature Families:**
- **INITIAL** (42D): 14D manual + 28D CNN
- **SUMMARY** (330D): Trajectory-level aggregated features
  - Encoded to 128D via MLP
- **TEMPORAL** (63D): Per-timestep features
  - Encoded to 128D via TemporalCNN

**Total Input:** 42D + 330D + 63D = 435D
**Total Encoded:** 42D + 128D + 128D = 298D (before VQ)

### New Architecture (v3.0.0)

**Feature Families:**
- **INITIAL** (42D): 14D manual + 28D CNN (unchanged)
- **TEMPORAL** (193D): Enhanced per-timestep features
  - Encoded to 256D via MLP

**Total Input:** 42D + 193D = 235D
**Total Encoded:** 42D + 256D = 298D (before VQ)

## Key Changes

### 1. Removed SUMMARY Family

The SUMMARY family contained trajectory-level features that required complete trajectories for extraction:
- Temporal dynamics (growth rates, oscillations, Lyapunov, autocorr)
- Causality (lag correlations, time irreversibility, spatial flow)
- Invariant drift (L1/L2/Linf/entropy/TV norm drift)
- Operator sensitivity (Lipschitz, gain curves, linearity)
- Nonlinear (RQA, correlation dimension, permutation entropy)

**Why removed:** Incompatible with online perturbation-based NOA where episodes extract features at each timestep without access to complete trajectories.

### 2. Enhanced TEMPORAL Family (63D → 193D)

The TEMPORAL family was expanded from 63D to 193D to capture equivalent or superior information quality compared to the removed SUMMARY features:

**Old TEMPORAL (63D):**
- Spatial: 24D
- Spectral: 27D
- Cross-channel: 12D

**New TEMPORAL (193D):**
- Spatial: 24D (unchanged)
- Spectral: 27D (unchanged)
- Cross-channel: 12D (unchanged)
- **Enhanced temporal: 130D** (new)
  - Instantaneous dynamics: 22D (energy, dissipation, spectral, structure, stats)
  - Local temporal: 28D (autocorr, trends, windowed stats, oscillations, growth)
  - Local stability: 24D (Lipschitz, stability, divergence, regularity)
  - Phase space geometry: 26D (flow, vorticity, strain, topology, manifold)
  - Multi-scale temporal: 30D (hierarchical averaging, cross-scale, persistence)

**Key insight:** These enhanced temporal features capture equivalent information to trajectory-level features using only local temporal context (windowed history buffers).

### 3. Increased TEMPORAL Encoder Capacity

To accommodate the larger TEMPORAL feature space (63D → 193D), the encoder capacity was increased:

**Old:**
```yaml
temporal:
  encoder: TemporalCNNEncoder
  encoder_params:
    embedding_dim: 128
```

**New:**
```yaml
temporal:
  encoder: MLPEncoder
  encoder_params:
    hidden_dims: [384, 256]
    output_dim: 256
```

**Rationale:**
- Larger hidden layers (384) to process richer features
- Higher output dimension (256 vs 128) to preserve more information
- Switched from CNN to MLP for simplicity (all features are already per-timestep)

## Code Changes Required

### Minimal Changes

The VQ-VAE architecture code (`categorical_vqvae.py`) is **already generic** and requires **no changes**. It accepts any input dimension and feature groupings through configuration.

### Configuration Changes Only

**New config file:** `configs/vqvae/enhanced_temporal.yaml`

Key changes:
1. Removed `summary` family entirely
2. Updated `temporal` encoder specification
3. Adjusted `dataset_path` to point to regenerated datasets
4. Updated comments and documentation

## Dataset Regeneration

Before training the VQ-VAE with the new architecture, feature datasets must be regenerated:

### Scripts Created

1. **`scripts/regenerate_cno_features.py`**
   - Regenerates CNO reference features (100K samples)
   - Output: `datasets/cno_features_100k_enhanced.h5`
   - Features: [N, T, 193]

2. **`scripts/regenerate_mno_features.py`**
   - Regenerates MNO rollout features (100K samples)
   - Output: `datasets/mno_features_100k_enhanced.h5`
   - Features: [N, T, 193]

3. **`scripts/validate_feature_extraction.py`**
   - Validates feature extraction pipeline
   - Checks dimensions, NaN/Inf, value ranges, variability
   - Generates diagnostic plots

### Regeneration Process

```bash
# 1. Validate feature extraction
python scripts/validate_feature_extraction.py

# 2. Regenerate CNO reference features
python scripts/regenerate_cno_features.py \
    --input datasets/cno_100k_stratified.h5 \
    --output datasets/cno_features_100k_enhanced.h5 \
    --batch-size 32 \
    --device cuda

# 3. Regenerate MNO features
python scripts/regenerate_mno_features.py \
    --input datasets/mno_rollouts_100k.h5 \
    --output datasets/mno_features_100k_enhanced.h5 \
    --batch-size 32 \
    --device cuda
```

## Training with New Architecture

### Training Command

```bash
spinlock train-vqvae \
    --config configs/vqvae/enhanced_temporal.yaml \
    --device cuda \
    --verbose
```

### Expected Performance

**Target Metrics:**
- Reconstruction loss: < 0.02 (match or beat old 0.018)
- Codebook utilization: > 60% per level
- Category separation: orthogonality < 0.15
- Training time: ~3-4 hours on RTX 3060 Ti

**Quality Validation:**
- PCA variance ≥ old trajectory-level features
- Feature extraction latency: < 10ms per timestep (CPU)
- VQ-VAE tokenization maintains semantic coherence

### Checkpoint Location

Checkpoints saved to: `checkpoints/vqvae/enhanced_temporal_v3/`

Files:
- `best_model.pt`: Best model checkpoint
- `normalization_stats.npz`: Per-category normalization stats
- `training_history.json`: Training metrics history
- `config.yaml`: Resolved configuration

## Backward Compatibility

### Legacy Support

For backward compatibility with existing code that references the old architecture:

1. **Legacy aliases** are provided in `src/spinlock/features/temporal/__init__.py`:
   ```python
   SummaryExtractor = TemporalFeatureOrchestrator
   SummaryConfig = TemporalFeatureConfig
   # etc.
   ```

2. **Old checkpoints** are incompatible with the new architecture (different dimensions)
   - Mark old checkpoints as `v2.x` and archive
   - Retrain from scratch with v3.0.0

3. **Config migration:** Old configs referencing "summary" family should be updated to "temporal" only

### Migration Checklist

- [ ] Regenerate feature datasets with enhanced temporal architecture
- [ ] Update VQ-VAE config to use `enhanced_temporal.yaml` template
- [ ] Retrain VQ-VAE from scratch
- [ ] Validate reconstruction quality (< 0.02)
- [ ] Update episode code to use new checkpoints
- [ ] Archive old v2.x checkpoints with clear naming

## References

- **Plan:** `/home/daniel/.claude/plans/valiant-giggling-penguin.md`
- **Migration guide:** `docs/feature_architecture_migration.md` (to be created in Phase 7)
- **Feature extraction:** `src/spinlock/features/temporal/`
- **VQ-VAE code:** `src/spinlock/encoding/categorical_vqvae.py`
- **Training CLI:** `src/spinlock/cli/train_vqvae.py`
