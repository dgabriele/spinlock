# VQ-VAE MVP Implementation Summary

## Overview

Implemented MVP system for training VQ-VAE on SD (State Descriptor) features using existing infrastructure from unisim.system, following DRY principles.

## Completed Tasks

### 1. Category Discovery Script ✅

**File**: `scripts/discover_sd_categories.py`

Creates semantic category-to-indices mapping for GroupedFeatureExtractor by organizing aggregated SDF v2.0 features.

**Features**:
- Loads feature registry from HDF5 dataset
- Groups trajectory features by category (temporal, causality, invariant_drift, operator_sensitivity, nonlinear, integrated)
- Maps each category's 3 aggregations (mean, std, cv) to indices
- Validates total feature count matches dataset
- Outputs JSON file with `group_indices` mapping

**Usage**:
```bash
poetry run python scripts/discover_sd_categories.py \
  --dataset datasets/test_1k_inline_features.h5 \
  --feature-path "/features/sdf/aggregated/features" \
  --output configs/vqvae/sd_category_mapping.json
```

### 2. Enhanced train-vqvae CLI ✅

**File**: `src/spinlock/cli/train_vqvae.py`

Extended existing CLI to support loading pre-computed category mappings from JSON files.

**Added**:
- New `_load_category_mapping()` method to load and validate JSON category files
- Modified category assignment logic to support three modes:
  1. Auto-discovery via clustering (`category_assignment: "auto"`)
  2. **Load from JSON file** (`category_mapping_file: "path/to/mapping.json"`) ← NEW
  3. Resume from checkpoint (`resume_from: "path/to/checkpoint.pt"`)

**Benefits**:
- DRY: Reuses existing VQ-VAE infrastructure
- Fast: Skips expensive clustering step
- Reproducible: Same categories across runs
- Interpretable: Semantic categories match SDF v2.0 design

### 3. VQ-VAE Training Config ✅

**File**: `configs/vqvae/sd_features_1k.yaml`

Production-ready config for training categorical hierarchical VQ-VAE on SD features.

**Configuration**:
```yaml
# Dataset
input_path: "datasets/test_1k_inline_features.h5"
feature_type: "aggregated"  # 120 trajectory × 3 aggregations
feature_family: "sdf"

# Category mapping (from discover_sd_categories.py)
category_mapping_file: "configs/vqvae/sd_category_mapping.json"

# VQ-VAE architecture (GroupedFeatureExtractor from unisim)
group_embedding_dim: 64
group_hidden_dim: 128

# Hierarchical levels (3-level proven architecture)
factors:
  - {latent_dim: 32, num_tokens: 128}  # Coarse
  - {latent_dim: 16, num_tokens: 256}  # Medium
  - {latent_dim: 8, num_tokens: 512}   # Fine

# Training (from unisim agent_training_v1)
epochs: 410
batch_size: 1024
learning_rate: 0.0007
commitment_cost: 0.45
```

**Design Decisions**:
- **Encoder**: GroupedFeatureExtractor (already ported, proven on unisim U tokens)
- **Hierarchy**: 3 levels (coarse → medium → fine granularity)
- **Hyperparameters**: Taken from unisim's production config (proven to work)
- **Loss weights**: Balanced for SD features (commitment=0.45, ortho=0.1, info=0.1)

## Next Steps

Once 1K dataset generation completes (~45 minutes remaining):

1. **Test category discovery**:
   ```bash
   poetry run python scripts/discover_sd_categories.py \
     --dataset datasets/test_1k_inline_features.h5 \
     --feature-path "/features/sdf/aggregated/features" \
     --output configs/vqvae/sd_category_mapping.json
   ```

2. **Train VQ-VAE**:
   ```bash
   poetry run spinlock train-vqvae \
     --config configs/vqvae/sd_features_1k.yaml \
     --verbose
   ```

3. **Validate tokenization**:
   - Reconstruction quality (MSE < threshold)
   - Codebook utilization (>80% per quantizer)
   - Orthogonality (low correlation between categories)
   - Informativeness (partial decoders work well)

## Future: Multi-Family Extension

After SD training validates, add multi-family system for joint training of:
- SD (State Descriptor) features ← CURRENT MVP
- Trajectory features (raw temporal evolution)
- IC (Initial Condition) embeddings
- Parameter embeddings
- Time Series (derivatives, rates of change)
- Functors (operations between SD embeddings)

**Architecture**:
- Separate VQ-VAE per family
- Named encoder registry
- Adaptive loss weighting
- Joint optimization

See plan file: `.claude/plans/composed-singing-scott.md`

## Files Modified/Created

### Created:
1. `scripts/discover_sd_categories.py` - Category discovery script
2. `configs/vqvae/sd_features_1k.yaml` - VQ-VAE training config
3. `docs/vqvae-mvp-implementation.md` - This file

### Modified:
1. `src/spinlock/cli/train_vqvae.py` - Added JSON category mapping support

## Key Design Principles

1. **DRY**: Reuse existing CategoricalHierarchicalVQVAE from unisim
2. **MVP First**: Get SD features working before multi-family
3. **Proven Architectures**: Use unisim's successful encoder and hyperparameters
4. **Modular**: Easy to extend to multi-family later
5. **Semantic Categories**: Manual grouping matches SDF v2.0 design

## Background Context

- **GroupedFeatureExtractor**: The "unisim_hierarchical" encoder - already ported
- **CategoricalHierarchicalVQVAE**: Production-ready from unisim.system
- **SD Features**: 120 trajectory features × 3 aggregations = 360 total features
- **Categories**: temporal, causality, invariant_drift, operator_sensitivity, nonlinear, integrated
- **Dataset**: 1K samples, 250 timesteps, 3 realizations, 128×128 grids

## Performance Expectations

**Training time** (1K samples):
- Dataset size: ~10GB (feature-only mode, no trajectories)
- Training time: ~2-3 hours @ 410 epochs with batch_size=1024
- GPU memory: ~4GB (fits on 8GB GPU)

**Quality metrics** (from unisim):
- Reconstruction MSE: <0.1
- Codebook utilization: >80%
- Orthogonality: <0.15
- Informativeness loss: <0.5

## References

- Plan file: `.claude/plans/composed-singing-scott.md`
- Unisim reference: `unisim.system.models.categorical_vqvae`
- Unisim U token training: Successful encoder for CA behavior features
- SDF v2.0 spec: 174 features across 8 categories
