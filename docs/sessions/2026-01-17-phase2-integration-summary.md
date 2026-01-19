# Phase 2 Integration Summary

## Work Completed

### 1. Feature Extraction Pipeline Fix ✅

**Problem Identified**: The original `SimpleFeatureExtractor` created incompatible features (100-200D spatial/spectral) that didn't match what the VQ-VAE was trained on.

**Solution Implemented**:
- Created `VQVAEFeatureExtractor` (`src/spinlock/noa/vqvae_feature_extraction.py`) that extracts the exact features used during VQ-VAE training
- Uses `SummaryExtractor.extract_per_timestep()` to get per-timestep features
- Properly configured to extract:
  - 24 spatial features (statistics, gradients, percentiles)
  - 27 spectral features (FFT power, frequency analysis)
  - 12 cross-channel features (correlation, mutual information)
  - **Total: 63D for 2-channel data, 62D for 1-channel data**

**Key Insight**: The VQ-VAE's TEMPORAL encoder was trained on per-timestep features from the `extract_per_timestep()` method, not custom features.

### 2. Model Loading Infrastructure ✅

**Created**: `src/spinlock/noa/validation_utils.py`
- `load_mno_checkpoint()`: Properly extracts model config from nested checkpoint structure
- `load_vqvae_checkpoint()`: Constructs `CategoricalVQVAEConfig` from checkpoint, strips "vqvae." prefix from state dict
- `sample_initial_condition()`: Supports multiple IC types (smooth_random, random, blob, zero)
- `get_vqvae_num_categories_and_levels()`: Extracts VQ-VAE architecture info

**Fixed Issues**:
- MNO config extraction from `checkpoint['config']['model']`
- VQ-VAE state dict prefix stripping ("vqvae." → "")
- Proper config object construction for CategoricalHierarchicalVQVAE

### 3. File Organization ✅

**Moved Files to Correct Locations**:
- `noa/*.py` → `src/spinlock/noa/*.py`
- All Phase 2 infrastructure now in proper package structure

### 4. End-to-End Integration Test ✅

**Created**: `test_episode_integration.py`
- Loads MNO and VQ-VAE checkpoints successfully
- Creates EpisodeRunner with proper feature extraction
- Samples ICs and perturbations correctly

**Test Results**:
- ✅ MNO loads: 226M parameters, 1→1 channels
- ✅ VQ-VAE loads: 1.1M parameters, 5 categories, 106D input
- ✅ Feature extraction works: produces 62D features for 1-channel MNO
- ⚠️ **Architecture mismatch discovered** (see below)

## Critical Issue Discovered: VQ-VAE Architecture Mismatch

### Problem

The current VQ-VAE checkpoint (`mno_10k_with_reference_reg`) was trained on **106D features** combining three families:

1. **INITIAL features** (~14D): Manual features extracted from raw ICs
2. **SUMMARY features** (~30D): Per-trajectory aggregated statistics (mean, std, CV of temporal features)
3. **TEMPORAL features** (62D): Per-timestep features (spatial + spectral + cross-channel)

**For autonomous episodes**, we can only extract **TEMPORAL features** (62D) since we have single MNO states at each timestep, not:
- Full trajectories (needed for SUMMARY aggregation)
- Raw ICs (needed for INITIAL features)

### Why This Matters

The VQ-VAE's `GroupedFeatureExtractor` has:
```python
group_indices = {
    'cluster_2': [45, 47, 50, 53, 58, 60, ...],  # Indices into 106D space
    'cluster_3': [0, 2, 6, 8, 10, ...],
    ...
}
```

When we pass 62D features, indexing fails: `IndexError: index 63 is out of bounds for dimension 0 with size 62`

### Solution Options

#### Option 1: Train TEMPORAL-Only VQ-VAE (Recommended)

**Pros**:
- Clean architecture for per-timestep tokenization
- No distribution mismatch
- Smaller model (62D input vs 106D)

**Cons**:
- Requires retraining VQ-VAE
- Need to regenerate features dataset (TEMPORAL only)

**Steps**:
1. Extract only `features/temporal/features` from `datasets/mno_features_100k.h5`
2. Train new VQ-VAE on 62D per-timestep features
3. Update validation scripts to use new checkpoint

#### Option 2: Multi-Family VQ-VAE with Single-Family Inference

**Pros**:
- Uses existing checkpoint
- Could support multiple tokenization modes

**Cons**:
- Requires modifying CategoricalHierarchicalVQVAE architecture
- Complex: need to remap group_indices for TEMPORAL subset
- May have distribution mismatch issues

**Steps**:
1. Add `forward_temporal_only()` method to VQ-VAE
2. Extract TEMPORAL group_indices from full group_indices
3. Create subset feature extractor for TEMPORAL features only

#### Option 3: Use Different Checkpoint

**Check if there's an existing TEMPORAL-only VQ-VAE checkpoint**:
```bash
# Look for checkpoints with smaller input_dim
find checkpoints/vqvae -name "best_model.pt" -exec \
  python -c "import torch; ckpt = torch.load('{}', weights_only=False); \
  print('{}: input_dim={}'.format('{}', ckpt['model_config']['input_dim']))" \;
```

### Immediate Next Steps

1. **Decision**: Which solution to pursue?
2. **If Option 1**: Create script to extract TEMPORAL features and retrain VQ-VAE
3. **If Option 2**: Modify VQ-VAE architecture for subset inference
4. **If Option 3**: Find appropriate checkpoint

## Validation Experiments Status

All 4 validation scripts are ready to run once VQ-VAE issue is resolved:

1. ✅ **01_perturbation_response_divergence.py**: Tests if different perturbations produce different token sequences
2. ✅ **02_token_regime_clustering.py**: Validates token-based behavioral clustering
3. ✅ **03_early_stopping_efficiency.py**: Measures computational savings from early stopping
4. ✅ **04_reproducibility.py**: Verifies deterministic tokenization

**All scripts use**:
- Proper model loading from `validation_utils.py`
- Real EpisodeRunner with VQVAEFeatureExtractor
- Actual MNO + VQ-VAE checkpoints

## Files Modified/Created

### Created:
- `src/spinlock/noa/vqvae_feature_extraction.py` - Feature extraction matching VQ-VAE training
- `src/spinlock/noa/validation_utils.py` - Model loading and IC sampling
- `test_episode_integration.py` - End-to-end integration test
- `PHASE2_INTEGRATION_SUMMARY.md` - This document

### Modified:
- `src/spinlock/noa/episode.py` - Integrated VQVAEFeatureExtractor, removed extract_features_fn parameter
- Validation scripts (01-04) - Updated to use real models (already done in previous work)

### Removed:
- `noa/simple_feature_extraction.py` - Deprecated, replaced by VQVAEFeatureExtractor

## Lessons Learned

1. **Checkpoint Metadata is Critical**: Feature extraction configuration should be saved in checkpoints to avoid reverse-engineering
2. **Distribution Alignment**: Features at inference must exactly match training distribution
3. **Multi-Family Architectures**: Need clear separation between training (multi-family) and inference modes (single-family)
4. **1-Channel vs 2-Channel**: Cross-channel features differ for single-channel systems (62D vs 63D)

## Recommendations

1. **Short-term**: Train TEMPORAL-only VQ-VAE for autonomous episode tokenization
2. **Long-term**: Design VQ-VAE architecture with explicit multi-mode support:
   - `forward_all()`: Uses all features (INITIAL + SUMMARY + TEMPORAL)
   - `forward_temporal()`: Per-timestep tokenization (TEMPORAL only)
   - `forward_summary()`: Trajectory-level tokenization (SUMMARY + INITIAL)

3. **Checkpoint Design**: Save feature extraction config in checkpoints:
   ```python
   checkpoint = {
       'model_state_dict': ...,
       'model_config': ...,
       'feature_config': {  # NEW
           'families': ['initial', 'summary', 'temporal'],
           'per_family_dims': {'initial': 14, 'summary': 30, 'temporal': 62},
           'extractor_config': summary_config.model_dump(),
       }
   }
   ```
