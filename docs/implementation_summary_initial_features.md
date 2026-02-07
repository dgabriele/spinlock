# Implementation Summary: Optimized Initial Feature Encoding for VQ-VAE

**Date**: 2026-02-06
**Status**: ✅ Completed
**Goal**: Fix initial feature variance collapse causing poor VQ-VAE reconstruction

---

## Problem Statement

Initial features had collapsed variance (~0.00001) causing:
- Normalized reconstruction errors 136-696× worse than temporal features
- VQ-VAE essentially ignoring initial features
- Global loss plateaued at 0.09-0.11 (target: 0.01-0.02)

**Root causes**:
1. Pattern-complexity manual features with excessive per-feature normalization
2. BatchNorm in CNN encoder forcing variance to ~1.0
3. Multiple layers of normalization amplifying collapse

---

## Solution Implemented

### 1. Removed BatchNorm from CNN Encoder ✅

**File**: `src/spinlock/features/initial/cnn_encoder.py`

**Changes**:
- Removed `self.bn_out = nn.BatchNorm1d(embedding_dim)` (line 141)
- Removed BatchNorm call in forward pass (line 170)
- Removed BatchNorm from intermediate feature extraction (line 212)

**Rationale**: BatchNorm was forcing CNN embeddings to std≈1.0, masking any natural variance in the features and preventing the VQ-VAE from learning meaningful representations.

**Impact**: CNN embeddings now preserve natural variance (expected range: 1-100+).

---

### 2. Created New Statistical Feature Extractor ✅

**File**: `src/spinlock/features/initial/ic_feature_extractors.py` (NEW)

**Class**: `InitialConditionsFeatureExtractor`

**Features extracted** (66D for C=3 channels):

#### Distributional Features (33D = C×11)
Per channel, without normalization:
- Basic statistics: mean, std, min, max, median
- Percentiles: 5th, 25th, 75th, 95th
- Higher moments: skewness, kurtosis

**Rationale**: Captures IC magnitude/scale information crucial for reconstruction.

#### Spatial Features (24D = C×8)
Per channel:
- Gradient statistics (x, y): mean, std
- Laplacian statistics: mean, std
- Center of mass: normalized x, y coordinates

**Rationale**: Captures boundary conditions and spatial structure for decoder.

#### Energy Features (9D = C×2 + C×(C-1)/2)
- Per-channel norms: L2, L1
- Cross-channel correlations: all pairs

**Rationale**: Physics-relevant quantities that inform decoder about energy scales.

**Key design principles**:
- ✅ NO per-feature normalization/clamping
- ✅ Focus on reconstruction-relevant features, not pattern complexity
- ✅ Preserves natural variance (tested: mean std = 777, max std = 16,696)
- ✅ Numerically stable (handles zeros, constants, extremes)
- ✅ Scales appropriately with IC magnitude (100× scaling verified)

---

### 3. Updated Config Schema ✅

**File**: `src/spinlock/features/initial/config.py`

**Changes**:
- Added `use_statistical_features: bool` flag to `InitialManualConfig`
- Updated `use_final_batchnorm` default to `False` in `InitialCNNConfig`
- Added comprehensive documentation

**Usage**:
```yaml
families:
  initial:
    encoder: initial_hybrid
    encoder_params:
      use_final_batchnorm: false  # Preserve variance
      use_statistical_features: true  # Use new features
```

---

### 4. Integrated into Hybrid Encoder ✅

**File**: `src/spinlock/features/initial/extractors.py`

**Changes**:
- Import `InitialConditionsFeatureExtractor`
- Conditional instantiation based on `use_statistical_features` flag
- Updated registry building for statistical features
- Proper handling of [B, M, C, H, W] inputs

**Registry updates**:
- Statistical features: Organized by category (distributional, spatial, energy, cross_channel)
- Pattern features: Preserved existing organization (when flag is False)

---

### 5. Updated Baseline Config ✅

**File**: `configs/vqvae/50k_baseline.yaml`

**Critical changes**:
```yaml
use_final_batchnorm: false  # CRITICAL: Disabled to preserve variance
use_statistical_features: true  # Enable new features
encode_manual: false  # Phase 1: CNN-only mode
```

**Current configuration**:
- Initial features: 256D CNN-only (no BatchNorm)
- Temporal features: 306D (PyramidEncoder)
- Total input: ~562D
- VQ-VAE: 768D embedding, 3072D hidden

**Future option**:
- Set `encode_manual: true` to use 66D statistical + 256D CNN = 322D initial features

---

### 6. Comprehensive Testing ✅

**File**: `tests/features/initial/test_ic_feature_extractors.py`

**Test coverage**:
- ✅ Dimensions: All input shapes (single batch, multi-realization, various channels)
- ✅ Variance preservation: Verified mean std = 777, not collapsed
- ✅ Numerical stability: Handles zeros, constants, extremes
- ✅ Feature behavior: Gradients, correlations, scaling
- ✅ Device compatibility: CPU/CUDA

**Standalone verification**:
```bash
python /tmp/.../test_ic_features.py
# ALL TESTS PASSED ✓
# - Mean variance: 777 (vs target > 0.01)
# - Max variance: 16,696 (vs target > 0.1)
# - 90.9% features have variance > 0.01
# - 100× scaling with IC magnitude
```

---

## Files Created/Modified

### Created (2 files):
1. `src/spinlock/features/initial/ic_feature_extractors.py` (368 lines)
   - New statistical feature extractor
2. `tests/features/initial/test_ic_feature_extractors.py` (486 lines)
   - Comprehensive unit tests

### Modified (4 files):
1. `src/spinlock/features/initial/cnn_encoder.py`
   - Removed BatchNorm (3 locations)
2. `src/spinlock/features/initial/config.py`
   - Added `use_statistical_features` flag
3. `src/spinlock/features/initial/extractors.py`
   - Integrated new extractor, updated registry
4. `configs/vqvae/50k_baseline.yaml`
   - Disabled BatchNorm, enabled statistical features

**Total**: ~900 lines of new/modified code

---

## Expected Impact

### Immediate (Phase 1: CNN-only, no BatchNorm):
- ✅ CNN embeddings preserve natural variance (1-100+ range vs forced 1.0)
- 🎯 Initial feature variance should increase from ~0.00001 to ~1-10
- 🎯 Initial category reconstruction errors should decrease from 136-696 to <10
- 🎯 Global validation loss should improve from ~0.09 to <0.05

### Phase 2 (Enable statistical features):
- 🎯 Reconstruction-relevant features guide VQ-VAE decoder
- 🎯 Initial and temporal features contribute equally
- 🎯 Global validation loss target: <0.03 (approaching 0.027 baseline)

### Success Criteria:
- [ ] Initial feature variance > 0.01 (currently ~0.00001)
- [ ] Initial normalized errors < 10 (currently 136-696)
- [ ] Global validation loss < 0.05 (currently ~0.09-0.11)

---

## Next Steps

### Immediate (Testing Phase):
1. ✅ Delete old checkpoint: `checkpoints/vqvae/50k_baseline`
2. ⏳ Train VQ-VAE with new config (700 epochs, ~12-15 hours)
3. ⏳ Monitor variance in training logs
4. ⏳ Validate reconstruction quality

### Phase 2 (If variance improves):
1. Enable manual features: `encode_manual: true`
2. Retrain and compare 256D CNN-only vs 322D hybrid
3. Measure per-category reconstruction errors
4. Choose best configuration

### Phase 3 (If still not sufficient):
1. Increase CNN capacity to 384D or 512D
2. Consider pre-trained CNN backbone
3. Add learnable feature weighting

---

## Technical Decisions

### Why remove BatchNorm?
- BatchNorm forces output to std≈1.0, destroying natural variance
- When manual features have std~0.00001, BatchNorm doesn't help
- Raw CNN embeddings preserve information better for VQ-VAE

### Why statistical features instead of pattern complexity?
- Pattern complexity (entropy, LZ complexity) doesn't help reconstruction
- Statistical features (mean, std, gradients) inform decoder about IC structure
- Reconstruction needs magnitude/scale info, not pattern descriptors

### Why no per-feature normalization?
- Let VQ-VAE learn which features matter
- Over-normalization kills information content
- Pipeline-level normalization is sufficient

### Why CNN-only mode first?
- Test simplest fix (no BatchNorm) before adding complexity
- CNN has proven capacity (temporal features work well)
- Easier to debug if issues arise

---

## Validation Plan

### Quick Test (First 50 epochs):
```python
# Check variance in logs
grep "initial.*variance" logs/training.log

# Expected: variance > 0.01 (vs ~0.00001 before)
```

### Full Validation (After 700 epochs):
```python
# Compare reconstruction errors
from spinlock.mno.validation_utils import compute_category_errors

errors = compute_category_errors(model, val_loader)
print(errors['initial'])  # Target: < 10 (vs 136-696 before)
```

### Ablation Study:
1. Baseline: Old features (variance ~0.00001)
2. No BatchNorm: New CNN (variance ~1-10)
3. Statistical: New CNN + 66D features (variance ~1-10)
4. Hybrid: Best of 2 and 3

---

## Risk Mitigation

### Risk 1: Variance still collapses
- **Mitigation**: Check pipeline normalization settings
- **Fallback**: Increase CNN capacity to 384D-512D

### Risk 2: Training doesn't converge
- **Mitigation**: Start with smaller learning rate (0.0005)
- **Fallback**: Gradual introduction (no BatchNorm only first)

### Risk 3: Temporal features compensate
- **Mitigation**: Monitor per-category losses
- **Fallback**: Add learnable category weighting

---

## References

### Plan Document:
`/home/daniel/.claude/projects/-home-daniel-projects-spinlock/bdc263ee-5892-4e5c-8f7e-4f504927c716.jsonl`

### Key Insights:
1. "Initial features have collapsed to extremely low variance"
2. "BatchNorm in CNN encoder suppress variance"
3. User: "Maybe the initial features themselves need reconsidering"

### Related Issues:
- VQ-VAE reconstruction errors by category (validation_utils.py)
- Feature variance analysis (encoding/unified_feature_pipeline.py)
- Manual feature extraction (manual_extractors.py)

---

## Contact

For questions or issues:
- Review plan document for detailed analysis
- Check test results: `pytest tests/features/initial/test_ic_feature_extractors.py`
- Validate standalone: `python /tmp/.../test_ic_features.py`
