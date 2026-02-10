# VQTokenizer Training Metrics Implementation

**Date**: 2026-02-09
**Status**: ✅ Complete

## Overview

Implemented comprehensive metrics tracking for VQTokenizer training to address the issue where EMA cluster sizes showed misleading 100% utilization (from training transients) when reality was 1-3 codes per quantizer after convergence.

## Problem Statement

**Before**:
- Training showed "100% utilization" based on EMA cluster sizes
- EMA accumulates training transients (pre-convergence random assignments)
- No per-quantizer utilization tracking in final metrics
- 22.6% of temporal features had zero/near-zero variance but weren't filtered
- Metrics computed during training, not post-convergence

**After**:
- Real token frequency tracking during validation
- Per-quantizer utilization computed from actual usage (not EMA artifacts)
- Post-training metrics capture true post-convergence behavior
- Feature variance filtering enabled via config
- Per-category reconstruction MSE tracking

## Implementation Details

### Phase 1: Feature Variance Filtering (Config Only)

**File**: `configs/vqvae_50k.yaml`

Added `feature_cleaning` section to enable zero-variance feature removal:

```yaml
feature_cleaning:
  enabled: true
  pre_categorization: true  # Clean before feature grouping
  variance_threshold: 1.0e-8  # Remove features with std < 1e-8
  deduplicate_threshold: 0.99  # Remove features with |corr| > 0.99
  use_intelligent_dedup: true  # Keep more informative feature
  outlier_method: "percentile"  # Use percentile-based outlier capping
  percentile_range: [0.5, 99.5]  # Clip at 0.5% and 99.5% percentiles
```

**Integration**:
- System already implemented in `src/spinlock/encoding/feature_processor.py`
- No code changes needed - just config enablement
- Will remove ~22 zero-variance features (22.6% of 320 temporal features)

### Phase 2: Per-Quantizer Utilization Tracking

**File**: `src/spinlock/tokens/trainer.py`

**Changes**:

1. **Modified `_validate_epoch()` to track token frequencies**:
   ```python
   # Initialize frequency counters
   token_frequencies = {}
   for quantizer_name in self.model.quantizers.keys():
       num_codes = self.model.quantizers[quantizer_name].num_embeddings
       token_frequencies[quantizer_name] = torch.zeros(num_codes, dtype=torch.long)

   # In validation loop
   if 'token_indices' in outputs:
       for quantizer_name, token_idxs in outputs['token_indices'].items():
           flat_tokens = token_idxs.flatten()
           counts = torch.bincount(flat_tokens, minlength=token_frequencies[quantizer_name].shape[0])
           token_frequencies[quantizer_name] += counts.cpu()
   ```

2. **Added helper method `_compute_token_utilization()`**:
   ```python
   def _compute_token_utilization(self, token_frequencies: Dict[str, torch.Tensor]) -> Dict[str, float]:
       """Compute utilization from token frequency counts.

       Utilization = (codes used at least once) / codebook_size
       """
       utilizations = {}
       for quantizer_name, frequencies in token_frequencies.items():
           num_used = (frequencies > 0).sum().item()
           codebook_size = len(frequencies)
           utilizations[quantizer_name] = (num_used / codebook_size) * 100.0
       return utilizations
   ```

3. **Return per-quantizer utilization in validation metrics**:
   ```python
   per_quantizer_utilization = self._compute_token_utilization(token_frequencies)

   return {
       ...,
       'per_quantizer_utilization': per_quantizer_utilization,
   }
   ```

### Phase 3: Post-Training Metrics Computation

**File**: `src/spinlock/tokens/trainer.py`

**Changes**:

1. **Added `_compute_final_validation_metrics()` method**:
   - Runs one final validation pass after training completes
   - Captures post-convergence behavior (no training transients)
   - Formats metrics for visualization dashboard compatibility
   - Parses quantizer names like `"temporal_group_1_L0"` → category + level
   - Returns dict with keys: `"{category}/level_{level}/utilization"`

2. **Integrated into `train()` method**:
   ```python
   # After training loop completes
   logger.info("Computing final validation metrics...")
   final_metrics = self._compute_final_validation_metrics(val_loader)
   self.training_history['final_metrics'] = final_metrics
   ```

3. **Included in checkpoint metadata**:
   - Checkpoint now contains `metadata['training_history']['final_metrics']`
   - Visualization dashboards can load metrics directly from checkpoint
   - No need to re-compute or pretokenize for visualization

### Phase 4: Per-Category MSE Tracking

**File**: `src/spinlock/tokens/trainer.py`

**Changes**:

1. **Track per-category reconstruction MSE in `_validate_epoch()`**:
   ```python
   category_mse = {}
   category_counts = {}

   # In validation loop
   original = outputs['original_encoded']
   reconstructed = outputs['reconstructed']
   for family_cat, indices in self.group_indices.items():
       cat_orig = original[:, indices]
       cat_recon = reconstructed[:, indices]
       cat_mse = torch.mean((cat_orig - cat_recon) ** 2).item()

       if family_cat not in category_mse:
           category_mse[family_cat] = 0.0
           category_counts[family_cat] = 0

       category_mse[family_cat] += cat_mse
       category_counts[family_cat] += 1
   ```

2. **Return per-category MSE in validation metrics**:
   ```python
   per_category_mse = {
       cat: category_mse[cat] / category_counts[cat]
       for cat in category_mse.keys()
   }

   return {
       ...,
       'per_category_mse': per_category_mse,
   }
   ```

3. **Include in final metrics with visualization-compatible keys**:
   - Keys: `"{category}/reconstruction_mse"`
   - Engineering dashboard can show per-category MSE bars

### Phase 5: Configuration Schema

**File**: `src/spinlock/tokens/config.py`

**Changes**:

Added `FeatureCleaningConfig` Pydantic model:
```python
class FeatureCleaningConfig(BaseModel):
    """Feature cleaning configuration (pre-training)."""
    enabled: bool = Field(default=False)
    pre_categorization: bool = Field(default=True)
    variance_threshold: float = Field(default=1e-8, ge=0.0)
    deduplicate_threshold: float = Field(default=0.99, ge=0.0, le=1.0)
    use_intelligent_dedup: bool = Field(default=True)
    outlier_method: Literal["percentile", "iqr", "mad", "none"] = Field(default="percentile")
    percentile_range: tuple[float, float] = Field(default=(0.5, 99.5))
```

Added to `TokenizerConfig`:
```python
class TokenizerConfig(BaseModel):
    ...
    feature_cleaning: Optional[FeatureCleaningConfig] = None
    ...
```

## Testing

**Test Script**: `scripts/validation/test_trainer_metrics.py`

**Results**:
```
✓ Feature cleaning config loads correctly
✓ Per-quantizer utilization tracked (48 quantizers)
✓ Average utilization: 11.12% (not 100%!)
✓ Per-category MSE tracked (16 categories)
✓ Final metrics computed with 64 keys
✓ Visualization-compatible key format verified
```

**Sample Output**:
```
Per-Quantizer Utilization:
  temporal_group_0_L0: 3.57%
  temporal_group_0_L1: 10.00%
  temporal_group_0_L2: 16.67%
  Average utilization: 11.12%

Final Metrics (visualization format):
  temporal_group_0/level_0/utilization: 3.57%
  temporal_group_0/level_1/utilization: 10.00%
  temporal_group_0/level_2/utilization: 16.67%
  temporal_group_0/reconstruction_mse: 0.417792
```

## Files Modified

| File | Changes | Risk |
|------|---------|------|
| `configs/vqvae_50k.yaml` | Added `feature_cleaning` section | Low |
| `src/spinlock/tokens/config.py` | Added `FeatureCleaningConfig` model | Low |
| `src/spinlock/tokens/trainer.py` | Added metrics tracking + post-training computation | Medium |
| `scripts/validation/test_trainer_metrics.py` | New test script | None |

## Expected Outcomes

### Immediate (After Implementation)

✅ **Metrics tracking works correctly**:
- Per-quantizer utilization shows real values (3-16%, not 100%)
- Per-category MSE tracked for debugging
- Final metrics saved in checkpoint metadata
- Visualization dashboards work without errors

### After Feature Filtering (Next Training Run)

**If utilization improves (10-30% per quantizer)**:
- ✅ Success! Feature filtering unlocked codebook capacity
- Proceed with MNO tokenizer training
- Continue with alignment layer work

**If utilization stays low (1-10% per quantizer)**:
- ⚠️ Dataset diversity issue - features cleaned but data still narrow
- Need to generate more diverse CNO dataset (100K-200K samples)
- Use wider parameter ranges in Sobol sampling
- Verify parameter space coverage spans full [0,1]^14 hypercube
- Retrain CNO tokenizer on diverse data
- Then proceed with MNO work

## Verification Commands

### Run Test Script
```bash
poetry run python scripts/validation/test_trainer_metrics.py
```

### Train with New Metrics (50K dataset)
```bash
poetry run spinlock train-vq-tokenizer \
  --config configs/vqvae_50k.yaml \
  --output checkpoints/vqvae_50k_with_metrics
```

### Inspect Checkpoint Metrics
```python
import torch

checkpoint = torch.load("checkpoints/vqvae_50k_with_metrics/vq_tokenizer_final.pt")
final_metrics = checkpoint['metadata']['training_history']['final_metrics']

# Check utilization metrics
util_keys = [k for k in final_metrics.keys() if 'utilization' in k]
print(f"Found {len(util_keys)} utilization metrics")

# Sample values
for k in sorted(util_keys)[:5]:
    print(f"{k}: {final_metrics[k]:.4f}%")
```

### Generate Visualizations
```bash
poetry run spinlock visualize-vqvae \
  --checkpoint checkpoints/vqvae_50k_with_metrics \
  --output visualizations/
```

## Next Steps

### Immediate (Before Training)

1. ✅ **Implementation Complete**
   - All phases implemented and tested
   - Config schema updated
   - Test script passing

### Short-Term (Next Training Run)

2. **Run Training with New Metrics**
   ```bash
   poetry run spinlock train-vq-tokenizer --config configs/vqvae_50k.yaml
   ```

3. **Analyze Results**
   - Check final_metrics in checkpoint
   - Verify feature filtering removed ~22 features
   - Assess per-quantizer utilization (target: 3-10% minimum, 10-30% ideal)

4. **Generate Visualizations**
   - Engineering dashboard should show realistic utilization gradient
   - Per-category MSE bars for debugging

### Decision Point (Based on Results)

**If Utilization Improves (10-30%)**:
- ✅ Proceed with MNO tokenizer training
- Continue with alignment layer work
- Dual tokenizer architecture as planned

**If Utilization Stays Low (1-10%)**:
- ⚠️ Generate diverse CNO dataset (100K-200K, wider parameters)
- Retrain CNO tokenizer on diverse data
- Verify token diversity before MNO work
- Then continue with dual tokenizer architecture

## Design Trade-Offs

### Pre-Categorization vs Post-Categorization Cleaning
**Chosen**: Pre-categorization (clean before feature grouping)
- **Pros**: Simpler, faster, prevents noise from influencing grouping
- **Cons**: May remove globally-zero but locally-informative features
- **Status**: Already implemented in FeatureProcessor, just needs config enabling

### Token Frequency vs Embedding Norms
**Chosen**: Token frequency during validation
- **Pros**: Reflects actual inference usage, more accurate post-convergence
- **Cons**: Requires validation pass, slightly slower
- **Status**: Preferred for final_metrics, embedding norms still used during training

### Metrics Timing
**Chosen**: Compute after training completes
- **Pros**: Avoids transients, matches visualization expectations
- **Cons**: Adds extra validation pass at end
- **Status**: Clean separation of training vs evaluation

## References

- **Memory Document**: `/home/daniel/.claude/projects/-home-daniel-projects-spinlock/memory/MEMORY.md`
- **Diversity Analysis**: `docs/tokenizer-diversity-CORRECTED-FINAL.md`
- **Feature Processor**: `src/spinlock/encoding/feature_processor.py`
- **Trainer**: `src/spinlock/tokens/trainer.py`
- **Config**: `src/spinlock/tokens/config.py`
- **Test Script**: `scripts/validation/test_trainer_metrics.py`
