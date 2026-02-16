# VQTokenizer "Best" Model Selection Fix

**Date**: 2026-02-16
**Issue**: Model with 100× better reconstruction loss not being saved as "best"
**Root Cause**: Utilization weight was 100× too large

---

## The Problem

### Observed Behavior
- **Current "best" model**: `recon_loss = 0.01`
- **New checkpoint**: `recon_loss = 0.001` (100× better!)
- **Not saving as "best"**: Despite massive reconstruction improvement

### Root Cause Analysis

The "best model" metric in `src/spinlock/tokens/trainer.py` (line 202-207) was:

```python
best_metric = (
    recon                    # Reconstruction loss (~0.001-0.01)
    + roundtrip              # Roundtrip consistency (~0.0-0.01)
    + 0.1 * topo             # Topographic loss (de-emphasized)
    - 0.01 * util_epoch      # ← PROBLEM: Utilization bonus (0-100%)
)
```

**Why this failed**:

Example scenario:
- Reconstruction improves: `0.01 → 0.001` (change = **-0.009**)
- Utilization drops: `50% → 20%` (change = **-30%**)
- Utilization penalty: `-0.01 × (-30) = **+0.30**`
- **Net change in best_metric**: `-0.009 + 0.30 = **+0.291**` (WORSE!)

The utilization weight of `0.01` was **100× too large**. A 30% drop in utilization created a 0.30 penalty that completely overwhelmed the 0.009 reconstruction improvement!

---

## The Fix

**Changed**: Line 206 in `src/spinlock/tokens/trainer.py`

**Before**:
```python
- 0.01 * util_epoch  # Reward higher codebook utilization (%)
```

**After**:
```python
- 0.0001 * util_epoch  # Tiny utilization bonus (was 0.01, too large!)
```

**Weight reduction**: `0.01 → 0.0001` (100× smaller)

### Impact of Fix

Same scenario now:
- Reconstruction improves: `0.01 → 0.001` (change = **-0.009**)
- Utilization drops: `50% → 20%` (change = **-30%**)
- Utilization penalty: `-0.0001 × (-30) = **+0.003**`
- **Net change in best_metric**: `-0.009 + 0.003 = **-0.006**` (BETTER! ✓)

Now reconstruction quality **dominates** as it should. A model with 100× better reconstruction will save as "best" even if utilization drops moderately.

---

## Metric Priorities (After Fix)

The "best model" metric now prioritizes in this order:

1. **Reconstruction quality** (`recon`) - Weight: 1.0 ✓
   - Most important: how well does it reconstruct features?
   - Range: ~0.001-0.01

2. **Roundtrip consistency** (`roundtrip`) - Weight: 1.0
   - Secondary: encode → quantize → decode consistency
   - Range: ~0.0-0.01

3. **Topographic loss** (`topo`) - Weight: 0.1
   - De-emphasized: smoothness of codebook manifold
   - Range: ~0.0-0.1

4. **Codebook utilization** (`util_epoch`) - Weight: 0.0001
   - Tiny tiebreaker: prefer higher utilization if all else equal
   - Range: 0-100%

### Typical "best_metric" values:
- **Good model**: `0.001 + 0.001 + 0.01 - 0.005 = 0.007`
- **Bad model**: `0.01 + 0.005 + 0.05 - 0.003 = 0.062`

A 100× reconstruction improvement (`0.01 → 0.001`) dominates any reasonable utilization change.

---

## What This Means for Training

### Before Fix ❌
- Training could get stuck with poor reconstruction
- High utilization models were overvalued
- 100× reconstruction improvements were ignored

### After Fix ✓
- Best model = best reconstruction quality
- Utilization is a minor tiebreaker only
- Improvements in reconstruction are properly recognized

### If utilization is still too important

You can reduce it further or remove it entirely:

```python
# Option 1: Even smaller weight
- 0.00001 * util_epoch  # 1000× smaller than original

# Option 2: Remove entirely
# - 0.0 * util_epoch  # Don't consider utilization at all

# Option 3: Only recon + roundtrip (simplest)
best_metric = recon + roundtrip
```

---

## How to Verify the Fix

### Check training logs

Look for:
```
New best model saved: metric=0.007123 (recon=0.0012, roundtrip=0.0008, topo=0.0450, util_epoch=45.2%)
```

Now if reconstruction improves significantly (e.g., `0.01 → 0.001`), it **will** save as best even if utilization drops.

### Compare metrics

Old best:
- `recon=0.01`, `util=50%` → `best_metric = 0.01 + ... - 0.01*50 = 0.01 - 0.50 = -0.49` (WRONG!)

New checkpoint:
- `recon=0.001`, `util=20%` → `best_metric = 0.001 + ... - 0.0001*20 = 0.001 - 0.002 = -0.001` (BETTER! ✓)

With the fix, the new checkpoint is correctly recognized as better (lower metric = better).

---

## Resume Training

If you want the current training run to re-evaluate with the fixed metric:

```bash
# Training will continue and properly evaluate new checkpoints
# The next validation that improves reconstruction will save as "best"
```

Or restart from the checkpoint with `recon=0.001`:

```bash
poetry run spinlock train-vq-tokenizer \
    --config your_config.yaml \
    --resume checkpoints/path/to/recon_0.001_checkpoint.pt
```

---

## Summary

**Fixed**: Utilization weight reduced from `0.01` to `0.0001` (100× smaller)

**Result**: Reconstruction quality now properly dominates "best model" selection

**Impact**: Models with better reconstruction will save as "best", even if utilization drops

**Your case**: Model with `recon=0.001` will now save over model with `recon=0.01` ✓
