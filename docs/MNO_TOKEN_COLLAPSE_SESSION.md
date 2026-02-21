# MNO Training - Token Space Mode Collapse

## Session Date: 2026-02-20

## Problem Summary

MNO training exhibits **severe token space mode collapse**: as physics losses improve, token diversity collapses and contrastive accuracy drops from 20% → 5%.

###  Observed Behavior

| Batch | traj (physics) | token_contrastive | tc_accuracy | Diagnosis |
|-------|----------------|-------------------|-------------|-----------|
| 10    | 305.2          | 1.61              | 20.0%       | Baseline  |
| 20    | 283.0          | 2.25              | 15.0%       | ↓         |
| 30    | 265.7          | 2.74              | 10.0%       | ↓↓        |
| 40    | 250.8          | 2.92              | 7.5%        | ↓↓↓ COLLAPSE |

**Root cause**: MNO learns a "one-size-fits-all" rollout that reduces physics error on average but ignores parameter specificity. All (θ, IC) → similar rollouts → similar tokens → contrastive loss increasing, accuracy dropping.

---

## Configuration History

### Initial Config (FAILED - mode collapse)
```yaml
lambda_traj: 0.1                    # Weak physics
lambda_ic: 0.1
lambda_param_recon: 0.0             # DISABLED
lambda_token_contrastive: 80.0      # PRIMARY (gradient exploded to 850!)
lambda_roundtrip: 1.0
```

**Result**: token_contrastive gradient exploded (101 → 287 → 589 → 851), completely dominated optimization. Token entropy collapsed to 0.11 bits (vs GT: 0.49 bits).

### Attempted Fix #1 (FAILED - mode collapse persisted)
```yaml
lambda_param_recon: 2.0             # ENABLED to force param-specific rollouts
lambda_token_contrastive: 80.0      # Still too high
```

**Result**: param_recon loss decreased (30 → 12), but tc_accuracy still collapsed (20% → 8%). token_contrastive gradient still exploded to 851.

### Current Config (NEEDS TESTING)
```yaml
lambda_traj: 0.1
lambda_ic: 0.1
lambda_param_recon: 2.0             # Prevents mode collapse
lambda_token_contrastive: 5.0       # REDUCED from 80.0 (16× reduction)
lambda_roundtrip: 1.0
batch_size: 1                       # Reduced for 7GB GPU
gradient_accumulation_steps: 8
```

**File**: `configs/cno/mno/pure_physics.yaml`

**Status**: Not yet successfully tested (OOM from zombie processes).

---

## Ground Truth Token Diversity Analysis

Ran diagnostic on GT dataset (`datasets/50k_cno_v3_tokenized_temporal_res.h5`):

```
GT Token Utilization (n=1000 samples):
  Mean codebook utilization: 0.450 (45% of codes used)
  Unique token combinations: 1000/1000 (100% unique)
  Mean token entropy: 0.49 bits

MNO Token Utilization (batch 100, lambda=80.0):
  Mean codebook utilization: 0.335
  Unique token combinations: 2/2 (only 2 samples collected!)
  Mean token entropy: 0.11 bits  ← SEVERE COLLAPSE
```

**Conclusion**: GT HAS good diversity. MNO collapse is not due to GT being inherently low-diversity.

---

## Key Insights

1. **token_contrastive gradient explosion**: At lambda=80.0, gradients grew from 101 → 851 by batch 200, completely overwhelming other losses (param_recon ~50, traj ~60).

2. **Contrastive loss misleading**: Loss can increase while model gets WORSE (accuracy dropping). This happens when the queue fills with increasingly similar embeddings as training progresses.

3. **param_recon necessary but insufficient**: Enabling it reduced param_recon loss but didn't prevent token collapse when token_contrastive was too strong.

4. **Queue dynamics**: With batch_size=1 + queue_size=64 → 65 options per sample. Random chance = 1.5% accuracy. At 20% accuracy (batch 10), model is 13× better than random. But by batch 40, accuracy drops to 7.5% (only 5× better than random).

---

## Next Steps

1. **Clean GPU memory**: Kill all zombie processes before running.
   ```bash
   pkill -9 -f "spinlock train"
   ps aux | grep python | grep spinlock  # Verify clean
   ```

2. **Test current config**: Run with `lambda_token_contrastive=5.0` + `lambda_param_recon=2.0`.

3. **Monitor these metrics**:
   - `tc_accuracy`: Should STABILIZE or INCREASE (not collapse)
   - `param_recon`: Should decrease to ~5-10 (learning param encoding)
   - `token_contrastive` gradient norm: Should stay <100 (not explode)

4. **If still collapsing**, further reduce token_contrastive to 1.0 or disable entirely, relying only on roundtrip + param_recon.

5. **If stable**, wait for batch 100 token diagnostic:
   ```
   [TOKEN DIAG batch 100]
     Mean codebook utilization: should be >0.35
     Unique token combinations: should be >80/100
     Mean token entropy: should be >0.30 bits
   ```

---

## Files Modified

1. **`src/spinlock/cli/train_meta_operator.py`**:
   - Line 1933: Added `'tc_accuracy'` to metrics whitelist
   - Line 2171: Added `'tc_accuracy'` to validation metrics whitelist

2. **`configs/cno/mno/pure_physics.yaml`**:
   - Line 68: `lambda_param_recon: 0.0 → 2.0`
   - Line 86: `lambda_token_contrastive: 80.0 → 5.0`
   - Line 37: `batch_size: 2 → 1`
   - Line 38: `gradient_accumulation_steps: 4 → 8`

3. **`scripts/check_gt_token_diversity.py`** (NEW):
   - Diagnostic script for checking GT token diversity
   - Run: `poetry run python scripts/check_gt_token_diversity.py`

4. **`src/spinlock/mno/diagnostics/token_utilization.py`**:
   - Already existed, used for in-training diagnostics

---

## Training Command

```bash
# Clean start
pkill -9 -f "spinlock train"
sleep 3

# Run training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  poetry run spinlock train-meta-operator \
  --config configs/cno/mno/pure_physics.yaml
```

**Expected runtime**: ~10s/batch × 9000 batches/epoch = 25 hours/epoch on 7GB GPU.

---

## Critical Warnings

1. **DO NOT** pipe output through `head` or `grep` - need full logs to monitor.
2. **DO NOT** run multiple training instances simultaneously - causes OOM.
3. **ALWAYS** verify no zombie processes before starting:
   ```bash
   ps aux | grep python | grep spinlock | grep -v grep
   ```
4. **STALE GPU ALLOCATION**: Process 1094464 holds 4.2GB GPU memory but doesn't exist in ps.
   - Cannot be killed with kill -9
   - Cannot be cleared with `torch.cuda.empty_cache()`
   - **SOLUTION**: Reboot OR `sudo systemctl restart display-manager` (will close desktop session)

---

## Success Criteria

Training is successful if by batch 100:
- tc_accuracy >= 15% (at least 10× better than random)
- tc_accuracy NOT decreasing monotonically
- token entropy >= 0.30 bits (at least 60% of GT)
- param_recon loss < 10 (model encoding params)
- token_contrastive gradient norm < 200 (not exploding)

If criteria not met, reduce lambda_token_contrastive further or disable entirely.
