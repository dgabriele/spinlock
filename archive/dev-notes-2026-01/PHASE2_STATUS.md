# Phase 2 Experiment Status

**Started**: 2026-01-10
**Status**: RUNNING

---

## Experiment 2A: Baseline 1K Samples

**Goal**: Validate that Phase 1 improvements (11.7% gain) scale to larger datasets

### Configuration

```yaml
model:
  base_channels: 32    # Standard capacity
  modes: 16
  afno_blocks: 4

training:
  n_samples: 1000      # 10x larger than Phase 1
  batch_size: 4
  epochs: 30
  weight_decay: 1.0e-4  # From winning Phase 1 config (1E)
  clip_grad: 0.5        # From winning Phase 1 config (1E)
  early_stopping_patience: 10
```

**Key Features**:
- ✅ Conditional GPU cache clearing (code-level improvement)
- ✅ Stronger regularization (weight_decay=1e-4, clip_grad=0.5)
- ✅ Early stopping (patience=10)
- ❌ No warmup (not beneficial with small batch size)
- ❌ No gradient accumulation (hurts performance)

### Status

**Current**: Epoch 1/30 in progress
**Dataset**: 900 train samples, 100 val samples
**Batch size**: 4 (225 batches per epoch)
**Estimated time per epoch**: ~4-5 minutes
**Total estimated time**: ~2-3 hours (with early stopping likely triggering around epoch 15-20)

### Expected Results

Based on Phase 1 findings:

| Metric | Phase 1 (100 samples) | Expected 2A (1K samples) |
|--------|----------------------|--------------------------|
| **Val Loss** | 0.454 | **< 0.40** ✨ |
| **vs Baseline** | +11.7% | **+20-25%** |
| **Training Stability** | Stable | More stable |
| **Convergence** | ~8 epochs | ~15-20 epochs |

### Success Criteria

✅ **Primary**: Val loss < 0.40 (confirms improvements scale)
✅ **Secondary**: Training stable, no divergence
✅ **Tertiary**: Early stopping triggers appropriately

If these are met, proceed to Experiment 2B (capacity test).

---

## Monitoring Commands

### Check Current Progress
```bash
# View live training output
tail -f /tmp/claude/-home-daniel-projects-spinlock/tasks/bfd54f4.output

# View training log (once epochs complete)
tail -f checkpoints/experiments/phase2/exp2a_baseline_1k/training_log.txt

# Quick status check
watch -n 10 "tail -20 /tmp/claude/-home-daniel-projects-spinlock/tasks/bfd54f4.output"
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Parse Current Results
```bash
# Get best validation loss so far
grep -v "^#" checkpoints/experiments/phase2/exp2a_baseline_1k/training_log.txt | \
  awk -F',' '{print $3}' | sort -g | head -1
```

---

## Timeline

| Milestone | Est. Time | Status |
|-----------|-----------|--------|
| Experiment starts | 11:30 | ✅ Running |
| First epoch complete | 11:35 | 🔄 In progress |
| Epoch 10 | 12:15 | ⏳ Pending |
| Epoch 20 | 13:00 | ⏳ Pending |
| Likely completion (early stop) | 13:30-14:00 | ⏳ Pending |

---

## What Happens Next

### If 2A Succeeds (val loss < 0.40)

**Immediate**:
1. ✅ Phase 1 improvements validated at scale
2. ✅ Ready to test capacity increase

**Next Experiment**: 2B - Increased Capacity
- Moderate capacity increase (base_channels=40, modes=24, afno_blocks=5)
- Tests if larger model helps with more data
- Runtime: ~4-5 hours

### If 2A Fails (val loss > 0.40)

**Diagnosis**:
1. Check if training diverged (unlikely given Phase 1 success)
2. Check if early stopping triggered too soon
3. Compare train/val gap for overfitting

**Possible Actions**:
- Adjust learning rate schedule
- Increase early stopping patience
- Add more regularization
- Investigate data issues

### After 2A and 2B Complete

**Decision Point**: Choose configuration for Phase 3 (10K samples)
- If 2B > 2A: Use increased capacity
- If 2B ≈ 2A: Use standard capacity (faster training)

**Phase 3 Target**: Val loss < 0.30 on 10K samples (40% improvement over baseline!)

---

## Phase 1 Context (For Reference)

Completed experiments on 100 samples:

| Experiment | Val Loss | Result |
|-----------|----------|--------|
| **1E: Stronger Regularization** 🥇 | **0.454** | Winner |
| 1C: No Cache Clearing 🥈 | 0.463 | Great |
| 1A: LR Warmup | 1.317 | Poor |
| 1F: Combined | 1.511 | Poor |
| 1B: Gradient Accumulation | 4.135 | Failed |

**Key Insight**: Simple improvements (conditional cache + regularization) beat complex combinations.

---

## Background Task Info

**Task ID**: bfd54f4
**Output File**: `/tmp/claude/-home-daniel-projects-spinlock/tasks/bfd54f4.output`
**Config**: `configs/noa/experiments/phase2/exp2a_baseline_1k.yaml`
**Checkpoint Dir**: `checkpoints/experiments/phase2/exp2a_baseline_1k/`

---

**Last Updated**: 2026-01-10 11:30
**Auto-updating**: This file is static - check task output for live progress
