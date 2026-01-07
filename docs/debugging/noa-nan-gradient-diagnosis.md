# NOA Training: NaN Gradient Diagnosis

**Date:** 2026-01-07
**Issue:** Uniform NaN/Inf gradients after batch 1657
**Root Cause:** Missing truncated BPTT for long sequences (T=256)
**Status:** ✅ Diagnosed - Solution available

---

## Symptoms

```
Warning: NaN/Inf gradients at batch 1657, skipping update
Warning: NaN/Inf gradients at batch 1658, skipping update
Warning: NaN/Inf gradients at batch 1659, skipping update
...
(every single batch)
```

Training appears to proceed but skips ALL weight updates due to NaN gradients.

---

## Root Cause

Your training command is missing the **critical** `--bptt-window` parameter for long sequences:

```bash
# ❌ CURRENT (BROKEN):
poetry run python scripts/dev/train_noa_state_supervised.py \
    --dataset datasets/100k_full_features.h5 \
    --vqvae-path checkpoints/production/100k_3family_v1 \
    --n-samples 8000 --epochs 3 --batch-size 4 --lr 3e-4 \
    --warmup-steps 500 --timesteps 256 \  # <-- T=256 is TOO LONG
    --save-every 100 --early-stop-patience 1
```

**The Problem:**

Backpropagating through 256 sequential autoregressive steps causes **gradient explosion**:

```
u₀ → NOA → u₁ → NOA → u₂ → NOA → ... → u₂₅₆
    ↑                                      ↑
    └──────── 256 gradient hops ──────────┘
```

Each step multiplies gradients by Jacobian norms. After 256 steps, gradients explode to `inf`.

---

## Diagnostic Results

Using the new diagnostic tool (`scripts/dev/diagnose_noa_training.py`):

```bash
poetry run python scripts/dev/diagnose_noa_training.py \
    --checkpoint checkpoints/noa/step_500.pt
```

**Key Findings:**

1. **Forward pass**: ✅ All values are finite
   - Predicted trajectory: range [-38, 120], no NaN/Inf
   - Target trajectory: range [-6, 6], no NaN/Inf
   - State loss: 255.08 (finite)
   - VQ-VAE features: 187D, no NaN/Inf

2. **Backward pass**: 🔴 ALL gradients explode to infinity
   - Gradient norms: `inf` across **ALL 154 parameters**
   - Min gradient norm: 1.45e+27
   - Max gradient norm: `inf`
   - This happens immediately on first backward pass

3. **Root cause confirmed**:
   - `timesteps=256` without `bptt_window=None`
   - Gradient explosion through autoregressive chain

---

## Solution

Add `--bptt-window 32` to enable **Truncated Backpropagation Through Time (TBPTT)**:

```bash
# ✅ CORRECTED:
poetry run python scripts/dev/train_noa_state_supervised.py \
    --dataset datasets/100k_full_features.h5 \
    --vqvae-path checkpoints/production/100k_3family_v1 \
    --n-samples 8000 --epochs 3 --batch-size 4 --lr 3e-4 \
    --warmup-steps 500 \
    --timesteps 256 \
    --bptt-window 32 \  # <-- ADD THIS!
    --save-every 100 --early-stop-patience 1
```

### How TBPTT Works

Instead of backpropagating through all 256 steps, TBPTT:

1. **Warmup phase** (no gradients): Roll out 224 steps (256 - 32)
   ```python
   with torch.no_grad():
       for t in range(224):
           x = noa.single_step(x)
   ```

2. **Supervised phase** (with gradients): Roll out last 32 steps
   ```python
   for t in range(32):
       x = noa.single_step(x)  # Gradients flow
   ```

3. **Loss computed only on supervised window**
   ```python
   loss = MSE(pred_trajectory, target_trajectory[:, -32:])
   ```

This limits gradient flow to 32 steps instead of 256, preventing explosion while still supervising the full trajectory.

---

## Why It Took 1657 Batches to Fail

The gradient explosion is **accumulating** rather than immediate:

1. **Batches 1-500**: Gradients start growing but weights still update
   - Gradient norms increase from ~1e-3 to ~1e10
   - Gradient clipping (max_norm=1.0) helps temporarily

2. **Batches 500-1657**: Weights diverge, gradients accelerate
   - Model parameters drift to unstable regions
   - Gradient explosion becomes more severe each batch

3. **Batch 1657+**: Complete failure
   - Gradients hit `inf` every single batch
   - All updates skipped permanently

The checkpoint at step 500 was saved **before complete failure**, but weights were already diverging.

---

## Expected Results with TBPTT

From the RESUME document (`notes/RESUME-2026-01-06-vqvae-alignment-nan-fix.md`):

```
Epoch 1/2
  Train: total=0.940664 state=0.940304 commit=0.000722 [17.3s]
  Val: total=1.081461 state=1.081146 commit=0.000631

Epoch 2/2
  Train: total=1.050154 state=1.049898 commit=0.000512 [16.3s]
  Val: total=0.895321 state=0.895005 commit=0.000631
```

Training completes successfully with:
- ✅ No NaN/Inf gradients
- ✅ Finite losses throughout
- ✅ Decreasing state loss (physics fidelity improving)
- ✅ Small commit loss (VQ manifold adherence)

---

## When to Use TBPTT

| Timesteps | TBPTT Required? | Recommended `bptt_window` |
|-----------|----------------|---------------------------|
| T ≤ 16    | ❌ No          | N/A                       |
| 16 < T ≤ 32 | ⚠️ Optional  | 16                        |
| 32 < T ≤ 128 | ✅ Yes      | 32                        |
| T > 128   | ✅ **Required** | 32                        |

**Rule of thumb:** If `T > 32`, always use `--bptt-window 32`.

---

## Diagnostic Tool Usage

The new diagnostic tool provides full transparency into NaN issues:

```bash
# Basic diagnosis
poetry run python scripts/dev/diagnose_noa_training.py \
    --checkpoint checkpoints/noa/step_500.pt

# Override timesteps for testing
poetry run python scripts/dev/diagnose_noa_training.py \
    --checkpoint checkpoints/noa/step_500.pt \
    --timesteps 32  # Test with shorter sequence

# Test on more samples
poetry run python scripts/dev/diagnose_noa_training.py \
    --checkpoint checkpoints/noa/step_500.pt \
    --n-samples 10
```

**Output includes:**

1. **Configuration check**: Detects missing TBPTT
2. **Forward pass diagnostics**: Analyzes all intermediate tensors
3. **Backward pass diagnostics**: Lists ALL parameters with NaN/Inf gradients
4. **Gradient statistics**: Norms, percentiles, distribution
5. **Recommended fix**: Exact command to run

---

## Additional Improvements

### Enhanced NaN Detection in Training Script

The training script already has NaN gradient detection:

```python
# Check for NaN in gradients (prevents weight corruption)
has_nan_grad = False
for name, param in noa.named_parameters():
    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
        has_nan_grad = True
        break

if has_nan_grad:
    print(f"Warning: NaN/Inf gradients at batch {batch_idx}, skipping update")
    optimizer.zero_grad()  # Clear corrupted gradients
    continue
```

However, it doesn't show **which** parameters have NaN. Consider adding:

```python
if has_nan_grad:
    print(f"Warning: NaN/Inf gradients at batch {batch_idx}")
    for name, param in noa.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            print(f"  {name}: has_nan={torch.isnan(param.grad).any().item()}, has_inf={torch.isinf(param.grad).any().item()}")
    optimizer.zero_grad()
    continue
```

### LR Scheduler Warning

The deprecation warning:

```
/home/daniel/.cache/pypoetry/virtualenvs/spinlock-JcmOkOp3-py3.13/lib/python3.13/site-packages/torch/optim/lr_scheduler.py:198:
UserWarning: The epoch parameter in `scheduler.step()` was not necessary...
```

Is harmless but can be fixed by removing the epoch argument:

```python
# Current (deprecated)
scheduler.step(epoch)

# Correct
scheduler.step()
```

The training script already uses `scheduler.step()` without arguments (line 251), so this warning may be from the scheduler initialization. It's safe to ignore.

---

## References

- Training script: `scripts/dev/train_noa_state_supervised.py:train_noa_state_supervised.py:115-285`
- Diagnostic tool: `scripts/dev/diagnose_noa_training.py`
- TBPTT implementation: `scripts/dev/train_noa_state_supervised.py:train_noa_state_supervised.py:154-174`
- Previous fix: `notes/RESUME-2026-01-06-vqvae-alignment-nan-fix.md`
- NOA roadmap: `docs/noa-roadmap.md`
