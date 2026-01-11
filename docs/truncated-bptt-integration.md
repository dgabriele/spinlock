# Truncated BPTT Integration Guide

## Overview

The dataset uses 256-step CNO trajectories, but training on full 256-step rollouts with full backpropagation can cause:
- GPU memory overflow (activations for 256 steps)
- Gradient explosion (long autoregressive chains)
- Slow training (256x more computation per batch)

**Solution**: Truncated Backpropagation Through Time (TBPTT)
- Rollout for full 256 steps (matching dataset generation)
- Only backprop through last 32 steps (prevent gradient explosion)
- Detach gradients after first 224 steps (save memory)

## Implementation

The `TruncatedBPTT` wrapper has been abstracted from the archived training script and is now available as a reusable OOP module.

### Module Location

```python
from spinlock.noa import TruncatedBPTT
```

### Basic Usage

```python
from spinlock.noa import NOABackbone, TruncatedBPTT

# Create NOA model
noa = NOABackbone(
    base_channels=40,
    encoder_levels=3,
    modes=16,
    afno_blocks=4,
    token_conditioning=True,
    # ... other params
)

# Wrap with truncated BPTT
tbptt = TruncatedBPTT(
    model=noa,
    timesteps=256,       # Full rollout length (matches dataset)
    bptt_window=32,      # Backprop window (prevents explosion)
)

# Training loop
for batch in dataloader:
    ic = batch["ic"]
    tokens = batch["tokens"]

    # Generate trajectory with truncated BPTT
    # Returns: [B, 33, C, H, W] (warmup final state + 32 supervised steps)
    pred_trajectory = tbptt.rollout(ic, tokens=tokens)

    # Generate full target trajectory
    target_trajectory = replayer.rollout(ic, params, timesteps=256)

    # Align predicted and target for loss computation
    # This handles windowing automatically
    pred_states, target_states = tbptt.align_for_loss(
        pred_trajectory,
        target_trajectory,
        skip_ic=True,
    )

    # Compute loss on aligned states
    # pred_states: [B, 32, C, H, W] (supervised window)
    # target_states: [B, 32, C, H, W] (last 32 states of target)
    loss = F.mse_loss(pred_states, target_states)

    loss.backward()
    optimizer.step()
```

## Integration into `train_meta_operator.py`

### Step 1: Import TruncatedBPTT

Add to imports (around line 30):
```python
from spinlock.noa import NOABackbone, TruncatedBPTT
```

### Step 2: Add Config Parameters

In experiment config YAML (e.g., `exp2e_token_stable_large.yaml`):
```yaml
training:
  timesteps: 256           # Full rollout length (matches dataset)
  bptt_window: 32          # Backprop window (NEW)
  batch_size: 2
  epochs: 30
  # ... rest unchanged
```

### Step 3: Initialize TruncatedBPTT Wrapper

After creating NOA model (around line 520):
```python
# Create NOA backbone
noa = NOABackbone(
    spatial_dim=config["model"]["spatial_dim"],
    # ... other params
).to(device)

# Wrap with truncated BPTT
timesteps = config["training"]["timesteps"]
bptt_window = config["training"].get("bptt_window")

if bptt_window is not None and bptt_window < timesteps:
    print(f"  Using truncated BPTT: {timesteps} steps, backprop window={bptt_window}")
    noa_rollout = TruncatedBPTT(noa, timesteps=timesteps, bptt_window=bptt_window)
else:
    print(f"  Using full backprop: {timesteps} steps")
    # Create wrapper that just passes through to noa.rollout()
    class FullBPTTWrapper:
        def __init__(self, model, timesteps):
            self.model = model
            self.timesteps = timesteps

        def rollout(self, ic, tokens=None):
            return self.model.rollout(ic, steps=self.timesteps, return_all_steps=True, tokens=tokens)

        def align_for_loss(self, pred_traj, target_traj, skip_ic=True):
            if skip_ic:
                return pred_traj[:, 1:, :, :, :], target_traj[:, 1:, :, :, :]
            else:
                return pred_traj, target_traj

    noa_rollout = FullBPTTWrapper(noa, timesteps)
```

### Step 4: Update Training Loop

Modify rollout generation (around line 766):
```python
# Before:
# pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True, tokens=batch_tokens)

# After:
pred_trajectory = noa_rollout.rollout(ic, tokens=batch_tokens)
```

Modify loss computation (around line 800):
```python
# Before:
# pred_states = pred_trajectory[:, 1:, :, :, :]
# target_states = target_trajectory[:, 1:, :, :, :]

# After:
pred_states, target_states = noa_rollout.align_for_loss(
    pred_trajectory,
    target_trajectory,
    skip_ic=True,
)
```

### Step 5: Update Validation Loop

Apply same changes to validation (around line 900):
```python
# Generate trajectory
pred_trajectory = noa_rollout.rollout(ic, tokens=batch_tokens)

# Align for loss
pred_states, target_states = noa_rollout.align_for_loss(
    pred_trajectory,
    target_trajectory,
    skip_ic=True,
)

# Compute loss
val_loss = loss_fn.compute(
    pred_trajectory=pred_states,
    target_trajectory=target_states,
    ic=ic,
    noa=noa,
)
```

## Memory and Performance Impact

### Without Truncated BPTT (256 full steps)
- Activation memory: ~5-8 GB
- Gradient memory: ~3-5 GB
- **Total: ~10-13 GB per batch**
- Risk: Gradient explosion, OOM errors

### With Truncated BPTT (256 steps, backprop window=32)
- Activation memory: ~1-2 GB (only 32 steps tracked)
- Gradient memory: ~0.5-1 GB (only 32 steps)
- Warmup phase: 224 steps with `torch.no_grad()` (minimal memory)
- **Total: ~2-3 GB per batch**
- Benefit: 4-5x memory reduction, stable gradients

## Configuration Examples

### Experiment 2F: 256-Step Training with Truncated BPTT

Create `configs/noa/experiments/phase2/exp2f_256step_tbptt.yaml`:
```yaml
# Experiment 2F: Token-Conditioned MNO with 256-Step Training
# - Full-horizon rollouts matching dataset generation
# - Truncated BPTT with 32-step backprop window
# - Stable training: 5e-5 LR + warmup
# Goal: Learn long-horizon dynamics, achieve <0.3 MSE

model:
  spatial_dim: 64
  in_channels: 1
  out_channels: 1
  base_channels: 40          # Increased capacity
  encoder_levels: 3
  modes: 16
  afno_blocks: 4

  # Token conditioning
  token_conditioning: true
  token_embed_dim: 64
  vqvae_checkpoint: "checkpoints/production/100k_full_features/best_model.pt"

training:
  n_samples: 1000
  batch_size: 2              # Reduced for 256-step rollouts
  epochs: 30
  learning_rate: 5.0e-5      # Stable LR
  warmup_steps: 2250         # 5 epochs × 450 batches
  weight_decay: 1.0e-4
  clip_grad: 0.5

  # Long-horizon training with truncated BPTT
  timesteps: 256             # Full rollout (matches dataset)
  bptt_window: 32            # Backprop window (prevents explosion)

  early_stopping_patience: 10

loss:
  lambda_traj: 1.0

data:
  dataset_path: "datasets/100k_full_features.h5"
  oracle_token_path: "datasets/100k_oracle_tokens_1k.h5"
  cno_config: "configs/experiments/local_100k_optimized.yaml"
  val_split: 0.1
  num_workers: 4

checkpointing:
  save_dir: "checkpoints/experiments/phase2/exp2f_256step_tbptt"
  save_every: 5
  keep_best: true

device: "cuda"
seed: 42
```

### Experiment 2G: Clean Baseline with 256 Steps (No Tokens)

Create `configs/noa/experiments/phase2/exp2g_256step_clean.yaml`:
```yaml
# Same as 2F but without token conditioning

model:
  spatial_dim: 64
  in_channels: 1
  out_channels: 1
  base_channels: 32          # Baseline capacity
  encoder_levels: 3
  modes: 16
  afno_blocks: 4

  token_conditioning: false  # No tokens

training:
  n_samples: 1000
  batch_size: 4              # Can use larger batch without tokens
  epochs: 30
  learning_rate: 5.0e-5
  warmup_steps: 1125         # 5 epochs × 225 batches
  weight_decay: 1.0e-4
  clip_grad: 0.5

  timesteps: 256
  bptt_window: 32

  early_stopping_patience: 10

loss:
  lambda_traj: 1.0

data:
  dataset_path: "datasets/100k_full_features.h5"
  cno_config: "configs/experiments/local_100k_optimized.yaml"
  val_split: 0.1
  num_workers: 4

checkpointing:
  save_dir: "checkpoints/experiments/phase2/exp2g_256step_clean"
  save_every: 5
  keep_best: true

device: "cuda"
seed: 42
```

## Expected Results

### Hypothesis
Training on full 256-step trajectories should improve long-horizon accuracy:
- Better capture of transient dynamics (first 50 steps)
- Improved late-timestep stability (steps 200-256)
- Lower overall MSE when evaluated on full trajectories

### Comparison Matrix

| Experiment | Timesteps | BPTT Window | Tokens | Capacity | Expected Val Loss |
|------------|-----------|-------------|---------|----------|-------------------|
| 2E (current) | 32 | N/A (full) | Yes | 226M | ~0.35-0.40 |
| 2F (new) | 256 | 32 | Yes | 226M | **~0.25-0.30** |
| 2G (baseline) | 256 | 32 | No | 144M | ~0.40-0.45 |

**Key Insight**: The bottleneck may not be rollout error accumulation (as shown by analysis), but rather **insufficient supervision on long-horizon dynamics**. Training on only 32 steps means the model never learns what happens at t=100, t=200, etc.

## Validation

After implementing, verify:

1. **Memory Usage**:
   ```bash
   nvidia-smi
   # Should show ~2-3 GB per GPU (vs 10+ GB without truncation)
   ```

2. **Gradient Flow**:
   - Check for NaNs/Infs (should be rare with truncation)
   - Monitor loss stability (should not explode mid-epoch)

3. **Rollout Quality**:
   ```bash
   python scripts/analysis/analyze_rollout_error.py \
       --checkpoint checkpoints/experiments/phase2/exp2f_256step_tbptt/meta_operator_best.pt \
       --config configs/noa/experiments/phase2/exp2f_256step_tbptt.yaml \
       --oracle-tokens datasets/100k_oracle_tokens_1k.h5 \
       --timesteps 256 \
       --output /tmp/exp2f_rollout_error_256step.png
   ```

4. **Compare Error at Different Horizons**:
   - t=32: Should match or beat Exp 2E
   - t=128: Should be significantly better than Exp 2E (new supervision!)
   - t=256: Should be significantly better than Exp 2E (new supervision!)

## References

- Original implementation: `scripts/archived/dev/train_noa_state_supervised.py`
- Architecture docs: `docs/noa-architecture.md`
- Debugging guide: `docs/debugging/noa-nan-gradient-diagnosis.md`
