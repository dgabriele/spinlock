# NOA Training Guide

Complete guide for training Neural Operator Agents (NOA) with VQ-VAE alignment, including L_latent loss, checkpointing, and resume functionality.

---

## Table of Contents

- [Overview](#overview)
- [Training Architecture](#training-architecture)
- [Quick Start](#quick-start)
- [Training Configuration](#training-configuration)
- [Loss Functions](#loss-functions)
- [Checkpointing and Resume](#checkpointing-and-resume)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Diagnostics](#diagnostics)
- [Troubleshooting](#troubleshooting)

---

## Overview

NOA training uses **state-level supervision** with optional **VQ-VAE alignment** to learn physics-native rollout generation. The training objective combines three complementary losses:

```
L_total = L_traj + λ_commit·L_commit + λ_latent·L_latent
```

- **L_traj**: MSE between NOA predictions and CNO ground truth (physics fidelity)
- **L_commit**: VQ-VAE commitment loss (manifold adherence)
- **L_latent**: NOA-VQ latent alignment loss (representation learning)

### Key Features

- **U-AFNO Backbone**: Physics-native neural operator with spectral mixing
- **Truncated BPTT**: Prevents gradient explosion for long sequences (T=256)
- **Three-Loss Training**: Learns physics + behavioral representations simultaneously
- **Checkpoint Resume**: Robust resumption from training interruptions
- **Diagnostic Tools**: Comprehensive alignment quality evaluation

---

## Training Architecture

```
Input: (IC, operator_params) → CNO rollout (ground truth)
                              → NOA rollout (predicted)
                                    ↓
                    ┌─────────────────────────────────┐
                    │        Loss Computation         │
                    ├─────────────────────────────────┤
                    │ L_traj   = MSE(NOA, CNO)       │
                    │ L_commit = VQ commitment        │
                    │ L_latent = NOA ↔ VQ alignment  │
                    └─────────────────────────────────┘
                                    ↓
                    Backprop through last 32 steps (TBPTT)
```

### NOA Components

1. **U-AFNO Operator**: Spectral mixing in Fourier domain
2. **Latent Projector**: Maps U-AFNO bottleneck → VQ latent space (optional, for L_latent)
3. **CNO Replayer**: Generates ground truth trajectories from saved parameters
4. **VQ-VAE Encoder**: Extracts behavioral features → discrete tokens (frozen)

---

## Quick Start

### Basic Training (Physics Only)

Train NOA to match CNO rollouts without VQ-VAE alignment:

```bash
poetry run python scripts/dev/train_noa_state_supervised.py \
    --dataset datasets/100k_full_features.h5 \
    --n-samples 5000 \
    --epochs 10 \
    --batch-size 4 \
    --lr 3e-4 \
    --bptt-window 32 \
    --timesteps 256
```

**Expected**: `L_traj` decreases from ~600 → <10 over 10 epochs.

### Training with VQ-VAE Alignment (Recommended)

Add VQ-VAE commitment loss for better manifold adherence:

```bash
poetry run python scripts/dev/train_noa_state_supervised.py \
    --dataset datasets/100k_full_features.h5 \
    --vqvae-path checkpoints/production/100k_3family_v1 \
    --n-samples 5000 \
    --epochs 10 \
    --batch-size 4 \
    --lr 3e-4 \
    --bptt-window 32 \
    --timesteps 256 \
    --lambda-commit 0.5
```

**Expected**: `L_commit` stays low (~0.0005), indicating NOA outputs are VQ-tokenizable.

### Full Training with L_latent (Advanced)

Enable latent alignment for representation learning:

```bash
poetry run python scripts/dev/train_noa_state_supervised.py \
    --dataset datasets/100k_full_features.h5 \
    --vqvae-path checkpoints/production/100k_3family_v1 \
    --n-samples 5000 \
    --epochs 10 \
    --batch-size 4 \
    --lr 3e-4 \
    --bptt-window 32 \
    --warmup-steps 500 \
    --timesteps 256 \
    --lambda-commit 0.5 \
    --enable-latent-loss \
    --lambda-latent 0.5 \
    --latent-sample-steps 8 \
    --save-every 200
```

**Expected**:
- `L_traj`: 600 → <10
- `L_commit`: ~0.0005 (stable)
- `L_latent`: 0.7 → 0.5 (alignment improving)

---

## Training Configuration

### Required Arguments

| Argument | Description | Recommended Value |
|----------|-------------|-------------------|
| `--dataset` | Path to HDF5 dataset | `datasets/100k_full_features.h5` |
| `--n-samples` | Number of training samples | 5000-10000 |
| `--epochs` | Training epochs | 10-20 |
| `--batch-size` | Batch size (GPU memory limited) | 4 |
| `--lr` | Learning rate | 3e-4 |
| `--bptt-window` | Truncated BPTT window | 32 |
| `--timesteps` | Rollout length | 256 |

### VQ-VAE Alignment (Optional)

| Argument | Description | Recommended Value |
|----------|-------------|-------------------|
| `--vqvae-path` | Path to VQ-VAE checkpoint | `checkpoints/production/100k_3family_v1` |
| `--lambda-commit` | Commitment loss weight | 0.5 |
| `--enable-latent-loss` | Enable L_latent | (flag) |
| `--lambda-latent` | Latent alignment weight | 0.1-0.5 |
| `--latent-sample-steps` | Timesteps to sample for L_latent | 3-8 |

### LR Scheduling

| Argument | Description | Recommended Value |
|----------|-------------|-------------------|
| `--lr-schedule` | Schedule type | `cosine` |
| `--warmup-steps` | Warmup batches | 500 |

### Checkpointing

| Argument | Description | Recommended Value |
|----------|-------------|-------------------|
| `--checkpoint-dir` | Checkpoint directory | `checkpoints/noa` |
| `--save-every` | Save every N batches | 200 |
| `--early-stop-patience` | Stop if no improvement for N epochs | 2 |

### Model Architecture

| Argument | Description | Recommended Value |
|----------|-------------|-------------------|
| `--base-channels` | U-AFNO base channels | 32 |
| `--encoder-levels` | U-Net encoder levels | 3 |
| `--modes` | Fourier modes | 16 |
| `--afno-blocks` | AFNO blocks per level | 4 |

---

## Loss Functions

### L_traj: Physics Fidelity

**What it measures**: How well NOA matches CNO ground truth trajectories.

**Computation**:
```python
L_traj = MSE(NOA_rollout, CNO_rollout)  # [B, T, C, H, W]
```

**Interpretation**:
- `L_traj = 600`: Random initialization
- `L_traj = 50-100`: Learning basic dynamics
- `L_traj = 10-20`: Good physics matching
- `L_traj < 5`: Excellent physics fidelity

**Why it's needed**: Core objective for learning operator dynamics.

### L_commit: VQ Manifold Adherence

**What it measures**: How easily VQ-VAE can tokenize NOA outputs.

**Computation**:
```python
features = extract_features(NOA_rollout)
z = VQ_encode(features)  # Pre-quantization latents
z_q = quantize(z)        # Nearest codebook vectors
L_commit = MSE(z, z_q.detach())
```

**Interpretation**:
- `L_commit ≈ 0.0005`: NOA outputs are on VQ manifold (good)
- `L_commit > 0.001`: NOA drifting off manifold (concerning)
- `L_commit increasing`: NOA learning physics that VQ-VAE can't represent

**Why it's needed**: Ensures NOA outputs remain tokenizable for downstream applications.

### L_latent: Representation Alignment

**What it measures**: Alignment between NOA's internal features and VQ-VAE's learned embeddings.

**Computation**:
```python
# Extract NOA bottleneck features [B, 256, 8, 8]
noa_bottleneck = NOA.get_intermediate_features(state_t, "bottleneck")

# Project to VQ space [B, 780]
noa_latents = projector(noa_bottleneck)

# Get VQ latents from features
features = extract_features(NOA_rollout)
vq_latents = VQ_encode(features)

# Align (sample N timesteps for efficiency)
L_latent = MSE(mean(noa_latents_sampled), vq_latents.detach())
```

**Interpretation**:
- `L_latent = 0.7`: Random initialization
- `L_latent = 0.5`: Moderate alignment (good)
- `L_latent = 0.3`: Strong alignment (excellent)
- `L_latent < 0.1`: Very strong alignment (rare)

**Why it's needed**:
- Encourages NOA to learn VQ-VAE's behavioral representations
- Enables interpretability (NOA features → VQ codes)
- Improves transfer learning to downstream tasks

**Memory tradeoff**: Sampling fewer timesteps (3-8) reduces overhead from 48% → 15-20%.

---

## Checkpointing and Resume

### Checkpoint Contents

New checkpoints (saved after implementing resume) include:

```python
checkpoint = {
    "model_state_dict": ...,      # NOA weights
    "optimizer_state_dict": ...,  # Adam state
    "scheduler_state_dict": ...,  # LR schedule
    "epoch": 3,                   # Current epoch (0-indexed)
    "global_step": 675,           # Total batches processed
    "history": {                  # Loss curves
        "train_loss": [...],
        "val_loss": [...]
    },
    "best_val_loss": 12.345,      # Best validation so far
    "alignment_state": ...,       # Latent projector weights (if L_latent enabled)
    "config": {                   # Model architecture
        "base_channels": 32,
        "encoder_levels": 3,
        "modes": 16,
        "afno_blocks": 4,
    },
    "args": {...}                 # Full training args
}
```

### Automatic Checkpointing

Checkpoints are saved automatically:

1. **Periodic**: Every `--save-every` batches (e.g., `step_200.pt`, `step_400.pt`)
2. **Per-Epoch**: After each epoch (`epoch_1.pt`, `epoch_2.pt`)
3. **Best Model**: When validation loss improves (`best_model.pt`)

### Resuming Training

#### From Latest Checkpoint

```bash
# Resume from last saved checkpoint
poetry run python scripts/dev/train_noa_state_supervised.py \
    --resume checkpoints/noa/epoch_5.pt \
    --dataset datasets/100k_full_features.h5 \
    --vqvae-path checkpoints/production/100k_3family_v1 \
    --epochs 10 \
    --batch-size 4 \
    --enable-latent-loss \
    --lambda-latent 0.5
```

**What happens**:
1. Loads model, optimizer, scheduler state
2. Resumes from epoch 5, continues to epoch 10
3. Preserves training history and best validation loss
4. LR schedule continues from correct step (no warmup restart)
5. Loads projector weights if L_latent was enabled

#### From Mid-Epoch Checkpoint

```bash
# Resume from step checkpoint (e.g., batch 200 of epoch 1)
poetry run python scripts/dev/train_noa_state_supervised.py \
    --resume checkpoints/noa/step_200.pt \
    --dataset datasets/100k_full_features.h5 \
    --vqvae-path checkpoints/production/100k_3family_v1 \
    --epochs 5 \
    --batch-size 4 \
    --enable-latent-loss
```

**What happens**:
1. Detects checkpoint is from batch 200 of epoch 1
2. **Skips first 200 batches** of epoch 1 (already processed)
3. Continues from batch 201
4. LR synced to step 200 (correct value, no warmup)

#### Expected Resume Output

```
Resuming from checkpoint: checkpoints/noa/step_200.pt
  ✓ Loaded model weights
  ✓ Loaded optimizer state
  ✓ Loaded scheduler state
  ✓ Resuming from epoch 1, step 200
    (Will skip first 200 batches of epoch 1)
  ✓ Best val loss so far: 15.234567
  ✓ Loaded latent projector weights

Epoch 1/5
  Skipping first 200 batches (already processed)...
  [201/225] loss=13.88 traj=13.54 commit=0.000561 latent=0.534 lr=1.51e-04 8.2s/b
  [202/225] loss=13.82 traj=13.48 commit=0.000560 latent=0.533 lr=1.51e-04 8.1s/b
  ...
```

### Old Checkpoint Format (Pre-Resume)

If resuming from checkpoints saved before resume functionality was added:

```
Resuming from checkpoint: checkpoints/noa/step_200.pt
  ✓ Loaded model weights
  ✓ Loaded optimizer state
  ⚠ Old checkpoint format detected - inferred global_step=200 from filename
  ⚠ Syncing scheduler to step 200...
  ✓ Scheduler synced to step 200, lr=1.50e-04
  ✓ Resuming from epoch 1, step 200
    (Will skip first 200 batches of epoch 1)
  ✓ No validation history (first epoch incomplete)
  ⚠⚠ WARNING: Old checkpoint has no projector weights!
      Projector will restart from random initialization.
      L_latent training will be inconsistent with pre-crash training.
      Recommend: Either disable --enable-latent-loss or train from scratch.
```

**Recommendations**:
1. **If L_latent is critical**: Train from scratch to get consistent projector training
2. **If L_latent is optional**: Disable `--enable-latent-loss` and resume with L_commit only
3. **Accept inconsistency**: Resume with L_latent, but projector will reinitialize (loss curve will jump)

### Best Practices

1. **Save frequently**: Use `--save-every 200` for large datasets
2. **Monitor checkpoints**: Check `checkpoints/noa/` periodically to ensure saves are working
3. **Keep best model**: Always preserve `best_model.pt` for deployment
4. **Clean up**: Delete old `step_*.pt` files to save disk space
5. **Test resume**: After first epoch, try resuming to verify checkpoint format

---

## Hyperparameter Tuning

### λ_commit: Commitment Loss Weight

**What it controls**: How strongly NOA is pushed toward VQ manifold.

| Value | Effect | When to Use |
|-------|--------|-------------|
| 0.0 | No VQ alignment | Testing physics learning only |
| 0.1 | Weak alignment | VQ-VAE already well-matched to data |
| 0.5 | **Recommended** | Standard training |
| 1.0 | Strong alignment | NOA drifting off manifold |
| 2.0+ | Very strong | Force manifold adherence (may hurt physics) |

**Tuning guide**:
- If `L_commit` increasing during training → increase λ_commit
- If `L_traj` not decreasing → decrease λ_commit (too much constraint)

### λ_latent: Latent Alignment Weight

**What it controls**: How strongly NOA's internal features align with VQ latents.

| Value | Effect | When to Use |
|-------|--------|-------------|
| 0.0 | No latent alignment | Baseline (L_traj + L_commit only) |
| 0.1 | Weak alignment | Initial experiments |
| 0.5 | **Recommended** | Strong alignment without compromising physics |
| 1.0 | Very strong | Prioritize representation learning |
| 2.0+ | Dominant | Force alignment (may hurt physics) |

**Tuning guide**:
- Start with 0.1, increase to 0.5 if `L_latent` plateaus
- If `L_traj` convergence slows → decrease λ_latent
- If `L_latent` doesn't decrease → increase λ_latent or `--latent-sample-steps`

**Ablation results** (preliminary):
```
λ_latent=0.1, n_samples=3:  L_latent: 0.646 → 0.580 (plateau)
λ_latent=0.5, n_samples=8:  L_latent: 0.705 → 0.561 (2× faster, breaks plateau)
```

### --latent-sample-steps: Timestep Sampling

**What it controls**: How many trajectory timesteps to sample for L_latent computation.

| Value | Memory Overhead | Latent Loss Quality | When to Use |
|-------|----------------|---------------------|-------------|
| 3 | +15% | Good (first, middle, last) | **Recommended**, memory limited |
| 8 | +48% | Better (rich temporal context) | Strong alignment needed |
| -1 | +200% | Best (all timesteps) | Small BPTT windows only |

**Tradeoff**: More samples → richer alignment signal but slower training.

### Batch Size vs GPU Memory

| GPU VRAM | Batch Size | Notes |
|----------|-----------|-------|
| 8 GB | 1-2 | May OOM with L_latent |
| 16 GB | 4 | **Recommended** |
| 24 GB | 8 | Faster convergence |
| 40 GB+ | 16 | Diminishing returns |

**If OOM**:
1. Reduce `--batch-size` (4 → 2)
2. Reduce `--latent-sample-steps` (8 → 3)
3. Disable `--enable-latent-loss`

### Learning Rate Schedule

**Recommended**: Cosine annealing with warmup

```bash
--lr 3e-4 \
--lr-schedule cosine \
--warmup-steps 500
```

**Why warmup**: Prevents early instability when optimizer hasn't seen data yet.

**Warmup schedule**:
```
Steps 0-500:    LR ramps 3e-5 → 3e-4 (linear)
Steps 500+:     LR decays 3e-4 → 0 (cosine)
```

---

## Diagnostics

### During Training

Monitor these metrics every epoch:

```
Epoch 5/10
  Train: total=8.5432 traj=8.0123 commit=0.000543 latent=0.529 [1234.5s]
  Val:   total=9.1234 traj=8.5678 commit=0.000556 latent=0.556
```

**Health checks**:
- ✅ `L_traj` decreasing steadily
- ✅ `L_commit` stable around 0.0005
- ✅ `L_latent` decreasing (if enabled)
- ❌ `L_commit` increasing → NOA drifting off manifold
- ❌ `L_latent` stuck → increase λ_latent or sample more timesteps
- ❌ `L_traj` not decreasing → learning rate too high/low

### Post-Training Evaluation

Run comprehensive diagnostic after training:

```bash
poetry run python scripts/dev/diagnose_latent_alignment.py \
    --noa-checkpoint checkpoints/noa/best_model.pt \
    --vqvae-path checkpoints/production/100k_3family_v1 \
    --dataset datasets/100k_full_features.h5 \
    --n-samples 100 \
    --timesteps 256
```

**Output**:
```
============================================================
L_latent Alignment Diagnostics
============================================================

1. VQ Reconstruction Quality
Total MSE: 0.3245
Per-category reconstruction errors:
  INITIAL     : 0.2134
  SUMMARY     : 0.3891
  TEMPORAL    : 0.3710
➜ Quality Assessment: Good

2. Token Diversity
Overall utilization: 67.3% (1234/1834 codes)
Token entropy: 5.83
➜ Diversity Assessment: Good

3. Alignment Correlation
Cosine similarity: 0.623 ± 0.084
L_latent (MSE): 0.521
➜ Correlation Assessment: Moderate

4. Temporal Consistency
Latent norm: 12.345 ± 0.678
Coefficient of variation: 0.055
➜ Consistency Assessment: Good

Overall Summary
VQ Reconstruction:      Good
Token Diversity:        Good
Alignment Correlation:  Moderate
Temporal Consistency:   Good

➜ Final Verdict: GOOD - L_latent provides meaningful alignment
```

**Interpretation**:
- **VQ Reconstruction < 0.5**: NOA outputs tokenize well
- **Token Diversity > 50%**: NOA explores diverse behaviors
- **Cosine Similarity > 0.5**: Moderate alignment achieved
- **CV < 0.1**: Stable alignment across trajectory

See [Diagnostics](#diagnostics) section for detailed metric definitions.

---

## Troubleshooting

### Training Crashes / OOM

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:
1. Reduce batch size: `--batch-size 4` → `--batch-size 2`
2. Reduce latent sampling: `--latent-sample-steps 8` → `--latent-sample-steps 3`
3. Disable L_latent: Remove `--enable-latent-loss`
4. Check GPU usage: `nvidia-smi` to see if other processes are using memory

### NaN Gradients

**Symptoms**:
```
Warning: NaN/Inf gradients at batch 157, skipping update
```

**Solutions**:
1. **Already handled**: Training skips NaN batches automatically
2. If frequent (>10% of batches): Reduce `--lr` (3e-4 → 1e-4)
3. If persistent: Check dataset for NaN values

### L_latent Not Decreasing

**Symptoms**:
```
Epoch 5: latent=0.65
Epoch 10: latent=0.64  (barely changed)
```

**Solutions**:
1. Increase λ_latent: `0.1` → `0.5`
2. Sample more timesteps: `--latent-sample-steps 3` → `--latent-sample-steps 8`
3. Train longer: May need 20+ epochs for strong alignment
4. Check projector is learning: Load checkpoint and verify weights changed

### L_commit Increasing

**Symptoms**:
```
Epoch 1: commit=0.0005
Epoch 5: commit=0.0015  (increasing!)
```

**Root cause**: NOA learning physics that VQ-VAE can't represent.

**Solutions**:
1. Increase λ_commit: `0.5` → `1.0`
2. Check VQ-VAE quality: May need to retrain VQ-VAE on more diverse data
3. Reduce λ_latent: May be pulling NOA off manifold

### Resume Warnings

**Symptom**:
```
⚠⚠ WARNING: Old checkpoint has no projector weights!
    Projector will restart from random initialization.
```

**Solution**:
1. Train from scratch (recommended for clean L_latent)
2. Disable `--enable-latent-loss` and resume without L_latent
3. Accept inconsistency (projector reinitializes)

### Different Loss Values When Resuming

**Symptom**: Loss at batch 1 of resumed training doesn't match loss at batch 225 of crashed training.

**Root cause**: DataLoader reshuffles data each epoch.

**Solution**: This is expected! Training loop now skips already-processed batches, so loss values will match the original run once it reaches batch 201.

---

## Advanced Topics

### Custom VQ-VAE Configurations

If using a different VQ-VAE architecture:

```bash
# System automatically infers VQ latent dimension
poetry run python scripts/dev/train_noa_state_supervised.py \
    --vqvae-path checkpoints/my_custom_vqvae \
    --enable-latent-loss
```

**Supported**: Any VQ-VAE with `.encode()` method that returns list of latents.

### Multi-GPU Training

Not yet supported. Stay tuned for distributed training implementation.

### Transfer Learning

Use trained NOA as initialization for domain-specific tasks:

```bash
# Train on general dataset
poetry run python scripts/dev/train_noa_state_supervised.py \
    --dataset datasets/100k_full_features.h5 \
    --epochs 20 \
    --checkpoint-dir checkpoints/noa_pretrained

# Fine-tune on domain-specific data
poetry run python scripts/dev/train_noa_state_supervised.py \
    --resume checkpoints/noa_pretrained/best_model.pt \
    --dataset datasets/domain_specific.h5 \
    --epochs 5 \
    --lr 1e-4  # Lower LR for fine-tuning
```

---

## See Also

- [NOA Roadmap](noa-roadmap.md) - Phase 0-3 implementation plan
- [Architecture](architecture.md) - System design
- [Debugging Guide](debugging/noa-nan-gradient-diagnosis.md) - NaN gradient troubleshooting
- [VQ-VAE Training](../README.md#vq-vae-behavioral-tokenization) - Tokenizer training

---

**Last Updated**: 2026-01-07
**Status**: Phase 1 In Development (core training working, L_latent operational)
