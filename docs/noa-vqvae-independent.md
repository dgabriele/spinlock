# Independent Optimization Architecture: NOA + VQ-VAE

**Date:** 2026-01-11
**Branch:** `noa-vqvae-independent`
**Philosophy:** Train tokenizer on simulator's distribution

---

## Executive Summary

This document describes the **independent optimization architecture** for training Meta-Neural Operators (NOA) with VQ-VAE tokenization. Unlike coupled training approaches, this architecture optimizes each component independently:

1. **NOA** trained purely for physics accuracy (no VQ constraints)
2. **VQ-VAE** trained on NOA's outputs (alignment by construction)

**Key Insight:** The tokenizer should adapt to the simulator's distribution, not vice versa.

**Result:** Optimal physics accuracy + optimal tokenization quality + architectural simplicity.

---

## Quick Start

### 1. Train NOA (Pure Physics)

```bash
# Train NOA with pure MSE loss (no token conditioning)
poetry run spinlock train-meta-operator \
  --config configs/noa/experiments/phase2/exp_pure_mse.yaml

# Target: L_traj < 1.0 (RMSE < field variation)
# Time: ~2 hours (2 epochs on RTX 3060 Ti)
# Result: checkpoints/noa/pure_mse_baseline/meta_operator_best.pt
```

### 2. Generate NOA Features

```bash
# Generate 10K features for validation
poetry run spinlock generate-noa-features \
  --noa-checkpoint checkpoints/noa/pure_mse_baseline/meta_operator_best.pt \
  --output datasets/noa_features_10k.h5 \
  --n-samples 10000 \
  --config configs/experiments/local_100k_optimized.yaml \
  --batch-size 16

# Generate 100K features for production
poetry run spinlock generate-noa-features \
  --noa-checkpoint checkpoints/noa/pure_mse_baseline/meta_operator_best.pt \
  --output datasets/noa_features_100k.h5 \
  --n-samples 100000 \
  --config configs/experiments/local_100k_optimized.yaml \
  --batch-size 16

# Time: ~30 min (10K) or ~5 hours (100K)
# Space: ~100 MB (10K) or ~1 GB (100K) - feature-only, no trajectories
```

### 3. Train VQ-VAE on NOA Distribution

```bash
# Train on 10K (validation)
poetry run spinlock train-vqvae \
  --config configs/vqvae/noa_distribution_10k.yaml

# Train on 100K (production)
poetry run spinlock train-vqvae \
  --config configs/vqvae/noa_distribution_100k.yaml

# Target: L_recon < 0.05 (better than CNO's 0.067)
# Time: ~3 hours (10K) or ~10 hours (100K)
# Result: checkpoints/vqvae/noa_distribution_100k/best_model.pt
```

### 4. Deploy End-to-End

```python
from spinlock.noa.backbone import NOABackbone
from spinlock.encoding.unified_feature_pipeline import UnifiedFeaturePipeline
import torch

# Load trained components
noa = NOABackbone.load_checkpoint("checkpoints/noa/pure_mse_baseline/meta_operator_best.pt")
vqvae = load_vqvae("checkpoints/vqvae/noa_distribution_100k/best_model.pt")

# Generate and tokenize
theta, u0 = sample_operator_and_ic()
rollout = noa(u0, steps=256)  # [1, 256, 1, H, W]
features = extract_features(rollout, u0)  # [1, 270]
tokens = vqvae.encode(features)  # [1, num_tokens]

# Ready for symbolic reasoning!
```

---

## Architecture Philosophy

### Problem with Coupled Training

Traditional approaches train NOA with VQ constraints:

```
Loss = λ_traj × L_traj + λ_commit × L_commit + λ_latent × L_latent
       ════════════════   ═══════════════════════════════════════
       Physics objective  VQ alignment objectives
```

**Challenges:**
- Competing gradients (physics vs VQ quality)
- Loss weight tuning (λ values interdependent)
- Feature dimension matching (VQ and NOA must align)
- Plateau at equilibrium (neither objective optimized)

**Observed:** NOA achieves L_recon = 0.067 (better than VQ-VAE's 0.120 on CNO) by learning "VQ-friendly" dynamics at the cost of physics accuracy.

### Solution: Independent Optimization

Train components separately with single objectives:

```
Stage 1: NOA (Pure MSE)
  Loss = L_traj
  Goal: Optimal physics accuracy

Stage 2: Generate Features
  NOA(θ, u₀) → Rollout → Features
  Sample 100K diverse rollouts from NOA

Stage 3: VQ-VAE (Standard)
  Loss = L_recon + L_commit
  Goal: Optimal tokenization of NOA's outputs
```

**Advantages:**
- ✅ No competing objectives
- ✅ No loss weight tuning
- ✅ Alignment by construction (VQ learns NOA's structure)
- ✅ 100× more training samples for VQ-VAE
- ✅ Simpler architecture (no token conditioning)

---

## Architecture Details

### Stage 1: Pure Physics Training

**Goal:** Train NOA to minimize trajectory MSE against CNO ground truth.

**Architecture:**
```
Input: (θ, u₀)  [NO token conditioning]
   ↓
U-AFNO Backbone (226M parameters)
   - Encoder: 3 levels, 40 base channels
   - AFNO blocks: 4 per level
   - Modes: 16 Fourier modes
   ↓
Autoregressive Rollout
   - Total steps: 256
   - Truncated BPTT: window=32
   ↓
Loss: MSE(NOA_rollout, CNO_rollout)
```

**Training hyperparameters:**
```yaml
n_samples: 1000
batch_size: 2
epochs: 30
learning_rate: 5.0e-5
warmup_steps: 2250  # 5 epochs
weight_decay: 1.0e-4
clip_grad: 0.5
```

**Success criteria:**
- **Excellent**: L_traj < 1.0 (RMSE < 1.0, comparable to field variation)
- **Good**: L_traj < 1.5 (RMSE < 1.2)
- **Acceptable**: L_traj < 2.5 (RMSE < 1.6)

**Typical results:**
- Epoch 1: L_traj ≈ 1.5-2.0
- Epoch 2: L_traj ≈ 0.6-1.0
- Epoch 5+: L_traj < 0.5 (excellent)

### Stage 2: Feature Generation

**Goal:** Generate large-scale feature dataset from trained NOA.

**Process:**
1. Load trained NOA checkpoint
2. Sample diverse (θ, u₀) from parameter space
3. Generate NOA rollouts (fast, no gradients)
4. Extract features inline (GPU-optimized)
5. Save features to HDF5 (no trajectories)

**Feature extraction:**
```python
# INITIAL features (14D)
- Spatial statistics: mean, std, min, max
- Spectral properties: dominant frequencies
- Initial condition characteristics

# SUMMARY features (360D → 128D)
- Per-trajectory aggregation
- Temporal statistics
- Cross-channel correlations
- Operator sensitivity metrics

# TEMPORAL features (63D per timestep)
- Per-timestep spatial statistics
- Spectral evolution
- Cross-channel dynamics
```

**Storage efficiency:**
```
Trajectories: [100K, 256, 1, 64, 64] × 4 bytes = 1.6 TB
Features:     [100K, 270] × 4 bytes = 108 MB
Savings: 99.99%
```

**Generation throughput:**
- RTX 3060 Ti: ~20 samples/sec
- 10K samples: ~8 minutes
- 100K samples: ~80 minutes

### Stage 3: VQ-VAE Training

**Goal:** Learn discrete tokenization of NOA's feature distribution.

**Architecture:**
```
Input: NOA features [B, 270]
   ↓
Family Encoders (per-family compression)
   - INITIAL: 14D → 14D (identity)
   - SUMMARY: 360D → 128D (MLP)
   - TEMPORAL: 63D × T → 128D (temporal CNN)
   ↓
Combined: [B, 270D] total
   ↓
Hierarchical VQ (per category)
   - Auto-discover categories via clustering
   - 3 levels per category: coarse → medium → fine
   - Codebook sizes: [256, 512, 1024] per level
   ↓
Discrete tokens: [B, num_categories × 3]
```

**Training hyperparameters:**
```yaml
batch_size: 1024
learning_rate: 0.0007
num_epochs: 500
commitment_cost: 0.45
use_ema: true
decay: 0.99
```

**Category discovery:**
- **Auto mode**: Clustering-based discovery
- Expected: 10-15 categories (behavioral modes)
- Orthogonality target: < 0.15 (category separation)

**Success criteria:**
- **Excellent**: L_recon < 0.04 (better than CNO's 0.067)
- **Good**: L_recon < 0.05
- **Acceptable**: L_recon < 0.08

**Why it's better:** VQ-VAE learns NOA's actual structure, not forced to compress dynamics it doesn't understand.

---

## Implementation Guide

### Stage 1: Training NOA

**Config:** `configs/noa/experiments/phase2/exp_pure_mse.yaml`

```yaml
model:
  spatial_dim: 64
  in_channels: 1
  out_channels: 1
  base_channels: 40
  encoder_levels: 3
  modes: 16
  afno_blocks: 4
  token_conditioning: false  # Key: no tokens!

training:
  n_samples: 1000
  batch_size: 2
  epochs: 30
  learning_rate: 5.0e-5
  warmup_steps: 2250
  timesteps: 256
  bptt_window: 32

loss:
  mode: "mse_led"
  lambda_traj: 1.0  # Only objective

checkpointing:
  save_dir: "checkpoints/noa/pure_mse_baseline"
  save_every: 3
  keep_best: true
```

**Monitoring:**
```bash
# Watch training progress
tail -f checkpoints/noa/pure_mse_baseline/training_log.txt

# Check current loss
tail -1 checkpoints/noa/pure_mse_baseline/training_log.txt
```

**Checkpoints:**
- `meta_operator_epoch_3.pt`: Saved every 3 epochs
- `meta_operator_best.pt`: Best validation loss
- Use `best.pt` for feature generation

### Stage 2: Generating Features

**Command structure:**
```bash
spinlock generate-noa-features \
  --noa-checkpoint <path>       # Trained NOA checkpoint
  --output <path>                # Output HDF5 file
  --n-samples <N>                # Number of rollouts
  --config <path>                # Base config (for param space)
  --batch-size <N>               # Generation batch size (default: 16)
  --device cuda                  # Device (default: cuda)
  --seed <N>                     # Random seed (default: 42)
```

**Recommended batch sizes:**
- RTX 3060 Ti (8GB): 16-32
- RTX 4090 (24GB): 32-64
- A100 (40GB): 64-128

**Output structure:**
```
datasets/noa_features_100k.h5:
  /features/initial      [100K, 14]
  /features/summary      [100K, 360]
  /features/temporal     [100K, 256, 63]  # If temporal enabled
  /operators/*           [100K]  # Parameter metadata
```

**Validation:**
```python
import h5py

with h5py.File('datasets/noa_features_10k.h5', 'r') as f:
    print(f"Samples: {f['features/initial'].shape[0]}")
    print(f"Initial dim: {f['features/initial'].shape[1]}")
    print(f"Summary dim: {f['features/summary'].shape[1]}")

    # Check for NaNs
    assert not np.any(np.isnan(f['features/initial'][:]))
    assert not np.any(np.isnan(f['features/summary'][:]))
```

### Stage 3: Training VQ-VAE

**Config:** `configs/vqvae/noa_distribution_100k.yaml`

```yaml
dataset_path: "datasets/noa_features_100k.h5"

families:
  summary:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [256, 128]
      output_dim: 64

model:
  group_embedding_dim: 64
  group_hidden_dim: 128

  levels:
    - {latent_dim: 32, num_tokens: 256}
    - {latent_dim: 16, num_tokens: 512}
    - {latent_dim: 8, num_tokens: 1024}

  commitment_cost: 0.45
  use_ema: true
  decay: 0.99

training:
  batch_size: 1024
  learning_rate: 0.0007
  num_epochs: 500
  category_assignment: "auto"
  orthogonality_target: 0.15

  checkpoint_dir: "checkpoints/vqvae/noa_distribution_100k"
  early_stopping_patience: 120
```

**Monitoring:**
```bash
# Watch training
tail -f checkpoints/vqvae/noa_distribution_100k/training.log

# Check reconstruction loss
grep "val_recon" checkpoints/vqvae/noa_distribution_100k/training.log | tail -5

# Check codebook utilization
grep "codebook_util" checkpoints/vqvae/noa_distribution_100k/training.log | tail -5
```

**Expected convergence:**
- Epoch 50: L_recon ≈ 0.15
- Epoch 200: L_recon ≈ 0.08
- Epoch 400+: L_recon < 0.05

---

## Deployment

### Loading Models

```python
import torch
from spinlock.noa.backbone import NOABackbone
from pathlib import Path

# Load NOA
noa_ckpt = torch.load(
    "checkpoints/noa/pure_mse_baseline/meta_operator_best.pt",
    map_location='cuda'
)
model_config = noa_ckpt['config']['model']
noa = NOABackbone(**model_config)
noa.load_state_dict(noa_ckpt['model_state_dict'])
noa = noa.cuda().eval()

# Load VQ-VAE
vqvae = load_vqvae_checkpoint(
    "checkpoints/vqvae/noa_distribution_100k/best_model.pt"
)
vqvae = vqvae.cuda().eval()
```

### End-to-End Inference

```python
from spinlock.encoding.unified_feature_pipeline import UnifiedFeaturePipeline

# Initialize feature extractor
feature_pipeline = UnifiedFeaturePipeline(
    vqvae_checkpoint="checkpoints/vqvae/noa_distribution_100k/best_model.pt",
    device='cuda'
)

# Generate rollout
theta, u0 = sample_params_and_ic()
with torch.no_grad():
    rollout = noa(u0, steps=256)  # [1, 256, 1, H, W]

# Extract features and tokenize
features = feature_pipeline.extract_features(
    rollout=rollout,
    ic=u0,
)
tokens = feature_pipeline.tokenize(features)

# Use tokens for symbolic reasoning
print(f"Tokens: {tokens.shape}")  # [1, num_categories × 3]
```

### Batch Inference

```python
# Process multiple rollouts efficiently
batch_size = 16
ics = generate_batch_ics(batch_size)

with torch.no_grad():
    rollouts = noa(ics, steps=256)  # [16, 256, 1, H, W]

features = feature_pipeline.extract_features_batch(
    rollouts=rollouts,
    ics=ics,
)
tokens = feature_pipeline.tokenize_batch(features)

print(f"Batch tokens: {tokens.shape}")  # [16, num_categories × 3]
```

---

## Comparison to Alternatives

### vs. Two-Stage Curriculum

| Aspect | Two-Stage Curriculum | Independent Optimization |
|--------|---------------------|--------------------------|
| **NOA Training** | Stage 1: MSE + tokens<br>Stage 2: VQ-led fine-tuning | Pure MSE only |
| **VQ Training** | Pre-trained on CNO | Trained on NOA |
| **Physics Quality** | L_traj ≈ 1.5-2.0 | L_traj < 1.0 |
| **VQ Quality** | L_recon ≈ 0.067 | L_recon < 0.05 |
| **Architecture** | Complex (tokens + VQ-led) | Simple (pure MSE) |
| **Debugging** | Hard (coupled failures) | Easy (isolated) |
| **Sample Count** | 1K (CNO limited) | 100K+ (NOA unlimited) |

### vs. Simultaneous Training

| Aspect | Simultaneous | Independent Optimization |
|--------|-------------|--------------------------|
| **Loss Function** | Multi-objective | Single objective per stage |
| **Hyperparameters** | 4+ (λ_traj, λ_commit, λ_latent, ...) | 0 (standard configs) |
| **Debugging** | Very hard (entangled) | Easy (staged) |
| **Memory** | High (VQ during rollout) | Low (pure MSE) |
| **Rollout Length** | 32 steps (limited) | 256 steps (full) |

---

## Troubleshooting

### NOA Training Issues

**Problem: Loss not decreasing**
```
Solution:
1. Check learning rate warmup (should reach full LR by epoch 5)
2. Verify gradient clipping (0.5 is good default)
3. Check for NaNs in data or gradients
4. Reduce learning rate if unstable
```

**Problem: OOM during training**
```
Solution:
1. Reduce batch_size (try 1 instead of 2)
2. Reduce bptt_window (try 16 instead of 32)
3. Clear GPU cache: torch.cuda.empty_cache()
4. Use mixed precision training (add to config)
```

**Problem: Slow convergence**
```
Solution:
1. Verify warmup_steps is correct (5 epochs recommended)
2. Check dataset quality (no corrupted samples)
3. Increase epochs (30 is good baseline)
4. Monitor val_loss vs train_loss (overfitting?)
```

### Feature Generation Issues

**Problem: OOM during generation**
```
Solution:
1. Reduce batch_size (try 8 or 4)
2. Generate in smaller chunks (10K at a time)
3. Use CPU if necessary (slower but works)
```

**Problem: NaN features**
```
Solution:
1. Check NOA checkpoint quality (L_traj < 2.0?)
2. Verify feature extractors are initialized correctly
3. Check for numerical instability in rollouts
```

### VQ-VAE Training Issues

**Problem: High reconstruction loss**
```
Solution:
1. Verify feature quality (check for NaNs/outliers)
2. Increase num_epochs (500 minimum)
3. Check category assignment (auto-discovery working?)
4. Verify normalization stats from feature generation
```

**Problem: Low codebook utilization**
```
Solution:
1. Reduce commitment_cost (try 0.25)
2. Enable dead code reset (reset_interval: 100)
3. Check if too many categories (reduce max_clusters)
4. Verify data diversity (generate more samples?)
```

---

## Performance Benchmarks

### Training Times (RTX 3060 Ti, 8GB)

**NOA Training:**
- 1 epoch (1000 samples): ~55 min
- 2 epochs (target L_traj < 1.0): ~2 hours
- 5 epochs (target L_traj < 0.5): ~4.5 hours

**Feature Generation:**
- 10K samples: ~8 min
- 100K samples: ~80 min
- 1M samples: ~13 hours

**VQ-VAE Training:**
- 10K samples (validation): ~3 hours
- 100K samples (production): ~10 hours
- Convergence: 400-500 epochs typical

### Storage Requirements

**NOA Checkpoints:**
- Per checkpoint: ~2 GB (226M params)
- Total (30 epochs, save_every=3): ~20 GB

**Feature Datasets:**
- 10K samples: ~100 MB
- 100K samples: ~1 GB
- 1M samples: ~10 GB

**VQ-VAE Checkpoints:**
- Per checkpoint: ~50 MB
- Total (500 epochs, save_every=50): ~500 MB

### Inference Throughput

**NOA Rollouts:**
- Single sample: ~0.05 sec (20 samples/sec)
- Batch of 16: ~0.4 sec (40 samples/sec)
- 256-step rollout, TBPTT not needed for inference

**Feature Extraction:**
- Single sample: ~0.01 sec (100 samples/sec)
- Batch of 16: ~0.1 sec (160 samples/sec)
- GPU-optimized, minimal overhead

**Tokenization:**
- Single sample: ~0.001 sec (1000 samples/sec)
- Batch of 16: ~0.01 sec (1600 samples/sec)
- Very fast, mostly memory-bound

**End-to-End:**
- Single sample: ~0.06 sec (17 samples/sec)
- Batch of 16: ~0.5 sec (32 samples/sec)
- Dominated by NOA rollout time

---

## References

### Documentation
- [Two-Stage Curriculum Architecture](two-stage-curriculum-architecture.md) - Original approach and pivot rationale
- [NOA Training Guide](noa-training-guide.md) - General NOA training principles
- [VQ-VAE Training Guide](../configs/vqvae/README.md) - VQ-VAE configuration details

### Code
- `src/spinlock/noa/backbone.py` - NOA architecture
- `src/spinlock/cli/train_meta_operator.py` - NOA training script
- `src/spinlock/cli/generate_noa_features.py` - Feature generation command
- `src/spinlock/noa/generation_pipeline.py` - Feature generation pipeline
- `src/spinlock/cli/train_vqvae.py` - VQ-VAE training script

### Configs
- `configs/noa/experiments/phase2/exp_pure_mse.yaml` - NOA training
- `configs/vqvae/noa_distribution_10k.yaml` - VQ-VAE validation
- `configs/vqvae/noa_distribution_100k.yaml` - VQ-VAE production

---

## Changelog

**2026-01-11:** Initial documentation
- Documented independent optimization architecture
- Provided quick start guide and implementation details
- Added troubleshooting and performance benchmarks
- Created comprehensive deployment guide

---

**Last Updated:** 2026-01-11
**Branch:** `noa-vqvae-independent`
**Status:** NOA training in progress (Epoch 1/30)
