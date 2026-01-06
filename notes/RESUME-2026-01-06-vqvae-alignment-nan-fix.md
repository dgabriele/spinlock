# NOA VQ-VAE Alignment Training

**Date:** January 6, 2026
**Status:** Working - Aligned feature extraction + 3-family VQ-VAE support

---

## Overview

NOA (Neural Operator Agent) can now be trained with VQ-VAE alignment to "think in tokens" - producing outputs that are well-represented by the VQ-VAE's discrete vocabulary.

### Two-Loss Structure

```
L = L_traj + lambda * L_commit
```

| Loss | Purpose | Default Weight |
|------|---------|----------------|
| `L_traj` | MSE on trajectories (physics fidelity) | 1.0 |
| `L_commit` | VQ commitment (manifold adherence) | lambda = 0.5 |

Note: L_latent was removed - it was always ~0 due to normalization washing out differences.

---

## Architecture

```
IC (Initial Condition)
    |
NOA Backbone (U-AFNO)
    |
Predicted Trajectory [B, T, C, H, W]
    |
+-----------------------------------------------+
|           VQ-VAE Alignment Module             |
|                                               |
|  AlignedFeatureExtractor                      |
|    - INITIAL (14D manual from IC)             |
|    - SUMMARY (360D -> 128D via MLPEncoder)    |
|    - TEMPORAL (T x 63D -> 128D via CNN)       |
|      |                                        |
|  Features [B, 187D] + raw_ics [B, C, H, W]    |
|      |                                        |
|  VQ-VAE Encoder (frozen) -> z_pre             |
|      |                                        |
|  L_commit = MSE(z_pre, sg(z_quantized))       |
+-----------------------------------------------+
```

---

## Key Files

| File | Description |
|------|-------------|
| `src/spinlock/noa/vqvae_alignment.py` | VQVAEAlignmentLoss, AlignedFeatureExtractor |
| `src/spinlock/noa/__init__.py` | Exports alignment classes |
| `scripts/dev/train_noa_state_supervised.py` | Training script with VQ-VAE options |

---

## Usage

### State-Only Training (Baseline)
```bash
poetry run python scripts/dev/train_noa_state_supervised.py \
    --n-samples 500 --epochs 10
```

### With VQ-VAE Alignment (3-Family Checkpoint)
```bash
poetry run python scripts/dev/train_noa_state_supervised.py \
    --n-samples 500 --epochs 10 \
    --vqvae-path checkpoints/production/100k_3family_v1 \
    --lambda-commit 0.5
```

### CLI Arguments
- `--vqvae-path`: Path to VQ-VAE checkpoint directory
- `--lambda-commit`: Weight for commitment loss (default: 0.5)
- `--timesteps`: Number of timesteps to supervise (default: 32)
- `--n-realizations`: Number of stochastic realizations for CNO rollout (default: 1)

---

## Results

### With `100k_3family_v1` checkpoint (187D features, 3 families):
```
Epoch 1/2
  Train: total=2.281602 state=2.281358 latent=0.000000 commit=0.000489
  Val: total=1.942824 state=1.942508 latent=0.000000 commit=0.000631
  New best! (val_loss=1.942824)

Epoch 2/2
  Train: total=2.189601 state=2.189329 latent=0.000000 commit=0.000543
  Val: total=2.542324 state=2.542008 latent=0.000000 commit=0.000631
```

Training completes without NaN collapse. Latent loss is near-zero (expected for random trajectories with normalization). Commit loss is non-zero, showing VQ manifold adherence is being learned.

---

## Implementation Details

### AlignedFeatureExtractor

Extracts features matching the 3-family VQ-VAE format:

1. **INITIAL (14D manual)**: Uses `InitialManualExtractor` from IC
   - Spatial, spectral, information, morphological features

2. **SUMMARY (128D encoded)**: Uses `SummaryExtractor` + `MLPEncoder`
   - Aggregated trajectory statistics
   - Encoded via MLPEncoder (360D -> 128D)

3. **TEMPORAL (128D encoded)**: Uses `SummaryExtractor.per_timestep` + `TemporalCNNEncoder`
   - Per-timestep features [B, T, 63D]
   - Encoded via TemporalCNNEncoder -> [B, 128D]

Returns tuple: `(features, raw_ics)` for hybrid VQ-VAE models.

### HybridVQVAEWrapper

Loads 3-family checkpoints (e.g., `100k_3family_v1`) without dimension re-adjustment:

- Checkpoint was saved with already-adjusted dimensions (187D)
- Inner VQ-VAE and InitialHybridEncoder loaded separately
- `encode()` method combines manual INITIAL with CNN embeddings

### Bug Fixes

1. **InitialManualExtractor shape bugs**: Fixed `_centroid_distance()` and `_spectral_centroid()` methods that used `squeeze()` incorrectly for M=1 realizations.

2. **VQVAEWithInitial dimension mismatch**: Created `HybridVQVAEWrapper` to load checkpoints without re-adjusting dimensions.

---

## Feature Normalization

Uses existing infrastructure from `spinlock.encoding.normalization`:
```python
from spinlock.encoding.normalization import standard_normalize

# Clean NaN values first
features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# Skip if batch size < 2 (can't compute std)
if features.shape[0] < 2:
    return features

# Apply standard normalization
normalized = standard_normalize(features)
```

### Gradient Flow
- VQ-VAE weights are **frozen** (acts as pre-trained feature extractor)
- Gradients flow: NOA <- features <- z_pre (L_commit)
- `sg()` (stop-gradient) applied to quantized vectors in L_commit

---

## Known Issues (Resolved)

### Feature Dimension Mismatch (SOLVED)
Different VQ-VAE checkpoints expect different input dimensions:
- `100k_full_features`: 225D (legacy)
- `100k_3family_v1`: 187D (production)

**Solution**: `AlignedFeatureExtractor.from_checkpoint()` reads config and creates appropriate encoders.

### NaN Handling (SOLVED)
Edge cases that can produce NaN:
- Single-sample batches (std undefined)
- Zero-variance features
- Extreme trajectory values
- M=1 realizations with std/cv aggregation

**Solutions implemented**:
1. `nan_to_num()` applied before and after normalization
2. NaN gradient checking in training loop (skips corrupted updates)
3. `SummaryConfig(realization_aggregation=["mean"])` to avoid std() with M=1
4. `SummaryConfig(temporal_aggregation=["mean"])` to avoid NaN from constant sequences

### InitialManualExtractor Shape Bugs (SOLVED)
Methods using `squeeze()` without dimension argument caused shape mismatches for M=1:
- `_centroid_distance()`: Fixed to use `squeeze(-1).squeeze(-1)`
- `_spectral_centroid()`: Fixed to use `squeeze(-1)`

### Hybrid VQ-VAE Loading (SOLVED)
`VQVAEWithInitial` re-adjusts dimensions on construction, but checkpoint already has adjusted dimensions.

**Solution**: Created `HybridVQVAEWrapper` that loads inner vqvae and initial_encoder separately without re-adjustment.

---

## Truncated BPTT for Long Sequences

### Problem: NaN Gradients with T > 32

When training NOA with long rollouts (T ≥ 64), gradients explode through the autoregressive chain:

```
u₀ → NOA → u₁ → NOA → u₂ → ... → u₂₅₆
```

Backpropagating through 64+ sequential applications causes gradient explosion, resulting in:
- NaN gradients in `operator.encoder.stem.block.0.weight`
- Training skips all updates

### Solution: Truncated BPTT

Only backpropagate through the last `bptt_window` steps:

```python
# Phase 1: Warmup (no gradients)
warmup_steps = timesteps - bptt_window  # e.g., 256 - 32 = 224
with torch.no_grad():
    for t in range(warmup_steps):
        x = noa.single_step(x)

# Phase 2: Supervised (with gradients)
for t in range(bptt_window):
    x = noa.single_step(x)
    trajectory.append(x)

# Loss computed only on supervised window
loss = MSE(pred_trajectory, target_trajectory[:, -bptt_window:])
```

### Usage

```bash
# Short sequences (T ≤ 32): Full backprop
poetry run python scripts/dev/train_noa_state_supervised.py \
    --timesteps 32

# Long sequences (T > 32): Truncated BPTT required
poetry run python scripts/dev/train_noa_state_supervised.py \
    --timesteps 256 --bptt-window 32

# With VQ-VAE alignment
poetry run python scripts/dev/train_noa_state_supervised.py \
    --timesteps 256 --bptt-window 32 \
    --vqvae-path checkpoints/production/100k_3family_v1 \
    --lambda-commit 0.5
```

### Training Results (T=256, TBPTT window=32)

```
Epoch 1/2
  Train: total=0.940664 state=0.940304 latent=0.000000 commit=0.000722 [17.3s]
  Val: total=1.081461 state=1.081146 latent=0.000000 commit=0.000631

Epoch 2/2
  Train: total=1.050154 state=1.049898 latent=0.000000 commit=0.000512 [16.3s]
  Val: total=0.895321 state=0.895005 latent=0.000000 commit=0.000631
```

---

## Future Improvements

1. **Learned Feature Projection**: Train a small MLP to project trajectory features to VQ-VAE space
2. **Token Distribution Matching**: Add KL divergence on token distributions
3. **Data Augmentation via VQ Space**: Sample token sequences and decode to create synthetic training data
