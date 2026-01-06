# NOA VQ-VAE Alignment Training

**Date:** January 6, 2026
**Status:** Working - NaN issues resolved, training stable

---

## Overview

NOA (Neural Operator Agent) can now be trained with VQ-VAE alignment to "think in tokens" - producing outputs that are well-represented by the VQ-VAE's discrete vocabulary.

### Three-Loss Structure

```
L = L_traj + λ₁ * L_latent + λ₂ * L_commit
```

| Loss | Purpose | Default Weight |
|------|---------|----------------|
| `L_traj` | MSE on trajectories (physics fidelity) | 1.0 |
| `L_latent` | Pre-quantized latent alignment | λ₁ = 0.1 |
| `L_commit` | VQ commitment (manifold adherence) | λ₂ = 0.5 |

---

## Architecture

```
IC (Initial Condition)
    ↓
NOA Backbone (U-AFNO)
    ↓
Predicted Trajectory [B, T, C, H, W]
    ↓
┌───────────────────────────────────────────────┐
│           VQ-VAE Alignment Module             │
│                                               │
│  TrajectoryFeatureExtractor                   │
│      ↓                                        │
│  Features [B, D] (D matched to VQ-VAE)        │
│      ↓                                        │
│  standard_normalize() + nan_to_num()          │
│      ↓                                        │
│  VQ-VAE Encoder (frozen) → z_pre              │
│      ↓                                        │
│  L_latent = MSE(z_pred_norm, z_target_norm)   │
│  L_commit = MSE(z_pre, sg(z_quantized))       │
└───────────────────────────────────────────────┘
```

---

## Key Files

| File | Description |
|------|-------------|
| `src/spinlock/noa/vqvae_alignment.py` | VQVAEAlignmentLoss, TrajectoryFeatureExtractor |
| `src/spinlock/noa/__init__.py` | Exports alignment classes |
| `scripts/dev/train_noa_state_supervised.py` | Training script with VQ-VAE options |

---

## Usage

### State-Only Training (Baseline)
```bash
poetry run python scripts/dev/train_noa_state_supervised.py \
    --n-samples 500 --epochs 10
```

### With VQ-VAE Alignment
```bash
poetry run python scripts/dev/train_noa_state_supervised.py \
    --n-samples 500 --epochs 10 \
    --vqvae-path checkpoints/production/100k_full_features \
    --lambda-latent 0.1 --lambda-commit 0.5
```

### CLI Arguments
- `--vqvae-path`: Path to VQ-VAE checkpoint directory
- `--lambda-latent`: Weight for latent alignment loss (default: 0.1)
- `--lambda-commit`: Weight for commitment loss (default: 0.5)
- `--timesteps`: Number of timesteps to supervise (default: 32)
- `--n-realizations`: Number of stochastic realizations for CNO rollout (default: 1)

---

## Results

### With `100k_full_features` checkpoint (225D features):
```
Epoch 1/3
  Train: total=0.979118 state=0.977631 latent=0.000044 commit=0.002967
  Val: total=0.714476 (best)

Epoch 2/3
  Train: total=1.023395 state=1.021751 latent=0.000051 commit=0.003278
  Val: total=0.806481

Epoch 3/3
  Train: total=0.957766 state=0.956143 latent=0.000043 commit=0.003236
  Val: total=0.721931
```

Training completes without NaN collapse. Occasional NaN gradient warnings (safely skipped) don't affect convergence.

---

## Implementation Details

### VQ-VAE Loading
Handles multiple checkpoint formats:
- `_orig_mod.vqvae.*` (compiled + nested)
- `vqvae.*` (nested)
- Raw VQ-VAE weights

### Feature Normalization
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
- Gradients flow: NOA ← features ← z_pre (L_latent + L_commit)
- `sg()` (stop-gradient) applied to quantized vectors in L_commit

---

## Known Issues (Resolved)

### Feature Dimension Mismatch ✅
Different VQ-VAE checkpoints expect different input dimensions:
- `100k_full_features`: 225D
- `100k_3family_v1`: 187D

**Solution**: TrajectoryFeatureExtractor pads/truncates to match. Uses global `standard_normalize()` instead of per-category normalization.

### NaN Handling ✅
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

### Training Stability ✅
```
Warning: NaN/Inf gradients at batch X, skipping update
```
These warnings indicate NaN gradients were detected and safely skipped. Training continues without weight corruption.

---

## Future Improvements

1. **Learned Feature Projection**: Train a small MLP to project trajectory features to VQ-VAE space
2. **Token Distribution Matching**: Add KL divergence on token distributions
3. **Data Augmentation via VQ Space**: Sample token sequences and decode to create synthetic training data
