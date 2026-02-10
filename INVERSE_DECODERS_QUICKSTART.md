# VQTokenizer Inverse Decoders - Quick Start

## What Was Implemented

✅ **Problem Solved**: VQTokenizer can now properly decode tokens → (theta, ICs)
- Before: Returned encoded features [B, 32] and [B, 426]
- After: Returns actual parameters [B, 14] and spatial grids [B, 3, 64, 64]

✅ **Architecture**: Two supervised inverse decoders
1. `ThetaInverseMLP`: theta_encoded [B, 32] → theta [B, 14] in [0,1]
2. `InitialInverseCNN`: initial_encoded [B, 426] → ICs [B, 3, 64, 64]

---

## Training Inverse Models (Required Next Steps)

### 1. Train Theta Inverse (~2 hours)

```bash
poetry run python scripts/train_theta_inverse.py \
    --tokenizer checkpoints/v2/vqvae/vq_tokenizer_best.pt \
    --tokenized-dataset datasets/50k_tokenized_v2_cleaned.h5 \
    --raw-dataset datasets/50k_baseline.h5 \
    --output checkpoints/theta_inverse.pt \
    --epochs 100 \
    --batch-size 256
```

**Target**: Val MSE < 0.01

### 2. Train Initial Inverse (~3 hours)

```bash
poetry run python scripts/train_initial_inverse.py \
    --tokenizer checkpoints/v2/vqvae/vq_tokenizer_best.pt \
    --tokenized-dataset datasets/50k_tokenized_v2_cleaned.h5 \
    --raw-dataset datasets/50k_baseline.h5 \
    --output checkpoints/initial_inverse.pt \
    --epochs 100 \
    --batch-size 128
```

**Target**: Val MSE < 0.05

### 3. Validate End-to-End (~10 minutes)

```bash
poetry run python scripts/validate_inverse_decoders.py \
    --tokenizer checkpoints/v2/vqvae/vq_tokenizer_best.pt \
    --theta-inverse checkpoints/theta_inverse.pt \
    --initial-inverse checkpoints/initial_inverse.pt \
    --dataset datasets/50k_baseline.h5 \
    --num-samples 100
```

**Outputs**:
- Reconstruction metrics (MSE, MAE, max error)
- Visualizations in `results/inverse_validation/`
- Success/failure diagnostics

---

## Usage After Training

### Load Tokenizer with Inverse Models

```python
from spinlock.tokens.tokenizer import VQTokenizer

tokenizer = VQTokenizer.from_checkpoint(
    "checkpoints/v2/vqvae/vq_tokenizer_best.pt",
    theta_inverse_path="checkpoints/theta_inverse.pt",
    initial_inverse_path="checkpoints/initial_inverse.pt",
)
```

### Decode Tokens → (theta, ICs)

```python
# Sample from diffusion or load from dataset
tokens = {...}  # Dict[str, Tensor]

# Decode to actual parameters and ICs
theta, u0 = tokenizer.decode(tokens)

# theta: [B, 14] in [0,1] - operator parameters
# u0: [B, 3, 64, 64] - spatial initial conditions

# Generate trajectories
trajectories = cno_replayer.rollout_batch(theta, u0, num_steps=256)
```

### New decode() Signature

**Before** (broken):
```python
theta, u0, temporal = tokenizer.decode(tokens)
# theta: [B, 32] encoded features ❌
# u0: [B, 426] encoded features ❌
# temporal: [B, 320] encoded features
```

**After** (fixed):
```python
theta, u0 = tokenizer.decode(tokens)
# theta: [B, 14] actual parameters in [0,1] ✅
# u0: [B, 3, 64, 64] spatial ICs ✅
```

**Key Changes**:
- Returns `Tuple[Tensor, Tensor]` (not Optional types)
- Requires inverse models loaded (no fallbacks)
- Raises clear errors if models missing
- Always returns proper shapes

---

## Files Created

```
src/spinlock/tokens/inverse_models.py          # ThetaInverseMLP, InitialInverseCNN
scripts/train_theta_inverse.py                 # Training script
scripts/train_initial_inverse.py               # Training script
scripts/validate_inverse_decoders.py           # Validation script
tests/test_inverse_models.py                   # Unit tests ✅ 12/12 passing
docs/inverse-decoders-implementation.md        # Full documentation
INVERSE_DECODERS_QUICKSTART.md                 # This file
```

## Files Modified

```
src/spinlock/tokens/model.py                   # Added inverse model attributes
src/spinlock/tokens/tokenizer.py               # Updated decode(), added load methods
```

---

## Decision Point: Success or Plan B?

After training both inverse models, check validation metrics:

### ✅ SUCCESS (Proceed with Current Architecture)
- Theta MSE < 0.01
- IC MSE < 0.05
- Visual quality good
- CNO rollouts stable

**Next**: Update sampling pipeline to use proper decode()

### ⚠️ FAILURE (Switch to Plan B)
- Theta MSE > 0.01 or IC MSE > 0.05
- Poor visual quality
- CNO instability

**Root Cause**: Quantization too lossy → encoded features don't preserve info

**Solution**: End-to-end VQTokenizer retraining (~8+ hours)
- Add reconstruction heads to decoder
- Train with combined loss: encoded + final reconstruction
- Guide quantizers to preserve reconstruction-critical information

See `docs/inverse-decoders-implementation.md` for Plan B details.

---

## Summary

**Status**: Implementation complete, training required

**Total Time**: ~5 hours training + 10 min validation

**Files**: 6 new, 2 modified, 12 tests passing

**Next Step**: Run training scripts above in sequence

**Documentation**: See `docs/inverse-decoders-implementation.md` for full details
