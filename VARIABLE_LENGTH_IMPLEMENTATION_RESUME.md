# Variable-Length Trajectory Training Implementation - Resume Document

**Date:** 2026-01-31
**Status:** Implementation complete, testing in progress
**Command to run:** `poetry run spinlock train-vqvae --config configs/vqvae/baseline_vqvae_variable_length.yaml`

---

## Summary

Implemented variable-length trajectory support for VQ-VAE training, enabling the model to train on trajectories of different lengths (T ∈ [16, 32, 64, 128, 256]) with adaptive pyramid level selection. This allows the model to learn scale-invariant representations for operator discovery and meta-learning.

---

## Problem Statement

The original VQ-VAE pipeline assumed fixed-length temporal trajectories (T=500). The initial variable-length implementation encountered a critical architectural mismatch:

1. **Initial Error:** Training failed with "Variable-length mode requires temporal features [N, T, D], got shape torch.Size([5688, 338])"
   - Root cause: Pipeline pre-encoded temporal features to [N, 338] during loading
   - Variable-length mode needs raw [N, T, D] sequences for length sampling during training

2. **Secondary Error:** After skipping pre-encoding, got "All features have zero variance!"
   - Root cause: Category discovery (clustering) needs encoded features
   - Kept raw features broke clustering

---

## Solution: Dual Encoding Strategy

### Architecture Changes

**1. Feature Loading (`_load_features()` in `train_vqvae.py`)**

When variable-length is enabled for temporal family:

```python
# STEP 1: Encode at full length for category discovery
temporal_encoder = get_encoder(...)
encoded_for_clustering = temporal_encoder(raw_features)  # [N, 320]

# STEP 2: Store raw features for variable-length training
raw_temporal_features = family_features  # [N, T, D]

# STEP 3: Store encoder for runtime encoding
self._temporal_encoder = temporal_encoder

# Use encoded features for clustering
all_features.append(encoded_for_clustering)
```

**2. Dataloader Creation (`_create_data_loaders()` in `train_vqvae.py`)**

```python
# Pass raw temporal features to dataset (for length sampling)
dataset_features = raw_temporal_features  # [N, T, D]

# Store pre-encoded initial features separately
self._encoded_initial_features = initial_features  # [N, D_initial]

# Dataset creates masks and samples lengths
```

**3. Trainer (`VQVAETrainer` in `trainer.py`)**

Added variable-length support parameters:
- `temporal_encoder`: Encoder to apply at runtime
- `temporal_encoder_output_dim`: Output dimension (320D)
- `encoded_initial_features`: Pre-encoded initial features to concatenate

Modified training loop:

```python
# In train_epoch() and validate():
if self.temporal_encoder is not None:
    # Extract mask and length from batch
    mask = batch["mask"]  # [B, T]
    length = batch["length"]  # [B]

    # Encode temporal with variable lengths
    encoded_temporal = self.temporal_encoder(features, mask=mask, lengths=length)  # [B, 320]

    # Concatenate with pre-encoded initial features
    if self.encoded_initial_features_tensor is not None:
        features = torch.cat([initial_batch, encoded_temporal], dim=1)
    else:
        features = encoded_temporal

# Pass to VQ-VAE
outputs = self.model(features)
```

---

## Data Flow

### Category Discovery Phase (One-time)
```
Raw temporal [N, T=256, D]
  → Encode at full length
  → [N, 320] encoded
  → Clustering/category discovery
  → 14 categories (4 initial + 3 P0 + 4 P1 + ... + 8 P3)
```

### Training Phase (Every batch)
```
Raw temporal [N, T=256, D]
  → Dataset: Sample length T_i ∈ {16, 32, 64, 128, 256}
  → Create mask [N, T] (True×T_i, False×(T-T_i))
  → Training loop: Encode with mask
  → [N, 320] encoded (adaptive pyramid levels)
  → Concatenate with initial [N, 128]
  → [N, 448] total features
  → VQ-VAE forward pass
```

---

## Configuration

**File:** `configs/vqvae/baseline_vqvae_variable_length.yaml`

Key settings:

```yaml
families:
  temporal:
    encoder: PyramidTemporalEncoder
    encoder_params:
      level_dims: [32, 64, 96, 128]  # Per-level output (total 320D)
      downsample_factors: [1, 2, 4, 8]  # Pyramid factors

      variable_length:
        enabled: true  # ENABLED for variable-length mode
        min_timesteps: 16  # Minimum: 2^4
        max_timesteps: 256  # Maximum: 2^8
        sampling_strategy: "fixed_bins"
        length_bins: [16, 32, 64, 128, 256]  # Powers of 2
        adaptive_pyramid: true  # Auto-skip invalid levels
        mask_downsample_method: "ceil"  # Conservative masking

training:
  batch_size: 1024
  num_epochs: 100  # Variable-length: ~400-500 hours (17-21 days)
  warmup_epochs: 20
```

---

## Modified Files

### Core Implementation
1. **`src/spinlock/cli/train_vqvae.py`**
   - Added variable-length detection in `_load_features()`
   - Dual encoding: encode for clustering, keep raw for training
   - Store temporal encoder and initial features
   - Modified `_create_data_loaders()` to use raw temporal features
   - Modified `_create_trainer()` to pass temporal encoder

2. **`src/spinlock/encoding/training/trainer.py`**
   - Added parameters: `temporal_encoder`, `temporal_encoder_output_dim`, `encoded_initial_features`
   - Modified `train_epoch()`: encode temporal at runtime, concatenate with initial
   - Modified `validate()`: same runtime encoding logic

### Supporting Infrastructure (Already Implemented)
3. **`src/spinlock/encoding/trajectory_length_sampler.py`** (NEW)
   - `TrajectoryLengthSampler` with strategies: uniform, geometric, fixed_bins
   - `LengthCurriculumScheduler` for gradual min→max transition
   - `create_mask()` utility for boolean masks

4. **`src/spinlock/encoding/temporal_pyramid.py`** (MODIFIED)
   - Added `get_valid_levels()` for adaptive pyramid level selection
   - Added `adaptive` parameter and `min_pyramid_length`
   - Returns (levels, level_masks, valid_factors) tuple

5. **`src/spinlock/encoding/encoders/pyramid_temporal.py`** (MODIFIED)
   - Added `variable_length_config` parameter
   - Modified `forward()` to accept mask and lengths
   - Added `_pad_to_full_dim()` for zero-padding missing levels
   - Returns (embeddings, mask_info) when mask provided

6. **`src/spinlock/encoding/training/losses.py`** (MODIFIED)
   - Added sample weighting by valid fraction
   - Modified `reconstruction_loss()` to handle `mask_info`

7. **`src/spinlock/encoding/variable_length_utils.py`** (NEW)
   - `parse_variable_length_config()` - extract and validate config
   - `create_length_sampler()` - instantiate sampler
   - `augment_dataset_with_lengths()` - wrap dataset with length sampling
   - `extract_mask_info_from_batch()` - extract mask metadata

8. **`src/spinlock/encoding/training/data_utils.py`** (NEW - refactored from train_vqvae.py)
   - `FeatureDataset` class
   - `create_train_val_dataloaders()` with variable-length support

9. **`configs/vqvae/baseline_vqvae_variable_length.yaml`** (NEW)
   - Full configuration with variable_length enabled
   - Powers of 2 length bins for clean pyramid alignment

10. **`docs/vqvae/temporal-pyramid.md`** (UPDATED)
    - Added "Variable-Length Trajectory Support" section (~200 lines)
    - Documented length sampling, adaptive pyramid, masking

11. **`README.md`** (UPDATED)
    - Added variable-length callout in VQ-VAE section

---

## Testing Status

### Previous Errors (FIXED)
1. ✅ "Variable-length mode requires temporal features [N, T, D], got shape torch.Size([5688, 338])"
   - Fixed by skipping pre-encoding for temporal features

2. ✅ "All features have zero variance!"
   - Fixed by dual encoding (encode for clustering, keep raw for training)

### Current Status
- Training started successfully
- Initialization phase in progress (torch.compile warmup may take 3-5 minutes)
- No errors in feature loading or category discovery
- Expected to see training progress soon

### Expected Output
```
Loading features from datasets/100k_baseline_dev.h5...
  initial: Hybrid mode - 38D manual + raw ICs (6320, 3, 64, 64)
  temporal: Variable-length mode detected - dual encoding strategy
    Encoding at full length (T=256) for category discovery...
    Encoded for clustering: (6320, 320)
    Keeping raw features for variable-length training: (6320, 256, 345)
    Proceeding with encoded features for category discovery

DISCOVERED CATEGORY STRUCTURE
Number of categories: 14
...

Variable-length mode: Using raw temporal features for runtime encoding:
  Raw temporal shape: (6320, 256, 345)
  Pre-encoded initial features shape: (6320, 142)
  These will be concatenated with encoded temporal features during training

Variable-length mode enabled:
  TrajectoryLengthSampler(min_T=16, max_T=256, strategy='fixed_bins', bins=[16, 32, 64, 128, 256])

Epoch 1/100: [Training starts]
```

---

## Key Insights

### Why Dual Encoding?
- **Clustering needs encoded features**: Category discovery requires fixed-size embeddings
- **Training needs raw features**: Variable-length sampling requires [N, T, D] sequences
- **Solution**: Encode once at full length for clustering, keep raw for training

### Why Powers of 2?
- Pyramid factors [1, 2, 4, 8] divide cleanly into powers of 2
- T=16: levels [16, 8, 4, 2] - all valid
- T=32: levels [32, 16, 8, 4] - all valid
- T=64: levels [64, 32, 16, 8] - all valid
- No rounding, integer boundaries at every level

### Adaptive Pyramid Levels
- For T=16 with factors [1, 2, 4, 8]: all 4 levels valid (16/8 = 2 ≥ 1)
- For T=4 with factors [1, 2, 4, 8]: only [1, 2, 4] valid (skip 8×)
- Output zero-padded to full dimension (320D) when levels skipped

### Sample Weighting
- Prevents short trajectories from dominating gradients
- Weight = (num_valid / max_valid)
- T=64 batch: weight = 64/256 = 0.25
- T=256 batch: weight = 256/256 = 1.00

---

## Commits

1. **849514b** - "fix: implement variable-length trajectory support for VQ-VAE training"
   - Modified `_load_features()` to skip encoding for temporal
   - Modified `_create_data_loaders()` to use raw temporal
   - Modified `VQVAETrainer` for runtime encoding

2. **44620f1** - "fix: dual encoding strategy for variable-length"
   - Encode at full length for clustering
   - Keep raw for training
   - Fixes "zero variance" error

---

## Next Steps

1. **Monitor Training**
   - Watch for successful epoch completion
   - Check loss convergence
   - Verify codebook utilization (~18% expected)

2. **Validation**
   - After a few epochs, check reconstruction quality
   - Verify variable-length batches work correctly
   - Check that different length bins are sampled

3. **Optimization** (if needed)
   - If training is too slow, reduce batch_size or use fewer pyramid levels
   - If memory issues, reduce max_timesteps or model capacity
   - If convergence is poor, adjust commitment_cost or learning_rate

4. **Early Stopping**
   - Training will stop early if validation loss plateaus
   - Early stopping patience: 30 epochs (reduced from 100 proportionally)
   - Min delta: 0.01

---

## Troubleshooting

### If training crashes
**Error:** Out of memory
- Reduce `batch_size` from 1024 to 512
- Reduce `max_timesteps` from 256 to 128
- Reduce model capacity (group_embedding_dim, group_hidden_dim)

**Error:** Slow training
- Reduce `num_epochs` from 100 to 50
- Check torch.compile is enabled and working
- Verify GPU utilization (should be >80%)

**Error:** Poor reconstruction quality
- Increase `num_epochs` for more training
- Adjust `commitment_cost` (try 0.15-0.30 range)
- Check orthogonality target is being met

**Error:** Encoder-related errors
- Check that temporal_encoder is on correct device (GPU)
- Verify mask and length tensors have correct shapes
- Check that encoded_initial_features matches batch size

### If results are unexpected
- Check that length sampling is working (inspect batch["length"])
- Verify adaptive pyramid is selecting correct levels
- Check that sample weighting is applied correctly
- Monitor codebook utilization per category

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE LOADING (Once)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Raw Temporal [N, T=256, D=345]                                    │
│         │                                                           │
│         ├──→ Encode at full length ──→ [N, 320] ──→ Clustering     │
│         │                                                           │
│         └──→ Store raw [N, T, D] ──────────────────────────────────┼──┐
│                                                                     │  │
│  Initial [N, C, H, W]                                               │  │
│         │                                                           │  │
│         └──→ Extract manual ──→ [N, 38] ──→ Concat with CNN later  │  │
│                                                                     │  │
└─────────────────────────────────────────────────────────────────────┘  │
                                                                          │
┌─────────────────────────────────────────────────────────────────────┐  │
│                  TRAINING LOOP (Every Batch)                         │  │
├─────────────────────────────────────────────────────────────────────┤  │
│                                                                     │  │
│  Dataset __getitem__:                                               │◀─┘
│    1. Sample length T_i ∈ {16, 32, 64, 128, 256}                   │
│    2. Create mask [T]: [True×T_i, False×(T-T_i)]                    │
│    3. Return: features=[T, D], mask=[T], length=T_i                 │
│                                                                     │
│  Training loop (trainer.py):                                        │
│    1. features = batch["features"]  # [B, T, D] raw                 │
│    2. mask = batch["mask"]          # [B, T] boolean                │
│    3. length = batch["length"]      # [B] actual lengths            │
│                                                                     │
│    4. # Encode with variable lengths                                │
│       encoded_temporal = temporal_encoder(                          │
│           features,                                                 │
│           mask=mask,                                                │
│           lengths=length                                            │
│       )  # [B, 320] with adaptive pyramid levels                    │
│                                                                     │
│    5. # Concatenate with initial features                           │
│       features = concat([initial_batch, encoded_temporal], dim=1)   │
│       # [B, 142+320] = [B, 462]                                     │
│                                                                     │
│    6. # Forward pass                                                │
│       outputs = model(features)                                     │
│                                                                     │
│    7. # Compute loss (with sample weighting)                        │
│       loss = reconstruction_loss(outputs, targets, mask_info)       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## References

- **Plan document:** Plan file in previous conversation
- **Temporal pyramid docs:** `docs/vqvae/temporal-pyramid.md`
- **Example config:** `configs/vqvae/baseline_vqvae_variable_length.yaml`
- **Test files:** `tests/encoding/test_variable_length_*.py`
- **Original implementation:** All core components in `src/spinlock/encoding/`

---

## Contact Points for Issues

1. **Encoder not found errors** → Check `src/spinlock/encoding/encoders/__init__.py` registry
2. **Mask shape errors** → Check `src/spinlock/encoding/temporal_pyramid.py` mask downsampling
3. **Concatenation errors** → Check `src/spinlock/encoding/training/trainer.py` feature concatenation
4. **Loss computation errors** → Check `src/spinlock/encoding/training/losses.py` sample weighting
5. **Dataset errors** → Check `src/spinlock/encoding/training/data_utils.py` and `variable_length_utils.py`

---

**END OF RESUME DOCUMENT**
