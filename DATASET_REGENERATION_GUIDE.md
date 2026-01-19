# Dataset Regeneration Guide (v3.0.0)

**Last Updated:** 2026-01-18

After the v3.0.0 feature architecture refactoring, existing feature datasets need to be regenerated with the new per-timestep-only architecture.

## Quick Summary

**What Changed:**
- **v2.x:** 4 families (INITIAL 42D, ARCHITECTURE 21D, SUMMARY 420-520D, TEMPORAL 63D)
- **v3.0:** 3 families (INITIAL 42D, ARCHITECTURE ~20D, TEMPORAL ~328D per-timestep)

**Key Changes:**
- ❌ Removed SUMMARY features (incompatible with online prediction)
- ✨ Enhanced TEMPORAL from 63D → ~328D per-timestep
- ✨ Parameter space expanded from 12D → 14D (added dt and alpha)
- ✅ All features now per-timestep computable (no trajectory lookahead)

**Status:**
- ✅ Code fully updated and validated
- ✅ Test dataset (100 samples) successfully regenerated
- ⏳ Production datasets (10K-100K) pending regeneration

## Prerequisites

- **Trained MNO checkpoint**: `checkpoints/noa/pure_mse_v4_10k_contiguous/meta_operator_best.pt`
- **GPU access**: Highly recommended for large-scale generation
- **Disk space**: ~10-20 GB for 10K samples, ~100-200 GB for 100K samples

## Regeneration Steps

### Option 1: Quick Validation (10K Samples, ~2-3 hours on GPU)

```bash
# Step 1: Generate MNO rollouts (10K samples)
poetry run spinlock generate-noa-dataset \
    --noa-checkpoint checkpoints/noa/pure_mse_v4_10k_contiguous/meta_operator_best.pt \
    --output datasets/mno_rollouts_10k_v3.h5 \
    --n-samples 10000 \
    --config configs/experiments/noa_features_10k.yaml \
    --device cuda \
    --batch-size 32 \
    --verbose

# Step 2: Regenerate features from MNO rollouts
poetry run python scripts/regenerate_mno_features.py \
    --input datasets/mno_rollouts_10k_v3.h5 \
    --output datasets/mno_features_10k_enhanced.h5 \
    --batch-size 32 \
    --device cuda

# Step 3: Train VQ-VAE on new features
poetry run spinlock train-vqvae \
    --config configs/vqvae/enhanced_temporal.yaml \
    --device cuda
```

### Option 2: Full-Scale (100K Samples, ~20-30 hours on GPU)

```bash
# Step 1: Generate 100K MNO rollouts
poetry run spinlock generate-noa-dataset \
    --noa-checkpoint checkpoints/noa/pure_mse_v4_10k_contiguous/meta_operator_best.pt \
    --output datasets/mno_rollouts_100k_v3.h5 \
    --n-samples 100000 \
    --config configs/experiments/local_100k_optimized.yaml \
    --device cuda \
    --batch-size 32 \
    --verbose

# Step 2: Regenerate features
poetry run python scripts/regenerate_mno_features.py \
    --input datasets/mno_rollouts_100k_v3.h5 \
    --output datasets/mno_features_100k_enhanced.h5 \
    --batch-size 32 \
    --device cuda

# Step 3: Train VQ-VAE
poetry run spinlock train-vqvae \
    --config configs/vqvae/enhanced_temporal.yaml \
    --device cuda
```

### Option 3: CNO Reference Features (if CNO trajectories available)

```bash
# If you have existing CNO trajectory datasets
poetry run python scripts/regenerate_cno_features.py \
    --input datasets/cno_100k_stratified.h5 \
    --output datasets/cno_features_100k_enhanced.h5 \
    --batch-size 32 \
    --device cuda
```

## Validation

After regeneration, validate the new features:

```bash
# Run validation script on generated features
poetry run python scripts/validate_feature_extraction.py \
    --num-samples 10 \
    --num-timesteps 50 \
    --device cuda \
    --output-dir validation_output
```

**Expected output:**
- Dimension check: PASS (~328D TEMPORAL per-timestep)
- No infinities
- Minimal NaNs (robust statistics handle edge cases)
- Feature registry validated (spatial, spectral, cross-channel, temporal categories)

## VQ-VAE Training

Update the VQ-VAE config to point to your regenerated dataset:

```yaml
# configs/vqvae/enhanced_temporal.yaml
dataset_path: "datasets/mno_features_10k_enhanced.h5"  # or 100k version
```

**Expected VQ-VAE metrics:**
- Reconstruction loss: < 0.02 (target: match or beat old 0.018)
- Codebook utilization: > 60% per level
- Training time: ~3-4 hours on RTX 3060 Ti

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size:
```bash
--batch-size 16  # or even 8 for large models
```

### Issue: "ModuleNotFoundError: No module named 'spinlock.features.summary'"
**Solution:** SUMMARY features were removed in v3.0. Update your code to use TEMPORAL features only. Archived SUMMARY code available in `src/spinlock/features/temporal_old_v2/summary/` for reference.

### Issue: MNO generation very slow
**Solution:** 
- Use GPU (`--device cuda`)
- Increase batch size if memory allows
- Consider generating smaller dataset first (1K-10K for testing)

### Issue: Features have many NaNs
**Expected (v3.0):** TEMPORAL features use robust statistics and should have minimal NaNs:
- Most features are NaN-safe (use `nanmean`, `nanstd`, etc.)
- Edge cases handled (t=0, uniform fields, single-channel data)
- Feature cleaning pipeline removes NaN-containing features

**Unexpected:** If features have excessive NaNs, check:
1. Input trajectories are valid (no NaN/Inf in rollouts)
2. Feature extraction uses latest v3.0 code (robust statistics)
3. Numerical stability in FFT and gradient computations
4. Run feature validation script to identify problematic features

## Performance Estimates

| Dataset Size | GPU | Generation Time | Feature Extraction | Total |
|-------------|-----|----------------|-------------------|-------|
| 1K samples  | RTX 3060 Ti | ~15 min | ~2 min | ~17 min |
| 10K samples | RTX 3060 Ti | ~2-3 hours | ~15 min | ~3.5 hours |
| 100K samples| RTX 3060 Ti | ~20-30 hours | ~2 hours | ~32 hours |
| 1K samples  | CPU | ~2 hours | ~15 min | ~2.5 hours |
| 10K samples | CPU | ~20+ hours | ~2 hours | ~22+ hours |

**Note:** CPU generation not recommended for >1K samples.

## File Sizes (v3.0)

| Dataset | Trajectories | Features (v3.0) | Compression Ratio |
|---------|-------------|----------------|------------------|
| 100 samples | 436 MB | ~2.5 MB | 99.4% |
| 1K samples | ~4.3 GB | ~25 MB | 99.4% |
| 10K samples | ~43 GB | ~250 MB | 99.4% |
| 100K samples | ~430 GB | ~2.5 GB | 99.4% |

**v3.0 Note:** Enhanced TEMPORAL features (~328D vs 63D) increase feature file sizes by ~1.7x compared to v2.x, but still achieve >99% compression vs. storing full trajectories.

**Tip:** You can delete trajectory datasets after feature extraction to save space.

## Next Steps After Regeneration

1. **Validate features:** Check dimensions, NaN counts, value ranges
2. **Train VQ-VAE:** Use enhanced_temporal.yaml config
3. **Compare performance:** Ensure reconstruction quality matches/beats old 0.018
4. **Update checkpoints:** Retrain any downstream models using new features
5. **Archive old datasets:** Keep old checkpoints marked as v2.x for reference

## References

- Main implementation: `FEATURE_EXTRACTION_V3_SUMMARY.md`
- VQ-VAE architecture: `docs/vqvae_architecture_update_v3.md`
- Regeneration scripts: `scripts/regenerate_*.py`
- Validation script: `scripts/validate_feature_extraction.py`

## Test Dataset (v3.0)

A test dataset with 100 samples should be regenerated with v3.0 features:

```bash
# Generate test dataset with v3.0 features
poetry run spinlock generate-dataset \
    --config configs/experiments/test_100_v3.yaml \
    --output datasets/test_features_100_v3.h5 \
    --device cuda
```

**Expected v3.0 output:**
```
✅ datasets/test_features_100_v3.h5
   - Input: [100, C, H, W] initial conditions
   - TEMPORAL features: [100, T, ~328] per-timestep
   - Parameters: [100, 14] Sobol unit cube
   - Validation: All dimension checks PASS
   - Feature families: TEMPORAL only (SUMMARY removed)
```

This validates the v3.0 pipeline works correctly.
