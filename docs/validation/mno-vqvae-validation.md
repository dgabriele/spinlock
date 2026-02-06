# MNO-VQ-VAE Distribution Alignment Validation

## Overview

This validation system verifies that the VQ-VAE (trained on CNO ground truth) can reliably tokenize MNO-generated rollouts with acceptable reconstruction quality. This validates the core assumption that `relative_l2 ≈ 1.0` is sufficient for distribution alignment.

**Critical Question**: Can the VQ-VAE tokenize MNO outputs, or is there a distribution mismatch?

## Architecture

### Components

```
src/spinlock/noa/validation/
├── __init__.py                 # Package exports
├── config.py                   # Validation configuration schemas
├── metrics.py                  # Validation metrics computation
├── mno_vqvae_validator.py     # Main validator class
└── report.py                   # Results reporting
```

### Design Principles

1. **Composition over Inheritance**: Reuses existing components (MNO loader, VQ-VAE loader, feature pipeline)
2. **Single Responsibility**: Each class has one clear purpose
3. **Modular and Testable**: Clean interfaces, easy to test
4. **Production Quality**: Well-documented, type-annotated, error-handling

## Usage

### CLI Command

```bash
# Basic validation with 100 samples
spinlock validate-mno-vqvae \
    --mno-checkpoint checkpoints/mno/50k_baseline/meta_operator_best.pt \
    --vqvae-checkpoint checkpoints/vqvae/50k_baseline/best_model.pt \
    --dataset datasets/cno_50k_v3_1.h5 \
    --output-dir validation_results/

# Quick test with 10 samples
spinlock validate-mno-vqvae \
    --mno-checkpoint checkpoints/mno/50k_baseline/meta_operator_best.pt \
    --vqvae-checkpoint checkpoints/vqvae/50k_baseline/best_model.pt \
    --dataset datasets/cno_50k_v3_1.h5 \
    --num-samples 10 \
    --output-dir validation_results/quick_test/
```

### Programmatic Usage

```python
from spinlock.noa.validation import MNOVQVAEValidator, ValidationConfig

# Create validator
config = ValidationConfig(num_samples=50, batch_size=4)
validator = MNOVQVAEValidator(
    mno_checkpoint="checkpoints/mno/50k_baseline/meta_operator_best.pt",
    vqvae_checkpoint="checkpoints/vqvae/50k_baseline/best_model.pt",
    config=config
)

# Run validation
result = validator.validate(
    dataset_path="datasets/cno_50k_v3_1.h5",
    num_samples=50
)

# Check result
if result.pass_threshold:
    print(f"✓ Validation passed! Ratio: {result.reconstruction_ratio:.3f}x")
else:
    print(f"✗ Validation failed. Ratio: {result.reconstruction_ratio:.3f}x")
```

## Validation Workflow

The validator performs the following steps:

1. **Load Validation Samples**: Loads ICs and parameters from dataset
2. **Generate MNO Rollouts**: Creates 256-step rollouts using MNO
3. **Extract Features**: Uses UnifiedFeaturePipeline to extract features from rollouts
4. **Tokenize with VQ-VAE**: Encodes features into discrete tokens
5. **Decode Tokens**: Reconstructs features from tokens
6. **Measure Reconstruction Error**: Computes MSE between original and reconstructed features
7. **Compare to CNO Baseline**: Calculates ratio of MNO/CNO reconstruction error

## Success Criteria

**Pass Threshold**: `reconstruction_ratio < 2.0`

### Interpretation

- **Ratio < 1.5**: ✓ EXCELLENT - MNO outputs match CNO distribution closely
- **Ratio 1.5-2.0**: ✓ GOOD - Minor drift but acceptable for tokenization
- **Ratio 2.0-3.0**: ⚠ ACCEPTABLE - Degraded quality, consider improvement
- **Ratio > 3.0**: ✗ POOR - Distribution mismatch, MNO needs improvement

## Metrics

The validation computes:

- **Reconstruction MSE**: Mean squared error between original and reconstructed features
- **Reconstruction MAE**: Mean absolute error
- **Relative L2**: Normalized reconstruction error
- **Reconstruction Ratio**: MNO/CNO error ratio (primary metric)
- **Per-dimension Correlation**: Feature-wise correlation between original and reconstructed
- **Token Entropy**: Shannon entropy of token distribution
- **Unique Tokens**: Number of distinct tokens used

## Output

The validation generates:

1. **Console Summary**: Real-time progress and final results
2. **Markdown Report**: Detailed report saved to `{output_dir}/validation_report.md`

### Example Report

```markdown
# MNO-VQ-VAE Distribution Alignment Validation Report

## Summary

**Status**: ✓ PASS

**Samples Tested**: 100

## Reconstruction Quality

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MNO Reconstruction Error** | 0.032 | <0.054 | ✓ |
| **CNO Baseline Error** | 0.027 | - | Reference |
| **Ratio (MNO/CNO)** | 1.185x | <2.0x | ✓ |

## Interpretation

### Reconstruction Ratio: 1.185x

**✓ EXCELLENT** - VQ-VAE tokenizes MNO outputs with quality matching CNO training data.
Distribution alignment is excellent. No action needed.

## Recommendation

System is production-ready for downstream NOA experimentation.
```

## Expected Results

Based on current MNO training metrics:
- **MNO L_traj**: 0.378-0.410 (good)
- **MNO relative_l2**: 0.998-1.002 (excellent)
- **VQ-VAE CNO reconstruction**: 0.027 (excellent)

**Expected validation result**:
- Reconstruction ratio: ~1.2-1.5x (GOOD to EXCELLENT)
- Pass threshold: Likely YES

## Next Steps

### If Validation PASSES (ratio < 2.0)

1. ✓ System is ready for downstream NOA experimentation
2. ✓ VQ-VAE tokenization is reliable on MNO outputs
3. ✓ Proceed with trajectory completion experiments

### If Validation FAILS (ratio > 2.0)

1. Investigate MNO training:
   - Check L_traj convergence
   - Check relative_l2 values
   - Visualize MNO vs CNO rollouts
2. Consider architecture improvements:
   - Increase MNO capacity
   - Adjust FiLM conditioning
   - Longer training
3. Re-run validation after improvements

## Implementation Details

### Reused Components

The validator leverages existing infrastructure:

- `load_mno_checkpoint()`: From `spinlock.noa.validation_utils`
- `load_vqvae_checkpoint()`: From `spinlock.noa.validation_utils`
- `UnifiedFeaturePipeline`: From `spinlock.encoding.unified_feature_pipeline`
- VQ-VAE inference: From `spinlock.encoding.categorical_vqvae`

This ensures:
- No code duplication
- Consistent behavior with training
- Maintainable codebase

### Testing

Unit tests are provided in `tests/noa/validation/test_mno_vqvae_validator.py`:

- Configuration validation
- Metrics computation
- Token entropy calculation
- End-to-end validation (if checkpoints available)

## References

- **10K MNO Baseline**: `docs/baselines/10k-mno-baseline.md`
- **50K VQ-VAE Baseline**: `docs/baselines/50k-vqvae-baseline.md`
- **NOA Architecture**: `docs/noa-architecture.md`
