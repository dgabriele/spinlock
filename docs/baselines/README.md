# Baselines

Production datasets and VQ-VAE tokenizers for Neural Operator Agent research.

## Available Baselines

### Datasets

| Dataset | Samples | Features | Size | Status |
|---------|---------|----------|------|--------|
| [**100K Full Features**](100k-full-features-dataset.md) | 100,000 | INITIAL+SUMMARY+TEMPORAL+ARCHITECTURE | ~10 GB | PRODUCTION |

### VQ-VAE Tokenizers

| Tokenizer | Dataset | Val Loss | Quality | Utilization | Categories | Status |
|-----------|---------|----------|---------|-------------|------------|--------|
| [**100K Full Features**](100k-full-features-vqvae.md) | 100k_full_features.h5 | **0.172** | 0.9517 | 66.7% | 15 | PRODUCTION |

## Recommended Baseline

**100K Full Features with INITIAL** is the recommended baseline for:
- NOA agent training (Phase 1+)
- Behavioral token analysis
- Transfer learning experiments
- Production deployments

### Quick Reference

| Component | Path |
|-----------|------|
| Dataset | `datasets/100k_full_features.h5` |
| Checkpoint | `checkpoints/production/100k_with_initial/` |
| Dataset Config | [100k-full-features-dataset.md](100k-full-features-dataset.md) |
| VQ-VAE Config | [100k-full-features-vqvae.md](100k-full-features-vqvae.md) |

### Feature Summary

| Family | Raw Dim | Encoded Dim | Encoder |
|--------|---------|-------------|---------|
| **INITIAL** | 14 + CNN | 42 | InitialHybridEncoder (end-to-end) |
| SUMMARY | 360 | 128 | MLPEncoder [512, 256] |
| TEMPORAL | 256×63 | 128 | TemporalCNNEncoder |
| ARCHITECTURE | 12 | 12 | IdentityEncoder |
| **Total** | - | **282** | - |

After cleaning: **175 features** → **15 categories**

## Adding New Baselines

When creating new production baselines:

1. **Dataset:** Document in `docs/baselines/{name}-dataset.md`
2. **VQ-VAE:** Document in `docs/baselines/{name}-vqvae.md`
3. **Config:** Store in `configs/vqvae/production/`
4. Update this README with comparison tables
5. Reference from main README.md
