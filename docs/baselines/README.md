# Baselines

Production datasets and VQ-VAE tokenizers for Neural Operator Agent research.

## Available Baselines

### Datasets

| Dataset | Samples | Features | Size | Status |
|---------|---------|----------|------|--------|
| [**100K Full Features**](100k-full-features-dataset.md) | 100,000 | SUMMARY+TEMPORAL+ARCHITECTURE | ~10 GB | PRODUCTION |

### VQ-VAE Tokenizers

| Tokenizer | Dataset | Val Loss | Quality | Utilization | Categories | Status |
|-----------|---------|----------|---------|-------------|------------|--------|
| [**100K Full Features**](100k-full-features-vqvae.md) | 100k_full_features.h5 | **0.169** | 0.957 | 71.7% | 14 | PRODUCTION |

## Recommended Baseline

**100K Full Features** is the recommended baseline for:
- NOA agent training (Phase 1+)
- Behavioral token analysis
- Transfer learning experiments
- Production deployments

### Quick Reference

| Component | Path |
|-----------|------|
| Dataset | `datasets/100k_full_features.h5` |
| Checkpoint | `checkpoints/production/100k_full_features/` |
| Dataset Config | [100k-full-features-dataset.md](100k-full-features-dataset.md) |
| VQ-VAE Config | [100k-full-features-vqvae.md](100k-full-features-vqvae.md) |

### Feature Summary

| Family | Raw Dim | Encoded Dim | Encoder |
|--------|---------|-------------|---------|
| SUMMARY | 360 | ~150 | MLPEncoder [512, 256] |
| TEMPORAL | 256×63 | ~38 | TemporalCNNEncoder |
| ARCHITECTURE | 12 | 12 | IdentityEncoder |
| **Total** | - | **200** | - |

After cleaning: **200 features** → **14 categories**

## Adding New Baselines

When creating new production baselines:

1. **Dataset:** Document in `docs/baselines/{name}-dataset.md`
2. **VQ-VAE:** Document in `docs/baselines/{name}-vqvae.md`
3. **Config:** Store in `configs/vqvae/production/`
4. Update this README with comparison tables
5. Reference from main README.md
