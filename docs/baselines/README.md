# Baselines

Production datasets and VQ-VAE tokenizers for Neural Operator Agent research.

**Last Updated:** 2026-01-18 (v3.0)

## Available Baselines

### Datasets

| Dataset | Samples | Features | Size | Status |
|---------|---------|----------|------|--------|
| [**100K Full Features v3.0**](100k-full-features-dataset.md) | 100,000 | TEMPORAL (~328D) | ~12 GB | PRODUCTION |

**v3.0 Changes:** SUMMARY features removed, TEMPORAL enhanced from 63D → ~328D per-timestep, parameter space 12D → 14D

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

### Feature Summary (v3.0)

| Family | Raw Dim | Encoded Dim | Encoder |
|--------|---------|-------------|---------|
| INITIAL | 42 | ~32 | MLPEncoder [128, 64] |
| TEMPORAL | T×~328 | ~180 | TemporalCNNEncoder |
| ARCHITECTURE | 14 | Excluded | N/A (NOA knows params) |
| **Total** | - | **~212** | - |

**v3.0 Architecture:**
- TEMPORAL features: Enhanced to ~328D per-timestep (was 63D in v2.x)
- SUMMARY features: Removed (incompatible with online prediction)
- INITIAL features: Computed inline from inputs (42D)
- ARCHITECTURE: Stored in `/parameters/params [N, 14]` but excluded from VQ-VAE training

After cleaning: **~212 features** → **~14-16 behavioral categories**

## Adding New Baselines

When creating new production baselines:

1. **Dataset:** Document in `docs/baselines/{name}-dataset.md`
2. **VQ-VAE:** Document in `docs/baselines/{name}-vqvae.md`
3. **Config:** Store in `configs/vqvae/production/`
4. Update this README with comparison tables
5. Reference from main README.md
