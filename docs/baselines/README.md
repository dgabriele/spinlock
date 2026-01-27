# Baselines

Production datasets, VQ-VAE tokenizers, and MNO world models for Neural Operator Agent research.

**Last Updated:** 2026-01-27 (v3.1)

## Available Baselines

### Datasets

| Dataset | Samples | Features | Size | Status |
|---------|---------|----------|------|--------|
| [**100K Full Features v3.0**](100k-full-features-dataset.md) | 100,000 | TEMPORAL (~328D) | ~12 GB | PRODUCTION |
| **CNO 50K v3.1** | 50,000 | TEMPORAL (~328D) | ~6 GB | PRODUCTION |

**v3.1 Changes:** Enhanced temporal features, 14D parameter space, Sobol sampling for prefix-optimality

### VQ-VAE Tokenizers

| Tokenizer | Dataset | Val Loss | Quality | Utilization | Categories | Status |
|-----------|---------|----------|---------|-------------|------------|--------|
| [**100K Full Features**](100k-full-features-vqvae.md) | 100k_full_features.h5 | **0.169** | 0.957 | 71.7% | 14 | PRODUCTION |
| [**50K CNO Baseline**](50k-vqvae-baseline.md) | cno_50k_v3_1.h5 | **0.067** | 0.985 | 20.5% | 8 | PRODUCTION |

### MNO World Models

| Model | Dataset | Samples | Val L_traj | Val Loss | Parameters | Status |
|-------|---------|---------|------------|----------|------------|--------|
| [**10K CNO Baseline**](10k-mno-baseline.md) | cno_50k_v3_1.h5 | 10,240 | **0.5343** | 0.641 | 227M | PRODUCTION |

## Recommended Baselines

### For NOA Integration (Phase 1)

**CNO-Trained Components** (50K VQ-VAE + 10K MNO):
- VQ-VAE: [50K CNO Baseline](50k-vqvae-baseline.md) (8 categories, 99.4% quality)
- MNO: [10K CNO Baseline](10k-mno-baseline.md) (L_traj=0.53, 227M params)
- **Status**: ✅ Both production ready, ready for NOA experimentation
- **Use case**: Perturbation-driven exploration with symbolic reasoning

### For Large-Scale Analysis

**100K Full Features** for:
- Behavioral token analysis
- Transfer learning experiments
- Cross-domain vocabulary alignment

### Quick Reference

| Component | Dataset | Checkpoint | Config |
|-----------|---------|------------|--------|
| **VQ-VAE (50K)** | `datasets/cno_50k_v3_1.h5` | `checkpoints/vqvae/50k_baseline/` | [50k-vqvae-baseline.md](50k-vqvae-baseline.md) |
| **MNO (10K)** | `datasets/cno_50k_v3_1.h5` | `checkpoints/noa/10k_baseline/` | [10k-mno-baseline.md](10k-mno-baseline.md) |
| **VQ-VAE (100K)** | `datasets/100k_full_features.h5` | `checkpoints/production/100k_full_features/` | [100k-full-features-vqvae.md](100k-full-features-vqvae.md) |

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

## Dataset Regeneration

For migrating from v2.x datasets or regenerating datasets with v3.0 features, see:
- [Dataset Regeneration Guide](dataset-regeneration-guide.md)
