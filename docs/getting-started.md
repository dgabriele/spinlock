# Getting Started with Spinlock

This guide walks through the complete workflow from operator generation to VQ-VAE tokenization.

## Quick Start

### 1. Generate Operator Dataset

```bash
poetry run spinlock generate \
    --config configs/experiments/baseline_10k.yaml \
    --output datasets/my_operators.h5
```

This will:
- Sample 10,000 operator parameter vectors using Sobol stratification
- Construct CNN-based neural operators
- Generate 500-timestep stochastic rollouts (3 realizations each)
- Extract IC, NOP, SDF, TD features inline
- Store everything in HDF5 format

**Expected time:** ~12 hours on GPU

### 2. Inspect Dataset

```bash
poetry run spinlock inspect datasets/my_operators.h5
```

View dataset contents:
- Number of operators
- Feature dimensions (IC, NOP, SDF, TD)
- Metadata (IC types, evolution policies, parameter stratification)

### 3. Train VQ-VAE Tokenizer

```bash
poetry run spinlock train-vqvae \
    --dataset datasets/my_operators.h5 \
    --config configs/vqvae/production.yaml \
    --output checkpoints/vqvae/
```

This will:
- Load and concatenate all feature families (IC+NOP+SDF+TD)
- Automatically clean features (NaN removal, variance filtering, deduplication)
- Discover ~8-15 categories via hierarchical clustering
- Train 3-level hierarchical VQ-VAE
- Save checkpoints and training history

**Expected time:** ~2-6 hours on GPU

### 4. Tokenize Operators

```python
import torch
from spinlock.encoding import CategoricalHierarchicalVQVAE

# Load trained model
checkpoint = torch.load("checkpoints/vqvae/best_model.pt")
model = CategoricalHierarchicalVQVAE.from_checkpoint(checkpoint)

# Load features for new operators
features = ...  # Shape: [N, D] where D is total feature dimension

# Extract behavioral tokens
with torch.no_grad():
    tokens_coarse, tokens_medium, tokens_fine = model.get_tokens(features)

# tokens_coarse: [N] - Coarse behavioral category
# tokens_medium: [N] - Medium-grained behavior
# tokens_fine: [N] - Fine-grained behavior pattern
```

## Tutorials

### Example 1: Generate Small Test Dataset

```bash
# Generate 100 operators for quick testing
poetry run spinlock generate \
    --config configs/experiments/test_100.yaml \
    --output datasets/test_100.h5
```

### Example 2: Visualize Rollouts

```bash
# Visualize first 16 operators
poetry run spinlock visualize-dataset \
    --dataset datasets/test_100.h5 \
    --num-ops 16 \
    --output visualizations/
```

### Example 3: Extract Features Only

If you already have rollouts and want to extract features:

```python
from spinlock.features.ic import ICFeatureExtractor
from spinlock.features.sdf import SDFFeatureExtractor

# Extract IC features
ic_extractor = ICFeatureExtractor()
ic_features = ic_extractor.extract(initial_conditions)  # [N, M, 42]

# Extract SDF features
sdf_extractor = SDFFeatureExtractor()
sdf_features = sdf_extractor.extract(rollouts)  # [N, M, 420-520]
```

## Configuration

### Experiment Configs

Located in `configs/experiments/`:
- `test_100.yaml` - Small test dataset (100 operators)
- `baseline_10k.yaml` - Standard 10K dataset
- `benchmark_10k.yaml` - Benchmark configuration

### VQ-VAE Configs

Located in `configs/vqvae/`:
- `production.yaml` - Production-ready configuration
- `fast_training.yaml` - Quick training for testing

### Custom Configurations

Create your own YAML config:

```yaml
dataset:
  name: "custom_dataset"
  num_operators: 5000
  num_realizations: 3
  num_timesteps: 500
  grid_size: 128

parameter_space:
  architecture:
    num_layers: [2, 8]
    kernel_size: [3, 9]
    # ... more parameters

features:
  ic: true
  nop: true
  sdf: true
  td: true
```

## Common Workflows

### Workflow 1: End-to-End Training

```bash
# 1. Generate dataset
poetry run spinlock generate --config configs/experiments/baseline_10k.yaml

# 2. Train VQ-VAE
poetry run spinlock train-vqvae --dataset datasets/baseline_10k.h5

# 3. Evaluate tokenization
poetry run spinlock evaluate-vqvae --checkpoint checkpoints/vqvae/best_model.pt
```

### Workflow 2: Dataset Iteration

```bash
# Generate multiple datasets with different configs
for config in configs/experiments/*.yaml; do
    poetry run spinlock generate --config $config
done

# Train VQ-VAE on combined datasets
poetry run spinlock train-vqvae --datasets datasets/*.h5
```

## Next Steps

- **Explore operators:** Use visualization tools to understand operator diversity
- **Analyze features:** Investigate feature distributions and correlations
- **Tune VQ-VAE:** Experiment with codebook sizes and category counts
- **Build ANO:** Use tokens to train Neural Operator Agent (Phase 1+)

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size in config
- Use smaller grid size (64×64 instead of 128×128)
- Enable mixed precision training

### Slow Generation
- Ensure GPU is being used (`nvidia-smi`)
- Check if CPU-bound (increase num_workers)
- Profile with `scripts/dev/profile_temporal_rollout.py`

### Poor Tokenization Quality
- Increase dataset size (more operator diversity)
- Adjust feature cleaning thresholds
- Tune VQ-VAE hyperparameters (latent dims, codebook sizes)

## References

- [Architecture](architecture.md) - System design details
- [Feature Families](features/README.md) - IC, NOP, SDF, TD documentation
- [VQ-VAE Training](vqvae/training-guide.md) - Tokenization pipeline
- [NOA Roadmap](noa-roadmap.md) - Future development plan
