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
- Extract INITIAL, ARCHITECTURE, SUMMARY, TEMPORAL features inline
- Store everything in HDF5 format

**Expected time:** ~12 hours on GPU

### 2. Inspect Dataset

```bash
poetry run spinlock inspect datasets/my_operators.h5
```

View dataset contents:
- Number of operators
- Feature dimensions (INITIAL, ARCHITECTURE, SUMMARY, TEMPORAL)
- Metadata (INITIAL types, evolution policies, parameter stratification)

### 3. Understanding Feature Semantics

The four feature families provide complementary perspectives on operator behavior. Understanding what each feature measures enables interpretable discovery and validation:

```python
import h5py
import numpy as np
from pathlib import Path

# Load dataset
with h5py.File("datasets/my_operators.h5", "r") as f:
    initial_features = f["features/initial"][:]      # [N, M, 42]
    arch_features = f["features/architecture"][:]    # [N, 21+]
    summary_features = f["features/summary"][:]      # [N, 420-520]
    temporal_features = f["features/temporal"][:]    # [N, M, T, D]

    # Feature metadata
    initial_names = f["features/initial"].attrs["feature_names"]
    summary_names = f["features/summary"].attrs["feature_names"]

# Example 1: Interpret spatial characteristics (INITIAL features)
# High spatial gradient → sharp interfaces or localized structures
spatial_gradient_idx = list(initial_names).index("ic_spatial_gradient_mean")
spatial_gradients = initial_features[:, :, spatial_gradient_idx].mean(axis=1)

print("Operators with high spatial gradients (sharp structures):")
high_gradient_ops = np.where(spatial_gradients > np.percentile(spatial_gradients, 90))[0]
print(f"  Found {len(high_gradient_ops)} operators in top 10%")

# Example 2: Cross-validate ARCHITECTURE and SUMMARY features
# Do high-noise operators show high SUMMARY entropy?
noise_scale_idx = 0  # First parameter in architecture features
noise_scales = arch_features[:, noise_scale_idx]

# Find SUMMARY entropy feature
entropy_idx = [i for i, name in enumerate(summary_names) if "entropy" in name][0]
summary_entropy = summary_features[:, entropy_idx]

correlation = np.corrcoef(noise_scales, summary_entropy)[0, 1]
print(f"\nNoise scale vs. SUMMARY entropy correlation: {correlation:.3f}")
print("  High correlation confirms features capture related behavioral aspects")

# Example 3: Identify behavioral regimes via SUMMARY spectral features
# Strong spectral peaks → periodic or quasi-periodic behavior
spectral_peak_indices = [i for i, name in enumerate(summary_names)
                         if "spectral" in name and "peak" in name]
spectral_strength = summary_features[:, spectral_peak_indices].max(axis=1)

print(f"\nOperators with strong periodic components:")
periodic_ops = np.where(spectral_strength > np.percentile(spectral_strength, 80))[0]
print(f"  Found {len(periodic_ops)} operators in top 20%")

# Example 4: Temporal evolution patterns
# Examine how variance evolves over time
variance_trajectory = temporal_features[:, :, :, 0]  # Assuming first feature is variance
mean_variance_trajectory = variance_trajectory.mean(axis=1)  # Average across realizations

# Classify temporal behaviors
early_variance = mean_variance_trajectory[:, :50].mean(axis=1)
late_variance = mean_variance_trajectory[:, -50:].mean(axis=1)
variance_growth = (late_variance - early_variance) / (early_variance + 1e-8)

print(f"\nTemporal behavior classification:")
print(f"  Growing operators (variance increases): {(variance_growth > 0.5).sum()}")
print(f"  Stable operators (variance constant): {(np.abs(variance_growth) < 0.5).sum()}")
print(f"  Decaying operators (variance decreases): {(variance_growth < -0.5).sum()}")
```

**Interpretation Tips:**

| Feature Family | High Values Indicate | Low Values Indicate |
|---------------|---------------------|-------------------|
| **INITIAL spatial gradients** | Sharp interfaces, localized structures | Smooth, diffuse initial conditions |
| **INITIAL spectral peaks** | Periodic initial patterns | Broadband or noisy initial conditions |
| **ARCHITECTURE noise scale** | High stochasticity, variability | Deterministic or low-noise dynamics |
| **SUMMARY entropy** | Chaotic or irregular dynamics | Ordered or simple patterns |
| **SUMMARY spectral power** | Periodic or quasi-periodic behavior | Aperiodic or chaotic behavior |
| **SUMMARY spatial variance** | Heterogeneous spatial patterns | Homogeneous or uniform states |
| **TEMPORAL growth rates** | Expanding or unstable dynamics | Contracting or stable dynamics |

**Cross-Validation Strategy:**

Multi-modal features enable consistency checking across perspectives:
- If **ARCHITECTURE** suggests chaotic behavior (high noise), do **SUMMARY** entropy features confirm?
- If **TEMPORAL** shows period-doubling, do **SUMMARY** spectral features detect harmonics?
- If **INITIAL** indicates smooth inputs, does **SUMMARY** show expected spatial autocorrelation?

This cross-validation increases confidence that discovered categories reflect genuine behavioral differences, not statistical artifacts.

### 4. Train VQ-VAE Tokenizer

```bash
poetry run spinlock train-vqvae \
    --dataset datasets/my_operators.h5 \
    --config configs/vqvae/production.yaml \
    --output checkpoints/vqvae/
```

This will:
- Load and concatenate all feature families (INITIAL+ARCHITECTURE+SUMMARY+TEMPORAL)
- Automatically clean features (NaN removal, variance filtering, deduplication)
- Discover ~8-15 categories via hierarchical clustering
- Train 3-level hierarchical VQ-VAE
- Save checkpoints and training history

**Expected time:** ~2-6 hours on GPU

### 5. Tokenize Operators

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

# Extract INITIAL features
ic_extractor = ICFeatureExtractor()
ic_features = ic_extractor.extract(initial_conditions)  # [N, M, 42]

# Extract SUMMARY features
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
- **Build NOA:** Use tokens to train Neural Operator Agent (Phase 1+)

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
- [Feature Families](features/README.md) - INITIAL, ARCHITECTURE, SUMMARY, TEMPORAL documentation
- [VQ-VAE Training](vqvae/training-guide.md) - Tokenization pipeline
- [NOA Roadmap](noa-roadmap.md) - Future development plan
