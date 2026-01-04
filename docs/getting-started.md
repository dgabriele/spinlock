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
- Number of operators and realizations
- Feature dimensions (SUMMARY aggregated, TEMPORAL if enabled)
- Metadata (initial condition types, evolution policies, parameter stratification)

### 3. Understanding Feature Semantics

The feature families provide complementary perspectives on operator behavior. Understanding what each feature measures enables interpretable discovery and validation.

**HDF5 Layout:** See [HDF5 Layout Reference](features/hdf5-layout.md) for the complete dataset schema.

```python
import h5py
import numpy as np
import json
from pathlib import Path

# Load dataset
with h5py.File("datasets/my_operators.h5", "r") as f:
    # SUMMARY features: aggregated behavioral statistics [N, D]
    summary_features = f["/features/summary/aggregated/features"][:]

    # TEMPORAL features: per-timestep time series [N, T, D] (if enabled)
    if "/features/temporal/features" in f:
        temporal_features = f["/features/temporal/features"][:]
    else:
        temporal_features = None

    # ARCHITECTURE features: from parameter vectors [N, P]
    arch_features = f["/parameters/vectors"][:]

    # Feature registry for interpretability
    registry_json = f["/features/summary"].attrs["feature_registry"]
    registry = json.loads(registry_json)

# Example 1: Find features by category
# Registry structure: {category: {feature_name: index}}
def get_feature_indices(registry, category):
    """Get feature indices for a category."""
    if category not in registry:
        return []
    return list(registry[category].values())

spatial_indices = get_feature_indices(registry, "spatial")
spectral_indices = get_feature_indices(registry, "spectral")
temporal_indices = get_feature_indices(registry, "temporal")

print(f"Feature dimensions by category:")
print(f"  Spatial: {len(spatial_indices)} features")
print(f"  Spectral: {len(spectral_indices)} features")
print(f"  Temporal: {len(temporal_indices)} features")

# Example 2: Analyze SUMMARY spectral features
# Strong spectral peaks → periodic or quasi-periodic behavior
if spectral_indices:
    spectral_features = summary_features[:, spectral_indices]
    spectral_strength = spectral_features.max(axis=1)

    print(f"\nOperators with strong periodic components:")
    periodic_ops = np.where(spectral_strength > np.percentile(spectral_strength, 80))[0]
    print(f"  Found {len(periodic_ops)} operators in top 20%")

# Example 3: Temporal evolution patterns (if TEMPORAL enabled)
if temporal_features is not None:
    # Examine how features evolve over time
    early_mean = temporal_features[:, :50, :].mean(axis=(1, 2))
    late_mean = temporal_features[:, -50:, :].mean(axis=(1, 2))
    feature_growth = (late_mean - early_mean) / (np.abs(early_mean) + 1e-8)

    print(f"\nTemporal behavior classification:")
    print(f"  Growing operators: {(feature_growth > 0.5).sum()}")
    print(f"  Stable operators: {(np.abs(feature_growth) < 0.5).sum()}")
    print(f"  Decaying operators: {(feature_growth < -0.5).sum()}")
else:
    print("\nTEMPORAL features not available in this dataset")
```

**Interpretation Tips:**

| Feature Category | High Values Indicate | Low Values Indicate |
|-----------------|---------------------|-------------------|
| **Spatial gradients** | Sharp interfaces, localized structures | Smooth, diffuse patterns |
| **Spectral peaks** | Periodic or quasi-periodic behavior | Aperiodic or chaotic behavior |
| **Spectral entropy** | Chaotic or irregular dynamics | Ordered or simple patterns |
| **Temporal autocorrelation** | Persistent dynamics | Rapidly changing states |
| **Causality metrics** | Strong information flow | Weak dependencies |
| **Invariant drift** | Evolving behavioral regimes | Stable dynamics |

**Cross-Validation Strategy:**

Multi-modal features enable consistency checking across perspectives:
- If **parameter vectors** suggest high noise, do **SUMMARY** entropy features confirm chaotic behavior?
- If **TEMPORAL** features show period-doubling, do **SUMMARY** spectral features detect harmonics?
- Compare spatial features at early vs. late timesteps to detect pattern evolution.

This cross-validation increases confidence that discovered categories reflect genuine behavioral differences, not statistical artifacts.

### 4. Train VQ-VAE Tokenizer

```bash
poetry run spinlock train-vqvae \
    --dataset datasets/my_operators.h5 \
    --config configs/vqvae/production.yaml \
    --output checkpoints/vqvae/
```

This will:
- Load SUMMARY aggregated features from `/features/summary/aggregated/features`
- Optionally concatenate TEMPORAL features if available
- Automatically clean features (NaN removal, variance filtering, deduplication)
- Discover ~8-15 categories via hierarchical clustering
- Train 3-level hierarchical VQ-VAE
- Save checkpoints and training history

**Expected time:** ~2-6 hours on GPU

### 5. Tokenize Operators

```python
import torch
import yaml
from pathlib import Path
from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig

# Load VQ-VAE configuration
with open("checkpoints/vqvae/config.yaml") as f:
    config_dict = yaml.safe_load(f)

# Construct model from config
config = CategoricalVQVAEConfig(**config_dict["model"])
model = CategoricalHierarchicalVQVAE(config)

# Load trained weights
checkpoint = torch.load("checkpoints/vqvae/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load features for new operators
features = ...  # Shape: [N, D] where D is total feature dimension

# Extract behavioral tokens
with torch.no_grad():
    # Returns [N, num_categories * num_levels] token tensor
    # Organized as [category_1_L0, category_1_L1, category_1_L2, category_2_L0, ...]
    tokens = model.get_tokens(features)
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
from spinlock.features.summary import SummaryExtractor, SummaryConfig
import torch

# Configure and create SUMMARY extractor
summary_config = SummaryConfig(
    per_channel=True,
    temporal_aggregation=["mean", "std", "trend"],
    realization_aggregation=["mean", "std", "cv"]
)
summary_extractor = SummaryExtractor(summary_config, device=torch.device('cuda'))

# Extract SUMMARY features from rollouts [N, M, T, C, H, W]
summary_features = summary_extractor.extract_all(rollouts)  # [N, ~360]
```

See [Feature Families README](features/README.md) for details on available extractors.

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
metadata:
  name: "custom_dataset"

sampling:
  total_samples: 5000

simulation:
  num_realizations: 3
  num_timesteps: 500
  operator_type: "cnn"

  input_generation:
    method: "sampled"
    grid_size: 128

# Feature extraction config
features:
  temporal:
    enabled: false  # Disable per-timestep features to save space
  summary:
    enabled: true   # Enable aggregated SUMMARY features
```

See [HDF5 Layout Reference](features/hdf5-layout.md) for details on the feature storage structure.

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
- [Feature Families](features/README.md) - Feature family documentation (TEMPORAL, SUMMARY)
- [HDF5 Layout](features/hdf5-layout.md) - Dataset schema reference
- [VQ-VAE Training](vqvae/training-guide.md) - Tokenization pipeline
- [NOA Roadmap](noa-roadmap.md) - Future development plan
