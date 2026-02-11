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
- Sample 10,000 operator parameter vectors using Sobol stratification (14D parameter space)
- Construct CNN-based neural operators
- Generate 500-timestep stochastic rollouts (3 realizations each)
- Extract INITIAL and TEMPORAL features inline (~328D per-timestep, v3.0)
- Store everything in HDF5 format

**v3.0 Note:** SUMMARY features removed. All features are now per-timestep computable for online NOA operation.

**Expected time:** ~12 hours on GPU

### 2. Inspect Dataset

```bash
poetry run spinlock inspect datasets/my_operators.h5
```

View dataset contents:
- Number of operators and realizations
- Feature dimensions (TEMPORAL ~328D per-timestep)
- Metadata (initial condition types, evolution policies, parameter stratification)
- Parameter space (14D Sobol unit cube)

### 3. Understanding Feature Semantics

The feature families provide complementary perspectives on operator behavior. Understanding what each feature measures enables interpretable discovery and validation.

**HDF5 Layout:** See [HDF5 Layout Reference](features/hdf5-layout.md) for the complete dataset schema.

```python
import h5py
import numpy as np
import json
from pathlib import Path

# Load dataset (v3.0)
with h5py.File("datasets/my_operators.h5", "r") as f:
    # TEMPORAL features: per-timestep behavioral features [N, T, D_temporal]
    temporal_features = f["/features/temporal/features"][:]  # [N, T, ~328]

    # ARCHITECTURE parameters: 14D Sobol unit cube [N, 14]
    params = f["/parameters/params"][:]

    # Initial conditions
    inputs = f["/inputs/fields"][:]

    # Feature registry for interpretability
    registry_json = f["/features/temporal"].attrs["feature_registry"]
    registry = json.loads(registry_json)

    print(f"Dataset shapes:")
    print(f"  TEMPORAL: {temporal_features.shape}")  # [N, T, ~328]
    print(f"  Parameters: {params.shape}")  # [N, 14]
    print(f"  Inputs: {inputs.shape}")  # [N, C, H, W]

# Example 1: Find features by category
# Registry structure: {category: {feature_name: index}}
def get_feature_indices(registry, category):
    """Get feature indices for a category."""
    if category not in registry:
        return []
    return list(registry[category].values())

spatial_indices = get_feature_indices(registry, "spatial")
spectral_indices = get_feature_indices(registry, "spectral")
cross_channel_indices = get_feature_indices(registry, "cross_channel")
temporal_dynamics_indices = get_feature_indices(registry, "temporal_dynamics")

print(f"Feature dimensions by category:")
print(f"  Spatial: {len(spatial_indices)} features (~105)")
print(f"  Spectral: {len(spectral_indices)} features (~93)")
print(f"  Cross-channel: {len(cross_channel_indices)} features (~10)")
print(f"  Temporal dynamics: {len(temporal_dynamics_indices)} features (~120)")

# Example 2: Analyze spectral features across time
# Strong spectral peaks → periodic or quasi-periodic behavior
if spectral_indices:
    # Average spectral features over time for each operator
    spectral_time_avg = temporal_features[:, :, spectral_indices].mean(axis=1)  # [N, D_spectral]
    spectral_strength = spectral_time_avg.max(axis=1)

    print(f"\nOperators with strong periodic components:")
    periodic_ops = np.where(spectral_strength > np.percentile(spectral_strength, 80))[0]
    print(f"  Found {len(periodic_ops)} operators in top 20%")

# Example 3: Temporal evolution patterns
# Examine how features evolve over time
early_mean = temporal_features[:, :50, :].mean(axis=(1, 2))  # Average over early timesteps
late_mean = temporal_features[:, -50:, :].mean(axis=(1, 2))  # Average over late timesteps
feature_growth = (late_mean - early_mean) / (np.abs(early_mean) + 1e-8)

print(f"\nTemporal behavior classification:")
print(f"  Growing operators: {(feature_growth > 0.5).sum()}")
print(f"  Stable operators: {(np.abs(feature_growth) < 0.5).sum()}")
print(f"  Decaying operators: {(feature_growth < -0.5).sum()}")
```

**Interpretation Tips (v3.0):**

| Feature Category | High Values Indicate | Low Values Indicate |
|-----------------|---------------------|-------------------|
| **Spatial gradients** | Sharp interfaces, localized structures | Smooth, diffuse patterns |
| **Spectral peaks** | Periodic or quasi-periodic behavior | Aperiodic or chaotic behavior |
| **Spectral entropy** | Chaotic or irregular dynamics | Ordered or simple patterns |
| **Temporal autocorrelation (windowed)** | Persistent dynamics | Rapidly changing states |
| **Stability metrics** | Unstable/chaotic trajectories | Stable/convergent behavior |
| **Phase space features** | Complex attractor dynamics | Simple trajectories |

**Cross-Validation Strategy (v3.0):**

Multi-modal per-timestep features enable consistency checking:
- If **parameter vectors** suggest high noise, do **spatial entropy** features at each timestep confirm chaotic behavior?
- If **temporal autocorrelation** shows periodic patterns, do **spectral** features detect corresponding harmonics?
- Compare early vs. late timesteps to detect pattern evolution and regime transitions
- Use **stability metrics** to validate parameter-inferred stability properties

This cross-validation increases confidence that discovered categories reflect genuine behavioral differences, not statistical artifacts.

### 4. Train VQ-VAE Tokenizer

```bash
poetry run spinlock train-vqvae \
    --dataset datasets/my_operators.h5 \
    --config configs/vqvae/production.yaml \
    --output checkpoints/vqvae/
```

This will:
- Load TEMPORAL per-timestep features from `/features/temporal/features`
- Optionally concatenate INITIAL features (computed inline from inputs)
- Automatically clean features (NaN removal, variance filtering, deduplication)
- Discover ~8-15 behavioral categories via hierarchical clustering
- Train 3-level hierarchical VQ-VAE on per-timestep features
- Save checkpoints and training history

**v3.0 Note:** VQ-VAE now operates on per-timestep TEMPORAL features, enabling online tokenization for NOA predictions.

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

### Example 2: Visualize Operator Dynamics

Create videos showing temporal evolution of operators with multiple aggregate views:

```bash
# Basic visualization with default settings
poetry run spinlock visualize-dataset \
    --dataset datasets/test_100.h5 \
    --output visualizations/evolution.mp4

# Diverse sampling (feature-based) to find interesting behaviors
poetry run spinlock visualize-dataset \
    --dataset datasets/100k_full_features.h5 \
    --output visualizations/diverse_operators.mp4 \
    --sampling-method diverse \
    --n-operators 8

# Custom aggregates: PCA, variance, entropy, spectral (FFT)
poetry run spinlock visualize-dataset \
    --dataset datasets/100k_full_features.h5 \
    --output visualizations/full_analysis.mp4 \
    --aggregates pca variance entropy spectral mean \
    --size 128x128 \
    --sampling-method diverse

# Filter by evolution policy (convex produces more dynamic/amoeba-like behavior)
poetry run spinlock visualize-dataset \
    --dataset datasets/100k_full_features.h5 \
    --output visualizations/convex_operators.mp4 \
    --evolution-policy convex \
    --sampling-method diverse \
    --aggregates pca variance mean
```

**Sampling methods:**
- `sobol` (default): Low-discrepancy space-filling sampling
- `diverse`: Feature-based interestingness scoring (entropy + outlier distance + variance)
- `random`: Uniform random sampling
- `sequential`: First N operators

**Evolution policy filter:**
- `--evolution-policy convex`: Select operators using convex mixing (more dynamic, amoeba-like)
- `--evolution-policy residual`: Select operators using residual/Euler integration (more stable)

**Aggregate renderers:**
- `mean`: Mean field across realizations
- `variance`: Spatial variance map (uncertainty)
- `entropy`: Shannon entropy (structural uncertainty)
- `pca`: PCA modes as RGB (PC1/PC2/PC3)
- `spectral`: FFT power spectrum
- `envelope`: Min/max range
- `ssim`: Structural similarity

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

## Working with Theta Parameters

### Generate Dataset with Theta Tokens

```bash
# Standard CNO dataset (includes theta parameters)
poetry run spinlock generate-cno-dataset \
  --num-samples 50000 \
  --output datasets/cno_50k.h5
```

### Train Theta Tokenizer

```bash
# Multi-family tokenizer (temporal + initial + theta)
poetry run spinlock train-vq-tokenizer \
  --config configs/vqvae_50k.yaml \
  --dataset datasets/cno_50k.h5 \
  --checkpoint-dir checkpoints/theta_tokenizer
```

### Verify Theta Reconstruction

```python
from spinlock.tokens import VQTokenizer
import torch

tokenizer = VQTokenizer.from_checkpoint('checkpoints/theta_tokenizer/best_model.pt')

# Tokenize parameters
theta_params = torch.randn(10, 14).sigmoid()  # [B, 14] in [0,1]
tokens = tokenizer.encode(theta_features=theta_params)

# Decode and check roundtrip
outputs = tokenizer.forward(theta_features=theta_params)
reconstructed = outputs['decoded']['theta']

# Measure reconstruction quality
mse = torch.nn.functional.mse_loss(reconstructed, theta_params)
print(f"Theta reconstruction MSE: {mse:.6f}")  # Should be <0.01

# Check roundtrip consistency
roundtrip_tokens = tokenizer.encode(theta_features=reconstructed)
match_rate = sum(
    (roundtrip_tokens[k] == tokens[k]).float().mean()
    for k in tokens
) / len(tokens)
print(f"Token match rate: {match_rate:.2%}")  # Should be >90%
```

## Working with Quantum Features

### Generate QBM Dataset

```bash
poetry run spinlock generate-qbm-dataset \
  --num-samples 10000 \
  --num-realizations 3 \
  --output datasets/qbm_10k.h5
```

### Train QBM Tokenizer

```bash
# Uses 188D temporal features (178 standard + 10 quantum)
poetry run spinlock train-vq-tokenizer \
  --config configs/vqvae_qbm.yaml \
  --dataset datasets/qbm_10k.h5 \
  --checkpoint-dir checkpoints/qbm_tokenizer
```

### Analyze Quantum Features

```python
from spinlock.data import SpinlockDataset

dataset = SpinlockDataset.from_file('datasets/qbm_10k.h5')

with dataset.open():
    temporal = dataset.features.temporal.load_all()  # [N, T, 188]

    # Extract quantum subset
    quantum_feats = temporal[..., 178:]  # Last 10 dimensions

    # Analyze decoherence
    purity = quantum_feats[:, :, 0]  # First quantum feature

    import matplotlib.pyplot as plt
    plt.plot(purity[0], label='Sample 0')
    plt.xlabel('Timestep')
    plt.ylabel('Purity Tr(ρ²)')
    plt.title('Quantum Decoherence')
    plt.legend()
    plt.show()
```

## Choosing a Training Regime

See [Training Regimes Guide](training-regimes-guide.md) for detailed comparison.

**Quick Recommendation:**
- **General use**: Roundtrip-first training (`configs/vqvae_50k.yaml`)
- **Ablation studies**: Independent training (`configs/ablation_independent.yaml`)
- **Memory-constrained**: Independent without inverse heads

```bash
# Recommended: Roundtrip-first (150 epochs, best metrics)
poetry run spinlock train-vq-tokenizer \
  --config configs/vqvae_50k.yaml \
  --dataset datasets/cno_50k.h5

# Ablation: Independent (200 epochs, baseline)
poetry run spinlock train-vq-tokenizer \
  --config configs/ablation_independent.yaml \
  --dataset datasets/cno_50k.h5
```

## References

- [Architecture](architecture.md) - System design details
- [Feature Families](features/README.md) - Feature family documentation (TEMPORAL, INITIAL, Theta)
- [HDF5 Layout](features/hdf5-layout.md) - Dataset schema reference
- [VQ-VAE Training](vqvae/training-guide.md) - Tokenization pipeline
- [Theta Features Guide](theta-features-guide.md) - Theta parameter tokenization
- [Quantum Features Guide](quantum-features-guide.md) - QBM and quantum observables
- [Training Regimes Guide](training-regimes-guide.md) - Roundtrip vs independent training
- [Dataset Generation](dataset-generation.md) - CNO, QBM, and MNO dataset creation
- [NOA Roadmap](noa-roadmap.md) - Future development plan
