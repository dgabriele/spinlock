# VQ-VAE Checkpoint Format

**Location**: `docs/vqvae/checkpoint-format.md`
**Last Updated**: 2026-01-10
**Related**: `docs/baselines/100k-full-features-dataset.md`

This document describes the structure and contents of VQ-VAE model checkpoints produced by `spinlock train-vqvae`.

## Directory Structure

A complete VQ-VAE checkpoint directory contains:

```
checkpoints/production/100k_3family_v1/
├── best_model.pt              # Best model checkpoint (primary file)
├── final_model.pt             # Final epoch checkpoint
├── normalization_stats.npz    # Feature normalization parameters
├── config.yaml                # Resolved training configuration
└── training_history.json      # Metrics history across epochs
```

## File Descriptions

### 1. `best_model.pt` (Primary Checkpoint)

PyTorch checkpoint containing the complete model state and training metadata.

**Top-Level Keys**:

| Key | Type | Description |
|-----|------|-------------|
| `model_state_dict` | dict | PyTorch state dict with all model weights |
| `optimizer_state_dict` | dict | Optimizer state for resuming training |
| `epoch` | int | Training epoch when checkpoint was saved |
| `val_loss` | float | Validation loss at this checkpoint |
| `metrics` | dict | All training/validation metrics |
| `config` | dict | Full training configuration (raw YAML) |
| `model_config` | dict | Model architecture parameters (see below) |
| `normalization_stats` | dict | Per-category normalization statistics |
| `pre_model_group_indices` | dict | Category assignments before model init |
| `feature_names` | list | Names of all features (143 total) |
| `feature_mask` | ndarray | Boolean mask for feature selection |
| `feature_cleaning_params` | dict | Parameters for feature preprocessing |
| `encoder_state_dicts` | dict | Pre-trained encoder states (if applicable) |
| `prng_states` | dict | Random number generator states for reproducibility |

**`model_config` Structure** (Canonical Model Parameters):

```python
{
    "input_dim": 225,              # Input feature dimension (after encoding)
    "group_indices": {             # Category → feature index mapping
        "cluster_1": [0, 3, 7, ...],
        "cluster_2": [1, 5, 9, ...],
        ...
    },
    "group_embedding_dim": 256,    # Embedding dimension per category
    "group_hidden_dim": 512,       # Hidden dimension in group MLPs
    "levels": [                    # Hierarchical VQ levels per category
        {"num_tokens": 24, "latent_dim": 48},
        {"num_tokens": 24, "latent_dim": 12},
        {"num_tokens": 24, "latent_dim": 8},
    ]
}
```

**`normalization_stats` Structure**:

```python
{
    "cluster_1": NormalizationStats(mean=..., std=...),
    "cluster_2": NormalizationStats(mean=..., std=...),
    ...
}
```

Each category has its own `NormalizationStats` object containing:
- `mean`: Per-feature mean values (numpy array)
- `std`: Per-feature standard deviation (numpy array)

These stats are used to normalize features before VQ encoding:
```python
normalized = (features - mean) / (std + epsilon)
```

### 2. `normalization_stats.npz` (NumPy Archive)

Standalone normalization parameters for feature preprocessing. This file mirrors the `normalization_stats` dict from the checkpoint but in a more portable format.

**Keys**:

```
_normalization_method: shape=(1,), dtype=<U8       # "standard" or "mad"
cluster_1_mean: shape=(7,), dtype=float32
cluster_1_std: shape=(7,), dtype=float32
cluster_2_mean: shape=(17,), dtype=float32
cluster_2_std: shape=(17,), dtype=float32
...
```

**Loading**:

```python
import numpy as np

stats = np.load("normalization_stats.npz")
cluster_1_mean = stats["cluster_1_mean"]
cluster_1_std = stats["cluster_1_std"]
normalization_method = str(stats["_normalization_method"][0])  # "standard"
```

### 3. `config.yaml` (Training Configuration)

Resolved training configuration with all defaults expanded. This is the actual config used during training, not the user-provided YAML.

**Key Sections**:

```yaml
# Dataset
dataset_path: datasets/100k_full_features.h5
batch_size: 1024

# Feature Families
families:
  initial:
    encoder: initial_hybrid
    encoder_params: {...}
  summary:
    encoder: MLPEncoder
    encoder_params: {...}
  temporal:
    encoder: TemporalCNNEncoder
    encoder_params: {...}

# Model Architecture
model:
  group_embedding_dim: 256
  group_hidden_dim: 512
  levels: []  # Auto-scaling
  compression_ratios: "auto"

# Training Hyperparameters
training:
  learning_rate: 0.001
  num_epochs: 550
  optimizer: adam
  ...
```

### 4. `training_history.json` (Metrics Log)

Complete training history with per-epoch metrics.

**Structure**:

```json
{
  "train_loss": [3.334, 1.572, 1.230, ...],
  "val_loss": [0.394, 0.394, 0.394, 0.238, ...],
  "reconstruction_quality": [0.891, 0.915, 0.923, ...],
  "codebook_utilization": [16.8, 15.0, 14.5, ...],
  "topographic_pre": [0.809, 0.850, 0.868, ...],
  "topographic_post": [0.998, 0.998, 0.998, ...],
  ...
}
```

Each key contains a list of values (one per epoch).

## Loading Checkpoints

### Basic Model Loading

```python
import torch
from spinlock.encoding import CategoricalHierarchicalVQVAE

# Load checkpoint
checkpoint = torch.load("best_model.pt", map_location="cpu", weights_only=False)

# Method 1: Use model_config (recommended)
model_config = checkpoint["model_config"]
vqvae = CategoricalHierarchicalVQVAE(
    input_dim=model_config["input_dim"],
    group_indices=model_config["group_indices"],
    group_embedding_dim=model_config["group_embedding_dim"],
    group_hidden_dim=model_config["group_hidden_dim"],
    levels=model_config["levels"],
)

# Load weights
vqvae.load_state_dict(checkpoint["model_state_dict"])
vqvae.eval()
```

### Loading with Normalization Stats

```python
from spinlock.mno.vqvae_alignment import VQVAEAlignmentLoss

# Load VQ-VAE with alignment (includes normalization)
alignment = VQVAEAlignmentLoss.from_checkpoint(
    vqvae_path="checkpoints/production/100k_3family_v1",
    device="cuda",
    use_aligned_extractor=True,
)

# Alignment includes:
# - alignment.vqvae (frozen VQ-VAE model)
# - alignment.feature_extractor (feature extraction pipeline)
# - alignment.normalization_stats (per-category stats)
# - alignment.group_indices (category assignments)
```

### Accessing Normalization Stats

```python
# From checkpoint
checkpoint = torch.load("best_model.pt", weights_only=False)
norm_stats = checkpoint["normalization_stats"]

# From .npz file
import numpy as np
stats = np.load("normalization_stats.npz")

# Apply normalization
def normalize_features(features, category_name, norm_stats):
    """Normalize features using per-category statistics."""
    mean = norm_stats[f"{category_name}_mean"]
    std = norm_stats[f"{category_name}_std"]
    return (features - mean) / (std + 1e-8)
```

## Key Design Decisions

### Why Two Normalization Files?

1. **In checkpoint (`normalization_stats` dict)**:
   - Primary source of truth
   - Guaranteed to match model weights
   - Includes Python objects (NormalizationStats dataclasses)

2. **Standalone `.npz` file**:
   - Portable format (NumPy arrays only)
   - Can be used independently of PyTorch
   - Easier for inspection and debugging

**Best Practice**: Always load from checkpoint's `normalization_stats` dict to ensure consistency with model weights.

### Feature Dimension Mismatch Prevention

The checkpoint contains multiple sources for feature dimensions:

1. **`model_config["input_dim"]`**: Canonical input dimension (e.g., 225)
2. **`feature_mask`**: Boolean array marking which raw features are used
3. **`group_indices`**: Category assignments (sum of lengths = input_dim)
4. **`normalization_stats`**: Per-category stats (dimensions must match group_indices)

**Validation**:
```python
# Verify consistency
input_dim = model_config["input_dim"]
total_features = sum(len(indices) for indices in group_indices.values())
assert input_dim == total_features, f"Mismatch: {input_dim} vs {total_features}"

# Verify normalization stats match
for cat_name, indices in group_indices.items():
    expected_dim = len(indices)
    actual_dim = len(normalization_stats[f"{cat_name}_mean"])
    assert expected_dim == actual_dim, f"{cat_name}: {expected_dim} vs {actual_dim}"
```

## Backward Compatibility

### Handling Old Checkpoints

Checkpoints created before 2026-01-10 may not have `normalization_stats` in the checkpoint dict.

**Fallback Strategy**:

```python
checkpoint = torch.load("best_model.pt", weights_only=False)

# Try checkpoint dict first
normalization_stats = checkpoint.get("normalization_stats")

# Fallback to .npz file if not in checkpoint
if normalization_stats is None:
    npz_path = checkpoint_dir / "normalization_stats.npz"
    if npz_path.exists():
        normalization_stats = dict(np.load(npz_path))
    else:
        # No normalization was used during training
        normalization_stats = None
        print("Warning: No normalization stats found. VQ-VAE was trained without normalization.")
```

## Common Issues

### Issue 1: Dimension Mismatch in Normalization

**Symptom**: `RuntimeError: The size of tensor a (37) must match the size of tensor b (9)`

**Cause**: Normalization stats file doesn't match checkpoint's `group_indices`.

**Solution**:
```python
# Always use normalization_stats from checkpoint, not .npz file
checkpoint = torch.load("best_model.pt", weights_only=False)
normalization_stats = checkpoint["normalization_stats"]  # Guaranteed to match
```

### Issue 2: Missing `model_config`

**Symptom**: `KeyError: 'model_config'`

**Cause**: Old checkpoint format (pre-2024).

**Solution**:
```python
# Fallback to config dict
if "model_config" in checkpoint:
    model_config = checkpoint["model_config"]
else:
    # Reconstruct from config
    config = checkpoint["config"]
    model_config = {
        "input_dim": config.get("input_dim", 187),
        "group_indices": checkpoint.get("pre_model_group_indices", {}),
        "group_embedding_dim": config["model"]["group_embedding_dim"],
        "group_hidden_dim": config["model"]["group_hidden_dim"],
        "levels": config["model"].get("levels", []),
    }
```

### Issue 3: Compiled Model Prefix

**Symptom**: State dict keys have `_orig_mod.` prefix

**Cause**: Model was saved after `torch.compile()`.

**Solution**:
```python
state_dict = checkpoint["model_state_dict"]

# Remove compiled prefix
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

vqvae.load_state_dict(state_dict)
```

## Production Checkpoints

### Current Production Model

**Location**: `checkpoints/production/100k_3family_v1/`

**Specifications**:
- **Dataset**: 100K samples, 3 feature families (INITIAL, SUMMARY, TEMPORAL)
- **Architecture**: Hybrid VQ-VAE with CNN-based INITIAL encoder
- **Input Dimension**: 225 features (after family encoders)
- **Categories**: 10 auto-discovered clusters
- **Levels**: 3 hierarchical levels per category (auto-scaled)
- **Codebook Sizes**: ~24 tokens per level (adaptive compression)
- **Training Epochs**: 550
- **Best Val Loss**: 0.211

**Performance Metrics**:
- Reconstruction Quality: 94.0%
- Codebook Utilization: 14.5%
- Topographic Similarity: 83.5% (pre), 99.8% (post)

## Related Documentation

- **Dataset Format**: `docs/baselines/100k-full-features-dataset.md`
- **VQ-VAE Architecture**: `docs/vqvae/multi-family-encoders.md`
- **Training Guide**: CLI help via `spinlock train-vqvae --help`
- **Feature Extraction**: `src/spinlock/noa/feature_extraction.py` (docstrings)

## Version History

- **2026-01-10**: Added `normalization_stats` to checkpoint dict (previously only in .npz)
- **2025-01**: Added `model_config` for canonical architecture parameters
- **2024-12**: Added `encoder_state_dicts` for pre-trained encoder support
- **2024-11**: Initial checkpoint format with `pre_model_group_indices`
