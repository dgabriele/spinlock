# Learnable Category Assignments - Quick Start Guide

## What is Learnable Assignment?

Traditional VQ-VAE uses **static pre-clustering** to assign features to categories. Learnable assignment replaces this with **end-to-end gradient-based optimization**, where category assignments are learned during training to minimize reconstruction loss.

### Benefits
- 🎯 **Task-optimal categories** - Learned to minimize reconstruction, not just correlation
- ⚖️ **Automatic balancing** - No dead code reset needed
- 🔧 **Simpler config** - Fewer hyperparameters to tune
- 🔀 **End-to-end** - Gradients flow through entire pipeline

## Quick Start

### 1. Use the Pre-configured YAML

```bash
poetry run spinlock train-vqvae \
  --config configs/vqvae/learnable_assignment.yaml \
  --verbose
```

### 2. Or Add to Your Existing Config

```yaml
# In your VQ-VAE config YAML:
training:
  category_assignment: learnable  # Enable learnable mode

learnable_assignment:
  # Temperature annealing (1.0 → 0.1)
  temperature_start: 1.0
  temperature_end: 0.1
  temperature_schedule: "linear"  # Options: linear, exponential, cosine

  # Loss weights
  orthogonality_weight: 0.1
  balance_weight: 0.05
  family_constraint_weight: 1.0  # For per-family mode

  # Optimization
  assignment_lr: 0.001  # Lower than main LR for stability
```

### 3. Or Use CLI Flag

```bash
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae.yaml \
  --learnable \
  --verbose
```

## How It Works

### 1. Initialization (Epoch 0)
```
features [N, D]
  → clustering (silhouette method)
  → init_logits [D, K]  # High confidence (5.0) per cluster
  → SoftAssignmentMatrix(init_logits)  # Learnable parameter
```

### 2. Training Loop (Epochs 1-500)
```
x [B, D]
  → temperature = scheduler(epoch)  # 1.0 → 0.1
  → soft_assign [D, K] = assignment_matrix(temperature)  # Gumbel-Softmax
  → category_embs [B, K, E] = soft_router(x, soft_assign)  # Weighted routing
  → latents → quantize → decode → reconstruction
  → L_total = L_recon + L_vq + L_ortho + L_balance
  → backward() → optimizer.step()  # Updates both model AND assignments
```

### 3. Inference (After Training)
```
learned_assignments = assignment_matrix.get_hard_assignments()  # Argmax
model.freeze_assignments()  # Convert to static GroupedFeatureExtractor
(Standard VQ-VAE inference)
```

## Configuration Options

### Temperature Annealing

Controls how "soft" the assignments are:

```yaml
learnable_assignment:
  temperature_start: 1.0   # Soft (uniform-like)
  temperature_end: 0.1     # Hard (sharp)
  temperature_schedule: "linear"  # linear, exponential, or cosine
```

- **High temperature (1.0)**: Assignments are soft, features route to multiple categories
- **Low temperature (0.1)**: Assignments are hard, features route to single category
- **Annealing**: Gradually transition from soft → hard over training

### Loss Weights

```yaml
learnable_assignment:
  orthogonality_weight: 0.1   # Penalize correlated categories
  balance_weight: 0.05        # Penalize unbalanced category sizes
  family_constraint_weight: 1.0  # Penalize cross-family assignments (per-family mode)
```

### Freezing Assignments

Optionally freeze assignments after convergence:

```yaml
learnable_assignment:
  freeze_after_epochs: 300  # Freeze at epoch 300
  # Or null to keep learning throughout training
```

## Comparison: Static vs Learnable

| Aspect | Static Pre-Clustering | Learnable Assignment |
|--------|----------------------|---------------------|
| **Objective** | Minimize correlation distance | Minimize reconstruction loss |
| **Optimization** | One-time (before training) | Continuous (during training) |
| **Balancing** | Manual (dead code reset) | Automatic (balance loss) |
| **Flexibility** | Fixed after clustering | Adapts during training |
| **Complexity** | More hyperparameters | Fewer hyperparameters |
| **Speed** | Faster (no assignment updates) | 1.5-2x slower (dual optimization) |

## Monitoring Training

Key metrics to watch:

```
Epoch 50/500 (12.3s): train=0.0234 val=0.0289
  Components: recon=0.0180 vq=0.0024 ortho=0.0015 info=0.0008 topo=0.0007
              assign_orthogonality=0.0003 assign_balance=0.0001 assign_total=0.0004
  Temperature: 0.90
  Utilization: 23.4%
```

**What to look for:**
- `assign_orthogonality` decreasing (categories becoming more distinct)
- `assign_balance` decreasing (categories becoming more balanced)
- `temperature` annealing from 1.0 → 0.1
- `utilization` stable (no category collapse)

## Troubleshooting

### Category Collapse
**Symptom:** All features assign to 1-2 categories

**Solutions:**
1. Increase `balance_weight` (0.05 → 0.10)
2. Use stronger clustering initialization
3. Increase `temperature_end` (0.1 → 0.3) for softer final assignments

### Slow Convergence
**Symptom:** Assignment losses not decreasing

**Solutions:**
1. Increase `assignment_lr` (0.001 → 0.003)
2. Use exponential annealing (faster early decrease)
3. Decrease `orthogonality_weight` (may be too restrictive)

### NaN Losses
**Symptom:** Training crashes with NaN values

**Solutions:**
1. Enable gradient clipping (default: 1.0, try 0.5)
2. Decrease `assignment_lr` (0.001 → 0.0005)
3. Check feature cleaning is enabled

### Cross-Family Leakage (Per-Family Mode)
**Symptom:** Features from family A routing to family B categories

**Solutions:**
1. Increase `family_constraint_weight` (1.0 → 2.0)
2. Check `PerFamilyAssignmentMatrix` is being used
3. Verify family definitions in config

## Advanced Usage

### Programmatic API

```python
from spinlock.encoding import (
    CategoricalVQVAEConfig,
    LearnableCategoricalVQVAE,
    LearnableAssignmentConfig
)
from spinlock.encoding.learnable_assignment import initialize_from_clustering
from spinlock.encoding.training import LearnableVQVAETrainer
from spinlock.encoding.training.annealing import TemperatureScheduler

# Create configs
vqvae_config = CategoricalVQVAEConfig(
    input_dim=features.shape[1],
    group_indices=group_indices,
    group_embedding_dim=512,
    # ... other params
)

learnable_config = LearnableAssignmentConfig(
    temperature_start=1.0,
    temperature_end=0.1,
    orthogonality_weight=0.1,
    balance_weight=0.05,
    assignment_lr=0.001
)

# Create model
model = LearnableCategoricalVQVAE(vqvae_config, learnable_config)

# Initialize assignment matrix from clustering
init_logits, categories_per_family = initialize_from_clustering(
    features, feature_names, per_family=True, clustering_params=params
)

# Create trainer
temp_scheduler = TemperatureScheduler(
    start=1.0, end=0.1, total_epochs=500, schedule="linear"
)

trainer = LearnableVQVAETrainer(
    model, train_loader, val_loader,
    temp_scheduler=temp_scheduler,
    assignment_lr=0.001
)

# Train
history = trainer.train(epochs=500)

# Freeze assignments
model.freeze_assignments()
```

## Testing

Run the test suite:

```bash
# Unit tests
poetry run pytest tests/test_learnable_assignment.py -v

# Integration tests
poetry run pytest tests/test_learnable_integration.py -v

# All tests
poetry run pytest tests/test_learnable*.py -v
```

Expected: 16/16 tests passing

## References

- **Plan**: Original implementation plan with architecture details
- **Code**: `src/spinlock/encoding/learnable_*.py` for core modules
- **Tests**: `tests/test_learnable_*.py` for examples
- **Config**: `configs/vqvae/learnable_assignment.yaml` for full example

## Questions?

Check the implementation summary (`IMPLEMENTATION_SUMMARY.md`) for:
- Design decisions and rationale
- Complete file listing
- Future enhancement ideas
- Known limitations
