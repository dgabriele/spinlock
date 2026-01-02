# VQ-VAE Training Configurations

This directory contains configuration files for training the Categorical Hierarchical VQ-VAE tokenizer on operator behavioral features.

## Configuration Format

All VQ-VAE training configs use a **multi-family nested format** for clarity and modularity.

### Multi-Family Nested Format

```yaml
# Dataset
dataset_path: "datasets/baseline_10k.h5"
max_samples: null  # Optional: limit dataset size

# Feature Families
families:
  summary:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [256, 128]
      output_dim: 64
      dropout: 0.1
      activation: "relu"
      batch_norm: true

  architecture:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [128, 64]
      output_dim: 64
      dropout: 0.1
      activation: "relu"
      batch_norm: true

# VQ-VAE Architecture
model:
  group_embedding_dim: 64
  group_hidden_dim: 128
  levels: []  # Empty for auto-scaling
  commitment_cost: 0.25
  use_ema: true
  decay: 0.99
  dropout: 0.1
  orthogonality_weight: 0.1
  informativeness_weight: 0.1

# Training
training:
  batch_size: 512
  learning_rate: 0.001
  num_epochs: 500
  optimizer: "adam"
  scheduler: null
  warmup_epochs: 0

  # Category discovery
  category_assignment: "auto"
  num_categories_auto: null
  orthogonality_target: 0.15
  min_features_per_category: 3
  max_clusters: 25

  # Loss weights
  reconstruction_weight: 1.0
  vq_weight: 1.0
  orthogonality_weight: 0.1
  informativeness_weight: 0.1
  topo_weight: 0.3
  topo_samples: 512

  # Checkpointing
  checkpoint_dir: "checkpoints/vqvae"
  save_every: null

  # Callbacks
  early_stopping_patience: 100
  early_stopping_min_delta: 0.001
  dead_code_reset_interval: 100
  dead_code_threshold: 10.0
  dead_code_max_reset_fraction: 0.25

  # Validation
  val_every_n_epochs: 5

  # Performance
  use_torch_compile: true

# Logging
logging:
  wandb: false
  log_interval: 100
  eval_interval: 500
  verbose: true

# Random seed
random_seed: 42
```

See `default.yaml` for complete template with all options documented.

## Available Configurations

### Validation Configs

Progressive validation configs for testing feature families:

- **`validation/1k_initial_only.yaml`** - INITIAL features (42D IC features)
- **`validation/1k_architecture_only.yaml`** - ARCHITECTURE features (21D operator structure)
- **`validation/1k_summary_only.yaml`** - SUMMARY features (275D aggregated statistics)
- **`validation/1k_arch_summary.yaml`** - Joint ARCHITECTURE + SUMMARY training
- **`validation/1k_arch_summary_highgpu.yaml`** - High-GPU-utilization version (8× batch, 4× model)

### Production Configs

- **`default.yaml`** - Template configuration (all options documented)
- **`summary_1k.yaml`** - SUMMARY features with manual category mapping
- **`summary_1k_auto.yaml`** - SUMMARY features with auto category discovery

## Usage

### Training from Scratch

```bash
# Train VQ-VAE on validation dataset (1K samples)
poetry run spinlock train-vqvae \
  --config configs/vqvae/validation/1k_arch_summary.yaml \
  --verbose

# Train on full dataset
poetry run spinlock train-vqvae \
  --config configs/vqvae/production/10k_arch_summary_400epochs.yaml \
  --verbose
```

### Resuming Training

```bash
# Resume from checkpoint
poetry run spinlock train-vqvae \
  --config configs/vqvae/validation/1k_arch_summary.yaml \
  --resume-from checkpoints/validation/1k_arch_summary/best_model.pt
```

### Configuration Overrides

```bash
# Override specific parameters via CLI
poetry run spinlock train-vqvae \
  --config configs/vqvae/validation/1k_arch_summary.yaml \
  --epochs 200 \
  --batch-size 128 \
  --learning-rate 0.0005
```

### Dry Run

```bash
# Validate configuration without training
poetry run spinlock train-vqvae \
  --config configs/vqvae/validation/1k_arch_summary.yaml \
  --dry-run \
  --verbose
```

## Custom Configuration

Copy `default.yaml` and customize:

```bash
cp configs/vqvae/default.yaml configs/vqvae/my_experiment.yaml
# Edit my_experiment.yaml with your parameters
poetry run spinlock train-vqvae --config configs/vqvae/my_experiment.yaml
```

## Expected Outputs

After training, the output directory contains:

```
checkpoints/vqvae/my_experiment/
├── best_model.pt              # Best model checkpoint
├── normalization_stats.npz    # Per-category normalization stats
├── training_history.json      # Training metrics history
├── config.yaml               # Resolved configuration
└── checkpoint_epoch_*.pt      # Periodic checkpoints (if save_every set)
```

### Checkpoint Contents

The `best_model.pt` checkpoint contains:

- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state (for resuming)
- `group_indices`: Discovered category assignments
- `normalization_stats`: Per-category mean/std
- `feature_names`: Input feature names
- `config`: Model configuration
- `history`: Training history
- `val_loss`: Best validation loss
- `metrics`: Final metrics (quality, utilization, etc.)

## Expected Metrics

Production targets after full training:

- **Reconstruction Quality**: >0.90 (MSE-based reconstruction accuracy)
- **Codebook Utilization**: >25% (fraction of codebook vectors used)
- **Inter-Category Orthogonality**: <0.25 (cosine similarity between category centroids)
- **Topographic Correlation**: >0.60 (input-latent distance correlation)

## Troubleshooting

### Poor Reconstruction Quality

- Increase `training.num_epochs` (try 500-600)
- Reduce `training.orthogonality_weight` temporarily
- Check normalization stats
- Verify dataset quality

### Low Codebook Utilization

- Reduce `training.dead_code_threshold` (more aggressive reset, e.g., 8.0)
- Decrease `training.dead_code_reset_interval` (more frequent, e.g., 50)
- Increase `model.commitment_cost` (e.g., 0.35)

### Slow Training

- Enable `training.use_torch_compile: true` (~30-40% speedup, enabled by default)
- Increase `training.val_every_n_epochs` to reduce validation overhead (e.g., 10)
- Use larger `training.batch_size` if GPU memory allows

### Categories Not Sufficiently Orthogonal

- Increase `training.orthogonality_weight` (e.g., 0.15-0.2)
- Extend training (`training.num_epochs: 500-600`)
- Reduce `training.orthogonality_target` to be more strict (e.g., 0.10)

## Design Rationale

The multi-family nested format was adopted to:

1. **Support Multiple Feature Families**: Train on combinations of ARCHITECTURE, SUMMARY, INITIAL, TEMPORAL features
2. **Modular Encoders**: Each family can use specialized encoders (MLP, CNN, etc.)
3. **Clear Organization**: Separate model, training, and logging concerns
4. **Better Validation**: Explicit structure enables better error messages
5. **Future Extensibility**: Easy to add new families and encoder types

## References

- **Architecture**: Categorical Hierarchical VQ-VAE with 3-level quantization
- **Category Discovery**: Hierarchical clustering with silhouette-optimized K
- **Normalization**: Per-category zero-mean, unit-variance normalization
- **Multi-Family Format**: See `docs/vqvae/multi-family-encoders.md`

## Next Steps

After training VQ-VAE:

1. **Analyze Results**: Use analysis scripts to evaluate codebook quality
2. **Encode Dataset**: Convert features to discrete tokens
3. **Train Downstream Models**: Use tokens for neural operator training
4. **Visualize Codebooks**: t-SNE/UMAP visualization of learned categories
