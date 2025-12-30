# VQ-VAE Training Configurations

This directory contains configuration files for training the Categorical Hierarchical VQ-VAE tokenizer on operator behavioral features.

## Production Training Configuration

The VQ-VAE is trained using a **single-stage production configuration** that has been proven effective in the unisim project for learning high-quality discrete token representations.

**Config**: `production.json`
**Duration**: ~8-12 hours (4K dataset), ~2-3 days (50K dataset)
**Goal**: Learn discrete token representations with high reconstruction quality and codebook utilization

### Key Parameters

**Model Architecture**:
- `embedding_dim: 126` - Fixed embedding size (unisim production default)
- `compression_ratios: "0.5:1:1.5"` - Hierarchical compression across 3 levels
- Auto-computed latent dimensions and token counts based on dataset size

**Training**:
- `epochs: 410` - Proven convergence point from unisim production runs
- `batch_size: 1024` - Large batch for stable training
- `learning_rate: 0.0007` - Conservative LR for quality convergence

**Loss Weights**:
- `commitment_cost: 0.5` - VQ commitment loss
- `orthogonality_weight: 0.1` - Codebook diversity (low emphasis)
- `informativeness_weight: 0.1` - Partial decoder quality
- `topo_weight: 0.45` - Topographic similarity (high emphasis)
- `topo_samples: 1024` - Large sample set for stable topology

**Category Discovery**:
- `category_assignment: "auto"` - Hierarchical clustering on feature correlations
- `num_categories_auto: null` - Auto-determine optimal K via silhouette
- `orthogonality_target: 0.25` - Target inter-category orthogonality
- `max_clusters: 50` - Maximum categories to consider

**Codebook Management**:
- `dead_code_reset_interval: 10000` - Infrequent resets (stable learning)
- `dead_code_threshold: 10.0` - Percentile threshold for dead code detection
- `ema_decay: 0.995` - High momentum for stable codebook updates

**Checkpointing**:
- `checkpoint_use_composite: true` - Composite metric (loss + topo + quality)
- `checkpoint_loss_weight: 1.0`
- `checkpoint_topo_weight: 0.5`
- `checkpoint_quality_weight: 0.5`

**Early Stopping**:
- `early_stopping_patience: 250` - High patience for thorough training
- `early_stopping_min_delta: 0.001` - Tight convergence threshold

## Usage

### Training from Scratch

```bash
# Train VQ-VAE on 4K dataset
poetry run spinlock train-vqvae --config configs/vqvae/production.json

# Train on larger dataset (50K)
poetry run spinlock train-vqvae \
    --config configs/vqvae/production.json \
    --input datasets/benchmark_50k.h5 \
    --output checkpoints/vqvae/production_50k
```

### Resuming Training

```bash
# Resume from checkpoint
poetry run spinlock train-vqvae \
    --config configs/vqvae/production.json \
    --resume-from checkpoints/vqvae/production_4k/best_model.pt
```

### Configuration Overrides

```bash
# Override specific parameters via CLI
poetry run spinlock train-vqvae \
    --config configs/vqvae/production.json \
    --epochs 500 \
    --batch-size 512 \
    --learning-rate 0.001
```

### Dry Run

```bash
# Validate configuration without training
poetry run spinlock train-vqvae \
    --config configs/vqvae/production.json \
    --dry-run \
    --verbose
```

## Custom Configuration

Copy `default.yaml` or `production.json` and customize:

```bash
cp configs/vqvae/production.json configs/vqvae/my_experiment.json
# Edit my_experiment.json with your parameters
poetry run spinlock train-vqvae --config configs/vqvae/my_experiment.json
```

## Expected Outputs

After training, the output directory contains:

```
checkpoints/vqvae/production_4k/
├── best_model.pt              # Best model checkpoint (composite metric)
├── normalization_stats.npz    # Per-category normalization stats
├── training_history.json      # Training metrics history
└── config.yaml               # Resolved configuration (with auto-filled defaults)
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
- **Codebook Utilization**: >0.80 (fraction of codebook vectors used)
- **Inter-Category Orthogonality**: <0.25 (cosine similarity between category centroids)
- **Topographic Correlation**: >0.85 (input-latent distance correlation)

## Troubleshooting

### Poor Reconstruction Quality

- Increase `epochs` (try 500-600)
- Reduce `orthogonality_weight` temporarily
- Check feature normalization stats
- Verify dataset quality

### Low Codebook Utilization

- Reduce `dead_code_threshold` (more aggressive reset, e.g., 8.0)
- Decrease `dead_code_reset_interval` (more frequent, e.g., 5000)
- Increase `commitment_cost` (e.g., 0.6)

### Slow Training

- Enable `use_torch_compile: true` (~30-40% speedup, enabled by default)
- Increase `val_every_n_epochs` to reduce validation overhead (e.g., 10)
- Use larger `batch_size` if GPU memory allows (e.g., 2048)

### Categories Not Sufficiently Orthogonal

- Increase `orthogonality_weight` (e.g., 0.15-0.2)
- Extend training (`epochs: 500-600`)
- Reduce `orthogonality_target` to be more strict (e.g., 0.20)

## Design Rationale

This configuration is based on unisim's production `stage_1_ukr_tokens.json`, which has been validated on 50K+ CA trajectories across multiple token families (U-tokens, K-tokens, R-tokens). Key adaptations for Spinlock:

1. **Removed Token Families**: Spinlock has a single feature family (operator behavioral features), unlike unisim's U/K/R separation
2. **Removed Family Stratification**: No need for multi-family balancing
3. **Simplified Category Assignment**: Auto-discovery only (no manual/hybrid modes initially)
4. **Adapted Feature Loading**: Spinlock uses HDF5 SDF features, not CA trajectory features
5. **Preserved Core Hyperparameters**: Learning rate, loss weights, EMA decay, etc. proven in production

## References

- **Source Config**: `/home/daniel/projects/unisim/config/production/agent_training_v1/stage_1_ukr_tokens.json`
- **Architecture**: Categorical Hierarchical VQ-VAE with 3-level quantization
- **Category Discovery**: Hierarchical clustering with silhouette-optimized K
- **Normalization**: Per-category zero-mean, unit-variance normalization

## Next Steps

After training VQ-VAE:

1. **Encode Dataset**: Convert features to discrete tokens
   ```bash
   poetry run spinlock encode-to-tokens \
       --dataset datasets/test_4k_phase1_phase2.h5 \
       --vqvae-model checkpoints/vqvae/production_4k/best_model.pt \
       --output datasets/test_4k_tokens.h5
   ```

2. **Train ANO (Atomic Neural Operator)**: Map tokens → operator parameters

3. **Evaluate Token Quality**: Discriminability, compression ratio, reconstruction error

4. **Visualize Codebooks**: t-SNE/UMAP visualization of learned categories
