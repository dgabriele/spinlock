# Token-to-Rollout VAE: Enable Sobol Sampling for Rollout Variations

## Overview

The Token-to-Rollout VAE is a standalone Variational Autoencoder that learns the inverse mapping from VQ tokens back to generative parameters. This enables Sobol sampling in latent space to generate diverse rollout variations around a given token embedding.

**Key Capabilities**:
- Decode tokens → (theta parameters, initial condition grids)
- Generate diverse variations via Sobol sampling in latent space
- Uncertainty quantification through ensemble sampling
- Counterfactual analysis ("what if this operator had different ICs?")
- Agent exploration strategies

## Architecture

```
Training Flow:
CNO Dataset → (theta, IC) [ground truth]
Pre-tokenized Dataset → tokens (96 indices)
Frozen VQTokenizer → token embeddings (96 × 64D = 6144D)
TokenToRolloutVAE:
  ├─ Encoder → latent z [512D]
  ├─ ParameterDecoder → theta [14D]
  └─ GridDecoder → grids [3, 64, 64]

Inference Flow (Sobol Sampling):
tokens → embeddings → VAE encoder → z_mean, z_std
  → Sobol samples in [0,1]^512
  → Inverse CDF transform → Gaussian samples
  → z_variations = z_mean + samples * z_std
  → VAE decoder → (theta, IC) variations
  → CNO/MNO simulate → diverse rollouts
```

### Components

1. **TokenEmbeddingExtractor** (Frozen)
   - Loads pre-trained VQTokenizer
   - Extracts embeddings for 96 token groups
   - Output: [B, 6144] flattened embeddings

2. **TokenVAEEncoder**
   - MLP encoder: embeddings → latent distribution (mu, logvar)
   - Hidden dims: [2048, 1024]
   - Output: [B, 512] latent codes

3. **ParameterDecoder**
   - MLP decoder: latent → theta parameters
   - Hidden dims: [256, 128]
   - Output: [B, 14] parameters

4. **GridDecoder**
   - ConvTranspose2d stack: latent → spatial grids
   - Adaptive upsampling: H//16 × W//16 → H × W
   - Output: [B, 3, 64, 64] initial grids

## Usage

### Training

**Option 1: Config file**
```bash
poetry run spinlock train-token-to-rollout-vae \
  --config configs/token_to_rollout_vae.yaml
```

**Option 2: CLI arguments**
```bash
poetry run spinlock train-token-to-rollout-vae \
  --vq-checkpoint checkpoints/vqvae/vq_tokenizer_best.pt \
  --cno-dataset datasets/50k_baseline.h5 \
  --tokenized-dataset datasets/50k_baseline_tokenized.h5 \
  --output-dir checkpoints/token_to_rollout_vae/ \
  --num-epochs 200 \
  --batch-size 256 \
  --learning-rate 1e-3
```

**Option 3: Config file + overrides**
```bash
poetry run spinlock train-token-to-rollout-vae \
  --config configs/token_to_rollout_vae.yaml \
  --num-epochs 100 \
  --device cpu
```

### Validation

```bash
python scripts/validation/validate_token_to_rollout_vae.py \
  --vae-checkpoint checkpoints/token_to_rollout_vae/best.pt \
  --cno-dataset datasets/50k_baseline.h5 \
  --tokenized-dataset datasets/50k_baseline_tokenized.h5 \
  --n-samples 100 \
  --output validation_results.json
```

### Sobol Sampling

```python
from spinlock.tokens.rollout_vae import TokenToRolloutVAE
from spinlock.tokens.sobol_sampler import generate_sobol_variations

# Load model
model = TokenToRolloutVAE.from_checkpoint("checkpoints/token_to_rollout_vae/best.pt")

# Load tokens (from tokenizer or dataset)
tokens = {f"temporal_group_{i}_L0": torch.tensor([idx]) for i in range(96)}

# Generate variations
variations = generate_sobol_variations(
    model,
    tokens,
    n_variations=100,
    device="cuda"
)

# variations["theta"]: [100, 14] parameter variations
# variations["grids"]: [100, 3, 64, 64] IC variations
# variations["z"]: [100, 512] latent codes
```

## Configuration

### Model Configuration

```yaml
model:
  latent_dim: 512  # Latent space dimensionality
  encoder:
    hidden_dims: [2048, 1024]
    dropout: 0.1
  param_decoder:
    hidden_dims: [256, 128]
    dropout: 0.1
  grid_decoder:
    hidden_channels: [512, 256, 128, 64, 32]
    dropout: 0.1
```

### Training Configuration

```yaml
training:
  num_epochs: 200
  batch_size: 256
  learning_rate: 0.001
  beta_schedule: "linear"  # KL annealing
  beta_max: 1.0
  beta_warmup_epochs: 100
  optimizer: "adam"
  scheduler_type: "cosine"
  grad_clip_norm: 1.0
```

### Data Configuration

```yaml
data:
  vq_checkpoint: "checkpoints/vqvae/vq_tokenizer_best.pt"
  cno_dataset: "datasets/50k_baseline.h5"
  tokenized_dataset: "datasets/50k_baseline_tokenized.h5"
  train_split: 0.9
  val_split: 0.1
```

## Loss Function

**VAE Loss** = Reconstruction Loss + β × KL Divergence

- **Theta Reconstruction**: MSE on 14D parameters
- **Grid Reconstruction**: MSE on [3, 64, 64] spatial fields
- **KL Divergence**: KL(q(z|x) || N(0, I))
- **β Annealing**: Ramps 0 → 1.0 over first 100 epochs (prevents posterior collapse)

## Runtime Dimension Resolution

All input/output dimensions are resolved at runtime:
- **theta_dim**: From CNO dataset `/parameters/params` shape
- **grid_shape**: From CNO dataset `/inputs/fields` shape
- **input_dim**: From VQTokenizer quantizers

This eliminates hardcoding and makes the system flexible to different datasets.

## Expected Performance

**Training**:
- Duration: ~4-6 hours for 200 epochs on single GPU
- Dataset: 50K samples, 90/10 train/val split

**Metrics** (rough targets):
- Theta MAE: <0.1 (normalized parameters)
- Grid MSE: <0.05 (normalized grids)
- KL divergence: ~1.0 (well-regularized)
- Sobol discrepancy: <0.01 (good coverage)

## Design Principles

### TODO: Use TODOs for incremental implementation
All code includes TODOs marking key implementation steps:
- Loading datasets
- Model initialization
- Forward passes
- Loss computation
- Checkpoint saving

### DRY (Don't Repeat Yourself)
- Reuses `StratifiedSobolSampler` from existing infrastructure
- Shares Pydantic config pattern with other spinlock modules
- Single source of truth for dimension resolution

### Functional Composition
- Modular components (Encoder, ParameterDecoder, GridDecoder)
- Pure functions for dimension resolution and loss computation
- Clear interfaces via config objects

### Runtime Dimension Resolution
- No hardcoded dimensions
- Inspects checkpoints and datasets at runtime
- Flexible to different dataset sizes and shapes

## Files Created

### Core Implementation (8 files)
1. `src/spinlock/tokens/rollout_vae.py` - VAE architecture
2. `src/spinlock/tokens/rollout_dataset.py` - Dataset loader
3. `src/spinlock/tokens/rollout_vae_config.py` - Pydantic configs
4. `src/spinlock/tokens/rollout_vae_trainer.py` - Training loop
5. `src/spinlock/tokens/sobol_sampler.py` - Sobol sampling utilities
6. `src/spinlock/cli/train_token_to_rollout_vae.py` - CLI command
7. `scripts/validation/validate_token_to_rollout_vae.py` - Validation
8. `configs/token_to_rollout_vae.yaml` - Default config

### Modified Files (1 file)
1. `src/spinlock/cli/__init__.py` - Registered CLI command

## Future Extensions

1. **Hierarchical Latent Space**: Separate latents for theta vs grids
2. **Conditional Sampling**: Sample grids conditioned on theta
3. **Multi-Modal Decoder**: Generate multiple plausible (theta, IC) per token
4. **Diffusion Prior**: Replace Gaussian prior with diffusion model
5. **MNO Integration**: Extend to MNO tokens (train on 100K MNO dataset)

## References

- VAE: Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
- Sobol Sequences: Sobol (1967) "On the distribution of points in a cube"
- VQ-VAE: van den Oord et al. (2017) "Neural Discrete Representation Learning"
