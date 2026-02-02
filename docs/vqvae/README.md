# VQ-VAE Documentation

Comprehensive guide to Spinlock's Vector-Quantized Variational Autoencoder for behavioral tokenization.

## Core Concepts

- **[Architecture](architecture.md)** - VQ-VAE design, encoding paths, and components
- **[Assignment Strategies](assignment-strategies.md)** - Static vs learnable category assignment
- **[Checkpoint Format](checkpoint-format.md)** - Model saving/loading specification

## Advanced Features

- **[Learnable Assignments](learnable-assignments.md)** - Gradient-based category learning
- **[Learnable Mode Guide](learnable-mode-guide.md)** - Complete usage guide for learnable mode
- **[torch.compile Optimization](torch-compile.md)** - Performance optimization with PyTorch compilation
- **[Variable-Length Encoding](variable-length-encoding.md)** - Temporal pyramid integration

## Architecture Components

- **[Temporal Pyramid](temporal-pyramid.md)** - Multi-scale temporal encoding
- **[Multi-Family Encoders](multi-family-encoders.md)** - Feature family handling

## Quick Start

### 1. Understand the Architecture

Start with [Architecture](architecture.md) to understand:
- How VQ-VAE converts features to discrete tokens
- The three encoding paths (fixed-length, variable-length, hybrid)
- Component responsibilities (encoder, projector, quantizer, decoder)

### 2. Choose an Assignment Strategy

Read [Assignment Strategies](assignment-strategies.md) to decide between:
- **Static** (default): Fast, deterministic, interpretable
- **Learnable**: Adaptive, task-optimal, requires more compute

### 3. Train Your Model

```bash
# Static assignment (default)
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae_variable_length.yaml \
  --epochs 500

# Learnable assignment
poetry run spinlock train-vqvae \
  --config configs/vqvae/learnable_hybrid_variable_length.yaml \
  --epochs 1000
```

### 4. Monitor Training

Watch for:
- Reconstruction loss decreasing (target: <0.05)
- VQ loss stabilizing (target: <0.01)
- Codebook utilization increasing (target: >15%)
- Temperature annealing (learnable mode only)

### 5. Load and Use Checkpoints

See [Checkpoint Format](checkpoint-format.md) for:
- Loading saved models
- Extracting discrete tokens
- Fine-tuning or transfer learning

## Common Workflows

### Workflow 1: Standard Training (Production)

**Goal:** Fast, deterministic, interpretable model

```bash
# Use baseline config with static assignments
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae_variable_length.yaml \
  --epochs 500

# Enable compilation for speed
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae_variable_length.yaml \
  --compile \
  --epochs 500
```

**Docs:**
- [Architecture](architecture.md) - Understanding the model
- [torch.compile Optimization](torch-compile.md) - Speedup details

### Workflow 2: Research & Optimization

**Goal:** Best reconstruction quality, flexible exploration

```bash
# Use learnable assignments for task-optimal categories
poetry run spinlock train-vqvae \
  --config configs/vqvae/learnable_hybrid_variable_length.yaml \
  --epochs 1000
```

**Docs:**
- [Learnable Assignments](learnable-assignments.md) - Implementation details
- [Learnable Mode Guide](learnable-mode-guide.md) - Usage guide
- [Assignment Strategies](assignment-strategies.md) - Comparison with static

### Workflow 3: Temporal Sequence Modeling

**Goal:** Capture multi-scale temporal dynamics

```bash
# Use variable-length encoding with temporal pyramid
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae_variable_length.yaml \
  --epochs 500
```

**Docs:**
- [Variable-Length Encoding](variable-length-encoding.md) - Temporal pyramid details
- [Temporal Pyramid](temporal-pyramid.md) - Multi-scale architecture
- [Architecture](architecture.md#2-variable-length-path-temporal) - Variable-length path

### Workflow 4: End-to-End CNN Learning

**Goal:** Learn optimal initial condition representations

```bash
# Use hybrid encoder with learnable assignments
poetry run spinlock train-vqvae \
  --config configs/vqvae/learnable_hybrid_variable_length.yaml \
  --epochs 1000
```

**Docs:**
- [Architecture](architecture.md#3-hybrid-initial-path-end-to-end-cnn) - Hybrid encoder details
- [Learnable Assignments](learnable-assignments.md) - End-to-end gradient flow

## Troubleshooting

### Low Codebook Utilization (<10%)

**Symptoms:** Many codebook vectors unused, poor diversity

**Solutions:**
1. Enable dead code reset in config
2. Increase codebook size
3. Adjust VQ commitment cost
4. Use learnable assignments with balance loss

**See:** [Architecture](architecture.md#dead-code-reset)

### Poor Reconstruction Quality (>0.1 loss)

**Symptoms:** High reconstruction error, features not recovered

**Solutions:**
1. Increase encoder hidden dimension
2. Add more hierarchical levels
3. Reduce number of categories
4. Try learnable assignments

**See:** [Assignment Strategies](assignment-strategies.md#comparison)

### Slow Training Speed

**Symptoms:** Long epoch times, inefficient GPU usage

**Solutions:**
1. Enable `torch.compile` (30-40% speedup)
2. Increase batch size
3. Use fixed-length encoding (avoid runtime temporal encoding)
4. Check for CPU-GPU data transfer bottlenecks

**See:** [torch.compile Optimization](torch-compile.md)

### Category Collapse (Learnable Mode)

**Symptoms:** All features assigned to 1-2 categories

**Solutions:**
1. Increase `balance_weight` (0.05 → 0.10)
2. Increase `temperature_end` (0.1 → 0.3)
3. Adjust initialization

**See:** [Assignment Strategies](assignment-strategies.md#issue-category-collapse-learnable)

### Compilation Errors

**Symptoms:** Errors during `torch.compile`, crashes

**Solutions:**
1. Disable compilation for variable-length models (limited benefit)
2. Use `mode: "default"` instead of `"reduce-overhead"`
3. Check for dynamic shapes or control flow

**See:** [torch.compile Optimization](torch-compile.md)

## Performance Benchmarks

### Training Speed (50k samples, V100 GPU)

| Configuration | Epoch Time | Speedup |
|---------------|------------|---------|
| Fixed-length (no compile) | 0.82s | Baseline |
| Fixed-length (compiled) | 0.51s | 1.6x |
| Variable-length (no compile) | 1.43s | 0.57x (slower) |
| Variable-length (compiled) | 1.21s | 0.68x (slower) |
| Hybrid + learnable (compiled) | 0.79s | 0.96x |

### Memory Usage (Batch size 256)

| Configuration | GPU Memory |
|---------------|------------|
| Fixed-length | 2.3 GB |
| Variable-length | 4.8 GB |
| Hybrid | 3.2 GB |
| Learnable (additional) | +0.3 GB |

### Reconstruction Quality (Baseline dataset)

| Configuration | Recon Loss | VQ Loss | Utilization |
|---------------|------------|---------|-------------|
| Static + Fixed | 0.0234 | 0.0024 | 18.3% |
| Static + Variable | 0.0198 | 0.0021 | 22.1% |
| Learnable + Variable | 0.0218 | 0.0021 | 23.7% |
| Learnable + Hybrid | 0.0203 | 0.0019 | 25.4% |

**See:** [Architecture](architecture.md#performance-characteristics)

## Configuration Reference

### Minimal Config (Static)

```yaml
families:
  initial:
    encoder: identity
  temporal:
    encoder: identity

training:
  category_assignment: auto
  epochs: 500
  batch_size: 256

categorical_vqvae:
  num_levels: 3
  codebook_size: 512
  codebook_dim: 64
```

### Full Config (Learnable + Hybrid)

```yaml
families:
  initial:
    encoder: initial_hybrid
    encoder_config:
      manual_dim: 14
      cnn_output_dim: 128
  temporal:
    encoder: PyramidTemporalEncoder
    encoder_config:
      hidden_dim: 64
      num_scales: 3

training:
  category_assignment: learnable
  epochs: 1000
  batch_size: 128
  compile:
    enabled: true
    mode: reduce-overhead

learnable_assignment:
  temperature_start: 1.0
  temperature_end: 0.1
  temperature_schedule: linear
  orthogonality_weight: 0.1
  balance_weight: 0.05
  assignment_lr: 0.001

categorical_vqvae:
  num_levels: 3
  codebook_size: 512
  codebook_dim: 64
  commitment_cost: 0.25
```

**See Example Configs:**
- `configs/vqvae/baseline_vqvae_variable_length.yaml`
- `configs/vqvae/learnable_hybrid_variable_length.yaml`

## API Reference

### Training

```python
from spinlock.cli.train_vqvae import main as train_vqvae

# Train with config
train_vqvae(config_path="configs/vqvae/baseline.yaml", epochs=500)
```

### Loading Checkpoints

```python
from spinlock.encoding import load_vqvae_checkpoint

# Load trained model
model, metadata = load_vqvae_checkpoint("checkpoints/vqvae_epoch_500.pt")

# Extract tokens
tokens = model.encode(features)  # [B, K*L] discrete tokens
```

### Inference

```python
# Encode features to tokens
tokens = model.encode(features)

# Reconstruct features from tokens
reconstructed = model.decode(tokens)

# Get quantized representations
quantized, _ = model.quantize(encoded_features)
```

## Further Reading

### Research Papers

- **VQ-VAE:** van den Oord et al., "Neural Discrete Representation Learning" (2017)
- **Gumbel-Softmax:** Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (2016)

### Related Documentation

- [Main README](../../README.md) - Project overview
- [Feature Extraction](../features/README.md) - Pre-processing pipeline
- [Training Guide](../training/README.md) - General training documentation

### Implementation Details

- [Decision Record](../decisions/2026-02-learnable-integration.md) - Learnable integration decisions
- Source code: `src/spinlock/encoding/`

## Contributing

When adding new VQ-VAE features:

1. Update [Architecture](architecture.md) with new components
2. Add configuration examples to this README
3. Document performance characteristics
4. Add troubleshooting tips
5. Update comparison tables

## Support

- **Issues:** [GitHub Issues](https://github.com/danielathyssens/spinlock/issues)
- **Discussions:** [GitHub Discussions](https://github.com/danielathyssens/spinlock/discussions)
- **Documentation:** This guide and linked pages
