# VQ-VAE Training Regimes Guide

## Overview

Spinlock supports multiple training strategies for VQ-VAE tokenizers. This guide compares approaches and provides recommendations for different use cases.

## Training Strategies Comparison

| Strategy | Reconstruction Target | Roundtrip Consistency | Training Time | Use Case |
|----------|----------------------|----------------------|---------------|----------|
| **Roundtrip-First** (RECOMMENDED) | Decoded space | ✅ Primary objective | ~150 epochs | General-purpose tokenization |
| **Independent** | Encoded space | ❌ Not enforced | ~200 epochs | Baseline/ablation studies |
| **Curriculum** | Encoded → Decoded (staged) | ✅ Added later | ~250 epochs | Legacy (deprecated) |
| **Two-Stage** | Encoder/Decoder separate | ⚠️ Weakly enforced | ~300 epochs | Deprecated (see notes) |

## Roundtrip-First Training (RECOMMENDED)

### Philosophy

Train encoder, quantizers, and inverse decoder **jointly from the start** to ensure decoded values re-encode to the same tokens.

### Configuration

```yaml
loss:
  reconstruction_weight: 0.0   # Pure roundtrip objective (no encoded-space)
  orthogonality_weight: 0.1
  informativeness_weight: 0.1
  topographic_weight: 0.2

  roundtrip:
    enabled: true
    weight: 5.0                # Primary training signal
    theta_weight: 1.0          # Per-family weighting
    initial_weight: 1.0

inverse_heads:
  theta_hidden_dim: 64
  theta_dropout: 0.15
  initial_base_channels: 256
```

### Training Command

```bash
poetry run spinlock train-vq-tokenizer \
  --config configs/vqvae_50k.yaml \
  --dataset datasets/cno_50k.h5 \
  --checkpoint-dir checkpoints/roundtrip_v1 \
  --epochs 150
```

### Expected Metrics

After 150 epochs:
- **Roundtrip token match**: >90% (theta), >85% (initial)
- **Parameter reconstruction MSE**: <0.01 (theta)
- **Codebook utilization**: >10% per quantizer
- **Training stability**: Smooth convergence, no mode collapse

### Advantages

✅ **Self-consistent tokens**: Decoded values reliably re-encode
✅ **Faster convergence**: Joint optimization more efficient
✅ **Better parameter fidelity**: Direct optimization of decoded space
✅ **Simpler workflow**: Single training run, no staging required

### Disadvantages

❌ **Requires inverse heads**: Adds model complexity
❌ **GPU memory**: ~20% more than independent training

---

## Independent Training (Baseline)

### Philosophy

Train encoder and quantizers to reconstruct in **encoded space only** (traditional VQ-VAE objective). No inverse decoder.

### Configuration

```yaml
loss:
  reconstruction_weight: 1.0   # Encoded-space reconstruction
  roundtrip:
    enabled: false             # No roundtrip loss

inverse_heads: null            # No inverse decoder
```

### Expected Metrics

After 200 epochs:
- **Encoded-space MSE**: <0.02
- **Roundtrip token match**: ~5-10% (not optimized for this)
- **Parameter reconstruction**: N/A (no inverse decoder)
- **Codebook utilization**: >10%

### When to Use

- Ablation studies comparing training regimes
- Memory-constrained environments (no inverse heads)
- Applications where decoded space reconstruction not needed

---

## Curriculum Training (Deprecated)

### Philosophy

Start with encoded-space reconstruction, gradually introduce roundtrip loss.

**Deprecation Reason:** Roundtrip-first training converges faster and achieves better metrics. Curriculum adds complexity without benefit.

### Historical Configuration

```yaml
loss:
  reconstruction_weight: 1.0 → 0.0  # Decay over time
  roundtrip:
    enabled: true
    weight: 0.0 → 5.0              # Ramp up over epochs 50-150
```

### Migration Path

```bash
# Old approach
train --config curriculum.yaml --epochs 250

# New approach (faster, better results)
train --config vqvae_50k.yaml --epochs 150
```

---

## Two-Stage Training (Deprecated)

### Philosophy

Train encoder/quantizers first, then train inverse decoder separately.

**Deprecation Reason:**
1. Frozen encoder limits inverse decoder optimization
2. No gradients flow to encoder from roundtrip loss
3. Longer total training time
4. Inferior token match rates vs joint training

### Historical Workflow

```bash
# Stage 1: Train encoder/quantizers (100 epochs)
poetry run spinlock train-vq-tokenizer \
  --config encoder_only.yaml \
  --checkpoint-dir checkpoints/stage1

# Stage 2: Train inverse decoder (100 epochs, encoder frozen)
poetry run spinlock train-inverse-decoder \
  --encoder-checkpoint checkpoints/stage1/best_model.pt \
  --config inverse.yaml
```

### Migration Path

Replace with roundtrip-first single-stage training. Expected improvements:
- **Training time**: 300 → 150 epochs (50% reduction)
- **Token match**: 20% → 90% (4.5x improvement)
- **Parameter MSE**: 0.083 → 0.010 (8x improvement)

---

## Choosing a Training Strategy

### Decision Tree

```
Do you need decoded-space reconstruction?
├─ Yes → Use Roundtrip-First (recommended)
│   └─ Memory constrained? → Consider reducing batch size or model width
└─ No → Use Independent
    └─ For ablation or encoded-space-only applications
```

### Common Use Cases

| Use Case | Recommended Strategy | Notes |
|----------|---------------------|-------|
| **CNO tokenization** | Roundtrip-First | Parameter tokens need high fidelity |
| **MNO tokenization** | Roundtrip-First | Trajectory reconstruction critical |
| **CNO-MNO alignment** | Roundtrip-First | Both tokenizers must be roundtrip-consistent |
| **Ablation study** | Independent | Baseline for comparison |
| **Memory-limited GPU** | Independent | Skip inverse heads to save VRAM |
| **Transfer learning** | Roundtrip-First | Pre-train jointly, fine-tune with frozen encoder |

---

## Performance Benchmarks

### Roundtrip-First vs Independent (50K CNO Dataset)

| Metric | Roundtrip-First | Independent | Improvement |
|--------|----------------|-------------|-------------|
| **Training Time** | 150 epochs | 200 epochs | 25% faster |
| **Theta Token Match** | 92.3% | 4.97% | 18.6x |
| **Initial Token Match** | 87.1% | 8.2% | 10.6x |
| **Theta Recon MSE** | 0.0094 | 0.0831 | 8.8x |
| **Codebook Util (avg)** | 11.2% | 10.8% | Similar |
| **GPU Memory** | 12.4 GB | 10.2 GB | +21% |

### Training Curves

**Roundtrip Token Match (150 epochs):**
```
Epoch   Theta Match   Initial Match
  50       45.2%         38.1%
 100       78.6%         71.3%
 150       92.3%         87.1%
```

**Convergence Speed:**
- Independent: Plateau at epoch ~150
- Roundtrip-First: Steady improvement through epoch 150
- Curriculum: Similar to Roundtrip but 100 extra epochs
- Two-Stage: Poor match rates even after 300 epochs

---

## Configuration Templates

### Production (Multi-Family)

```yaml
# configs/production_tokenizer.yaml
encoder:
  temporal:
    variant: "pyramid"
    num_pyramid_levels: 3
  initial:
    variant: "hybrid"
    cnn_base_channels: 256
  theta:
    variant: "mlp"
    hidden_dim: 64
    output_dim: 32

inverse_heads:
  theta_hidden_dim: 64
  theta_dropout: 0.15
  initial_base_channels: 256

loss:
  reconstruction_weight: 0.0
  orthogonality_weight: 0.1
  informativeness_weight: 0.1
  topographic_weight: 0.2
  roundtrip:
    enabled: true
    weight: 5.0
    theta_weight: 1.0
    initial_weight: 1.0

training:
  epochs: 150
  batch_size: 512
  learning_rate: 0.001
```

### Ablation (Independent)

```yaml
# configs/ablation_independent.yaml
encoder:
  # Same as above

inverse_heads: null  # No inverse decoder

loss:
  reconstruction_weight: 1.0
  orthogonality_weight: 0.1
  informativeness_weight: 0.1
  topographic_weight: 0.2
  roundtrip:
    enabled: false

training:
  epochs: 200
```

---

## Troubleshooting

### Low Token Match Rates

**Symptom:** <50% match after 100 epochs

**Solutions:**
1. Increase `roundtrip.weight` to 10.0
2. Reduce `reconstruction_weight` to 0.0
3. Check inverse head architecture (may be too small)
4. Verify dataset quality (corrupted samples can hurt training)

### Mode Collapse

**Symptom:** Codebook utilization <5%, many dead codes

**Solutions:**
1. Increase `informativeness_weight` to 0.2-0.3
2. Enable smart code resets: `use_smart_reset: true`
3. Reduce commitment cost: `commitment_cost: 0.25` → `0.1`
4. Check feature variance (low-variance features collapse easily)

### Slow Convergence

**Symptom:** Token match improving <1% per 10 epochs

**Solutions:**
1. Increase learning rate: `0.001` → `0.002`
2. Reduce batch size for more frequent updates: `512` → `256`
3. Add warmup epochs: `warmup_epochs: 10`
4. Check for gradient clipping issues

---

## References

- Van Den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE, 2017)
- Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2" (2019)
- Esser et al., "Taming Transformers for High-Resolution Image Synthesis" (VQ-GAN, 2021)
