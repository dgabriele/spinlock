# Theta Parameter Tokenization Guide

## Overview

Theta tokens discretely represent the **14-dimensional continuous operator parameters** that define PDE systems in Spinlock. This guide covers architecture, training, and use cases.

## Architecture

### Input: Operator Parameters

PDE operators in Spinlock are parameterized by 14 continuous values in [0,1]:

| Parameter | Description | Range |
|-----------|-------------|-------|
| θ₀-θ₃ | Diffusion coefficients | [0, 1] |
| θ₄-θ₇ | Advection velocities | [0, 1] |
| θ₈-θ₁₀ | Reaction rates | [0, 1] |
| θ₁₁-θ₁₃ | Boundary conditions | [0, 1] |

### ThetaMLPEncoder

**Purpose:** Map continuous parameters to quantization-ready embeddings.

**Architecture:**
```
Input: [B, 14] parameters
  ↓
Linear(14 → 64)
  ↓
LayerNorm(64) → ReLU → Dropout(0.15)
  ↓
Linear(64 → 32)
  ↓
LayerNorm(32)
  ↓
Output: [B, 32] encoded parameters
```

**Configuration:**
```yaml
encoder:
  theta:
    variant: "mlp"
    param_dim: 14
    hidden_dim: 64
    output_dim: 32
    dropout: 0.15
    use_layer_norm: true
```

### Hierarchical Quantization

Theta embeddings are quantized at 3 hierarchical levels:

```
32D Embeddings
  ↓
Projector → [latent_L0, latent_L1, latent_L2]
  ↓
VQ Quantizers:
  - theta_group_1_L0: Coarse discretization
  - theta_group_1_L1: Medium discretization
  - theta_group_1_L2: Fine discretization
  ↓
Output: 3 discrete token indices per sample
```

**Adaptive Codebook Sizing:**
- L0: ~16-32 tokens (coarse operator families)
- L1: ~8-16 tokens (medium-grain variations)
- L2: ~4-8 tokens (fine parameter adjustments)

### ThetaInverseMLP (Roundtrip Decoder)

**Purpose:** Reconstruct original parameters from token embeddings.

**Architecture:**
```
Input: [B, 32] encoded theta
  ↓
Linear(32 → 64)
  ↓
LayerNorm(64) → ReLU → Dropout(0.15)
  ↓
Linear(64 → 14)
  ↓
Sigmoid (clamp to [0,1])
  ↓
Output: [B, 14] reconstructed parameters
```

## Training

### Roundtrip Consistency Loss

```python
# Forward: parameters → tokens
tokens = model.encode(theta_params)  # 3 token indices per sample

# Inverse: tokens → reconstructed parameters
reconstructed = model.forward(theta_params)['decoded']['theta']

# Re-encode: reconstructed → roundtrip tokens
roundtrip_tokens = model.encode(reconstructed)

# Loss: Do roundtrip tokens match original?
loss = mse_loss(roundtrip_latents, target_token_embeddings)
```

### Expected Performance

With roundtrip training (150 epochs):
- **Token match rate**: >90% (same tokens after decode → re-encode)
- **Parameter MSE**: <0.01 (high-fidelity reconstruction)
- **Codebook utilization**: >10% per quantizer

## Use Cases

### 1. CNO-MNO Alignment

**Problem:** Conditional Neural Operators (CNO) train on specific parameter sets, while Meta Neural Operators (MNO) generalize across parameters. How to transfer knowledge?

**Solution:** Theta tokens provide a discrete bridge:

```python
# CNO: trained on 50K parameter sets with theta tokens
cno_tokens = cno_tokenizer.encode(theta_params)  # Discrete tokens

# MNO: trained on broader distribution
mno_latent = mno(initial_conditions)  # Continuous latent

# Alignment layer: map theta tokens → MNO latent space
aligned_latent = alignment_layer(cno_tokens)

# Semantic grounding: MNO latent guided by CNO token structure
```

### 2. Operator Discovery

**Problem:** Searching continuous 14D parameter space is expensive.

**Solution:** Search discrete token space instead:

```python
# Discrete search over token space
for theta_L0 in range(num_codes_L0):
    for theta_L1 in range(num_codes_L1):
        for theta_L2 in range(num_codes_L2):
            # Decode tokens to parameters
            params = inverse_decoder(theta_L0, theta_L1, theta_L2)

            # Evaluate operator performance
            score = evaluate_operator(params)
```

### 3. Transfer Learning

**Problem:** Training from scratch on new operator families is expensive.

**Solution:** Pre-train theta encoder on diverse parameter distributions:

```python
# Pre-train on broad distribution
pretrain_tokenizer(dataset='diverse_operators_100k')

# Fine-tune on specific family
finetune_tokenizer(dataset='target_family_5k', freeze_encoder=True)
```

## Integration Example

```python
from spinlock.tokens import VQTokenizer

# Load tokenizer with theta support
tokenizer = VQTokenizer.from_checkpoint('checkpoints/vqvae_with_theta.pt')

# Tokenize parameters
theta_params = torch.tensor([[0.5, 0.3, ...]])  # [B, 14]
tokens = tokenizer.encode(theta_features=theta_params)

# Output: {'theta_group_1_L0': [batch of token indices],
#          'theta_group_1_L1': [batch of token indices],
#          'theta_group_1_L2': [batch of token indices]}

# Decode tokens back to parameters
reconstructed = tokenizer.forward(theta_features=theta_params)['decoded']['theta']

# Verify roundtrip consistency
roundtrip_tokens = tokenizer.encode(theta_features=reconstructed)
match_rate = (roundtrip_tokens == tokens).float().mean()
print(f"Token match rate: {match_rate:.2%}")  # Should be >90%
```

## Configuration Templates

### Theta-Only Tokenizer (Testing)

```yaml
encoder:
  theta:
    variant: "mlp"
    param_dim: 14
    hidden_dim: 64
    output_dim: 32
    dropout: 0.1

inverse_heads:
  theta_hidden_dim: 64
  theta_dropout: 0.15

loss:
  roundtrip:
    enabled: true
    weight: 5.0
    theta_weight: 1.0
```

### Multi-Family Tokenizer (Production)

```yaml
encoder:
  temporal:
    variant: "pyramid"
    # ... temporal config

  initial:
    variant: "hybrid"
    # ... initial config

  theta:
    variant: "mlp"
    param_dim: 14
    hidden_dim: 64
    output_dim: 32
    dropout: 0.15

inverse_heads:
  theta_hidden_dim: 64
  theta_dropout: 0.15
  initial_base_channels: 256

loss:
  roundtrip:
    enabled: true
    weight: 5.0
    theta_weight: 1.0
    initial_weight: 1.0
```

## Limitations

1. **Fixed dimensionality**: Currently hardcoded to 14 parameters
2. **Single grouping**: All parameters in one semantic group (may not be optimal for all operator families)
3. **Range assumption**: Assumes parameters normalized to [0,1]

## Future Directions

- Adaptive parameter dimensionality
- Multi-group theta encoding for heterogeneous operators
- Hierarchical parameter clustering (like temporal features)
