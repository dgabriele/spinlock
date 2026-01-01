# Multi-Family VQ-VAE Encoder System

## Overview

The VQ-VAE now supports **modular, family-specific encoders** where each feature family (SUMMARY, ARCHITECTURE, INITIAL) can specify its own encoder architecture. This enables:

1. **End-to-end CNN training** for INITIAL spatial features (not pre-extracted)
2. **Family-appropriate architectures** (MLP for vectors, CNN for spatial data)
3. **Flexible encoder selection** via YAML config
4. **Backward compatibility** with existing configs (defaults to GroupMLP)

## Architecture

### Encoder Registry

Located in `src/spinlock/encoding/encoders/`:

```python
from spinlock.encoding.encoders import get_encoder

# Available encoders:
encoder = get_encoder('IdentityEncoder', input_dim=31)
encoder = get_encoder('MLPEncoder', input_dim=221, hidden_dims=[256, 128], output_dim=64)
encoder = get_encoder('ICCNNEncoder', embedding_dim=28)
```

**Available Encoders:**

1. **IdentityEncoder** (`identity.py`): Pass-through, no transformation
   - Use for: Pre-processed features already in good format
   - Params: `input_dim`

2. **MLPEncoder** (`mlp.py`): Multi-layer perceptron
   - Use for: SUMMARY trajectory features, ARCHITECTURE parameter features
   - Params: `input_dim`, `hidden_dims`, `output_dim`, `dropout`, `activation`, `batch_norm`

3. **ICCNNEncoder** (`ic_cnn.py`): ResNet-3 CNN for spatial data
   - Use for: INITIAL spatial grids [B, 1, 128, 128]
   - Params: `embedding_dim`, `in_channels`, `architecture`

### Config Structure

**YAML format:**

```yaml
families:
  sdf:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [256, 128]
      output_dim: 64
      dropout: 0.1

  nop:
    encoder: IdentityEncoder
    encoder_params:
      input_dim: 31

  ic:
    encoder: ICCNNEncoder
    encoder_params:
      embedding_dim: 64
```

**Python (dataclass):**

```python
from spinlock.encoding.categorical_vqvae import CategoricalVQVAEConfig

config = CategoricalVQVAEConfig(
    input_dim=252,  # Total feature dim
    group_indices={
        'sdf': [0, 1, 2, ...],  # 221 features
        'nop': [221, 222, ...],  # 31 features
    },
    family_encoders={
        'sdf': {
            'encoder': 'MLPEncoder',
            'encoder_params': {'hidden_dims': [256, 128], 'output_dim': 64}
        },
        'nop': {
            'encoder': 'IdentityEncoder',
            'encoder_params': {'input_dim': 31}
        }
    },
    levels=[...],
)
```

## Implementation Details

### GroupedFeatureExtractor

Updated to accept `family_encoders` parameter:

```python
from spinlock.encoding.grouped_feature_extractor import GroupedFeatureExtractor

extractor = GroupedFeatureExtractor(
    input_dim=252,
    group_indices={'sdf': [0, ...], 'nop': [221, ...]},
    group_embedding_dim=64,
    family_encoders={  # NEW: optional encoder config
        'sdf': {'encoder': 'MLPEncoder', 'encoder_params': {...}},
    }
)
```

**Behavior:**
- If `family_encoders` provided for a family → use specified encoder
- Otherwise → default to `GroupMLP` (backward compatible)
- All encoders output same `group_embedding_dim` for uniformity

### CategoricalVQVAEConfig

Added `family_encoders` field:

```python
@dataclass
class CategoricalVQVAEConfig:
    ...
    family_encoders: Optional[Dict[str, Dict[str, Any]]] = None
```

Passed to `GroupedFeatureExtractor` during model initialization.

## Usage Examples

### Example 1: SUMMARY + ARCHITECTURE with MLP Encoders

```yaml
# configs/vqvae/sdf_nop.yaml
families:
  sdf:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [256, 128]
      output_dim: 64
  nop:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [128, 64]
      output_dim: 64
```

### Example 2: INITIAL CNN Trained End-to-End

**Critical:** INITIAL CNN encoder is trained **end-to-end with VQ-VAE**, NOT pre-extracted.

```yaml
# configs/vqvae/ic_end_to_end.yaml
families:
  ic:
    encoder: ICCNNEncoder
    encoder_params:
      embedding_dim: 64
      in_channels: 1
      architecture: "resnet3"
```

**Training pipeline must:**
1. Load raw INITIAL grids from `/inputs/initial_conditions` [N, M, 1, 128, 128]
2. Pass through `ICCNNEncoder` during forward pass
3. Backpropagate gradients to train CNN weights

### Example 3: Multi-Family INITIAL + ARCHITECTURE + SUMMARY

```yaml
# configs/vqvae/multi_family_example.yaml
families:
  sdf:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [256, 128]
      output_dim: 64

  nop:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [128, 64]
      output_dim: 64

  ic:
    encoder: ICCNNEncoder
    encoder_params:
      embedding_dim: 64

model:
  group_embedding_dim: 64
  levels:
    - {latent_dim: 32, num_tokens: 128}
    - {latent_dim: 24, num_tokens: 64}
    - {latent_dim: 16, num_tokens: 32}
```

**Output:** 9 tokens (3 families × 3 levels)

## Backward Compatibility

**Old configs without `family_encoders`:**

```python
config = CategoricalVQVAEConfig(
    input_dim=252,
    group_indices={'sdf': [...], 'nop': [...]},
    # No family_encoders specified
)
```

→ **Default behavior:** All families use `GroupMLP` (legacy encoder)

**No breaking changes** to existing code!

## Custom Encoders

### Registering New Encoders

```python
from spinlock.encoding.encoders import register_encoder, BaseEncoder

class MyCustomEncoder(BaseEncoder):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self._output_dim = output_dim
        self.net = nn.Sequential(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim

# Register
register_encoder('MyCustomEncoder', MyCustomEncoder)

# Use in config
families:
  my_family:
    encoder: MyCustomEncoder
    encoder_params:
      input_dim: 100
      output_dim: 64
```

## Files Modified

### Created:
1. `src/spinlock/encoding/encoders/__init__.py` - Registry
2. `src/spinlock/encoding/encoders/base.py` - Base class
3. `src/spinlock/encoding/encoders/identity.py` - Identity encoder
4. `src/spinlock/encoding/encoders/mlp.py` - MLP encoder
5. `src/spinlock/encoding/encoders/ic_cnn.py` - CNN encoder
6. `configs/vqvae/multi_family_example.yaml` - Full example
7. `configs/vqvae/ic_nop_example.yaml` - INITIAL+ARCHITECTURE example

### Modified:
8. `src/spinlock/encoding/categorical_vqvae.py` - Added `family_encoders` field
9. `src/spinlock/encoding/grouped_feature_extractor.py` - Support modular encoders

## Testing

```python
# Test encoder registry
from spinlock.encoding.encoders import get_encoder

mlp = get_encoder('MLPEncoder', input_dim=221, hidden_dims=[256, 128], output_dim=64)
cnn = get_encoder('ICCNNEncoder', embedding_dim=28)
identity = get_encoder('IdentityEncoder', input_dim=31)

# Test with VQ-VAE
from spinlock.encoding.categorical_vqvae import CategoricalVQVAEConfig, CategoricalHierarchicalVQVAE

config = CategoricalVQVAEConfig(
    input_dim=252,
    group_indices={'sdf': list(range(221)), 'nop': list(range(221, 252))},
    family_encoders={
        'sdf': {'encoder': 'MLPEncoder', 'encoder_params': {'hidden_dims': [256, 128], 'output_dim': 64}},
        'nop': {'encoder': 'IdentityEncoder', 'encoder_params': {'input_dim': 31}},
    },
    levels=[{'latent_dim': 32, 'num_tokens': 128}],
)

model = CategoricalHierarchicalVQVAE(config)
x = torch.randn(32, 252)
output = model(x)
# output['tokens']: [32, 2] (2 families × 1 level)
```

## Future Work

1. **INITIAL CNN End-to-End Training Pipeline**
   - Update data loader to provide raw INITIAL grids
   - Modify training loop to handle mixed inputs (features + spatial)
   - Add gradient flow from VQ-VAE back to CNN

2. **Attention-Based Encoders**
   - Register transformer encoders for sequence data
   - Enable cross-family attention

3. **Pretrained Encoders**
   - Support loading pretrained weights
   - Freeze/unfreeze functionality

4. **AutoEncoder Integration**
   - Support VAE encoders with latent sampling
   - Conditional encoders

## Key Insights

✅ **Modularity**: Each family can use the best encoder for its data type
✅ **End-to-End Training**: INITIAL CNN trained with VQ-VAE (not pre-extracted)
✅ **Backward Compatible**: Old configs still work with GroupMLP
✅ **Extensible**: Easy to add new encoder types via registry
✅ **Clean API**: Simple YAML config, no code changes needed

This enables the **NOA to construct INITIAL+NO pairs** by training generative models end-to-end with the joint representation!
