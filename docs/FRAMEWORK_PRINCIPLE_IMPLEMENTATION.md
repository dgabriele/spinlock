# Framework Principle Implementation

## Fundamental Principle

**Spinlock is a FRAMEWORK, not an application.**

All dataset-dependent parameters must be automatically detected at runtime,
never hardcoded in configurations or assumptions.

## Problem

Previously, the VQTokenizer required manual specification of:
- `initial.manual_dim`: Initial feature dimension
- `initial.in_channels`: Number of input channels
- `theta.param_dim`: Parameter dimension

This violated the framework principle because:
1. Users had to manually inspect their dataset to find dimensions
2. Configs were dataset-specific and not portable
3. Easy to make mistakes (wrong dimension → training failure)
4. Framework couldn't adapt to new dataset structures

## Solution: Auto-Detection at Runtime

### 1. Dataset Introspection Module

Created `src/spinlock/tokens/dataset_introspection.py`:

```python
from spinlock.tokens.dataset_introspection import DatasetIntrospector

introspector = DatasetIntrospector("datasets/qbm_50k.h5")
info = introspector.introspect_all()
# Returns:
# {
#     'num_samples': 50000,
#     'initial_manual_dim': 93,
#     'initial_raw_channels': 6,
#     'theta_param_dim': 9,
#     'temporal_feature_dim': 247,
#     'temporal_timesteps': 256,
#     ...
# }
```

### 2. Automatic Config Update

The introspector automatically updates config with detected dimensions:

```python
from spinlock.tokens.dataset_introspection import introspect_and_update_config

config_dict = yaml.safe_load(open('config.yaml'))
config_dict = introspect_and_update_config(config_dict, 'datasets/qbm_50k.h5')
# Config now has correct dimensions from actual dataset
```

### 3. Optional Config Fields

Updated `src/spinlock/tokens/config.py` to make dimensions optional:

```python
class InitialEncoderConfig(BaseModel):
    """Dimensions are automatically detected from dataset if not specified."""
    manual_dim: Optional[int] = Field(default=None, description="Auto-detected if None")
    in_channels: Optional[int] = Field(default=None, description="Auto-detected if None")

class ThetaEncoderConfig(BaseModel):
    """Parameter dimension is automatically detected from dataset."""
    param_dim: Optional[int] = Field(default=None, description="Auto-detected if None")
```

### 4. Integrated into Training Pipeline

Updated `src/spinlock/cli/train_vq_tokenizer.py`:

```python
# Before creating config, introspect dataset
config_dict = introspect_and_update_config(
    config_dict,
    dataset_path,
    verbose=True
)

# Now config has correct dimensions
config = TokenizerConfig(**config_dict)
```

## Adaptive Shape Handling

The introspector intelligently handles various input shapes:

| Input Shape | Interpretation | Result |
|-------------|----------------|--------|
| `[N, H, W]` | Single-channel | 1 channel |
| `[N, C, H, W]` | Multi-channel | C channels |
| `[N, M, H, W]` | Realizations | 1 channel (first realization) |
| `[N, M, C, H, W]` | Realizations + channels | C channels |
| `[N, C, S, H, W]` | **Channels × Species** | **C × S channels** |

Example for QBM:
- Input: `[50000, 3, 2, 64, 64]`
- Detected: 3 channels × 2 species = 6 channels
- Features: 93D (distributional + energy for 6 channels)

## Benefits

### For Users
1. **Zero manual dimension configuration** - Just point to dataset
2. **Portable configs** - Same config works with different datasets
3. **No dimension mismatch errors** - Framework ensures consistency
4. **Clear warnings** - If manually specified dimensions don't match data

### For Developers
1. **Framework adapts automatically** - Works with any dataset structure
2. **DRY principle** - Single source of truth (the dataset itself)
3. **Easier maintenance** - No hardcoded assumptions to update
4. **Better error messages** - Clear when data doesn't match expectations

## Example: QBM Training

### Old Way (Manual Dimensions)
```yaml
encoder:
  initial:
    manual_dim: 93  # Had to manually check dataset
    in_channels: 6  # Had to manually count channels × species
  theta:
    param_dim: 9  # Had to manually check params
```

### New Way (Auto-Detection)
```yaml
encoder:
  initial:
    # manual_dim: auto-detected from /features/initial/aggregated/features
    # in_channels: auto-detected from /inputs/fields shape
  theta:
    # param_dim: auto-detected from /parameters/params shape
```

Training command:
```bash
poetry run spinlock train-vq-tokenizer --config configs/vqvae_qbm_50k.yaml
```

Output:
```
Introspecting dataset structure...
  Raw inputs: (50000, 3, 2, 64, 64) -> 3 channels × 2 species = 6 effective channels
  Initial manual features: (50000, 93) -> 93D
  Parameters: (50000, 9) -> 9D
  Temporal features: (50000, 256, 247) -> 256 timesteps × 247D
Applied dataset-detected dimensions to config
```

## Validation

The introspector validates manually-specified dimensions against data:

```python
introspector = DatasetIntrospector("datasets/qbm_50k.h5")
is_valid, warnings = introspector.validate_config(config_dict)

if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")
    # e.g., "Config specifies manual_dim=27 but dataset has 93D features"
```

## Implementation Files

- **`src/spinlock/tokens/dataset_introspection.py`** - Introspection logic
- **`src/spinlock/tokens/config.py`** - Optional dimension fields
- **`src/spinlock/cli/train_vq_tokenizer.py`** - Integrated auto-detection
- **`src/spinlock/features/initial/extraction_pipeline.py`** - Adaptive extraction
- **`src/spinlock/dataset/pipeline.py`** - Uses introspection-aware extraction

## Future Extensions

### 1. MNO Dataset Generation
Auto-detect CNO feature dimensions for MNO generation.

### 2. Alignment Layer
Auto-detect token dimensions from both tokenizers.

### 3. CLI Info Command
Show detected dimensions without training:
```bash
spinlock info --dataset datasets/qbm_50k.h5 --introspect
```

### 4. Config Templates
Generate starter configs from datasets:
```bash
spinlock generate-config --from-dataset datasets/qbm_50k.h5 --output configs/qbm.yaml
```

## Testing Auto-Detection

```python
from spinlock.tokens.dataset_introspection import DatasetIntrospector

# Test introspection
introspector = DatasetIntrospector("datasets/qbm_50k.h5")
info = introspector.introspect_all()

assert info['initial_manual_dim'] == 93
assert info['initial_raw_channels'] == 6
assert info['theta_param_dim'] == 9
assert info['temporal_feature_dim'] == 247

# Test config update
config_dict = {'encoder': {'initial': {}, 'theta': {}}}
updated = introspector.get_encoder_config_overrides()

assert updated['encoder']['initial']['manual_dim'] == 93
assert updated['encoder']['initial']['in_channels'] == 6
assert updated['encoder']['theta']['param_dim'] == 9
```

## Migration Guide

### Existing Configs
No action required! Auto-detection will override any manually-specified dimensions
and warn if they don't match.

### New Datasets
1. Extract features (initial + temporal)
2. Create minimal config (only algorithmic choices)
3. Run training - dimensions detected automatically

### Minimal Config Template
```yaml
# Only specify algorithmic choices, NOT data dimensions
encoder:
  initial:
    variant: "hybrid"
    cnn_embedding_dim: 384
    pretrained_cnn_path: null
    encode_manual: true
  temporal:
    variant: "pyramid"
    level_dims: [32, 64, 96, 128]
  theta:
    variant: "mlp"
    hidden_dim: 64
    output_dim: 32

quantizer:
  embedding_dim: 64
  use_ema: true

training:
  num_epochs: 1000
  batch_size: 768
  learning_rate: 0.001
```

## Principle Enforcement

To ensure this principle is maintained:

1. **Code Review**: Check that new parameters aren't hardcoded
2. **Testing**: Verify framework works with various dataset structures
3. **Documentation**: Keep FUNDAMENTAL PRINCIPLE visible in memory
4. **Validation**: Introspector warns about manual overrides

Remember: **If it comes from data, detect it at runtime. If it's an algorithmic choice, configure it.**
