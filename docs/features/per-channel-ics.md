# Per-Channel Independent Initial Conditions

**Version:** 3.2
**Status:** Production Ready ✅
**Date:** 2026-01-30

## Overview

Per-channel independent IC generation allows each of the 3 channels to have different IC types, parameters, and characteristics. This creates **richer behavioral diversity** for VQ-VAE category discovery and enables **compositional reasoning** in NOA training.

## Key Benefits

1. **Maximum Diversity**: Each channel explores different IC families independently
2. **VQ-VAE Category Discovery**: Discovers richer cross-channel interaction categories
3. **NOA Compositional Reasoning**: Learns rules like "when channel 0 is localized AND channel 1 is structured → behavior X"
4. **Better State Space Coverage**: Explores more of the joint IC space

## Quick Start

### Configuration

```yaml
simulation:
  input_generation:
    method: "per_channel"  # Enable per-channel ICs

    channel_configs:
      channel_0:  # Density field - fine-grained features
        ic_type_weights:
          gaussian_random_field: 0.4
          localized: 0.3
          multiscale_grf: 0.3

        gaussian_random_field:
          length_scale: 0.05
          variance: 1.0

        localized:
          num_blobs: 5
          min_width: 3.0
          max_width: 10.0

        multiscale_grf:
          scales: [0.02, 0.04, 0.06]
          variance: 1.0

      channel_1:  # Velocity field - structured patterns
        ic_type_weights:
          structured: 0.5
          gaussian_random_field: 0.5

        structured:
          num_structures: 3
          structure_types: ["vortex", "stripe"]

        gaussian_random_field:
          length_scale: 0.1
          variance: 1.0

      channel_2:  # Auxiliary field - coarse features
        ic_type_weights:
          gaussian_random_field: 0.6
          multiscale_grf: 0.4

        gaussian_random_field:
          length_scale: 0.15
          variance: 0.5

        multiscale_grf:
          scales: [0.20, 0.25, 0.30]
          variance: 0.8
```

### Generate Dataset

```bash
# Test dataset (100 samples)
poetry run spinlock generate \
    --config configs/experiments/test_per_channel_100.yaml \
    --device cuda

# Production dataset (50k samples)
poetry run spinlock generate \
    --config configs/experiments/cno_50k_per_channel.yaml \
    --device cuda
```

### Verify Results

```python
import h5py

with h5py.File('datasets/test_per_channel_100.h5', 'r') as f:
    # Show IC types
    ic_types = f['metadata/ic_types'][:10]
    for i, ic_type in enumerate(ic_types):
        decoded = ic_type.decode('utf-8') if isinstance(ic_type, bytes) else ic_type
        print(f"Sample {i}: {decoded}")

# Output:
# Sample 0: ch0:local|ch1:struct|ch2:mgrf
# Sample 1: ch0:grf|ch1:struct|ch2:grf
# Sample 2: ch0:grf|ch1:grf|ch2:mgrf
# ...
```

## Configuration Reference

### Required Fields

- `method: "per_channel"` - Enable per-channel IC generation
- `channel_configs` - Dict mapping channel names to configurations

### Channel Configuration

Each channel config (`channel_0`, `channel_1`, `channel_2`) requires:

**ic_type_weights** (required):
- Dict mapping IC type names to probability weights
- Weights are automatically normalized to sum to 1.0
- Example: `{gaussian_random_field: 0.5, localized: 0.5}`

**IC-type-specific parameters** (optional):
- Parameters for each IC type listed in `ic_type_weights`
- Each IC type can have its own parameter dict
- Example: `gaussian_random_field: {length_scale: 0.05, variance: 1.0}`

### Available IC Types

| IC Type | Description | Key Parameters |
|---------|-------------|----------------|
| `gaussian_random_field` | Smooth random field | `length_scale`, `variance` |
| `localized` | Sparse Gaussian blobs | `num_blobs`, `min_width`, `max_width` |
| `multiscale_grf` | Multi-scale superposition | `scales`, `variance` |
| `structured` | Geometric patterns | `num_structures`, `structure_types` |
| `composite` | Structured + noise | `base_field`, `perturbation_field`, `mix_ratio` |
| `heavy_tailed` | Power-law spectrum | `alpha`, `variance` |

For domain-specific ICs, see `src/spinlock/dataset/generators.py`.

## Metadata Format

### Single IC Type (Legacy)

```python
ic_types[0] = "gaussian_random_field"
ic_types[1] = "localized"
```

### Per-Channel IC Types (v3.2)

```python
ic_types[0] = "ch0:grf|ch1:struct|ch2:mgrf"
ic_types[1] = "ch0:local|ch1:grf|ch2:grf"
```

**Format:** `ch{i}:{type}|ch{j}:{type}|...`

**Abbreviations:**
- `grf` = gaussian_random_field
- `local` = localized
- `mgrf` = multiscale_grf
- `struct` = structured
- `comp` = composite
- `heavy` = heavy_tailed

## Programmatic Usage

```python
from spinlock.dataset.generators import InputFieldGenerator, PerChannelICGenerator
from spinlock.config.loader import load_config

# Load config
config = load_config("configs/experiments/test_per_channel_100.yaml")

# Create generators
input_gen = InputFieldGenerator(grid_size=64, num_channels=3, device="cuda")
per_channel_gen = PerChannelICGenerator(
    input_generator=input_gen,
    channel_configs=config.simulation.input_generation.channel_configs,
    num_channels=3,
)

# Generate batch
fields, metadata = per_channel_gen.generate_batch(batch_size=100)
# fields: [100, 3, 64, 64]
# metadata: List[Dict[str, str]] with IC type info per channel

print(f"Generated {len(metadata)} samples")
print(f"Sample 0 ICs: {metadata[0]}")
# Output: {'channel_0': 'gaussian_random_field', 'channel_1': 'structured', 'channel_2': 'multiscale_grf'}
```

## VQ-VAE Training

**No changes needed!** VQ-VAE training works unchanged because:

1. VQ-VAE only uses field tensors, not IC metadata
2. Field tensor shape is unchanged: `[N, M, C, H, W]`
3. IC metadata is only for analysis/tracking

```bash
# Train VQ-VAE on per-channel dataset
poetry run spinlock train-vqvae \
    --dataset datasets/cno_50k_per_channel.h5 \
    --config configs/vqvae/baseline.yaml
```

## Design Principles

1. **Clean Separation**: IC selection logic separate from generation logic
2. **DRY**: Reuse existing IC type generators, no duplication
3. **Composability**: Per-channel configs compose naturally
4. **Type Safety**: Strong typing with Pydantic dataclasses
5. **Single Responsibility**: Each class/method does one thing well
6. **Efficient Batching**: Groups samples by IC type for GPU efficiency

## Performance

- **Batching Optimization**: Samples with the same IC type are batched together
- **GPU Efficiency**: Minimizes kernel launches by grouping operations
- **Throughput**: ~1.4 samples/sec on CPU, ~10+ samples/sec on GPU (depends on IC complexity)

## Examples

See `configs/experiments/`:
- `test_per_channel_100.yaml` - Small test dataset (100 samples)
- `cno_50k_per_channel.yaml` - Production dataset (50k samples)

## Testing

Unit tests: `tests/test_per_channel_ics.py`

```bash
poetry run pytest tests/test_per_channel_ics.py -v
```

**Test coverage:**
- ✅ Config validation and normalization
- ✅ IC type sampling distribution
- ✅ Batch generation shapes
- ✅ Channel independence
- ✅ Metadata tracking
- ✅ Edge cases (zero batch, varying params)

All 13 tests passing ✅

## Troubleshooting

### Weights don't sum to 1.0

**Automatic:** Weights are normalized automatically - you can provide any positive numbers:

```yaml
ic_type_weights:
  gaussian_random_field: 2.0  # Will be normalized to 0.5
  localized: 2.0              # Will be normalized to 0.5
```

### Missing channel config

**Error:** `"Missing required config: channel_0"`

**Solution:** Ensure all channels have configs:
```yaml
channel_configs:
  channel_0: { ... }
  channel_1: { ... }
  channel_2: { ... }
```

### IC type parameter mismatch

**Error:** IC generator complains about missing parameters

**Solution:** Ensure IC-type-specific parameters match the IC types in `ic_type_weights`:

```yaml
channel_0:
  ic_type_weights:
    gaussian_random_field: 1.0  # This IC type is used
  gaussian_random_field:         # So provide its parameters
    length_scale: 0.1
    variance: 1.0
```

## References

- **Implementation**: `docs/per_channel_ic_implementation.md`
- **HDF5 Layout**: `docs/features/hdf5-layout.md`
- **Unit Tests**: `tests/test_per_channel_ics.py`
- **Example Configs**: `configs/experiments/`

## Changelog

**v3.2 (2026-01-30):**
- Initial release of per-channel independent IC generation
- Full pipeline integration
- Comprehensive testing (13 unit tests)
- Production-ready with example configurations
