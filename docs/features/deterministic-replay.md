# Deterministic Rollout Replay

## Overview

Spinlock datasets support **deterministic replay** of operator rollouts without storing full trajectory data. This enables:

- **Storage efficiency**: ~644 MB for 4K samples (vs. ~1.2 TB with trajectories)
- **Future extensibility**: Extract new feature families on-demand
- **Bit-exact reproduction**: Same rollout every time

## What Makes Replay Deterministic

All sources of randomness are seeded deterministically:

### 1. Operator Weight Initialization

Operator weights are initialized using PyTorch's default initializers (Kaiming for conv layers, etc.). Seeding is based on parameter hash:

```python
# In pipeline.py (automatic during generation)
torch.manual_seed(hash(str(params)) % (2**31))
operator = builder.build_simple_cnn(param_dict)
```

**For replay:**
```python
params = dataset['/parameters/params'][i]
param_dict = map_parameters(params)  # Convert to dict
torch.manual_seed(hash(str(params)) % (2**31))
operator = builder.build_simple_cnn(param_dict)
```

### 2. Stochastic Noise During Rollout

Noise is seeded per operator using the operator index:

```python
# In pipeline.py (automatic)
trajectories = rollout.evolve_operator(
    operator,
    initial_condition,
    num_realizations=M,
    base_seed=op_idx  # Deterministic
)
```

### 3. Initial Conditions

Stored explicitly in HDF5:

```python
ic = dataset['/inputs/fields'][i]  # [C, H, W]
```

## Deterministic Replay Procedure

### Full Replay Example

```python
import torch
import h5py
from spinlock.operators import OperatorBuilder
from spinlock.rollout import OperatorRollout
from spinlock.features.sdf import SDFExtractor

# Load dataset
with h5py.File('dataset.h5', 'r') as f:
    # Select operator to replay
    op_idx = 42

    # Load stored data
    ic = torch.from_numpy(f['/inputs/fields'][op_idx]).cuda()  # [C, H, W]
    params = f['/parameters/params'][op_idx]  # [14]

    # Reconstruct parameter dict
    param_dict = {
        'num_layers': int(params[0]),
        'base_channels': int(params[1]),
        # ... map all 14 parameters
    }

    # Set seed for deterministic initialization
    torch.manual_seed(hash(str(params)) % (2**31))

    # Build operator (deterministic weights)
    builder = OperatorBuilder()
    model = builder.build_simple_cnn(param_dict)
    operator = NeuralOperator(model).cuda()

    # Replay rollout (deterministic noise)
    rollout = OperatorRollout(
        policy_type='convex',  # Or read from metadata
        num_timesteps=250,
        device=torch.device('cuda')
    )

    trajectories, _, _ = rollout.evolve_operator(
        operator,
        ic,
        num_realizations=3,
        base_seed=op_idx  # Same seed as generation
    )

    # Extract new features
    sdf_extractor = SDFExtractor(device=torch.device('cuda'))
    features = sdf_extractor.extract_all(trajectories)

    print(f"Replayed operator {op_idx}")
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"New features extracted: {features['aggregated'].shape}")
```

### Extract New Features for Entire Dataset

```python
def extract_new_features(dataset_path, output_path, new_extractor):
    """Extract new features via deterministic replay."""

    with h5py.File(dataset_path, 'r') as f_in:
        num_samples = f_in['/parameters/params'].shape[0]

        # Initialize feature storage
        with HDF5FeatureWriter(output_path, mode='a') as writer:
            # Create storage for new feature family
            writer.create_feature_group(...)

            # Process in batches for GPU efficiency
            batch_size = 16
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_indices = range(batch_start, batch_end)

                # Replay batch
                batch_trajectories = []
                for i in batch_indices:
                    ic, params = load_data(f_in, i)
                    operator = build_operator_deterministic(params)
                    traj = replay_rollout(operator, ic, base_seed=i)
                    batch_trajectories.append(traj)

                # Extract new features in batch (GPU-optimized)
                batch_traj = torch.stack(batch_trajectories)
                new_features = new_extractor.extract(batch_traj)

                # Write to HDF5
                writer.write_batch(batch_start, new_features)
```

## Verification

### Test Determinism

```python
def test_deterministic_replay():
    """Verify replay produces identical trajectories."""

    # Replay twice
    traj1 = replay_operator(dataset, op_idx=5)
    traj2 = replay_operator(dataset, op_idx=5)

    # Should be bit-exact
    assert torch.all(traj1 == traj2)
    print("✓ Deterministic replay verified")
```

### Common Pitfalls

❌ **Forgot to seed before operator build:**
```python
# Wrong - non-deterministic weights
operator = builder.build_simple_cnn(param_dict)
```

✅ **Correct:**
```python
# Correct - deterministic weights
torch.manual_seed(hash(str(params)) % (2**31))
operator = builder.build_simple_cnn(param_dict)
```

❌ **Different base_seed:**
```python
# Wrong - different noise
rollout.evolve_operator(..., base_seed=random.randint(0, 1000))
```

✅ **Correct:**
```python
# Correct - same seed as generation
rollout.evolve_operator(..., base_seed=op_idx)
```

## Storage Comparison

| Mode | Storage (4K) | Storage (10K) | Replay Cost | Use Case |
|------|-------------|---------------|-------------|----------|
| **Feature-only + Inline** | 644 MB | 1.6 GB | 0s (SDF features) | **Production** |
| **Feature-only + Replay** | 644 MB | 1.6 GB | ~45 min (new features) | Future features |
| **Full trajectories** | 2.4 GB | 6 TB | 0s (all features) | Research/debug only |

## When to Use Each Approach

**Inline Extraction** (implemented):
- ✅ Use for: Standard SDF features (always needed)
- ✅ Benefit: Zero replay cost
- ✅ Overhead: ~2% generation time

**Deterministic Replay** (documented):
- ✅ Use for: Experimental/future feature families
- ✅ Benefit: Minimal storage (~1.6 GB for 10K)
- ✅ Cost: Replay time (once per new feature family)

**Full Trajectory Storage**:
- ❌ Avoid: 6 TB for 10K samples
- ✅ Only use for: Small validation datasets (<1000 samples)

## Implementation Status

✅ **Deterministic seeding**: Enabled (parameter-hash based)
✅ **Inline SDF extraction**: Implemented
✅ **Replay infrastructure**: Available (via visualization system)
📝 **Post-hoc extractor**: Update to use deterministic seeding (TODO)

## Future Enhancements

1. **Cache operator checkpoints**: Optional 500MB storage for instant replay
2. **Parallel replay**: Multi-GPU batch replay for faster extraction
3. **Incremental features**: Add features without full re-extraction
