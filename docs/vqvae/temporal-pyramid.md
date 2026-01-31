# Temporal Pyramid Encoder

**Multi-resolution temporal feature extraction for scale-disentangled VQ-VAE encoding.**

## Overview

The **Temporal Pyramid Encoder** extends the VQ-VAE with multi-resolution temporal analysis, creating a **dual-hierarchy architecture**:

1. **Spatial hierarchy**: 3-level coarse→fine VQ quantization per category (existing)
2. **Temporal hierarchy**: 4-scale pyramid encoder (new: P0→P1→P2→P3)

This enables the model to disentangle fast dynamics (high-frequency fluctuations) from slow trends (global behavioral patterns) into independent representations.

### Why Multi-Resolution?

A single temporal encoding compresses all timescales into one vector, losing the distinction between fast transients and slow trends. The pyramid solves this by processing the same sequence at multiple temporal granularities:

**Analogy:** Stock price charts
- 1-minute resolution: Captures noise, rapid fluctuations
- 1-hour resolution: Captures intraday patterns
- 1-day resolution: Captures trends
- 1-week resolution: Captures strategic movements

Each resolution reveals different structure. The pyramid makes this explicit.

---

## Architecture

### 1. Temporal Downsampling (`TemporalPyramid`)

Input: `[B, T, 345]` per-timestep features

Output: 4 views at different resolutions
- **P0**: `[B, T, 345]` — full resolution (every timestep)
- **P1**: `[B, T/2, 345]` — half resolution (avg-pooled pairs)
- **P2**: `[B, T/4, 345]` — quarter resolution
- **P3**: `[B, T/8, 345]` — eighth resolution

Implementation: `nn.AdaptiveAvgPool1d` along time axis.

### 2. Shared ResNet-1D Backbone

Each pyramid level passes through the **same** ResNet-1D CNN (shared weights):

```
[B, D, T_i] → Conv1d(D→32, k=7, s=2) + MaxPool
           → ResBlock(32→64, s=2)
           → ResBlock(64→128, s=2)
           → ResBlock(128→256, s=2)
           → GlobalAvgPool → [B, 256]
```

**Key insight**: The backbone learns universal temporal pattern detectors. A filter that detects "rapid oscillation" fires strongly on P0 (fine resolution) but weakly on P3 (smoothed out). Same filter, same weights, different responses.

### 3. Per-Level Projection Heads

After the shared backbone produces a 256D vector for each level, separate heads project to different-sized embeddings:

- **P0**: 256D → 32D (fine dynamics, compact)
- **P1**: 256D → 64D (medium patterns)
- **P2**: 256D → 96D (slow dynamics)
- **P3**: 256D → 128D (global trends, largest)

Increasing dimensions reflect that coarser scales capture more structurally important variation.

### 4. Concatenation & Per-Level Families

Final output: `[32 + 64 + 96 + 128] = 320D`, concatenated with 14D initial features → **334D** reconstruction target.

During training, the 320D temporal output is split into 4 feature families:
- `temporal_p0::temporal_p0_0` through `temporal_p0::temporal_p0_31` (32 features)
- `temporal_p1::temporal_p1_0` through `temporal_p1::temporal_p1_63` (64 features)
- `temporal_p2::temporal_p2_0` through `temporal_p2::temporal_p2_95` (96 features)
- `temporal_p3::temporal_p3_0` through `temporal_p3::temporal_p3_127` (128 features)

Each family undergoes **independent clustering** to discover behavioral categories at that temporal scale.

---

## What the Model Learns

### Shared Backbone
Learns temporal pattern detectors (convolutional filters) that respond to:
- Oscillation frequencies
- Decay rates
- Transient events
- Plateau patterns

Because the backbone is shared, these detectors must be general enough to be useful across all temporal scales.

### Per-Level Heads
Learn what aspects of those patterns matter at each scale:
- **P0 head**: Extracts fast dynamics (turbulent fluctuations, high-frequency modes)
- **P1 head**: Extracts medium-scale patterns
- **P2 head**: Extracts slow dynamics
- **P3 head**: Extracts global trends (which attractor, overall trajectory shape)

### Clustering
Discovers that different simulations group differently at different scales:
- A simulation might cluster with group A at P0 (similar fast dynamics)
- But cluster with group B at P3 (different long-term trend)

**Example from training:** P0 found 3 clusters, P3 found 8 clusters. Fine-scale dynamics have fewer distinct patterns than coarse-scale ones.

---

## Integration with VQ-VAE

Each pyramid level becomes a separate feature family for clustering and VQ coding. This means the codebook can assign different discrete tokens to different temporal scales independently.

A simulation gets described by:

```
initial:     token 4   (spatial IC pattern)
temporal_p0: token 7   (fine dynamics type)
temporal_p1: token 2   (medium dynamics type)
temporal_p2: token 11  (slow dynamics type)
temporal_p3: token 3   (global trend type)
```

This is richer than a single 128D temporal embedding — it disentangles temporal structure across scales rather than compressing everything into one vector.

---

## Configuration

### Basic Usage

```yaml
# configs/vqvae/pyramid_example.yaml
families:
  initial:
    encoder: initial_hybrid
    encoder_params:
      manual_dim: 14
      cnn_embedding_dim: 128
      encode_manual: false
      in_channels: 3

  temporal:
    encoder: PyramidTemporalEncoder
    encoder_params:
      level_dims: [32, 64, 96, 128]       # Per-level output dimensions
      downsample_factors: [1, 2, 4, 8]    # Temporal downsampling
      architecture: "resnet1d_3"            # Shared backbone

training:
  category_assignment_config:
    per_family_clustering: true
    per_family_params:
      initial:
        min_clusters: 2
        max_clusters: 5
      temporal_p0:
        min_clusters: 2
        max_clusters: 10
      temporal_p1:
        min_clusters: 2
        max_clusters: 10
      temporal_p2:
        min_clusters: 2
        max_clusters: 15
      temporal_p3:
        min_clusters: 2
        max_clusters: 20
```

### Customization

**Different resolutions:**
```yaml
downsample_factors: [1, 3, 9, 27]  # Powers of 3
level_dims: [16, 32, 64, 128]       # Different size progression
```

**Fewer levels:**
```yaml
downsample_factors: [1, 4]  # Just two scales
level_dims: [64, 128]
```

---

## Training Results

**Example training run (experimental, 1000 epochs):**

| Metric | Value |
|--------|-------|
| **Quality** | 96.93% (L_recon=0.0307) |
| **Categories** | 23 (4 initial + 3 P0 + 4 P1 + 4 P2 + 8 P3) |
| **Features** | 204D after cleaning |
| **Utilization** | 17.7% |
| **Topology** | pre=0.971, post=1.000 |

**Per-level category distribution:**
- P0 (fine): 3 categories → simpler fast dynamics
- P1 (medium): 4 categories
- P2 (slow): 4 categories
- P3 (coarse): 8 categories → more complex slow trends

This reflects the intrinsic complexity at each temporal scale.

---

## Implementation Details

### Code Location
- **Pyramid downsampling**: `src/spinlock/encoding/temporal_pyramid.py`
- **Encoder**: `src/spinlock/encoding/encoders/pyramid_temporal.py`
- **Registry**: `src/spinlock/encoding/encoders/__init__.py`

### Key Attributes
The `PyramidTemporalEncoder` exposes:
- `output_dims_per_level: List[int]` — Used by training pipeline to detect pyramid and split features
- `output_dim: int` — Total dimension (sum of level_dims)
- `forward_per_level() -> List[Tensor]` — Returns per-level embeddings separately

### Training Pipeline
During feature loading (`train_vqvae.py`):
1. Detect pyramid via `hasattr(encoder, 'output_dims_per_level')`
2. Split concatenated 320D output into 4 arrays: [32D, 64D, 96D, 128D]
3. Name each level's features with `temporal_p{i}::` prefix
4. Existing clustering code treats each as independent family

---

## Comparison with Single-Scale

| Aspect | Single-Scale (TemporalCNNEncoder) | Multi-Scale (PyramidTemporalEncoder) |
|--------|----------------------------------|--------------------------------------|
| **Output** | 128D single vector | 320D (32+64+96+128) across 4 scales |
| **Timescales** | Compressed together | Disentangled into P0-P3 |
| **Categories** | Single temporal family | 4 independent temporal families |
| **Clustering** | All features cluster together | Per-scale clustering |
| **Expressivity** | Lower (fewer token positions) | Higher (more compositional structure) |
| **Training time** | Faster (simpler architecture) | Similar (shared backbone) |

**When to use pyramid:**
- Dynamics have distinct fast/slow components
- Want independent control over multi-scale behavior
- Need richer token vocabulary for complex systems

**When to use single-scale:**
- Simple dynamics with single characteristic timescale
- Prioritize simplicity over expressivity
- Smaller model footprint desired

---

---

## Variable-Length Trajectory Support

**NEW (2026-01):** The pyramid encoder now supports **variable-length temporal sequences** for meta-learning and operator discovery.

### Motivation

Different operators have intrinsic timescales:
- **Fast equilibration**: T=16-32 timesteps sufficient
- **Slow dynamics**: T=256+ timesteps needed

Training on **mixed-length trajectories** helps the model learn **scale-invariant representations** where dynamics are recognized regardless of temporal resolution.

### How It Works

**1. Length Sampling (per batch)**

Each sample gets ONE randomly sampled length when loaded:

```python
# During training, sample 42 might be seen as:
Epoch 1, Batch 5:  T=64   (random from bins [16,32,64,128,256])
Epoch 1, Batch 18: T=128  (different random choice)
Epoch 2, Batch 3:  T=32   (different again)
```

**2. Masking**

Create validity mask for each sample:
```python
Sample with T=64:
  features: [500, D]  # Full padded trajectory
  mask:     [True×64, False×436]  # First 64 valid
  length:   64
```

**3. Adaptive Pyramid Levels**

Pyramid automatically adjusts levels based on trajectory length:

```python
# T=256: All levels valid
[1×] → T=256
[2×] → T=128
[4×] → T=64
[8×] → T=32  ✓ All valid

# T=16: Some levels skipped
[1×] → T=16
[2×] → T=8
[4×] → T=4
[8×] → T=2   ✓ Still valid (min_pyramid_length=1)

# T=4: Adaptive skipping
[1×] → T=4
[2×] → T=2
[4×] → T=1
[8×] → T=0.5  ✗ SKIP (< min_pyramid_length)
# Only uses [1×, 2×, 4×] levels
```

**4. Mask Propagation**

Masks downsample through pyramid using "ceil" (conservative):
```python
Original mask: [True×64, False×436] at T=500
Level 0 (1×):  [True×64, False×436]  # Full resolution
Level 1 (2×):  [True×32, False×218]  # Pooled pairs
Level 2 (4×):  [True×16, False×109]  # Pooled groups of 4
Level 3 (8×):  [True×8,  False×54]   # Pooled groups of 8
```

"Ceil" method: position valid if **ANY** source timestep valid (conservative).

**5. Length-Invariant Encoding**

Global average pooling in backbone makes encoding **length-invariant**:

```python
# Short trajectory (T=16)
[B, 16, 64] → Backbone → GAP → [B, 256]

# Long trajectory (T=256)
[B, 256, 64] → Backbone → GAP → [B, 256]

# Same 256D embedding dimension!
```

**6. Zero-Padding for Missing Levels**

If adaptive pyramid skips levels, output is zero-padded:

```python
# All levels active (T=256):
embeddings = concat([32D, 64D, 96D, 128D]) = 320D

# Only 3 levels active (T=4, skips 8×):
embeddings = concat([32D, 64D, 96D, 0×128D]) = 320D (padded)
```

**7. Sample Weighting in Loss**

Reconstruction loss weighted by valid fraction:

```python
loss = MSE(reconstruction, target) * (num_valid / max_valid)

# Examples:
Sample A (T=64):  weight = 64/256  = 0.25
Sample B (T=256): weight = 256/256 = 1.00
```

Prevents short trajectories from dominating gradient updates!

### Configuration

```yaml
families:
  temporal:
    encoder: PyramidTemporalEncoder
    encoder_params:
      level_dims: [32, 64, 96, 128]
      downsample_factors: [1, 2, 4, 8]
      architecture: "resnet1d_3"

      # Variable-length support
      variable_length:
        enabled: true
        min_timesteps: 16          # Powers of 2
        max_timesteps: 256
        sampling_strategy: "fixed_bins"
        length_bins: [16, 32, 64, 128, 256]  # Aligns with pyramid
        adaptive_pyramid: true     # Auto-skip invalid levels
        mask_downsample_method: "ceil"  # Conservative
```

### Training Strategy

**Powers of 2 for clean alignment:**

Length bins `[16, 32, 64, 128, 256]` divide cleanly by factors `[1, 2, 4, 8]`:
- T=16: levels produce [16, 8, 4, 2]
- T=32: levels produce [32, 16, 8, 4]
- T=64: levels produce [64, 32, 16, 8]
- T=128: levels produce [128, 64, 32, 16]
- T=256: levels produce [256, 128, 64, 32]

No rounding, integer boundaries at every level!

**Epoch requirements:**

With 5 bins and uniform sampling:
- After 5 epochs: ~63% samples seen at all lengths
- After 10 epochs: ~89% coverage
- After 20 epochs: ~99% coverage

**Recommended:** 100-150 epochs for good multi-scale learning.

### What the Model Learns

With variable-length training:

1. **Scale-invariant pattern recognition**
   - Same dynamics recognized at T=16 or T=256
   - Filters learn temporal patterns, not absolute timescales

2. **Robust multi-resolution encoding**
   - Reconstruction quality uniform across lengths
   - No length-specific shortcuts or memorization

3. **Adaptive temporal processing**
   - Short trajectories use fewer pyramid levels
   - Long trajectories use full pyramid hierarchy
   - Output dimension always consistent (320D)

### Implementation

- **Length sampling**: `src/spinlock/encoding/trajectory_length_sampler.py`
- **Adaptive pyramid**: `src/spinlock/encoding/temporal_pyramid.py`
- **Variable-length encoder**: `src/spinlock/encoding/encoders/pyramid_temporal.py`
- **Masked loss**: `src/spinlock/encoding/training/losses.py`
- **Integration**: `src/spinlock/cli/train_vqvae.py`
- **Tests**: `tests/encoding/test_variable_length_*.py`

---

## References

- **Training script**: `src/spinlock/cli/train_vqvae.py`
- **Inference pipeline**: `src/spinlock/encoding/unified_feature_pipeline.py`
- **Example configurations**:
  - `configs/vqvae/baseline_vqvae.yaml` (standard)
  - `configs/vqvae/baseline_vqvae_variable_length.yaml` (variable-length enabled)
- **Unit tests**:
  - `tests/encoding/test_trajectory_length_sampler.py`
  - `tests/encoding/test_pyramid_adaptive.py`
  - `tests/encoding/test_variable_length_integration.py`
