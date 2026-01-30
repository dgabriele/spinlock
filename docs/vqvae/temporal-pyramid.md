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

## References

- **Training script**: `src/spinlock/cli/train_vqvae.py`
- **Inference pipeline**: `src/spinlock/encoding/unified_feature_pipeline.py`
- **Example configuration**: `configs/vqvae/50k_3channel.yaml`
- **Unit tests**: `tests/test_pyramid_encoder.py` (if implemented)
