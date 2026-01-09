# Adaptive Compression Ratio System

## Overview

The adaptive compression ratio system automatically computes optimal compression ratios for each category in the hierarchical VQ-VAE based on feature characteristics. This replaces the uniform compression ratios (e.g., `[0.5, 1.0, 1.5]`) with per-category ratios tailored to feature complexity.

## Motivation

Diagnostic analysis revealed severe imbalance in reconstruction quality:
- **cluster_2**: MSE=3611.91 (46,578x worse than baseline)
- **cluster_1**: MSE=0.12 (lowest error)

Uniform compression ratios apply the same bottleneck to all categories regardless of complexity, causing:
1. **Over-compression** of complex categories → high reconstruction error
2. **Under-compression** of simple categories → wasted capacity
3. **Poor codebook utilization** → many unused codes (69% dead codes)

## Implementation

### Core Analysis Functions (`latent_dim_defaults.py`)

#### `analyze_category_characteristics(features)`
Analyzes feature characteristics to determine optimal compression strategy:

```python
def analyze_category_characteristics(features: np.ndarray) -> Dict[str, float]:
    """Returns:
    - variance_score [0-1]: High variance → preserve detail (less compression)
    - dimensionality_score [0-1]: High dim → compress more aggressively
    - information_score [0-1]: PCA concentration → compress if redundant
    - correlation_score [0-1]: High correlation → compress more
    """
```

#### `compute_adaptive_compression_ratios(features, strategy)`
Computes optimal compression ratios using 4 strategies:

**"variance"** - Prioritizes high-variance features (good for TEMPORAL):
```python
if variance_score > 0.7:  # High variance
    return [0.25, 0.75, 2.0]  # More expansion at fine levels
elif variance_score > 0.4:  # Medium
    return [0.4, 1.0, 1.8]
else:  # Low variance
    return [0.5, 1.0, 1.5]  # Standard
```

**"dimensionality"** - Based on feature count:
```python
if dimensionality_score > 0.8:  # Very high dim (>50 features)
    return [0.3, 0.8, 1.5]  # Aggressive bottleneck
elif dimensionality_score > 0.5:  # High dim (20-50)
    return [0.4, 1.0, 1.8]
else:  # Low dim (<20)
    return [0.5, 1.2, 2.0]  # Preserve information
```

**"information"** - Based on redundancy (PCA + correlation):
```python
redundancy = (information_score + correlation_score) / 2.0
if redundancy > 0.7:  # High redundancy
    return [0.3, 0.7, 1.2]  # Aggressive compression
elif redundancy > 0.4:
    return [0.5, 1.0, 1.5]  # Standard
else:  # Low redundancy (complex, orthogonal)
    return [0.6, 1.5, 2.5]  # Preserve complexity
```

**"balanced"** (default, recommended) - Weighted combination:
```python
# Weight strategies by dominant characteristic
weights = {
    'variance': metrics['variance_score'],
    'dimensionality': metrics['dimensionality_score'],
    'information': 1.0 - metrics['information_score'],  # Low info → high weight
}
# Normalize and compute weighted average
ratios = weighted_average([variance_ratios, dim_ratios, info_ratios], weights)
```

### Training Pipeline Integration (`train_vqvae.py`)

The `_precompute_compression_ratios()` method:
1. Checks if `compression_ratios: "auto"` in config
2. Analyzes cleaned features per category
3. Computes optimal ratios using selected strategy
4. Stores results with full metrics for reproducibility
5. Passes per-category ratios to model initialization

Example output:
```
======================================================================
ADAPTIVE COMPRESSION RATIO COMPUTATION
======================================================================
Analyzing 7 categories to compute optimal ratios...

  cluster_1:
    Features: 24
    Variance:      1.000
    Dimensionality: 0.697
    Information:   0.983
    Correlation:   0.793
    → Ratios: [0.31, 0.85, 1.91] (L0=0.31, L1=0.85, L2=1.91)

  cluster_7:
    Features: 31
    Variance:      1.000
    Dimensionality: 0.751
    Information:   0.954
    Correlation:   0.663
    → Ratios: [0.31, 0.85, 1.9] (L0=0.31, L1=0.85, L2=1.9)
======================================================================
```

### Model Support (`categorical_vqvae.py`)

Updated `CategoricalVQVAEConfig.__post_init__()` to handle:
- `compression_ratios: "auto"` → triggers adaptive computation
- `compression_ratios: [0.5, 1.0, 1.5]` → uniform ratios (backward compatible)
- `compression_ratios: {cluster_1: [0.3, 0.8, 1.8], ...}` → per-category (new)

## Usage

### Configuration

Create a config with `compression_ratios: "auto"`:

```yaml
# configs/vqvae/adaptive_compression_example.yaml
model:
  # ADAPTIVE COMPRESSION RATIOS
  compression_ratios: "auto"

  # Strategy for adaptive computation (default: balanced)
  # Options: "balanced", "variance", "dimensionality", "information"
  auto_compression_strategy: "balanced"

  # Standard VQ-VAE parameters
  group_embedding_dim: 256
  group_hidden_dim: 128
  commitment_cost: 0.45
```

### Training

```bash
poetry run spinlock train-vqvae \
    --config configs/vqvae/adaptive_compression_example.yaml \
    --verbose
```

The training pipeline will:
1. Load and clean features
2. Discover categories via clustering
3. **Analyze each category and compute adaptive ratios**
4. Build VQ-VAE with per-category ratios
5. Train normally

### Checkpoint Metadata

Computed ratios are saved with the checkpoint for reproducibility:

```yaml
# In saved config.yaml
model:
  compression_ratios: "auto"
  compression_ratios_strategy: "balanced"

  # Computed ratios (for reproducibility)
  compression_ratios_computed:
    cluster_1: [0.31, 0.85, 1.91]
    cluster_2: [0.32, 0.88, 2.0]
    cluster_3: [0.3, 0.83, 1.91]
    # ...

  # Metadata
  compression_ratios_metrics:
    cluster_1:
      variance_score: 1.000
      dimensionality_score: 0.697
      information_score: 0.983
      correlation_score: 0.793
```

## Architecture Philosophy

### Dynamic Dimension Inference

**Principle:** Never hard-code dimensions. All dimensions must be resolved at runtime.

**Why?** The repo is designed for experimental configurations with:
- Different feature sets across experiments
- Dynamic feature cleaning that removes features
- Multiple dataset formats

**Implementation:**
1. Input dimensions inferred from cleaned features
2. Category dimensions discovered via clustering
3. Latent dimensions computed from compression ratios
4. All projectors/quantizers built dynamically

**Example Flow:**
```
Raw features (270D)
  → Feature cleaning (removes duplicates, outliers)
  → Cleaned features (118D)  [DYNAMIC - varies by experiment]
  → Category discovery
  → 7 categories with varying sizes [DYNAMIC]
  → Adaptive ratios computed per category
  → VQ-VAE built with computed dimensions
```

### DRY Principle

All dimensional computations use shared functions:
- `compute_default_latent_dims()` - Computes latent dims from feature count + ratios
- `compute_adaptive_compression_ratios()` - Analyzes features → optimal ratios
- `fill_missing_num_tokens()` - Computes codebook sizes from latent dims

No dimension should appear in multiple places. All dimensions flow from:
1. **Input**: Cleaned feature count (resolved at runtime)
2. **Ratios**: Compression ratios (uniform or adaptive)
3. **Constraints**: GPU alignment (multiples of 4)

## Expected Impact

Based on baseline diagnostics (10K samples):

### Baseline (Uniform Ratios)
- cluster_2: MSE=3611.91 (worst)
- cluster_1: MSE=0.12 (best)
- Codebook utilization: 20.4%
- Dead codes: 693 (69%)

### Expected with Adaptive Ratios
- **Reduce high-error category reconstruction** by 20-30%
  - cluster_2 gets more capacity (less compression)
- **Maintain low-error category quality**
  - cluster_1 can use more compression (frees capacity)
- **Improve codebook utilization** to 40-60%
  - Better capacity allocation across categories
- **Reduce dead codes** by 20-30%
  - Right-sized codebooks per category

### Validation Results

**Initial test (10K samples, 100 epochs):**
- Training time: 90.4s (1.5 minutes)
- Reconstruction error: 0.1057
- Codebook utilization: 23.92%
- 7 categories with adaptive ratios

Full diagnostics pending...

## Backward Compatibility

The system is fully backward compatible:

**Explicit uniform ratios (existing):**
```yaml
model:
  compression_ratios: "0.5:1.0:1.5"  # Applied to all categories
```

**Explicit per-category ratios (new):**
```yaml
model:
  compression_ratios:
    cluster_1: [0.3, 0.8, 1.8]
    cluster_2: [0.5, 1.2, 2.0]
```

**Adaptive (new):**
```yaml
model:
  compression_ratios: "auto"
  auto_compression_strategy: "balanced"
```

## Future Work

1. **Strategy tuning**: Experiment with different strategies per family type
2. **Per-level strategies**: Allow different strategies for L0/L1/L2
3. **Learned ratios**: Use RL to optimize ratios during training
4. **Cross-validation**: Validate on multiple datasets
5. **Documentation**: Add examples for each strategy

## References

- Plan: `/home/daniel/.claude/plans/elegant-chasing-stallman.md`
- Diagnostics: `scripts/dev/diagnose_vqvae_recon.py`
- Config example: `configs/vqvae/adaptive_compression_example.yaml`
