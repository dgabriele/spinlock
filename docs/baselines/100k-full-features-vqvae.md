# VQ-VAE Baseline: 100K Full Features (INITIAL + SUMMARY + TEMPORAL + ARCHITECTURE)

**Date:** January 4, 2026
**Dataset:** `datasets/100k_full_features.h5`
**Checkpoint:** `checkpoints/production/100k_with_initial/`
**Status:** PRODUCTION READY

---

## Executive Summary

Production VQ-VAE tokenizer trained on 100,000 neural operator samples with joint encoding of **four feature families**, including hybrid INITIAL features with end-to-end CNN training:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Val Loss** | 0.1644 | <0.20 | **EXCEEDED** |
| **Quality** | 0.9554 | >0.85 | **EXCEEDED** |
| **Codebook Utilization** | 93.7% | >25% | **EXCEEDED** |
| **Reconstruction Error** | 0.0446 | - | Excellent |
| **Categories Discovered** | 7 | auto | Data-driven |
| **Total Parameters** | 3.2M | - | - |

**Improvement over previous baseline:** 13% better val_loss (0.164 vs 0.189)

---

## Dataset Configuration

### Feature Families

| Family | Raw Dimensions | Encoder | Output Dimensions |
|--------|----------------|---------|-------------------|
| **INITIAL** | 14 (manual) + CNN | InitialHybridEncoder | 14 + 28 = 42 |
| **SUMMARY** | 360 | MLPEncoder [512, 256] | 128 |
| **TEMPORAL** | 256 × 63 | TemporalCNNEncoder (ResNet-1D) | 128 |
| **ARCHITECTURE** | 12 | IdentityEncoder | 12 |
| **Total** | - | - | **282** (pre-cleaning) |

**INITIAL Hybrid Encoder:** 14D manual features (spatial stats, spectral, entropy, morphological) passed through directly + 28D CNN embeddings trained end-to-end with the VQ-VAE via backpropagation.

After feature cleaning (variance filtering, deduplication, outlier capping): **175 features**

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 100,000 operators |
| Realizations per Sample | 5 |
| Grid Size | 64×64 |
| Timesteps | 256 |
| Dataset Size | ~10 GB |

### HDF5 Structure

```
datasets/100k_full_features.h5
├── features/
│   ├── summary/
│   │   ├── aggregated/features    [100000, 360]
│   │   └── per_trajectory/features [100000, 5, 120]
│   ├── temporal/features          [100000, 256, 63]
│   └── architecture/
│       └── aggregated/features    [100000, 12]
├── inputs/fields                  [100000, 3, 64, 64]
├── parameters/params              [100000, 12]
└── metadata/
    ├── ic_types                   [100000]
    ├── evolution_policies         [100000]
    ├── grid_sizes                 [100000]
    └── noise_regimes              [100000]
```

---

## VQ-VAE Architecture

### Model Configuration

```yaml
families:
  initial:
    encoder: initial_hybrid
    encoder_params:
      manual_dim: 14          # Manual features passed through directly
      cnn_embedding_dim: 28   # CNN trained end-to-end
      encode_manual: false
      in_channels: 1

  summary:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [512, 256]
      output_dim: 128
      dropout: 0.1
      activation: "relu"
      batch_norm: true

  temporal:
    encoder: TemporalCNNEncoder
    encoder_params:
      embedding_dim: 128
      architecture: "resnet1d_3"

  architecture:
    encoder: IdentityEncoder
    encoder_params: {}
    # Parameters already normalized to [0, 1]

model:
  group_embedding_dim: 256
  group_hidden_dim: 512
  levels: []  # auto-computed
  commitment_cost: 0.25
  use_ema: true
  decay: 0.99
```

### Training Configuration

```yaml
training:
  batch_size: 1024           # Increased for better GPU utilization
  learning_rate: 0.001
  num_epochs: 300            # Extended for INITIAL feature training
  optimizer: "adam"

  # Category discovery (pure clustering - no gradient refinement)
  category_assignment: "auto"
  category_assignment_config:
    method: "clustering"     # Pure clustering (faster, better reconstruction)
  orthogonality_target: 0.15
  min_features_per_category: 2  # Allow smaller clusters for granularity

  # Loss weights
  reconstruction_weight: 1.0
  vq_weight: 1.0
  orthogonality_weight: 0.1
  informativeness_weight: 0.1
  topo_weight: 0.3
  topo_samples: 1024         # Increased for more topographic signal

  # Callbacks
  early_stopping_patience: 30
  dead_code_reset_interval: 100  # Less frequent for better convergence
  val_every_n_epochs: 5
  use_torch_compile: true
```

---

## Training Results

### Convergence

| Epoch | Quality | Utilization | Val Loss | Train Loss |
|-------|---------|-------------|----------|------------|
| 1 | 0.842 | 23.3% | 0.411 | 2.964 |
| 50 | 0.908 | 24.8% | 0.210 | 0.257 |
| 100 | 0.924 | 31.8% | 0.200 | 0.222 |
| 150 | 0.942 | 59.7% | 0.182 | 0.199 |
| 200 | 0.946 | 71.4% | 0.176 | 0.191 |
| 250 | 0.952 | 73.6% | 0.168 | 0.185 |
| 300 | 0.957 | 72.9% | **0.164** | 0.180 |

### Final Metrics

```
Final Metrics:
  val_loss: 0.164376
  utilization: 0.9368
  reconstruction_error: 0.0446
  quality: 0.9554
```

### Per-Cluster Performance

| Cluster | Reconstruction MSE | Utilization (L0/L1/L2) |
|---------|-------------------|------------------------|
| cluster_1 | 0.3145 | 0.93 / 0.93 / 1.00 |
| cluster_2 | 0.2274 | 1.00 / 0.92 / 0.83 |
| cluster_3 | 0.2311 | 0.89 / 0.89 / 1.00 |
| cluster_4 | 0.3544 | 0.82 / 1.00 / 1.00 |
| cluster_5 | 0.2797 | 0.97 / 0.94 / 1.00 |
| cluster_6 | 0.8247 | 0.83 / 1.00 / 0.83 |
| cluster_7 | 0.3153 | 0.94 / 0.94 / 1.00 |

**Note on metrics:** The global `reconstruction_error` (0.045) uses the **shared decoder** that combines all 21 codebooks (7 categories × 3 levels) to reconstruct the full feature vector. The per-cluster **Reconstruction MSE** (0.23-0.82) measures how well each category's codes alone can reconstruct its own features—inherently a harder task used for informativeness regularization.

### Training Time

- **Total Duration:** ~27 minutes (300 epochs)
- **Per-Epoch:** ~5.2 seconds (after torch.compile warmup)
- **torch.compile Warmup:** ~90 seconds (epoch 1)
- **Hardware:** Single GPU with TF32 matmul

---

## Hierarchical Codebook Architecture

### Auto-Scaling Codebook Sizes

The VQ-VAE uses **compression ratio-based auto-scaling** to determine codebook sizes at each level of the hierarchy:

| Level | Compression Ratio | Purpose | Typical Size |
|-------|------------------|---------|--------------|
| L0 (Coarse) | 0.5 | Broad behavioral categories | 15-32 codes |
| L1 (Medium) | 1.0 | Sub-category distinctions | 8-17 codes |
| L2 (Fine) | 1.5 | Specific behavioral variants | 4-8 codes |

**Design rationale:** Fine-grained distinctions are inherently sparse. If 80% of operators fall into a few broad behavioral regimes, only rare edge cases exhibit truly unique fine-scale dynamics. The compression ratio hierarchy reflects this information-theoretic insight:
- **Coarse codes** must tile a large behavioral space → need more codes
- **Fine codes** are refinements within already-narrow regions → fewer meaningful distinctions exist

### Dead Code Reset Mechanism

During training, the system monitors codebook utilization and performs **dead code resets** at configurable intervals (every 100 epochs in this configuration):

1. **Detection:** Codes with EMA cluster size below the 10th percentile are flagged as "dead"
2. **Reset:** Dead codes are re-initialized to perturbed versions of high-usage codes
3. **Pruning effect:** Over training, codebooks naturally stabilize to their "right size"
4. **Reset limit:** Maximum 25% of codes can be reset per interval to prevent disruption

**Interpretation:** If a Level 2 codebook stabilizes at 6 codes with 83% utilization (5 active), this is the system *discovering* the natural capacity for fine-grained distinctions in that category—not a limitation but empirical evidence of inherent sparsity.

### Per-Level Utilization Analysis

From the production model:

| Level | Mean Utilization | Interpretation |
|-------|-----------------|----------------|
| L0 | 91% | Excellent coverage of broad categories |
| L1 | 95% | Healthy mid-level distinctions |
| L2 | 95% | Fine codes well-utilized |

**Key insight:** The combination of auto-scaling + dead code resets + pure clustering ensures codebooks are neither too large (wasted capacity) nor too small (forced collapse). The 90%+ utilization across all levels indicates the architecture found appropriate vocabulary sizes for the data.

### Category Discovery: Pure Clustering

This model uses **pure hierarchical clustering** (no gradient refinement) to discover 7 categories from 175 features. Key benefits:
- **Better reconstruction**: Gradient refinement optimizes for orthogonality at the expense of reconstruction quality
- **Natural category sizes**: Clustering respects the inherent structure of feature correlations
- **Faster training**: No additional optimization loop for category assignments

**Configuration:**
- `method: "clustering"` (pure agglomerative clustering)
- `min_features_per_category: 2` (allow smaller, more granular categories)
- `max_clusters: 25` (upper bound, actual discovered: 7)

### Category Composition

The 7 discovered categories integrate features across all four families:

| Category | Features | Primary Content |
|----------|----------|-----------------|
| cluster_1 | 23 | Mixed SUMMARY + TEMPORAL |
| cluster_2 | 15 | TEMPORAL dynamics |
| cluster_3 | 41 | SUMMARY statistics |
| cluster_4 | 16 | ARCHITECTURE + INITIAL |
| cluster_5 | 30 | SUMMARY spatial |
| cluster_6 | 15 | INITIAL + SUMMARY |
| cluster_7 | 35 | TEMPORAL + SUMMARY |

**Key observation:** With 4 feature families (INITIAL+SUMMARY+TEMPORAL+ARCHITECTURE), the clustering discovers categories that meaningfully integrate across modalities rather than strictly separating by family. This reflects the learned correlations between initial conditions, operator parameters, and emergent behaviors.

**Interpretation:**
- **INITIAL features** (14D manual + 28D CNN) cluster with ARCHITECTURE and SUMMARY features, capturing how initial conditions influence behavioral outcomes
- **TEMPORAL features** (FFT, autocorrelation, periodicity) form dedicated clusters (C2, C7) and also integrate with SUMMARY
- **SUMMARY features** distribute across most categories, serving as the "glue" that captures aggregate behavioral signatures
- **ARCHITECTURE features** cluster with INITIAL, reflecting the operator-level determinism of parameter→behavior mapping

---

## Feature Cleaning Summary

The VQ-VAE training pipeline applies automatic feature cleaning:

1. **INITIAL Protection:** 14 manual INITIAL features are protected from cleaning (always retained)

2. **NaN Replacement:**
   - SUMMARY: 3,000,000 NaN → 0 (operator_sensitivity features disabled in extraction)
   - TEMPORAL: 0 NaN (fixed at source in spatial.py)

3. **Variance Filtering:** Remove features with variance < 1e-8

4. **Deduplication:** Remove features with correlation > 0.99

5. **Outlier Capping:** MAD-based clipping at 5σ

**Result:** 282 features → 175 features after cleaning

---

## Usage

### Load Trained Model

```python
import torch
import yaml
from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig

# Load configuration
with open("checkpoints/production/100k_full_features/config.yaml") as f:
    config_dict = yaml.safe_load(f)

# Build model
config = CategoricalVQVAEConfig(**config_dict["model"])
model = CategoricalHierarchicalVQVAE(config)

# Load weights
checkpoint = torch.load("checkpoints/production/100k_full_features/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Extract Tokens

```python
import h5py
import numpy as np

# Load features
with h5py.File("datasets/100k_full_features.h5", "r") as f:
    summary = f["features/summary/aggregated/features"][:]
    temporal = f["features/temporal/features"][:]
    architecture = f["features/architecture/aggregated/features"][:]

# Apply encoders and get tokens
# (See spinlock.cli.train_vqvae for full pipeline)
```

### Retrain with Different Configuration

```bash
poetry run spinlock train-vqvae \
    --config configs/vqvae/production/100k_full_features.yaml \
    --verbose
```

---

## Files

| File | Description |
|------|-------------|
| `configs/vqvae/production/100k_full_features.yaml` | Training configuration |
| `datasets/100k_full_features.h5` | Dataset with all features |
| `checkpoints/production/100k_with_initial/best_model.pt` | Best model weights (val_loss: 0.164) |
| `checkpoints/production/100k_with_initial/final_model.pt` | Final model weights (epoch 300) |
| `checkpoints/production/100k_with_initial/config.yaml` | Saved model config |
| `checkpoints/production/100k_with_initial/normalization_stats.npz` | Feature normalization |
| `checkpoints/production/100k_with_initial/training_history.json` | Full training metrics |

---

## Known Issues and Fixes

### SUMMARY NaN (Expected)

30 features (operator_sensitivity) are NaN because `extract_operator_features: false` in dataset generation config. These are automatically replaced with 0 during training.

### TEMPORAL NaN (Fixed)

Skewness/kurtosis features were NaN at t=0 for structured initial conditions (symmetric distributions). **Fixed in `src/spinlock/features/summary/spatial.py`** by adding `torch.nan_to_num()` to handle undefined moments for symmetric distributions.

---

## Comparison to Previous Baselines

| Metric | 10K Baseline | 100K (3-family) | **100K (4-family)** |
|--------|--------------|-----------------|---------------------|
| Samples | 10,000 | 100,000 | 100,000 |
| Feature Families | SUMMARY only | SUM+TEM+ARCH | **INIT+SUM+TEM+ARCH** |
| Raw Features | 46 | 268 | **282** |
| Cleaned Features | ~40 | 147 | **175** |
| Categories | ~6-8 | 11 | **7** |
| Val Loss | - | 0.189 | **0.164** |
| Quality | ~0.85 | 0.9475 | **0.9554** |
| Utilization | ~30% | 93.7% | **93.7%** |

**Key improvements in 4-family model:**
- **13% better val_loss** (0.164 vs 0.189) through pure clustering + INITIAL features
- **INITIAL CNN trained end-to-end** - learns spatial patterns that correlate with behavioral outcomes
- **Fewer, more meaningful categories** (7 vs 11) - pure clustering produces more natural groupings
- **Better reconstruction** (0.045 vs 0.053) - less information loss in tokenization

---

## Visualization Dashboards

Three visualization dashboards are available for analyzing trained VQ-VAE models:

```bash
# Generate all dashboards
poetry run spinlock visualize-vqvae \
    --checkpoint checkpoints/production/100k_with_initial/ \
    --output visualizations/ \
    --type all
```

> **Note:** The topological and semantic dashboards require updates to support VQVAEWithInitial models (hybrid encoder architecture). Currently only the engineering dashboard is fully supported.

### Engineering Dashboard (`--type engineering`)

Technical overview for model evaluation and debugging:

| Panel | Content |
|-------|---------|
| Architecture Schematic | Flow diagram: Input → Encoders → Categories → Levels → Decoder |
| Training Curves | Loss and quality metrics over 300 epochs |
| Utilization Heatmap | 7 categories × 3 levels with utilization percentages |
| Reconstruction MSE | Per-category reconstruction error bars |
| Summary Metrics | Quality (0.96), utilization (94%), parameters (3.2M), epochs (300) |

![Engineering Dashboard](images/100k_full_features_engineering.png)

### Topological Dashboard (`--type topological`)

> ⚠️ **Currently unavailable** for VQVAEWithInitial models. Visualization code needs update to parse hybrid model state dict.

Codebook embedding space analysis:

| Panel | Content |
|-------|---------|
| t-SNE Embedding | All codebook vectors projected to 2D, colored by category, shaped by level |
| Similarity Matrix | Cosine similarity between codebook centroids |
| Statistics | Total codes, active codes, embedding dimensions, model quality |

**Interpreting t-SNE:** Points are L2-normalized before projection to prevent artificial clustering from dimension padding. Clear category separation indicates the VQ-VAE learned distinct embedding spaces. Within-category clustering of levels (●L0, ■L1, ▲L2) shows hierarchical structure is preserved.

### Semantic Dashboard (`--type semantic`)

> ⚠️ **Currently unavailable** for VQVAEWithInitial models. Visualization code needs update to parse hybrid model state dict.

Feature-to-category mapping analysis:

| Panel | Content |
|-------|---------|
| Feature-Category Matrix | Which features belong to which category |
| Category Sizes | Number of features per category (bar chart) |
| Feature Families | INITIAL, Summary, Temporal, Architecture counts |
| Token Structure | Compositional vocabulary explanation |
| Category Correlation | Inter-category orthogonality |

---

## Next Steps

1. **NOA Phase 1:** Train agent to predict behavioral tokens from (θ, u₀)
2. **Ablation Studies:** Remove feature families to measure contribution
3. **Transfer Learning:** Fine-tune on domain-specific operators
4. **Token Analysis:** Interpret discovered behavioral categories

---

**Generated:** January 4, 2026
**Validated by:** Claude Opus 4.5
**Status:** PRODUCTION READY
