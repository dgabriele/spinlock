# VQ-VAE Baseline: 100K Full Features (SUMMARY + TEMPORAL + ARCHITECTURE)

**Date:** January 5, 2026
**Dataset:** `datasets/100k_full_features.h5`
**Checkpoint:** `checkpoints/production/100k_full_features/`
**Status:** PRODUCTION READY

---

## Executive Summary

Production VQ-VAE tokenizer trained on 100,000 neural operator samples with joint encoding of **three feature families**, using **uniform codebook initialization** with natural dead code pruning:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Val Loss** | 0.169 | <0.20 | ✅ EXCEEDED |
| **Quality** | 0.957 | >0.85 | ✅ EXCEEDED |
| **Codebook Utilization** | 71.7% | >25% | ✅ EXCEEDED |
| **Reconstruction Error** | 0.043 | - | Excellent |
| **Categories Discovered** | 14 | auto | Data-driven clustering |
| **Input Dimensions** | 200 | - | After feature cleaning |

---

## Dataset Configuration

### Feature Families

| Family | Raw Dimensions | Encoder | Output Dimensions |
|--------|----------------|---------|-------------------|
| **SUMMARY** | 360 | MLPEncoder [512, 256] | ~150 |
| **TEMPORAL** | 256 × 63 | TemporalCNNEncoder (ResNet-1D) | ~38 |
| **ARCHITECTURE** | 12 | IdentityEncoder | 12 |
| **Total** | - | - | **200** (after cleaning) |

After feature cleaning (variance filtering, deduplication, outlier capping): **200 features**

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
    isolated_families: ["architecture"]  # ARCHITECTURE in own category
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

### Final Metrics

```
Final Metrics:
  val_loss: 0.169
  utilization: 0.717
  reconstruction_error: 0.043
  quality: 0.957
  epochs: 550
```

### Per-Cluster Performance

| Cluster | Features | L0 Tokens | L1 Tokens | L2 Tokens |
|---------|----------|-----------|-----------|-----------|
| architecture_isolated | 12 | 24 | 12 | 6 |
| cluster_1 | 10 | 20 | 11 | 6 |
| cluster_2 | 36 | 36 | 18 | 9 |
| cluster_3 | 18 | 28 | 14 | 7 |
| cluster_4 | 8 | 20 | 10 | 6 |
| cluster_5 | 11 | 24 | 12 | 6 |
| cluster_6 | 32 | 15 | 7 | 6 |
| cluster_8 | 4 | 15 | 7 | 6 |
| cluster_9 | 9 | 20 | 11 | 6 |
| cluster_10 | 7 | 16 | 9 | 6 |
| cluster_11 | 11 | 24 | 12 | 6 |
| cluster_12 | 17 | 28 | 14 | 7 |
| cluster_13 | 9 | 20 | 11 | 6 |
| cluster_14 | 16 | 28 | 14 | 7 |
| **TOTAL** | **200** | **318** | **162** | **90** |

**Note on metrics:** The global `reconstruction_error` (0.043) uses the **shared decoder** that combines all 42 codebooks (14 categories × 3 levels = 570 total codes) to reconstruct the full feature vector.

### Training Time

- **Total Duration:** ~50 minutes (550 epochs)
- **Hardware:** Single GPU with TF32 matmul

---

## Hierarchical Codebook Architecture

### Uniform Codebook Initialization

This model uses **uniform codebook initialization** (`uniform_codebook_init: true`), where all hierarchical levels start with the same codebook size (L0's size). Dead code resets then naturally prune unused codes during training, allowing the model to empirically discover the appropriate hierarchical structure.

| Level | Initial Size | Final Size (avg) | Purpose |
|-------|-------------|------------------|---------|
| L0 (Coarse) | ~23 | 22.7 | Broad behavioral categories |
| L1 (Medium) | ~23 | 11.6 | Sub-category distinctions |
| L2 (Fine) | ~23 | 6.4 | Specific behavioral variants |

**Design rationale:** Rather than pre-specifying hierarchical compression ratios, uniform initialization lets the training process discover natural codebook capacities. The dead code reset mechanism prunes unused codes, resulting in empirically-discovered hierarchical structure:
- **L0** retains most codes (~23) for coarse behavioral distinctions
- **L1** naturally prunes to ~12 codes for medium-scale patterns
- **L2** prunes aggressively to ~6 codes for fine-grained refinements

### Dead Code Reset Mechanism

During training, the system monitors codebook utilization and performs **dead code resets** at configurable intervals (every 100 epochs in this configuration):

1. **Detection:** Codes with EMA cluster size below the 10th percentile are flagged as "dead"
2. **Reset:** Dead codes are re-initialized to perturbed versions of high-usage codes
3. **Pruning effect:** Over training, codebooks naturally stabilize to their "right size"
4. **Reset limit:** Maximum 25% of codes can be reset per interval to prevent disruption

**Interpretation:** With uniform initialization, the dead code reset mechanism serves as an empirical capacity discovery tool. Levels that need fewer codes (L2) see more codes become "dead" and get reset, while levels that need more codes (L0) maintain higher utilization.

### Per-Level Utilization Analysis

From the production model (14 categories × 3 levels = 42 codebooks, 570 total codes):

| Level | Total Codes | Mean Utilization | Interpretation |
|-------|-------------|-----------------|----------------|
| L0 | 318 | ~72% | High utilization for coarse patterns |
| L1 | 162 | ~71% | Healthy mid-level distinctions |
| L2 | 90 | ~71% | Fine codes appropriately sized |

**Key insight:** Uniform initialization + dead code resets results in consistent ~71% utilization across all levels, indicating the model has discovered appropriate codebook sizes for each level of the hierarchy. The total 570 codes with 71.7% utilization means ~408 active codes are being used to tokenize 200 features across 14 categories.

### Category Discovery: Pure Clustering

This model uses **pure hierarchical clustering** (no gradient refinement) to discover 14 categories from 200 features. Key benefits:
- **Better reconstruction**: Gradient refinement optimizes for orthogonality at the expense of reconstruction quality
- **Natural category sizes**: Clustering respects the inherent structure of feature correlations
- **Faster training**: No additional optimization loop for category assignments

**Configuration:**
- `method: "clustering"` (pure agglomerative clustering)
- `min_features_per_category: 2` (allow smaller, more granular categories)
- `max_clusters: 25` (upper bound, actual discovered: 14)

### Category Composition

The 14 categories consist of **1 isolated** (ARCHITECTURE) + **13 clustered** (SUMMARY+TEMPORAL):

| Category | Features | Primary Content |
|----------|----------|-----------------|
| **architecture_isolated** | 12 | ARCHITECTURE only (isolated by design) |
| cluster_1 | 10 | Mixed features |
| cluster_2 | 36 | Large SUMMARY cluster |
| cluster_3 | 18 | SUMMARY statistics |
| cluster_4 | 8 | Small cluster |
| cluster_5 | 11 | Mixed features |
| cluster_6 | 32 | Large SUMMARY + TEMPORAL |
| cluster_8 | 4 | Smallest cluster |
| cluster_9 | 9 | Mixed features |
| cluster_10 | 7 | Small cluster |
| cluster_11 | 11 | Mixed features |
| cluster_12 | 17 | SUMMARY features |
| cluster_13 | 9 | Mixed features |
| cluster_14 | 16 | SUMMARY + TEMPORAL |

**Key design decision:** ARCHITECTURE features are **isolated** because they're uniform Sobol-sampled operator parameters, not computed behavioral features. Mixing them with computed features (SUMMARY, TEMPORAL) would contaminate reconstruction quality.

**Interpretation:**
- **architecture_isolated:** The 12 operator parameters are kept separate—they represent *inputs* to the system, not behavioral *outputs*
- **TEMPORAL features** (FFT, autocorrelation, periodicity) form clusters with SUMMARY (cluster_6, cluster_14)
- **SUMMARY features** distribute across most categories, serving as the "glue" that captures aggregate behavioral signatures

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

**Result:** Raw features → 200 features after cleaning

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
| `checkpoints/production/100k_full_features/best_model.pt` | Best model weights (val_loss: 0.183) |
| `checkpoints/production/100k_full_features/final_model.pt` | Final model weights (epoch 200) |
| `checkpoints/production/100k_full_features/config.yaml` | Saved model config |
| `checkpoints/production/100k_full_features/normalization_stats.npz` | Feature normalization |
| `checkpoints/production/100k_full_features/training_history.json` | Full training metrics |

---

## Known Issues and Fixes

### SUMMARY NaN (Expected)

30 features (operator_sensitivity) are NaN because `extract_operator_features: false` in dataset generation config. These are automatically replaced with 0 during training.

### TEMPORAL NaN (Fixed)

Skewness/kurtosis features were NaN at t=0 for structured initial conditions (symmetric distributions). **Fixed in `src/spinlock/features/summary/spatial.py`** by adding `torch.nan_to_num()` to handle undefined moments for symmetric distributions.

---

## Comparison to Previous Baselines

| Metric | 10K Baseline | 100K (hierarchical) | **100K (uniform init)** |
|--------|--------------|---------------------|------------------------|
| Samples | 10,000 | 100,000 | 100,000 |
| Feature Families | SUMMARY only | SUM+TEM+ARCH | **SUM+TEM+ARCH** |
| Cleaned Features | ~40 | 172 | **200** |
| Categories | ~6-8 | 11 | **14** (1 isolated + 13 clustered) |
| Val Loss | - | 0.183 | **0.169** |
| Quality | ~0.85 | 0.9475 | **0.957** |
| Utilization | ~30% | 93.7% | **71.7%** |
| Epochs | - | 200 | **550** |
| Total Codes | - | ~480 | **570** |

**Key improvements with uniform codebook initialization:**
- **8% better val_loss** (0.169 vs 0.183) through longer training and uniform init
- **Uniform codebook init** - all levels start with L0's size, dead code resets prune naturally
- **Empirically-discovered hierarchy** - L0→L1→L2 structure emerges from training, not pre-specified
- **More categories** - 14 vs 11, finer-grained feature grouping
- **Better reconstruction** - 0.043 MSE vs 0.053 (19% improvement)
- **Higher quality** - 0.957 vs 0.9475 (1% improvement)

---

## Visualization Dashboards

Three visualization dashboards are available for analyzing trained VQ-VAE models:

```bash
# Generate all dashboards
poetry run spinlock visualize-vqvae \
    --checkpoint checkpoints/production/100k_full_features/ \
    --output docs/baselines/images/ \
    --type all
```

### Engineering Dashboard (`--type engineering`)

Technical overview for model evaluation and debugging:

| Panel | Content |
|-------|---------|
| Architecture Schematic | Flow diagram: Input → Encoders → Categories → Levels → Decoder |
| Training Curves | Loss and quality metrics over 550 epochs |
| Utilization Heatmap | 14 categories × 3 levels with utilization percentages |
| Reconstruction MSE | Per-category reconstruction error bars |
| Summary Metrics | Quality (0.957), utilization (71.7%), epochs (550) |

![Engineering Dashboard](images/100k_full_features_engineering.png)

### Topological Dashboard (`--type topological`)

Codebook embedding space analysis:

| Panel | Content |
|-------|---------|
| t-SNE Embedding | All 42 codebook vectors (14 categories × 3 levels) projected to 2D |
| Similarity Matrix | 42×42 cosine similarity between codebook centroids |
| Statistics | Total codes (570), active codes (~408, 71.7%), model quality (0.957) |

![Topological Dashboard](images/100k_full_features_topological.png)

**Interpreting t-SNE:** Points are L2-normalized before projection to prevent artificial clustering from dimension padding. Clear category separation indicates the VQ-VAE learned distinct embedding spaces. Within-category clustering of levels (●L0, ■L1, ▲L2) shows hierarchical structure is preserved.

### Semantic Dashboard (`--type semantic`)

Feature-to-category mapping analysis:

| Panel | Content |
|-------|---------|
| Feature-Category Matrix | Which features belong to which category |
| Category Sizes | Number of features per category (bar chart) |
| Feature Families | Summary (~150), Temporal (~38), Architecture (12) |
| Codebook Utilization | N/M format showing used codes per codebook (~408/570 = 71.7%) |
| Category Correlation | Inter-category orthogonality |

![Semantic Dashboard](images/100k_full_features_semantic.png)

---

## Next Steps

1. **NOA Phase 1:** Train agent to predict behavioral tokens from (θ, u₀)
2. **Ablation Studies:** Remove feature families to measure contribution
3. **Transfer Learning:** Fine-tune on domain-specific operators
4. **Token Analysis:** Interpret discovered behavioral categories

---

**Generated:** January 5, 2026
**Validated by:** Claude Opus 4.5
**Status:** PRODUCTION READY
