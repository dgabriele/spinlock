# Category Assignment Strategies

## Overview

Category assignment determines how operator features are grouped into semantic categories before hierarchical encoding and vector quantization. The strategy significantly impacts model interpretability, training speed, and reconstruction quality.

**Two approaches:**
1. **Static Assignment** - Pre-computed clustering (default)
2. **Learnable Assignment** - Gradient-based optimization (optional)

## What is Category Assignment?

### The Problem

Given `D` operator features and `K` desired categories, how do we map features to categories?

**Example:**
- 462 total features
- 14 categories
- Need to assign each feature to exactly one category

**Why it matters:**
- Related features should be grouped together
- Categories enable hierarchical encoding
- Assignment affects reconstruction quality and interpretability

### Visualization

```
Features [D=462]:
  feature_0: ic_u0
  feature_1: ic_u1
  feature_2: temporal_u_mean_0
  ...
  feature_461: spatial_laplacian_max

Categories [K=14]:
  category_0: Initial conditions (u, v)
  category_1: Temporal statistics (mean, std)
  category_2: Spatial derivatives
  ...
  category_13: Interaction terms

Assignment:
  feature_0 → category_0  (ic_u0 is initial condition)
  feature_1 → category_0  (ic_u1 is initial condition)
  feature_2 → category_1  (temporal mean is temporal stat)
  ...
```

## Static Assignment (Clustering-Based)

### How It Works

**Step 1: Compute Feature Correlations**
```python
# Compute pairwise correlations on training data
correlations = np.corrcoef(features.T)  # [D, D]
```

**Step 2: K-Means Clustering**
```python
# Cluster features based on correlation patterns
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K)
assignments = kmeans.fit_predict(correlations)  # [D]
```

**Step 3: Fixed Mapping**
```python
# Create deterministic feature → category mapping
category_assignments = {
    0: [0, 1, 2, 5, 8],      # Category 0 contains features 0, 1, 2, 5, 8
    1: [3, 4, 6, 7, 9, 10],  # Category 1 contains features 3, 4, 6, 7, 9, 10
    ...
}
```

**Step 4: Hard Routing During Training**
```python
def group_features(features, assignments):
    grouped = []
    for k in range(K):
        # Extract features assigned to category k
        feature_indices = assignments[k]
        grouped.append(features[:, feature_indices])
    return grouped  # List of [B, D_k] tensors
```

### Advantages

**1. Fast**
- No gradient computation for assignments
- Simple indexing operations
- ~5% faster than learnable

**2. Interpretable**
- Categories based on correlation structure
- Easy to inspect which features are grouped
- Stable across training runs

**3. Deterministic**
- Same assignments every time
- Reproducible results
- Easy to debug

**4. Lower Memory**
- No assignment matrix gradients
- Smaller optimizer state
- More efficient for large feature sets

### Configuration

```yaml
training:
  category_assignment: auto  # Default (uses static)

# No additional configuration needed
```

### When to Use

**Best for:**
- Production deployments (deterministic, fast)
- Interpretability requirements
- Limited compute resources
- Well-understood feature correlations

**Example use cases:**
- Analyzing learned categories for domain insights
- Debugging feature importance
- Comparing models across experiments

## Learnable Assignment (Gradient-Based)

### How It Works

**Step 1: Initialize from Clustering**
```python
# Start with static assignments
initial_assignments = kmeans_clustering(features)

# Convert to logits
assignment_logits = torch.zeros(D, K)
for d in range(D):
    assignment_logits[d, initial_assignments[d]] = 5.0  # High logit for assigned category
```

**Step 2: Soft Assignment Matrix**
```python
class SoftAssignmentMatrix(nn.Module):
    def __init__(self, num_features, num_categories):
        super().__init__()
        self.logits = nn.Parameter(assignment_logits)  # [D, K]

    def forward(self, temperature=1.0):
        # Gumbel-Softmax for differentiable sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
        soft_assignments = F.softmax((self.logits + gumbel_noise) / temperature, dim=-1)
        return soft_assignments  # [D, K], rows sum to 1
```

**Step 3: Soft Routing**
```python
def soft_group_features(features, soft_assignments):
    # features: [B, D]
    # soft_assignments: [D, K]

    # Weighted feature grouping
    grouped = torch.einsum('bd,dk->bkd', features, soft_assignments)
    # grouped: [B, K, D] (each category gets weighted sum of features)

    return grouped
```

**Step 4: Temperature Annealing**
```python
# Epoch 1: temperature=1.0 (soft, diffuse assignments)
# Epoch 500: temperature=0.1 (hard, nearly one-hot assignments)

temperature = temperature_start + (temperature_end - temperature_start) * (epoch / total_epochs)
```

**Step 5: Gradient Descent**
```python
# Separate optimizer for assignment matrix
assignment_optimizer = torch.optim.Adam([assignment_matrix.logits], lr=0.001)

# Backward pass
total_loss.backward()  # Gradients flow through soft routing

# Update assignments
assignment_optimizer.step()
```

### Architecture Details

```
Features [B, D]
    ↓
SoftAssignmentMatrix(temperature=τ)
    ├─ Logits [D, K] (learnable parameters)
    ├─ Gumbel noise (sampling)
    ├─ Softmax((logits + Gumbel) / τ)
    └─ Soft assignments [D, K]
    ↓
Einsum: features @ assignments
    ↓
Weighted Features [B, K, D_weighted]
    ├─ Category k receives weighted contributions from all features
    └─ Weights determined by soft assignment matrix
    ↓
GroupedFeatureExtractor (per-category MLPs)
    ↓
[Continue with VQ-VAE encoding]
```

### Regularization Losses

**1. Orthogonality Loss**
```python
# Encourage distinct category assignments
# (Different categories should use different features)

assignment_matrix = soft_assignments  # [D, K]
orthogonality_loss = torch.norm(
    assignment_matrix.T @ assignment_matrix - torch.eye(K)
)

# Penalizes overlap between category assignments
```

**2. Balance Loss**
```python
# Prevent category collapse
# (Each category should get similar number of features)

category_sizes = soft_assignments.sum(dim=0)  # [K]
balance_loss = torch.var(category_sizes)

# Penalizes uneven category sizes
```

**Total Assignment Loss:**
```python
L_assign = orthogonality_weight * orthogonality_loss + balance_weight * balance_loss
L_total = L_recon + L_vq + L_assign
```

### Advantages

**1. Task-Optimal**
- Learned to minimize reconstruction loss
- Not constrained by correlation-based clustering
- Can discover non-obvious groupings

**2. End-to-End Differentiable**
- Gradients flow through entire pipeline
- Unified optimization objective
- No separate clustering step

**3. Adaptive**
- Assignments evolve during training
- Can adjust to data distribution
- Flexible feature sharing between categories

**4. Simpler Configuration**
- No need to tune clustering hyperparameters
- Fewer assumptions about feature structure
- Automatic category balancing (via balance loss)

### Configuration

```yaml
training:
  category_assignment: learnable

learnable_assignment:
  # Temperature annealing schedule
  temperature_start: 1.0      # Soft assignments at start
  temperature_end: 0.1        # Nearly hard assignments at end
  temperature_schedule: linear  # "linear", "exponential", "cosine"

  # Regularization weights
  orthogonality_weight: 0.1   # Encourage distinct categories
  balance_weight: 0.05        # Prevent category collapse

  # Optimization
  assignment_lr: 0.001        # Separate learning rate for assignments
  gradient_clip: 1.0          # Clip assignment gradients

  # Freezing (optional)
  freeze_after_epochs: null   # Freeze assignments after N epochs
  freeze_threshold: 0.95      # Freeze when max(assignment) > threshold
```

### When to Use

**Best for:**
- Research and optimization
- Unknown feature correlations
- End-to-end learning (e.g., with hybrid CNN encoder)
- Maximizing reconstruction quality

**Example use cases:**
- Discovering optimal feature groupings for new datasets
- Joint optimization with hybrid initial encoder
- Experiments requiring maximum flexibility

## Comparison

| Aspect | Static (Clustering) | Learnable (Gradient) |
|--------|---------------------|----------------------|
| **Speed** | Faster (~0.5s/epoch) | Slower (~0.53s/epoch, +5%) |
| **Memory** | Lower (~2.5 GB) | Higher (~2.8 GB, +12%) |
| **Interpretability** | High (correlation-based) | Moderate (learned) |
| **Flexibility** | Fixed | Adaptive |
| **Reproducibility** | Deterministic | Stochastic (Gumbel noise) |
| **Configuration** | Minimal | More hyperparameters |
| **Gradient Flow** | Not applicable | End-to-end |
| **Category Balance** | Manual tuning | Automatic (via loss) |
| **Use Case** | Production, interpretability | Research, optimization |

### Reconstruction Quality

**Typical results on same dataset:**

| Metric | Static | Learnable | Difference |
|--------|--------|-----------|------------|
| Reconstruction Loss | 0.0234 | 0.0218 | -6.8% (better) |
| VQ Loss | 0.0024 | 0.0021 | -12.5% (better) |
| Codebook Utilization | 18.3% | 23.7% | +29% (better) |
| Training Time (500 epochs) | 4.2 min | 4.4 min | +4.8% (slower) |

**Interpretation:** Learnable assignments achieve slightly better reconstruction and utilization at modest cost.

## Hybrid Initial Encoder Compatibility

Both assignment strategies work with the hybrid initial encoder, enabling end-to-end gradient flow.

### Static + Hybrid

```yaml
families:
  initial:
    encoder: initial_hybrid

training:
  category_assignment: auto
```

**Gradient flow:**
- CNN encoder gradients: ✓ (from reconstruction loss)
- Assignment gradients: ✗ (fixed assignments)

### Learnable + Hybrid

```yaml
families:
  initial:
    encoder: initial_hybrid

training:
  category_assignment: learnable
```

**Gradient flow:**
- CNN encoder gradients: ✓ (from reconstruction loss)
- Assignment gradients: ✓ (from reconstruction + assignment losses)

**Benefit:** Full end-to-end optimization of both CNN and category assignments.

## Troubleshooting

### Issue: Category Collapse (Learnable)

**Symptoms:**
- All features assigned to 1-2 categories
- Other categories have near-zero assignments
- High balance_loss in logs

**Solutions:**
1. Increase `balance_weight` from 0.05 to 0.10
2. Increase `temperature_end` from 0.1 to 0.3 (slower annealing)
3. Initialize with better clustering (more balanced)

### Issue: Slow Convergence (Learnable)

**Symptoms:**
- Assignments change slowly
- High assignment losses for many epochs
- Reconstruction not improving

**Solutions:**
1. Increase `assignment_lr` from 0.001 to 0.003
2. Use "exponential" temperature schedule (faster annealing)
3. Decrease `orthogonality_weight` to allow more flexibility

### Issue: Gradient Instability (Learnable)

**Symptoms:**
- Training loss spikes
- NaN in assignment matrix
- Gradient norms exploding

**Solutions:**
1. Decrease `assignment_lr` from 0.001 to 0.0005
2. Enable `gradient_clip: 1.0` (already default)
3. Check for dead codes (might need dead code reset)

### Issue: Poor Interpretability (Learnable)

**Symptoms:**
- Learned categories don't match domain expectations
- Features mixed in unexpected ways
- Hard to understand groupings

**Solutions:**
1. Use static assignment instead (more interpretable)
2. Increase `orthogonality_weight` to enforce distinction
3. Freeze assignments after initial learning: `freeze_after_epochs: 100`

### Issue: Inconsistent Categories Across Runs (Learnable)

**Symptoms:**
- Different assignments each training run
- Results not reproducible
- Comparisons difficult

**Solutions:**
1. Set random seed: `torch.manual_seed(42)`
2. Use static assignment for reproducibility
3. Average results across multiple seeds

## Monitoring Assignment Quality

### Static Assignment

**Check category sizes:**
```python
for k, indices in category_assignments.items():
    print(f"Category {k}: {len(indices)} features")
```

**Inspect correlations within categories:**
```python
for k, indices in category_assignments.items():
    category_features = features[:, indices]
    corr_matrix = np.corrcoef(category_features.T)
    avg_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)].mean()
    print(f"Category {k} avg correlation: {avg_corr:.3f}")
```

### Learnable Assignment

**Monitor training metrics:**
```
Epoch 10/500:
  assign_orthogonality: 0.0234  ← Lower is better (distinct categories)
  assign_balance: 0.0123        ← Lower is better (balanced sizes)
  Temperature: 0.98             ← Annealing towards 0.1
```

**Inspect learned assignments:**
```python
# After training
soft_assignments = model.assignment_matrix()  # [D, K]
hard_assignments = soft_assignments.argmax(dim=-1)  # [D]

# Category sizes
for k in range(K):
    size = (hard_assignments == k).sum()
    print(f"Category {k}: {size} features")

# Top features per category
for k in range(K):
    weights = soft_assignments[:, k]
    top_features = torch.topk(weights, k=5)
    print(f"Category {k} top features: {top_features.indices}")
```

## Implementation Examples

### Static Assignment (CLI)

```bash
# Default behavior
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae_variable_length.yaml \
  --epochs 500
```

### Learnable Assignment (CLI)

```bash
# Via config
poetry run spinlock train-vqvae \
  --config configs/vqvae/learnable_hybrid_variable_length.yaml \
  --epochs 1000

# Via flag (any config)
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae_variable_length.yaml \
  --learnable \
  --epochs 1000
```

### Programmatic Usage

```python
from spinlock.encoding import CategoricalHierarchicalVQVAE
from spinlock.encoding.learnable_assignment import SoftAssignmentMatrix

# Static (default)
model = CategoricalHierarchicalVQVAE(config)

# Learnable
assignment_matrix = SoftAssignmentMatrix(
    num_features=462,
    num_categories=14,
    initial_assignments=clustering_result
)
model = CategoricalHierarchicalVQVAE(
    config,
    assignment_matrix=assignment_matrix
)
```

## Best Practices

### Choosing a Strategy

**Use Static if:**
- You need deterministic, reproducible results
- Interpretability is critical
- Compute resources are limited
- Feature correlations are well-understood

**Use Learnable if:**
- You're optimizing for best reconstruction quality
- You're using hybrid CNN encoder (end-to-end benefits)
- You're exploring a new dataset
- You have sufficient compute for longer training

### Configuration Tips

**Static:**
- No configuration needed (works out of the box)
- Focus tuning on other hyperparameters (codebook size, levels, etc.)

**Learnable:**
- Start with default configuration
- Monitor `assign_balance` - if high, increase `balance_weight`
- Monitor `assign_orthogonality` - if high, increase `orthogonality_weight`
- Adjust `temperature_schedule` based on convergence speed

### Evaluation

**Compare both strategies on your dataset:**
1. Train with static assignment (baseline)
2. Train with learnable assignment (same other hyperparameters)
3. Compare reconstruction quality, utilization, training time
4. Choose based on requirements (speed vs quality vs interpretability)

## See Also

- [Architecture](architecture.md) - Overall VQ-VAE architecture
- [Learnable Assignments](learnable-assignments.md) - Implementation details
- [Learnable Mode Guide](learnable-mode-guide.md) - Complete usage guide
- [Variable-Length Encoding](variable-length-encoding.md) - Temporal pyramid integration
