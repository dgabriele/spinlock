# VQ-VAE Architecture

## Overview

Spinlock's Vector-Quantized Variational Autoencoder (VQ-VAE) converts operator features into discrete behavioral tokens through hierarchical encoding, category assignment, and vector quantization. The architecture supports multiple encoding paths and assignment strategies to handle diverse simulation data.

**Purpose:** Behavioral tokenization for efficient sequence modeling and analysis

**Key Features:**
- Multiple encoding paths (fixed-length, variable-length, hybrid)
- Flexible category assignment (static clustering or learnable)
- Hierarchical multi-level representation
- Dead code reset for codebook utilization

## High-Level Flow

```
Input Features → Category Assignment → Grouped Encoding → Vector Quantization → Discrete Tokens
      ↓                    ↓                    ↓                    ↓
   [B, D]            [B, K, D_k]          [B, K, L, C]         [B, K*L]
```

**Notation:**
- `B`: Batch size
- `D`: Total feature dimension
- `K`: Number of categories
- `D_k`: Features per category k
- `L`: Hierarchical levels
- `C`: Codebook size per level

## Encoding Paths

### 1. Fixed-Length Path (Standard)

For pre-computed feature vectors from SDF simulations.

```
Features [B, D]
    ↓
CategoryAssignment (static or learnable)
    ↓
GroupedFeatureExtractor (per-category MLPs)
    ├─ Category 1 [B, D_1] → MLP → [B, hidden_dim]
    ├─ Category 2 [B, D_2] → MLP → [B, hidden_dim]
    └─ Category K [B, D_K] → MLP → [B, hidden_dim]
    ↓
CategoricalProjector (hierarchical levels)
    ├─ Level 1 [B, K, hidden_dim] → Linear → [B, K, codebook_dim]
    ├─ Level 2 [B, K, hidden_dim] → Linear → [B, K, codebook_dim]
    └─ Level L [B, K, hidden_dim] → Linear → [B, K, codebook_dim]
    ↓
VectorQuantizer (per-category per-level codebooks)
    ├─ Codebook[cat=1, level=1]: [C, codebook_dim]
    ├─ Codebook[cat=1, level=2]: [C, codebook_dim]
    └─ ... (K*L total codebooks)
    ↓
Discrete Tokens [B, K*L] (indices into codebooks)
```

**Configuration:**
```yaml
families:
  initial:
    encoder: identity  # Pass-through
  temporal:
    encoder: identity  # No temporal encoding
```

### 2. Variable-Length Path (Temporal)

For raw temporal sequences with multi-scale dynamics.

```
Raw Temporal Features [B, T, D_temporal]
    ↓
PyramidTemporalEncoder (multi-scale convolutions)
    ├─ Scale 1 (fine):   Conv1d(kernel=3, stride=1) → [B, T, hidden]
    ├─ Scale 2 (medium): Conv1d(kernel=5, stride=2) → [B, T/2, hidden]
    └─ Scale 3 (coarse): Conv1d(kernel=7, stride=4) → [B, T/4, hidden]
    ↓
Global Pooling (max + mean) → [B, 6*hidden]
    ↓
Concatenate with Initial Features [B, D_initial]
    ↓
Combined Features [B, D_initial + 6*hidden]
    ↓
[Continue with Fixed-Length Path above]
```

**Configuration:**
```yaml
families:
  initial:
    encoder: identity
  temporal:
    encoder: PyramidTemporalEncoder
    encoder_config:
      hidden_dim: 64
      num_scales: 3
```

**Performance:** Runtime temporal encoding (not pre-computed)

### 3. Hybrid Initial Path (End-to-End CNN)

For learning optimal initial condition representations.

```
Raw Initial Conditions [B, 14]
    ↓
InitialHybridEncoder (CNN)
    ├─ Conv1d(14 → 32, kernel=3)
    ├─ ReLU + BatchNorm
    ├─ Conv1d(32 → 64, kernel=3)
    ├─ ReLU + BatchNorm
    ├─ Conv1d(64 → 128, kernel=3)
    └─ Global Average Pooling → [B, 128]
    ↓
Concatenate with Pre-computed Features [B, D_features]
    ↓
Expanded Features [B, 128 + D_features]
    ↓
[Continue with Fixed-Length Path above]
```

**Configuration:**
```yaml
families:
  initial:
    encoder: initial_hybrid
    encoder_config:
      manual_dim: 14
      cnn_output_dim: 128
```

**Benefits:**
- End-to-end trainable initial encoder
- Gradients flow from VQ-VAE losses back to CNN
- Can discover better IC representations than hand-crafted features

**Wrapper:** Uses `VQVAEWithInitial` wrapper around `CategoricalHierarchicalVQVAE`

## Category Assignment Strategies

### Static Assignment (Default)

**Method:** Pre-computed clustering on feature correlations

**Process:**
1. Compute pairwise feature correlations
2. K-means clustering to group correlated features
3. Fixed mapping: feature `i` → category `j`
4. Deterministic routing during training

**Advantages:**
- Fast (no gradient computation for assignments)
- Interpretable (categories based on correlation structure)
- Deterministic
- Lower memory usage

**Configuration:**
```yaml
training:
  category_assignment: auto  # Default
```

**Implementation:** Feature indices pre-assigned to categories

### Learnable Assignment (Optional)

**Method:** Gradient-based optimization with Gumbel-Softmax

**Process:**
1. Initialize assignment matrix from clustering
2. Soft assignment matrix `A ∈ ℝ^(D×K)` with logits
3. Gumbel-Softmax sampling: `softmax((logits + Gumbel) / τ)`
4. Temperature annealing: `τ = 1.0 → 0.1` over epochs
5. Gradients flow through soft routing

**Architecture:**
```
Features [B, D]
    ↓
SoftAssignmentMatrix(temperature=τ)
    ├─ Assignment logits [D, K]
    ├─ Gumbel-Softmax sampling
    └─ Soft assignments [D, K] (sum to 1 per feature)
    ↓
Weighted Features [B, K, D_k]
    ├─ Feature i contributes to multiple categories (weighted)
    └─ Weights determined by soft assignments
    ↓
[Continue with GroupedFeatureExtractor]
```

**Advantages:**
- End-to-end optimization
- Adapts to reconstruction task
- Can discover better groupings
- Flexible feature sharing between categories

**Configuration:**
```yaml
training:
  category_assignment: learnable

learnable_assignment:
  temperature_start: 1.0
  temperature_end: 0.1
  temperature_schedule: linear
  assignment_lr: 0.001
  orthogonality_weight: 0.1
  balance_weight: 0.05
```

**Trade-offs:** ~5% slower, higher memory (gradient storage)

**See Also:** [Assignment Strategies Guide](assignment-strategies.md)

## Component Details

### GroupedFeatureExtractor

**Purpose:** Per-category feature encoding with separate MLPs

**Architecture:**
```python
class GroupedFeatureExtractor:
    def __init__(self, category_dims, hidden_dim):
        self.encoders = nn.ModuleList([
            MLP(dim_k, hidden_dim) for dim_k in category_dims
        ])

    def forward(self, grouped_features):
        # grouped_features: [B, K, D_k]
        encoded = []
        for k, encoder in enumerate(self.encoders):
            encoded.append(encoder(grouped_features[:, k]))
        return torch.stack(encoded, dim=1)  # [B, K, hidden_dim]
```

**Configuration:**
```yaml
encoder_config:
  hidden_dim: 256
  num_layers: 3
  dropout: 0.1
```

### CategoricalProjector

**Purpose:** Hierarchical multi-level projections per category

**Architecture:**
```python
class CategoricalProjector:
    def __init__(self, num_categories, num_levels, hidden_dim, codebook_dim):
        self.projectors = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dim, codebook_dim)
                for _ in range(num_levels)
            ]) for _ in range(num_categories)
        ])

    def forward(self, encoded_features):
        # encoded_features: [B, K, hidden_dim]
        projections = []
        for k in range(self.num_categories):
            level_projections = [
                proj(encoded_features[:, k])
                for proj in self.projectors[k]
            ]
            projections.append(torch.stack(level_projections, dim=1))
        return torch.stack(projections, dim=1)  # [B, K, L, codebook_dim]
```

**Configuration:**
```yaml
categorical_vqvae:
  num_levels: 3
  codebook_dim: 64
```

### VectorQuantizer

**Purpose:** Discrete codebook learning with commitment loss

**Architecture:**
```python
class VectorQuantizer:
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        # Find nearest codebook vectors
        distances = torch.cdist(inputs, self.embedding.weight)
        indices = torch.argmin(distances, dim=-1)
        quantized = self.embedding(indices)

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Commitment loss
        commitment_loss = F.mse_loss(quantized.detach(), inputs)

        return quantized, indices, commitment_loss
```

**Total Codebooks:** `K * L` (one per category per level)

**Configuration:**
```yaml
categorical_vqvae:
  codebook_size: 512
  commitment_cost: 0.25
```

### Decoder

**Purpose:** Reconstruct original features from quantized codes

**Architecture:**
```python
class CategoricalDecoder:
    def __init__(self, num_categories, num_levels, codebook_dim, hidden_dim):
        # Combine all quantized codes
        self.combiner = nn.Linear(num_categories * num_levels * codebook_dim, hidden_dim)

        # Per-category decoders
        self.decoders = nn.ModuleList([
            MLP(hidden_dim, dim_k) for dim_k in category_dims
        ])

    def forward(self, quantized):
        # quantized: [B, K, L, codebook_dim]
        combined = self.combiner(quantized.flatten(1))  # [B, K*L*C]
        decoded = [decoder(combined) for decoder in self.decoders]
        return torch.cat(decoded, dim=-1)  # [B, D]
```

## Training Process

### Loss Components

**1. Reconstruction Loss**
```python
L_recon = MSE(reconstructed_features, original_features)
```
**Weight:** 1.0 (primary objective)

**2. VQ Commitment Loss**
```python
L_vq = commitment_cost * MSE(quantized.detach(), encoded)
```
**Weight:** 0.25 (encourages encoder to commit to codebook)

**3. Orthogonality Loss** (optional)
```python
# Encourage distinct category representations
L_ortho = orthogonality_weight * sum(
    (encoded[k] @ encoded[j].T)**2
    for k != j in categories
)
```
**Weight:** 0.1

**4. Informativeness Loss** (optional)
```python
# Encourage diverse representations within categories
L_info = informativeness_weight * sum(
    -log(det(Cov(encoded[k])))
    for k in categories
)
```
**Weight:** 0.01

**5. Topographic Loss** (optional)
```python
# Encourage smooth codebook organization
L_topo = topographic_weight * sum(
    distance(code[i], code[j]) * distance(index[i], index[j])
    for adjacent codes
)
```
**Weight:** 0.01

**6. Assignment Losses** (learnable mode only)
```python
# Orthogonality: encourage distinct category assignments
L_assign_ortho = orthogonality_weight * (A @ A.T - I).norm()

# Balance: prevent category collapse
L_assign_balance = balance_weight * Var(A.sum(dim=0))

L_assign = L_assign_ortho + L_assign_balance
```
**Weights:** 0.1 (ortho), 0.05 (balance)

**Total Loss:**
```python
L_total = L_recon + L_vq + L_ortho + L_info + L_topo + L_assign
```

### Dead Code Reset

**Problem:** Some codebook vectors become unused (dead codes)

**Solution:** Periodically reset dead codes to active data points

**Strategy:**
```python
def reset_dead_codes(model, data_loader, threshold=0.01):
    # Track code usage over epoch
    usage_counts = count_code_usage(model, data_loader)

    # Identify dead codes (usage < threshold)
    dead_codes = usage_counts < threshold * total_samples

    # Reset to random active samples
    for codebook_idx in dead_codes:
        random_sample = sample_from_data(data_loader)
        model.codebooks[codebook_idx] = random_sample
```

**Configuration:**
```yaml
training:
  dead_code_reset:
    enabled: true
    threshold: 0.01
    reset_strategy: random_active
```

### Checkpointing

**Saved State:**
- Model weights (encoder, projector, codebooks, decoder)
- Optimizer state
- Assignment matrix (if learnable)
- Category assignments (if static)
- Training metrics history

**Format:** PyTorch state dict with metadata

**See Also:** [Checkpoint Format](checkpoint-format.md)

## torch.compile Integration

### Selective Compilation

**Strategy:** Compile compute-intensive components, skip dynamic parts

**Fixed-Length Models:**
```python
# Compile entire forward pass
model.forward = torch.compile(model.forward, mode="reduce-overhead")
```
**Speedup:** ~30-40% faster training

**Variable-Length Models:**
```python
# Compile only static components
model.encoder = torch.compile(model.encoder)
model.decoder = torch.compile(model.decoder)
# Skip temporal encoder (dynamic shapes)
```
**Speedup:** ~15-25% (limited by dynamic temporal encoding)

### Configuration

```yaml
training:
  compile:
    enabled: true
    mode: reduce-overhead  # or "default", "max-autotune"
```

**Trade-offs:**
- First epoch slower (compilation overhead)
- Subsequent epochs much faster
- Higher memory usage during compilation

**See Also:** [torch.compile Optimization](torch-compile.md)

## Performance Characteristics

### Memory Usage

**Fixed-Length:** ~2-3 GB GPU memory (batch size 256)
- Dominated by codebooks (K*L*C*D)
- Efficient for pre-computed features

**Variable-Length:** ~4-6 GB GPU memory (batch size 128)
- Additional temporal encoder states
- Dynamic sequence lengths

**Hybrid:** ~3-4 GB GPU memory (batch size 256)
- CNN encoder parameters
- Gradient storage for end-to-end training

### Training Speed

**Fixed-Length (compiled):**
- ~0.5s per epoch (50k samples, V100)
- Scales linearly with batch size

**Variable-Length (compiled):**
- ~1.2s per epoch (50k samples, V100)
- Runtime temporal encoding overhead

**Hybrid (compiled):**
- ~0.8s per epoch (50k samples, V100)
- CNN forward/backward adds ~60% overhead

### Reconstruction Quality

**Typical Results (after convergence):**
- Reconstruction loss: 0.01-0.05 (normalized features)
- VQ loss: 0.001-0.005
- Codebook utilization: 15-30%
- Feature recovery: >95% (correlation with original)

**By Encoding Path:**
- Fixed-length: Fastest, good quality
- Variable-length: Best for temporal dynamics, slower
- Hybrid: Best for end-to-end IC learning, medium speed

## Usage Examples

### Standard Training (Fixed-Length)

```bash
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae.yaml \
  --epochs 500
```

### Variable-Length with Temporal Encoding

```bash
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae_variable_length.yaml \
  --epochs 500
```

### Hybrid with Learnable Assignments

```bash
poetry run spinlock train-vqvae \
  --config configs/vqvae/learnable_hybrid_variable_length.yaml \
  --epochs 1000
```

### Enable Compilation

```bash
poetry run spinlock train-vqvae \
  --config configs/vqvae/baseline_vqvae.yaml \
  --compile \
  --epochs 500
```

## See Also

- [Assignment Strategies](assignment-strategies.md) - Static vs learnable categories
- [Learnable Assignments](learnable-assignments.md) - Gradient-based category learning
- [Variable-Length Encoding](variable-length-encoding.md) - Temporal pyramid details
- [torch.compile Optimization](torch-compile.md) - Performance optimization
- [Checkpoint Format](checkpoint-format.md) - Model saving/loading
