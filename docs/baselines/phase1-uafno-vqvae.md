# NOA Phase 1 Baseline: U-AFNO Backbone with VQ-VAE Alignment

**Date:** January 6, 2026
**Status:** IMPLEMENTED - Core training infrastructure working
**Dependencies:** VQ-VAE Baseline (`100k_full_features.h5`, `checkpoints/production/100k_full_features/`)

---

## Executive Summary

This document specifies the Phase 1 NOA architecture: a **U-AFNO neural operator backbone** trained with **VQ-VAE perceptual loss**. The NOA operates in continuous function space, generating rollouts whose behavioral features are encoded into discrete tokens via a frozen VQ-VAE.

**Key architectural shift:** The NOA is itself a neural operator (not a transformer on tokens). This creates a physics-native, self-consistent system where the agent operates in the same function space as the dynamics it studies.

### Architecture Overview

```
(θ, u₀) → U-AFNO NOA → Predicted Rollout/Features → VQ-VAE (frozen) → Loss
```

| Component | Role |
|-----------|------|
| **Input** | θ (operator parameters) + u₀ (initial grid) + optional context |
| **Backbone** | U-AFNO neural operator (reuses `operators/u_afno.py`) |
| **Output** | Predicted next grid + feature heads (INITIAL-like, SUMMARY-like, TEMPORAL-like) |
| **Loss Encoder** | Frozen VQ-VAE from Phase 0 (perceptual loss) |
| **Training Signal** | VQ-VAE reconstruction + grid MSE + commitment loss |

---

## Why U-AFNO as NOA Backbone?

| Property | Benefit |
|----------|---------|
| **Physics-native** | Operates directly in continuous function space matching the studied dynamics |
| **Resolution-independent** | Spectral mixing captures global patterns regardless of grid size |
| **Proven infrastructure** | Leverages existing dataset-generation U-AFNO architecture |
| **Self-consistent** | Enables emergent self-modeling and law discovery in the same function space |
| **Efficient** | 4–9× inference speedup vs pure CNN, with global receptive field via FFT |

---

## U-AFNO Backbone Architecture

### Model Configuration

The NOA backbone reuses the existing U-AFNO implementation (`src/spinlock/operators/u_afno.py`):

```yaml
noa_backbone:
  type: "u_afno"
  encoder_levels: 3                # Multi-scale encoding (H/8 bottleneck)
  base_channels: 32                # Feature channels at first level
  afno_blocks: 4                   # Number of spectral mixing blocks
  modes: 16                        # FFT modes retained per dimension
  dropout: 0.1                     # Regularization

  # Optional stochastic elements
  stochastic_block: true           # Noise injection for trajectory diversity
  noise_schedule: "constant"       # Or "annealing", "periodic"
```

### Key Components

**UNetEncoder** (progressive downsampling):
- Input: `[B, C_in, H, W]` → progressively doubled channels
- Skip connections at each level for multi-scale information
- Output: `(bottleneck, skips)` tuple

**AFNOBlock** (spectral bottleneck):
- 2D FFT → learned complex weights → inverse FFT
- Global receptive field via frequency-domain mixing
- Residual connections + feedforward MLP

**UNetDecoder** (progressive upsampling):
- Concatenates skip connections at each level
- Refinement blocks for clean reconstruction
- Output: `[B, C_out, H, W]`

### Intermediate Feature Extraction

The key hook for NOA feature heads is `get_intermediate_features()`:

```python
features = operator.get_intermediate_features(
    x,
    extract_from="all",        # "bottleneck", "skips", or "all"
    skip_levels=[0, 1, 2]      # Which encoder levels to extract
)

# Returns:
# {
#   "bottleneck": [B, C_bottleneck, H/8, W/8],  # AFNO output (most compressed)
#   "skip_0": [B, C_0, H/2, W/2],               # Shallow encoder features
#   "skip_1": [B, C_1, H/4, W/4],               # Mid encoder features
#   "skip_2": [B, C_2, H/8, W/8],               # Deep encoder features
# }
```

---

## VQ-VAE Integration (Training Loss)

### Three-Loss Structure (Implemented)

**Location:** `src/spinlock/noa/vqvae_alignment.py`

The Phase 1 training uses a three-loss structure that combines trajectory supervision with VQ-VAE alignment:

```python
L = L_traj + λ₁ * L_latent + λ₂ * L_commit
```

| Loss | Description | Default Weight |
|------|-------------|----------------|
| `L_traj` | MSE on trajectories vs CNO replay (physics fidelity) | 1.0 |
| `L_latent` | Pre-quantized latent alignment (smooth gradients) | λ₁ = 0.1 |
| `L_commit` | VQ commitment loss (manifold adherence) | λ₂ = 0.5 |

**Why This Structure?**
1. **L_traj**: Anchors physics correctness—NOA must match CNO trajectories
2. **L_latent**: Uses pre-quantization embeddings for smooth gradients (avoids quantization artifacts)
3. **L_commit**: Forces NOA outputs to be expressible in VQ vocabulary (manifold adherence)

### Implementation Details

```python
from spinlock.noa import VQVAEAlignmentLoss

# Load frozen VQ-VAE alignment module
alignment = VQVAEAlignmentLoss.from_checkpoint(
    vqvae_path="checkpoints/production/100k_full_features",
    device="cuda",
)

# In training loop
losses = alignment.compute_losses(pred_trajectory, target_trajectory, ic=ic)
total_loss = state_loss + lambda_latent * losses['latent'] + lambda_commit * losses['commit']
```

### TrajectoryFeatureExtractor

**Location:** `src/spinlock/noa/vqvae_alignment.py:267`

Extracts SUMMARY/TEMPORAL features from NOA-generated trajectories, matching VQ-VAE input format:

```python
# Features extracted from trajectory
summary_features = SummaryExtractor.extract_all(trajectory)  # Spatial, spectral, temporal stats
temporal_agg = temporal_features.mean(dim=1)  # Aggregated temporal dynamics

# Config avoids NaN for M=1 realizations
config = SummaryConfig(
    realization_aggregation=["mean"],  # No std/cv (undefined for M=1)
    temporal_aggregation=["mean"],      # No std (can produce NaN)
)
```

### Why Frozen VQ-VAE?

| Approach | Pros | Cons |
|----------|------|------|
| **Frozen VQ-VAE** | Stable training, proven tokenization, no codebook drift | Less adaptive |
| **Joint Training** | End-to-end optimization, potentially better alignment | Risk of codebook collapse, training instability |

**Current Approach:** Frozen VQ-VAE for Phase 1. Consider joint fine-tuning in Phase 2 if needed.

---

## Feature Heads

The NOA produces auxiliary outputs aligned with Phase 0 feature families:

### INITIAL-like Head
- **Source:** U-AFNO bottleneck at first timestep
- **Aggregation:** Global average pooling + MLP projection
- **Output:** 42D (matching Phase 0 INITIAL encoder output)
- **Purpose:** Characterizes generated initial condition quality

### SUMMARY-like Head
- **Source:** U-AFNO bottleneck across full trajectory (T timesteps)
- **Aggregation:** Temporal mean/max pooling + MLP projection
- **Output:** 128D (matching Phase 0 SUMMARY encoder output)
- **Purpose:** Aggregated behavioral statistics

### TEMPORAL-like Head
- **Source:** Skip connection features across trajectory
- **Aggregation:** 1D ResNet over temporal dimension
- **Output:** 128D (matching Phase 0 TEMPORAL encoder output)
- **Purpose:** Trajectory dynamics embedding

---

## Training Infrastructure (Implemented)

### CNO Replay for State Supervision

**Location:** `src/spinlock/noa/cno_replay.py`

The CNOReplayer reconstructs CNO operators from parameter vectors, enabling state-level supervision without storing full trajectories:

```python
from spinlock.noa import CNOReplayer

# Create replayer from config
replayer = CNOReplayer.from_config(
    "configs/experiments/local_100k_optimized.yaml",
    device="cuda",
    cache_size=8,  # LRU cache for operator reuse
)

# Generate ground-truth trajectory from parameter vector
target_trajectory = replayer.rollout(
    params_vector=params,           # [D] parameter vector
    ic=initial_condition,           # [1, C, H, W]
    timesteps=32,
    num_realizations=1,
    return_all_steps=True,
)  # Returns [1, T+1, C, H, W]
```

### Training Command

```bash
# Basic training (state-only)
poetry run python scripts/dev/train_noa_state_supervised.py \
    --n-samples 500 --epochs 10

# With VQ-VAE alignment (recommended)
poetry run python scripts/dev/train_noa_state_supervised.py \
    --n-samples 500 --epochs 10 \
    --vqvae-path checkpoints/production/100k_full_features \
    --lambda-latent 0.1 --lambda-commit 0.5
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-samples` | 1000 | Number of training samples |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 4 | Batch size (GPU memory limited) |
| `--timesteps` | 32 | Supervision timesteps |
| `--n-realizations` | 1 | Stochastic realizations |
| `--vqvae-path` | None | VQ-VAE checkpoint for alignment |
| `--lambda-latent` | 0.1 | Latent alignment weight |
| `--lambda-commit` | 0.5 | Commitment loss weight |

### Training Results (50 samples, 3 epochs)

```
Epoch 1/3
  Train: total=0.979 state=0.978 latent=0.000044 commit=0.003
  Val: total=0.714 (best)

Epoch 2/3
  Train: total=1.023 state=1.022 latent=0.000051 commit=0.003
  Val: total=0.806

Epoch 3/3
  Train: total=0.958 state=0.956 latent=0.000043 commit=0.003
  Val: total=0.722
```

Training completes without NaN collapse. Occasional NaN gradient warnings are safely skipped.

### NaN Handling

**Resolved issues:**
- NaN gradient checking prevents weight corruption
- Safe aggregation config (mean only) for M=1 realizations
- `nan_to_num()` applied before/after normalization

See `notes/RESUME-2026-01-06-vqvae-alignment-nan-fix.md` for details.

---

## First Training Task: Rollout Feature Encoding

### Objective

Train U-AFNO NOA to generate rollouts whose features are well-reconstructed by VQ-VAE, yielding high-quality discrete behavioral symbols.

### Training Dataset

| Source | Usage |
|--------|-------|
| **θ (parameters)** | Sampled from existing stratified dataset (100K operators) |
| **u₀ (initial grid)** | From `inputs/fields` in HDF5 |
| **Ground-truth trajectories** | Generated via CNOReplayer |
| **VQ-VAE features** | Extracted from trajectories via TrajectoryFeatureExtractor |

---

## Expected Metrics

### Success Criteria (Phase 1 → Phase 2)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **VQ Reconstruction Error** | < 0.15 | `MSE(pred_z_q, gt_z_q)` on validation set |
| **Token Distribution Match** | Entropy > 0.9 × dataset | Compare NOA token entropy to dataset |
| **Codebook Utilization** | > 40% | Same methodology as Phase 0 VQ-VAE |
| **Rollout Fidelity** | MSE < 0.1 | Grid-level comparison over 64 steps |
| **Behavioral Clustering** | >80% cluster agreement | NOA-generated vs. ground-truth cluster assignments |

### Monitoring Metrics

- VQ-VAE reconstruction loss (primary training signal)
- Grid MSE (rollout quality)
- Token entropy (discrete symbol diversity)
- Feature head alignment (INITIAL/SUMMARY/TEMPORAL MSE)
- Training/validation loss curves

---

## Evaluation Methodology

### Rollout Fidelity

Compare NOA-generated rollouts to ground-truth operators:

```python
def evaluate_rollout_fidelity(noa, test_set, steps=256):
    mse_list = []
    for theta, u0, gt_trajectory in test_set:
        pred_trajectory = noa.rollout(theta, u0, steps=steps)
        mse = F.mse_loss(pred_trajectory, gt_trajectory)
        mse_list.append(mse.item())
    return np.mean(mse_list), np.std(mse_list)
```

### Token Distribution Analysis

Compare discrete token distributions:

```python
def analyze_token_distribution(noa, vqvae, test_set):
    noa_tokens = []
    gt_tokens = []

    for theta, u0, gt_features in test_set:
        # NOA-generated tokens
        pred_features = noa(theta, u0)
        pred_z = vqvae.encode(pred_features)
        _, pred_tok, _ = vqvae.quantize(pred_z)
        noa_tokens.append(pred_tok)

        # Ground-truth tokens
        gt_z = vqvae.encode(gt_features)
        _, gt_tok, _ = vqvae.quantize(gt_z)
        gt_tokens.append(gt_tok)

    # Compare distributions
    noa_entropy = compute_entropy(noa_tokens)
    gt_entropy = compute_entropy(gt_tokens)
    token_agreement = compute_agreement(noa_tokens, gt_tokens)

    return noa_entropy, gt_entropy, token_agreement
```

### Behavioral Clustering Coherence

Do NOA-generated rollouts cluster correctly with ground-truth behaviors?

```python
def evaluate_clustering_coherence(noa, vqvae, test_set):
    # Extract embeddings for all samples
    noa_embeddings = []
    gt_embeddings = []

    for theta, u0, gt_features in test_set:
        pred_features = noa(theta, u0)
        pred_z = vqvae.encode(pred_features)
        gt_z = vqvae.encode(gt_features)

        noa_embeddings.append(torch.cat(pred_z, dim=1))
        gt_embeddings.append(torch.cat(gt_z, dim=1))

    # Cluster both
    noa_labels = cluster(noa_embeddings, n_clusters=12)
    gt_labels = cluster(gt_embeddings, n_clusters=12)

    # Measure agreement
    ari = adjusted_rand_index(noa_labels, gt_labels)
    return ari
```

---

## Implementation Roadmap

### Step 1: NOA Model Class

```python
# src/spinlock/noa/backbone.py

class NOABackbone(nn.Module):
    """U-AFNO NOA backbone with feature heads."""

    def __init__(self, config):
        super().__init__()
        self.u_afno = UAFNOOperator(**config.u_afno_params)
        self.initial_head = InitialHead(config.bottleneck_dim, output_dim=42)
        self.summary_head = SummaryHead(config.bottleneck_dim, output_dim=128)
        self.temporal_head = TemporalHead(config.skip_dims, output_dim=128)

    def forward(self, theta, u0, steps=64):
        trajectory = self.rollout(theta, u0, steps)
        features = self.extract_features(trajectory)
        return trajectory, features

    def rollout(self, theta, u0, steps):
        # Generate trajectory autoregressively
        states = [u0]
        x = u0
        for t in range(steps):
            x = self.u_afno(x)
            states.append(x)
        return torch.stack(states, dim=1)

    def extract_features(self, trajectory):
        # Extract bottleneck and skip features
        intermediate = self.u_afno.get_intermediate_features(
            trajectory[:, 0], extract_from="all"
        )

        initial_feat = self.initial_head(intermediate["bottleneck"])
        summary_feat = self.summary_head(trajectory, intermediate["bottleneck"])
        temporal_feat = self.temporal_head(trajectory, intermediate)

        return torch.cat([initial_feat, summary_feat, temporal_feat], dim=1)
```

### Step 2: VQ-VAE Perceptual Loss

```python
# src/spinlock/noa/losses.py

class VQVAEPerceptualLoss(nn.Module):
    """Frozen VQ-VAE as perceptual loss encoder."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.vqvae = load_vqvae(checkpoint_path)
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def forward(self, predicted_features, ground_truth_features):
        with torch.no_grad():
            gt_z = self.vqvae.encode(ground_truth_features)
            gt_z_q, _, _ = self.vqvae.quantize(gt_z)

            pred_z = self.vqvae.encode(predicted_features)
            pred_z_q, _, _ = self.vqvae.quantize(pred_z)

        # Code-space loss
        loss = sum(F.mse_loss(p, g) for p, g in zip(pred_z_q, gt_z_q))
        return loss / len(pred_z_q)
```

### Step 3: Training Pipeline

```python
# src/spinlock/noa/training.py

class NOATrainer:
    def __init__(self, config):
        self.noa = NOABackbone(config.noa)
        self.vq_loss = VQVAEPerceptualLoss(config.vqvae_checkpoint)
        self.optimizer = torch.optim.AdamW(self.noa.parameters(), lr=config.lr)

    def train_step(self, batch):
        theta, u0, gt_features, gt_trajectory = batch

        # Forward pass
        pred_trajectory, pred_features = self.noa(theta, u0)

        # Losses
        vq_loss = self.vq_loss(pred_features, gt_features)
        grid_loss = F.mse_loss(pred_trajectory, gt_trajectory)

        total_loss = vq_loss + 0.5 * grid_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"total": total_loss, "vq": vq_loss, "grid": grid_loss}
```

---

## Implementation Files

| File | Status | Description |
|------|--------|-------------|
| `src/spinlock/operators/u_afno.py` | ✅ | U-AFNO backbone |
| `src/spinlock/noa/backbone.py` | ✅ | NOABackbone class (144M params) |
| `src/spinlock/noa/vqvae_alignment.py` | ✅ | VQVAEAlignmentLoss, TrajectoryFeatureExtractor |
| `src/spinlock/noa/cno_replay.py` | ✅ | CNOReplayer for state supervision |
| `src/spinlock/noa/losses.py` | ✅ | VQVAEPerceptualLoss, FeatureProjector |
| `src/spinlock/noa/training.py` | ✅ | Training utilities, NOAPhase1Trainer |
| `scripts/dev/train_noa_state_supervised.py` | ✅ | Main training script |

---

## Dependencies

- **VQ-VAE Checkpoint:** `checkpoints/production/100k_full_features/` (225D features)
- **Dataset:** `datasets/100k_full_features.h5`
- **Config:** `configs/experiments/local_100k_optimized.yaml` (for CNO replay)

---

## Next Steps

### Remaining Phase 1 Tasks
- [ ] Full-scale training run (1000+ samples, 50+ epochs)
- [ ] Evaluation metrics and benchmarks
- [ ] Hyperparameter tuning (λ₁, λ₂, learning rate)

### Phase 2 (After Phase 1 Success Criteria Met)
1. **Multi-step context:** Lightweight transformer heads on VQ code sequences
2. **Curiosity-driven exploration:** Prediction error as exploration signal
3. **Joint VQ-VAE fine-tuning:** Consider unfreezing VQ-VAE for end-to-end optimization

---

**Generated:** January 6, 2026
**Status:** IMPLEMENTED - Core training infrastructure working
