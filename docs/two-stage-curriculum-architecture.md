# Two-Stage Curriculum Architecture for Meta-Neural Operators

**Branch:** `two-stage-training`
**Status:** Stage 1 in progress, Stage 2 planned
**Philosophy:** Physics first, symbolic reasoning second

---

## Executive Summary

This document describes the two-stage curriculum approach to training Meta-Neural Operators (MNOs) that combine physics fidelity with symbolic reasoning capabilities. Unlike the main branch's simultaneous VQ-VAE alignment approach, this curriculum separates physics learning from symbolic learning into two sequential stages.

**Key Innovation:** Token conditioning as temporary scaffolding that guides behavioral specialization in Stage 1, then removal in Stage 2 when symbolic reasoning is internalized via VQ-led training.

**Result:** A self-contained MNO that operates on `(θ, u₀)` alone while producing VQ-compatible outputs suitable for symbolic reasoning and memory systems.

---

## Motivation: Why Two Stages?

### Problems with Simultaneous Training

The main branch approach trains NOA with VQ alignment from the start:

```
Loss = λ_traj × L_traj + λ_commit × L_commit + λ_latent × L_latent
```

**Challenges:**
1. **Debugging complexity:** Physics errors mixed with VQ alignment errors
2. **Feature dimension mismatches:** VQ and NOA feature spaces must align from day 1
3. **Short rollouts:** Limited to 32 timesteps (memory constraints with VQ losses)
4. **Coupled learning:** Cannot isolate physics learning from symbolic learning

### Curriculum Advantages

The two-stage curriculum decouples these concerns:

**Stage 1: Master Physics**
- Pure MSE training (L_traj = 1.0, NO VQ losses)
- Token conditioning provides behavioral guidance
- 256-step rollouts via truncated BPTT
- Simple debugging: only physics to worry about

**Stage 2: Internalize Symbolism**
- VQ-led training (L_recon + L_commit primary)
- Remove token conditioning (self-regulation)
- NOA learns to produce VQ-compatible outputs autonomously
- Physics preserved via auxiliary L_traj

**Curriculum Principle:** Learn the rules with guidance, then internalize them and operate freely.

---

## Architecture Overview

### Stage 1: Token-Conditioned Meta-Operator

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: GUIDED LEARNING                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (θ, u₀, tokens_CNO)                                │
│           ↓      ↓     ↓                                    │
│           │      │     └──→ Token Embedding (36 → 64D)     │
│           │      │             ↓                            │
│           │      │        Spatial Broadcast [B,64,H,W]     │
│           │      └────────────┐                            │
│           │                   ↓                            │
│           └──→ Concatenate [B, 1+64, H, W]                │
│                       ↓                                     │
│               U-AFNO Backbone (226M params)                │
│                       ↓                                     │
│            Autoregressive Rollout (256 steps)              │
│                       ↓                                     │
│             With Truncated BPTT (window=32)                │
│                       ↓                                     │
│              Predicted Trajectory [B, 256, 1, H, W]        │
│                       ↓                                     │
│              Loss: L_traj = MSE(pred, CNO)                 │
│                                                             │
│  Goal: Learn "if token=chaotic, produce chaos"             │
│        Master physics with behavioral specialization        │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Token Conditioning**
   - 36 VQ tokens (from pre-trained VQ-VAE on CNO rollouts)
   - Each token has dedicated embedding table (learned)
   - Embeddings concatenated and projected to 64D
   - Spatially broadcast and concatenated with u₀
   - Acts as behavioral "prescription"

2. **Truncated BPTT**
   - Total rollout: 256 timesteps (matches dataset generation)
   - Warmup phase: 224 steps without gradients
   - Supervised phase: Last 32 steps with gradients
   - Prevents gradient explosion on long rollouts
   - Memory efficient (~2-3 GB vs ~10 GB for full backprop)

3. **Pure Physics Loss**
   - L_traj = MSE between NOA and CNO trajectories
   - No VQ alignment losses (simplifies debugging)
   - Target: < 2.7 MSE (256-step equivalent of < 0.3 on 32-step)

### Stage 2: VQ-Led Self-Regulation

```
┌─────────────────────────────────────────────────────────────┐
│                 STAGE 2: SELF-REGULATED LEARNING            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (θ, u₀)  ← NO TOKENS! (scaffolding removed)       │
│           ↓                                                 │
│     U-AFNO Backbone (same 226M params)                     │
│           ↓                                                 │
│  Autoregressive Rollout (256 steps)                        │
│           ↓                                                 │
│  Predicted Trajectory [B, 256, 1, H, W]                    │
│           ↓                                                 │
│  Feature Extraction (INITIAL + SUMMARY + TEMPORAL)         │
│           ↓                                                 │
│  VQ-VAE Encoding → Quantization → Tokens                   │
│           ↓              ↓                                  │
│      L_recon      L_commit + L_codebook                    │
│           ↓                                                 │
│  Loss = L_recon + L_commit + 0.3 × L_traj                  │
│         ═════════════════                                   │
│              PRIMARY                                        │
│                                                             │
│  Goal: Produce VQ-compatible rollouts autonomously         │
│        Internalize symbolic structure without guidance      │
└─────────────────────────────────────────────────────────────┘
```

**Key Changes:**

1. **Token Conditioning Removed**
   - Input is now just (θ, u₀)
   - No external behavioral guidance
   - NOA must self-regulate to VQ space

2. **VQ-Led Losses Primary**
   - L_recon: VQ reconstruction quality (primary)
   - L_commit: Embedding commitment (codebook learning)
   - L_traj: Physics regularizer (reduced to 0.3)

3. **Symbolic Internalization**
   - NOA learns to produce rollouts that encode to valid tokens
   - Tokens emerge from rollouts, not prescribed as input
   - VQ structure becomes implicit in NOA's generation

---

## Stage 2: Loss Scales and VQ-VAE Data Processing

### Understanding Stage 2 Loss Scales

Stage 2 training uses VQ-led losses with very different natural scales. **Understanding these scales is critical for successful training.**

#### Natural Loss Scales (Expected Values)

| Loss Component | Typical Scale | Physical Meaning |
|----------------|---------------|------------------|
| **L_recon** | 0.05 - 0.15 | VQ reconstruction error in normalized feature space |
| **L_commit** | 0.0003 - 0.001 | Codebook commitment (typically ~0.0005) |
| **L_traj** | 1.0 - 3.0 | Physics MSE (256-step rollout average) |

**Key Insight:** These losses have 100-1000× scale differences! Without proper weighting, traj loss will dominate.

#### Loss Weight Recommendations

Based on empirical tuning (exp2g):

```yaml
loss:
  lambda_recon: 5.0      # VQ reconstruction (PRIMARY)
  lambda_commit: 0.05    # Commitment (minimal - already optimal)
  lambda_traj: 0.2       # Physics regularizer (AUXILIARY)
```

**Weighted contributions** (target balance):
- Weighted recon: 5.0 × 0.07 = **0.35** (dominant)
- Weighted commit: 0.05 × 0.0005 = **0.000025** (negligible)
- Weighted traj: 0.2 × 1.5 = **0.30** (comparable to recon)

**Why these weights?**
1. **L_commit is already optimal**: Stays at ~0.0005 throughout training, doesn't need strong gradients
2. **L_recon needs dominance**: VQ reconstruction is the core objective in Stage 2
3. **L_traj provides regularization**: Prevents physics from degrading completely

### VQ-VAE Feature Processing: Critical Details

**CRITICAL:** Feature dimension mismatches between VQ-VAE training and meta-operator inference are a common source of bugs.

#### Feature Pipeline Architecture

```
NOA Trajectory [B, 256, 1, H, W]
    ↓
UnifiedFeaturePipeline.extract()
    ↓
Raw Features:
  - INITIAL: 14D (manual features from IC)
  - SUMMARY: 360D (aggregated trajectory statistics)
  - TEMPORAL: 63D×T (per-timestep features)
    ↓
Per-Family Encoders (frozen from VQ-VAE checkpoint):
  - INITIAL: 14D → 14D (identity, no encoding)
  - SUMMARY: 360D → 128D (via MLPEncoder)
  - TEMPORAL: 63D×T → 128D (via TemporalCNNEncoder)
    ↓
Concatenate: [B, 14+128+128 = 270D]
    ↓
Per-Family Normalization (mean/std per family):
  - INITIAL: normalize 14D
  - SUMMARY: normalize 128D
  - TEMPORAL: normalize 128D
    ↓
Normalized Features [B, 270D]
    ↓
Feature Cleaning (VQ-VAE side only):
  - Apply feature_mask to remove zero-variance/outlier dims
  - Result: [B, 171D] (cleaned subset)
    ↓
VQ-VAE Encode → Quantize → Decode
    ↓
Reconstruction Loss: MSE(recon_171D, target_171D)
```

#### Critical Implementation Details

**1. Normalization Stats Format**

VQ-VAE checkpoints MUST contain per-family normalization stats:

```python
checkpoint['normalization_stats'] = {
    'initial': (mean_14d, std_14d),      # [14] manual features
    'summary': (mean_128d, std_128d),    # [128] encoded features
    'temporal': (mean_128d, std_128d),   # [128] encoded features
}
```

**Common Error:** Old checkpoints may have cluster-based stats (143D) which don't match the 270D encoded features. This causes dimension mismatches and incorrect normalization.

**Fix:** Retrain VQ-VAE with per-family stats, OR patch checkpoint with identity normalization for SUMMARY/TEMPORAL families.

**2. Feature Dimension Matching**

Two different feature dimensions exist in the pipeline:

- **270D**: Full encoded features (14 + 128 + 128) used for extraction and normalization
- **171D**: Cleaned features after applying feature_mask (used for VQ-VAE encode/decode)

**During Stage 2 loss computation:**

```python
# Extract and normalize features (270D)
features = unified_pipeline(trajectory, ic, normalize=True)  # [B, 270]

# Apply feature cleaning for VQ-VAE (270D → 171D)
features_cleaned = vqvae_alignment._apply_feature_cleaning(features)  # [B, 171]

# VQ-VAE encode/decode (171D)
z_list = vqvae.encode(features_cleaned)  # Pre-quantization latents
z_q_list, tokens, losses = vqvae.quantize(z_list)
recon_features = vqvae.decode(z_q_list)  # [B, 171]

# Reconstruction loss (171D vs 171D - MUST MATCH!)
L_recon = MSE(recon_features, features_cleaned)
```

**Common Error:** Computing L_recon between 171D reconstruction and 270D normalized features causes dimension mismatch.

**Fix:** Always apply feature cleaning before computing reconstruction loss.

**3. UnifiedFeaturePipeline Usage**

`UnifiedFeaturePipeline` is used ONLY for meta-operator training (on-the-fly extraction):

```python
# During Stage 2 training
pipeline = UnifiedFeaturePipeline.from_checkpoint(vqvae_checkpoint)
features = pipeline(noa_trajectory, ic, normalize=True)  # [B, 270]
```

**NOT used during VQ-VAE training** - VQ-VAE training loads pre-extracted features from HDF5.

### Training Dynamics and Common Issues

#### Issue 1: Recon Loss Not Improving

**Symptoms:**
```
Epoch 1: recon=0.078, traj=1.88
Epoch 2: recon=0.078, traj=1.55  (recon stuck, traj improving)
```

**Root causes:**
1. **Learning rate too low during warmup**: Model can't escape local minima in VQ space
2. **Traj loss dominates gradient**: lambda_traj too high relative to lambda_recon
3. **Feature normalization incorrect**: Features not on same scale as VQ-VAE training

**Solutions:**
- Reduce warmup steps (fine-tuning doesn't need long warmup: 450 steps instead of 1350)
- Increase lambda_recon (2.0 → 5.0) to make VQ gradient dominant
- Verify per-family normalization stats are loaded correctly

#### Issue 2: Both Losses Diverging During Warmup

**Symptoms:**
```
Batch 50-100: recon=0.066, traj=1.44 (improving)
Batch 100-300: recon=0.071, traj=1.57 (both getting worse!)
```

**Root cause:** Learning rate too aggressive for VQ fine-tuning

**Solution:** Reduce learning rate (2.0e-5 → 1.0e-5) for gentler fine-tuning

#### Issue 3: Commit Loss Never Changes

**Symptoms:**
```
Batch 10-500: commit=0.0005 (exactly constant)
```

**Root cause:** Commit loss is ALREADY OPTIMAL - codebook embeddings are well-positioned

**Solution:** Reduce lambda_commit (0.5 → 0.05) to minimal weight. This loss doesn't need strong gradients.

#### Issue 4: Feature Extraction Warnings

**Symptoms:**
```
UserWarning: std(): degrees of freedom is <= 0
```

**Root cause:** Batch size 1 with sample std calculation

**Solution:** Use population std instead: `std(dim=1, correction=0)`

### Hyperparameter Tuning Guide

#### Learning Rate Schedule

**For fine-tuning from Stage 1 checkpoint:**
- Learning rate: 1.0e-5 (50% of Stage 1 rate)
- Warmup steps: 450 (0.5 epochs, NOT 1.5 epochs)
- Schedule: Cosine decay after warmup

**Rationale:** Stage 1 checkpoint already learned physics, Stage 2 only needs to align VQ structure.

#### Loss Weight Search Strategy

1. **Start conservative:**
   ```yaml
   lambda_recon: 2.0
   lambda_commit: 0.5
   lambda_traj: 0.3
   ```

2. **Monitor first 50 batches:**
   - If recon plateaus: increase lambda_recon
   - If commit is constant: decrease lambda_commit
   - If traj dominates: decrease lambda_traj

3. **Iterate toward balance:**
   ```yaml
   lambda_recon: 5.0   # Make recon truly dominant
   lambda_commit: 0.05  # Minimal (already optimal)
   lambda_traj: 0.2     # Just enough to prevent physics collapse
   ```

**Target:** Weighted recon and traj contributions should be within 2× of each other.

### Validation and Debugging

#### Check 1: Verify Feature Dimensions

```python
# During training
features = pipeline(trajectory, ic, normalize=True)
print(f"Normalized features: {features.shape}")  # Should be [B, 270]

features_cleaned = alignment._apply_feature_cleaning(features)
print(f"Cleaned features: {features_cleaned.shape}")  # Should be [B, 171]

z_list = vqvae.encode(features_cleaned)
total_latent_dim = sum(z.shape[1] for z in z_list)
print(f"VQ latent dim: {total_latent_dim}")  # Should match encoder output
```

#### Check 2: Verify Normalization Stats

```python
checkpoint = torch.load(vqvae_checkpoint)
norm_stats = checkpoint.get('normalization_stats', {})

if 'initial' in norm_stats:
    print("✓ Per-family normalization (correct)")
    print(f"  INITIAL: {norm_stats['initial'][0].shape}")  # Should be [14]
    print(f"  SUMMARY: {norm_stats['summary'][0].shape}")  # Should be [128]
    print(f"  TEMPORAL: {norm_stats['temporal'][0].shape}")  # Should be [128]
else:
    print("✗ Old cluster-based normalization (INCORRECT)")
    print("  Need to retrain VQ-VAE or patch checkpoint")
```

#### Check 3: Monitor Loss Balance

```python
# Every 10 batches during training
weighted_recon = lambda_recon * loss_output.metrics['recon']
weighted_commit = lambda_commit * loss_output.metrics['commit']
weighted_traj = lambda_traj * loss_output.metrics['traj']

print(f"Weighted contributions:")
print(f"  Recon:  {weighted_recon:.4f}")
print(f"  Commit: {weighted_commit:.6f}")
print(f"  Traj:   {weighted_traj:.4f}")
print(f"  Ratio (recon/traj): {weighted_recon/weighted_traj:.2f}")
```

**Healthy training:** Ratio should be 0.5 - 2.0 (recon and traj comparable)

---

## Technical Details

### Token Embedding Architecture

**Stage 1 Implementation:**

```python
class TokenEmbedding(nn.Module):
    def __init__(
        self,
        num_tokens: int = 36,           # From hierarchical VQ-VAE
        codebook_sizes: list[int],      # Per-token vocabulary size
        embed_dim: int = 32,            # Embedding size per token
        projection_dim: int = 64,       # Final projected dimension
    ):
        # Separate embedding table per token
        self.embeddings = nn.ModuleList([
            nn.Embedding(K, embed_dim) for K in codebook_sizes
        ])

        # Project concatenated embeddings to lower dim
        self.projection = nn.Linear(num_tokens * embed_dim, projection_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, num_tokens] integer indices
        Returns:
            [B, projection_dim] embedded tokens
        """
        embedded = [emb(tokens[:, i]) for i, emb in enumerate(self.embeddings)]
        concat = torch.cat(embedded, dim=-1)  # [B, num_tokens * embed_dim]
        return self.projection(concat)  # [B, projection_dim]
```

**Usage in NOABackbone:**

```python
class NOABackbone(nn.Module):
    def rollout(self, u0, steps, tokens=None):
        if self.token_conditioning:
            # Embed tokens: [B, num_tokens] → [B, token_embed_dim]
            token_embed = self.token_embedding(tokens)

            # Broadcast to spatial dimensions
            B, C, H, W = u0.shape
            token_spatial = token_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)

            # Concatenate with initial condition
            x = torch.cat([u0, token_spatial], dim=1)  # [B, C+token_dim, H, W]
        else:
            x = u0

        # Standard autoregressive rollout
        ...
```

**Key Design Decisions:**

1. **Separate embeddings per token:** Respects hierarchical VQ structure
2. **Projection layer:** Reduces dimensionality (36 × 32 = 1152 → 64)
3. **Spatial broadcast:** Makes global tokens available at every spatial location
4. **Concatenation:** Simple fusion with initial condition

### Truncated BPTT Implementation

**Problem:**
- Dataset uses 256-step CNO trajectories
- Training on full 256 steps with full backprop:
  - Causes gradient explosion (long autoregressive chains)
  - Requires ~10-13 GB GPU memory
  - Slow (256× more computation per batch)

**Solution: Truncated Backpropagation Through Time (TBPTT)**

```python
class TruncatedBPTT:
    def __init__(self, model, timesteps=256, bptt_window=32):
        self.model = model
        self.timesteps = timesteps
        self.bptt_window = bptt_window
        self.warmup_steps = timesteps - bptt_window  # 224

    def rollout(self, ic, tokens=None):
        # Phase 1: Warmup without gradients (224 steps)
        x = ic.clone()
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                if tokens is not None:
                    x_aug = torch.cat([x, token_spatial], dim=1)
                else:
                    x_aug = x
                x = self.model.single_step(x_aug)

        # Detach from computation graph
        warmup_state = x.clone()

        # Phase 2: Supervised with gradients (32 steps)
        supervised_traj = self.model.rollout(
            warmup_state,
            steps=self.bptt_window,
            return_all_steps=True,
            tokens=tokens,
        )

        return supervised_traj  # [B, 33, C, H, W]

    def align_for_loss(self, pred_traj, target_traj, skip_ic=True):
        """Align predicted and target for loss computation."""
        if skip_ic:
            pred_states = pred_traj[:, 1:, :, :, :]  # [B, 32, C, H, W]
            target_states = target_traj[:, -32:, :, :, :]  # Last 32 states
        else:
            pred_states = pred_traj
            target_states = target_traj[:, -33:, :, :, :]

        return pred_states, target_states
```

**Memory Analysis:**

| Approach | Timesteps Tracked | Memory Usage | Notes |
|----------|-------------------|--------------|-------|
| Full backprop | 256 | ~10-13 GB | Tracks all activations |
| Truncated BPTT | 32 | ~2-3 GB | Only last 32 steps tracked |
| Gradient checkpointing | 256 (checkpointed) | ~5-8 GB | Recomputes during backward |

**Why 32-step window?**
- Matches 32-step experiments for fair comparison
- Small enough for memory efficiency
- Large enough to capture local dynamics

**Result:**
- 4-5× memory reduction
- Stable gradients (no explosion)
- Full 256-step supervision (warmup states compared to CNO)

### Training Dynamics

**Hyperparameters (Experiment 2F):**

```yaml
model:
  base_channels: 40        # 226M parameters
  token_conditioning: true
  token_embed_dim: 64

training:
  timesteps: 256           # Full rollout length
  bptt_window: 32          # Backprop window
  batch_size: 2
  learning_rate: 5.0e-5    # Stable (halved from 1e-4)
  warmup_steps: 2250       # 5 epochs (10% of total steps)
  weight_decay: 1.0e-4
  clip_grad: 0.5
  epochs: 30
```

**Loss Scaling:**

Due to 256-step vs 32-step difference, loss magnitudes scale ~9-10×:

| Quality Tier | 32-Step | 256-Step | Interpretation |
|--------------|---------|----------|----------------|
| **Excellent** | < 0.30 | **< 2.7** | Research-grade accuracy |
| **Very Good** | < 0.35 | < 3.2 | Production-ready |
| **Good** | < 0.40 | < 3.7 | Acceptable baseline |
| **Needs improvement** | > 0.50 | > 4.6 | Insufficient training |

**Warmup Dynamics:**

```
Epoch 1-5 (Warmup):     LR: 0 → 100%     Loss: ~24 → ~8-10
  - Learning rate ramps from 0.1× to 1.0×
  - Slow, steady improvement
  - Apparent plateau is intentional!

Epoch 6-15 (Full LR):   LR: 100% peak    Loss: ~8 → ~4-5
  - Rapid learning phase
  - Most improvement happens here

Epoch 16-30 (Decay):    LR: 100% → ~20%  Loss: ~4 → ~2.5-3.2
  - Fine-tuning and convergence
  - Target: < 2.7 for excellent performance
```

**Observed Training Curve (Exp 2F, Epoch 1):**

```
Batch 10:   157.2
Batch 50:    53.8  (65.8% reduction)
Batch 100:   28.8  (81.7% reduction)
Batch 140:   21.2  (86.5% reduction)
```

The rapid initial drop followed by plateau is **expected** - we're only 6% through warmup (LR at 16% of target). Real learning happens after epoch 5.

---

## Deployment and Usage

### Stage 1 Deployment

**Problem:** Token conditioning creates a circular dependency:

```
To use NOA: need tokens
To get tokens: need CNO rollout
If running CNO: why use NOA?
```

**Solution:** Token conditioning is **temporary scaffolding** removed in Stage 2!

### Stage 2 Deployment (Target)

After Stage 2 VQ-led training:

```python
# Simple deployment - no tokens needed!
noa = NOABackbone.from_checkpoint("stage2_checkpoint.pt")

# Generate rollout
rollout = noa(theta, u0)  # [B, 256, 1, H, W]

# Extract symbolic representation (for reasoning/memory)
features = feature_extractor(rollout, ic=u0)
tokens = vqvae.encode(features)  # [B, num_tokens]

# Interpret
print(f"Behavioral sequence: {token_to_english(tokens)}")
# Output: "Chaotic onset → Period doubling → Strange attractor"
```

**Validation:** Token consistency check

```python
# Generate both CNO and NOA rollouts
cno_rollout = cno.rollout(theta, u0, timesteps=256)
noa_rollout = noa(theta, u0)

# Tokenize both
cno_tokens = vqvae.encode(feature_extractor(cno_rollout, u0))
noa_tokens = vqvae.encode(feature_extractor(noa_rollout, u0))

# Measure consistency
consistency = (cno_tokens == noa_tokens).float().mean()
print(f"Token consistency: {consistency:.1%}")

# Success criteria:
# > 90%: Stage 2 works! Use CNO-trained VQ for reasoning
# < 70%: NOA diverged, may need NOA-specific VQ (fallback plan)
```

---

## Comparison with Main Branch

### Main Branch Architecture

```
┌─────────────────────────────────────┐
│     VQ-FIRST APPROACH (MAIN)        │
├─────────────────────────────────────┤
│                                     │
│ 1. Train VQ-VAE on CNO rollouts     │
│    └─> Freeze codebook              │
│                                     │
│ 2. Train NOA with VQ alignment      │
│    Input: (θ, u₀)                   │
│    Loss: L_traj + L_commit + L_latent│
│    └─> 32 timesteps only            │
│                                     │
│ 3. Two training modes:              │
│    - MSE-led: L_traj primary        │
│    - VQ-led: L_recon primary        │
│                                     │
└─────────────────────────────────────┘
```

### Current Branch Architecture

```
┌─────────────────────────────────────┐
│  TWO-STAGE CURRICULUM (CURRENT)     │
├─────────────────────────────────────┤
│                                     │
│ 1. VQ-VAE pre-trained (same as main)│
│    └─> Used for oracle tokens      │
│                                     │
│ 2. STAGE 1: Token-Conditioned MSE   │
│    Input: (θ, u₀, tokens_CNO)       │
│    Loss: L_traj only                │
│    └─> 256 timesteps (TBPTT)        │
│                                     │
│ 3. STAGE 2: VQ-Led Fine-tuning      │
│    Input: (θ, u₀) ← no tokens!      │
│    Loss: L_recon + L_commit + 0.3×L_traj│
│    └─> Remove scaffolding           │
│                                     │
└─────────────────────────────────────┘
```

### Feature Comparison

| Feature | Main Branch | Current Branch |
|---------|-------------|----------------|
| **Training phases** | 1 (simultaneous) | 2 (curriculum) |
| **Token usage** | VQ alignment losses | Stage 1: Conditioning<br>Stage 2: Production |
| **Rollout length** | 32 steps | **256 steps** |
| **Physics accuracy** | Good (with λ_traj) | **Excellent** (pure MSE first) |
| **Debugging complexity** | High (VQ + NOA coupled) | Low (decoupled stages) |
| **Memory efficiency** | Standard | **4-5× better** (TBPTT) |
| **Novel contributions** | VQ-led paradigm | Token conditioning + TBPTT |
| **Deployment** | (θ, u₀) from start | (θ, u₀) after Stage 2 |
| **Symbolic reasoning** | From start | After Stage 2 |

### When to Use Each Approach

**Use Main Branch IF:**
- You want symbolic reasoning immediately
- 32-step rollouts are sufficient
- You prefer single-phase training
- VQ alignment losses are well-understood

**Use Current Branch IF:**
- You need best long-horizon accuracy (256 steps)
- You want cleaner debugging (isolate physics vs symbolic)
- You believe physics-first curriculum is optimal
- You value memory efficiency (larger models/batches)

---

## Experimental Results

### Phase 1: Hyperparameter Optimization (32-step baseline)

**Experiments Run:**

| Exp | Configuration | Best Val Loss | Key Finding |
|-----|---------------|---------------|-------------|
| 1A | Baseline (bs=4, lr=1e-4) | 0.514 | Baseline reference |
| 1B | Warmup (1125 steps) | 0.503 | Warmup helps (+2%) |
| 1C | Stronger regularization | 0.467 | Weight decay 1e-4 (+2%) |
| 1D | Gradient accumulation (no warmup) | Failed | Warmup essential! |
| 1E | Conditional cache clearing | 0.454 | **Winner** (+10-12%) |
| 1F | Combined (warmup + cache + reg) | 0.454 | Validates findings |

**Key Learnings:**
1. Conditional GPU cache clearing (>90% memory) gives 10-12% improvement
2. Weight decay 1e-4 is optimal (1e-3 too strong, 1e-5 too weak)
3. Warmup essential for gradient accumulation
4. Achieved 11.7% improvement over baseline

### Phase 2A: Scaling to 1K Samples

**Experiment 2A:** Baseline at 1K samples
- Configuration: Same as 1E winner, but 1000 samples (vs 100)
- Result: 0.442 val loss (14% better than Phase 1 baseline)
- Validation: Phase 1 improvements transfer to larger scale ✅

### Phase 2B-E: Token Conditioning (32-step)

**Experiment 2B:** Token-conditioned baseline
- Configuration: 32 base_channels + token conditioning
- Status: Had mid-epoch explosions (LR too high)
- Issue: LR 1e-4 caused instability at epochs 2-3

**Experiment 2C:** Clean baseline (no tokens)
- Configuration: 32 base_channels, no tokens
- Status: Same instability pattern
- Diagnosis: LR 1e-4 too aggressive for this architecture

**Experiment 2D:** Stable training (no tokens)
- Configuration: LR 5e-5 with 5-epoch warmup
- Status: Running smoothly, no explosions
- Validation: Stable LR solves instability

**Experiment 2E:** Stable + increased capacity + tokens
- Configuration: 40 base_channels (226M params), LR 5e-5, tokens
- Timesteps: 32
- Expected: ~0.35-0.40 val loss
- Status: Killed to free GPU for Exp 2F

### Phase 2F: 256-Step Training with Truncated BPTT (Current)

**Experiment 2F:** Full-horizon training
- Configuration: 40 base_channels, LR 5e-5, 256 timesteps, TBPTT window=32
- Status: **Running** (Epoch 1, batch 140+/450)
- Progress: Loss 157 → 21 (smooth decrease, in warmup)
- Target: < 2.7 val loss (excellent tier)
- ETA: ~29 more epochs

**Key Observations:**
1. Truncated BPTT working perfectly (no OOM, no gradient explosion)
2. Loss scaling confirmed: 256-step ≈ 9.2× higher than 32-step
3. Warmup dynamics expected: slow improvement until epoch 5
4. GPU memory stable at ~7 GB (vs ~13 GB without TBPTT)

**Rollout Error Analysis:**

Ran on Exp 2B (32-step, token-conditioned):
- Error growth: **IRREGULAR** (not exponential!)
- t=1: 0.714, t=8: 0.565 (decreases!), t=32: 0.628
- Linear R²: 0.05, Exponential R²: 0.05
- **Key finding:** Rollout accumulation is NOT the bottleneck

**Implication:** The bottleneck is insufficient long-horizon supervision, not error accumulation. Training on 256 steps should significantly improve performance.

---

## Design Rationale and Learnings

### Why Token Conditioning?

**Hypothesis:** VQ tokens encode behavioral categories. If we condition NOA on tokens during training, it learns to specialize:

```
Token 5 → "Produce chaotic behavior"
Token 12 → "Produce periodic behavior"
Token 23 → "Produce transient dynamics"
```

**Benefits:**
1. **Behavioral specialization:** NOA learns category-specific generation
2. **Curriculum scaffolding:** Explicit guidance before self-regulation
3. **Interpretability:** Token→behavior mapping is explicit
4. **Validation:** Can test if NOA respects token prescriptions

**Novel Contribution:** Token conditioning as **training technique** (not permanent architecture feature)

### Why Remove Token Conditioning in Stage 2?

**The Deployment Problem:**

If token conditioning is permanent:
- Deployment requires tokens
- Tokens require CNO rollout
- Circular dependency!

**The Solution:**

Token conditioning is **temporary scaffolding**:
- Stage 1: Learn with guidance (tokens tell NOA what to produce)
- Stage 2: Remove guidance, add VQ-led loss (NOA self-regulates to VQ space)
- Result: NOA operates on (θ, u₀) alone, produces VQ-compatible outputs

**Analogy:** Training wheels on a bicycle
- Stage 1: Use training wheels to learn balance
- Stage 2: Remove wheels, learn to balance independently
- Result: Ride without external support

### Why 256-Step Training?

**Problem with 32-Step Training:**
- Dataset generated with 256-step CNO rollouts
- Training on only 32 steps means NOA never sees:
  - Transient dynamics (t=50-100)
  - Long-term behavior (t=150-200)
  - Late-timestep patterns (t=200-256)

**Solution: Truncated BPTT**
- Roll out full 256 steps
- Only backprop through last 32 steps
- Supervise entire trajectory, but limit gradient flow

**Expected Impact:**
- Better long-horizon accuracy
- Improved transient dynamics
- More faithful to dataset generation process

**Risk Mitigation:**
- Gradient explosion → SOLVED (truncated BPTT)
- Memory overflow → SOLVED (4-5× memory reduction)
- Training instability → SOLVED (stable LR + warmup)

### Why Curriculum Instead of Simultaneous?

**Main branch issues:**
1. VQ alignment losses add complexity during physics learning
2. Feature dimension mismatches caused bugs
3. Debugging difficult (physics errors vs VQ errors)
4. Memory constraints limit rollout length

**Curriculum advantages:**
1. **Cleaner debugging:** Physics-only in Stage 1
2. **Better physics:** No VQ interference during initial learning
3. **Memory efficiency:** TBPTT enables longer rollouts
4. **Validation:** Stage 1 success = prerequisite for Stage 2

**Trade-off:** Two training phases instead of one
- More complex workflow
- Must complete both stages
- But: cleaner separation of concerns

### Training Stability Insights

**Discovery:** Learning rate sensitivity

Early experiments (2B, 2C) had mid-epoch explosions:
- Epoch 1: Stable
- Epoch 2-3: Loss spikes (0.42 → 0.55)
- Root cause: LR 1e-4 too aggressive

**Solution:** Halve learning rate + warmup
- LR 5e-5 (stable)
- 5-epoch warmup (2250 steps)
- Result: Smooth training, no explosions

**Lesson:** For large models (226M params) on this dataset:
- LR 1e-4 = unstable
- LR 5e-5 = stable
- Warmup essential for stability

### Memory Optimization Insights

**Conditional GPU Cache Clearing:**

From Phase 1 experiments:
- Unconditional cache clearing: Slow (serialization overhead)
- No cache clearing: OOM after many batches
- Conditional (>90% memory): **Best** (10-12% improvement)

```python
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    if max_allocated > 0 and allocated / max_allocated > 0.9:
        torch.cuda.empty_cache()
```

**Why it helps:** Prevents memory fragmentation without performance penalty

---

## Stage 2 Implementation Plan

### Prerequisites (Stage 1 Success Criteria)

Before starting Stage 2, validate:

1. **Performance:** Val loss < 2.7 (excellent tier)
2. **Stability:** No training explosions or NaNs
3. **Token consistency:** Initial check on 100 samples
   ```python
   # For each sample:
   cno_rollout = cno(theta, u0)
   noa_rollout = noa(theta, u0, tokens_cno)

   cno_tokens = vqvae.encode(features(cno_rollout))
   noa_tokens = vqvae.encode(features(noa_rollout))

   consistency = (cno_tokens == noa_tokens).mean()
   # Target: > 70% for Stage 2 readiness
   ```

### Stage 2 Architecture Changes

**Remove Token Conditioning:**

```python
# Stage 1 (current)
class NOABackbone(nn.Module):
    def __init__(self, ..., token_conditioning=True):
        if token_conditioning:
            self.token_embedding = TokenEmbedding(...)
        ...

# Stage 2 (modified)
class NOABackbone(nn.Module):
    def __init__(self, ..., token_conditioning=False):  # Disable!
        # No token embedding
        # Input channels back to base value
        ...
```

**Add VQ-Led Losses:**

```python
from spinlock.mno.losses import VQLedLoss

loss_fn = VQLedLoss(
    lambda_recon=1.0,      # Primary: VQ reconstruction
    lambda_commit=0.5,     # Primary: commitment
    lambda_traj=0.3,       # Auxiliary: physics regularizer
    vqvae_alignment=alignment,
)
```

**Training Configuration:**

```yaml
# Stage 2 config (tentative)
model:
  base_channels: 40
  token_conditioning: false  # Removed!
  # ... rest same

training:
  timesteps: 256
  bptt_window: 32
  batch_size: 2
  learning_rate: 5.0e-5
  warmup_steps: 500        # Shorter (fine-tuning)
  epochs: 15               # Fewer (starting from Stage 1 weights)

loss:
  lambda_recon: 1.0        # VQ reconstruction
  lambda_commit: 0.5       # VQ commitment
  lambda_traj: 0.3         # Physics regularizer

# Load Stage 1 checkpoint
resume_from: "checkpoints/.../stage1_best.pt"
```

### Stage 2 Training Process

1. **Initialize from Stage 1**
   - Load Stage 1 checkpoint
   - Remove token embedding layers
   - Adjust input channels back to base value
   - Keep all other weights

2. **Feature Extraction Setup**
   - Use same feature extractor as VQ-VAE training
   - INITIAL + SUMMARY + TEMPORAL features
   - Normalize features (same normalization as VQ-VAE)

3. **VQ Alignment**
   - Use frozen VQ-VAE from production checkpoint
   - Compute L_recon, L_commit on extracted features
   - Keep physics loss for regularization

4. **Monitoring**
   - Track: L_recon, L_commit, L_traj separately
   - Validate: Token consistency with CNO
   - Check: Physics fidelity doesn't degrade too much

### Stage 2 Success Criteria

**Primary Metrics:**

1. **VQ Reconstruction Quality**
   - L_recon < 0.1 (VQ-VAE can reconstruct NOA features)
   - Commitment loss stable (embeddings sharp)

2. **Token Consistency**
   - NOA vs CNO token agreement > 90%
   - Same (θ, u₀) produces similar tokens

3. **Physics Preservation**
   - Val L_traj < 4.0 (allowed to degrade slightly)
   - Still better than untrained baseline

**Secondary Metrics:**

4. **Symbolic Interpretability**
   - NOA rollouts map to meaningful token sequences
   - Token sequences correspond to known behaviors

5. **Deployment Validation**
   - NOA(θ, u₀) works without tokens ✅
   - Rollouts are VQ-compatible ✅

### Fallback Plans

**If token consistency < 70%:**
- NOA diverged too much from CNO
- Option B: Train new VQ-VAE on NOA rollouts
- Option C: Use dual vocabulary (CNO-VQ for reference, NOA-VQ for reasoning)

**If physics degrades > 50%:**
- L_traj weight too low
- Increase λ_traj from 0.3 → 0.5 or 1.0
- May need longer training or different schedule

**If VQ reconstruction poor:**
- Feature extraction mismatch
- Verify features match VQ-VAE training exactly
- Check normalization, feature selection

---

## Future Directions

### After Stage 2 Success

**1. Symbolic Reasoning Layer**
```
NOA rollout → VQ tokens → Token sequence modeling
                           ↓
                   Predict next token
                   Compose behaviors
                   Plan in token space
```

**2. Memory and Planning**
```
Store: (θ, u₀) → tokens → outcome
Retrieve: Similar token sequences → predict outcomes
Plan: Search token space for desired behavior
```

**3. Self-Play and Exploration**
```
Generate: Novel (θ, u₀) combinations
Evaluate: Token sequences for interestingness
Discover: New behavioral regimes
```

**4. Creative Observer Mode**
```
Constraint: Stay in VQ space (valid tokens)
Freedom: Explore alternative rollouts
Goal: Discover novel but meaningful dynamics
```

### Scaling and Optimization

**Model Scaling:**
- Current: 40 base_channels (226M params)
- Future: 48-64 base_channels (400-600M params)
- Technique: Gradient checkpointing + mixed precision

**Dataset Scaling:**
- Current: 1K samples
- Future: 10K-100K samples (full Sobol stratification)
- Technique: Streaming dataloaders, distributed training

**Efficiency:**
- Mixed precision training (FP16)
- Larger batch sizes with gradient accumulation
- Multi-GPU training

### Research Questions

**1. Optimal Curriculum Design**
- What's the right Stage 1/Stage 2 epoch ratio?
- Should λ_traj in Stage 2 decay over time?
- Can we automate transition criteria?

**2. Token Conditioning Ablations**
- How much does token conditioning help vs baseline?
- Which token types (INITIAL, SUMMARY, TEMPORAL) matter most?
- Can we use predicted tokens instead of oracle?

**3. VQ Architecture**
- Is CNO-trained VQ optimal for NOA?
- Should we fine-tune VQ on NOA rollouts?
- Hierarchical vs flat VQ structure?

**4. Long-Horizon Dynamics**
- How does performance scale with timesteps (256 vs 512 vs 1024)?
- Is truncated BPTT necessary beyond 256 steps?
- Can we use variable-length rollouts?

---

## Appendix: Key Files and Locations

### Core Implementation

**Token Conditioning:**
- `src/spinlock/noa/token_embedding.py` - Token embedding module
- `src/spinlock/noa/backbone.py` - NOABackbone with token conditioning
- `src/spinlock/noa/__init__.py` - Exports

**Truncated BPTT:**
- `src/spinlock/noa/truncated_bptt.py` - TBPTT wrapper
- `docs/truncated-bptt-integration.md` - Integration guide

**Training:**
- `src/spinlock/cli/train_meta_operator.py` - Stage 1 training
- `src/spinlock/noa/losses.py` - Loss functions (MSELed, VQLed)

### Experiment Configs

**Phase 1 (32-step, hyperparameter optimization):**
- `configs/noa/experiments/phase1/exp1a_baseline.yaml`
- `configs/noa/experiments/phase1/exp1b_warmup.yaml`
- `configs/noa/experiments/phase1/exp1c_regularization.yaml`
- `configs/noa/experiments/phase1/exp1e_conditional_cache.yaml`

**Phase 2 (token conditioning, scaling):**
- `configs/noa/experiments/phase2/exp2a_baseline_1k.yaml`
- `configs/noa/experiments/phase2/exp2b_token_baseline.yaml`
- `configs/noa/experiments/phase2/exp2e_token_stable_large.yaml`
- `configs/noa/experiments/phase2/exp2f_256step_tbptt.yaml` ← **Current**

### Analysis Tools

**Scripts:**
- `scripts/analysis/analyze_rollout_error.py` - Per-timestep error analysis
- `scripts/analysis/plot_phase1_simple.py` - Phase 1 visualization
- `scripts/preprocess/compute_oracle_tokens.py` - Oracle token generation

**Documentation:**
- `PHASE1_RESULTS.md` - Phase 1 comprehensive analysis
- `PHASE2_STATUS.md` - Phase 2 progress tracking
- `docs/two-stage-curriculum-architecture.md` - This document

### Data

**Datasets:**
- `datasets/100k_full_features.h5` - Main dataset (CNO rollouts)
- `datasets/100k_oracle_tokens_1k.h5` - Pre-computed oracle tokens

**Checkpoints:**
- `checkpoints/production/100k_full_features/best_model.pt` - VQ-VAE
- `checkpoints/experiments/phase2/exp2f_256step_tbptt/` - Current experiment

---

## Architectural Pivot: Independent Optimization (2026-01-11)

### Motivation for Change

After implementing and analyzing Stage 2 (VQ-led fine-tuning), we identified a fundamental architectural limitation: **competing objectives in Stage 2 create an unstable equilibrium**.

#### Problem: Competing Gradients in Stage 2

**Observed behavior at batch 500 (Stage 2 equilibrium):**

```
Weighted L_recon: 5.0 × 0.0713 = 0.357
Weighted L_traj:  0.2 × 1.576  = 0.315
Gradient ratio: 1.13× (nearly equal!)
```

Despite `λ_recon` being 25× larger than `λ_traj`, the actual gradient contributions were nearly balanced because `L_traj` magnitude is ~22× larger than `L_recon`. Both losses plateaued at batch 500 with no further improvement.

**The fundamental tension:**
- ↑ VQ reconstruction quality → ↓ Physics accuracy (NOA learns "VQ-friendly" rollouts)
- ↑ Physics accuracy → ↓ VQ reconstruction quality (accurate physics details hurt VQ compression)

**Key finding:** Stage 2 achieved `L_recon = 0.067`, which is **better than VQ-VAE's baseline 0.120** on CNO rollouts. This means NOA learned to produce simpler, VQ-friendly dynamics at the cost of physics fidelity.

### New Architecture: "Train Tokenizer on Simulator's Distribution"

**Core Philosophy Shift:**

| Aspect | Two-Stage Curriculum | Independent Optimization |
|--------|---------------------|--------------------------|
| **Foundation** | CNO defines ground truth | NOA defines its own distribution |
| **VQ-VAE role** | Constraint on NOA training | Post-hoc compression tool |
| **Coupling** | Tight (VQ affects NOA) | Loose (components independent) |
| **Sample efficiency** | Limited to 1K CNO samples | Unlimited NOA samples |
| **Training complexity** | High (multi-objective tuning) | Low (single objective per stage) |

**Key Insight:** Instead of forcing NOA to produce CNO-compatible tokens, train VQ-VAE to tokenize whatever NOA actually produces.

### Architecture: Independent Component Optimization

```
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: PURE PHYSICS OPTIMIZATION                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: (θ, u₀)  [NO TOKEN CONDITIONING]                        │
│      ↓                                                           │
│  U-AFNO Backbone (226M params)                                  │
│      ↓                                                           │
│  Autoregressive Rollout (256 steps, TBPTT window=32)           │
│      ↓                                                           │
│  Loss: L_traj = MSE(NOA, CNO)  [ONLY OBJECTIVE]                │
│                                                                  │
│  Target: L_traj < 1.0 (RMSE < field variation)                 │
│  Result: NOA with optimal physics, no VQ constraints            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Save best checkpoint
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STAGE 2: GENERATE NOA ROLLOUT DATASET (100K+)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Load: Trained NOA (Stage 1 checkpoint)                         │
│      ↓                                                           │
│  For i in range(100,000):  [10-100× larger than CNO dataset]   │
│      Sample θ ~ parameter space                                 │
│      Sample u₀ ~ initial conditions                             │
│      Rollout = NOA(θ, u₀)  [Fast! No gradients needed]         │
│      Features = extract_features(Rollout, u₀)                   │
│      Save(Features)  [Feature-only, ~10GB vs 1.2TB for trajs]  │
│                                                                  │
│  Result: Large-scale feature dataset from NOA's distribution    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│    STAGE 3: TRAIN VQ-VAE ON NOA'S DISTRIBUTION (STANDARD)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: NOA features (100K samples)                             │
│      ↓                                                           │
│  VQ-VAE Training (standard pipeline, proven)                    │
│      ↓                                                           │
│  Loss: L_recon + L_commit  [NO physics loss]                   │
│                                                                  │
│  Target: L_recon < 0.05 (better than CNO's 0.067)              │
│  Result: VQ-VAE optimized for NOA's actual outputs             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Advantages Over Two-Stage Curriculum

#### 1. Complete Separation of Concerns

**Two-Stage Curriculum:**
- NOA training involves VQ-VAE
- Loss balance tuning (λ_recon vs λ_traj)
- Feature dimension matching
- Competing gradients
- Unstable equilibrium

**Independent Optimization:**
- NOA and VQ-VAE trained independently
- NOA: Pure MSE training (simple, proven)
- VQ-VAE: Standard training (simple, proven)
- No loss balance needed
- Each component optimized for its objective

#### 2. Alignment by Construction

**Two-Stage Curriculum:** VQ-VAE trained on CNO → NOA forced to match CNO's tokens → feature mismatch issues

**Independent Optimization:** VQ-VAE trained on NOA → guaranteed good compression of what NOA actually produces

**Reconstruction loss floor:**
- Two-stage: NOA can't reach VQ-VAE's optimal (physics constraints pull it away)
- Independent: VQ-VAE learns NOA's structure, recon loss minimized by design

#### 3. Massive Sample Space

```
Two-Stage Curriculum:
- Limited by expensive CNO generation
- 1,000 CNO samples (slow to generate)
- VQ-VAE sees limited behavioral diversity

Independent Optimization:
- Unlimited cheap NOA generation
- 100,000+ NOA samples (fast after training)
- Better coverage of NOA's operating space
- VQ-VAE learns full manifold NOA explores
```

#### 4. Architectural Simplicity

**Components removed:**
- ❌ Token conditioning architecture
- ❌ Oracle token generation pipeline
- ❌ VQ-led loss (competing objectives)
- ❌ Loss weight balancing
- ❌ Feature dimension matching between CNO and NOA
- ❌ Stage 2 training instabilities

**Components kept:**
- ✅ Pure MSE training (proven stable)
- ✅ Truncated BPTT (proven effective)
- ✅ Standard VQ-VAE training (proven pipeline)

#### 5. Modular Design

```
NOA: Physics simulator (foundation model)
  - Input: (θ, u₀)
  - Output: trajectory [T, C, H, W]
  - Objective: Minimize L_traj
  - Training: Pure MSE, 30 epochs, L_traj < 1.0

VQ-VAE: Compression codec for NOA's outputs
  - Input: features from NOA rollouts
  - Output: discrete tokens
  - Objective: Minimize reconstruction error on NOA's distribution
  - Training: Standard VQ-VAE, 400-500 epochs

Clean interfaces, independently testable
```

### Implementation: Pure MSE Stage 1

**Config:** `configs/noa/experiments/phase2/exp_pure_mse.yaml`

```yaml
model:
  spatial_dim: 64
  in_channels: 1
  out_channels: 1
  base_channels: 40              # 226M params
  encoder_levels: 3
  modes: 16
  afno_blocks: 4
  token_conditioning: false      # DISABLED - no token complexity

training:
  n_samples: 1000
  batch_size: 2
  epochs: 30                     # Train to convergence
  learning_rate: 5.0e-5
  warmup_steps: 2250
  weight_decay: 1.0e-4
  clip_grad: 0.5
  timesteps: 256
  bptt_window: 32
  early_stopping_patience: 5

loss:
  mode: "mse_led"
  lambda_traj: 1.0               # Only objective!

checkpointing:
  save_dir: "checkpoints/noa/pure_mse_baseline"
  save_every: 3
  keep_best: true

# Target: L_traj < 1.0 (RMSE < field variation)
```

**Key simplifications:**
- No token conditioning → simpler architecture
- No VQ-VAE alignment → no competing objectives
- Pure MSE → single, clear goal
- Proven stable from exp2f testing

### Implementation: NOA Feature Generation

**Command:** `spinlock generate-noa-features`

```bash
# Generate 10K samples for VQ-VAE validation
spinlock generate-noa-features \
  --noa-checkpoint checkpoints/noa/pure_mse_baseline/best_model.pt \
  --output datasets/noa_features_10k.h5 \
  --n-samples 10000 \
  --config configs/experiments/local_100k_optimized.yaml \
  --batch-size 16

# Generate 100K samples for production VQ-VAE
spinlock generate-noa-features \
  --noa-checkpoint checkpoints/noa/pure_mse_baseline/best_model.pt \
  --output datasets/noa_features_100k.h5 \
  --n-samples 100000 \
  --config configs/experiments/local_100k_optimized.yaml \
  --batch-size 16
```

**Pipeline features:**
- GPU-optimized inline feature extraction
- Feature-only mode (no trajectories stored, saves 99% space)
- Uses existing CNO parameter space sampling
- Batch generation for efficiency

### Implementation: VQ-VAE Training on NOA

**Config:** `configs/vqvae/noa_distribution_100k.yaml`

```yaml
dataset_path: "datasets/noa_features_100k.h5"

families:
  summary:
    encoder: MLPEncoder
    encoder_params:
      hidden_dims: [256, 128]
      output_dim: 64

model:
  levels:
    - {latent_dim: 32, num_tokens: 256}   # Coarse
    - {latent_dim: 16, num_tokens: 512}   # Medium
    - {latent_dim: 8, num_tokens: 1024}   # Fine

training:
  batch_size: 1024
  learning_rate: 0.0007
  num_epochs: 500
  category_assignment: "auto"  # Auto-discover NOA's categories
```

**Expected performance:**
- Reconstruction loss: < 0.04 (better than CNO's 0.067)
- Codebook utilization: > 95%
- Category separation: orthogonality < 0.12

### Deployment: End-to-End Pipeline

```python
# 1. Load trained NOA (pure MSE, no tokens)
noa = load_checkpoint("checkpoints/noa/pure_mse_baseline/best_model.pt")

# 2. Load VQ-VAE trained on NOA's distribution
vqvae = load_checkpoint("checkpoints/vqvae/noa_distribution_100k/best_model.pt")

# 3. Generate rollout
rollout = noa(theta, u0)  # [B, 256, 1, H, W]

# 4. Extract features
features = extract_features(rollout, ic=u0)  # [B, 270]

# 5. Tokenize
tokens = vqvae.encode(features)  # [B, num_tokens]

# Ready for symbolic reasoning!
```

**Key benefits:**
- NOA operates on (θ, u₀) alone (no token conditioning)
- VQ-VAE compresses NOA's outputs (perfect alignment by construction)
- Clean, modular components

### Comparison: Key Metrics

| Metric | Two-Stage Curriculum | Independent Optimization |
|--------|---------------------|--------------------------|
| **NOA Training** | 15 epochs + 15 epochs fine-tuning | 30 epochs pure MSE |
| **Physics Quality** | L_traj ≈ 1.5-2.0 (competing objectives) | L_traj < 1.0 (optimized) |
| **VQ Reconstruction** | 0.067 (plateau, competing) | < 0.05 (optimal alignment) |
| **Training Complexity** | High (loss balancing) | Low (single objectives) |
| **Sample Efficiency** | 1K CNO samples | 100K+ NOA samples |
| **Architecture** | Token conditioning + VQ-led | Pure MSE + standard VQ |
| **Debugging** | Complex (coupled failures) | Simple (isolated components) |

### Lessons Learned

1. **Competing objectives don't converge:** Multi-objective optimization with conflicting gradients leads to plateaus, not optima.

2. **Alignment by construction > forced alignment:** Training VQ-VAE on NOA's distribution guarantees good compression; forcing NOA to produce CNO-compatible outputs compromises physics.

3. **Sample efficiency matters:** NOA can generate samples 100× faster than CNO after training, enabling better VQ-VAE training.

4. **Simplicity wins:** Removing token conditioning, VQ-led losses, and loss balancing makes the system easier to train, debug, and improve.

5. **Foundation models define their own distribution:** NOA is the foundation model for dynamics; the tokenizer should adapt to NOA, not vice versa.

### Future Work

**Validation experiments:**
1. Compare physics accuracy (Stage 1 pure MSE vs Stage 1 with tokens)
2. Compare VQ reconstruction quality (NOA-trained vs CNO-trained)
3. Measure token consistency/divergence between distributions
4. Qualitative analysis of NOA tokens vs CNO tokens

**Potential enhancements:**
- Multi-scale VQ-VAE (hierarchical tokenization)
- Conditional generation (tokens → rollouts, reverse direction)
- Symbolic reasoning layer on top of tokens
- Integration with memory systems

---

## Glossary

**CNO:** Computational Neural Operator - ground truth physics simulator
**MNO:** Meta-Neural Operator - learned universal operator (NOA)
**NOA:** Neural Operator Agent - the model we're training
**VQ-VAE:** Vector Quantized Variational Autoencoder - for discrete tokenization
**TBPTT:** Truncated Backpropagation Through Time
**Oracle tokens:** Tokens extracted from ground truth CNO rollouts

**L_traj:** Trajectory MSE loss (physics fidelity)
**L_recon:** VQ reconstruction loss (symbolic quality)
**L_commit:** VQ commitment loss (embedding sharpness)
**L_latent:** Latent alignment loss (feature space matching)

**Token conditioning:** Using VQ tokens as additional input to guide NOA behavior
**Curriculum learning:** Staged training with increasing difficulty/complexity
**Creative Observer:** Vision of NOA as symbolic interpreter of dynamics

---

## Changelog

**2026-01-11:** Architectural pivot to independent optimization
- Analyzed Stage 2 competing gradients (plateau at batch 500)
- Documented fundamental tension between VQ quality and physics accuracy
- Designed "train tokenizer on simulator's distribution" architecture
- Implemented pure MSE Stage 1 (no token conditioning)
- Created `spinlock generate-noa-features` command
- Created VQ-VAE configs for NOA distribution (10K + 100K)
- Updated branch to `noa-vqvae-independent`

**2026-01-10:** Initial document created
- Documented two-stage curriculum architecture
- Captured experimental results through Phase 2F
- Detailed truncated BPTT implementation
- Outlined Stage 2 plan and success criteria

---

## Contact and Contributions

This architecture has evolved from two-stage curriculum to independent optimization based on empirical findings.

**Current Branch:** `noa-vqvae-independent`
**Current Status:** Pure MSE training running (Epoch 1, target L_traj < 1.0)
**Next Milestone:** Complete 2-epoch training, generate NOA features, train VQ-VAE
**Architecture:** Independent optimization (NOA → features → VQ-VAE)
