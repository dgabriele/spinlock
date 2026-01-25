# NOA Architecture: CNO-Trained Components

## Overview

The NOA (Neural Operator Agent) architecture consists of two independently-trained components that compose for downstream exploration and reasoning:

1. **VQ-VAE Tokenizer**: Discrete behavioral vocabulary (trained on CNO ground truth)
2. **MNO World Model**: High-fidelity physics simulator (trained on CNO ground truth)

Both components train on the same CNO dataset independently, then compose for NOA cognitive capabilities.

## Design Philosophy

### Key Architectural Principle

**Both VQ-VAE and MNO train independently on CNO ground truth, then compose for NOA.**

- **VQ-VAE**: Learns discrete symbolic representation from CNO behavioral features
- **MNO**: Learns continuous physics simulation from CNO trajectories
- **NOA**: Uses MNO for exploration + VQ-VAE for symbolic reasoning

### Why CNO-Trained Instead of MNO-Generated?

The original 3-stage approach (train MNO → generate MNO features → train VQ on MNO) had the philosophy of "train tokenizer on simulator's distribution." The new approach trains both on ground truth CNO data.

**Advantages:**

1. **Simplicity**: Two independent training pipelines instead of sequential dependency
2. **Modularity**: Each component can be validated independently against CNO
3. **Efficiency**: No need to generate 100K+ MNO rollouts before training VQ-VAE
4. **Parallelism**: VQ-VAE and MNO can be trained simultaneously
5. **Validation**: If MNO achieves high fidelity (L_traj < 1.0), its outputs should be distributionally similar to CNO

**Assumption:**

If MNO achieves physics fidelity with L_traj < 1.0 (RMSE less than typical field variation), then MNO rollouts are distributionally similar enough to CNO that a CNO-trained VQ-VAE will reconstruct MNO outputs with comparable quality.

**Post-Training Validation:**

After training both components, verify that VQ-VAE reconstruction quality on MNO outputs remains high (~0.006 L_recon).

---

## Component 1: VQ-VAE Tokenizer

### Purpose

Provide a discrete behavioral vocabulary for symbolic reasoning over PDE dynamics.

### Training Data

**Input**: CNO ground truth features from `cno_50k_v3_1.h5`
- 50K samples with v3.1 enhanced features
- INITIAL features (16 dims): IC statistics, parameter encoding
- SUMMARY features (18 dims): Global trajectory statistics
- TEMPORAL features (variable): Per-timestep spatiotemporal snapshots

### Architecture

**3-Level Hierarchical VQ-VAE:**

```
Level 1 (INITIAL):  16 dims → Encoder → z_init  → Quantize → 8 tokens
Level 2 (SUMMARY):  18 dims → Encoder → z_summ  → Quantize → 8 tokens
Level 3 (TEMPORAL): T×d dims → Encoder → z_temp → Quantize → 6 tokens (variable per family)

Total: ~22 tokens per trajectory
```

**Codebook Configuration:**
- Adaptive compression ratios per feature family
- Per-family clustering for category discovery
- Entropy regularization for balanced utilization
- Residual vector quantization within families

### Training Objective

```python
L_total = L_recon + β_commit * L_commit + β_entropy * L_entropy

L_recon = MSE(features_reconstructed, features_original)
L_commit = MSE(z_encoder, sg(z_quantized))  # Encoder commitment
L_entropy = -H(codebook_usage)  # Utilization entropy
```

**No physics loss** - VQ-VAE never sees raw PDE trajectories, only extracted features.

### Production Baseline (50K CNO)

**Status**: ✅ Complete

**Results:**
- 8 discovered categories (emergent behavioral clustering)
- L_recon = 0.006 (99.4% reconstruction quality)
- ~22 tokens per trajectory (adaptive per family)
- High codebook utilization (entropy-regularized)

**Config**: `configs/vqvae/50k_baseline.yaml`
**Checkpoint**: `checkpoints/vqvae/50k_baseline/vqvae_best.pt`

### Usage in NOA

```python
# Tokenize a trajectory (from MNO or CNO)
features = extract_features(u_trajectory, params)
tokens, z_quantized = vqvae.encode(features)

# Reconstruct (validate tokenization quality)
features_recon = vqvae.decode(z_quantized)
L_recon = mse(features, features_recon)
```

---

## Component 2: MNO World Model

### Purpose

Provide a sparse high-accuracy physics simulator for NOA perturbation-driven exploration.

### Training Data

**Input**: CNO ground truth trajectories from `cno_50k_v3_1.h5`
- 10K samples (stratified subset)
- Full spatiotemporal fields: u(x,t) ∈ ℝ^(128×128×256)
- Conditioned on (θ, u0): PDE parameters + initial conditions

### Architecture

**U-AFNO with FiLM Conditioning:**
- 227M parameters
- Multi-scale attention in Fourier space
- FiLM (Feature-wise Linear Modulation) for parameter conditioning
- Truncated BPTT: 256 timesteps, 32-step windows

### Training Objective

```python
L_total = L_traj + λ_ic * L_ic

L_traj = MSE(u_pred, u_true)  # Trajectory rollout loss
L_ic = MSE(u_pred[t=0], u0)   # Initial condition reconstruction
```

**Pure physics loss** - no VQ constraints, no token conditioning during training.

**Target**: L_traj < 1.0 (RMSE less than typical field variation ≈ 2.0)

### Production Baseline (10K CNO)

**Status**: 🔄 In progress

**Config**: `configs/noa/10k_baseline.yaml`

**Training Command:**
```bash
spinlock train-meta-operator \
    --config configs/noa/10k_baseline.yaml \
    --verbose
```

### Usage in NOA

**MNO serves as a "world model" for exploration:**

1. **Perturbation-driven exploration**: Sample (θ, u0) near known regions
2. **Cheap simulation**: MNO generates rollout u(x,t)
3. **Hypothesis generation**: "What if we perturb damping coefficient?"
4. **Validation**: Compare MNO output to CNO in explored regions

```python
# Generate MNO rollout for exploration
theta_perturbed = theta_known + delta_theta
u_mno = mno.rollout(theta_perturbed, u0, timesteps=256)

# Tokenize for symbolic reasoning
tokens = vqvae.encode(extract_features(u_mno, theta_perturbed))

# Track surprisal (how different from known behaviors)
surprisal = compute_surprisal(tokens, known_token_distribution)
```

---

## Integration: NOA Cognitive Architecture

### Workflow

```
1. MNO generates (θ, u0) rollouts via perturbation
   ↓
2. Extract features from MNO outputs
   ↓
3. Tokenize with CNO-trained VQ-VAE
   ↓
4. Reason over token sequences (symbolic layer)
   ↓
5. Track surprisal → refine exploration
   ↓
6. Validate with CNO in high-uncertainty regions
```

### MNO + VQ-VAE Composition

**MNO provides:**
- Continuous physics simulation (cheap, sparse, high-accuracy)
- Perturbation-driven exploration
- Hypothesis generation

**VQ-VAE provides:**
- Discrete symbolic representation
- Behavioral category discovery
- Communication/reasoning language

**CNO provides:**
- Ground truth validation
- Surprisal calibration
- Refinement targets

### Exploration Strategy

**Phase 1: Broad Exploration (MNO-driven)**
- Sample (θ, u0) from unexplored parameter space
- Generate MNO rollouts
- Tokenize and cluster behaviors
- Identify high-surprisal regions

**Phase 2: Targeted Validation (CNO-driven)**
- Run CNO in high-surprisal regions
- Compare CNO vs MNO outputs
- Update MNO if systematic errors found
- Refine token distribution

**Phase 3: Symbolic Reasoning (VQ-driven)**
- Analyze token transition patterns
- Discover behavioral rules
- Communicate findings in discrete language

---

## Validation

### Post-Training Validation Plan

After training both VQ-VAE (on CNO) and MNO (on CNO), verify that the CNO-trained tokenizer works well on MNO outputs.

#### Step 1: Generate MNO Rollouts

```bash
# Generate 1K MNO rollouts from validation set
spinlock generate-mno-features \
    --mno-checkpoint checkpoints/mno/10k_baseline/meta_operator_best.pt \
    --config configs/noa/10k_baseline.yaml \
    --output datasets/mno_validation_1k.h5 \
    --n-samples 1000
```

#### Step 2: Tokenize with CNO-Trained VQ-VAE

```python
# Load CNO-trained VQ-VAE
vqvae = load_vqvae("checkpoints/vqvae/50k_baseline/vqvae_best.pt")

# Load MNO-generated features
mno_features = load_features("datasets/mno_validation_1k.h5")

# Tokenize and reconstruct
tokens, z_q = vqvae.encode(mno_features)
recon_features = vqvae.decode(z_q)

# Compute reconstruction error
L_recon_mno = mse(mno_features, recon_features)
print(f"VQ reconstruction on MNO outputs: {L_recon_mno:.6f}")
```

**Success Criterion:** L_recon_mno ≈ 0.006 (comparable to CNO baseline)

#### Step 3: Compare MNO vs CNO Trajectories

```python
# Load CNO ground truth for same (θ, u0) samples
cno_features = load_features("datasets/cno_validation_1k.h5")

# Compare physics fidelity
L_traj = mse(mno_trajectories, cno_trajectories)
print(f"MNO physics fidelity: {L_traj:.4f}")

# Compare feature distributions
kl_div = compute_kl_divergence(mno_features, cno_features)
print(f"Feature distribution KL divergence: {kl_div:.4f}")
```

**Success Criteria:**
- L_traj < 1.0 (high physics fidelity)
- KL divergence < 0.1 (similar feature distributions)

#### Step 4: Token Distribution Analysis

```python
# Tokenize both CNO and MNO features
cno_tokens = vqvae.encode(cno_features)[0]
mno_tokens = vqvae.encode(mno_features)[0]

# Compare token usage distributions
cno_token_hist = compute_token_histogram(cno_tokens)
mno_token_hist = compute_token_histogram(mno_tokens)

# Visualize
plot_token_distributions(cno_token_hist, mno_token_hist)
```

**Success Criterion:** Similar token usage patterns (high overlap in discovered categories)

---

## Training Recipes

### VQ-VAE Training (CNO)

```bash
# 50K CNO baseline (production)
spinlock train-vqvae \
    --config configs/vqvae/50k_baseline.yaml \
    --verbose

# Expected results:
# - Training time: ~8 hours on V100
# - Final L_recon: 0.006
# - Categories discovered: 8
# - Tokens per sample: ~22
```

### MNO Training (CNO)

```bash
# 10K CNO baseline (production)
spinlock train-meta-operator \
    --config configs/noa/10k_baseline.yaml \
    --verbose

# Expected results:
# - Training time: ~24 hours on V100
# - Final L_traj: < 1.0 (target)
# - Parameters: 227M
```

### Parallel Training (Recommended)

Since VQ-VAE and MNO are independent, train them in parallel to save time:

```bash
# Terminal 1: VQ-VAE
spinlock train-vqvae --config configs/vqvae/50k_baseline.yaml --verbose

# Terminal 2: MNO
spinlock train-meta-operator --config configs/noa/10k_baseline.yaml --verbose
```

---

## Comparison to Old 3-Stage Approach

### OLD: Sequential 3-Stage Pipeline

```
Stage 1: Train MNO on CNO (L_traj only)
         ↓
Stage 2: Generate 100K+ MNO rollouts + features
         ↓
Stage 3: Train VQ-VAE on MNO distribution
```

**Philosophy:** "Train tokenizer on simulator's distribution"

**Issues:**
- Sequential dependency (VQ-VAE waits for MNO + feature generation)
- Need to generate 100K+ MNO rollouts before VQ training
- More complex pipeline with extra dataset generation step

### NEW: Independent CNO Training

```
Component 1: Train VQ-VAE on CNO features
Component 2: Train MNO on CNO trajectories
             ↓
      (Train in parallel)
             ↓
   Validate composition post-training
```

**Philosophy:** "Train both on ground truth, validate composition"

**Advantages:**
- Parallel training (faster iteration)
- Simpler pipeline (no intermediate dataset generation)
- Modular validation (each component tested independently)
- Same end result (if MNO achieves high fidelity)

---

## Future Extensions

### Cross-Domain Transfer

Train per-domain on CNO ground truth:
- Domain A: VQ-VAE_A + MNO_A (both CNO-trained)
- Domain B: VQ-VAE_B + MNO_B (both CNO-trained)

Then test token transfer:
- Can VQ-VAE_A tokenize MNO_B outputs?
- Are discovered categories domain-specific or universal?

### Hybrid Exploration

- Use MNO for cheap exploration (high throughput)
- Use CNO for validation (high accuracy)
- Adaptively decide when to invoke expensive CNO based on surprisal

### Online Learning

- Start with frozen CNO-trained components
- Fine-tune VQ-VAE on MNO outputs if distribution shift detected
- Fine-tune MNO on CNO corrections in high-error regions

---

## Key Takeaways

1. **Two Independent Components**: VQ-VAE and MNO both train on CNO ground truth
2. **MNO as World Model**: Not a feature generator, but a sparse high-accuracy simulator
3. **VQ-VAE as Symbolic Layer**: CNO-trained discrete representation used for reasoning
4. **Composition Over Coupling**: Independent training + post-training validation
5. **Assumption**: High-fidelity MNO (L_traj < 1.0) → similar distribution to CNO
6. **Validation**: Verify VQ reconstruction quality on MNO outputs post-training

---

## References

- Production VQ-VAE baseline: `docs/baselines/50k-vqvae-baseline.md`
- MNO training paradigms: `docs/noa-training-paradigms.md`
- Feature extraction: `docs/features/feature-catalog.md`
- Training guide: `docs/noa-training-guide.md`
- NOA roadmap: `docs/noa-roadmap.md`
