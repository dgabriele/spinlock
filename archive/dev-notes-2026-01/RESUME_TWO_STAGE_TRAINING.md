# Two-Stage NOA Training: Context & Next Steps

## What We've Been Doing

We've been investigating how to train NOA (Neural Operator Agent) to think and express itself symbolically via VQ-VAE tokens - the "Creative Observer" paradigm.

### Experiments Conducted (All Failed)

1. **VQ-led training** (frozen VQ encoder): Latent loss plateaued at 0.0217
2. **VQ-led with unfrozen encoder**: Latent improved briefly then degraded (0.0223 → 0.0204 → 0.0212)
3. **Feature-space loss** (bypassing encoder): Plateaued around 1.05
4. **Manual weight balancing** (λ_recon=100): Same degradation pattern
5. **Adaptive loss balancing** (EMA normalization): Normalized latent degraded from 0.50 → 0.93

### Root Cause

**Joint optimization of physics (L_traj) and VQ alignment (L_latent) is fundamentally incompatible:**
- VQ-VAE trained on CNO's distribution, NOA's distribution is different
- Even with perfect gradient balancing, latent objective degrades
- Not a magnitude problem - the objectives conflict at a deeper level
- VQ-VAE's latent space may not capture the right features for this task

**Full investigation documented in:** `FINDINGS_VQ_LED_INVESTIGATION.md`

## The New Approach: Two-Stage Training

### Stage 1: Train Precision Meta-Operator
Train NOA purely on physics (trajectory matching) without any VQ involvement:
- Loss: L_traj (MSE vs CNO rollouts)
- Potentially other instrumental regularizers (smoothness, energy conservation, etc.)
- No VQ-VAE, no symbolic objectives, no joint optimization
- Goal: NOA becomes the best possible physics simulator

### Stage 2: Train VQ-VAE on NOA Rollouts
Train VQ-VAE to tokenize NOA's actual distribution:
- Load trained meta-operator from Stage 1
- Generate Sobol-sampled rollouts using NOA (not CNO!)
- Train VQ-VAE on these rollouts
- VQ-VAE learns NOA's manifold, guaranteeing perfect expressibility

### Why This Works

**Decoupled objectives:**
- Physics quality optimized in Stage 1 (no compromises)
- Symbolic representation optimized in Stage 2 (no conflicts)

**No distribution mismatch:**
- VQ-VAE sees NOA's outputs during training and deployment
- Tokenization guaranteed to work (VQ trained on NOA's manifold)

**Simpler to debug:**
- Stage 1: Single metric (MSE vs CNO)
- Stage 2: Single metric (VQ reconstruction on NOA rollouts)

**Same end result:**
- High-quality physics simulator (NOA)
- Fully tokenizable outputs (VQ-VAE aligned to NOA)
- No optimization conflicts!

## Implementation Plan Scope

### What We Need to Build

1. **New CLI target: `train-meta-operator` or `train-noa-stage-1`**
   - Pure MSE-led training (no VQ)
   - Runnable via spinlock CLI
   - Configuration via YAML or CLI args
   - Checkpoint saving/loading

2. **Update existing `train-vqvae` CLI target**
   - Load trained meta-operator checkpoint
   - Generate Sobol-sampled NOA rollouts
   - Train VQ-VAE on NOA rollouts (not CNO rollouts)
   - Decision: 100K samples (same as dataset) or scale to 1M?

3. **Supporting infrastructure**
   - Rollout generation script/utility
   - Sobol sampling coordinator (reuse existing or expand?)
   - VQ-VAE training modifications to accept NOA-generated data

### Open Questions for Planning

1. **Sobol sampling strategy:**
   - Reuse same 100K Sobol samples from original dataset?
   - Expand to larger space (1M samples) for better VQ coverage?
   - Trade-off: More data → better VQ, but longer training time

2. **Instrumental regularizers for Stage 1:**
   - Pure L_traj, or add physics-informed regularizers?
   - Smoothness penalties, energy conservation, PDE constraints?

3. **CLI organization:**
   - Two separate commands (`train-meta-operator`, `train-vqvae`)?
   - Or unified workflow with `--stage 1/2` flag?

4. **Checkpoint coordination:**
   - How to pass Stage 1 checkpoint to Stage 2?
   - Configuration format for linking stages?

## Next Steps

**Enter plan mode** to design:
1. CLI interface for two-stage training
2. Code refactoring required (remove VQ from MSE-led training path)
3. VQ-VAE training modifications (load NOA, generate rollouts, train on NOA distribution)
4. Sobol sampling strategy (100K vs 1M decision)
5. Integration with existing spinlock CLI infrastructure

**Branch strategy:**
- Create new branch: `two-stage-training` or `noa-vqvae-decoupled`
- Commit current state first
- Develop two-stage approach
- Merge to main if successful

**Success criteria:**
- Stage 1: NOA achieves low MSE vs CNO on validation set
- Stage 2: VQ-VAE achieves low reconstruction error on NOA rollouts
- End-to-end: NOA outputs are perfectly tokenizable by VQ-VAE
