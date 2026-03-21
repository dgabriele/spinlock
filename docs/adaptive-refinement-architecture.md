# Adaptive Guided Refinement for D3PM Inverse Generation

## Theoretical Framing

The D3PM refinement pipeline solves an **inverse problem**: given observed temporal dynamics of a Lenia configuration, infer the physics parameters (theta) and initial conditions (IC) that produced them. The D3PM acts as an **approximate posterior** q(theta, IC | temporal), learned by training on the forward mapping.

The adaptive refinement extends this with a two-level search:
1. **Token-space exploration**: Multiple D3PM denoising trajectories sample from the learned posterior. Different stochastic paths through the reverse diffusion produce different (theta, IC) proposals.
2. **Physics-space exploration**: Local perturbation around the D3PM's best decoded proposal, separately in both parameter spaces:
   - **Theta** (34 dims): Clipped Gaussian perturbation in [0,1]^34 Sobol space (`LocalParameterPerturber`). Categorical dimensions (kernel_type, growth_type) are frozen.
   - **IC** (36 dims): Sobol quasi-random offsets in [0,1]^36 Fourier parameter space (`FourierICPerturber`). The D3PM's decoded IC grids are inverted via 2D FFT to recover (amplitude, orientation, phase) per mode per channel, normalized to [0,1], then perturbed using low-discrepancy Sobol offsets. Frequencies are recomputed from the perturbed theta's kernel radii, maintaining theta-IC coupling.

This hybrid approach is motivated by the observation that the D3PM's proposal may be *close* to a valid solution in parameter space, even when it doesn't pass the discrete token-level agreement threshold. Using Sobol offsets (rather than Gaussian noise) for IC perturbation gives better coverage of the 36-dimensional neighborhood — in high dimensions, Gaussian samples concentrate in a thin shell near ||x||=sqrt(D), while Sobol fills the local region uniformly.

## Causal Structure

The inverse problem has a specific causal structure:

```
theta --> IC (via FourierICGenerator, radii coupling)
theta + IC --> temporal dynamics (via Lenia simulator)
temporal dynamics --> temporal tokens (via VQTokenizer)
```

The D3PM learns to invert this chain: temporal tokens --> (theta, IC) tokens. The difficulty varies by sample because:
- Some (theta, IC) pairs produce distinctive dynamics (easy to invert)
- Others produce dynamics similar to many parameter configurations (hard, many-to-many)
- The tokenization quantizes the parameter space, so nearby thetas may map to different tokens

## Compute Scaling with Difficulty

The budget allocation is difficulty-proportional:

```
distance = clamp(1 - agreement / threshold, 0, 1)
total_extra = min_extra + ceil(distance * (max_extra - min_extra))
```

| Agreement | Distance | Total Extra | Cost vs Fixed-4 |
|-----------|----------|-------------|-----------------|
| >= 0.80   | 0.0      | 0           | 0.5x (2 initial vs 4 fixed) |
| 0.70      | 0.125    | 4           | 1.5x |
| 0.50      | 0.375    | 7           | 2.25x |
| 0.00      | 1.0      | 16          | 4.5x |

The initial D3PM round uses 2 candidates (vs 4 in the fixed approach). For the ~70-80% of samples that pass on first try, this halves compute. The extra budget is spent only on hard samples.

Each refinement round consists of:
- 1 D3PM re-sample (token-space diversity)
- N perturbation candidates (physics-space diversity, default N=4), with **joint theta+IC perturbation**:
  - Theta: Gaussian offsets in [0,1]^34, categorical dims frozen
  - IC: Sobol offsets in [0,1]^36 (Fourier params extracted from decoded grids via FFT)
  - Frequencies recomputed from perturbed radii (theta-IC coupling preserved)
- Progressive sigma widening: 0.03 -> 0.045 -> 0.0675 -> 0.10 (same sigma for both theta and IC)

Categorical dimensions (kernel_type at dim 32, growth_type at dim 33) are frozen during theta perturbation because small noise can flip the category entirely, producing a fundamentally different system rather than a nearby one.

## Data Flow

```
For each novel Sobol sample in a batch:

  PHASE 1 — GT Generation:
    Sobol [B,34] --> sobol_batch_to_tensors --> physical params
    FourierICGenerator(radii) --> ICs [B,C,H,W]
    RolloutProvider.rollout --> trajectories [B,T+1,C,H,W]
    VQTokenizer.tokenize --> GT tokens (all 160 keys)
    TokenFilter.contract --> GT active tokens (102 keys)

  PHASE 2 — Initial D3PM (cheap path):
    D3PM.sample(observed=temporal, mask=theta+IC) x 2 candidates
    For each: expand --> decode --> rollout --> retokenize --> score
    Accept if agreement >= 0.8 (most samples stop here)

  PHASE 3 — Adaptive Refinement (hard samples only):
    CandidateBudgetAllocator.allocate(agreements) --> per-sample budgets

    For each round (up to 4, sigma widening):
      D3PM re-sample: 1 additional denoising trajectory
        --> evaluate as Phase 2

      Joint theta+IC perturbation:
        1. Decode best tokens --> theta [R', 34] + IC grids [R', C, H, W]
        2. Theta: LocalParameterPerturber.perturb(sigma)
           --> [R'*P, 34] perturbed thetas (Gaussian, categorical frozen)
        3. sobol_batch_to_tensors --> physical params (radii for IC freqs)
        4. IC: FourierICPerturber.perturb_from_decoded(grids, radii, sigma)
           --> FFT extract (A, theta, phi) from decoded grids [R', C, K]
           --> normalize to [0,1]^36
           --> Sobol quasi-random offsets centered on each point [R'*P, 36]
           --> de-normalize, reconstruct with perturbed radii [R'*P, C, H, W]
        5. Rollout from (perturbed theta, perturbed IC)
           --> tokenize --> score against GT temporal
           --> update best if improved

      Stop sample if accepted or budget exhausted
      sigma *= 1.5

  PHASE 4 — Collect Results:
    All accepted (initial + recovered) become hard targets
    Hard target = {GT temporal tokens (observed), best theta+IC tokens (target)}
```

## Dynamic Stopping

Three levels of stopping prevent wasted compute:

1. **Per-sample**: Stop refinement when accepted or budget exhausted (max 4 rounds)
2. **Per-cycle**: Stop generating proposals when acceptance rate drops below 5% over last 200 proposals (within-cycle comparison is valid since samples are from the same Sobol stream)
3. **Eval-based early stopping** (v11): Stop refinement cycles when held-out eval agreement doesn't improve for `early_stopping_patience` cycles. When triggered, the best cycle's checkpoint is restored. This replaces the v9 loss-based cross-cycle criterion, which was unreliable because each cycle draws different Sobol samples with different difficulty.

## v10 Post-Mortem: GT-Token Training

### Experiment
v10 trained the D3PM directly on ground-truth tokens (`train_on_gt: true`), bypassing D3PM inference and quality filtering entirely. Every novel Sobol sample became a training target with perfect agreement.

### Results

| Metric | Baseline | Cycle 1 | Cycle 2 | Cycle 3 |
|--------|----------|---------|---------|---------|
| Mean agreement | 0.7452 | 0.7198 | 0.7041 | 0.6949 |

Agreement **degraded monotonically** over 3 cycles.

### Root Cause Analysis

1. **Distribution mismatch**: GT targets are teacher-forced — the model sees perfect conditioning tokens during training. At inference, it conditions on its own noisy predictions. This exposure bias means improvements on GT targets don't transfer to on-policy performance.
2. **Catastrophic forgetting**: 1000 novel samples per cycle, no replay buffer, fresh optimizer each cycle, no weight constraints. The model forgets v8's learned inverse mapping.
3. **No LR scheduling**: Static 2e-5 with no warmup or decay. Fine-tuning a converged model at full learning rate causes destructive updates.

### Key Takeaway
On-policy training (D3PM's own roundtrip-verified completions) is essential. The quality filter ensures training targets match inference-time distribution.

## v11 Stabilization

v11 returns to v9's on-policy approach with four stabilization mechanisms:

### Weight Anchoring
L2 penalty toward v8 base checkpoint weights:
```
anchor_loss = sum((p - p_anchor)^2 for all params)
loss = ce_loss + anchor_weight * anchor_loss
```
Prevents catastrophic drift while still allowing adaptation. Anchors are captured **once** after loading the initial checkpoint and remain fixed across all cycles.

### Replay Buffer
Reservoir sampling accumulates on-policy targets across cycles:
- Each cycle's accepted targets are added to the buffer
- Training mixes current targets with replay samples: `replay_fraction` controls the ratio
- Reservoir sampling ensures uniform coverage when buffer is full (`max_replay_size`)
- Buffer state is persisted in checkpoints for `--resume`

Expected replay growth: cycle 1 = 0 replay, cycle 2 = ~300, cycle 3 = ~600 (at `replay_fraction=0.3`)

### LR Scheduling
Two-level schedule:
- **Within-cycle**: Cosine decay with linear warmup (`use_cosine_schedule`, `warmup_fraction`)
- **Across-cycle**: Multiplicative decay (`per_cycle_lr_decay`), e.g., 2e-5 → 1.8e-5 → 1.62e-5 at decay=0.9

### Eval-Based Early Stopping
Patience mechanism with best-checkpoint recovery:
- After each eval, compare agreement to best seen
- If no improvement for `early_stopping_patience` cycles → stop, load best checkpoint
- State (best agreement, patience counter, best cycle) persisted for `--resume`

### Updated Data Flow
```
For each cycle:
  1. Generate on-policy hard targets (same as v9)
  2. Add targets to replay buffer (reservoir sampling)
  3. Mix: current targets + replay_buffer.sample(n_replay)
  4. Fine-tune with:
     - Effective LR = base_lr * per_cycle_lr_decay^cycle
     - Cosine schedule with warmup within cycle
     - CE loss + anchor_weight * L2(params - anchor_params)
  5. Evaluate on held-out set
  6. Early stopping check → break if patience exhausted
```

## v12 Surprise-Driven Refinement

v11 peaked at cycle 4 (agreement 0.7452→0.7698, +3.3%) then degraded through cycle 7 (0.7238), triggering early stopping. Root cause: ~85% acceptance rate means most training targets are uninformative — easy examples dilute the learning signal from the ~15% of hard targets the model actually needs to learn from.

v12 addresses signal dilution with a **three-level surprise hierarchy**:

```
Level 1 — REPLAY SELECTION (coarsest)
  PrioritizedReplayBuffer.sample(): P(i) ∝ (1 - agreement_i)^α + ε
  Signal: agreement-surprise (static, objective-aligned)
  Controls WHICH past targets appear in the training batch

Level 2 — SAMPLE WEIGHTING (per-sample)
  sample_weight_i = √(agreement_surprise_i × loss_surprise_i)
  Signal: geometric mean of static + dynamic surprise
  Controls HOW MUCH gradient each sample contributes

Level 3 — POSITION WEIGHTING (finest, unchanged)
  focal_weight = (1 - p_correct)^γ
  Signal: per-position model confidence (online)
  Controls gradient allocation WITHIN each sample across positions
```

### Why geometric mean for Level 2

The geometric mean `√(a × b)` requires both signals to agree for high weight:
- Both hard (high agreement-surprise × high loss-surprise): highest weight
- Agreement-hard, loss-easy: moderate — model learned this, less useful
- Agreement-easy, loss-hard: moderate — denoiser struggling at this timestep
- Both easy: lowest weight — uninformative

This naturally handles stale priorities: as the model learns a hard sample, its CE loss drops, pulling the combined weight down even though agreement stays fixed.

### Literature grounding

- **Prioritized Generative Replay** (ICLR 2025 oral): curiosity-based relevance for replay
- **SuRe** (2025): surprise-driven prioritised replay with dual-learner (fast + slow)
- **OHEM / Focal Loss lineage**: per-sample hard example selection + per-position reshaping

### v12 Updated Data Flow
```
For each cycle:
  1. Generate on-policy hard targets (same as v9/v11)
  2. Add targets to PrioritizedReplayBuffer (reservoir sampling)
  3. Mix: current targets + buffer.sample(n_replay)
     Replay sampling: P(i) ∝ (1 - agreement_i)^α  [Level 1]
  4. Fine-tune with:
     - Per-sample loss [B] from focal CE  [Level 3]
     - Surprise weights = √(agreement_surprise × loss_surprise)  [Level 2]
     - Weighted CE + anchor_weight * L2(params - anchor)
     - Cosine schedule with warmup, per-cycle decay
  5. Evaluate on held-out set
  6. Early stopping check → break if patience exhausted
```

## Module Structure

```
src/spinlock/experimental/diffusion/refinement/
    __init__.py                 -- Module exports
    adaptive_search.py          -- AdaptiveRefinementSearch (main orchestrator)
    candidate_budget.py         -- CandidateBudgetAllocator
    ic_perturber.py             -- FourierICPerturber (Sobol in Fourier param space)
    local_perturber.py          -- LocalParameterPerturber (Gaussian in Sobol space)
    replay_buffer.py            -- PrioritizedReplayBuffer (v12)

experiments/diffusion/scripts/
    refine_d3pm.py              -- Surprise-weighted fine-tuning loop

experiments/diffusion/configs/
    v9_adaptive_refinement.yaml  -- On-policy, no stabilization
    v10_gt_refinement.yaml       -- GT-token training (FAILED)
    v11_stabilized_refinement.yaml -- On-policy + stabilization
    v12_surprise_refinement.yaml -- Surprise-driven (current)
```

## Configuration

The adaptive refinement is controlled by `AdaptiveRefinementConfig` in `RefinementConfig.adaptive`, with sensible defaults that make it backward-compatible with v8 configs (the adaptive field defaults are applied automatically).

Key tuning knobs:
- `initial_d3pm_candidates`: Initial cheap round (default 2, was 4 fixed)
- `perturbation.initial_sigma`: Starting perturbation radius (0.03 = 3% of [0,1] range)
- `budget.max_extra_candidates`: Max extra candidates for hardest samples (16)
- `fine_tuning.anchor_weight`: L2 anchor strength (0.05, 0 = disabled)
- `fine_tuning.replay_fraction`: Replay mixing ratio (0.3, 0 = disabled)
- `fine_tuning.per_cycle_lr_decay`: Cross-cycle LR decay (0.9, 1.0 = disabled)
- `fine_tuning.surprise_alpha`: Replay priority exponent (1.0, 0 = uniform)
- `fine_tuning.max_surprise_weight`: Per-sample weight cap (5.0)
- `early_stopping_patience`: Eval-based stopping (3 cycles, 0 = disabled)
