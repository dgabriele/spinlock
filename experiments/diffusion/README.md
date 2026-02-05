# Discrete Diffusion for Trajectory Completion (Future Work)

**Status:** Planning Phase
**Created:** 2026-02-05
**Priority:** To be implemented if/when transformer baseline results warrant exploration

---

## Overview

This package contains comprehensive planning for a discrete diffusion-based approach to trajectory completion as an alternative to the transformer implementation in `experiments/trajectory_completion/`.

**Core Concept:** Use D3PM-style discrete diffusion with RePaint-style inpainting to iteratively denoise missing token positions, allowing hierarchical coarse-to-fine structure to emerge naturally over diffusion timesteps.

---

## Why Diffusion?

The trajectory completion task (given 30% start + 20% end tokens, predict middle 50%) is fundamentally an **interpolation** problem, where diffusion models excel:

### Advantages

1. **Inpainting Strength**
   - Keep observed tokens fixed (RePaint-style conditioning)
   - Gradually fill gaps respecting boundary constraints
   - Proven success in image inpainting translates to discrete tokens

2. **Natural Hierarchical Emergence**
   - Early diffusion steps: Coarse (L0) tokens converge
   - Middle steps: Medium (L1) tokens refine
   - Late steps: Fine (L2) details emerge
   - **No manual engineering required** (vs transformer's explicit coarse→fine guidance)

3. **Iterative Refinement Quality**
   - 50 denoising steps vs 1 forward pass (transformer)
   - Each step improves completion coherence
   - Potentially higher quality at cost of speed

4. **Uncertainty Quantification**
   - Stochastic sampling reveals prediction confidence
   - Multiple samples show completion diversity
   - Useful for ambiguous physics scenarios

### Trade-offs

| Aspect | Transformer | Diffusion-50 | Diffusion-20 | Hybrid |
|--------|-------------|--------------|--------------|--------|
| Speed | 10ms | 800ms | 300ms | 310ms |
| Quality (expected) | Good (62%) | Best (65%) | Moderate (60%) | Good (63%) |
| Hierarchy | Manual | Natural | Natural | Both |
| Implementation | Simple | Complex | Complex | Most Complex |

---

## Implementation Plan

See `IMPLEMENTATION_PLAN.md` for comprehensive technical specification including:

- **Architecture:** D3PM discrete diffusion process, transformer denoising network, hierarchical conditioning
- **Experiments:** Baseline (50 steps), Fast (20 steps), Hybrid (transformer L0 + diffusion L1/L2)
- **Roadmap:** 4-phase implementation plan (~2-3 weeks total)
- **Code:** Production-ready code sketches for all components
- **Metrics:** Hierarchical emergence tracking, speed benchmarks, quality comparisons

### Key Components

```
experiments/diffusion/
├── models/
│   ├── discrete_diffusion.py        # Core D3PM process (~300 lines)
│   ├── denoising_network.py         # Transformer denoiser (~250 lines)
│   ├── time_embeddings.py           # Sinusoidal embeddings (~50 lines)
│   └── hybrid_model.py              # Transformer + Diffusion (~200 lines)
├── training/
│   ├── diffusion_trainer.py         # Training loop (~200 lines)
│   └── schedules.py                 # Noise schedules (~80 lines)
├── evaluation/
│   └── hierarchical_metrics.py      # Emergence tracking (~150 lines)
├── configs/
│   ├── baseline_diffusion.yaml      # 50 steps, 6 layers
│   ├── fast_diffusion.yaml          # 20 steps, 8 layers
│   └── hybrid.yaml                  # Transformer (L0) + Diffusion (L1/L2)
└── run_experiment.py                # Main entry point (~150 lines)

Total: ~1,380 lines (comparable to trajectory_completion at ~1,476)
```

---

## When to Implement

**Proceed if:**
1. ✅ Transformer baseline achieves >55% token accuracy (proves concept works)
2. ✅ Transformer shows clear hierarchical token relationships (L0→L1→L2)
3. ✅ Quality ceiling reached (diminishing returns from transformer tuning)
4. ✅ GPU resources available (~4-6 hours training per variant)

**Skip if:**
1. ❌ Transformer fails (<40% accuracy) → fundamental issue with token space
2. ❌ No clear hierarchy in transformer → diffusion won't help
3. ❌ Transformer already exceeds 70% → diminishing returns
4. ❌ GPU time constrained → focus on other priorities

**Current Status:** Waiting for transformer baseline results (experiments/trajectory_completion/)

---

## Expected Performance

Based on recent discrete diffusion work (D3PM, Multinomial Diffusion):

### Quality Improvements (vs Transformer)

- **Token Accuracy:** +2-5% improvement (62% → 65-67%)
- **Reconstruction MSE:** -10% error (0.095 → 0.085)
- **Hierarchical Coherence:** Quantifiable coarse→fine emergence
- **Extreme Masking:** Better performance on 10%+10% case

### Speed Costs

- **Baseline (50 steps):** 80x slower (10ms → 800ms)
- **Fast (20 steps):** 30x slower (10ms → 300ms)
- **Hybrid:** 31x slower (10ms → 310ms) but matches transformer quality

### Best Use Cases

1. **Offline analysis:** Quality matters more than speed
2. **Hierarchical validation:** Test if learned structure is real
3. **Uncertainty quantification:** Need multiple plausible completions
4. **Extreme interpolation:** Very sparse observations (10% total)

---

## Quick Start (When Ready)

### Prerequisites

1. Transformer baseline trained and evaluated
2. Results show >55% token accuracy
3. GPU available for 4-6 hour training runs

### Phase 1: Basic Discrete Diffusion

```bash
# Create baseline diffusion model
poetry run python -m experiments.diffusion.run_experiment \
    --config experiments/diffusion/configs/baseline_diffusion.yaml

# Expected output:
# - Training: ~5 hours (30 epochs)
# - Token accuracy: 60-65% (target: match/beat transformer)
# - Inference: ~800ms per sample (50 steps)
```

### Phase 2: Fast Variant

```bash
# Train faster variant (20 steps)
poetry run python -m experiments.diffusion.run_experiment \
    --config experiments/diffusion/configs/fast_diffusion.yaml

# Expected output:
# - Training: ~6 hours (40 epochs, deeper network)
# - Token accuracy: 58-62% (acceptable drop)
# - Inference: ~300ms per sample (20 steps)
```

### Phase 3: Hybrid Model

```bash
# Train hybrid (transformer L0 + diffusion L1/L2)
poetry run python -m experiments.diffusion.run_experiment \
    --config experiments/diffusion/configs/hybrid.yaml

# Expected output:
# - Training: ~6 hours (25 epochs)
# - Token accuracy: 62-64% (match transformer)
# - Inference: ~310ms (10ms transformer + 300ms diffusion)
```

---

## Technical Highlights

### Discrete Diffusion (D3PM)

**Forward Process:** Gradually corrupt tokens toward uniform categorical distribution
```python
q(x_t | x_{t-1}) = Cat(x_t; p = x_{t-1} Q_t)
where Q_t[i,j] = (1-β_t)δ_ij + β_t/K
```

**Reverse Process:** Learn to denoise using transformer
```python
p_θ(x_{t-1} | x_t, x_obs) = Cat(x_{t-1}; p = f_θ(x_t, t, x_obs))
```

**Inpainting:** Keep observed tokens fixed at each step (RePaint-style)
```python
x_{t-1} = {
    x_obs_noisy[i]  if i is observed
    x_pred[i]       if i is masked
}
```

### Hierarchical Emergence

**Hypothesis:** Diffusion naturally produces coarse→fine structure:
- Step 50 → 40: Random noise → Coarse (L0) structure emerges
- Step 40 → 20: L0 refines → Medium (L1) structure emerges
- Step 20 → 0: L1 refines → Fine (L2) details emerge

**Validation:** Track per-level accuracy over diffusion steps, expect:
```
L0: 80% accuracy reached at step ~35
L1: 80% accuracy reached at step ~25
L2: 80% accuracy reached at step ~10
```

---

## References

### Papers

1. **D3PM** - Austin et al., NeurIPS 2021
   - Discrete denoising diffusion probabilistic models
   - [ArXiv](https://arxiv.org/abs/2107.03006)

2. **Multinomial Diffusion** - Hoogeboom et al., NeurIPS 2021
   - Learning categorical distributions with diffusion
   - [ArXiv](https://arxiv.org/abs/2102.05379)

3. **RePaint** - Lugmayr et al., CVPR 2022
   - Inpainting using denoising diffusion
   - [ArXiv](https://arxiv.org/abs/2201.09865)

4. **ARDM** - Hoogeboom et al., ICLR 2022
   - Autoregressive diffusion models
   - [ArXiv](https://arxiv.org/abs/2110.02037)

### Related Experiments

- **Transformer Baseline:** `experiments/trajectory_completion/`
- **VQ-VAE Training:** `src/spinlock/encoding/models/categorical_vqvae.py`
- **Common Infrastructure:** `experiments/common/`

---

## Decision Framework

Use this flowchart to decide whether to implement diffusion:

```
┌─────────────────────────────────────────┐
│ Transformer baseline results ready?     │
└─────────────┬───────────────────────────┘
              │
              ↓ YES
┌─────────────────────────────────────────┐
│ Token accuracy > 55%?                   │
└─────────────┬───────────────────────────┘
              │
              ↓ YES
┌─────────────────────────────────────────┐
│ Clear hierarchical structure (L0→L1→L2)?│
└─────────────┬───────────────────────────┘
              │
              ↓ YES
┌─────────────────────────────────────────┐
│ Want higher quality (trade speed)?      │
└─────────────┬───────────────────────────┘
              │
              ↓ YES
┌─────────────────────────────────────────┐
│ GPU resources for ~15 hours total?      │
└─────────────┬───────────────────────────┘
              │
              ↓ YES
┌─────────────────────────────────────────┐
│ ✅ PROCEED WITH DIFFUSION               │
│ Start with Phase 1: baseline_diffusion  │
└──────────────────────────────────────────┘

              ↓ NO (at any step)
┌─────────────────────────────────────────┐
│ ❌ DEFER DIFFUSION                      │
│ Focus on other experiments/improvements │
└──────────────────────────────────────────┘
```

---

## Contact

**Questions about this plan:**
- See `IMPLEMENTATION_PLAN.md` for comprehensive technical details
- Agent ID af5ffa6 can be resumed for clarifications/extensions

**When ready to implement:**
- Follow phased roadmap in implementation plan
- Reuse components from `experiments/common/`
- Extend transformer trainer patterns

**If results don't warrant implementation:**
- Archive this planning package as reference
- Document decision rationale in experiments/README
