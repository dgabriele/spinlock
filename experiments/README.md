# Spinlock Experiments

Experiments demonstrating learned representations and token-based modeling for physics trajectory completion using the VQ-VAE + MNO system.

---

## Overview

This package explores three complementary approaches to trajectory completion: predicting missing portions of physics trajectories using only discrete token representations from the VQ-VAE.

**Core Task:** Given partial observations (e.g., first 30% + last 20% of trajectory tokens), predict the missing middle tokens, then decode back to continuous space.

**Why Tokens?** Trajectories are encoded as ~30-60 discrete tokens with hierarchical structure (coarse/medium/fine) and temporal pyramid organization (multiple time scales). Completing in token space tests whether the learned representation captures meaningful physics structure.

---

## Three Approaches

### 1. Transformer Completion (`trajectory_completion/`)

**Architecture:** BERT-style masked token prediction with hierarchical guidance

**Method:**
- Single forward pass predicts all masked tokens simultaneously
- Bidirectional transformer attention over observed tokens
- Explicit coarse→fine guidance via residual connections
- Per-level token embeddings and output projections

**Key Features:**
- Fast inference (~10ms, 1 forward pass)
- Parallel prediction treats tokens as conditionally independent
- Manual hierarchical structure (coarse L0 influences fine L1, L2)

**Use Case:** Baseline approach proving token completion is feasible

**Experiments:**
- Baseline: 30% start + 20% end masking
- Extreme: 10% start + 10% end (stress test)
- Coarse-only: Only L0 tokens given (hierarchy test)
- Ablations: No hierarchy, random masking

---

### 2. Pyramid Autoregressive (`trajectory_completion/`)

**Architecture:** Multi-pass transformer respecting temporal pyramid structure

**Method:**
- 4 sequential passes predict tokens in pyramid order (coarse → fine)
- Pass 1: Predict p3 (coarsest, 32 timesteps) + initial conditions
- Pass 2: Predict p2 (64 timesteps) conditioned on p3
- Pass 3: Predict p1 (128 timesteps) conditioned on p3 + p2
- Pass 4: Predict p0 (256 timesteps, finest) conditioned on all coarser levels

**Key Features:**
- Models temporal scale dependencies explicitly
- Physically motivated (slow dynamics constrain fast dynamics)
- Natural coarse-to-fine cascade matching multi-scale physics
- Moderate speed (~40ms, 4 forward passes)

**Use Case:** Tests whether temporal pyramid structure is meaningful

**Rationale:** If coarser temporal scales truly constrain finer scales (as in multi-scale physics), autoregressive prediction should outperform parallel by capturing these dependencies.

---

### 3. Discrete Diffusion (`diffusion/`)

**Architecture:** D3PM-style discrete diffusion with RePaint inpainting

**Method:**
- Forward: Gradually corrupt tokens toward uniform categorical noise over T steps
- Reverse: Iteratively denoise from noise → clean tokens over T steps
- Inpainting: Keep observed tokens fixed during reverse process (RePaint-style)
- Each denoising step predicts full token sequence, masked positions updated

**Key Features:**
- Hierarchical emergence: Coarse tokens naturally converge before fine tokens
- Iterative refinement: 50 denoising steps (vs 1 transformer pass)
- Uncertainty quantification via stochastic sampling
- Slower inference (~800ms, 50 forward passes)

**Use Case:** Validates hierarchical structure emergence without manual engineering

**Rationale:** If the VQ-VAE learned meaningful hierarchy, diffusion should naturally produce coarse→fine structure over timesteps without explicit guidance. Early steps refine coarse structure, late steps add fine details.

**Experiments:**
- Baseline diffusion: 50 steps, full quality
- Fast diffusion: 20 steps, speed/quality balance
- Hybrid: Transformer for L0 (coarse) + Diffusion for L1/L2 (fine)

---

## Comparison

| Approach | Passes | Speed | Hierarchy | Temporal Deps | Best For |
|----------|--------|-------|-----------|---------------|----------|
| **Transformer** | 1 | ~10ms | Manual | No | Baseline, speed-critical |
| **Pyramid AR** | 4 | ~40ms | Explicit | Yes (p3→p0) | Quality + interpretability |
| **Diffusion** | 50 | ~800ms | Emergent | Natural | Validation, offline analysis |

**Hierarchy:**
- Manual: Engineered coarse→fine guidance (transformer)
- Explicit: Autoregressive conditioning on coarser levels (pyramid)
- Emergent: Unsupervised emergence over diffusion steps (diffusion)

**Temporal Dependencies:**
- Transformer: Assumes independence (parallel prediction)
- Pyramid: Models pyramid scale dependencies (p3→p2→p1→p0)
- Diffusion: Gradual refinement respects all dependencies

---

## Shared Infrastructure

**`experiments/common/`** - Reusable components:
- Config system: Pydantic schemas + YAML loading
- Model wrappers: TrainedVQVAE, TrainedMNO interfaces
- Base trainer: Training loop, checkpointing, metrics
- Data utilities: Feature loading, trajectory handling

**Design Principles:**
- DRY: Shared infrastructure across experiments
- Clean abstraction: Config, data, models, training separated
- OOP patterns: Inheritance and composition
- Extensibility: Easy to add new experiments

---

## Directory Structure

```
experiments/
├── README.md                    # This overview
├── common/                      # Shared infrastructure
│   ├── config/                  # Pydantic + YAML config system
│   ├── models/                  # VQ-VAE & MNO wrappers
│   ├── training/                # Base trainer
│   └── data/                    # Data loading utilities
│
├── trajectory_completion/       # Transformer & Pyramid AR
│   ├── models/                  # Completion models
│   ├── training/                # Trainers
│   ├── data/                    # Masking & datasets
│   ├── evaluation/              # Metrics & analysis
│   ├── configs/                 # Experiment configs
│   └── README.md                # Detailed docs
│
├── diffusion/                   # Discrete diffusion
│   ├── IMPLEMENTATION_PLAN.md   # Technical specification
│   └── README.md                # Overview & decision framework
│
└── token_coverage/              # Future: Token space analysis
    └── README.md
```

---

## Key Insights

### Why Three Approaches?

Each approach tests different hypotheses about the learned representation:

**Transformer:**
- **Tests:** Can tokens predict other tokens? (feasibility)
- **Validates:** Token space has sufficient information for completion

**Pyramid Autoregressive:**
- **Tests:** Do temporal scales have real dependencies? (structure)
- **Validates:** Coarse scales constrain fine scales (physics-informed)

**Diffusion:**
- **Tests:** Does hierarchy emerge naturally? (learned structure)
- **Validates:** VQ-VAE learned meaningful coarse-to-fine organization

### Complementary Evidence

- If **transformer works** → Token completion is feasible
- If **pyramid improves** → Temporal dependencies are real
- If **diffusion shows emergence** → Hierarchy is fundamental, not engineered

Together, these experiments comprehensively validate the learned representation from multiple perspectives: computational (transformer), physical (pyramid), and structural (diffusion).

---

## References

**VQ-VAE Training:**
- Token structure: `src/spinlock/encoding/models/categorical_vqvae.py`
- Feature extraction: `src/spinlock/noa/generation_pipeline.py`

**MNO Training:**
- Trajectory generation: `src/spinlock/noa/backbone.py`
- Training infrastructure: `src/spinlock/cli/train_meta_operator.py`

**Checkpoints:**
- VQ-VAE: `checkpoints/vqvae/50k_baseline/best_model.pt`
- MNO: `checkpoints/mno/50k_baseline/meta_operator_best.pt`
- Dataset: `datasets/50k_baseline.h5`
