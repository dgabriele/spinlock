# Spinlock System Overview

A guide for new developers and agents joining the project. Covers what the system does, why it's built this way, and how the pieces fit together — from continuous physics to discrete tokens to generative imagination to emergent communication.

---

## What is Spinlock?

Spinlock is a **domain-agnostic framework** for training neural agents that can **perceive**, **imagine**, and eventually **communicate about** the behavior of complex dynamical systems.

The framework accepts any system that produces spatiotemporal trajectories from configurable parameters. The current proof-of-concept domain is [Lenia](https://chakazul.github.io/lenia.html) — a continuous cellular automaton with rich, life-like dynamics emerging from a 34-dimensional parameter space. Lenia examples are used throughout this document, but the pipeline (dataset generation → VQ tokenization → D3PM training → refinement) is parameterized by YAML configs and makes no domain-specific assumptions in its core modules.

The core pipeline has three stages:

```
Physical system        →   Discrete tokens        →   Generative model
(Lenia rollouts)           (VQ tokenization)           (D3PM diffusion)
                                                           ↓
                                                    Agent perception
                                                    & imagination
                                                           ↓
                                                    Emergent language
                                                    (LFM, future)
```

## Why discrete tokens?

A rollout from a spatiotemporal dynamical system is a high-dimensional tensor — for Lenia, `[T, C, H, W]` (hundreds of timesteps of a 3-channel 128x128 grid). You can't learn a joint distribution over raw state. And even if you could, the agent needs to *reason* about what it sees, not just reproduce it.

The VQ tokenizer compresses each rollout into ~160 discrete token indices organized into three families:

| Family | What it encodes | Example |
|--------|----------------|---------|
| **Temporal** (~60 tokens) | How the system evolves over time | 30 groups of CNN-extracted features, each quantized at 2 hierarchy levels |
| **Initial** (~32 tokens) | The spatial initial condition | 16 spatial patches, each quantized at 2 levels |
| **Theta** (~68 tokens) | The operator parameters | 34 physical parameters (kernel radii, growth rates, coupling), each quantized at 2 levels |

These tokens are **grounded**: the tokenizer guarantees ~95% roundtrip consistency (tokenize → decode → re-tokenize = same tokens). They're discrete, compositional, and small enough to learn a joint distribution over.

### Why three families?

Because the physical system has three distinct causes of what you observe:

1. **Theta** determines the rules — what kernel shape, what growth function, how channels couple. Two different thetas produce fundamentally different physics.
2. **Initial conditions** determine the starting state — the spatial pattern of channel activations at t=0. Same theta + different IC = same rules, different trajectory.
3. **Temporal dynamics** are the *result* — what actually happens when theta acts on IC over time.

The causal chain is: **theta → IC → dynamics**. This matters deeply for how the generative model is trained.

## The D3PM: learning the joint distribution

The D3PM (Discrete Denoising Diffusion Probabilistic Model) learns P(temporal, initial, theta) — the joint distribution over all ~160 token positions simultaneously. This is fundamentally different from autoregressive models (which generate left-to-right) because all positions are predicted in parallel, conditioned on each other through iterative refinement.

### How diffusion works (intuition)

Imagine a completed jigsaw puzzle (the clean tokens). Diffusion training works in two directions:

**Forward process** (corruption): Randomly replace puzzle pieces with blank pieces. At t=0, the puzzle is complete. At t=T, almost everything is blank.

**Reverse process** (denoising): A neural network learns to fill in the blanks. Given a partially-blanked puzzle at noise level t, predict what the original pieces were. The network sees which pieces remain and uses their pattern to infer the missing ones.

At inference time, start from an all-blank puzzle and iteratively fill in pieces, using each round's predictions as context for the next round.

### Absorbing transitions

Spinlock uses **absorbing** diffusion: tokens are either their true value or a special MASK token. This is cleaner than uniform noise (where tokens randomly swap to other valid codes) because the denoiser always knows which positions are real information vs. masked — there's no ambiguity about what's noise.

### The denoising network

The denoiser is a Transformer operating on the ~160 token positions as a sequence:

1. Each position gets its own learned embedding table (vocab sizes vary from 6 to 28 per quantizer)
2. Sinusoidal timestep embeddings tell the network what noise level it's operating at
3. Hierarchical guidance: L0 (coarse) token representations are broadcast to all positions as a structural prior
4. Standard Transformer self-attention with pre-norm, 8 layers, 8 heads, 256 hidden dim
5. Per-position output heads project back to each position's vocabulary

The Transformer architecture is natural here because every token position can attend to every other — temporal tokens see theta tokens, IC tokens see temporal tokens, etc. The attention pattern encodes the cross-family dependencies that define the physical system.

## The graded noise schedule: why it matters

This is the key architectural innovation. Standard diffusion applies uniform noise to all positions. Spinlock applies **graded noise** that encodes the causal hierarchy of the physical system.

### The problem

At global noise level t=30 (out of 50), a standard D3PM has corrupted ~60% of all positions uniformly. The denoiser must simultaneously predict theta tokens (the *cause*) and temporal tokens (the *effect*) from the same partial information. But temporal tokens are only predictable once theta is known — asking the network to predict effects before causes is asking it to solve an ill-posed problem.

### The solution

Scale factors assign each token family a fraction of the global noise:

```
theta:     scale = 0.15  →  at global t=30, theta sees effective t=4  (nearly clean)
IC:        scale = 0.25  →  at global t=30, IC sees effective t=8     (mostly clean)
temporal:  scale = 0.3-1.0 → at global t=30, temporal sees t=9 to t=30 (noisy)
```

This means theta resolves first during denoising, then IC, then short-horizon temporal, then long-horizon temporal. **The denoising trajectory mirrors physical causality**: parameters → initial conditions → early dynamics → late dynamics.

### Why this enables perception

An agent reading intermediate denoising states gets coarse-to-fine perceptions:

- Early in denoising: "This is a system with these kernel parameters" (theta resolved, everything else uncertain)
- Midway: "Starting from this initial condition, early dynamics look like this" (IC + short temporal resolved)
- Late: "The full 512-step trajectory unfolds like this" (everything resolved)

The agent doesn't need to wait for full denoising to act — it can make decisions from partial information, just like a human scientist who first identifies the type of system, then examines initial conditions, then watches the dynamics unfold.

## Training: the 3-stage curriculum

Total training: 20 epochs across 3 stages with different masking strategies.

### Stage 1: Joint Random (8 epochs)

Every position has a 50% chance of being masked (target) or observed (context), independently. The denoiser learns basic token co-occurrence statistics: which theta values tend to co-occur with which temporal patterns, how IC tokens correlate with early dynamics, etc. This is the broadest learning phase.

### Stage 2: Inverse Generation (8 epochs)

**Temporal tokens are always observed. Theta and IC tokens are always masked.** The denoiser must infer parameters from dynamics — this is the inverse problem. Given what the system did, what could have caused it?

This is the core capability for agent imagination. The masking forces the denoiser to develop a genuine understanding of the dynamics→parameters mapping, not just memorize token patterns.

### Stage 3: Consolidation (4 epochs)

Back to random 50% masking at a lower learning rate. This prevents the denoiser from overfitting to the inverse-generation direction and ensures it retains general-purpose completion ability.

## Loss functions

### Primary: focal cross-entropy

Standard cross-entropy on target positions, with focal weighting (γ=2.0) that down-weights easy predictions. This is critical because rollouts from similar parameter regions share 70-80% of tokens — without focal loss, the denoiser would coast on predicting the shared majority and ignore the 20-30% that actually distinguish different dynamical regimes.

### Roundtrip consistency loss

The primary CE operates purely in token space — it doesn't know that token 5 and token 6 might decode to nearly identical physics, or that token 5 and token 20 decode to completely different dynamics.

The roundtrip loss threads through the frozen VQ pipeline:

```
D3PM predicted logits
  → soft-decode through VQ codebooks (differentiable)
  → frozen decoder → frozen temporal inverse → frozen re-encoder
  → frozen quantizer distances → roundtrip logits
  → CE against ground-truth tokens at a noise-matched truncation level
```

**Multi-truncation matching**: At high noise, compare against short-truncation GT (T=32 — only early dynamics are predictable). At low noise, compare against full-truncation GT (T=512). The VQ tokenizer is trained with variable-length truncation bins [32, 64, 128, 256, 512], and the pretokenized dataset stores tokens at all 5 levels.

**Soft Jaccard coherence**: Beyond per-position accuracy, measures whether the overall *set of codes used* matches the ground truth's code usage pattern. This catches cases where every position is slightly wrong but the aggregate code distribution has drifted.

### Trajectory probe (validation only)

During validation, runs conditional sampling with snapshot recording at 4 denoising steps (80%, 60%, 40%, 20% of T). Measures agreement between each snapshot and the nearest truncation-level ground truth. Logged as `probe_t{step}_agree` — a direct measure of whether the denoising trajectory is recapitulating the physical temporal unfolding.

## File structure

### Core pipeline

| File | Role |
|------|------|
| `src/spinlock/experimental/diffusion/models/discrete_d3pm.py` | D3PM forward/reverse process, graded schedule, `sample()` with snapshot recording |
| `src/spinlock/experimental/diffusion/models/denoising_network.py` | Transformer denoiser with hierarchical guidance |
| `src/spinlock/experimental/diffusion/training/diffusion_trainer.py` | Training loop, validation, trajectory probe |
| `src/spinlock/experimental/diffusion/training/curriculum_trainer.py` | Multi-stage curriculum (inherits DiffusionTrainer) |
| `src/spinlock/experimental/diffusion/training/roundtrip_loss.py` | Roundtrip consistency loss + soft Jaccard |
| `src/spinlock/experimental/diffusion/training/physics_loss.py` | Physics-aware auxiliary loss (soft-decode to continuous params) |
| `src/spinlock/experimental/diffusion/config.py` | All Pydantic config models |

### Data pipeline

| File | Role |
|------|------|
| `src/spinlock/experimental/diffusion/data/pretokenized_dataset.py` | Loads pretokenized HDF5, aux truncation stores |
| `src/spinlock/experimental/diffusion/data/completion_dataset.py` | On-the-fly tokenization dataset, `collate_dict_batch` |
| `src/spinlock/experimental/diffusion/data/hierarchical_masking.py` | Masking strategies with family overrides |
| `src/spinlock/tokens/pretokenized_store.py` | Low-level HDF5 token loading, truncation key remapping |

### VQ tokenizer (upstream)

| File | Role |
|------|------|
| `src/spinlock/tokens/model.py` | VQTokenizerModel — encoder, quantizers, decoder, inverse heads |
| `src/spinlock/tokens/trainer.py` | VQ training loop with roundtrip consistency |
| `src/spinlock/tokens/tokenizer.py` | VQTokenizer — high-level tokenize/detokenize API |
| `src/spinlock/lenia/replayer.py` | Lenia simulator (CFL-adaptive substeps) |
| `src/spinlock/lenia/replay_adapter.py` | Adapter: Lenia rollouts → VQTokenizer-compatible features |

### Refinement (offline quality improvement)

| File | Role |
|------|------|
| `src/spinlock/experimental/diffusion/refinement/adaptive_search.py` | D3PM → rollout → retokenize → quality filter → fine-tune loop |
| `src/spinlock/experimental/diffusion/refinement/replay_buffer.py` | Surprise-weighted replay for fine-tuning |
| `experiments/diffusion/scripts/refine_d3pm.py` | Refinement entry point script |

### CLI and scripts

| File | Role |
|------|------|
| `src/spinlock/cli/train_diffusion.py` | `spinlock train-diffusion` CLI command |
| `src/spinlock/cli/pretokenize_dataset.py` | `spinlock tokenize-dataset` with `--temporal-resolution` |
| `experiments/diffusion/scripts/train.py` | Training script (loaded by CLI) |
| `experiments/diffusion/scripts/calibrate_trajectory.py` | Empirical noise-boundary calibration |
| `experiments/diffusion/scripts/compute_position_scales.py` | Per-position scale factors from cross-truncation divergence |

## Config structure

Training is configured via YAML. A complete config (e.g., `experiments/diffusion/configs/v13_trajectory_50k.yaml`) specifies:

```yaml
dataset:
  tokenized_path: "..."          # Pretokenized HDF5
  tokenizer_checkpoint: "..."    # VQ checkpoint (for vocab sizes + roundtrip head)
  truncation_length: 512         # Primary truncation
  aux_truncation_lengths: [32, 64, 128, 256]  # For roundtrip loss

diffusion:
  num_timesteps: 50
  transition_type: "absorbing"
  graded_schedule:
    enabled: true
    family_scale_overrides: {theta: 0.15, initial: 0.25}

model:
  hidden_dim: 256
  num_layers: 8
  num_heads: 8

training:
  focal_gamma: 2.0
  roundtrip_loss:
    enabled: true
    weight: 0.1
    set_coherence_weight: 0.05
    trajectory_probe_frequency: 1

curriculum:
  stages:
    - name: "joint_random"       # 8 epochs, random 50%
    - name: "inverse_generation" # 8 epochs, temporal observed, theta+IC masked
    - name: "consolidation"      # 4 epochs, random 50%, low LR
```

## Running training

```bash
# Tokenize dataset (one-time, ~20 hours for 300K samples with temporal resolution)
poetry run spinlock tokenize-dataset \
    --dataset datasets/ds_lenia_fourier_50k_perturbed.h5 \
    --tokenizer checkpoints/lenia/vq/v3_fourier_50k_perturbed/vq_tokenizer_latest.pt \
    --output datasets/ds_lenia_fourier_50k_perturbed_pretokenized.h5 \
    --temporal-resolution --device cuda

# Train D3PM
poetry run spinlock train-diffusion \
    --config experiments/diffusion/configs/v13_trajectory_50k.yaml

# Resume from checkpoint
poetry run spinlock train-diffusion \
    --config experiments/diffusion/configs/v13_trajectory_50k.yaml \
    --resume experiments/diffusion/results/v13_trajectory/v13_trajectory_d3pm_best.pt
```

## What comes next

The D3PM produces discrete token sets that describe dynamical system rollouts. An agent equipped with this model can:

1. **Perceive**: Observe a rollout → tokenize → read off what happened
2. **Imagine**: Observe dynamics (temporal tokens) → inpaint theta + IC → "what could have caused this?"
3. **Explore**: Sample unconditionally → discover novel parameter configurations

These capabilities are domain-agnostic — the same pipeline applies to any system where the VQ tokenizer has been trained, whether that's Lenia, fluid dynamics, reaction-diffusion systems, or biological networks.

The missing piece is **communication**. The agent's internal representations (token sets, denoising trajectories) need to be externalized as structured language — not English (which would impose human ontology), but a new language with its own morphology and syntax, translatable by a pretrained multilingual LLM.

This is the role of [LFM (Language Faculty Model)](https://github.com/dgabriele/lfm). The graded noise schedule's causal hierarchy maps directly onto information structure in the emergent language: what resolves first (theta) becomes background information; what resolves last (long-horizon temporal) becomes focus. The denoising trajectory is the agent's inner monologue.

## Further reading

- [D3PM Architecture and Training Dynamics](d3pm-architecture.md) — Deep technical reference for every component
- [Why the Language-First Approach](https://github.com/dgabriele/lfm/blob/main/docs/why-language-first.md) — Why emergent language, not math or alignment
- [Adaptive Refinement Architecture](adaptive-refinement-architecture.md) — Offline quality improvement loop
