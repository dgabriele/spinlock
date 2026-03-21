# D3PM Architecture and Training Dynamics

A technical introduction to the discrete diffusion model at the core of Spinlock's inverse generation pipeline: what it operates on, why the graded noise schedule encodes physical causality, how roundtrip consistency connects discrete predictions back to continuous dynamics, and what happens inside each training step.

---

## What the D3PM operates on

A single training sample is a dictionary of ~160 discrete token indices, organized into three families:

| Family | Tokens | Encodes |
|--------|--------|---------|
| **Temporal** | ~60 | Dynamics of a Lenia rollout (30 groups x 2 hierarchy levels) |
| **Initial** | ~32 | Spatial initial condition (16 spatial patches x 2 levels) |
| **Theta** | ~68 | Operator parameters (34 Sobol dimensions x 2 levels) |

Each token is an index into a learned VQ codebook with variable vocabulary size per quantizer (L0: ~28 codes, L1: ~11-13 codes). These tokens were produced by the VQTokenizer, which guarantees roundtrip consistency: tokenize a rollout, decode the tokens, re-tokenize — and you get the same tokens back with ~95% agreement.

The D3PM's job is to learn the joint distribution P(temporal, initial, theta) over all ~160 positions simultaneously. This is what enables inverse generation: observe temporal tokens (what the system did) → sample theta and IC tokens (what could have produced it).

## The forward diffusion process

D3PM is a discrete analog of continuous diffusion models. Instead of adding Gaussian noise to continuous values, it randomly corrupts discrete tokens according to transition matrices.

**Absorbing transitions**: At each diffusion step, each token either stays the same (with probability 1-β_t) or transitions to a special MASK token (with probability β_t). At t=0, tokens are clean. At t=T, most tokens are masked — the denoiser sees almost nothing.

The noise schedule β_t follows a cosine curve (Nichol & Dhariwal, 2021), with 50 total timesteps. The cosine schedule concentrates most of the information loss in the middle range, giving the denoiser more signal at both extremes.

## The graded noise schedule: encoding causal hierarchy

This is the architectural innovation that makes the D3PM physically meaningful rather than just a token completion model.

### The problem with uniform noise

In a standard D3PM, all token positions see the same noise level at the same global timestep t. But the physical system has causal structure:

- **Theta** (operator parameters) determines the kernel, growth function, and coupling — it is the *cause* of everything.
- **Initial conditions** set the starting state — they interact with theta to determine early dynamics.
- **Temporal tokens** encode the *result* of theta + IC playing out over time. Short-horizon temporal tokens (early dynamics) are largely determined by IC and theta. Long-horizon temporal tokens (late dynamics) are the most sensitive to everything.

Applying uniform noise ignores this hierarchy. At any given noise level, the denoiser would need to simultaneously resolve cause (theta) and effect (long-horizon temporal) — but the effect is only predictable once the cause is known.

### How the graded schedule works

The graded schedule maps each global timestep t to a per-position *effective* timestep:

```
effective_t(key, global_t) = clamp(round(global_t × scale_factor[key]), 0, T-1)
```

Scale factors are assigned in a 3-tier resolution:

1. **Per-key** (highest priority): Computed from cross-truncation token divergence by `compute_position_scales.py`. Positions whose tokens change between T=32 and T=512 truncations get high scale factors (they encode late-resolving information). Positions that stay stable get low factors.

2. **Per-family**: `theta: 0.15`, `initial: 0.25`. These encode the causal hierarchy directly — theta gets 15% of the global noise, IC gets 25%.

3. **Global fallback**: `non_temporal_scale: 0.3` for any unmatched position.

### What this means during denoising (t=T→0)

At **high t** (early in denoising): global noise is high, but theta's effective noise is only 15% of that — theta is already nearly clean. The denoiser can predict theta accurately while temporal positions are still mostly masked.

As **t decreases**: IC resolves next (scale 0.25), then short-horizon temporal, then long-horizon temporal (scale ~0.7-1.0).

At **low t** (late in denoising): everything is nearly resolved. The denoiser makes fine-grained adjustments to the most sensitive long-horizon temporal positions, conditioned on already-resolved theta and IC.

This mirrors how physical causality unfolds: parameters → initial conditions → short dynamics → long dynamics. The denoising trajectory *is* the temporal unfolding. An agent reading intermediate denoising states gets coarse-to-fine perceptions of progressively longer rollouts — without waiting for full denoising to complete.

## Roundtrip consistency loss: connecting tokens back to physics

The primary CE loss operates purely in token space — it measures whether the denoiser predicts the right token index. But two different token distributions can decode to similar or different physics, and the CE loss can't distinguish between a prediction that's "wrong by one code but physically close" and one that's "wrong by one code and physically catastrophic."

The roundtrip loss provides a structural consistency signal that threads through the actual physics of the VQ pipeline.

### The roundtrip path

```
D3PM logits [B, V_k] per position
  → soft-decode: softmax(logits/τ) @ codebook.weight → continuous embedding [B, D_k]
  → frozen shared decoder → reconstructed features [B, total_encoded_dim]
  → frozen TemporalInverseMLP → reconstructed CNN features [B, T_rt, D_cnn]
  → per-group: frozen PyramidTemporalEncoder → frozen rt_projection
  → frozen HierarchicalProjector → per-level latents
  → frozen quantizer codebook distances → roundtrip logits [B, V_k]
  → CE(roundtrip_logits, GT tokens at truncation T_k)
```

Every component after the soft-decode is frozen — gradients flow only through `softmax(logits/τ) @ codebook.weight` back to the D3PM's logits and into the denoiser.

### Multi-truncation matching

The key insight: the roundtrip comparison target depends on the noise level.

At **high noise** (early denoising), the D3PM's uncertain logits produce a diffuse soft-decode — a weighted average across codebook entries. This coarse representation should match **short-truncation** ground truth (T=32 tokens), because only early dynamics are resolvable at high noise.

At **low noise** (late denoising), sharp logits produce a precise embedding that should match **long-truncation** ground truth (T=512 tokens), because the full trajectory is now resolvable.

The mapping from noise level to truncation level uses configurable boundaries:

```
noise_fraction = effective_t / T

High noise (frac near 1) → coarse truncation (T=32)
Low noise (frac near 0)  → fine truncation (T=512)
```

These boundaries can be set to uniform spacing (default) or loaded from empirical calibration via `calibrate_trajectory.py`, which runs the full denoising trajectory on real data and finds the truncation level that maximizes agreement at each step.

### Soft set-level coherence (Jaccard term)

Beyond per-position CE, the roundtrip loss optionally includes a differentiable set-level Jaccard term. This measures whether the *set of codes used* across all temporal positions matches the ground truth's code usage pattern — not just whether each individual position is correct, but whether the overall code distribution is right.

```
p = softmax(roundtrip_logits / τ)  — soft code usage from predictions
q = one_hot(GT tokens)              — hard code usage from ground truth
J = sum(min(p, q)) / sum(max(p, q)) — differentiable Jaccard similarity
loss += set_coherence_weight × (1 - J)
```

This provides a global coherence signal that per-position CE cannot: even if every position's top-1 prediction is slightly off, the set-level loss penalizes predictions whose aggregate code usage profile diverges from what the physics demands.

## What happens in a single training step

Each training step processes one mini-batch (typically 128 samples). Here is the complete sequence:

### 1. Load and mask

A batch of pretokenized samples arrives as `{key: Tensor[B]}` dictionaries for tokens, observed masks, and target masks. The masking strategy determines which positions the denoiser must predict (target) and which it can see (observed):

- **Random masking**: Each position is independently masked with probability 0.5.
- **Curriculum overrides**: In the inverse_generation stage, temporal positions are always observed and theta+IC positions are always masked — forcing the denoiser to learn the inverse mapping.

### 2. Sample timesteps

A random timestep t is drawn uniformly from [0, T) independently for each sample in the batch. This means a single batch contains samples at all noise levels simultaneously — some nearly clean, some heavily corrupted.

### 3. Compute effective timesteps

If the graded schedule is enabled, the global t for each sample is mapped to per-key effective timesteps. For a sample at global t=40:

```
theta positions:    effective_t = round(40 × 0.15) = 6  (nearly clean)
IC positions:       effective_t = round(40 × 0.25) = 10
temporal (short):   effective_t = round(40 × 0.55) = 22
temporal (long):    effective_t = round(40 × 1.00) = 40 (full noise)
```

### 4. Forward diffusion

Noise is applied to target positions only (observed positions stay clean). Each target position is corrupted according to its effective timestep using the absorbing transition: token → MASK with probability determined by the cumulative noise schedule at the effective timestep.

### 5. Denoiser forward pass

The denoiser is a Transformer that:

1. **Embeds** each token (per-key embedding tables, variable vocab sizes).
2. **Adds** sinusoidal timestep embedding and learned position embeddings.
3. **Adds** hierarchical guidance: L0 token representations are projected and added to all positions as a coarse-to-fine prior.
4. **Attends** through `num_layers` Transformer blocks (with norm_first pre-norm). During training, attention masking prevents unobserved positions from attending to each other — predictions must come from observed context only.
5. **Projects** each position back to its vocabulary space via per-key output heads.

Output: `predicted_logits[key] = [B, V_key]` for every position.

### 6. Compute losses

The loss is a sum of up to four components:

**Primary CE loss** (always active):
Per-position cross-entropy on target positions only, with:
- **Focal loss** (γ=2.0): Down-weights easy predictions by (1-p_correct)^γ. Critical because rollouts share 70-80% of tokens — without focal loss, the denoiser coasts on the easy shared tokens and ignores the 20-30% that actually differ.
- **SNR weighting**: Per-timestep weight 1/β_t, so high-noise steps (where prediction is harder) get proportionally less gradient. With graded schedule, uses per-key effective timesteps for accurate noise-aware weighting.
- **Vocab-size weighting**: Normalizes per-key loss by log(V_k)/log(V_max), so keys with large vocabularies don't dominate.

**Physics loss** (optional, after warmup):
Soft-decodes logits through the frozen VQ pipeline to continuous physics parameters (theta values, IC spatial features), then MSE against ground truth. Gated by a bell-shaped timestep function that focuses the signal on mid-noise steps where the denoiser is making non-trivial predictions.

**Roundtrip loss** (optional, after warmup):
The full soft-roundtrip path described above, comparing against truncation-matched ground truth tokens. Gated by a cosine timestep function that weights low-noise steps more heavily (where the roundtrip comparison is most meaningful). Includes optional soft Jaccard term.

**Set coherence loss** (optional):
The differentiable Jaccard term, weighted and added to the roundtrip loss when `set_coherence_weight > 0`.

### 7. Backward pass and update

Standard gradient computation, clipped to max norm 1.0, followed by AdamW update. Learning rate follows cosine annealing with linear warmup.

## The 3-stage curriculum

Total training: 20 epochs across 3 stages, each with different masking and learning rate.

### Stage 1: Joint Random (8 epochs)

**Masking**: Random 50% across all families.
**Learning rate**: 1e-4 (full).
**What it learns**: Basic token co-occurrence statistics. The denoiser discovers that certain theta tokens predict certain temporal patterns, that IC tokens constrain early dynamics, and that tokens within a family are correlated. This is the broadest learning phase — every position sometimes serves as context and sometimes as target.

### Stage 2: Inverse Generation (8 epochs)

**Masking**: Temporal always observed, theta + IC always masked.
**Learning rate**: 5e-5 (reduced, fine-tuning).
**What it learns**: The inverse mapping — given temporal dynamics, predict the parameters and initial conditions that produced them. This is the core capability for agent "imagination": the agent observes what the system did and infers what could have caused it. The masking constraint forces the denoiser to develop a genuine understanding of the dynamics→parameters mapping, not just exploit token correlations.

### Stage 3: Consolidation (4 epochs)

**Masking**: Random 50% again, all families.
**Learning rate**: 2e-5 (lowest, stabilization).
**What it learns**: Stability under mixed masking after the inverse generation specialization. Without this stage, the denoiser would overfit to the temporal→theta/IC direction and lose its ability to do general-purpose completion. Consolidation ensures the model retains flexibility for arbitrary inpainting patterns at inference time.

## Variation within a single epoch

Every batch in an epoch sees different:

- **Timesteps**: Each sample draws an independent random t ∈ [0, 50), so a single batch spans the full noise range. The denoiser must simultaneously handle nearly-clean samples (t≈0, easy) and heavily-masked samples (t≈49, hard).
- **Masking patterns**: Each sample gets an independent random mask (in stages 1 and 3). Two samples in the same batch might have completely different observed/target splits.
- **Effective timesteps**: Even at the same global t, different keys see different noise levels due to the graded schedule. A theta position at global t=30 is nearly resolved; a long-horizon temporal position at the same t is still heavily masked.
- **Truncation matching**: For the roundtrip loss, each sample's timestep maps to a different truncation level. Samples at high t compare against T=32 ground truth; samples at low t compare against T=512.

This variance is by design — it forces the denoiser to be robust across the full range of conditions it will encounter during sampling, where it must handle every noise level in sequence as t decreases from T to 0.

## Connection to the denoising trajectory

At inference time, the `sample()` method runs the full reverse loop: initialize with noise (or MASK tokens in absorbing mode), then iteratively denoise from t=T to t=0. At each step, the denoiser predicts clean tokens, and the reverse step either accepts the prediction (t=0) or blends it with remaining noise (t>0).

With snapshot recording (`snapshot_steps` parameter), intermediate states can be captured at any denoising step. Combined with the graded schedule, this means:

- At t=40 (80% noise): theta is nearly resolved, everything else is uncertain → a coarse perception of "what kind of system is this?"
- At t=20 (40% noise): theta + IC resolved, short temporal emerging → "what does it look like early on?"
- At t=10 (20% noise): most temporal resolved → "what happens over the full rollout?"
- At t=0 (clean): full resolution → precise 160-token description

The denoising trajectory recapitulates the physical temporal unfolding. An agent doesn't need to wait for full denoising to perceive — it can read off intermediate states as progressively refined perceptions, exactly matching how physical causality reveals structure from parameters to dynamics.

## Connection to emergent communication

The D3PM's inverse generation capability — observing dynamics, imagining causes — is the perceptual foundation for agent communication. But the agent's internal token representations need a medium for externalization: a structured, compositional language that preserves the agent's native ontology rather than collapsing it onto human categories.

This is the role of [LFM (Language Faculty Model)](https://github.com/dgabriele/lfm), a companion framework that imposes morphosyntactic and phonotactic constraints on agent communication, producing an emergent language that is alien but structurally regular — regular enough for a pretrained multilingual LLM to learn to translate it. The graded noise schedule's causal hierarchy maps directly onto information structure in the emergent language: what the agent perceives first (theta, at low noise) becomes given/background information; what resolves last (long-horizon temporal) becomes focus/new information. These are exactly the information-structural distinctions that LFM's syntax modules are designed to learn.

For more on why this language-first approach to interpreting agent representations is both necessary and underexplored, see [Why the Language-First Approach Is Underexplored](https://github.com/dgabriele/lfm/blob/main/docs/why-language-first.md).
