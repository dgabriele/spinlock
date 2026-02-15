# ADAPT Phase: Online VQTokenizer Refinement in the Synthesis Loop

## Motivation

The synthesis pipeline runs an EXPLORE/REFINE loop where the D3PM diffusion
model generates token combinations, QBM simulates them, and high-priority
discoveries train the D3PM to generate better combinations. But the
**VQTokenizer** — the component that defines what tokens *mean* — is frozen.

This creates a representational ceiling:

1. **353 unused codebook entries** (out of 1320) can never be allocated to newly
   discovered physical regimes.
2. **Novel physics gets quantized to nearest existing code**, losing the
   information that makes it interesting.
3. **The D3PM can never learn distinctions the tokenizer can't express.** If two
   physically distinct parameter regimes map to the same token, the
   generalization signal is blind to that difference.

The ADAPT phase makes the VQTokenizer a co-evolving component of the synthesis
pipeline, turning the system into a fully self-improving representation learner.

## Architecture Overview

```
Current:    EXPLORE → REFINE  (repeat)
Proposed:   EXPLORE → REFINE → [ADAPT every K cycles]

EXPLORE:  D3PM proposes → QBM simulates → retokenize → push to queue
REFINE:   D3PM trains on high-priority experiences
ADAPT:    VQTokenizer fine-tunes on accumulated experiences
          → retokenize queue + probe buffer → cascade updates
```

The ADAPT phase runs periodically (every K cycles, e.g., K=5), not every cycle.
This is critical for stability — EMA codebook updates need sufficiently large
batches to avoid jitter.

## Detailed Design

### 1. Experience Buffer

During EXPLORE, the pipeline already calls `_extract_features_from_rollout()`
to get continuous features before tokenization. We retain these features in an
**experience buffer** for later VQTokenizer training.

**New instance variables in `SynthesisVerificationPipeline`:**

```python
# In __init__:
self._experience_buffer: List[Dict[str, torch.Tensor]] = []
self._experience_buffer_max: int = config.adapt.max_experiences  # e.g., 2000
```

**Accumulation in `_explore_step()`:**

After calling `_extract_features_from_rollout()` and before tokenization,
store the features (on CPU to save GPU memory):

```python
# In _explore_step(), after feature extraction:
if len(self._experience_buffer) < self._experience_buffer_max:
    self._experience_buffer.append({
        'temporal_features': features['temporal_features'].cpu(),
        'initial_manual': features['initial_manual'].cpu(),
        'initial_raw': features['initial_raw'].cpu(),
        'theta_features': features['theta_features'].cpu(),
    })
```

**Memory cost**: ~50KB per experience (features only). 2000 experiences = ~100MB.

### 2. ADAPT Phase Implementation

The ADAPT phase uses the VQTokenizer's **existing training infrastructure** —
no modifications to the VQ-VAE model, loss functions, or EMA mechanics.

**New method: `_adapt_tokenizer()`**

```python
def _adapt_tokenizer(self) -> Dict[str, float]:
    """Periodic VQTokenizer refinement on accumulated experiences.

    Fine-tunes the VQTokenizer on a mix of:
    - Original training features (anti-forgetting, loaded from dataset)
    - Newly discovered features from EXPLORE (experience buffer)

    Then retokenizes the priority queue and probe buffer with the
    updated codebooks.

    Returns:
        Dict with metrics: recon_loss_before, recon_loss_after, etc.
    """
```

**Step-by-step flow:**

#### Step 2a: Measure pre-ADAPT reconstruction quality

Before fine-tuning, measure reconstruction loss on the experience buffer
to establish a baseline.

```python
recon_before = self._compute_tokenizer_recon_loss(self._experience_buffer)
```

#### Step 2b: Prepare combined training features

Combine original training data with discovered experiences. The mix ratio
controls the forgetting/adaptation tradeoff.

```python
# Load a random subsample of original training features
original_features = self._load_original_features_subsample(
    ratio=config.adapt.original_mix_ratio,  # e.g., 0.3 → 30% of 50K = 15K
)

# Stack discovered features from buffer
discovered_features = self._stack_experience_buffer()

# Combine
combined = {
    key: torch.cat([original_features[key], discovered_features[key]], dim=0)
    for key in ['temporal_features', 'initial_manual', 'initial_raw', 'theta_features']
}
```

#### Step 2c: Fine-tune the VQTokenizer

Create a short training run using the existing trainer infrastructure.

```python
# Create a fresh trainer with current model weights
trainer = VQTokenizerTrainer(
    model=self._tokenizer.model,
    config=adapt_config,  # Fewer epochs, lower LR
    group_indices=self._tokenizer.group_indices,
    normalization_stats=self._tokenizer.normalization_stats,
    feature_metadata=self._tokenizer.feature_metadata,
)

# Short fine-tuning (e.g., 5-10 epochs, lower LR than initial training)
history = trainer.train_on_features(
    temporal_features=combined['temporal_features'],
    initial_manual=combined['initial_manual'],
    initial_raw=combined['initial_raw'],
    theta_features=combined['theta_features'],
)
```

**Training config for ADAPT** (shorter, gentler than initial training):

```yaml
adapt:
  enabled: true
  frequency: 5              # Run ADAPT every 5 synthesis cycles
  num_epochs: 10             # Short fine-tuning (vs 200 for initial training)
  learning_rate: 1.0e-5      # 10x lower than initial training LR
  original_mix_ratio: 0.3    # 30% of original data mixed in (anti-forgetting)
  max_experiences: 2000      # Max experiences to accumulate before ADAPT
  min_experiences: 100       # Min experiences needed to trigger ADAPT
```

#### Step 2d: Retokenize cascade

After codebook update, all stored tokens become stale and must be refreshed.

```python
# Retokenize priority queue items
self._retokenize_queue()

# Retokenize probe buffer (for generalization measurement)
self._retokenize_probe_buffer()

# Clear experience buffer (features consumed)
self._experience_buffer.clear()
```

#### Step 2e: Measure post-ADAPT reconstruction quality

```python
recon_after = self._compute_tokenizer_recon_loss(self._experience_buffer_snapshot)
logger.info(
    f"ADAPT: recon loss {recon_before:.4f} → {recon_after:.4f} "
    f"(Δ={recon_before - recon_after:+.4f})"
)
```

### 3. Retokenization Details

#### Queue Retokenization

Each `PriorityItem` stores `retokenized_tokens: Dict[str, int]`. After ADAPT,
we need to re-tokenize from the original features. This requires storing the
raw features alongside tokens.

**Change to PriorityItem:**

```python
@dataclass(order=True)
class PriorityItem:
    priority: float
    index: int
    surprisal: float
    generated_tokens: Dict[str, int]
    retokenized_tokens: Dict[str, int]
    theta: Optional[np.ndarray]
    features: Optional[np.ndarray]          # [5] priority features
    raw_features: Optional[Dict[str, np.ndarray]] = None  # NEW: for retokenization
```

**Implementation of `_retokenize_queue()`:**

```python
def _retokenize_queue(self):
    """Retokenize all queue items with updated VQTokenizer codebooks."""
    items = self._queue.get_all_items()
    retokenized_count = 0

    for item in items:
        if item.raw_features is None:
            continue  # Legacy items without stored features

        # Reconstruct feature tensors
        features = {
            k: torch.tensor(v, device=self.device).unsqueeze(0)
            for k, v in item.raw_features.items()
        }

        # Retokenize with updated codebooks
        new_tokens = self._tokenizer.tokenize(**features)
        item.retokenized_tokens = {
            k: v.item() for k, v in new_tokens.items()
        }
        retokenized_count += 1

    # Re-score priorities (token rarity may have changed)
    self._queue.recompute_all_priorities()

    logger.info(f"ADAPT: Retokenized {retokenized_count}/{len(items)} queue items")
```

#### Probe Buffer Retokenization

The probe buffer stores token dicts. We need the raw features to retokenize.

**Change to probe buffer storage:**

```python
# In _populate_probe_buffer():
self._probe_buffer_features: List[Dict[str, np.ndarray]] = []  # NEW: raw features
# ... existing _probe_buffer: List[Dict[str, int]] stays for tokens
```

**Implementation of `_retokenize_probe_buffer()`:**

```python
def _retokenize_probe_buffer(self):
    """Retokenize probe buffer with updated VQTokenizer codebooks."""
    if not self._probe_buffer_frozen:
        return

    new_probe_buffer = []
    for features_dict in self._probe_buffer_features:
        features = {
            k: torch.tensor(v, device=self.device).unsqueeze(0)
            for k, v in features_dict.items()
        }
        new_tokens = self._tokenizer.tokenize(**features)
        new_probe_buffer.append({
            k: v.item() for k, v in new_tokens.items()
        })

    self._probe_buffer = new_probe_buffer
    logger.info(f"ADAPT: Retokenized {len(new_probe_buffer)} probe buffer items")
```

### 4. Normalization Stats Strategy

The VQTokenizer normalizes features before encoding. When fine-tuning on new
experiences, normalization stats must be handled carefully.

**Options:**

| Strategy | Pros | Cons |
|----------|------|------|
| **Freeze original stats** | Stable, no cascading changes | New features may be out-of-distribution |
| **Recompute on combined data** | Accurate for combined distribution | Changes encoding of all old data |
| **EMA update stats** | Gradual adaptation | Custom implementation needed |

**Recommendation: Freeze original stats.**

The normalization stats from the 50K training set should remain fixed. New
experiences from synthesis come from the same QBM domain, so feature
distributions are similar. If drift becomes significant, it would show up as
increasing reconstruction loss — measurable and actionable.

### 5. VQTokenizerTrainer Changes

The existing trainer expects to create its own dataloaders from raw tensors.
We need a method that accepts pre-prepared features.

**New method: `train_on_features()`**

```python
def train_on_features(
    self,
    temporal_features: Optional[torch.Tensor] = None,
    initial_manual: Optional[torch.Tensor] = None,
    initial_raw: Optional[torch.Tensor] = None,
    theta_features: Optional[torch.Tensor] = None,
    temporal_mask: Optional[torch.Tensor] = None,
    temporal_lengths: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Train on pre-extracted, pre-normalized features.

    Unlike train(), this does NOT re-extract or re-normalize features.
    Used for fine-tuning during synthesis ADAPT phase.
    """
    # Apply existing normalization stats (frozen from initial training)
    normalized = self._apply_normalization(
        temporal_features, initial_manual, initial_raw, theta_features
    )

    # Create dataloaders (reuse existing _create_dataloaders)
    train_loader, val_loader = self._create_dataloaders(**normalized)

    # Train for configured epochs
    for epoch in range(self.config.training.num_epochs):
        train_metrics = self._train_epoch(train_loader, epoch)
        if val_loader:
            val_metrics = self._validate_epoch(val_loader)

    return self.training_history
```

### 6. Integration into Main Synthesis Loop

**Modified `run()` method:**

```python
def run(self):
    """Main synthesis loop with ADAPT phase."""
    for cycle in range(self.config.num_cycles):
        match self._mode:
            case Mode.EXPLORE:
                # Populate probe buffer once
                if not self._probe_buffer_frozen:
                    self._populate_probe_buffer()

                # Explore: generate, simulate, retokenize, push
                for step in range(self.config.generation.explore_steps):
                    self._explore_step()
                    # Accumulate features for ADAPT
                    # (done inside _explore_step)

            case Mode.REFINE:
                # Measure pre-refine probe loss
                probe_before = self._compute_probe_loss()

                # Refine D3PM
                for epoch in range(self.config.training.refine_epochs):
                    self._refine_epoch()

                # Measure post-refine generalization
                probe_after = self._compute_probe_loss()
                generalization = probe_before - probe_after
                self._queue.set_generalization_score(generalization)

        # === ADAPT PHASE (periodic) ===
        if (self.config.adapt.enabled
            and (cycle + 1) % self.config.adapt.frequency == 0
            and len(self._experience_buffer) >= self.config.adapt.min_experiences):

            logger.info(f"=== ADAPT Phase (cycle {cycle}) ===")
            adapt_metrics = self._adapt_tokenizer()
            self._log_adapt_metrics(adapt_metrics, cycle)

        # Alternate modes
        self._mode = Mode.REFINE if self._mode == Mode.EXPLORE else Mode.EXPLORE
```

### 7. Config Schema Changes

**New `AdaptConfig` in `config.py`:**

```python
class AdaptConfig(BaseModel):
    """VQTokenizer online refinement configuration."""
    enabled: bool = Field(default=False,
        description="Enable periodic VQTokenizer fine-tuning")
    frequency: int = Field(default=5, ge=1,
        description="Run ADAPT every N synthesis cycles")
    num_epochs: int = Field(default=10, ge=1,
        description="Fine-tuning epochs per ADAPT phase")
    learning_rate: float = Field(default=1e-5, gt=0.0,
        description="Lower LR than initial training for stability")
    original_mix_ratio: float = Field(default=0.3, ge=0.0, le=1.0,
        description="Fraction of original 50K data mixed in (anti-forgetting)")
    max_experiences: int = Field(default=2000, ge=1,
        description="Max experiences to accumulate before ADAPT")
    min_experiences: int = Field(default=100, ge=1,
        description="Min experiences needed to trigger ADAPT")
    original_features_path: Optional[Path] = Field(default=None,
        description="Path to original training features for mixing")
    normalization_strategy: str = Field(default="freeze",
        pattern="^(freeze|recompute|ema)$",
        description="How to handle normalization stats during ADAPT")
```

**Updated `SynthesisConfig`:**

```python
class SynthesisConfig(BaseModel):
    # ... existing fields ...
    adapt: AdaptConfig = Field(default_factory=AdaptConfig)
```

### 8. Original Features Loading

The ADAPT phase needs a subsample of the original 50K training features for
anti-forgetting. Two options:

**Option A: Load from the original HDF5 dataset**

The synthesis pipeline already loads the QBM dataset for rollout generation.
We can extract features from a random subsample:

```python
def _load_original_features_subsample(self, ratio: float) -> Dict[str, torch.Tensor]:
    """Load a random subsample of original training features."""
    import h5py

    with h5py.File(self.config.adapt.original_features_path, 'r') as f:
        n_total = f['features/temporal'].shape[0]
        n_sample = int(n_total * ratio)
        indices = np.random.choice(n_total, n_sample, replace=False)
        indices.sort()  # Sequential access for HDF5 performance

        return {
            'temporal_features': torch.tensor(f['features/temporal'][indices]),
            'initial_manual': torch.tensor(f['features/initial_manual'][indices]),
            'initial_raw': torch.tensor(f['features/initial_raw'][indices]),
            'theta_features': torch.tensor(f['features/theta'][indices]),
        }
```

**Option B: Pre-extract and cache features**

Run `VQTokenizer._extract_features()` once on the 50K dataset and save to a
dedicated HDF5 file. Faster loading during synthesis.

```bash
# One-time preparation:
poetry run python scripts/extract_training_features.py \
    --dataset datasets/qbm_50k.h5 \
    --tokenizer-checkpoint checkpoints/v2/vqvae/vq_tokenizer_best.pt \
    --output datasets/qbm_50k_features.h5
```

**Recommendation:** Option B is cleaner. The feature extraction pipeline
(including temporal feature cleaning) runs once, and the ADAPT phase loads
pre-cleaned features directly.

### 9. D3PM Embedding Adaptation

When VQTokenizer codebooks shift, token semantics change. The D3PM's learned
embeddings for each token index may become misaligned.

**Key insight:** The D3PM's `DenoisingNetwork` has per-category embedding
layers that map token indices to hidden representations. After ADAPT, these
embeddings correspond to slightly different physical meanings.

**Mitigation strategies (in order of preference):**

1. **Do nothing (rely on REFINE):** The next REFINE phase naturally trains the
   D3PM on retokenized queue items. Since codebook shifts are gradual (EMA
   decay=0.99), the D3PM's existing embeddings are close enough to the new
   semantics that a few refine epochs bridge the gap.

2. **Warm-restart optimizer:** Reset the D3PM optimizer momentum after ADAPT
   to avoid stale gradient statistics pushing the wrong direction.

3. **Extended REFINE after ADAPT:** Run extra refine epochs in the cycle
   immediately following an ADAPT phase.

**Recommendation:** Start with (1). The EMA mechanism ensures codebook changes
are small per ADAPT step. Monitor the generalization signal — if it shows a
spike after ADAPT phases, add (2) or (3).

### 10. Metrics and Monitoring

**Per-ADAPT metrics to log:**

| Metric | Meaning |
|--------|---------|
| `adapt/recon_loss_before` | VQTokenizer reconstruction loss before fine-tuning |
| `adapt/recon_loss_after` | VQTokenizer reconstruction loss after fine-tuning |
| `adapt/recon_improvement` | Delta (positive = better) |
| `adapt/codebook_drift` | Average L2 distance of codebook entries before/after |
| `adapt/new_codes_activated` | Number of previously-unused codes that got assignments |
| `adapt/experiences_used` | Size of experience buffer consumed |
| `adapt/queue_retokenized` | Number of queue items retokenized |
| `adapt/token_change_rate` | Fraction of retokenized items whose tokens changed |

**Codebook drift computation:**

```python
def _compute_codebook_drift(self, old_state_dict, new_state_dict) -> float:
    """Average L2 distance between old and new codebook entries."""
    total_drift = 0.0
    n_entries = 0
    for key in old_state_dict:
        if 'embedding.weight' in key:
            old = old_state_dict[key]
            new = new_state_dict[key]
            drift = (old - new).norm(dim=1).mean().item()
            total_drift += drift
            n_entries += 1
    return total_drift / max(n_entries, 1)
```

### 11. Checkpoint Integration

**Additional state to save in `_save_cycle_checkpoint()`:**

```python
checkpoint['experience_buffer'] = self._experience_buffer
checkpoint['adapt_history'] = self._adapt_history
checkpoint['tokenizer_adapt_epoch'] = self._tokenizer_adapt_epoch
checkpoint['probe_buffer_features'] = self._probe_buffer_features  # For retokenization
```

**On resume:** Restore experience buffer and probe buffer features so ADAPT
can pick up where it left off.

## Implementation Order

1. **Config**: Add `AdaptConfig` to `config.py`, wire into `SynthesisConfig`
2. **PriorityItem**: Add `raw_features` field for retokenization
3. **Experience buffer**: Accumulate features during `_explore_step()`
4. **VQTokenizerTrainer**: Add `train_on_features()` method
5. **Feature extraction script**: Pre-extract original 50K features to HDF5
6. **ADAPT phase core**: `_adapt_tokenizer()`, retokenization cascade
7. **Metrics**: Codebook drift, token change rate, reconstruction improvement
8. **Main loop integration**: Wire ADAPT into `run()`
9. **YAML config**: Add `adapt:` section to `qbm_synthesis.yaml`

## Verification Plan

### Unit-level

1. **Experience buffer**: Push 10 items, verify buffer grows, check CPU storage
2. **Feature stacking**: Stack buffer into tensors, verify shapes match trainer expectations
3. **Retokenization**: Tokenize → modify codebook → retokenize → verify tokens changed
4. **Normalization**: Verify frozen stats applied correctly to new features

### Integration-level

5. **3-cycle ADAPT test**: Run 3 cycles with `adapt.frequency=1`, verify:
   - ADAPT triggers after each cycle
   - Reconstruction loss reported
   - Queue items retokenized
   - Probe buffer retokenized
   - No crashes or shape mismatches

6. **Codebook drift**: Verify drift is small (EMA with decay=0.99)
7. **Token change rate**: Verify only a fraction of tokens change per ADAPT
8. **Generalization signal**: Check that generalization doesn't spike after ADAPT

### End-to-end

9. **20-cycle run**: Full synthesis with ADAPT every 5 cycles
10. **Compare**: Codebook utilization before/after (unused entries activated?)
11. **Compare**: Token SET diversity before/after (Jaccard analysis)
12. **Compare**: Generalization trajectory with vs without ADAPT

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| EMA jitter from small batches | Medium | High | Mix in 30% original data, use low LR |
| Codebook collapse during ADAPT | Low | High | Monitor per-codebook utilization, add dead code reset |
| D3PM confusion after token shift | Medium | Medium | Rely on natural REFINE adaptation; extend if needed |
| Memory pressure from experience buffer | Low | Low | Cap at 2000 items (~100MB), CPU storage |
| Stale normalization stats | Low | Low | Freeze stats; features from same QBM domain |
| ADAPT makes things worse | Medium | Medium | Measure recon_before/after; rollback if degraded |

## Future Extensions

1. **Adaptive ADAPT frequency**: Trigger ADAPT when experience buffer divergence
   exceeds a threshold, rather than fixed schedule.

2. **Curriculum mixing**: Start with high original_mix_ratio (0.5) and decrease
   over cycles as the tokenizer becomes more robust.

3. **Codebook growth**: If all entries are utilized, add new entries (increase
   `num_embeddings`). This requires D3PM architecture changes but is the natural
   next step for vocabulary extension.

4. **repr_trace integration**: Once `DenoisingNetwork` exposes hidden states,
   measure how ADAPT affects the D3PM's internal representations of token
   combinations.

## Relationship to Other Plan Components

- **Significance-weighted priority** (implemented): The generalization signal
  becomes more meaningful when the tokenizer co-adapts — it measures whether
  the whole representation system improves, not just the D3PM.

- **D3PM retraining** (in progress): Must complete before ADAPT can be
  implemented. The D3PM needs correct vocab sizes first.

- **repr_trace / structural_growth** (stubbed): These measure D3PM internal
  changes. ADAPT adds a new dimension: tokenizer internal changes. Together,
  they give a complete picture of representation evolution.
