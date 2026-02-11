# Pyramid Autoregressive Trajectory Completion: Implementation Plan

**Status:** Planning Phase
**Date:** 2026-02-05
**Author:** Claude Sonnet 4.5
**Version:** 1.0.0
**Agent ID:** a1d158b (planning agent - can be resumed)

---

## Executive Summary

This document provides a production-ready implementation plan for extending the trajectory completion experiment with **pyramid-aware autoregressive prediction**. Instead of predicting all masked tokens in parallel (BERT-style), this variant predicts tokens in temporal pyramid order (coarse → fine) over 4 sequential passes.

**Key Innovation:** Respect temporal scale dependencies by predicting p3 (coarsest, 32 timesteps) → p2 (64 timesteps) → p1 (128 timesteps) → p0 (finest, 256 timesteps), where each finer scale conditions on all coarser scales.

**Motivation:** The VQ-VAE's temporal pyramid structure provides a natural ordering for autoregressive prediction that mirrors multi-scale physics (slow dynamics constrain fast dynamics).

**Trade-off:** 4x slower inference (~40ms vs ~10ms) for +2-5% accuracy improvement.

---

## Quick Navigation

1. [Context and Motivation](#context-and-motivation)
2. [Architecture Extension](#architecture-extension)
3. [Training Infrastructure](#training-infrastructure)
4. [Configuration](#configuration)
5. [Integration Points](#integration-points)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Expected Results](#expected-results)
9. [Technical Challenges](#technical-challenges)
10. [Code Sketches](#code-sketches)

---

## Context and Motivation

### Current Approach: Parallel Prediction

**Existing `TrajectoryCompletionModel`**:
- Predicts all masked tokens simultaneously in 1 forward pass
- Treats tokens as conditionally independent given context
- Fast (~10ms) but ignores temporal pyramid dependencies
- BERT-style masked language modeling

### Temporal Pyramid Structure

The VQ-VAE organizes tokens by temporal resolution:

```
Token Sequence (~30-60 tokens):
├─ initial_cat0: [L0, L1, L2]          # Initial conditions
├─ temporal_p0_cat0: [L0, L1, L2]      # 256 timesteps (finest)
├─ temporal_p1_cat0: [L0, L1, L2]      # 128 timesteps
├─ temporal_p2_cat0: [L0, L1, L2]      # 64 timesteps
└─ temporal_p3_cat0: [L0, L1, L2]      # 32 timesteps (coarsest)
```

**Natural Ordering:** p3 (global dynamics) → p2 → p1 → p0 (fine details)

### Why Pyramid Autoregressive?

**Physical Intuition:**
- Coarse temporal scales capture slow, global dynamics
- Fine temporal scales add high-frequency details
- In physics: slow modes constrain fast modes (like turbulent cascade)
- Predicting coarse-to-fine mirrors this natural hierarchy

**Computational Cost:**
- 4 forward passes (one per pyramid level)
- ~40ms total (4x slower than parallel, 20x faster than full autoregressive)
- Reasonable trade-off for accuracy gains

**Expected Benefits:**
1. Model temporal scale dependencies explicitly
2. +2-5% token accuracy improvement
3. Better reconstruction quality for fine temporal scales
4. Physical interpretability (matches multi-scale physics)

---

## Architecture Extension

### Overview

Extend the existing `TrajectoryCompletionModel` with pyramid-aware multi-pass prediction:

```
Pass 1: Predict initial + p3 (coarsest)
        ↓ (predicted tokens become observations)
Pass 2: Predict p2 | condition on initial + p3
        ↓
Pass 3: Predict p1 | condition on initial + p3 + p2
        ↓
Pass 4: Predict p0 | condition on initial + p3 + p2 + p1
```

### Component 1: Pyramid Token Indexer

**Purpose:** Map token positions to pyramid levels

**Key Features:**
- Parse VQ-VAE category names to identify pyramid levels
- Generate masks for tokens at each level
- Provide pyramid ordering (coarse → fine)

**Implementation:** `experiments/trajectory_completion/models/pyramid_utils.py`

<details>
<summary>Code Sketch: PyramidTokenIndexer</summary>

```python
from typing import List, Dict
import torch

class PyramidTokenIndexer:
    """Maps token positions to temporal pyramid levels."""

    def __init__(self, vqvae_categories: List[str], num_levels: int = 3):
        self.num_levels = num_levels
        self.pyramid_map = self._parse_categories(vqvae_categories)

    def _parse_categories(self, categories: List[str]) -> Dict[str, List[int]]:
        """
        Parse category names to identify pyramid levels.

        Example:
            categories = ['initial_0', 'temporal_p0_0', 'temporal_p1_0',
                         'temporal_p2_0', 'temporal_p3_0']
            → {'initial': [0,1,2], 'p0': [3,4,5], 'p1': [6,7,8], ...}
        """
        pyramid_map = {'initial': []}
        for i, cat_name in enumerate(categories):
            base_idx = i * self.num_levels
            token_indices = list(range(base_idx, base_idx + self.num_levels))

            if 'initial' in cat_name:
                pyramid_map['initial'].extend(token_indices)
            elif 'temporal_p0' in cat_name:
                pyramid_map.setdefault('p0', []).extend(token_indices)
            elif 'temporal_p1' in cat_name:
                pyramid_map.setdefault('p1', []).extend(token_indices)
            elif 'temporal_p2' in cat_name:
                pyramid_map.setdefault('p2', []).extend(token_indices)
            elif 'temporal_p3' in cat_name:
                pyramid_map.setdefault('p3', []).extend(token_indices)

        return pyramid_map

    def get_pyramid_mask(self, seq_len: int, level: str) -> torch.BoolTensor:
        """Get mask for tokens at pyramid level."""
        mask = torch.zeros(seq_len, dtype=torch.bool)
        indices = self.pyramid_map.get(level, [])
        if indices:
            mask[indices] = True
        return mask

    def get_coarse_to_fine_order(self) -> List[str]:
        """Return pyramid levels in coarse→fine order."""
        return ['initial', 'p3', 'p2', 'p1', 'p0']
```
</details>

### Component 2: Pyramid Autoregressive Model

**Purpose:** Wrap `TrajectoryCompletionModel` for multi-pass prediction

**Key Features:**
- Execute 4 sequential forward passes
- Progressive conditioning (each pass conditions on previous predictions)
- Teacher forcing during training (scheduled annealing)
- Caching to avoid redundant computation

**Implementation:** `experiments/trajectory_completion/models/pyramid_autoregressive.py`

<details>
<summary>Code Sketch: PyramidAutoregressiveModel</summary>

```python
import torch
import torch.nn as nn
from typing import Dict, List

class PyramidAutoregressiveModel(nn.Module):
    """Pyramid-aware autoregressive trajectory completion."""

    def __init__(
        self,
        base_model: TrajectoryCompletionModel,
        indexer: PyramidTokenIndexer,
        use_caching: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        self.indexer = indexer
        self.use_caching = use_caching
        self.pyramid_order = indexer.get_coarse_to_fine_order()

    def forward(
        self,
        tokens_observed: torch.Tensor,
        mask_observed: torch.Tensor,
        mask_target: torch.Tensor,
        mode: str = "autoregressive"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with pyramid-aware prediction.

        Args:
            mode: "autoregressive" for 4-pass, "parallel" for baseline
        """
        if mode == "parallel":
            return self.base_model(tokens_observed, mask_observed, mask_target)
        return self._pyramid_forward(tokens_observed, mask_observed, mask_target)

    def _pyramid_forward(
        self,
        tokens_observed: torch.Tensor,
        mask_observed: torch.Tensor,
        mask_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Execute multi-pass pyramid prediction."""
        batch_size, seq_len = tokens_observed.shape

        tokens_current = tokens_observed.clone()
        mask_current = mask_observed.clone()

        intermediate_predictions = []

        for pyramid_level in self.pyramid_order:
            # Mask for tokens at this pyramid level
            level_mask = self.indexer.get_pyramid_mask(seq_len, pyramid_level)
            level_mask = level_mask.unsqueeze(0).expand(batch_size, -1)

            # Tokens to predict this pass: level tokens not yet predicted
            pass_target = level_mask & mask_target & (~mask_current)

            if not pass_target.any():
                continue

            # Predict with base model
            outputs = self.base_model(
                tokens_observed=tokens_current,
                mask_observed=mask_current,
                mask_target=pass_target
            )

            # Update tokens with predictions
            predictions = outputs['predictions']
            tokens_current = torch.where(pass_target, predictions, tokens_current)
            mask_current = mask_current | pass_target

            intermediate_predictions.append(predictions.clone())

        return {
            'predictions': tokens_current,
            'tokens_completed': tokens_current,
            'intermediate_predictions': intermediate_predictions
        }

    def forward_with_teacher_forcing(
        self,
        tokens_observed: torch.Tensor,
        tokens_true: torch.Tensor,
        mask_observed: torch.Tensor,
        mask_target: torch.Tensor,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward with teacher forcing.

        With probability teacher_forcing_ratio, use ground truth tokens
        from previous passes instead of predictions.
        """
        # Similar to _pyramid_forward but with teacher forcing logic
        # See full code sketch in detailed plan
        pass
```
</details>

---

## Training Infrastructure

### Teacher Forcing Strategy

**Challenge:** Autoregressive models suffer from error accumulation (prediction errors in p3 propagate to p2, p1, p0).

**Solution:** **Scheduled Sampling** - Start with full teacher forcing (use ground truth), gradually transition to autoregressive (use predictions)

**Schedule Options:**
1. **Linear**: `teacher_forcing_ratio = max(0, 1 - epoch/total_epochs)`
2. **Exponential**: `teacher_forcing_ratio = 0.5^(epoch * decay_rate)`
3. **Constant**: `teacher_forcing_ratio = 1.0` (always use ground truth - safest)

### Pyramid Trainer

**Extension:** `PyramidAutoregressiveTrainer` extends `CompletionTrainer`

**Key Additions:**
- Per-pyramid-level loss computation
- Teacher forcing scheduling
- Conditional accuracy metrics

**Implementation:** `experiments/trajectory_completion/training/pyramid_trainer.py`

---

## Configuration

### New Config Schema

```yaml
# experiments/trajectory_completion/baseline_50k/experiments/pyramid_autoregressive.yaml

metadata:
  name: "pyramid_autoregressive"
  description: "Pyramid-aware autoregressive trajectory completion"

# Existing configs (data, checkpoints, masking, model)...

# NEW: Pyramid-specific config
pyramid:
  enabled: true
  use_caching: true
  teacher_forcing_schedule: "linear"  # "linear" | "exponential" | "constant"

training:
  epochs: 30  # More epochs for autoregressive training
  # ... rest same as baseline
```

---

## Integration Points

### Minimal Code Changes

**File 1:** `experiments/trajectory_completion/run_experiment.py`

```python
# After creating base model:
if config.pyramid.enabled:
    from .models.pyramid_autoregressive import PyramidAutoregressiveModel
    from .models.pyramid_utils import PyramidTokenIndexer

    # Build indexer from VQ-VAE categories
    indexer = PyramidTokenIndexer(
        vqvae_categories=vqvae.get_category_names(),
        num_levels=3
    )

    # Wrap base model
    model = PyramidAutoregressiveModel(
        base_model=base_model,
        indexer=indexer,
        use_caching=config.pyramid.use_caching
    )

    # Use pyramid trainer
    from .training.pyramid_trainer import PyramidAutoregressiveTrainer
    trainer = PyramidAutoregressiveTrainer(...)
else:
    # Use existing parallel model and trainer
    model = base_model
    trainer = CompletionTrainer(...)
```

**File 2:** Add `pyramid` field to `CompletionExperimentConfig` (Pydantic schema)

**File 3:** Create new config YAML (shown above)

**Total new code:** ~800 lines across 3 new files
- `pyramid_utils.py` (~150 lines)
- `pyramid_autoregressive.py` (~350 lines)
- `pyramid_trainer.py` (~300 lines)

---

## Evaluation Metrics

### Standard Metrics (Reuse Existing)
- Token accuracy (overall)
- Per-level accuracy (L0, L1, L2)
- Reconstruction MSE

### Pyramid-Specific Metrics (New)

1. **Conditional Accuracy**
   - Accuracy of p0 given p3 is predicted
   - Accuracy of p1 given p3+p2 are predicted
   - Demonstrates temporal dependency modeling

2. **Error Accumulation**
   - Track accuracy after each pass
   - Measure error propagation from coarse to fine

3. **Per-Pyramid-Level Performance**
   - Accuracy breakdown: p3, p2, p1, p0 separately
   - Identify which scales benefit most from autoregressive approach

### Comparison Script

```bash
python -m experiments.trajectory_completion.evaluation.compare_pyramid \
    --baseline_dir results/baseline \
    --pyramid_dir results/pyramid_autoregressive \
    --output_dir results/comparison
```

Generates:
- Side-by-side accuracy curves
- Per-level accuracy bar charts
- Reconstruction error comparison
- Inference time benchmarks

---

## Implementation Roadmap

### Phase 1: Core Architecture (2-3 days)

**Goal:** Basic 4-pass prediction working

**Tasks:**
1. Implement `PyramidTokenIndexer`
   - Parse VQ-VAE categories
   - Generate pyramid masks
   - Unit tests for indexing logic

2. Implement `PyramidAutoregressiveModel`
   - Multi-pass forward function
   - Token update logic
   - Test with dummy data

3. Integration test
   - Load real VQ-VAE
   - Run 4-pass prediction
   - Verify token propagation

**Validation:**
- Model executes 4 passes without errors
- Each pass predicts correct subset of tokens
- Intermediate predictions stored correctly

### Phase 2: Training (2 days)

**Goal:** Training loop with teacher forcing

**Tasks:**
1. Implement `PyramidAutoregressiveTrainer`
   - Teacher forcing scheduling
   - Per-level loss computation
   - Metric tracking

2. Update `run_experiment.py`
   - Add pyramid config support
   - Conditional model/trainer selection

3. Create config YAML

4. Test training
   - Run for 5 epochs
   - Verify teacher forcing annealing
   - Check checkpointing works

**Validation:**
- Training runs without errors
- Losses decrease over epochs
- Teacher forcing ratio anneals correctly
- Checkpoints save successfully

### Phase 3: Evaluation (1-2 days)

**Goal:** Full comparison with baseline

**Tasks:**
1. Run experiments
   - Baseline (parallel) - 20 epochs
   - Pyramid autoregressive - 30 epochs

2. Implement pyramid metrics
   - Conditional accuracy
   - Error accumulation analysis

3. Generate comparison plots

4. Measure inference time
   - Baseline: ~10ms expected
   - Pyramid: ~40ms expected

**Validation:**
- Both experiments complete successfully
- Pyramid achieves +2-5% accuracy improvement
- Reconstruction quality maintained
- Inference time ~4x slower (acceptable)

### Phase 4: Analysis (1 day)

**Goal:** Deep dive into temporal dependencies

**Tasks:**
1. Analyze conditional accuracies
   - p0 | p3 vs p0 alone
   - Quantify dependency strength

2. Error propagation study
   - Where do errors accumulate?
   - Which pyramid levels are most sensitive?

3. Visualizations
   - Predictions at each pass
   - Error heatmaps

4. Documentation
   - Update README
   - Add findings to experiment docs

**Deliverables:**
- Analysis report
- Recommendation: when to use pyramid vs parallel
- Future work suggestions

---

## Expected Results

### Performance Targets

| Metric | Baseline | Pyramid AR | Delta |
|--------|----------|------------|-------|
| Token Accuracy | 60% | 62-65% | +2-5% |
| Reconstruction MSE | 0.10 | 0.09 | -10% |
| Inference Time | 10ms | 40ms | 4x |
| Training Time | 2-3hr | 3-4hr | 1.2x |

### Success Criteria

**Must Achieve:**
- ✅ Token accuracy ≥ baseline (no regression)
- ✅ Training converges without instability
- ✅ Inference time ≤ 50ms (5x baseline acceptable)

**Stretch Goals:**
- 🎯 Token accuracy > baseline + 3%
- 🎯 Conditional accuracy demonstrates temporal dependencies
- 🎯 Reconstruction MSE < 0.09

### When to Use Pyramid vs Parallel

**Use Pyramid Autoregressive if:**
- Quality matters more than speed
- Working offline (not real-time inference)
- Want to study temporal dependencies
- Extreme masking cases (10%+10%)

**Use Parallel Baseline if:**
- Speed is critical (<20ms required)
- Real-time applications
- Quality ceiling already reached
- Simple baseline sufficient

---

## Technical Challenges

### Challenge 1: Identifying Pyramid Levels

**Problem:** Token-to-pyramid mapping depends on VQ-VAE category configuration

**Solution:**
- Parse category names at runtime
- Build mapping dynamically
- Validate against expected structure

**Mitigation:** Unit tests with various VQ-VAE configurations

### Challenge 2: Error Accumulation

**Problem:** Prediction errors propagate across passes

**Solution:**
- Teacher forcing with scheduled annealing
- Hierarchical guidance from base model
- Monitor conditional accuracies

**Mitigation:** If severe, increase teacher forcing duration or add auxiliary losses

### Challenge 3: Loss Weighting

**Problem:** How to weight losses from different pyramid passes?

**Options:**
1. Equal weighting (simple)
2. Weight by importance (coarse > fine)
3. Weight by token count

**Recommendation:** Start with equal weighting, experiment if needed

### Challenge 4: Initial Tokens

**Problem:** Initial tokens don't have pyramid structure

**Solutions:**
1. Include in first pass (with p3)
2. Always observe initial (only predict temporal)
3. Predict initial separately

**Recommendation:** Option 1 (include in first pass) - simplest

---

## Code Sketches

### Complete PyramidTokenIndexer

[See detailed code in full plan document]

### Complete PyramidAutoregressiveModel

[See detailed code in full plan document]

### Complete PyramidAutoregressiveTrainer

[See detailed code in full plan document]

---

## Future Extensions

### Short-Term (Week 4-5)

1. **Bidirectional Pyramid**
   - Predict both coarse→fine AND fine→coarse
   - Ensemble predictions

2. **Adaptive Skipping**
   - Skip pyramid levels with high confidence
   - Dynamic pass count per example

### Medium-Term (Week 6-8)

1. **Multi-Scale Loss**
   - Add reconstruction loss at intermediate pyramid levels
   - Encourage physically meaningful intermediate representations

2. **Attention-Based Conditioning**
   - Replace masking with explicit cross-attention
   - Coarse levels attend to fine levels

### Long-Term (Month 2+)

1. **Generative Modeling**
   - Sample diverse trajectories via pyramid
   - Conditional generation (fix p3, sample p0)

2. **Meta-Learning**
   - Operator-specific pyramid orderings
   - Transfer learning across parameter regimes

---

## Documentation Updates

### README Section

Add to `experiments/trajectory_completion/README.md`:

```markdown
## Pyramid Autoregressive Variant

Extends the baseline transformer with pyramid-aware multi-pass prediction.

### Quick Start
\`\`\`bash
python -m experiments.trajectory_completion.run_experiment \\
    --config baseline_50k/experiments/pyramid_autoregressive.yaml
\`\`\`

### How It Works
Predicts tokens in 4 sequential passes following temporal pyramid structure:
- Pass 1: initial + p3 (coarsest, 32 timesteps)
- Pass 2: p2 (64 timesteps) | condition on p3
- Pass 3: p1 (128 timesteps) | condition on p3+p2
- Pass 4: p0 (256 timesteps) | condition on p3+p2+p1

### Performance
- Accuracy: +2-5% over parallel baseline
- Inference: ~40ms (4x slower, acceptable)
- Best for: Offline analysis, quality-critical applications

### Configuration
\`\`\`yaml
pyramid:
  enabled: true
  teacher_forcing_schedule: "linear"
\`\`\`
```

---

## Critical Files for Implementation

1. **`experiments/trajectory_completion/models/completion_model.py`**
   - Base architecture to wrap/extend
   - Token embedding patterns
   - Hierarchical guidance implementation

2. **`experiments/trajectory_completion/training/trainer.py`**
   - Training loop template
   - Loss computation patterns
   - Metric tracking structure

3. **`experiments/common/models/trained_vqvae.py`**
   - VQ-VAE interface
   - Category name queries (needed for pyramid indexing)
   - Codebook sizes

4. **`experiments/trajectory_completion/data/masking.py`**
   - Masking strategy patterns
   - Can extend for pyramid-level masking

5. **`experiments/trajectory_completion/run_experiment.py`**
   - Integration point
   - Model instantiation logic
   - Config schema

---

## Summary

This plan provides a complete roadmap for implementing pyramid-aware autoregressive trajectory completion. The implementation:

1. **Extends existing code** (minimal changes to `run_experiment.py`)
2. **Adds ~800 lines** across 3 new files
3. **Reuses infrastructure** (dataset, VQ-VAE wrapper, base trainer)
4. **Delivers in 5-7 days** (phased implementation)
5. **Expects +2-5% accuracy** for 4x inference cost

**When to implement:** After transformer baseline results show >55% accuracy and clear hierarchical structure.

**Planning Agent ID:** a1d158b (can be resumed for clarifications)

---

**End of Plan**
