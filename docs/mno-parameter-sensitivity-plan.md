# MNO Parameter Sensitivity Implementation Plan

**Date**: 2026-02-08
**Goal**: Fix weak parameter conditioning in MNO to enable dual tokenizer approach
**Current State**: MNO produces nearly identical rollouts regardless of parameters (diversity ratio: 0.006)
**Target State**: MNO produces diverse, parameter-specific rollouts (diversity ratio: >0.1)

---

## Problem Analysis

### Current Metrics (Broken)
- **Pairwise MSE**: 0.000488 (nearly identical outputs)
- **Diversity Ratio**: 0.006 (parameters have 166x LESS effect than time)
- **Variance Across Parameters**: 0.000244 (tiny)
- **Temporal Variance**: 0.081 (much larger)

### Target Metrics (Fixed)
- **Pairwise MSE**: >0.01 (20x increase)
- **Diversity Ratio**: >0.1 (16x increase)
- **Parameter Reconstruction Accuracy**: >80%
- **Contrastive Discrimination**: >90%

### Root Causes
1. **No parameter sensitivity in loss**: Pure MSE doesn't penalize ignoring parameters
2. **Weak FiLM gradients**: FiLM weights stayed near identity initialization
3. **Model shortcut**: Producing "average" dynamics minimizes loss easier than parameter-specific dynamics

---

## Implementation Phases

### Phase 1: Implement Loss Components (1-2 hours)

**Step 1.1: Parameter Reconstruction Loss**
- **File**: `src/spinlock/mno/losses/parameter_reconstruction.py` (NEW)
- **Purpose**: Force model to preserve parameter information in rollouts
- **Architecture**:
  ```python
  class ParameterReconstructor(nn.Module):
      """Predicts parameters from rollout features."""
      def __init__(self, rollout_feature_dim: int, param_dim: int = 14):
          # Extract rollout features (mean, std, spectral, etc.)
          # MLP: features → 256 → 128 → 14 params
          pass

      def forward(self, rollout: Tensor) -> Tensor:
          # Extract features from rollout
          features = self.extract_features(rollout)  # [B, feature_dim]
          # Predict parameters
          params_pred = self.mlp(features)  # [B, 14]
          return params_pred
  ```
- **Loss**: `MSE(params_pred, params_true)`
- **Weight**: λ_param_recon = 0.5

**Step 1.2: Contrastive Loss**
- **File**: `src/spinlock/mno/losses/contrastive.py` (NEW)
- **Purpose**: Ensure different parameters → different outputs
- **Architecture**:
  ```python
  def contrastive_loss(rollouts: Tensor, params: Tensor,
                       temperature: float = 0.1) -> Tensor:
      """
      InfoNCE-style contrastive loss.

      For each rollout i:
      - Positive: rollout i with params i
      - Negatives: rollouts j≠i with params j≠i

      Goal: Maximize similarity between rollout and its params,
            Minimize similarity to other params.
      """
      # Embed rollouts and params into same space
      rollout_features = extract_features(rollouts)  # [B, D]
      param_features = embed_params(params)  # [B, D]

      # Compute similarities
      logits = (rollout_features @ param_features.T) / temperature  # [B, B]
      labels = torch.arange(B)  # Diagonal are positives

      # Cross-entropy loss
      loss = F.cross_entropy(logits, labels)
      return loss
  ```
- **Weight**: λ_contrastive = 0.3

**Step 1.3: Parameter Sensitivity Regularization**
- **File**: `src/spinlock/mno/losses/sensitivity.py` (NEW)
- **Purpose**: Penalize if parameter changes don't affect outputs
- **Implementation**:
  ```python
  def parameter_sensitivity_loss(mno, ic, params, epsilon=0.01):
      """
      Measure output sensitivity to parameter changes.

      Use finite differences:
      - rollout_0 = MNO(ic, params)
      - rollout_1 = MNO(ic, params + ε)
      - sensitivity = ||rollout_1 - rollout_0|| / ε

      Penalize if sensitivity is too low.
      """
      rollout_0 = mno(ic, params)

      # Perturb parameters
      params_perturbed = params + epsilon * torch.randn_like(params)
      rollout_1 = mno(ic, params_perturbed)

      # Measure sensitivity
      diff = (rollout_1 - rollout_0) / epsilon
      sensitivity = torch.mean(diff ** 2)

      # Target: sensitivity should be similar to temporal variance
      temporal_var = torch.var(rollout_0, dim=1).mean()
      target_sensitivity = 0.1 * temporal_var  # 10% of temporal dynamics

      # Penalize deviation from target
      loss = F.mse_loss(sensitivity, target_sensitivity)
      return loss
  ```
- **Weight**: λ_sensitivity = 0.2

**Step 1.4: Integrate into Training Loop**
- **File**: `src/spinlock/mno/training.py` (MODIFY)
- **Changes**:
  ```python
  # Current loss
  loss = lambda_traj * L_traj + lambda_ic * L_ic

  # NEW: Add parameter sensitivity losses
  loss += lambda_param_recon * L_param_recon
  loss += lambda_contrastive * L_contrastive
  loss += lambda_sensitivity * L_sensitivity
  ```

---

### Phase 2: Implement FiLM Learning Rate Scaling (30 min)

**Step 2.1: Separate FiLM Parameters**
- **File**: `src/spinlock/operators/film.py` (MODIFY)
- **Changes**:
  ```python
  # In optimizer setup:
  film_params = [p for n, p in model.named_parameters() if 'film' in n.lower()]
  other_params = [p for n, p in model.named_parameters() if 'film' not in n.lower()]

  optimizer = torch.optim.Adam([
      {'params': other_params, 'lr': learning_rate},
      {'params': film_params, 'lr': learning_rate * 5.0}  # 5x higher LR
  ])
  ```

---

### Phase 3: Training & Validation (1-2 days)

**Step 3.1: Baseline Validation**
- Generate 20 test rollouts with current (broken) MNO
- Record diversity metrics as baseline
- Save to: `diagnostics/mno_baseline_diversity.json`

**Step 3.2: Train Parameter-Sensitive MNO**
```bash
poetry run spinlock train-meta-operator \
  --config configs/50k_baseline/mno/parameter_sensitive.yaml \
  --output checkpoints/mno/parameter_sensitive \
  --verbose
```
- **Expected time**: ~24-36 hours
- **Monitoring**: Track all loss components separately
- **Checkpoints**: Save every epoch

**Step 3.3: Validation During Training**
- Every 5 epochs: Generate 20 test rollouts
- Compute diversity metrics
- Plot diversity ratio over training
- Early stopping if diversity ratio > 0.1

**Step 3.4: Final Validation**
- Generate 50 test rollouts with trained MNO
- Compute full diversity analysis
- Compare to target metrics
- Document results

---

### Phase 4: Diagnostic & Iteration (if needed)

**If diversity ratio < 0.05 after training:**

**Diagnosis A: FiLM weights didn't change**
- Inspect: `model.film_generator` weights
- Check: Gradient magnitudes during training
- Fix: Increase FiLM LR multiplier to 10x

**Diagnosis B: Loss components fighting**
- Check: Loss component evolution plots
- Look for: One loss dominating, others going to zero
- Fix: Re-balance loss weights

**Diagnosis C: Model capacity issue**
- Check: Is model just memorizing average?
- Try: Increase model size or reduce dataset size

---

## Success Criteria

### Minimum Viable
- ✓ Diversity ratio > 0.05 (10x improvement)
- ✓ Parameter reconstruction accuracy > 60%
- ✓ Contrastive discrimination > 70%

### Target
- ✓ Diversity ratio > 0.1 (16x improvement)
- ✓ Parameter reconstruction accuracy > 80%
- ✓ Contrastive discrimination > 90%
- ✓ Visual inspection: Rollouts look clearly different

### Stretch
- ✓ Diversity ratio > 0.2 (33x improvement)
- ✓ MNO-CNO MSE competitive with baseline (< 2.0)

---

## Risk Mitigation

**Risk 1: Training diverges with new losses**
- Mitigation: Start with low loss weights (0.1x), gradually increase
- Fallback: Remove sensitivity loss first (most experimental)

**Risk 2: Parameter reconstruction is too hard**
- Mitigation: Use simpler extractor (just mean/std of rollout)
- Fallback: Reduce λ_param_recon to 0.1

**Risk 3: Contrastive loss conflicts with trajectory loss**
- Mitigation: Use separate projector for contrastive vs reconstruction
- Fallback: Disable contrastive, rely on param_recon + sensitivity

**Risk 4: Takes too long to train**
- Mitigation: Reduce to 10K samples, 2 epochs
- Monitor: If diversity improving by epoch 1, continue; else abort

---

## Timeline

| Phase | Task | Time | Dependencies |
|-------|------|------|--------------|
| 1.1 | Parameter reconstruction loss | 30 min | - |
| 1.2 | Contrastive loss | 30 min | - |
| 1.3 | Sensitivity loss | 30 min | - |
| 1.4 | Integrate into training | 30 min | 1.1, 1.2, 1.3 |
| 2.1 | FiLM LR scaling | 30 min | - |
| 3.1 | Baseline validation | 15 min | - |
| 3.2 | Training | 24-36 hrs | 1.4, 2.1 |
| 3.3 | Online validation | During 3.2 | 3.2 |
| 3.4 | Final validation | 30 min | 3.2 |
| 4.x | Diagnostics (if needed) | 2-4 hrs | 3.4 |

**Total**: ~28-40 hours (mostly unattended training)

---

## Next Steps

1. **Implement Phase 1**: Create the three new loss components
2. **Quick test**: Train for 1 epoch on small subset (1K samples)
3. **Validate**: Check if diversity ratio starts improving
4. **Full training**: If promising, run full 3-epoch training
5. **Iterate**: Adjust loss weights based on results

---

## Alternative: Quick Win Approach

If timeline is critical, try this simpler approach first:

**Minimal Implementation** (4 hours total):
1. Only implement parameter reconstruction loss (skip contrastive & sensitivity)
2. Only increase FiLM LR (5x multiplier)
3. Train for 1 epoch on 10K samples

This might be sufficient to get diversity ratio > 0.05, which could be "good enough" for the dual tokenizer approach to work.

If it works → continue with full implementation
If it doesn't → fall back to full 3-loss approach
