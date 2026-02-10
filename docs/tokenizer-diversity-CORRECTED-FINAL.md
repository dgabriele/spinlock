# VQTokenizer Diversity Analysis - CORRECTED FINAL REPORT

**Date**: 2026-02-09
**Model**: `checkpoints/vqvae/theta_baseline_50k`
**Dataset**: `datasets/50k_baseline.h5` (50,000 CNO ground truth samples)

---

## Executive Summary

**Initial Finding**: 39% of token categories produce only 1-2 unique tokens across 50K samples.

**Initial Misdiagnosis**: Claimed this was normal due to low feature variance, and that EMA cluster sizes showed 100% utilization (no mode collapse).

**CORRECTED DIAGNOSIS**: ⚠️ **SEVERE diversity limitation confirmed**
- **EMA cluster size is misleading** - shows 100% utilization due to training transients, NOT final behavior
- **Pretokenized dataset analysis is ground truth** - reveals real post-convergence utilization
- **Root cause**: 22.6% of temporal features have zero/near-zero variance in 50K training set
- **Impact**: Limits semantic diversity for downstream models (diffusion, MNO alignment)

---

## What We Got Wrong Initially

### Mistake #1: Trusting EMA Cluster Size as Utilization Metric

**What we observed**:
```python
quantizers.temporal_group_5_L0.ema_cluster_size: 28/28 codes "active" (all > 0)
```

**What we concluded**: ✅ No mode collapse, all codes are used.

**What's actually happening**:
```python
# Pretokenized dataset shows:
temporal_group_5_L0: Only token 2 is used (50,000 / 50,000 samples)

# But 27 other tokens have non-zero EMA:
Token 0: EMA=11.54 (NEVER used in final tokenization)
Token 1: EMA=23.09 (NEVER used in final tokenization)
...
Token 27: EMA=11.54 (NEVER used in final tokenization)
```

**Why this happens**:
1. **Early training**: Encoder not yet converged → varied code assignments
2. **EMA accumulation**: With decay 0.99, old assignments persist: `ema_new = 0.99 * ema_old + 0.01 * current`
3. **Late training**: Encoder converges → all samples use token 2
4. **Final state**: Token 2 dominates, but EMA still remembers old assignments

**Conclusion**: EMA cluster size tracks **cumulative historical usage**, not **current/final usage**.

---

## Corrected Analysis

### 1. Real Token Diversity (Ground Truth)

**Source**: `datasets/50k_baseline_tokenized_theta.h5` (frozen encoder applied to full 50K dataset)

**Findings**:
- **L0 (coarse)**: 8/23 categories (34.8%) use only 1 token
- **L1 (medium)**: 9/23 categories (39.1%) use only 1 token
- **L2 (fine)**: 10/23 categories (43.5%) use only 1 token

**Fully collapsed groups** (1 token across all 3 levels):
- `temporal_group_5`, `temporal_group_8`, `temporal_group_9`, `temporal_group_10`
- `temporal_group_12`, `temporal_group_13`, `temporal_group_14`, `temporal_group_19`

**Example: temporal_group_5**
| Level | Codebook Size | Tokens Used | Utilization |
|-------|---------------|-------------|-------------|
| L0    | 28            | 1 (token #2) | 3.6% |
| L1    | 6             | 1 (token #1) | 16.7% |
| L2    | 6             | 1 (token #0) | 16.7% |

---

### 2. Root Cause: Low Feature Variance

**Analysis**: 50K temporal features (shape: [50000, 256, 345], aggregated to [50000, 345])

| Variance Category | Count | Percentage | Implication |
|-------------------|-------|------------|-------------|
| Zero variance (< 1e-6) | 25 | 7.2% | Constant across all samples |
| Very low variance (< 0.01) | 53 | 15.4% | Nearly constant |
| **Total problematic** | **78** | **22.6%** | **Explain collapse** |
| Meaningful variance (> 0.01) | 267 | 77.4% | Diverse |

**Interpretation**:
- When features are constant, VQTokenizer correctly learns to encode them with a single code
- This is **correct compression** for low-diversity data
- But it limits downstream model capacity for semantic reasoning

---

### 3. Impact on Diffusion Model

**Diffusion validation accuracy**: 99.38%

**Breakdown**:
- 27/69 categories (39%) have 1-2 tokens → trivial to predict (high "free" accuracy)
- 42/69 categories (61%) have 3+ tokens → require real learning

**Adjusted accuracy estimate** (accounting for trivial categories):
```
Overall: 99.38% = (27 trivial + 42 non-trivial) / 69
Trivial: ~100% (predicting 1-2 options)
Non-trivial: ~98.9%
```

**Conclusion**: High accuracy is partially inflated by trivial categories, but non-trivial categories still show strong learning (~99%). However, overall semantic diversity is limited.

---

### 4. Why This Matters for Downstream Tasks

The low token diversity affects:

**Current Issue**:
- CNO tokenizer trained on 50K baseline dataset
- 22.6% of temporal features have low variance
- Results in 39% of token categories having 1-2 options

**Planned Downstream Work**:
1. **MNO Tokenizer**: Will inherit same data distribution issues
2. **Alignment Layer**: Limited semantic space to align (CNO has low diversity)
3. **Diffusion Generation**: Can only generate within narrow semantic range
4. **Agent Reasoning**: Limited diversity in generated rollouts

**Example Scenario**:
- Agent needs to explore diverse PDE behaviors
- But if CNO tokenizer can only represent narrow parameter space
- Then MNO-aligned generations will also be narrow
- Agent can't discover novel solutions outside training distribution

---

## Corrected Recommendations

### Option 1: Generate More Diverse Dataset (Recommended)

**Goal**: Increase feature variance to unlock full codebook capacity.

**Steps**:

1. **Audit current dataset diversity**:
   ```bash
   # Check which features have zero variance
   poetry run python -c "
   import h5py, numpy as np
   with h5py.File('datasets/50k_baseline.h5', 'r') as f:
       temporal = f['features/temporal/features'][:]
       temporal_agg = temporal.mean(axis=1)
       variance = temporal_agg.var(axis=0)
       zero_var = np.where(variance < 1e-6)[0]
       print(f'Zero variance features: {zero_var.tolist()}')
   "
   ```

2. **Generate new diverse dataset**:
   - Use Sobol sampling over **wider PDE parameter ranges**
   - Include more extreme initial conditions
   - Verify parameter coverage spans full [0,1]^14 hypercube
   - Target: 100K-200K samples with high diversity

3. **Retrain CNO tokenizer** on diverse dataset
   - Expect: 3-10 tokens per category (vs. current 1-2)
   - Better semantic coverage

4. **Then proceed with MNO tokenizer + alignment**

**Expected Impact**:
- ✅ Richer semantic space for diffusion
- ✅ Better MNO alignment (more CNO codes to map to)
- ✅ Agent can explore wider range of behaviors

---

### Option 2: Accept Limited Diversity (Not Recommended)

**If**: The 50K dataset accurately represents the "interesting" region of PDE space.

**Then**: Current tokenizer is optimal for this narrow distribution.

**Risks**:
- Agent limited to known parameter regimes
- Can't discover novel PDE behaviors
- MNO alignment only works for familiar cases

**When to choose this**:
- If downstream goal is interpolation/refinement (not exploration)
- If 50K distribution is intentionally narrow (e.g., "realistic physics only")

---

### Option 3: Investigate Feature Extraction

**If**: Zero-variance features are unexpected (possible bug).

**Action**: Audit which temporal features are constant and why.

**Example**:
```python
import h5py, numpy as np

with h5py.File('datasets/50k_baseline.h5', 'r') as f:
    temporal = f['features/temporal/features'][:]
    temporal_agg = temporal.mean(axis=1)  # [50000, 345]
    variance = temporal_agg.var(axis=0)

    zero_var_idx = np.where(variance < 1e-6)[0]
    print(f'Zero-variance feature indices: {zero_var_idx.tolist()}')

    # Check if they're padding, disabled, or bugs
    for idx in zero_var_idx[:5]:
        values = temporal_agg[:10, idx]
        print(f'Feature {idx}: {values} (all identical?)')
```

**Questions**:
- Are these supposed to be constant (padding, architecture flags)?
- Is feature extraction broken?
- Should they be removed from tokenizer input?

---

### Option 4: Fix Dashboard Visualization (Low Priority)

**Issue**: Dashboard shows 0.00% utilization (wrong metric).

**Fix**: Update `src/spinlock/visualization/vqvae/utils.py:extract_utilization_matrix()` to compute utilization from pretokenized dataset, NOT from EMA.

**Better metric**:
```python
def extract_utilization_matrix_from_tokens(
    tokenized_dataset_path: Path,
    data: VQVAECheckpointData
) -> np.ndarray:
    """Compute real utilization from pretokenized dataset."""
    with h5py.File(tokenized_dataset_path, 'r') as f:
        tokens_group = f['tokens']
        matrix = np.zeros((data.num_categories, data.num_levels))

        for i, cat in enumerate(data.category_names):
            for level in range(data.num_levels):
                key = f"{cat}_L{level}"
                if key in tokens_group:
                    tokens = tokens_group[key][:]
                    codebook_size = get_codebook_size(data, cat, level)
                    unique_tokens = len(np.unique(tokens))
                    matrix[i, level] = unique_tokens / codebook_size

        return matrix
```

---

## Validation Commands

### Check Real Utilization (Pretokenized Dataset)
```bash
poetry run python scripts/validation/compare_tokens_to_ema.py
```

### Check Feature Variance
```bash
poetry run python scripts/validation/analyze_collapsed_groups.py
```

### Full Diversity Report
```bash
poetry run python scripts/validation/verify_tokenizer_diversity.py
```

---

## Conclusion

### What We Learned

1. ❌ **EMA cluster size is misleading** for diversity assessment
2. ✅ **Pretokenized dataset analysis is ground truth**
3. ⚠️ **39% of categories have severe collapse** (1-2 tokens only)
4. 🔍 **Root cause**: 22.6% of temporal features have low/zero variance
5. 📉 **Impact**: Limits downstream semantic diversity

### Immediate Action Required

**Before proceeding with MNO tokenizer and alignment layer**:

1. ✅ **Generate more diverse dataset** (100K-200K samples, wider parameter ranges)
2. ✅ **Retrain CNO tokenizer** on diverse data
3. ✅ **Verify token diversity** (target: 3-10 tokens per category)
4. ✅ **Then train MNO tokenizer** (will inherit better diversity)

**Timeline Impact**:
- New dataset generation: ~1-2 hours
- CNO tokenizer retraining: ~2-4 hours (50 epochs)
- Verification: ~30 minutes
- **Total delay**: ~4-7 hours

**Benefit**:
- Much richer semantic space for agent reasoning
- Better MNO→CNO alignment
- Avoid having to retrain everything later

---

## Appendix: Evidence

### Proof that EMA is Misleading

```
temporal_group_5_L0:
  EMA cluster size: 28/28 codes "active" (all > 0)
  Real usage: 1/28 codes (only token #2)
  Unused tokens with non-zero EMA: 27

  Token 0: EMA=11.54 (NEVER used in final tokenization)
  Token 1: EMA=23.09 (NEVER used in final tokenization)
  ...
  Token 2: EMA=166.78 (100% of 50K samples)
  ...
  Token 27: EMA=11.54 (NEVER used in final tokenization)
```

**Interpretation**: EMA accumulated assignments from early training (when encoder was random). After convergence, only token 2 is used, but EMA remembers the transients.

---

**Status**: ⚠️ **Diversity issue confirmed. Dataset regeneration recommended before MNO work.**
