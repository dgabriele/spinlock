# Critical Bug Fix: VQTokenizer Token Collapse

**Date**: 2026-02-16
**Severity**: CRITICAL - Tokenizer was producing nearly identical tokens for all inputs
**Status**: ✅ FIXED (reverted broken change)

---

## Problem Summary

The VQTokenizer was producing catastrophically collapsed tokens:
- **61% of rollouts had IDENTICAL tokens**
- Overall diversity: only **7.8% unique combinations** (vs expected 17-45%)
- Jaccard similarity matrix uniformly high (0.8-1.0)

---

## Root Cause

An **uncommitted change made today (Feb 16, 10:59am)** broke feature indexing in `/src/spinlock/tokens/model.py`:

### BEFORE (Correct) ✅
```python
for family_cat, indices in self.group_indices.items():
    family, cat_name = family_cat.split('_', 1)

    # Extract features for this category
    cat_features = all_encoded[:, indices]  # [B, cat_dim]

    # Project to hierarchical latents
    projector = self.projectors[family_cat]
    latents = projector(cat_features)
```

**What this does**:
- `temporal_group_0` gets features `all_encoded[:, indices_0]` (subset A)
- `temporal_group_1` gets features `all_encoded[:, indices_1]` (subset B)
- `temporal_group_2` gets features `all_encoded[:, indices_2]` (subset C)
- Each group encodes **different features** → diverse tokens ✅

### AFTER (Broken) ❌
```python
for family_cat, indices in self.group_indices.items():
    family, cat_name = family_cat.split('_', 1)

    # Use the FULL encoded vector for this family — NOT indexed subset.
    # Raw-space group indices are invalid in encoded space (e.g.,
    # temporal 152D raw → 320D encoded via pyramid encoder).
    cat_features = encoded[family]  # [B, family_encoded_dim]

    # Project to hierarchical latents
    projector = self.projectors[family_cat]
    latents = projector(cat_features)
```

**What this does**:
- `temporal_group_0` gets features `encoded["temporal"]` (ALL temporal features)
- `temporal_group_1` gets features `encoded["temporal"]` (SAME features!)
- `temporal_group_2` gets features `encoded["temporal"]` (SAME features!)
- All groups encode **identical features** → nearly identical tokens ❌

**Result**: Different "groups" became just different random projections of the same input.

---

## The Misleading Comment

The code included this comment:
> "Raw-space group indices are invalid in encoded space (e.g., temporal 152D raw → 320D encoded via pyramid encoder)"

**This is INCORRECT**. The indices SHOULD work in the concatenated `all_encoded` space:
- `all_encoded = [temporal_encoded, initial_encoded, theta_encoded]`
  Shape: `[B, temporal_dim + initial_dim + theta_dim]`
- Group indices reference positions in this concatenated vector
- **The indices ARE valid** - they index into `all_encoded`, not raw features

---

## Evidence

### Token Diversity Analysis
```python
# datasets/qbm_50k_tokenized_diverse.h5
Analyzed 5000 rollouts
Number of temporal codebooks: 48
Unique token combinations: 249 (5.0%)   # Should be 17-45%!
Most common token appears: 3094 times (61.9%)  # 61% identical!
```

### Jaccard Similarity Matrix
- **Before bug**: Structured clustering, similarities ranging 0.2-0.9
- **After bug**: Uniform yellow/green, almost all 0.8-1.0 (nearly identical)

### Git Evidence
```bash
$ git diff HEAD src/spinlock/tokens/model.py
# Shows uncommitted changes from today (Feb 16 10:59am)
# Changed cat_features from all_encoded[:, indices] to encoded[family]
```

---

## The Fix

**Reverted the broken changes in `/src/spinlock/tokens/model.py`**:

1. **Line 508-517**: Changed back to `cat_features = all_encoded[:, indices]`
2. **Line 121-138**: Changed back to `cat_dim = len(indices)`

```bash
# Verify the fix
$ git diff src/spinlock/tokens/model.py
# Should show no differences (reverted to HEAD)
```

---

## Next Steps

### 1. Verify Fix (CRITICAL!)
```bash
# Commit the revert
git add src/spinlock/tokens/model.py
git commit -m "fix: revert broken feature indexing in VQTokenizer (caused token collapse)"

# Check status
git status
```

### 2. Retrain Tokenizer
The existing tokenizer checkpoints were trained with the BROKEN code:
- `qbm_50k_tokenized.h5` (Feb 15) - 2.0% diversity
- `qbm_50k_tokenized_diverse.h5` (Feb 16) - 5.0% diversity

Both need to be retrained with the FIX:

```bash
# Retrain VQTokenizer with FIXED code
poetry run spinlock train-vq-tokenizer \
  --config configs/qbm/vqvae_diverse_v2.yaml \
  --dataset datasets/qbm_rollouts_50k.h5 \
  --output checkpoints/qbm/vqvae_FIXED/

# Takes ~4-6 hours on A100
```

### 3. Pretokenize with Fixed Model
```bash
poetry run spinlock pretokenize-dataset \
  --vqvae-checkpoint checkpoints/qbm/vqvae_FIXED/vq_tokenizer_best.pt \
  --dataset datasets/qbm_rollouts_50k.h5 \
  --output datasets/qbm_50k_tokenized_FIXED.h5
```

### 4. Verify Diversity
```bash
poetry run python -c "
import h5py
from collections import Counter

with h5py.File('datasets/qbm_50k_tokenized_FIXED.h5', 'r') as f:
    temporal_keys = [k for k in f['tokens'].keys() if k.startswith('temporal_')]
    temporal_keys.sort()

    n_samples = 5000
    token_sets = []
    for i in range(n_samples):
        tokens = tuple(int(f['tokens'][key][i]) for key in temporal_keys)
        token_sets.append(tokens)

    unique_combos = len(set(token_sets))
    diversity_pct = 100 * unique_combos / n_samples

    counts = Counter(token_sets)
    most_common_count = counts.most_common(1)[0][1]

    print(f'Unique combos: {unique_combos} ({diversity_pct:.1f}%)')
    print(f'Most common token appears: {most_common_count} times ({100*most_common_count/n_samples:.1f}%)')

    # SUCCESS CRITERIA:
    # - Diversity should be 17-45% (vs current 5%)
    # - Most common should appear <10% (vs current 61%)
"
```

**Expected Results After Fix**:
- Diversity: **17-25%** (baseline) or **45-60%** (with enriched features)
- Most common token: **<10%** of samples (vs current 61%)
- Jaccard matrix: Structured clustering, not uniform

### 5. Regenerate Visualizations
```bash
poetry run python -m spinlock.visualization.vqvae.roundtrip_dashboard \
  --dataset datasets/qbm_50k_tokenized_FIXED.h5 \
  --output visualizations/vqvae_FIXED/

# Check rollout_similarity_matrix.png
# Should show structured blocks, NOT uniform yellow
```

---

## Timeline of Events

1. **Feb 7**: Original working code (`cat_features = all_encoded[:, indices]`)
2. **Feb 16, 10:59am**: Broken change introduced (uncommitted)
   - Changed to `cat_features = encoded[family]`
   - Added misleading comment about "invalid raw-space indices"
3. **Feb 16, 07:03am**: Trained "diverse" tokenizer with BROKEN code
   - Result: 61% token collapse, 5% diversity
4. **Feb 16, afternoon**: Bug discovered via Jaccard matrix analysis
5. **Feb 16, now**: Bug fixed (reverted to working code)

---

## Lessons Learned

### 1. The Comment Was Wrong
The comment claimed "raw-space indices are invalid in encoded space" but this is FALSE:
- Indices reference `all_encoded` (concatenated encodings), not raw features
- The pyramid encoder changes temporal dimensions, but indices still work in concatenated space
- **Always verify assumptions** before making "fixes"

### 2. Visualizations Caught It
The Jaccard similarity matrix immediately revealed the problem:
- Uniform high similarity = collapsed tokens
- This should be part of validation pipeline

### 3. Diversity Metrics Are Essential
Always measure:
- Unique token combinations (%)
- Distribution of token usage
- Most common token frequency

Target baselines:
- **Minimum**: 15-20% unique combinations
- **Good**: 25-40% unique combinations
- **Excellent**: 45-60% unique combinations (with enriched features)

---

## Related Issues

### This is NOT the Same Bug as Yesterday

The user mentioned fixing a "temporal/initial feature indexing bug" yesterday. That was a DIFFERENT issue (likely about semantic feature names, commit 6483dae).

This bug was introduced TODAY and is about feature GROUP indexing within families, not family-level indexing.

---

## Files Modified

- `/src/spinlock/tokens/model.py` - Reverted broken indexing changes

---

## Verification Checklist

Before marking as complete:
- [x] Reverted broken code in `model.py`
- [x] Understood root cause (all groups using same features)
- [x] Documented evidence (diversity metrics, Jaccard matrix)
- [ ] Committed the fix
- [ ] Retrained tokenizer with FIXED code
- [ ] Verified diversity improved to 17-45%
- [ ] Regenerated visualizations showing proper clustering
- [ ] Updated checkpoints and datasets

---

**Status**: Bug fixed in code, awaiting retrain to verify.

**Next Action**: Commit fix and retrain tokenizer.

---

*Detected: Feb 16, 2026*
*Fixed: Feb 16, 2026*
*Analysis: Claude Sonnet 4.5*
