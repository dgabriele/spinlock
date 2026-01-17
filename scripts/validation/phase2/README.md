# Phase 2 Validation Experiments

Validation experiments for perturbation framework and behavioral encoding.

## Overview

These experiments validate that the MNO responds meaningfully to perturbations when operated autonomously (without parameter conditioning), and that the behavioral encoding framework can capture and distinguish these responses.

## Experiments

### 1. Perturbation Response Divergence (`01_perturbation_response_divergence.py`)

**Hypothesis:** Different perturbations → different token sequences

**Success Criteria:**
- 90%+ of perturbation pairs produce divergent token sequences (>10% Hamming distance)
- Mean divergence > 0.3 across all pairs
- Spatial sensitivity: Corner blobs produce different responses than center blobs

**Methodology:**
1. Sample N=10 initial conditions
2. For each IC, apply 13 diverse perturbations:
   - Center blobs (3 amplitudes: 0.5, 1.0, 2.0)
   - Corner blobs (4 corners: TL, TR, BL, BR)
   - Uniform forcing (2 amplitudes)
   - Fourier modes (3 wavelengths: (1,1), (2,2), (1,2))
3. Compute token sequence divergence (Hamming distance)
4. Visualize divergence heatmaps

**Output:**
- `divergence_matrix_ic{i}.png` - Per-IC heatmaps
- `divergence_matrix_mean.png` - Aggregate across ICs
- `summary.txt` - Statistical summary and pass/fail

**Usage:**
```bash
python 01_perturbation_response_divergence.py \
    --mno-checkpoint checkpoints/mno/pure_mse_baseline/meta_operator_best.pt \
    --vqvae-checkpoint checkpoints/vqvae/mno_distribution_100k/vqvae_best.pt \
    --n-ics 10 \
    --output-dir results/phase2/exp1
```

---

### 2. Token Regime Clustering (`02_token_regime_clustering.py`)

**Hypothesis:** Token-based clustering captures behavioral regimes better than spatial clustering

**Success Criteria:**
- Token clustering silhouette score ≥ spatial clustering score
- Token clusters correspond to interpretable behavioral patterns
- Cluster purity > 0.7 when validated against known perturbation types

**Methodology:**
1. Generate 500 episodes with diverse perturbations/ICs
2. Extract behavioral signatures from token sequences
3. Cluster via K-means (K=5, 10, 15)
4. Compare token clustering vs spatial trajectory clustering
5. Visualize cluster characteristics

**Output:**
- `cluster_comparison.png` - Silhouette scores (token vs spatial)
- `cluster_visualization.png` - t-SNE embeddings with cluster labels
- `cluster_characteristics.txt` - Mean signatures per cluster
- `summary.txt` - Pass/fail assessment

**Usage:**
```bash
python 02_token_regime_clustering.py \
    --mno-checkpoint <path> \
    --vqvae-checkpoint <path> \
    --n-episodes 500 \
    --output-dir results/phase2/exp2
```

---

### 3. Early Stopping Efficiency (`03_early_stopping_efficiency.py`)

**Hypothesis:** Early stopping saves 30-50% computation vs fixed max_steps=256

**Success Criteria:**
- Mean stopping time < 180 steps (30% savings)
- 40%+ episodes stop before max_steps
- Stopped episodes achieve stable states (low ||Δu|| or token stability)

**Methodology:**
1. Run 1000 episodes with StandardStoppingPolicy
2. Track stopping times and reasons
3. Compare computational cost vs always running 256 steps
4. Validate stopped states are genuinely stable

**Output:**
- `stopping_time_distribution.png` - Histogram of stopping times
- `stopping_reasons.png` - Pie chart of stop reasons
- `convergence_validation.png` - ||Δu|| at stopping time
- `summary.txt` - Savings metrics and pass/fail

**Usage:**
```bash
python 03_early_stopping_efficiency.py \
    --mno-checkpoint <path> \
    --vqvae-checkpoint <path> \
    --n-episodes 1000 \
    --output-dir results/phase2/exp3
```

---

### 4. Reproducibility Testing (`04_reproducibility.py`)

**Hypothesis:** Same (u₀, perturbation) → same token sequence (>0.95 similarity)

**Success Criteria:**
- 95%+ episodes have token similarity > 0.95 across 5 runs
- Mean token similarity > 0.98
- Behavioral signature variance < 5% across runs

**Methodology:**
1. Create 100 (u₀, perturbation) pairs
2. Run each pair 5 times
3. Compute token sequence similarity across runs
4. Test determinism of MNO autonomous evolution

**Output:**
- `similarity_distribution.png` - Histogram of within-pair similarities
- `reproducibility_heatmap.png` - Similarity matrix for sample pairs
- `summary.txt` - Pass/fail and outlier analysis

**Usage:**
```bash
python 04_reproducibility.py \
    --mno-checkpoint <path> \
    --vqvae-checkpoint <path> \
    --n-pairs 100 \
    --n-runs 5 \
    --output-dir results/phase2/exp4
```

---

## Running All Experiments

```bash
# Set paths
MNO_CKPT=checkpoints/mno/pure_mse_baseline/meta_operator_best.pt
VQVAE_CKPT=checkpoints/vqvae/mno_distribution_100k/vqvae_best.pt

# Run all experiments
python 01_perturbation_response_divergence.py --mno-checkpoint $MNO_CKPT --vqvae-checkpoint $VQVAE_CKPT
python 02_token_regime_clustering.py --mno-checkpoint $MNO_CKPT --vqvae-checkpoint $VQVAE_CKPT
python 03_early_stopping_efficiency.py --mno-checkpoint $MNO_CKPT --vqvae-checkpoint $VQVAE_CKPT
python 04_reproducibility.py --mno-checkpoint $MNO_CKPT --vqvae-checkpoint $VQVAE_CKPT
```

## Success Summary

Phase 2 validation is considered successful if ALL experiments pass:
- ✓ Experiment 1: Perturbation divergence ≥ 90%
- ✓ Experiment 2: Token clustering silhouette ≥ spatial clustering
- ✓ Experiment 3: Early stopping saves ≥ 30% computation
- ✓ Experiment 4: Reproducibility ≥ 95%

If all pass → Phase 2 VALIDATED, proceed to Phase 3 implementation
If any fail → Investigate failure modes, iterate on design

## TODO: Integration

These scripts currently use placeholder model loading. Once MNO and VQ-VAE are integrated:

1. Replace `load_models()` with actual checkpoint loading
2. Implement `sample_initial_condition()` using proper IC sampling
3. Uncomment `EpisodeRunner.run_episode()` calls
4. Update `_mno_step()` in `episode.py` to use actual MNO forward pass

Then run full validation suite and update results.
