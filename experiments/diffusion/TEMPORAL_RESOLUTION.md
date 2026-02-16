# Temporal Resolution D3PM

## Overview

Temporal Resolution D3PM extends discrete diffusion to leverage the VQTokenizer's variable-length training by diffusing tokenizations at multiple truncation lengths, showing how the token representation "resolves" as more of the trajectory is observed.

## Key Innovation: Meta-Diffusion

The temporal resolution approach creates a **meta-level alignment**: the generative process (diffusion) mirrors the representation process (temporal resolution).

- **Standard D3PM**: Diffuses only the final tokenization at full length (256 timesteps) → single token set (~50 tokens)
- **Temporal Resolution D3PM**: Diffuses tokenizations at multiple truncation points [32, 64, 128, 256] → ~234 tokens

The VQTokenizer was already trained on variable-length sequences (pyramid binning), so it knows how to represent partial trajectories. This architecture asks: **"What's the relationship between early-time tokenization (t=32) and final-time tokenization (t=256)?"**

## Architecture

### Token Structure

```python
{
  # Temporal features at all truncation levels
  "temporal_group_1_trunc_T032_L0": [B],  # First 32 steps, coarse VQ
  "temporal_group_1_trunc_T032_L1": [B],  # First 32 steps, mid VQ
  "temporal_group_1_trunc_T032_L2": [B],  # First 32 steps, fine VQ

  "temporal_group_1_trunc_T064_L0": [B],  # First 64 steps
  ...
  "temporal_group_1_trunc_T256_L0": [B],  # Full trajectory

  # Initial/theta features (no truncation)
  "initial_group_1_L0": [B],
  ...
}
```

**Token count**:
- Temporal: 4 truncations × 3 VQ levels × ~17 groups ≈ 204 tokens
- Initial/theta: ~30 tokens (unchanged)
- **Total**: ~234 tokens (vs 50 baseline)

### Causal Temporal Attention

The TemporalResolutionDenoisingNetwork learns causal dependencies between truncation levels:

1. **Truncation Embeddings**: Embed which truncation level each token came from
2. **Learnable Temporal Bias**: [4, 4] attention bias matrix encoding temporal dependencies
3. **Causal Masking**: Enforce that future cannot attend to past (T256 cannot guide T032)

```python
# Causal attention bias (initialized)
# Rows = source truncation, Cols = target truncation
#          T032  T064  T128  T256
# T032  [  0.0   0.1   0.2   0.3 ]  # Early guides late: positive bias
# T064  [ -∞    0.0   0.1   0.2 ]  # Block non-causal: -∞
# T128  [ -∞    -∞    0.0   0.1 ]
# T256  [ -∞    -∞    -∞    0.0 ]
```

The network **learns** to refine this bias during training.

## Requirements

1. **VQTokenizer with pyramid temporal encoder**:
   - Required config: `encoder.temporal.variant: "pyramid"`
   - Truncation lengths derived from `downsample_factors`

2. **Pre-tokenized dataset with temporal resolution**:
   - Must be generated with `--temporal-resolution` flag
   - Stores tokens at all truncation levels

## Usage

### Step 1: Pretokenize Dataset

```bash
# Generate temporal resolution dataset
spinlock pretokenize-dataset \
  --dataset datasets/qbm_50k_baseline.h5 \
  --tokenizer checkpoints/v2/vqvae/vq_tokenizer_best.pt \
  --output datasets/qbm_50k_temporal_resolution.h5 \
  --temporal-resolution \
  --batch-size 128
```

This will:
- Check tokenizer uses pyramid encoder
- Extract truncation lengths from config (e.g., [32, 64, 128, 256])
- Tokenize at each truncation point
- Store with truncation suffixes (e.g., `_trunc_T064_`)

**Expected runtime**: ~2-3 hours on GPU for 50K samples

**Expected storage**: ~140 MB (with gzip compression)

### Step 2: Train Temporal Resolution D3PM

```bash
# Train for 100 epochs
python experiments/diffusion/scripts/train_temporal_resolution.py \
  --config experiments/diffusion/configs/temporal_resolution.yaml
```

**Expected training time**:
- Single GPU: ~32 hours per 100 epochs (~4x baseline due to larger token count)
- Multi-GPU (4×): ~8 hours per 100 epochs

### Step 3: Evaluate

Use standard diffusion evaluation scripts (they will automatically handle the multi-truncation tokens):

```bash
# Generate samples
python experiments/diffusion/scripts/generate_samples.py \
  --checkpoint experiments/diffusion/results/temporal_resolution/temporal_resolution_d3pm_best.pt \
  --num-samples 1000
```

## Configuration

See `experiments/diffusion/configs/temporal_resolution.yaml` for full config.

Key settings:

```yaml
model:
  # Standard diffusion settings
  hidden_dim: 256
  num_layers: 6
  num_heads: 8

  # Temporal resolution config
  temporal_resolution:
    enabled: true
    use_temporal_bias: true  # Learnable causal bias
    temporal_bias_init: "causal"  # Start with early → late prior
    temporal_bias_strength: 0.1
    enforce_causality: true  # Hard-mask non-causal attention

training:
  batch_size: 32  # Reduced from 64 (larger tokens)
  num_epochs: 100
  use_snr_weighting: true
  use_vocab_size_weighting: true
```

## Evaluation Metrics

### Quantitative

1. **Per-truncation reconstruction MSE**: Decode tokens at each truncation level
2. **Temporal consistency**: Correlation between adjacent truncations
3. **Early prediction accuracy**: Predict tokens₂₅₆ from tokens₃₂
4. **Information saturation**: When does representation stop improving?

### Qualitative

1. **Attention pattern visualization**: Plot learned temporal bias matrix
2. **Autoregressive generation**: Generate T032 → T064 → T128 → T256 sequentially
3. **Ablation studies**: Without causal bias, without temporal embeddings

## Success Criteria

**Minimum Viable**:
- ✅ Training converges with 234 tokens
- ✅ Per-truncation MSE ≤ baseline final MSE
- ✅ Clear causal attention pattern (early → late)

**Target**:
- ✅ 5-10% improvement in final reconstruction MSE
- ✅ Autoregressive generation works (T032 → T256 sequentially)
- ✅ Early prediction accuracy (tokens₆₄ predicts tokens₂₅₆ within 20% MSE)

**Stretch Goals**:
- ✅ Early stopping: Identify truncation length where representation saturates
- ✅ Uncertainty quantification: Measure confidence at each truncation
- ✅ Forecasting: Given partial trajectory, predict final tokens

## Implementation Details

### Components

1. **`src/spinlock/cli/pretokenize_dataset.py`**:
   - Added `--temporal-resolution` flag
   - Validates pyramid encoder usage
   - Extracts truncation lengths from config
   - Tokenizes at multiple lengths with suffix naming

2. **`experiments/diffusion/models/temporal_resolution_denoising_network.py`**:
   - Extends `DenoisingNetwork`
   - Adds truncation embeddings
   - Learnable causal attention bias
   - Enforces causality via masking

3. **`src/spinlock/experimental/diffusion/config.py`**:
   - Added `TemporalResolutionConfig` class
   - Embedded in `ModelConfig`

4. **`experiments/diffusion/configs/temporal_resolution.yaml`**:
   - Complete training configuration
   - Temporal resolution settings

5. **`experiments/diffusion/scripts/train_temporal_resolution.py`**:
   - Training script with temporal resolution support
   - Loads truncation lengths from dataset
   - Instantiates TemporalResolutionDenoisingNetwork

### Code Changes

- **New files**: 3 (~1,000 LOC)
- **Modified files**: 2 (~150 LOC)
- **Zero VQTokenizer modifications**: Leverages existing variable-length support

## Computational Cost

### Training Time

**Token count increase**: 50 → 234 tokens (4.7×)

**Transformer cost**: O(N²) → ~22× slower naively

**Mitigations**:
1. Causal masking (blocks half of attention): 2× speedup
2. Mixed precision (FP16): 2× speedup
3. Multi-GPU (4× RTX 3090): 4× speedup

**Revised estimate**: 22× / 2 / 2 / 4 = **~2.75× slower than baseline**

### Dataset Storage

- Baseline: ~30 MB
- Temporal resolution: ~140 MB (4.7× larger, matches token increase)
- With gzip compression: acceptable overhead

## Comparison to Alternatives

### Why Not Pyramid Decomposition?

Original consideration was to create pyramid token levels (coarse → fine) like image diffusion hierarchies. However:

- ✅ **Temporal truncation is simpler**: Uses VQTokenizer as-is
- ✅ **More interpretable**: "How does understanding evolve?" is intuitive
- ✅ **Better aligned with time-series**: Causal structure is natural
- ✅ **Enables autoregressive generation**: Sequential token generation

### Why Not Separate D3PM per Truncation?

Could train independent diffusion models for each truncation. However:

- ✅ **Joint diffusion learns dependencies**: Early → late conditioning
- ✅ **Single forward pass**: No sequential bottleneck
- ✅ **Shared capacity**: Model learns cross-truncation features

## Future Work

1. **Curriculum Learning**: Start with strong autoregressive bias (T032 → T256), gradually make uniform
2. **Per-Truncation Loss Weighting**: Prioritize final reconstruction quality
3. **Adaptive Truncation**: Dynamically determine which truncation levels to use
4. **Cross-Trajectory Attention**: Attend across samples at same truncation level
5. **Dual Tokenizer**: Extend to CNO + MNO dual tokenizer architecture

## References

- Plan document: See implementation plan in session transcript
- VQTokenizer: `src/spinlock/tokens/tokenizer.py`
- Base D3PM: `experiments/diffusion/models/discrete_d3pm.py`
- Base DenoisingNetwork: `experiments/diffusion/models/denoising_network.py`

## Citation

If this approach proves successful, consider it a novel contribution:

```
Temporal Resolution Discrete Diffusion for Trajectory Representation Evolution
- Multi-truncation tokenization diffusion
- Causal temporal attention bias
- Meta-diffusion: generative process mirrors representational refinement
```
