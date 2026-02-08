# Discrete Diffusion for VQTokenizer v2 Token Interpolation

Discrete diffusion implementation for predicting/interpolating masked VQTokenizer v2 tokens from CNO dataset rollouts.

## Architecture

### Core Components

1. **DiscreteD3PM** (`models/discrete_d3pm.py`)
   - D3PM discrete diffusion process for hierarchical dict tokens
   - Per-category-level transition matrices (variable vocab sizes)
   - Forward/reverse process with RePaint-style inpainting
   - Flexible noise schedules (linear, cosine, sqrt)

2. **DenoisingNetwork** (`models/denoising_network.py`)
   - Transformer-based denoiser with flatten-process-unflatten pattern
   - Sinusoidal time embeddings
   - Hierarchical guidance (L0 coarse → all levels)
   - Per-category-level output heads for variable vocab sizes

3. **HierarchicalMaskGenerator** (`data/hierarchical_masking.py`)
   - RANDOM: Primary training strategy (50% mask probability)
   - COARSE_ONLY: Keep L0, predict L1+L2 (hierarchical inference test)
   - HIERARCHICAL: Keep L0+L1, predict L2 (fine detail test)
   - FAMILY_SELECTIVE: Mask entire families (cross-family test)

4. **DiffusionCompletionDataset** (`data/completion_dataset.py`)
   - Load features, tokenize with VQTokenizer v2
   - Generate masked examples for training/evaluation

## Training

Uses pregenerated rollout features (initial + temporal) from CNO dataset where VQTokenizer was trained.

### Baseline Experiment (50 steps, RANDOM masking)

```bash
python experiments/diffusion/scripts/train.py \
  --config experiments/diffusion/configs/baseline_diffusion.yaml
```

## Unit Tests

All 22 tests pass:

```bash
pytest experiments/diffusion/tests/ -v
```

## Key Features

- **Dict Format Support**: Handles `Dict[str, Tensor]` tokens with variable vocab sizes
- **RePaint Inpainting**: Keeps observed tokens fixed during sampling
- **Hierarchical Guidance**: L0 (coarse) embeddings guide L1/L2 predictions
- **Flexible Masking**: 4 strategies for different evaluation scenarios

## References

- **D3PM**: Austin et al., NeurIPS 2021
- **RePaint**: Lugmayr et al., CVPR 2022
