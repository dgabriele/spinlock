# Diffusion Model Sampling Pipeline

Complete pipeline for generating PDE samples from trained diffusion models.

## Overview

This pipeline consists of 4 sequential stages:

1. **Token Sampling** (`DiffusionSampler`): Generate discrete token sequences using DiscreteD3PM
2. **Token Decoding** (`VQTokenizer.decode()`): Convert tokens to continuous theta parameters and ICs
3. **Trajectory Generation** (`TrajectoryGenerator`): Generate PDE rollouts using CNOReplayer
4. **Output & Visualization** (`SampleWriter`, `SampleVisualizer`): Save results and create plots

## Quick Start

### Generate 100 Unconditional Samples (Tokens Only)

```bash
poetry run python experiments/diffusion/scripts/generate_samples.py \
  --diffusion-checkpoint experiments/diffusion/experiments/diffusion/results/v2_tokenizer_baseline/diffusion_v2_pretokenized_best.pt \
  --tokenizer-checkpoint checkpoints/v2/vqvae/vq_tokenizer_best.pt \
  --num-samples 100 \
  --mode unconditional \
  --output-dir experiments/diffusion/samples/quick_100
```

**Output**: Tokens + decoded theta + decoded ICs (~2MB, ~40s runtime)

### Generate Samples with Ground Truth Trajectories

```bash
poetry run python experiments/diffusion/scripts/generate_samples.py \
  --diffusion-checkpoint experiments/diffusion/experiments/diffusion/results/v2_tokenizer_baseline/diffusion_v2_pretokenized_best.pt \
  --tokenizer-checkpoint checkpoints/v2/vqvae/vq_tokenizer_best.pt \
  --cno-config configs/experiments/cno_50k_v3_1.yaml \
  --num-samples 50 \
  --mode unconditional \
  --generate-trajectories \
  --num-timesteps 256 \
  --save-trajectories \
  --visualize \
  --output-dir experiments/diffusion/samples/full_50
```

**Output**: Tokens + theta + ICs + trajectories + plots (~5GB, ~4min runtime)

## Usage Examples

### 1. Quick Token-Only Generation

Generate discrete tokens without trajectories (fast, small files):

```bash
poetry run python experiments/diffusion/scripts/generate_samples.py \
  --diffusion-checkpoint <diffusion_ckpt> \
  --tokenizer-checkpoint <tokenizer_ckpt> \
  --num-samples 100 \
  --output-dir samples/tokens_only
```

### 2. Full Pipeline with Visualization

Generate complete samples with trajectories and plots:

```bash
poetry run python experiments/diffusion/scripts/generate_samples.py \
  --diffusion-checkpoint <diffusion_ckpt> \
  --tokenizer-checkpoint <tokenizer_ckpt> \
  --cno-config <cno_config> \
  --num-samples 50 \
  --generate-trajectories \
  --visualize \
  --output-dir samples/full_pipeline
```

### 3. Theta-Conditioned Sampling (Future)

Fix theta tokens and vary trajectories:

```bash
poetry run python experiments/diffusion/scripts/generate_samples.py \
  --diffusion-checkpoint <diffusion_ckpt> \
  --tokenizer-checkpoint <tokenizer_ckpt> \
  --cno-config <cno_config> \
  --num-samples 20 \
  --mode theta_cond \
  --condition-on-sample 42 \
  --tokenized-dataset <tokenized_dataset> \
  --generate-trajectories \
  --output-dir samples/theta_cond
```

### 4. Inpainting (Future)

Complete partial token sequences:

```bash
poetry run python experiments/diffusion/scripts/generate_samples.py \
  --diffusion-checkpoint <diffusion_ckpt> \
  --tokenizer-checkpoint <tokenizer_ckpt> \
  --cno-config <cno_config> \
  --num-samples 10 \
  --mode inpaint \
  --condition-on-sample 10 \
  --generate-trajectories \
  --visualize \
  --output-dir samples/inpainting
```

## Command-Line Arguments

### Model Checkpoints

- `--diffusion-checkpoint`: Trained diffusion model checkpoint (required)
- `--tokenizer-checkpoint`: Trained VQTokenizer checkpoint (required)
- `--cno-config`: CNO config YAML for ground truth rollouts (required if --generate-trajectories)
- `--mno-checkpoint`: MNO checkpoint for learned rollouts (optional, future)

### Sampling Parameters

- `--num-samples`: Number of samples to generate (default: 100)
- `--batch-size`: Batch size for diffusion sampling (default: 32)
- `--mode`: Sampling mode: unconditional, theta_cond, inpaint (default: unconditional)

### Trajectory Generation

- `--generate-trajectories`: Enable trajectory generation
- `--num-timesteps`: Rollout length (default: 256)
- `--num-realizations`: Stochastic realizations per sample (default: 1)
- `--use-mno`: Use MNO instead of CNOReplayer (requires --mno-checkpoint)
- `--traj-batch-size`: Batch size for trajectory generation (default: 8)

### Output

- `--output-dir`: Output directory (required)
- `--save-trajectories`: Save full trajectories to HDF5 (large files)
- `--visualize`: Create visualization plots

### Device

- `--device`: Computation device (default: cuda)
- `--seed`: Random seed (default: 42)

## Output Format

### HDF5 Structure (`samples.h5`)

```
/tokens/
  temporal_group_*_L*: [N] token indices
  initial_group_*_L*: [N] token indices
  theta_group_*_L*: [N] token indices

/decoded/
  theta: [N, 14] operator parameters
  u0: [N, C, H, W] initial conditions

/trajectories/ (optional)
  rollouts: [N, M, T+1, C, H, W] PDE trajectories

/metadata (attributes)
  num_samples, generation_mode, etc.
```

### JSON Metadata (`metadata.json`)

```json
{
  "timestamp": "2026-02-10T01:30:00",
  "num_samples": 100,
  "mode": "unconditional",
  "diffusion_checkpoint": "...",
  "tokenizer_checkpoint": "...",
  "device": "cuda",
  "seed": 42
}
```

### Visualizations (if --visualize)

- `param_distributions.png`: Histograms of sampled parameters
- `trajectory_example.png`: Single trajectory evolution
- `diversity_comparison.png`: Multiple trajectories side-by-side

## Performance

### Memory Requirements

- Token sampling: ~10MB per 100 samples
- Decoding: ~200MB per 100 samples
- Trajectories: ~8GB per 100 samples [100, 1, 257, 3, 64, 64]

### Speed Estimates

- Diffusion sampling: 0.3s/sample (30s for 100)
- Token decoding: 0.05s/sample (5s for 100)
- Trajectory generation: 2s/sample (200s for 100 @ 256 steps)
- **Total pipeline**: ~4min for 100 full samples

### Optimization Tips

1. **Skip trajectories for fast exploration**: Omit `--generate-trajectories` flag
2. **Reduce trajectory batch size**: Use `--traj-batch-size 4` if running out of memory
3. **Use smaller rollouts**: `--num-timesteps 128` for faster generation

## Testing

Run unit tests:

```bash
poetry run python experiments/diffusion/scripts/test_sampling_pipeline.py
```

Tests verify:
- Token sampling with mock models
- Output writer HDF5 format
- Visualizer plot generation

## Architecture

### Module Structure

```
sampling/
├── __init__.py
├── sampler.py              # DiffusionSampler class
├── trajectory_generator.py  # TrajectoryGenerator class
├── output_writer.py         # SampleWriter class
└── visualizer.py            # SampleVisualizer class
```

### Key Classes

**`DiffusionSampler`**: Generate token sequences
- `sample_unconditional()`: Full unconditional sampling
- `sample_theta_conditioned()`: Fix theta, vary temporal/initial
- `sample_inpainting()`: Complete partial sequences

**`TrajectoryGenerator`**: Generate PDE trajectories
- Supports CNOReplayer (ground truth) and MNO (future)
- Batched processing for memory efficiency

**`SampleWriter`**: Save outputs to disk
- HDF5 for large arrays
- JSON for metadata

**`SampleVisualizer`**: Create plots
- Trajectory evolution plots
- Parameter distributions
- Diversity comparisons

## Future Enhancements

1. **Theta Conditioning**: Implement loading from tokenized datasets
2. **Inpainting**: Add support for partial token sequences
3. **MNO Support**: Add learned operator trajectory generation
4. **Video Export**: Generate MP4 videos of trajectory evolution
5. **Diversity Metrics**: PCA/t-SNE analysis of trajectory space
6. **Batch Parallelization**: Multi-GPU support for large-scale generation

## Troubleshooting

### "No trained theta_inverse found"

The tokenizer doesn't have a trained inverse model for theta decoding. This is expected - the decode method uses an approximate fallback. For better quality:

1. Train an inverse MLP separately
2. Add to tokenizer checkpoint
3. Reload tokenizer

### Out of Memory

- Reduce `--batch-size` (for token sampling)
- Reduce `--traj-batch-size` (for trajectory generation)
- Reduce `--num-timesteps` (shorter rollouts)
- Don't use `--save-trajectories` (skip storing full arrays)

### Slow Generation

- Skip `--generate-trajectories` for token-only experiments
- Reduce `--num-timesteps` to 128 or 64
- Use smaller `--num-samples` for quick tests

## Contact

For questions or issues, see the main Spinlock project documentation.
