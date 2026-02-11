# Diffusion Trajectory Completion Experiments

Discrete diffusion models (D3PM) for token-based trajectory completion.

## Framework Code

All diffusion framework code has been moved to `spinlock.experimental.diffusion.*`:
- `spinlock.experimental.diffusion.models` - DiscreteD3PM, DenoisingNetwork
- `spinlock.experimental.diffusion.data` - Masking strategies, datasets
- `spinlock.experimental.diffusion.visualization` - TrajectoryInpainter, visualizers
- `spinlock.experimental.diffusion.training` - DiffusionTrainer

## Scripts

Standalone run scripts in `scripts/`:
- `train.py` - Train diffusion model
- `evaluate.py` - Evaluate trained model
- `generate_samples.py` - Generate completion samples
- `verify_baseline.py` - Verify setup before training

## CLI Command

Use the integrated CLI command for visualization:

```bash
poetry run spinlock visualize-diffusion-inpainting \
  --diffusion-checkpoint checkpoints/diffusion_model.pt \
  --tokenizer-checkpoint checkpoints/tokenizer.pt \
  --dataset datasets/50k_baseline.h5 \
  --output-dir visualizations/diffusion/ \
  --num-samples 5 \
  --mask-strategy temporal
```

## Results

Experiment outputs are stored in:
- `runs/` - Training checkpoints and logs
- `results/` - Evaluation metrics and visualizations
