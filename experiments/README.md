# Spinlock Experiments

This directory contains outputs from experimental runs using the Spinlock framework.

## Structure

- `diffusion/` - Discrete diffusion trajectory completion experiments
- `trajectory_completion/` - Transformer-based trajectory completion
- `token_coverage/` - Token coverage analysis results
- `clustering_comparison/` - Clustering comparison results

## Running Experiments

All experimental framework code is now in `spinlock.experimental.*`.
To run experiments, use the standalone scripts in each experiment's `scripts/` directory.

Example:
```bash
cd experiments/diffusion/scripts
python train.py --config configs/baseline.yaml
```

Alternatively, use the CLI commands:
```bash
poetry run spinlock visualize-diffusion-inpainting \
  --diffusion-checkpoint checkpoints/diffusion_model.pt \
  --tokenizer-checkpoint checkpoints/tokenizer.pt \
  --dataset datasets/50k_baseline.h5 \
  --output-dir visualizations/diffusion/
```
