# Experimental Components

The `spinlock.experimental` package contains cutting-edge research features.

## Modules

### `spinlock.experimental.common`
Shared experiment infrastructure:
- **Config system**: Pydantic schemas and YAML loading
- **Model wrappers** (LEGACY): `TrainedVQVAE` (wraps deprecated `CategoricalVQVAE`), `TrainedMNO` (convenience wrapper)
  - ⚠️ **Note**: These wrappers are legacy code and should not be used in new code
  - Use `VQTokenizer.from_checkpoint()` and `MetaOperator` directly instead
- **Training infrastructure**: `BaseExperimentTrainer`
- **Data utilities**: `TrajectoryDataLoader`

### `spinlock.experimental.diffusion`
Discrete diffusion models for trajectory completion:
- **Models**: `DiscreteD3PM`, `DenoisingNetwork`
- **Data**: Masking strategies, completion datasets
- **Visualization**: `TrajectoryInpainter`, `ComparisonVisualizer`
- **Training**: `DiffusionTrainer`, curriculum learning

### `spinlock.experimental.trajectory_completion`
Transformer-based trajectory completion from sparse observations.

### `spinlock.experimental.token_coverage`
Token usage analysis utilities.

### `spinlock.experimental.clustering_comparison`
Clustering algorithm comparison tools.

## Usage

```python
# Import models
from spinlock.experimental.diffusion.models import DiscreteD3PM, DenoisingNetwork

# Import common utilities
from spinlock.experimental.common import TrainedVQVAE, BaseExperimentTrainer

# Import visualization tools
from spinlock.experimental.diffusion.visualization import TrajectoryInpainter
```

## CLI Commands

### Diffusion Trajectory Completion

Visualize diffusion-based trajectory completion:

```bash
poetry run spinlock visualize-diffusion-inpainting \
  --diffusion-checkpoint checkpoints/diffusion_model.pt \
  --tokenizer-checkpoint checkpoints/tokenizer.pt \
  --dataset datasets/50k_baseline.h5 \
  --output-dir visualizations/diffusion/ \
  --num-samples 5 \
  --mask-strategy temporal \
  --device cuda
```

Options:
- `--mask-strategy {temporal,random,keyframe}` - How to mask observations
- `--mask-ratio R` - Fraction of trajectory to mask (default: 0.5)
- `--num-diffusion-steps T` - Diffusion steps (default: from checkpoint)
- `--format {frames,video,both}` - Output format

## Package Structure

```
src/spinlock/experimental/
├── __init__.py              # Public API
├── common/                  # Shared infrastructure
│   ├── config/             # Pydantic schemas, YAML loading
│   ├── data/               # Data loaders
│   ├── models/             # Model wrappers
│   └── training/           # Base trainer
├── diffusion/              # Diffusion models
│   ├── models/            # DiscreteD3PM, DenoisingNetwork
│   ├── data/              # Masking, datasets
│   ├── visualization/     # TrajectoryInpainter, visualizers
│   └── training/          # DiffusionTrainer
├── trajectory_completion/ # Transformer completion
├── token_coverage/        # Coverage analysis
└── clustering_comparison/ # Clustering tools
```

## Experiment Outputs

Experimental run outputs are stored in `experiments/`:

```
experiments/
├── diffusion/              # Diffusion experiment outputs
│   ├── scripts/           # Standalone run scripts
│   ├── runs/              # Training checkpoints
│   └── results/           # Evaluation results
└── trajectory_completion/ # Trajectory completion outputs
    ├── run_experiment.py  # Standalone script
    └── results/           # Experiment results
```

## Development

All production code lives in `src/spinlock/experimental/`.
Standalone experiment scripts live in `experiments/*/scripts/`.
Results and checkpoints go in `experiments/*/runs/` and `experiments/*/results/`.
