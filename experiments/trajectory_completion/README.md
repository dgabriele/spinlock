# Trajectory Completion Experiment

## Overview

This experiment demonstrates the VQ-VAE + MNO system's capability to predict missing portions of physics trajectories using only discrete token representations and coarse-to-fine temporal structure.

**Core Concept**: Given partial token sequences (e.g., first 30% + last 20% of trajectory), use coarse temporal tokens to guide inference of missing fine-grained tokens, then decode to continuous space and compare against ground truth.

## Quick Start

### Prerequisites

1. Trained VQ-VAE checkpoint: `checkpoints/vqvae/50k_baseline/best_model.pt`
2. Dataset with features: `datasets/50k_baseline.h5`
3. Python environment with dependencies installed

### Running the Baseline Experiment

```bash
# From project root
python -m experiments.trajectory_completion.run_experiment \
    --config experiments/trajectory_completion/baseline_50k/experiments/baseline.yaml
```

This will:
- Load the VQ-VAE model and tokenize all features
- Create masked trajectory examples (30% start + 20% end observed)
- Train the completion model for 20 epochs
- Save checkpoints every 5 epochs to `experiments/trajectory_completion/results/baseline/checkpoints/`
- Save training history to `experiments/trajectory_completion/results/baseline/training_history.json`

### Analyzing Results

```bash
python -m experiments.trajectory_completion.evaluation.analysis \
    --results_dir experiments/trajectory_completion/results/baseline
```

This generates:
- Summary statistics printed to console
- Training curves: `training_curves.png`

## Experiment Configurations

### Baseline (`baseline.yaml`)
- **Masking**: 30% start + 20% end observed, middle 50% to predict
- **Goal**: Demonstrate basic completion capability
- **Expected**: Token accuracy > 60%, reconstruction MSE < 0.1

### Extreme (`extreme.yaml`)
- **Masking**: 10% start + 10% end observed, middle 80% to predict
- **Goal**: Test limits of coarse-to-fine guidance with minimal observations
- **Note**: Uses increased hierarchical guidance weight (0.2)

### Coarse Only (`coarse_only.yaml`)
- **Masking**: Only L0 (coarse) tokens observed, L1 and L2 to predict
- **Goal**: Test hierarchical guidance - can coarse structure predict fine details?
- **Note**: Strongest test of hierarchical architecture

### Ablations

#### No Temporal Hierarchy (`ablation/no_temporal_hierarchy.yaml`)
- **Ablation**: `hierarchical_guidance_weight = 0.0`
- **Goal**: Measure contribution of coarse→fine guidance
- **Compare**: Baseline vs this to quantify hierarchy value

#### Random Masking (`ablation/random_masking.yaml`)
- **Ablation**: `random_windows` strategy instead of `start_end`
- **Goal**: Compare structured vs random temporal masking
- **Insight**: Does temporal contiguity help?

## Architecture

### Model: `TrajectoryCompletionModel`

**Pipeline**:
1. **Token Embedding**: Separate embeddings per hierarchical level
2. **Hierarchical Guidance**: Coarse (L0) embeddings modulate fine predictions via residual connections
3. **Transformer Encoder**: Bidirectional attention over observed tokens
4. **Output Projection**: Per-level projections to token logits

**Key Parameters**:
- `hidden_dim`: Embedding/hidden dimension (default: 256)
- `num_layers`: Transformer layers (default: 4)
- `num_heads`: Attention heads (default: 8)
- `hierarchical_guidance_weight`: Weight for coarse→fine influence (default: 0.1)

### Training

**Loss**: Cross-entropy on masked token positions only

**Metrics**:
- Token accuracy (on masked positions)
- Per-level token accuracy (L0, L1, L2)
- Feature reconstruction MSE (decode completed tokens)

**Optimizer**: Adam with weight decay

## Directory Structure

```
experiments/trajectory_completion/
├── README.md                          # This file
├── run_experiment.py                  # Main entry point
├── baseline_50k/
│   └── experiments/                   # Experiment configs
│       ├── baseline.yaml
│       ├── extreme.yaml
│       ├── coarse_only.yaml
│       └── ablation/
│           ├── no_temporal_hierarchy.yaml
│           └── random_masking.yaml
├── data/
│   ├── masking.py                     # TemporalMaskGenerator
│   └── completion_dataset.py          # TrajectoryCompletionDataset
├── models/
│   └── completion_model.py            # TrajectoryCompletionModel
├── training/
│   └── trainer.py                     # CompletionTrainer
├── evaluation/
│   ├── metrics.py                     # Completion metrics
│   └── analysis.py                    # Result analysis
└── results/                           # Generated during experiments
    ├── baseline/
    ├── extreme/
    └── ...
```

## Success Criteria

### Week 1-2 Goals (Core Implementation)
- ✅ Infrastructure implemented
- ✅ VQ-VAE wrapper loads and encodes/decodes
- ✅ Dataset loads and tokenizes features
- ✅ Model trains without errors
- 🎯 Baseline experiment completes 20 epochs
- 🎯 Token accuracy > 60% on validation set
- 🎯 Feature reconstruction MSE < 0.1 (normalized)

### Week 3 Goals (Analysis)
- 🎯 Analysis script generates training curves
- 🎯 Results documented
- 🎯 At least 2 ablation configs tested
- 🎯 Comparison table: baseline vs extreme vs coarse_only

## Design Principles

1. **DRY**: Shared infrastructure in `experiments/common/`
2. **Clean Abstraction**: Separate concerns (config, data, model, training)
3. **OOP**: Inheritance and composition patterns
4. **Testability**: Clear input/output contracts
5. **Extensibility**: New experiments reuse components

## Future Extensions

**Week 4+: Token Space Coverage Analysis**
- Analyze which tokens are used, which are rare/dead
- Visualize token co-occurrence matrices
- Hierarchical clustering of token usage

**Week 5+: Zero-Shot Transfer**
- Test completion on unseen parameter regimes
- Evaluate out-of-distribution generalization

**Week 6+: Inverse Problems**
- Given trajectory tokens → predict parameters
- Architecture: Encoder over token sequence → parameter vector

## References

**Internal Code**:
- VQ-VAE: `src/spinlock/encoding/models/categorical_vqvae.py`
- Feature extraction: `src/spinlock/noa/generation_pipeline.py`
- Config system: `src/spinlock/config/`

**Checkpoints**:
- VQ-VAE: `checkpoints/vqvae/50k_baseline/best_model.pt`
- Dataset: `datasets/50k_baseline.h5`
