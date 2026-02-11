# Trajectory Completion - Quick Start Guide

## When GPU is Available

### Step 1: Quick Validation (5 min)

```bash
# Verify VQ-VAE loads
poetry run python -c "
from experiments.common.models.trained_vqvae import TrainedVQVAE
vqvae = TrainedVQVAE('checkpoints/vqvae/50k_baseline/best_model.pt')
print(f'✅ VQ-VAE loaded')
print(f'   Categories: {vqvae.num_categories}')
print(f'   Codebook sizes: {vqvae.codebook_sizes}')
print(f'   Feature families: {vqvae.get_feature_families()}')
"
```

### Step 2: Run Baseline Experiment (2-3 hours)

```bash
# Run baseline experiment (30% start + 20% end masking)
poetry run python -m experiments.trajectory_completion.run_experiment \
    --config experiments/trajectory_completion/baseline_50k/experiments/baseline.yaml
```

**Expected output**:
- Dataset: ~20K samples loaded and tokenized
- Model: ~2.8M parameters
- Training: 20 epochs, saves every 5 epochs
- Results: `experiments/trajectory_completion/results/baseline/`

### Step 3: Analyze Results (5 min)

```bash
# Generate plots and summary
poetry run python -m experiments.trajectory_completion.evaluation.analysis \
    --results_dir experiments/trajectory_completion/results/baseline
```

**Generates**:
- Console: Summary statistics
- File: `training_curves.png` (loss, accuracy, reconstruction error)

### Step 4: Run Ablations (Optional, 2-3 hours each)

```bash
# Test extreme masking (10% + 10%)
poetry run python -m experiments.trajectory_completion.run_experiment \
    --config experiments/trajectory_completion/baseline_50k/experiments/extreme.yaml

# Test coarse-only (hierarchical guidance test)
poetry run python -m experiments.trajectory_completion.run_experiment \
    --config experiments/trajectory_completion/baseline_50k/experiments/coarse_only.yaml

# Test without hierarchical guidance (ablation)
poetry run python -m experiments.trajectory_completion.run_experiment \
    --config experiments/trajectory_completion/baseline_50k/experiments/ablation/no_temporal_hierarchy.yaml
```

---

## Success Criteria

### Week 2 Goals
- ✅ Model trains without OOM errors
- ✅ Token accuracy > 60% on validation set
- ✅ Feature reconstruction MSE < 0.1
- ✅ Checkpoints save correctly

### Week 3 Goals
- ✅ Ablation studies complete (5 configs)
- ✅ Comparison table generated
- ✅ Results documented

---

## Troubleshooting

### OOM Errors
- Reduce batch size: Edit config `data.batch_size: 16` → `8`
- Reduce model size: Edit config `model.hidden_dim: 256` → `128`

### Slow Training
- Increase batch size if GPU memory allows
- Reduce `data.num_workers` if CPU bottleneck

### Poor Accuracy
- Check VQ-VAE reconstruction quality first
- Try longer training: `training.epochs: 20` → `50`
- Increase hierarchical guidance: `model.hierarchical_guidance_weight: 0.1` → `0.2`

---

## File Locations

**Configs**: `experiments/trajectory_completion/baseline_50k/experiments/`
- `baseline.yaml` - 30% start + 20% end
- `extreme.yaml` - 10% start + 10% end
- `coarse_only.yaml` - L0 tokens only
- `ablation/no_temporal_hierarchy.yaml`
- `ablation/random_masking.yaml`

**Results**: `experiments/trajectory_completion/results/{experiment_name}/`
- `training_history.json` - Full metrics per epoch
- `checkpoints/checkpoint_epoch_*.pt` - Model checkpoints
- `training_curves.png` - Visualization (after analysis)

**Checkpoints Required**:
- `checkpoints/vqvae/50k_baseline/best_model.pt` ✅
- `datasets/50k_baseline.h5` ✅

---

## Quick Commands Reference

```bash
# List all experiments
ls experiments/trajectory_completion/baseline_50k/experiments/

# Run specific experiment
poetry run python -m experiments.trajectory_completion.run_experiment \
    --config experiments/trajectory_completion/baseline_50k/experiments/[CONFIG].yaml

# Analyze results
poetry run python -m experiments.trajectory_completion.evaluation.analysis \
    --results_dir experiments/trajectory_completion/results/[EXPERIMENT_NAME]

# Check GPU usage
nvidia-smi

# Monitor training (in another terminal)
watch -n 1 nvidia-smi
```

---

## Expected Timeline

| Task | Duration | Output |
|------|----------|--------|
| Baseline training | 2-3 hours | baseline/ results |
| Analysis | 5 minutes | training_curves.png |
| Extreme masking | 2-3 hours | extreme/ results |
| Coarse-only | 2-3 hours | coarse_only/ results |
| No hierarchy ablation | 2-3 hours | ablation/no_hierarchy/ results |
| Random masking ablation | 2-3 hours | ablation/random_masking/ results |

**Total**: ~10-15 hours GPU time for complete study

---

## Contact

**Issues**: Report in project issues or check:
- `experiments/trajectory_completion/README.md` - Full documentation
- `experiments/IMPLEMENTATION_SUMMARY.md` - Architecture details
