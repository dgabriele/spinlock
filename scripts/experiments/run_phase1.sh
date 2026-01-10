#!/bin/bash
# Phase 1 Experiments: Systematic Hyperparameter Search
# Runs 6 experiments sequentially on 100 samples, 20 epochs each

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================="
echo "Phase 1: Quick Experiments (100 samples)"
echo "======================================================================="
echo ""

# Experiment 1A: LR Warmup (already running, skip)
echo "Experiment 1A: LR Warmup - ALREADY RUNNING"
echo ""

# Experiment 1B: Gradient Accumulation
echo "======================================================================="
echo "Experiment 1B: Gradient Accumulation"
echo "======================================================================="
poetry run spinlock train-meta-operator \
    --config configs/noa/experiments/phase1/exp1b_gradaccum.yaml
echo ""

# Experiment 1C: No Cache Clearing
echo "======================================================================="
echo "Experiment 1C: No Cache Clearing"
echo "======================================================================="
poetry run spinlock train-meta-operator \
    --config configs/noa/experiments/phase1/exp1c_nocache.yaml
echo ""

# Experiment 1D: Increased Capacity
echo "======================================================================="
echo "Experiment 1D: Increased Capacity"
echo "======================================================================="
poetry run spinlock train-meta-operator \
    --config configs/noa/experiments/phase1/exp1d_capacity.yaml
echo ""

# Experiment 1E: Stronger Regularization
echo "======================================================================="
echo "Experiment 1E: Stronger Regularization"
echo "======================================================================="
poetry run spinlock train-meta-operator \
    --config configs/noa/experiments/phase1/exp1e_regularization.yaml
echo ""

# Experiment 1F: Combined Best (CRITICAL)
echo "======================================================================="
echo "Experiment 1F: Combined Best (CRITICAL)"
echo "======================================================================="
poetry run spinlock train-meta-operator \
    --config configs/noa/experiments/phase1/exp1f_combined.yaml
echo ""

echo "======================================================================="
echo "Phase 1 Complete!"
echo "======================================================================="
echo ""
echo "Results saved to: checkpoints/experiments/"
echo ""
echo "Next steps:"
echo "1. Analyze results: python scripts/analysis/plot_experiments.py"
echo "2. Identify best configuration for Phase 2"
echo "3. Scale to 1K-5K samples"
