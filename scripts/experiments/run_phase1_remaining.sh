#!/bin/bash
# Run remaining Phase 1 experiments (1B-1F) after 1A completes
# This script waits for Experiment 1A to finish, then runs experiments 1B-1F sequentially

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================="
echo "Waiting for Experiment 1A to complete..."
echo "======================================================================="

# Wait for 1A's best checkpoint to exist and stop being modified
while true; do
    if [ -f "checkpoints/experiments/exp1a_warmup/training_log.txt" ]; then
        # Check if training is complete (look for "Training Complete" in log)
        if grep -q "Training Complete" checkpoints/experiments/exp1a_warmup/training_log.txt 2>/dev/null; then
            echo "✓ Experiment 1A completed!"
            break
        fi
    fi
    echo "  Waiting... (checking every 30s)"
    sleep 30
done

echo ""
echo "======================================================================="
echo "Starting remaining Phase 1 experiments (1B-1F)"
echo "======================================================================="
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
echo "All experiment results saved to: checkpoints/experiments/"
echo ""
echo "Next steps:"
echo "1. Analyze results: python scripts/analysis/plot_experiments.py"
echo "2. Identify best configuration (likely exp1f_combined)"
echo "3. Proceed to Phase 2 (scale to 1K-5K samples)"
