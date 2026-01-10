#!/bin/bash
# Run remaining Phase 1 experiments (1C-1F) after 1B completes

set -e
cd "$(dirname "$0")/../.."

echo "Waiting for Experiment 1B to complete..."
while ! [ -f "checkpoints/experiments/exp1b_gradaccum/training_log.txt" ] || \
      ! grep -q "^[0-9]" "checkpoints/experiments/exp1b_gradaccum/training_log.txt" 2>/dev/null || \
      pgrep -f "exp1b_gradaccum" > /dev/null 2>&1; do
    sleep 30
done
echo "✓ Experiment 1B completed"
echo ""

# Run 1C
echo "======================================================================="
echo "Experiment 1C: No Cache Clearing"
echo "======================================================================="
poetry run spinlock train-meta-operator --config configs/noa/experiments/phase1/exp1c_nocache.yaml
echo ""

# Run 1D
echo "======================================================================="
echo "Experiment 1D: Increased Capacity"
echo "======================================================================="
poetry run spinlock train-meta-operator --config configs/noa/experiments/phase1/exp1d_capacity.yaml
echo ""

# Run 1E
echo "======================================================================="
echo "Experiment 1E: Stronger Regularization"
echo "======================================================================="
poetry run spinlock train-meta-operator --config configs/noa/experiments/phase1/exp1e_regularization.yaml
echo ""

# Run 1F (CRITICAL)
echo "======================================================================="
echo "Experiment 1F: Combined Best (CRITICAL)"
echo "======================================================================="
poetry run spinlock train-meta-operator --config configs/noa/experiments/phase1/exp1f_combined.yaml
echo ""

echo "✓ All Phase 1 experiments complete!"
