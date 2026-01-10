#!/bin/bash
# Wait for 1E to complete, then run 1F (low memory version)

cd "$(dirname "$0")/../.."

echo "Waiting for Experiment 1E to complete..."
while pgrep -f "exp1e_regularization" > /dev/null 2>&1; do
    sleep 30
done
echo "✓ Experiment 1E completed"
echo ""

echo "======================================================================="
echo "Experiment 1F: Combined Best (Low Memory)"
echo "======================================================================="
poetry run spinlock train-meta-operator --config configs/noa/experiments/phase1/exp1f_combined_lowmem.yaml
echo ""

echo "✓ All Phase 1 experiments complete!"
echo ""
echo "Run analysis: python scripts/analysis/plot_phase1_results.py"
