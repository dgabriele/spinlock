#!/bin/bash
# Test script for diffusion in-painting visualization CLI

set -e

echo "========================================="
echo "Testing Diffusion In-Painting Visualizer"
echo "========================================="

# Configuration
DIFFUSION_CKPT="experiments/diffusion/results/baseline_50steps_pretokenized/diffusion_baseline_pretokenized_best.pt"
TOKENIZER_CKPT="checkpoints/vqvae/vq_tokenizer_best.pt"
DATASET="datasets/50k_baseline.h5"
OUTPUT_DIR="visualizations/diffusion_inpainting_test"

# Clean previous output
rm -rf "$OUTPUT_DIR"

# Run CLI with minimal test (1 sample, frames only for speed)
echo ""
echo "Running in-painting visualization (1 sample)..."
poetry run spinlock visualize-diffusion-inpainting \
  --diffusion-checkpoint "$DIFFUSION_CKPT" \
  --tokenizer-checkpoint "$TOKENIZER_CKPT" \
  --dataset "$DATASET" \
  --num-samples 1 \
  --mask-strategy temporal \
  --mask-ratio 0.5 \
  --num-diffusion-steps 10 \
  --output-dir "$OUTPUT_DIR" \
  --format frames \
  --device cuda \
  --verbose

echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo "Output location: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/summary.json" ]; then
    echo ""
    echo "Summary:"
    cat "$OUTPUT_DIR/summary.json"
fi
