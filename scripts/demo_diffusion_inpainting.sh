#!/bin/bash
# Full demonstration of diffusion in-painting with token accuracy visualization

set -e

echo "=============================================="
echo "Diffusion Token In-Painting Demonstration"
echo "=============================================="

# Configuration
DIFFUSION_CKPT="experiments/diffusion/results/baseline_50steps_pretokenized/diffusion_baseline_pretokenized_best.pt"
TOKENIZER_CKPT="checkpoints/vqvae/vq_tokenizer_best.pt"
DATASET="datasets/50k_baseline.h5"
OUTPUT_DIR="visualizations/diffusion_inpainting_demo"

# Clean previous output
rm -rf "$OUTPUT_DIR"

# Run visualization with 3 samples, 50 diffusion steps
echo ""
echo "Running diffusion in-painting (3 samples, 50 steps)..."
poetry run spinlock visualize-diffusion-inpainting \
  --diffusion-checkpoint "$DIFFUSION_CKPT" \
  --tokenizer-checkpoint "$TOKENIZER_CKPT" \
  --dataset "$DATASET" \
  --num-samples 3 \
  --mask-strategy temporal \
  --mask-ratio 0.5 \
  --num-diffusion-steps 50 \
  --output-dir "$OUTPUT_DIR" \
  --device cuda

echo ""
echo "=============================================="
echo "Demo Complete!"
echo "=============================================="
echo "Output location: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"

echo ""
echo "Summary:"
cat "$OUTPUT_DIR/summary.json"
