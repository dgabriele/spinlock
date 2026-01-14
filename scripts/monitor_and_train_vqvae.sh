#!/bin/bash
# Monitor CNO generation and auto-start VQ-VAE training when complete

CNO_OUTPUT="/tmp/claude/-home-daniel-projects-spinlock/tasks/b0d0458.output"
MNO_DATASET="datasets/noa_features/mno_v3_10k.h5"
VQVAE_CONFIG="configs/vqvae/noa_distribution_10k_with_reference_reg.yaml"

echo "========================================================================"
echo "MONITORING CNO GENERATION → AUTO-START VQ-VAE TRAINING"
echo "========================================================================"
echo "CNO output: $CNO_OUTPUT"
echo "MNO dataset: $MNO_DATASET"
echo "VQ-VAE config: $VQVAE_CONFIG"
echo ""

# Wait for CNO generation to complete
echo "Waiting for CNO generation to complete..."
while true; do
    # Check if process is still running
    if ! pgrep -f "generate_cno_reference_features.py" > /dev/null; then
        echo ""
        echo "✓ CNO generation process completed"
        break
    fi

    # Show latest progress every 60 seconds
    sleep 60
    LATEST=$(tail -1 "$CNO_OUTPUT" 2>/dev/null | grep -o "[0-9]*%" | tail -1)
    if [ -n "$LATEST" ]; then
        echo "[$(date +%H:%M:%S)] CNO generation: $LATEST"
    fi
done

# Check if generation was successful
if tail -20 "$CNO_OUTPUT" | grep -q "GENERATION COMPLETE"; then
    echo "✓ CNO generation successful!"

    # Verify reference features exist
    echo ""
    echo "Verifying reference features..."
    poetry run python -c "
import h5py
with h5py.File('$MNO_DATASET', 'r') as f:
    if 'features/reference_features' in f:
        shape = f['features/reference_features'].shape
        print(f'✓ Reference features found: {shape}')
        exit(0)
    else:
        print('✗ Reference features NOT found!')
        exit(1)
"

    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================================================"
        echo "STARTING VQ-VAE TRAINING WITH REFERENCE REGULARIZATION"
        echo "========================================================================"
        echo ""

        # Start VQ-VAE training
        poetry run spinlock train-vqvae --config "$VQVAE_CONFIG"

        VQVAE_EXIT=$?
        echo ""
        echo "========================================================================"
        if [ $VQVAE_EXIT -eq 0 ]; then
            echo "✓ VQ-VAE TRAINING COMPLETED SUCCESSFULLY"
        else
            echo "✗ VQ-VAE TRAINING FAILED (exit code: $VQVAE_EXIT)"
        fi
        echo "========================================================================"
        exit $VQVAE_EXIT
    else
        echo "✗ Reference features verification failed"
        exit 1
    fi
else
    echo "✗ CNO generation failed or incomplete"
    tail -50 "$CNO_OUTPUT"
    exit 1
fi
