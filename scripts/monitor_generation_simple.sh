#!/bin/bash
# Simple monitoring script for dataset generation progress
# Usage: ./scripts/monitor_generation_simple.sh

LOG_FILE="/tmp/spinlock_10k_temporal_generation.log"

echo "=========================================="
echo "SPINLOCK 10K GENERATION MONITOR"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to exit monitoring"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "SPINLOCK 10K TEMPORAL DATASET GENERATION"
    echo "=========================================="
    date
    echo ""

    # Show last 40 lines of log
    if [ -f "$LOG_FILE" ]; then
        tail -40 "$LOG_FILE"
    else
        echo "Log file not found: $LOG_FILE"
    fi

    echo ""
    echo "=========================================="
    echo "System Status:"
    echo "=========================================="

    # GPU usage
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
        awk -F', ' '{printf "GPU: %s util | %s / %s MB | %s°C\n", $1, $2, $3, $4}'
    fi

    # RAM usage
    free -h | grep Mem: | awk '{printf "RAM: %s / %s\n", $3, $2}'

    # Dataset size if it exists
    if [ -f "datasets/vqvae_baseline_10k_temporal.h5" ]; then
        DATASET_SIZE=$(du -h datasets/vqvae_baseline_10k_temporal.h5 | cut -f1)
        echo "Dataset size: $DATASET_SIZE"
    fi

    echo ""
    echo "Refreshing in 30s... (Ctrl+C to exit)"
    sleep 30
done
