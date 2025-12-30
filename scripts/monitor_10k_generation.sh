#!/bin/bash
# Memory-monitored 10K dataset generation
# Monitors RAM/GPU usage and kills process if memory usage becomes dangerous

set -e

# Configuration
CONFIG_FILE="configs/experiments/vqvae_baseline_10k_temporal/dataset.yaml"
OUTPUT_DIR="data/10k_temporal_test"
LOG_FILE="10k_generation_memtest.log"
MEMORY_LOG="10k_memory_profile.csv"

# Memory thresholds (kill if exceeded)
MAX_RAM_GB=240  # Kill if RAM usage exceeds 240GB (out of 256GB total)
MAX_RAM_PERCENT=94  # Kill if RAM percentage exceeds 94%
CHECK_INTERVAL=30  # Check memory every 30 seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "10K DATASET GENERATION - MEMORY MONITORED"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Memory log: $MEMORY_LOG"
echo "RAM threshold: ${MAX_RAM_GB}GB / ${MAX_RAM_PERCENT}%"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "=========================================="
echo ""

# Cleanup old outputs
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing old output directory..."
    rm -rf "$OUTPUT_DIR"
fi

# Start CSV log
echo "timestamp,elapsed_sec,ram_used_gb,ram_percent,gpu_mem_mb,gpu_util_percent,status" > "$MEMORY_LOG"

# Start dataset generation in background
echo "Starting dataset generation..."
PYTHONUNBUFFERED=1 poetry run python scripts/cli.py generate \
    --config "$CONFIG_FILE" \
    > "$LOG_FILE" 2>&1 &

GENERATION_PID=$!
echo "Generation PID: $GENERATION_PID"
echo ""

# Memory monitoring loop
START_TIME=$(date +%s)
MAX_RAM_SEEN=0
KILL_REASON=""

monitor_memory() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - START_TIME))

    # Get RAM usage
    local ram_info=$(free -g | grep Mem:)
    local ram_used=$(echo $ram_info | awk '{print $3}')
    local ram_total=$(echo $ram_info | awk '{print $2}')
    local ram_percent=$(awk "BEGIN {printf \"%.1f\", ($ram_used/$ram_total)*100}")

    # Get GPU memory if available
    local gpu_mem="N/A"
    local gpu_util="N/A"
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0, 0")
        gpu_mem=$(echo $gpu_info | awk -F',' '{print $1}' | tr -d ' ')
        gpu_util=$(echo $gpu_info | awk -F',' '{print $2}' | tr -d ' ')
    fi

    # Update max RAM seen
    if (( $(echo "$ram_used > $MAX_RAM_SEEN" | bc -l) )); then
        MAX_RAM_SEEN=$ram_used
    fi

    # Log to CSV
    echo "$current_time,$elapsed,${ram_used},${ram_percent},${gpu_mem},${gpu_util},running" >> "$MEMORY_LOG"

    # Print status
    printf "[%5ds] RAM: %3dGB (%5.1f%%) | GPU: %5d MB (%2d%%) | Peak RAM: %3dGB\n" \
        $elapsed $ram_used $ram_percent $gpu_mem $gpu_util $MAX_RAM_SEEN

    # Check thresholds
    if (( $(echo "$ram_used > $MAX_RAM_GB" | bc -l) )); then
        KILL_REASON="RAM usage exceeded ${MAX_RAM_GB}GB threshold (current: ${ram_used}GB)"
        return 1
    fi

    if (( $(echo "$ram_percent > $MAX_RAM_PERCENT" | bc -l) )); then
        KILL_REASON="RAM percentage exceeded ${MAX_RAM_PERCENT}% threshold (current: ${ram_percent}%)"
        return 1
    fi

    return 0
}

echo "Monitoring memory usage (Ctrl+C to stop)..."
echo ""

# Monitor until process completes or threshold exceeded
while kill -0 $GENERATION_PID 2>/dev/null; do
    if ! monitor_memory; then
        # Threshold exceeded - kill process
        echo ""
        echo -e "${RED}=========================================="
        echo "MEMORY THRESHOLD EXCEEDED!"
        echo "==========================================${NC}"
        echo "Reason: $KILL_REASON"
        echo "Killing generation process (PID: $GENERATION_PID)..."

        kill -TERM $GENERATION_PID 2>/dev/null || true
        sleep 2
        kill -KILL $GENERATION_PID 2>/dev/null || true

        echo "$current_time,$elapsed,${ram_used},${ram_percent},${gpu_mem},${gpu_util},killed_threshold" >> "$MEMORY_LOG"

        echo ""
        echo "Peak RAM usage: ${MAX_RAM_SEEN}GB"
        echo "Memory log saved to: $MEMORY_LOG"
        echo "Generation log saved to: $LOG_FILE"
        exit 1
    fi

    sleep $CHECK_INTERVAL
done

# Process completed successfully
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}=========================================="
echo "GENERATION COMPLETED SUCCESSFULLY!"
echo "==========================================${NC}"
echo "Total time: ${TOTAL_TIME}s ($((TOTAL_TIME/60)) minutes)"
echo "Peak RAM usage: ${MAX_RAM_SEEN}GB"
echo "Memory log saved to: $MEMORY_LOG"
echo "Generation log saved to: $LOG_FILE"
echo ""

# Check if dataset was created
if [ -f "${OUTPUT_DIR}/dataset.h5" ]; then
    DATASET_SIZE=$(du -h "${OUTPUT_DIR}/dataset.h5" | cut -f1)
    echo "Dataset created: ${OUTPUT_DIR}/dataset.h5 (${DATASET_SIZE})"
else
    echo -e "${YELLOW}Warning: Dataset file not found at expected location${NC}"
fi

echo "=========================================="
