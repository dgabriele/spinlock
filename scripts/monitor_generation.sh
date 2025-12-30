#!/bin/bash
# Monitor 10K dataset generation progress

LOG_FILE="/tmp/spinlock_10k_generation.log"
OUTPUT_FILE="/tmp/claude/-home-daniel-projects-spinlock/tasks/b61bab5.output"

echo "============================================================"
echo "SPINLOCK 10K GENERATION MONITOR"
echo "============================================================"
echo ""
echo "Started: $(date)"
echo "Expected completion: ~19-28 hours from start"
echo ""
echo "============================================================"
echo "CURRENT PROGRESS"
echo "============================================================"
echo ""

# Show latest progress from log
if [ -f "$OUTPUT_FILE" ]; then
    # Extract latest progress line
    tail -50 "$OUTPUT_FILE" | grep -E "Generating:|Time per operator:|operators complete" | tail -5

    echo ""
    echo "============================================================"
    echo "RECENT ACTIVITY (last 20 lines)"
    echo "============================================================"
    tail -20 "$OUTPUT_FILE"

    echo ""
    echo "============================================================"
    echo "GPU STATUS"
    echo "============================================================"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader

else
    echo "Log file not found: $OUTPUT_FILE"
fi

echo ""
echo "============================================================"
echo "To watch live progress:"
echo "  tail -f $OUTPUT_FILE"
echo ""
echo "To check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo "============================================================"
