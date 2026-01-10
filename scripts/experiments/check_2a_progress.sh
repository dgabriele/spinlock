#!/bin/bash
# Quick progress check for Experiment 2A

echo "================================"
echo "Experiment 2A Progress"
echo "================================"
echo ""

LOG_FILE="checkpoints/experiments/phase2/exp2a_baseline_1k/training_log.txt"

if [ -f "$LOG_FILE" ]; then
    # Count completed epochs
    EPOCHS=$(grep -v "^#" "$LOG_FILE" | wc -l)

    # Get best validation loss
    BEST_VAL=$(grep -v "^#" "$LOG_FILE" | awk -F',' '{print $3}' | sort -g | head -1)

    # Get latest validation loss
    LATEST_VAL=$(grep -v "^#" "$LOG_FILE" | tail -1 | awk -F',' '{print $3}')

    echo "Epochs completed: $EPOCHS / 30"
    echo "Best val loss: $BEST_VAL"
    echo "Latest val loss: $LATEST_VAL"
    echo ""

    # Compare to baseline
    BASELINE=0.514
    if (( $(echo "$BEST_VAL < $BASELINE" | bc -l) )); then
        IMPROVEMENT=$(echo "scale=1; ($BASELINE - $BEST_VAL) / $BASELINE * 100" | bc)
        echo "✓ Beating baseline by ${IMPROVEMENT}%"
    else
        echo "⚠ Not yet beating baseline ($BASELINE)"
    fi

    # Compare to Phase 1 best
    PHASE1_BEST=0.454
    if (( $(echo "$BEST_VAL < $PHASE1_BEST" | bc -l) )); then
        echo "✓ Better than Phase 1!"
    elif (( $(echo "$BEST_VAL < 0.5" | bc -l) )); then
        echo "✓ Close to Phase 1 performance"
    fi

else
    echo "Training log not yet created"
    echo "Still on first epoch..."
fi

echo ""
echo "Live output:"
tail -10 /tmp/claude/-home-daniel-projects-spinlock/tasks/bfd54f4.output
