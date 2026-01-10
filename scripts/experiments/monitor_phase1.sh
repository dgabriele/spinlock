#!/bin/bash
# Monitor Phase 1 experiment progress

echo "Phase 1 Experiment Progress"
echo "=============================="
echo ""

for exp in exp1{a,b,c,d,e,f}; do
    exp_name=$(echo $exp | tr '[:lower:]' '[:upper:]')
    checkpoint_dir="checkpoints/experiments/${exp}_*"

    if [ -d checkpoints/experiments/${exp}_* 2>/dev/null ]; then
        echo "Experiment $exp_name:"

        # Check for training log
        log_file=$(find checkpoints/experiments/${exp}_* -name "training_log.txt" 2>/dev/null | head -1)
        if [ -f "$log_file" ]; then
            # Get last epoch and validation loss
            last_epoch=$(grep "Epoch" "$log_file" | tail -1 | awk '{print $2}')
            val_loss=$(grep "Val Loss" "$log_file" | tail -1 | awk '{print $3}')
            echo "  Status: Running/Complete"
            echo "  Last Epoch: $last_epoch"
            echo "  Val Loss: $val_loss"
        else
            echo "  Status: Starting..."
        fi
    else
        echo "Experiment $exp_name: Not started"
    fi
    echo ""
done

echo "=============================="
echo ""
echo "To view live progress of running experiment:"
echo "  tail -f /tmp/claude/-home-daniel-projects-spinlock/tasks/*.output"
