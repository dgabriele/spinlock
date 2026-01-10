#!/usr/bin/env python3
"""Quick summary of Phase 1 results."""

import sys
from pathlib import Path

# Experiment directories that actually exist
EXPERIMENTS = [
    ("exp1a_warmup", "1A: LR Warmup"),
    ("exp1b_gradaccum", "1B: Gradient Accumulation"),
    ("exp1c_nocache", "1C: No Cache Clearing"),
    ("exp1e_regularization", "1E: Stronger Regularization"),
    ("exp1f_combined_lowmem", "1F: Combined Best"),
]

def main():
    project_root = Path(__file__).parent.parent.parent
    checkpoint_dir = project_root / "checkpoints" / "experiments"

    print("=" * 70)
    print("Phase 1 Experiment Results")
    print("=" * 70)
    print()
    print(f"{'Exp':<5} {'Name':<30} {'Best Val Loss':<15} {'Epochs':<10}")
    print("-" * 70)

    results = []
    for exp_id, exp_name in EXPERIMENTS:
        log_path = checkpoint_dir / exp_id / "training_log.txt"

        if not log_path.exists():
            print(f"{exp_id:<5} {exp_name:<30} {'NOT RUN':<15} {'-':<10}")
            continue

        # Parse log file (CSV format: Epoch,TrainLoss,ValLoss,LR,Time)
        with open(log_path) as f:
            lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]

        if not lines:
            print(f"{exp_id:<5} {exp_name:<30} {'EMPTY LOG':<15} {'-':<10}")
            continue

        # Extract best validation loss
        best_val_loss = float('inf')
        num_epochs = len(lines)

        for line in lines:
            parts = line.split(',')
            if len(parts) >= 3:
                val_loss = float(parts[2])
                best_val_loss = min(best_val_loss, val_loss)

        results.append((exp_id, exp_name, best_val_loss, num_epochs))
        print(f"{exp_id:<5} {exp_name:<30} {best_val_loss:<15.6f} {num_epochs:<10}")

    print()
    print("=" * 70)
    print("Key Findings")
    print("=" * 70)
    print()

    # Find best experiment
    if results:
        results.sort(key=lambda x: x[2])  # Sort by val loss
        best = results[0]
        print(f"✓ Best experiment: {best[1]}")
        print(f"  Best val loss: {best[2]:.6f}")
        print(f"  Epochs: {best[3]}")
        print()

        # Compare to baseline
        baseline = 0.514
        improvement = (baseline - best[2]) / baseline * 100
        if best[2] < baseline:
            print(f"✓ Achieved {improvement:.1f}% improvement over baseline ({baseline})")
        else:
            print(f"⚠ Did not beat baseline ({baseline})")
        print()

        # Show all results sorted
        print("Ranking (best to worst):")
        for i, (exp_id, exp_name, val_loss, epochs) in enumerate(results, 1):
            print(f"  {i}. {exp_name}: {val_loss:.6f}")

if __name__ == "__main__":
    main()
