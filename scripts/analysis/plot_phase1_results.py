#!/usr/bin/env python3
"""
Analyze and visualize Phase 1 experiment results.

Plots validation loss curves for all 6 experiments to identify
the best hyperparameter configuration for Phase 2.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Experiment metadata
EXPERIMENTS = {
    "exp1a_warmup": {
        "name": "1A: LR Warmup",
        "color": "#1f77b4",
        "key_change": "warmup_steps=500",
    },
    "exp1b_gradaccum": {
        "name": "1B: Gradient Accumulation",
        "color": "#ff7f0e",
        "key_change": "gradient_accumulation_steps=8",
    },
    "exp1c_nocache": {
        "name": "1C: No Cache Clearing",
        "color": "#2ca02c",
        "key_change": "Conditional GPU cache",
    },
    "exp1d_capacity": {
        "name": "1D: Increased Capacity",
        "color": "#d62728",
        "key_change": "modes=32, afno_blocks=6",
    },
    "exp1e_regularization": {
        "name": "1E: Stronger Regularization",
        "color": "#9467bd",
        "key_change": "weight_decay=1e-4",
    },
    "exp1f_combined": {
        "name": "1F: Combined Best ⭐",
        "color": "#e377c2",
        "key_change": "All improvements",
        "linewidth": 3,  # Emphasize this experiment
    },
}

def parse_training_log(log_path):
    """Parse training log into DataFrame."""
    epochs = []
    train_losses = []
    val_losses = []
    lrs = []
    times = []

    with open(log_path, 'r') as f:
        current_epoch = None
        for line in f:
            line = line.strip()

            # Parse epoch number
            if line.startswith("Epoch "):
                parts = line.split()
                if len(parts) >= 2:
                    epoch_str = parts[1].split('/')[0]
                    try:
                        current_epoch = int(epoch_str)
                    except ValueError:
                        pass

            # Parse metrics
            elif "Train Loss:" in line:
                try:
                    train_loss = float(line.split("Train Loss:")[1].strip())
                    train_losses.append(train_loss)
                except (ValueError, IndexError):
                    pass

            elif "Val Loss:" in line:
                try:
                    val_loss = float(line.split("Val Loss:")[1].strip())
                    val_losses.append(val_loss)
                except (ValueError, IndexError):
                    pass

            elif "LR:" in line:
                try:
                    lr_str = line.split("LR:")[1].strip().split()[0]
                    lr = float(lr_str)
                    lrs.append(lr)
                except (ValueError, IndexError):
                    pass

            elif "Time:" in line:
                try:
                    time_str = line.split("Time:")[1].strip().split('s')[0]
                    time = float(time_str)
                    times.append(time)
                    if current_epoch is not None:
                        epochs.append(current_epoch)
                except (ValueError, IndexError):
                    pass

    df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses[:len(epochs)],
        'val_loss': val_losses[:len(epochs)],
        'lr': lrs[:len(epochs)],
        'time': times[:len(epochs)],
    })

    return df

def main():
    project_root = Path(__file__).parent.parent.parent
    checkpoint_dir = project_root / "checkpoints" / "experiments"

    print("=" * 70)
    print("Phase 1 Experiment Analysis")
    print("=" * 70)
    print()

    # Load all experiment data
    experiment_data = {}
    for exp_id, exp_info in EXPERIMENTS.items():
        exp_dir = checkpoint_dir / exp_id
        log_path = exp_dir / "training_log.txt"

        if not log_path.exists():
            print(f"⚠ {exp_info['name']}: No results found")
            continue

        try:
            df = parse_training_log(log_path)
            if len(df) == 0:
                print(f"⚠ {exp_info['name']}: Empty log")
                continue

            experiment_data[exp_id] = df

            # Print summary
            best_val_loss = df['val_loss'].min()
            final_val_loss = df['val_loss'].iloc[-1]
            total_time = df['time'].sum() / 60  # Convert to minutes

            print(f"✓ {exp_info['name']}")
            print(f"  Change: {exp_info['key_change']}")
            print(f"  Best val loss: {best_val_loss:.6f}")
            print(f"  Final val loss: {final_val_loss:.6f}")
            print(f"  Total time: {total_time:.1f} min")
            print()

        except Exception as e:
            print(f"✗ {exp_info['name']}: Error parsing log - {e}")
            print()

    if not experiment_data:
        print("No experiment data found!")
        sys.exit(1)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Validation Loss Curves
    ax = axes[0, 0]
    for exp_id, df in experiment_data.items():
        exp_info = EXPERIMENTS[exp_id]
        linewidth = exp_info.get('linewidth', 2)
        ax.plot(
            df['epoch'],
            df['val_loss'],
            label=exp_info['name'],
            color=exp_info['color'],
            linewidth=linewidth,
            marker='o',
            markersize=4,
        )

    ax.axhline(0.514, color='red', linestyle='--', linewidth=2, label='Baseline (0.514)')
    ax.axhline(0.30, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Phase 1 Target (0.30)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Phase 1: Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Training Loss Curves
    ax = axes[0, 1]
    for exp_id, df in experiment_data.items():
        exp_info = EXPERIMENTS[exp_id]
        linewidth = exp_info.get('linewidth', 2)
        ax.plot(
            df['epoch'],
            df['train_loss'],
            label=exp_info['name'],
            color=exp_info['color'],
            linewidth=linewidth,
            marker='o',
            markersize=4,
        )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Phase 1: Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Learning Rate Schedules
    ax = axes[1, 0]
    for exp_id, df in experiment_data.items():
        exp_info = EXPERIMENTS[exp_id]
        ax.plot(
            df['epoch'],
            df['lr'],
            label=exp_info['name'],
            color=exp_info['color'],
            linewidth=2,
        )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedules', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Final Performance Summary (Bar Chart)
    ax = axes[1, 1]
    exp_names = []
    final_val_losses = []
    colors = []

    for exp_id, df in experiment_data.items():
        exp_info = EXPERIMENTS[exp_id]
        exp_names.append(exp_info['name'].replace(': ', ':\n'))  # Line break for readability
        final_val_losses.append(df['val_loss'].iloc[-1])
        colors.append(exp_info['color'])

    bars = ax.barh(exp_names, final_val_losses, color=colors, alpha=0.8)
    ax.axvline(0.514, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.axvline(0.30, color='green', linestyle='--', linewidth=2, label='Target')
    ax.set_xlabel('Final Validation Loss', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, final_val_losses)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path = checkpoint_dir / "phase1_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")

    # Find best experiment
    print()
    print("=" * 70)
    print("Recommendation")
    print("=" * 70)

    best_exp_id = min(experiment_data.items(), key=lambda x: x[1]['val_loss'].min())[0]
    best_df = experiment_data[best_exp_id]
    best_val_loss = best_df['val_loss'].min()

    print(f"✓ Best experiment: {EXPERIMENTS[best_exp_id]['name']}")
    print(f"  Best val loss: {best_val_loss:.6f}")

    improvement = (0.514 - best_val_loss) / 0.514 * 100
    print(f"  Improvement over baseline: {improvement:.1f}%")

    if best_val_loss < 0.30:
        print(f"  ✓ Phase 1 target achieved (< 0.30)!")
        print()
        print("Next steps:")
        print("  1. Use exp1f_combined config as baseline for Phase 2")
        print("  2. Scale to 1K-5K samples")
        print("  3. Add TBPTT for longer rollouts (timesteps=64)")
    else:
        print(f"  ⚠ Phase 1 target not achieved (current: {best_val_loss:.3f}, target: < 0.30)")
        print()
        print("Next steps:")
        print("  1. Investigate why improvements are limited")
        print("  2. Consider: longer training, different LR, more regularization")
        print("  3. May need to scale dataset size sooner")

if __name__ == "__main__":
    main()
