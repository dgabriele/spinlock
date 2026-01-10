#!/usr/bin/env python3
"""Simple visualization of Phase 1 results."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Results from completed experiments
EXPERIMENTS = [
    ("1E", "Stronger\nRegularization", 0.454, "#2ecc71", "✓ Best"),
    ("1C", "No Cache\nClearing", 0.463, "#3498db", "✓ Great"),
    ("1A", "LR Warmup", 1.317, "#e74c3c", "⚠ Poor"),
    ("1F", "Combined\nBest", 1.511, "#e67e22", "⚠ Poor"),
    ("1B", "Gradient\nAccumulation", 4.135, "#95a5a6", "✗ Failed"),
]

BASELINE = 0.514

def main():
    project_root = Path(__file__).parent.parent.parent

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Bar chart of final validation losses
    exp_ids = [e[0] for e in EXPERIMENTS]
    names = [e[1] for e in EXPERIMENTS]
    losses = [e[2] for e in EXPERIMENTS]
    colors = [e[3] for e in EXPERIMENTS]

    bars = ax1.barh(range(len(EXPERIMENTS)), losses, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(EXPERIMENTS)))
    ax1.set_yticklabels([f"{e[0]}: {e[1]}" for e in EXPERIMENTS])
    ax1.axvline(BASELINE, color='red', linestyle='--', linewidth=2, label=f'Baseline ({BASELINE})')
    ax1.set_xlabel('Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Phase 1: Final Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, losses)):
        label_x = loss + 0.1
        if loss > 2.0:  # For very large values, put label inside
            label_x = loss - 0.3
            color = 'white'
        else:
            color = 'black'
        ax1.text(label_x, i, f'{loss:.3f}', va='center', fontsize=10, fontweight='bold', color=color)

    # Plot 2: Improvement vs baseline
    improvements = [(BASELINE - loss) / BASELINE * 100 for loss in losses]
    colors_sorted = [colors[i] for i in range(len(improvements))]

    bars2 = ax2.barh(range(len(EXPERIMENTS)), improvements, color=colors_sorted, alpha=0.8)
    ax2.set_yticks(range(len(EXPERIMENTS)))
    ax2.set_yticklabels([f"{e[0]}: {e[1]}" for e in EXPERIMENTS])
    ax2.axvline(0, color='red', linestyle='-', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Improvement vs Baseline (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Phase 1: Improvement Over Baseline (0.514)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars2, improvements)):
        if imp > 0:
            label_x = imp + 1
            label = f'+{imp:.1f}%'
            color = 'black'
        else:
            label_x = imp - 10
            label = f'{imp:.1f}%'
            color = 'white'
        ax2.text(label_x, i, label, va='center', fontsize=10, fontweight='bold', color=color)

    plt.tight_layout()

    # Save figure
    output_path = project_root / "checkpoints" / "experiments" / "phase1_results.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Also save as PDF for high quality
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ PDF version saved to: {pdf_path}")

    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Baseline: {BASELINE}")
    print(f"Best result: {min(losses):.6f} ({EXPERIMENTS[0][0]})")
    print(f"Improvement: {max(improvements):.1f}%")
    print(f"Worst result: {max(losses):.6f} ({EXPERIMENTS[-1][0]})")
    print()

if __name__ == "__main__":
    main()
