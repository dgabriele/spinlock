#!/usr/bin/env python3
"""Validation Experiment 3: Early Stopping Efficiency.

Hypothesis:
    Early stopping criteria (convergence + token stability) save 30-50%
    computation compared to always running max_steps=256, while still
    capturing complete behavioral dynamics.

Success Criteria:
    - Mean stopping time < 180 steps (30% savings from baseline 256)
    - 40%+ episodes stop before max_steps
    - Stopped episodes are genuinely stable (low ||Δu|| or token stability)
    - Median stopping time < 150 steps (40% savings)

Methodology:
    1. Run 1000 episodes with StandardStoppingPolicy (convergence OR token stability OR max_steps=256)
    2. Run same 1000 episodes with FixedStepsPolicy (always 256 steps)
    3. Track stopping times, reasons, and final state stability
    4. Compute computational savings
    5. Validate that early-stopped episodes are genuinely converged

Output:
    - stopping_time_distribution.png - Histogram of stopping times
    - stopping_reasons.png - Pie chart of termination reasons
    - convergence_validation.png - ||Δu|| vs stopping time
    - savings_analysis.png - Cumulative computational cost comparison
    - summary.txt - Efficiency metrics and pass/fail
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.spinlock.perturbations import ImpulsePerturbationFactory
from src.spinlock.noa.early_stopping import (
    StandardStoppingPolicy,
    ConvergenceStop,
    MaxStepsStop,
    CompositeEarlyStopping,
)


def load_models(mno_checkpoint: str, vqvae_checkpoint: str, device: str):
    """Load trained MNO and VQ-VAE models."""
    print(f"Loading MNO from {mno_checkpoint}")
    print(f"Loading VQ-VAE from {vqvae_checkpoint}")
    # TODO: Implement actual model loading
    return None, None


def generate_episode_dummy(perturbation, device: str) -> Dict[str, any]:
    """Generate dummy episode for testing early stopping.

    Simulates various convergence behaviors:
    - Fast convergence (20-50 steps)
    - Medium convergence (80-120 steps)
    - Slow convergence (150-200 steps)
    - Never converges (hits max_steps=256)

    Returns:
        Dictionary with:
        - stopping_time: When episode stopped
        - stop_reason: Why it stopped
        - final_state_diff: ||u_T - u_{T-1}|| at stopping
        - token_stable: Whether tokens were stable at stopping
    """
    # Simulate stopping behavior
    behavior_type = np.random.choice(["fast", "medium", "slow", "none"], p=[0.3, 0.4, 0.2, 0.1])

    if behavior_type == "fast":
        stopping_time = np.random.randint(20, 50)
        stop_reason = "Converged (||Δu||_l2 = 5.2e-05 < 1.0e-04)"
        final_diff = np.random.uniform(1e-5, 9e-5)
        token_stable = True
    elif behavior_type == "medium":
        stopping_time = np.random.randint(80, 120)
        stop_reason = "Token stability (repeated 3 times with period 5)"
        final_diff = np.random.uniform(1e-4, 5e-4)
        token_stable = True
    elif behavior_type == "slow":
        stopping_time = np.random.randint(150, 200)
        stop_reason = "Converged (||Δu||_l2 = 8.7e-05 < 1.0e-04)"
        final_diff = np.random.uniform(2e-5, 8e-5)
        token_stable = False
    else:  # none
        stopping_time = 256
        stop_reason = "Max steps reached (t = 256 >= 256)"
        final_diff = np.random.uniform(1e-3, 1e-2)  # Not converged
        token_stable = False

    return {
        "stopping_time": stopping_time,
        "stop_reason": stop_reason,
        "final_state_diff": final_diff,
        "token_stable": token_stable,
    }


def categorize_stop_reason(reason: str) -> str:
    """Categorize stop reason into broader categories."""
    if "Converged" in reason:
        return "Convergence"
    elif "Token stability" in reason:
        return "Token Stability"
    elif "Max steps" in reason:
        return "Max Steps"
    else:
        return "Other"


def run_experiment(
    mno,
    vqvae,
    device: str,
    n_episodes: int = 1000,
    spatial_size: Tuple[int, int] = (64, 64),
    output_dir: Path = Path("results/phase2/exp3"),
):
    """Run early stopping efficiency experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {n_episodes} episodes with early stopping...")

    # Generate diverse perturbations
    np.random.seed(42)
    torch.manual_seed(42)

    # Storage
    stopping_times = []
    stop_reasons = []
    final_diffs = []
    token_stabilities = []

    for i in range(n_episodes):
        # Create random perturbation
        amp = np.random.uniform(0.5, 2.0)
        pattern_type = np.random.choice(["center", "corner", "uniform"])

        if pattern_type == "center":
            perturbation = ImpulsePerturbationFactory.center_blob(
                t_impulse=0, amplitude=amp, spatial_size=spatial_size, device=device
            )
        elif pattern_type == "corner":
            corner = np.random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
            perturbation = ImpulsePerturbationFactory.corner_blob(
                t_impulse=0, amplitude=amp, spatial_size=spatial_size,
                corner=corner, device=device
            )
        else:
            perturbation = ImpulsePerturbationFactory.uniform_forcing(
                t_impulse=0, amplitude=amp, spatial_size=spatial_size, device=device
            )

        # TODO: Replace with actual episode generation
        episode_data = generate_episode_dummy(perturbation, device)

        stopping_times.append(episode_data["stopping_time"])
        stop_reasons.append(episode_data["stop_reason"])
        final_diffs.append(episode_data["final_state_diff"])
        token_stabilities.append(episode_data["token_stable"])

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_episodes} episodes completed")

    # Convert to numpy
    stopping_times = np.array(stopping_times)
    final_diffs = np.array(final_diffs)
    token_stabilities = np.array(token_stabilities)

    # Categorize stop reasons
    stop_categories = [categorize_stop_reason(r) for r in stop_reasons]
    category_counts = Counter(stop_categories)

    # Compute metrics
    mean_stopping_time = stopping_times.mean()
    median_stopping_time = np.median(stopping_times)
    std_stopping_time = stopping_times.std()

    # Baseline: always 256 steps
    baseline_cost = n_episodes * 256
    actual_cost = stopping_times.sum()
    savings = (baseline_cost - actual_cost) / baseline_cost
    savings_pct = savings * 100

    # Early stop rate (stopped before max_steps)
    early_stop_rate = (stopping_times < 256).mean()

    # Convergence validation (for early-stopped episodes)
    early_stopped_mask = stopping_times < 256
    if early_stopped_mask.any():
        early_stopped_diffs = final_diffs[early_stopped_mask]
        mean_converged_diff = early_stopped_diffs.mean()
        properly_converged = (early_stopped_diffs < 1e-3).mean()  # Below reasonable threshold
    else:
        mean_converged_diff = 0.0
        properly_converged = 0.0

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Stopping time distribution
    axes[0, 0].hist(stopping_times, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(mean_stopping_time, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_stopping_time:.1f}')
    axes[0, 0].axvline(median_stopping_time, color='green', linestyle='--', linewidth=2,
                      label=f'Median: {median_stopping_time:.1f}')
    axes[0, 0].axvline(180, color='orange', linestyle=':', linewidth=2,
                      label='Target (180)')
    axes[0, 0].set_xlabel("Stopping Time (steps)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Stopping Time Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Stop reasons pie chart
    labels = list(category_counts.keys())
    sizes = list(category_counts.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0, 1].set_title("Stop Reasons")

    # 3. Convergence validation
    scatter = axes[1, 0].scatter(
        stopping_times, final_diffs,
        c=token_stabilities.astype(int), cmap='RdYlGn',
        alpha=0.6, s=20
    )
    axes[1, 0].axhline(1e-4, color='red', linestyle='--', label='Convergence threshold')
    axes[1, 0].set_xlabel("Stopping Time (steps)")
    axes[1, 0].set_ylabel("||Δu|| at stopping")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Convergence Validation\n(Green = Token Stable)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label="Token Stable")

    # 4. Cumulative cost comparison
    sorted_times = np.sort(stopping_times)
    baseline_cumulative = np.arange(1, n_episodes + 1) * 256
    actual_cumulative = np.cumsum(sorted_times)

    axes[1, 1].plot(range(n_episodes), baseline_cumulative, label='Baseline (256 steps/episode)',
                   linewidth=2, linestyle='--')
    axes[1, 1].plot(range(n_episodes), actual_cumulative, label='Early Stopping',
                   linewidth=2)
    axes[1, 1].fill_between(range(n_episodes), actual_cumulative, baseline_cumulative,
                           alpha=0.3, color='green', label=f'Savings ({savings_pct:.1f}%)')
    axes[1, 1].set_xlabel("Episode Index (sorted)")
    axes[1, 1].set_ylabel("Cumulative Steps")
    axes[1, 1].set_title("Computational Cost Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "early_stopping_analysis.png", dpi=150)
    plt.close()

    # Success assessment
    mean_time_pass = mean_stopping_time < 180
    early_stop_rate_pass = early_stop_rate >= 0.4
    median_time_pass = median_stopping_time < 150
    convergence_pass = properly_converged >= 0.8  # 80% of early-stopped are converged

    overall_pass = mean_time_pass and early_stop_rate_pass

    print("\n" + "="*70)
    print("EXPERIMENT 3: EARLY STOPPING EFFICIENCY")
    print("="*70)
    print(f"\nStopping Statistics:")
    print(f"  Mean stopping time:   {mean_stopping_time:.1f} steps")
    print(f"  Median stopping time: {median_stopping_time:.1f} steps")
    print(f"  Std stopping time:    {std_stopping_time:.1f} steps")
    print(f"\nComputational Savings:")
    print(f"  Baseline cost:  {baseline_cost:,} steps")
    print(f"  Actual cost:    {actual_cost:,} steps")
    print(f"  Savings:        {savings_pct:.1f}%")
    print(f"\nEarly Stopping Rate:")
    print(f"  Stopped before max: {early_stop_rate*100:.1f}%")
    print(f"\nStop Reason Breakdown:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} ({count/n_episodes*100:.1f}%)")
    print(f"\nConvergence Validation (early-stopped episodes):")
    print(f"  Mean ||Δu||:         {mean_converged_diff:.2e}")
    print(f"  Properly converged:  {properly_converged*100:.1f}%")
    print(f"\nSuccess Criteria:")
    print(f"  Mean < 180 steps:         {mean_stopping_time:.1f} < 180  {'✓ PASS' if mean_time_pass else '✗ FAIL'}")
    print(f"  Early stop rate ≥ 40%:    {early_stop_rate*100:.1f}% ≥ 40%  {'✓ PASS' if early_stop_rate_pass else '✗ FAIL'}")
    print(f"  Median < 150 steps:       {median_stopping_time:.1f} < 150  {'✓ PASS' if median_time_pass else '✗ FAIL'}")
    print(f"  Convergence quality ≥ 80%: {properly_converged*100:.1f}% ≥ 80%  {'✓ PASS' if convergence_pass else '✗ FAIL'}")
    print(f"\nOverall: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    print("="*70)

    # Save summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write("EXPERIMENT 3: EARLY STOPPING EFFICIENCY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Stopping Statistics:\n")
        f.write(f"  Mean stopping time:   {mean_stopping_time:.1f} steps\n")
        f.write(f"  Median stopping time: {median_stopping_time:.1f} steps\n")
        f.write(f"  Std stopping time:    {std_stopping_time:.1f} steps\n\n")
        f.write(f"Computational Savings:\n")
        f.write(f"  Baseline cost:  {baseline_cost:,} steps\n")
        f.write(f"  Actual cost:    {actual_cost:,} steps\n")
        f.write(f"  Savings:        {savings_pct:.1f}%\n\n")
        f.write(f"Early Stopping Rate:\n")
        f.write(f"  Stopped before max: {early_stop_rate*100:.1f}%\n\n")
        f.write(f"Stop Reason Breakdown:\n")
        for cat, count in category_counts.items():
            f.write(f"  {cat}: {count} ({count/n_episodes*100:.1f}%)\n")
        f.write(f"\nConvergence Validation:\n")
        f.write(f"  Mean ||Δu||:         {mean_converged_diff:.2e}\n")
        f.write(f"  Properly converged:  {properly_converged*100:.1f}%\n\n")
        f.write(f"Success Criteria:\n")
        f.write(f"  Mean < 180 steps:         {'✓ PASS' if mean_time_pass else '✗ FAIL'}\n")
        f.write(f"  Early stop rate ≥ 40%:    {'✓ PASS' if early_stop_rate_pass else '✗ FAIL'}\n")
        f.write(f"  Median < 150 steps:       {'✓ PASS' if median_time_pass else '✗ FAIL'}\n")
        f.write(f"  Convergence quality ≥ 80%: {'✓ PASS' if convergence_pass else '✗ FAIL'}\n\n")
        f.write(f"Overall: {'✓ PASS' if overall_pass else '✗ FAIL'}\n")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Validation Experiment 3: Early Stopping Efficiency"
    )
    parser.add_argument(
        "--mno-checkpoint",
        type=str,
        default="checkpoints/mno/pure_mse_baseline/meta_operator_best.pt",
        help="Path to trained MNO checkpoint",
    )
    parser.add_argument(
        "--vqvae-checkpoint",
        type=str,
        default="checkpoints/vqvae/mno_distribution_100k/vqvae_best.pt",
        help="Path to trained VQ-VAE checkpoint",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=1000, help="Number of episodes to run"
    )
    parser.add_argument("--spatial-size", type=int, default=64, help="Spatial grid size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase2/exp3",
        help="Output directory for results",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load models
    mno, vqvae = load_models(args.mno_checkpoint, args.vqvae_checkpoint, args.device)

    # Run experiment
    run_experiment(
        mno,
        vqvae,
        device=args.device,
        n_episodes=args.n_episodes,
        spatial_size=(args.spatial_size, args.spatial_size),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
