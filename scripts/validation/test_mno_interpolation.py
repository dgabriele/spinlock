#!/usr/bin/env python
"""Test MNO parameter space interpolation behavior.

Investigates whether MNO rollouts in interpolated parameter space are:
1. Smooth & predictable (linear blends)
2. Non-predictable (phase transitions, discontinuities)

This informs the alignment layer design:
- If smooth: alignment can use simple linear mappings
- If non-predictable: alignment needs to handle regime transitions
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List

def load_mno(checkpoint_path: str, device: str = "cuda"):
    """Load trained MNO model."""
    from spinlock.mno.validation_utils import load_mno_checkpoint
    mno = load_mno_checkpoint(checkpoint_path, device=device)
    mno.eval()
    return mno

def load_cno_for_params(params: torch.Tensor, device: str = "cuda"):
    """Create CNO operator for given parameters."""
    from spinlock.operators.cno import ConvolutionalNeuralOperator
    from spinlock.operators.config import CNOConfig

    # Convert normalized params [0,1] back to CNO config
    # TODO: Implement proper parameter unpacking
    config = CNOConfig(
        grid_size=64,
        num_channels=3,
        # ... unpack params to config fields
    )
    cno = ConvolutionalNeuralOperator(config, device=device)
    return cno

def sample_parameter_pairs(dataset_path: str, n_pairs: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sample pairs of parameters from dataset for interpolation testing."""
    with h5py.File(dataset_path, 'r') as f:
        params = f['parameters/params'][:]
        n = len(params)

        pairs = []
        for _ in range(n_pairs):
            # Sample two different parameter points
            idx_a, idx_b = np.random.choice(n, size=2, replace=False)
            pairs.append((params[idx_a], params[idx_b]))

    return pairs

def interpolate_parameters(theta_a: np.ndarray, theta_b: np.ndarray,
                          n_steps: int = 5) -> np.ndarray:
    """Linearly interpolate between two parameter vectors.

    Returns:
        Array of shape [n_steps, param_dim] with interpolated parameters
    """
    t_values = np.linspace(0, 1, n_steps)
    interp_params = np.array([
        (1 - t) * theta_a + t * theta_b
        for t in t_values
    ])
    return interp_params

def generate_rollout(model, params: torch.Tensor, ic: torch.Tensor,
                    steps: int = 256) -> torch.Tensor:
    """Generate rollout from model."""
    with torch.no_grad():
        rollout = model.rollout(ic, params=params, steps=steps, return_all_steps=True)
    return rollout

def measure_interpolation_smoothness(rollouts: List[torch.Tensor]) -> dict:
    """Measure smoothness of rollout sequence.

    Args:
        rollouts: List of [T, C, H, W] rollouts at interpolated parameters

    Returns:
        Dictionary with smoothness metrics:
        - mse_diffs: MSE between consecutive rollouts
        - second_order_diffs: Acceleration (change in change)
        - is_smooth: Boolean, True if changes are gradual
    """
    mse_diffs = []
    for i in range(len(rollouts) - 1):
        mse = torch.mean((rollouts[i] - rollouts[i+1])**2).item()
        mse_diffs.append(mse)

    # Second-order differences (acceleration)
    second_order = np.diff(mse_diffs) if len(mse_diffs) > 1 else []

    # Heuristic: smooth if MSE changes are roughly constant (low acceleration)
    is_smooth = np.std(second_order) < np.mean(mse_diffs) * 0.5 if len(second_order) > 0 else True

    return {
        'mse_diffs': mse_diffs,
        'second_order_diffs': second_order.tolist() if len(second_order) > 0 else [],
        'is_smooth': is_smooth,
        'mean_mse_change': np.mean(mse_diffs),
        'std_mse_change': np.std(mse_diffs),
    }

def compare_mno_cno_interpolation(mno, theta_a, theta_b, ic,
                                 n_steps: int = 5, device: str = "cuda"):
    """Compare MNO vs CNO behavior under parameter interpolation.

    Returns:
        Dictionary with:
        - mno_rollouts: List of MNO rollouts at interpolated points
        - cno_rollouts: List of CNO rollouts at interpolated points
        - mno_smoothness: Smoothness metrics for MNO
        - cno_smoothness: Smoothness metrics for CNO
        - discrepancy: MSE between MNO and CNO at each point
    """
    # Generate interpolated parameters
    interp_params = interpolate_parameters(theta_a, theta_b, n_steps)

    mno_rollouts = []
    cno_rollouts = []
    discrepancies = []

    for params in interp_params:
        params_t = torch.from_numpy(params).float().to(device).unsqueeze(0)  # [1, param_dim]

        # MNO rollout
        mno_rollout = generate_rollout(mno, params_t, ic)
        mno_rollouts.append(mno_rollout)

        # CNO rollout (if available)
        # TODO: Implement CNO generation for arbitrary params
        # cno = load_cno_for_params(params_t, device)
        # cno_rollout = generate_rollout(cno, params_t, ic)
        # cno_rollouts.append(cno_rollout)

        # # Discrepancy
        # mse = torch.mean((mno_rollout - cno_rollout)**2).item()
        # discrepancies.append(mse)

    # Measure smoothness
    mno_smoothness = measure_interpolation_smoothness(mno_rollouts)
    # cno_smoothness = measure_interpolation_smoothness(cno_rollouts)

    return {
        'mno_rollouts': mno_rollouts,
        # 'cno_rollouts': cno_rollouts,
        'mno_smoothness': mno_smoothness,
        # 'cno_smoothness': cno_smoothness,
        # 'discrepancies': discrepancies,
        'interpolated_params': interp_params,
    }

def visualize_interpolation_path(results: dict, output_path: Path):
    """Visualize rollouts along interpolation path."""
    mno_rollouts = results['mno_rollouts']
    n_steps = len(mno_rollouts)

    fig, axes = plt.subplots(2, n_steps, figsize=(4*n_steps, 8))

    for i, rollout in enumerate(mno_rollouts):
        # Rollout shape: [B, T, C, H, W] - take first batch, first channel
        # Show t=0 (initial)
        axes[0, i].imshow(rollout[0, 0, 0].cpu().numpy(), cmap='viridis')
        axes[0, i].set_title(f't=0, α={i/(n_steps-1):.2f}')
        axes[0, i].axis('off')

        # Show t=128 (mid-rollout)
        axes[1, i].imshow(rollout[0, 128, 0].cpu().numpy(), cmap='viridis')
        axes[1, i].set_title(f't=128, α={i/(n_steps-1):.2f}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Run interpolation analysis."""
    # Configuration
    mno_checkpoint = "checkpoints/mno/50k_baseline/meta_operator_best.pt"
    dataset_path = "datasets/50k_baseline.h5"
    output_dir = Path("diagnostics/mno_interpolation")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MNO
    print("Loading MNO...")
    mno = load_mno(mno_checkpoint, device=device)

    # Sample parameter pairs (1.5x minimum for statistical significance)
    print("Sampling parameter pairs...")
    pairs = sample_parameter_pairs(dataset_path, n_pairs=45)

    # Generate random IC for testing
    ic = torch.randn(1, 3, 64, 64, device=device)

    # Test each pair
    results_all = []
    for pair_idx, (theta_a, theta_b) in enumerate(pairs):
        print(f"\nTesting pair {pair_idx + 1}/{len(pairs)}...")

        results = compare_mno_cno_interpolation(
            mno, theta_a, theta_b, ic,
            n_steps=5, device=device
        )

        # Print smoothness analysis
        smoothness = results['mno_smoothness']
        print(f"  MNO smoothness: {'SMOOTH' if smoothness['is_smooth'] else 'NON-SMOOTH'}")
        print(f"    Mean MSE change: {smoothness['mean_mse_change']:.6f}")
        print(f"    Std MSE change: {smoothness['std_mse_change']:.6f}")

        # Visualize (only first 10 to save time)
        if pair_idx < 10:
            vis_path = output_dir / f"interpolation_pair_{pair_idx}.png"
            visualize_interpolation_path(results, vis_path)
            print(f"    Visualization saved: {vis_path}")

        results_all.append(results)

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    smooth_count = sum(r['mno_smoothness']['is_smooth'] for r in results_all)
    print(f"Smooth interpolations: {smooth_count}/{len(results_all)}")
    print(f"Non-smooth interpolations: {len(results_all) - smooth_count}/{len(results_all)}")

    if smooth_count == len(results_all):
        print("\n✓ MNO exhibits SMOOTH interpolation behavior")
        print("  → Alignment layer can use simple linear mappings")
    else:
        print("\n⚠ MNO exhibits NON-SMOOTH interpolation behavior")
        print("  → Alignment layer needs to handle regime transitions")

if __name__ == "__main__":
    main()
