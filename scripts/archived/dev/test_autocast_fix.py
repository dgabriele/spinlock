#!/usr/bin/env python3
"""
Quick test to verify the autocast API fix works correctly.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from spinlock.operators.builder import OperatorBuilder, NeuralOperator
from spinlock.rollout.engine import OperatorRollout


def test_autocast_fix():
    """Test that the new autocast API works without warnings"""

    print("Testing new autocast API...")
    print(f"PyTorch version: {torch.__version__}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Build a small operator
    builder = OperatorBuilder()
    params = {
        "num_layers": 2,
        "base_channels": 16,
        "input_channels": 3,
        "output_channels": 3,
        "grid_size": 64,
        "kernel_size": 3,
        "normalization": "instance",
        "activation": "gelu",
        "dropout": 0.1,
        "has_stochastic": True,
        "noise_type": "gaussian",
        "noise_scale": 0.01,
    }

    base_model = builder.build_simple_cnn(params)
    base_model = base_model.to(device)
    operator = NeuralOperator(base_model, name="test_operator")

    # Create rollout engine with FP16
    engine = OperatorRollout(
        policy_type="convex",
        alpha=0.5,
        num_timesteps=10,  # Just 10 timesteps for quick test
        device=torch.device(device),
        precision="float16",
        compute_metrics=False
    )

    # Create initial condition
    initial_condition = torch.randn(3, 64, 64, device=device)

    # Run rollout (this will trigger autocast)
    print("\nRunning rollout with FP16...")
    trajectories, _, _ = engine.evolve_operator(
        operator=operator,
        initial_condition=initial_condition,
        num_realizations=2,
        base_seed=42
    )

    print(f"✓ Rollout completed successfully!")
    print(f"  Trajectory shape: {trajectories.shape}")
    print(f"  Trajectory dtype: {trajectories.dtype}")

    # Check if FutureWarning was raised
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Run one more step to capture warnings
        _ = engine.evolve_operator(
            operator=operator,
            initial_condition=initial_condition,
            num_realizations=2,
            base_seed=43
        )

        # Check for FutureWarning about autocast
        autocast_warnings = [warning for warning in w
                            if issubclass(warning.category, FutureWarning)
                            and 'autocast' in str(warning.message)]

        if autocast_warnings:
            print(f"\n✗ FAILED: FutureWarning still present!")
            for warning in autocast_warnings:
                print(f"  {warning.message}")
            return False
        else:
            print(f"\n✓ PASSED: No FutureWarning about autocast!")
            return True


if __name__ == "__main__":
    success = test_autocast_fix()
    sys.exit(0 if success else 1)
