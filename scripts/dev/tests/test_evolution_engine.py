#!/usr/bin/env python3
"""
Test script for temporal evolution engine.

Tests the core evolution engine with all three update policies
and verifies functionality end-to-end.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spinlock.rollout import (
    OperatorRollout,
    AutoregressivePolicy,
    ResidualPolicy,
    ConvexPolicy,
    create_update_policy,
    InitialConditionSampler,
)
from spinlock.operators.builder import OperatorBuilder


def create_simple_operator(device):
    """Create a simple test operator."""
    params = {
        "num_layers": 2,
        "base_channels": 16,
        "input_channels": 3,
        "output_channels": 3,
        "kernel_size": 3,
        "activation": "gelu",
        "normalization": "instance",
        "dropout_rate": 0.0,
        "noise_type": "gaussian",
        "noise_scale": 0.01,
        "grid_size": 64,
        # Evolution params (not used by builder, but for completeness)
        "update_policy": "convex",
        "alpha": 0.7,
        "dt": 0.01,
    }

    builder = OperatorBuilder()
    model = builder.build_simple_cnn(params)
    return model.to(device)


def test_update_policies():
    """Test all three update policies."""
    print("\n" + "="*60)
    print("TEST 1: Update Policies")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_prev = torch.randn(1, 3, 64, 64, device=device)
    O_theta_X = torch.randn(1, 3, 64, 64, device=device)

    # Test autoregressive
    policy = AutoregressivePolicy()
    X_next = policy.update(X_prev, O_theta_X)
    assert torch.allclose(X_next, O_theta_X), "Autoregressive failed"
    print(f"✓ {policy.name()}: X_next == O_theta_X")

    # Test residual
    policy = ResidualPolicy(dt=0.1)
    X_next = policy.update(X_prev, O_theta_X)
    expected = X_prev + 0.1 * O_theta_X
    assert torch.allclose(X_next, expected), "Residual failed"
    print(f"✓ {policy.name()}: X_next == X_prev + dt * O_theta_X")

    # Test convex
    policy = ConvexPolicy(alpha=0.7)
    X_next = policy.update(X_prev, O_theta_X)
    expected = 0.7 * X_prev + 0.3 * O_theta_X
    assert torch.allclose(X_next, expected), "Convex failed"
    print(f"✓ {policy.name()}: X_next == alpha * X_prev + (1-alpha) * O_theta_X")

    # Test factory
    policy = create_update_policy("convex", alpha=0.5)
    assert isinstance(policy, ConvexPolicy), "Factory failed"
    print(f"✓ Factory pattern works")

    print("\n✅ All policy tests passed!")


def test_single_evolution():
    """Test evolving a single operator."""
    print("\n" + "="*60)
    print("TEST 2: Single Operator Evolution")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create operator
    print("\nCreating neural operator...")
    operator = create_simple_operator(device)
    print(f"✓ Operator created with {sum(p.numel() for p in operator.parameters()):,} parameters")

    # Create initial condition
    print("\nCreating initial condition...")
    X0 = torch.randn(3, 64, 64, device=device)
    print(f"✓ Initial condition shape: {X0.shape}")

    # Create rollout engine
    print("\nCreating rollout engine...")
    rollout = OperatorRollout(
        policy_type="convex",
        alpha=0.7,
        num_timesteps=20,
        device=device,
        compute_metrics=True
    )
    print(f"✓ Rollout created with policy: {rollout.policy.name()}")

    # Evolve
    print("\nEvolving operator...")
    trajectories, metrics = rollout.evolve_operator(
        operator=operator,
        initial_condition=X0,
        num_realizations=3,
        base_seed=42,
        show_progress=True
    )

    print(f"\n✓ Trajectories shape: {trajectories.shape}")
    print(f"  Expected: [3, 20, 3, 64, 64]")
    assert trajectories.shape == (3, 20, 3, 64, 64), "Shape mismatch"

    print(f"✓ Metrics length: {len(metrics)} realizations")
    print(f"  Each realization: {len(metrics[0])} timesteps")
    assert len(metrics) == 3, "Wrong number of realizations"
    assert len(metrics[0]) == 20, "Wrong number of timesteps"

    # Check metrics content
    first_metric = metrics[0][0]
    print(f"\n✓ Metrics at t=0:")
    print(f"  Energy: {first_metric.energy:.4f}")
    print(f"  Entropy: {first_metric.entropy:.4f}")
    print(f"  Variance: {first_metric.variance:.4f}")
    print(f"  Mean magnitude: {first_metric.mean_magnitude:.4f}")
    print(f"  Autocorrelation: {first_metric.autocorrelation:.4f}")

    print("\n✅ Single evolution test passed!")
    return trajectories, metrics


def test_all_policies():
    """Test evolution with all three policies."""
    print("\n" + "="*60)
    print("TEST 3: All Policies Evolution")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    operator = create_simple_operator(device)
    X0 = torch.randn(3, 64, 64, device=device)

    policies = [
        ("autoregressive", {}),
        ("residual", {"dt": 0.01}),
        ("convex", {"alpha": 0.7}),
    ]

    for policy_type, kwargs in policies:
        print(f"\nTesting {policy_type} policy...")

        rollout = OperatorRollout(
            policy_type=policy_type,
            num_timesteps=10,
            device=device,
            compute_metrics=False,  # Faster
            **kwargs
        )

        trajectories, _ = rollout.evolve_operator(
            operator=operator,
            initial_condition=X0,
            num_realizations=2,
            base_seed=42
        )

        assert trajectories.shape == (2, 10, 3, 64, 64), f"{policy_type} failed"
        print(f"  ✓ Shape: {trajectories.shape}")
        print(f"  ✓ Energy range: [{trajectories.min().item():.2f}, {trajectories.max().item():.2f}]")

    print("\n✅ All policies test passed!")


def test_initial_condition_sampler():
    """Test initial condition sampling."""
    print("\n" + "="*60)
    print("TEST 4: Initial Condition Sampler")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test GRF sampling
    print("\nTesting GRF sampling...")
    sampler = InitialConditionSampler(
        method="grf",
        grid_size=64,
        num_channels=3,
        device=device
    )

    X0 = sampler.sample(batch_size=5, length_scale=0.1, seed=42)
    assert X0.shape == (5, 3, 64, 64), "GRF shape mismatch"
    print(f"✓ GRF shape: {X0.shape}")
    print(f"✓ GRF range: [{X0.min().item():.2f}, {X0.max().item():.2f}]")

    # Test zeros sampling
    print("\nTesting zeros sampling...")
    sampler = InitialConditionSampler(
        method="zeros",
        grid_size=64,
        num_channels=3,
        device=device
    )

    X0 = sampler.sample(batch_size=5)
    assert X0.shape == (5, 3, 64, 64), "Zeros shape mismatch"
    assert torch.allclose(X0, torch.zeros_like(X0)), "Not all zeros"
    print(f"✓ Zeros shape: {X0.shape}")
    print(f"✓ All zeros: {(X0 == 0).all().item()}")

    print("\n✅ Initial condition sampler test passed!")


def test_batch_evolution():
    """Test batch evolution."""
    print("\n" + "="*60)
    print("TEST 5: Batch Evolution")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create multiple operators
    print("\nCreating 3 operators...")
    operators = [create_simple_operator(device) for _ in range(3)]
    print(f"✓ Created {len(operators)} operators")

    # Create initial conditions
    initial_conditions = torch.randn(3, 3, 64, 64, device=device)
    print(f"✓ Initial conditions shape: {initial_conditions.shape}")

    # Create rollout
    rollout = OperatorRollout(
        policy_type="convex",
        alpha=0.7,
        num_timesteps=10,
        device=device,
        compute_metrics=True
    )

    # Evolve batch (in memory)
    print("\nEvolving batch...")
    trajectories, metrics = rollout.evolve_batch(
        operators=operators,
        initial_conditions=initial_conditions,
        num_realizations=2,
        show_progress=True
    )

    print(f"\n✓ Trajectories shape: {trajectories.shape}")
    print(f"  Expected: [3, 2, 10, 3, 64, 64]")
    assert trajectories.shape == (3, 2, 10, 3, 64, 64), "Batch shape mismatch"

    print(f"✓ Metrics structure: {len(metrics)} operators")
    print(f"  Each operator: {len(metrics[0])} realizations")
    print(f"  Each realization: {len(metrics[0][0])} timesteps")

    print("\n✅ Batch evolution test passed!")


def test_normalization_and_clamping():
    """Test post-processing."""
    print("\n" + "="*60)
    print("TEST 6: Normalization and Clamping")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    operator = create_simple_operator(device)
    X0 = torch.randn(3, 64, 64, device=device) * 10  # Large values

    # Test minmax normalization
    print("\nTesting minmax normalization...")
    rollout = OperatorRollout(
        policy_type="convex",
        alpha=0.7,
        num_timesteps=5,
        device=device,
        normalization="minmax",
        compute_metrics=False
    )

    trajectories, _ = rollout.evolve_operator(
        operator=operator,
        initial_condition=X0,
        num_realizations=1,
        base_seed=42
    )

    print(f"✓ Output range: [{trajectories.min().item():.4f}, {trajectories.max().item():.4f}]")
    assert trajectories.min() >= 0 and trajectories.max() <= 1, "Normalization failed"

    # Test clamping
    print("\nTesting clamping...")
    rollout = OperatorRollout(
        policy_type="convex",
        alpha=0.7,
        num_timesteps=5,
        device=device,
        clamp_range=(-1.0, 1.0),
        compute_metrics=False
    )

    trajectories, _ = rollout.evolve_operator(
        operator=operator,
        initial_condition=X0,
        num_realizations=1,
        base_seed=42
    )

    print(f"✓ Output range: [{trajectories.min().item():.4f}, {trajectories.max().item():.4f}]")
    assert trajectories.min() >= -1.0 and trajectories.max() <= 1.0, "Clamping failed"

    print("\n✅ Post-processing test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("OPERATOR ROLLOUT TEST SUITE")
    print("="*60)

    try:
        test_update_policies()
        test_single_evolution()
        test_all_policies()
        test_initial_condition_sampler()
        test_batch_evolution()
        test_normalization_and_clamping()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe operator rollout is working correctly.")
        print("You can now use it for:")
        print("  • Visualization")
        print("  • Feature extraction")
        print("  • Scientific analysis")
        print("  • Model debugging and replay")
        print("="*60 + "\n")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
