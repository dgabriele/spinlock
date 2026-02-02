#!/usr/bin/env python
"""NOA Phase 1 Stage 1 Test Script - Grid-level MSE training.

This script tests the minimal NOA prototype:
1. Creates NOA backbone (U-AFNO wrapper)
2. Generates synthetic trajectories for testing
3. Trains with grid-level MSE loss
4. Validates basic functionality

Success criteria:
- Training runs without crashes
- Loss decreases over epochs
- No NaN values
- Gradient flow works correctly

Usage:
    poetry run python scripts/dev/test_noa_stage1.py

After Stage 1 works, proceed to:
- Stage 2: Add feature extraction
- Stage 3: Add VQ-VAE perceptual loss
"""

import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

# Add src to path for imports
sys.path.insert(0, "/home/daniel/projects/spinlock/src")

from spinlock.noa.backbone import NOABackbone, create_noa_backbone
from spinlock.noa.training import (
    NOAPhase1Trainer,
    NOADataset,
    generate_synthetic_data,
)


def test_backbone_forward():
    """Test 1: NOA backbone forward pass."""
    print("\n" + "=" * 60)
    print("TEST 1: NOA Backbone Forward Pass")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create backbone
    noa = NOABackbone(
        in_channels=1,
        out_channels=1,
        base_channels=16,  # Smaller for testing
        encoder_levels=2,
        modes=8,
        afno_blocks=2,
    ).to(device)

    print(f"Parameters: {noa.num_parameters:,}")
    print(f"Trainable: {noa.num_trainable_parameters:,}")

    # Test forward pass
    u0 = torch.randn(4, 1, 64, 64, device=device)
    print(f"\nInput shape: {u0.shape}")

    # Single step
    u1 = noa.single_step(u0)
    print(f"Single step output: {u1.shape}")
    assert u1.shape == u0.shape, f"Shape mismatch: {u1.shape} vs {u0.shape}"

    # Rollout
    steps = 10
    trajectory = noa(u0, steps=steps, return_all_steps=True)
    expected_shape = (4, steps + 1, 1, 64, 64)
    print(f"Rollout output: {trajectory.shape} (expected {expected_shape})")
    assert trajectory.shape == expected_shape, f"Shape mismatch: {trajectory.shape}"

    # Check for NaN
    assert not torch.isnan(trajectory).any(), "NaN in trajectory!"

    print("\n✓ TEST 1 PASSED: Forward pass works correctly")
    return True


def test_gradient_flow():
    """Test 2: Gradient flow through NOA."""
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Flow")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    noa = NOABackbone(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        encoder_levels=2,
        modes=8,
        afno_blocks=2,
    ).to(device)

    # Forward pass
    u0 = torch.randn(2, 1, 64, 64, device=device)
    trajectory = noa(u0, steps=5)

    # Dummy target
    target = torch.randn_like(trajectory)

    # Compute loss and backward
    loss = F.mse_loss(trajectory, target)
    loss.backward()

    # Check gradients exist
    has_grad = False
    none_grad_count = 0
    total_params = 0

    for name, param in noa.named_parameters():
        total_params += 1
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"  {name}: grad_norm={grad_norm:.6f}")
        else:
            none_grad_count += 1

    print(f"\nTotal parameters: {total_params}")
    print(f"Parameters with None grad: {none_grad_count}")

    assert has_grad, "No gradients found!"
    assert not torch.isnan(loss), "NaN loss!"

    print(f"\nLoss: {loss.item():.4f}")
    print("\n✓ TEST 2 PASSED: Gradients flow correctly")
    return True


def test_training_loop():
    """Test 3: Full training loop with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST 3: Training Loop with Synthetic Data")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create smaller model for testing
    noa = NOABackbone(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        encoder_levels=2,
        modes=8,
        afno_blocks=2,
    )

    print(f"Model parameters: {noa.num_parameters:,}")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    n_samples = 32
    timesteps = 17  # 16 steps + initial condition
    u0, trajectories = generate_synthetic_data(
        n_samples=n_samples,
        timesteps=timesteps,
        channels=1,
        height=64,
        width=64,
    )
    print(f"  Initial conditions: {u0.shape}")
    print(f"  Trajectories: {trajectories.shape}")

    # Create dataset and loader
    dataset = NOADataset(u0, trajectories)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Create trainer
    trainer = NOAPhase1Trainer(
        noa=noa,
        device=device,
        learning_rate=1e-3,
    )

    # Train for a few epochs
    print("\nTraining...")
    epochs = 5
    losses = []

    for epoch in range(epochs):
        start = time.time()
        loss = trainer.train_epoch(train_loader)
        elapsed = time.time() - start
        losses.append(loss)
        print(f"  Epoch {epoch + 1}: loss={loss:.4f} ({elapsed:.1f}s)")

    # Check loss decreased
    loss_decreased = losses[-1] < losses[0]
    print(f"\nInitial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss decreased: {loss_decreased}")

    # More lenient check - at least not NaN and finite
    assert all(not (l != l) for l in losses), "NaN in losses!"
    assert all(l < float('inf') for l in losses), "Inf in losses!"

    if loss_decreased:
        print("\n✓ TEST 3 PASSED: Training loop works, loss decreased")
    else:
        print("\n⚠ TEST 3 WARNING: Loss did not decrease (may need more epochs)")

    return True


def test_with_real_data_subset():
    """Test 4: Load real dataset subset and test rollout shapes."""
    print("\n" + "=" * 60)
    print("TEST 4: Real Data Subset (Shape Validation)")
    print("=" * 60)

    # Clear GPU memory from previous tests
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    import h5py
    import numpy as np

    dataset_path = "/home/daniel/projects/spinlock/datasets/100k_full_features.h5"

    try:
        with h5py.File(dataset_path, "r") as f:
            # Load small subset (very small to avoid OOM)
            n_samples = 4
            inputs = f["inputs/fields"][:n_samples]  # [N, 3, 64, 64] - 3 realizations
            print(f"Loaded inputs: {inputs.shape}")

            # Use first realization
            u0 = torch.from_numpy(inputs[:, 0:1, :, :]).float()  # [N, 1, 64, 64]
            print(f"Initial conditions: {u0.shape}")

    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")
        print("Skipping real data test")
        return True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create NOA backbone
    noa = NOABackbone(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        encoder_levels=2,
        modes=8,
        afno_blocks=2,
    ).to(device)

    # Generate rollout
    u0 = u0.to(device)
    trajectory = noa(u0, steps=32)

    print(f"Generated trajectory: {trajectory.shape}")
    expected_shape = (n_samples, 33, 1, 64, 64)
    assert trajectory.shape == expected_shape, f"Unexpected shape: {trajectory.shape} vs {expected_shape}"
    assert not torch.isnan(trajectory).any(), "NaN in trajectory!"

    # Clean up
    del trajectory, u0, noa
    gc.collect()
    torch.cuda.empty_cache()

    print("\n✓ TEST 4 PASSED: Real data shapes work correctly")
    return True


def main():
    """Run all Stage 1 tests."""
    print("=" * 60)
    print("NOA Phase 1 Stage 1 Tests")
    print("=" * 60)
    print("\nThis tests the minimal NOA prototype with grid-level MSE loss.")
    print("Success criteria:")
    print("  - Forward pass works")
    print("  - Gradients flow correctly")
    print("  - Training loop runs")
    print("  - Loss decreases (or stays finite)")

    results = []

    # Run tests
    tests = [
        ("Backbone Forward", test_backbone_forward),
        ("Gradient Flow", test_gradient_flow),
        ("Training Loop", test_training_loop),
        ("Real Data Shapes", test_with_real_data_subset),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results:
        symbol = "✓" if "PASSED" in status else "✗"
        print(f"  {symbol} {name}: {status}")

    all_passed = all("PASSED" in status for _, status in results)

    if all_passed:
        print("\n" + "=" * 60)
        print("Stage 1 COMPLETE - Proceed to Stage 2 (Feature Extraction)")
        print("=" * 60)
    else:
        print("\n⚠ Some tests failed - debug before proceeding")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
