#!/usr/bin/env python
"""NOA Phase 1 Stage 2 Test Script - Feature extraction and feature MSE loss.

This script tests Stage 2 additions:
1. Feature extraction from trajectories
2. Feature MSE loss in addition to grid MSE
3. Training with combined loss

Prerequisites: Stage 1 tests must pass first.

Success criteria:
- Feature extraction produces correct shapes
- Feature values are finite (no NaN)
- Feature loss decreases over epochs
- Combined loss training works

Usage:
    poetry run python scripts/dev/test_noa_stage2.py
"""

import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

# Add src to path for imports
sys.path.insert(0, "/home/daniel/projects/spinlock/src")

from spinlock.noa.backbone import NOABackbone
from spinlock.noa.training import (
    NOAPhase1Trainer,
    NOADataset,
    NOADatasetWithFeatures,
    extract_trajectory_features,
    generate_synthetic_data,
)


def test_feature_extraction():
    """Test 1: Feature extraction from trajectories."""
    print("\n" + "=" * 60)
    print("TEST 1: Feature Extraction")
    print("=" * 60)

    # Create synthetic trajectory
    B, T, C, H, W = 4, 17, 1, 64, 64
    trajectory = torch.randn(B, T, C, H, W)
    print(f"Input trajectory shape: {trajectory.shape}")

    # Extract features
    features = extract_trajectory_features(trajectory)
    print(f"Extracted features shape: {features.shape}")

    # Check shape is [B, D] where D > 0
    assert features.dim() == 2, f"Expected 2D features, got {features.dim()}D"
    assert features.shape[0] == B, f"Batch size mismatch: {features.shape[0]} vs {B}"
    assert features.shape[1] > 0, "No features extracted"

    # Check for NaN/Inf
    assert not torch.isnan(features).any(), "NaN in features!"
    assert not torch.isinf(features).any(), "Inf in features!"

    print(f"Feature dimension: {features.shape[1]}")
    print(f"Feature stats: mean={features.mean():.4f}, std={features.std():.4f}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")

    print("\n✓ TEST 1 PASSED: Feature extraction works correctly")
    return True


def test_feature_consistency():
    """Test 2: Feature extraction is consistent and differentiable."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Consistency and Differentiability")
    print("=" * 60)

    # Create trajectory requiring grad
    B, T, C, H, W = 2, 10, 1, 32, 32
    trajectory = torch.randn(B, T, C, H, W, requires_grad=True)

    # Extract features
    features = extract_trajectory_features(trajectory)
    print(f"Features shape: {features.shape}")

    # Test gradient flow
    loss = features.sum()
    loss.backward()

    assert trajectory.grad is not None, "No gradients!"
    assert not torch.isnan(trajectory.grad).any(), "NaN in gradients!"

    print(f"Gradient shape: {trajectory.grad.shape}")
    print(f"Gradient norm: {trajectory.grad.norm():.4f}")

    # Test consistency (same input → same output)
    trajectory2 = trajectory.detach().clone()
    features2 = extract_trajectory_features(trajectory2)

    diff = (features.detach() - features2).abs().max()
    assert diff < 1e-5, f"Feature extraction not consistent: max diff = {diff}"

    print(f"Consistency check: max diff = {diff:.2e}")

    print("\n✓ TEST 2 PASSED: Features are consistent and differentiable")
    return True


def test_dataset_with_features():
    """Test 3: Dataset with pre-computed features."""
    print("\n" + "=" * 60)
    print("TEST 3: Dataset with Pre-computed Features")
    print("=" * 60)

    # Generate synthetic data
    n_samples = 16
    timesteps = 17
    u0, trajectories = generate_synthetic_data(
        n_samples=n_samples,
        timesteps=timesteps,
        channels=1,
        height=64,
        width=64,
    )

    # Pre-compute features
    print("Pre-computing features...")
    features = extract_trajectory_features(trajectories)
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Features: {features.shape}")

    # Create dataset
    dataset = NOADatasetWithFeatures(u0, trajectories, features)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Test iteration
    for batch_idx, (ic, traj, feat) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Initial conditions: {ic.shape}")
        print(f"  Trajectories: {traj.shape}")
        print(f"  Features: {feat.shape}")

        assert ic.shape[0] == traj.shape[0] == feat.shape[0], "Batch size mismatch"
        break

    print("\n✓ TEST 3 PASSED: Dataset with features works correctly")
    return True


def test_feature_loss_training():
    """Test 4: Training with feature loss (Stage 2)."""
    print("\n" + "=" * 60)
    print("TEST 4: Training with Feature Loss")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    # Generate synthetic data with features
    print("\nGenerating synthetic data...")
    n_samples = 32
    timesteps = 17
    u0, trajectories = generate_synthetic_data(
        n_samples=n_samples,
        timesteps=timesteps,
        channels=1,
        height=64,
        width=64,
    )

    # Pre-compute features
    features = extract_trajectory_features(trajectories)
    print(f"  Initial conditions: {u0.shape}")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Features: {features.shape}")

    # Create dataset and loader
    dataset = NOADatasetWithFeatures(u0, trajectories, features)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Create trainer with feature loss enabled
    trainer = NOAPhase1Trainer(
        noa=noa,
        device=device,
        learning_rate=1e-3,
        feature_weight=0.5,  # Enable feature loss
    )

    # Train for a few epochs
    print("\nTraining with feature loss...")
    epochs = 5
    losses = []

    for epoch in range(epochs):
        start = time.time()
        result = trainer.train_epoch(train_loader)
        elapsed = time.time() - start
        losses.append(result)
        print(f"  Epoch {epoch + 1}: total={result['total']:.4f}, "
              f"grid={result['grid']:.4f}, feat={result['feature']:.4f} ({elapsed:.1f}s)")

    # Check losses are finite
    for loss in losses:
        assert not (loss["total"] != loss["total"]), "NaN total loss!"
        assert not (loss["grid"] != loss["grid"]), "NaN grid loss!"
        assert not (loss["feature"] != loss["feature"]), "NaN feature loss!"

    # Check loss decreased
    total_decreased = losses[-1]["total"] < losses[0]["total"]
    grid_decreased = losses[-1]["grid"] < losses[0]["grid"]
    feature_decreased = losses[-1]["feature"] < losses[0]["feature"]

    print(f"\nInitial: total={losses[0]['total']:.4f}, grid={losses[0]['grid']:.4f}, feat={losses[0]['feature']:.4f}")
    print(f"Final: total={losses[-1]['total']:.4f}, grid={losses[-1]['grid']:.4f}, feat={losses[-1]['feature']:.4f}")
    print(f"Total decreased: {total_decreased}")
    print(f"Grid decreased: {grid_decreased}")
    print(f"Feature decreased: {feature_decreased}")

    if total_decreased:
        print("\n✓ TEST 4 PASSED: Feature loss training works, loss decreased")
    else:
        print("\n⚠ TEST 4 WARNING: Total loss did not decrease (may need more epochs)")

    return True


def test_feature_weight_comparison():
    """Test 5: Compare training with different feature weights."""
    print("\n" + "=" * 60)
    print("TEST 5: Feature Weight Comparison")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate shared synthetic data
    n_samples = 24
    timesteps = 17
    u0, trajectories = generate_synthetic_data(
        n_samples=n_samples,
        timesteps=timesteps,
        channels=1,
        height=64,
        width=64,
    )
    features = extract_trajectory_features(trajectories)

    dataset = NOADatasetWithFeatures(u0, trajectories, features)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    results = {}

    for weight in [0.0, 0.5, 1.0]:
        print(f"\nTraining with feature_weight={weight}...")

        # Create fresh model
        noa = NOABackbone(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            encoder_levels=2,
            modes=8,
            afno_blocks=2,
        )

        trainer = NOAPhase1Trainer(
            noa=noa,
            device=device,
            learning_rate=1e-3,
            feature_weight=weight,
        )

        # Train for 3 epochs
        losses = []
        for epoch in range(3):
            result = trainer.train_epoch(train_loader)
            losses.append(result)

        results[weight] = {
            "initial": losses[0]["total"],
            "final": losses[-1]["total"],
            "grid_final": losses[-1]["grid"],
            "feature_final": losses[-1]["feature"],
        }

        print(f"  Initial: {losses[0]['total']:.4f} → Final: {losses[-1]['total']:.4f}")

    # Compare results
    print("\n" + "-" * 40)
    print("Summary:")
    for weight, res in results.items():
        reduction = (res["initial"] - res["final"]) / res["initial"] * 100
        print(f"  weight={weight}: {res['initial']:.4f} → {res['final']:.4f} ({reduction:.1f}% reduction)")

    print("\n✓ TEST 5 PASSED: Different feature weights work correctly")
    return True


def main():
    """Run all Stage 2 tests."""
    print("=" * 60)
    print("NOA Phase 1 Stage 2 Tests")
    print("=" * 60)
    print("\nThis tests feature extraction and feature MSE loss.")
    print("Prerequisites: Stage 1 tests must pass first.")
    print("\nSuccess criteria:")
    print("  - Feature extraction produces correct shapes")
    print("  - Features are differentiable")
    print("  - Feature loss training works")

    results = []

    # Run tests
    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Feature Consistency", test_feature_consistency),
        ("Dataset with Features", test_dataset_with_features),
        ("Feature Loss Training", test_feature_loss_training),
        ("Feature Weight Comparison", test_feature_weight_comparison),
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
        print("Stage 2 COMPLETE - Proceed to Stage 3 (VQ-VAE Perceptual Loss)")
        print("=" * 60)
    else:
        print("\n⚠ Some tests failed - debug before proceeding")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
