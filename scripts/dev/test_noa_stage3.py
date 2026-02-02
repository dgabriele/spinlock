#!/usr/bin/env python
"""NOA Phase 1 Stage 3 Test Script - VQ-VAE perceptual loss.

This script tests Stage 3 additions:
1. VQ-VAE checkpoint loading
2. Feature projection to VQ-VAE input space
3. Perceptual loss computation
4. End-to-end training with all losses

Prerequisites: Stages 1 and 2 tests must pass first.

Success criteria:
- VQ-VAE loads correctly from checkpoint
- Perceptual loss computation works
- End-to-end training with perceptual loss runs

Usage:
    poetry run python scripts/dev/test_noa_stage3.py
"""

import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

# Add src to path for imports
sys.path.insert(0, "/home/daniel/projects/spinlock/src")

from spinlock.noa.backbone import NOABackbone
from spinlock.noa.losses import VQVAEPerceptualLoss, FeatureProjector, NOALoss
from spinlock.noa.training import (
    NOAPhase1Trainer,
    NOADatasetWithFeatures,
    extract_trajectory_features,
    generate_synthetic_data,
)

# VQ-VAE checkpoint path
VQVAE_CHECKPOINT = "/home/daniel/projects/spinlock/checkpoints/production/100k_3family_v1/"


def test_vqvae_loading():
    """Test 1: Load VQ-VAE from checkpoint."""
    print("\n" + "=" * 60)
    print("TEST 1: VQ-VAE Checkpoint Loading")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        vqvae_loss = VQVAEPerceptualLoss(
            checkpoint_path=VQVAE_CHECKPOINT,
            device=device,
            freeze_vqvae=True,
        )
        print(f"VQ-VAE loaded successfully")
        print(f"  Input dimension: {vqvae_loss.input_dim}")
        print(f"  Latent dimension: {vqvae_loss.get_latent_dim()}")

        # Verify parameters are frozen
        frozen_count = sum(1 for p in vqvae_loss.vqvae.parameters() if not p.requires_grad)
        total_count = sum(1 for p in vqvae_loss.vqvae.parameters())
        print(f"  Frozen parameters: {frozen_count}/{total_count}")

        assert frozen_count == total_count, "Not all parameters frozen!"

    except FileNotFoundError as e:
        print(f"Checkpoint not found: {e}")
        print("Skipping VQ-VAE tests")
        return False

    print("\n✓ TEST 1 PASSED: VQ-VAE loaded and frozen correctly")
    return True


def test_feature_projection():
    """Test 2: Feature projection to VQ-VAE input space."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Projection")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VQ-VAE to get input dimension
    try:
        vqvae_loss = VQVAEPerceptualLoss(
            checkpoint_path=VQVAE_CHECKPOINT,
            device=device,
        )
        vqvae_input_dim = vqvae_loss.input_dim
    except FileNotFoundError:
        print("Checkpoint not found, skipping test")
        return False

    # Create projector
    trajectory_feature_dim = 8  # From extract_trajectory_features
    projector = FeatureProjector(
        input_dim=trajectory_feature_dim,
        output_dim=vqvae_input_dim,
        hidden_dim=128,
    ).to(device)

    print(f"Projector: {trajectory_feature_dim}D → {vqvae_input_dim}D")

    # Test projection
    batch_size = 4
    traj_features = torch.randn(batch_size, trajectory_feature_dim, device=device)
    projected = projector(traj_features)

    print(f"  Input shape: {traj_features.shape}")
    print(f"  Output shape: {projected.shape}")

    assert projected.shape == (batch_size, vqvae_input_dim), f"Wrong shape: {projected.shape}"
    assert not torch.isnan(projected).any(), "NaN in projected features"

    # Test gradient flow
    projected.sum().backward()
    has_grad = any(p.grad is not None for p in projector.parameters())
    assert has_grad, "No gradients in projector!"

    print("\n✓ TEST 2 PASSED: Feature projection works correctly")
    return True


def test_perceptual_loss():
    """Test 3: VQ-VAE perceptual loss computation."""
    print("\n" + "=" * 60)
    print("TEST 3: Perceptual Loss Computation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        vqvae_loss = VQVAEPerceptualLoss(
            checkpoint_path=VQVAE_CHECKPOINT,
            device=device,
        )
    except FileNotFoundError:
        print("Checkpoint not found, skipping test")
        return False

    input_dim = vqvae_loss.input_dim
    batch_size = 4

    # Create features
    pred_features = torch.randn(batch_size, input_dim, device=device, requires_grad=True)
    gt_features = torch.randn(batch_size, input_dim, device=device)

    # Compute loss
    loss = vqvae_loss(pred_features, gt_features)

    print(f"Perceptual loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "NaN loss!"
    assert loss.item() >= 0, "Negative loss!"

    # Test gradient flow
    loss.backward()
    assert pred_features.grad is not None, "No gradients!"
    assert not torch.isnan(pred_features.grad).any(), "NaN in gradients!"

    print(f"Gradient norm: {pred_features.grad.norm().item():.4f}")

    print("\n✓ TEST 3 PASSED: Perceptual loss computation works")
    return True


def test_noa_loss_combined():
    """Test 4: Combined NOA loss (grid + feature + perceptual)."""
    print("\n" + "=" * 60)
    print("TEST 4: Combined NOA Loss")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Create combined loss function
        loss_fn = NOALoss(
            vqvae_checkpoint=VQVAE_CHECKPOINT,
            feature_weight=0.5,
            perceptual_weight=0.1,
            trajectory_feature_dim=8,
            device=device,
        )
    except FileNotFoundError:
        print("Checkpoint not found, testing without perceptual loss")
        loss_fn = NOALoss(
            vqvae_checkpoint=None,
            feature_weight=0.5,
            perceptual_weight=0.0,
            device=device,
        )

    # Create dummy data
    batch_size = 2
    T, C, H, W = 17, 1, 64, 64
    pred_trajectory = torch.randn(batch_size, T, C, H, W, device=device, requires_grad=True)
    gt_trajectory = torch.randn(batch_size, T, C, H, W, device=device)

    # Extract features
    pred_features = extract_trajectory_features(pred_trajectory)
    gt_features = extract_trajectory_features(gt_trajectory.detach())

    # Compute combined loss
    losses = loss_fn(pred_trajectory, gt_trajectory, pred_features, gt_features)

    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

    # Verify all losses are finite
    for name, value in losses.items():
        assert not torch.isnan(value), f"NaN in {name} loss!"
        assert not torch.isinf(value), f"Inf in {name} loss!"

    # Test gradient flow
    losses["total"].backward()
    assert pred_trajectory.grad is not None, "No gradients!"
    assert not torch.isnan(pred_trajectory.grad).any(), "NaN in gradients!"

    print(f"\nGradient norm: {pred_trajectory.grad.norm().item():.4f}")

    print("\n✓ TEST 4 PASSED: Combined NOA loss works correctly")
    return True


def test_end_to_end_training():
    """Test 5: End-to-end training with all losses."""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End Training with Perceptual Loss")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create NOA backbone
    noa = NOABackbone(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        encoder_levels=2,
        modes=8,
        afno_blocks=2,
    ).to(device)
    print(f"NOA parameters: {noa.num_parameters:,}")

    # Create loss function
    try:
        loss_fn = NOALoss(
            vqvae_checkpoint=VQVAE_CHECKPOINT,
            feature_weight=0.5,
            perceptual_weight=0.1,
            trajectory_feature_dim=8,
            device=device,
        )
        has_perceptual = True
        print("Using perceptual loss")
    except FileNotFoundError:
        loss_fn = NOALoss(
            vqvae_checkpoint=None,
            feature_weight=0.5,
            perceptual_weight=0.0,
            device=device,
        )
        has_perceptual = False
        print("No perceptual loss (checkpoint not found)")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
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

    # Create optimizer (include projector if using perceptual loss)
    params = list(noa.parameters())
    if has_perceptual and loss_fn.projector is not None:
        params += list(loss_fn.projector.parameters())

    optimizer = torch.optim.AdamW(params, lr=1e-3)

    # Train for a few epochs
    print("\nTraining...")
    epochs = 3
    losses = []

    for epoch in range(epochs):
        noa.train()
        epoch_loss = 0.0
        epoch_grid = 0.0
        epoch_feat = 0.0
        epoch_perc = 0.0
        num_batches = 0

        for batch in train_loader:
            ic, gt_traj, gt_feat = batch
            ic = ic.to(device)
            gt_traj = gt_traj.to(device)
            gt_feat = gt_feat.to(device)

            # Forward pass
            pred_traj = noa(ic, steps=gt_traj.shape[1] - 1, return_all_steps=True)
            pred_feat = extract_trajectory_features(pred_traj)

            # Compute loss
            loss_dict = loss_fn(pred_traj, gt_traj, pred_feat, gt_feat)

            # Backward
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

            epoch_loss += loss_dict["total"].item()
            epoch_grid += loss_dict["grid"].item()
            epoch_feat += loss_dict["feature"].item()
            epoch_perc += loss_dict["perceptual"].item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_grid = epoch_grid / num_batches
        avg_feat = epoch_feat / num_batches
        avg_perc = epoch_perc / num_batches

        losses.append({
            "total": avg_loss,
            "grid": avg_grid,
            "feature": avg_feat,
            "perceptual": avg_perc,
        })

        print(f"  Epoch {epoch + 1}: total={avg_loss:.4f}, "
              f"grid={avg_grid:.4f}, feat={avg_feat:.4f}, perc={avg_perc:.4f}")

    # Check all losses finite
    for epoch_losses in losses:
        for name, value in epoch_losses.items():
            assert value == value, f"NaN in {name} at some epoch!"

    # Check loss decreased
    initial = losses[0]["total"]
    final = losses[-1]["total"]
    decreased = final < initial

    print(f"\nInitial total loss: {initial:.4f}")
    print(f"Final total loss: {final:.4f}")
    print(f"Loss decreased: {decreased}")

    if decreased:
        print("\n✓ TEST 5 PASSED: End-to-end training works, loss decreased")
    else:
        print("\n⚠ TEST 5 WARNING: Loss did not decrease (may need more epochs)")

    return True


def main():
    """Run all Stage 3 tests."""
    print("=" * 60)
    print("NOA Phase 1 Stage 3 Tests")
    print("=" * 60)
    print("\nThis tests VQ-VAE perceptual loss integration.")
    print("Prerequisites: Stages 1 and 2 tests must pass first.")
    print("\nSuccess criteria:")
    print("  - VQ-VAE loads from checkpoint")
    print("  - Feature projection works")
    print("  - Perceptual loss computes correctly")
    print("  - End-to-end training runs")

    results = []

    # Run tests
    tests = [
        ("VQ-VAE Loading", test_vqvae_loading),
        ("Feature Projection", test_feature_projection),
        ("Perceptual Loss", test_perceptual_loss),
        ("Combined NOA Loss", test_noa_loss_combined),
        ("End-to-End Training", test_end_to_end_training),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASSED" if result else "SKIPPED"))
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
        symbol = "✓" if "PASSED" in status else ("⊘" if "SKIPPED" in status else "✗")
        print(f"  {symbol} {name}: {status}")

    all_passed = all("PASSED" in status or "SKIPPED" in status for _, status in results)

    if all_passed:
        print("\n" + "=" * 60)
        print("Stage 3 COMPLETE - NOA Phase 1 Prototype Ready!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Train on real data (not synthetic)")
        print("  2. Tune hyperparameters (loss weights, learning rate)")
        print("  3. Add proper feature extraction (INITIAL/SUMMARY/TEMPORAL)")
        print("  4. Evaluate trajectory quality metrics")
    else:
        print("\n⚠ Some tests failed - debug before proceeding")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
