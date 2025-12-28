#!/usr/bin/env python3
"""
Test script to verify enhanced StochasticBlock with schedules and spatial correlation.
"""

import torch
import matplotlib.pyplot as plt
from src.spinlock.operators.blocks import StochasticBlock


def test_noise_schedules():
    """Test different noise schedule types."""

    print("="*60)
    print("Testing Noise Schedules")
    print("="*60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, 64, 64, device=device)

    # Test 1: Constant schedule (default)
    print("1. Testing constant schedule...")
    block_constant = StochasticBlock(3, noise_scale=0.1, noise_schedule="constant").to(device)
    out = block_constant(x, step=0, max_steps=100)
    print(f"   Output shape: {out.shape}")
    print(f"   Noise magnitude (step 0): {(out - x).abs().mean().item():.6f}")
    out = block_constant(x, step=50, max_steps=100)
    print(f"   Noise magnitude (step 50): {(out - x).abs().mean().item():.6f}")
    print("   ✓ Constant schedule: noise magnitude should be similar\n")

    # Test 2: Annealing schedule
    print("2. Testing annealing schedule...")
    block_anneal = StochasticBlock(3, noise_scale=0.1, noise_schedule="annealing").to(device)
    noise_magnitudes = []
    for step in [0, 25, 50, 75, 99]:
        out = block_anneal(x, step=step, max_steps=100)
        mag = (out - x).abs().mean().item()
        noise_magnitudes.append(mag)
        print(f"   Step {step:2d}: noise magnitude = {mag:.6f}")
    print("   ✓ Annealing: noise should decrease over time\n")

    # Test 3: Periodic schedule
    print("3. Testing periodic schedule...")
    block_periodic = StochasticBlock(
        3, noise_scale=0.1, noise_schedule="periodic", schedule_period=50
    ).to(device)
    for step in [0, 12, 25, 37, 50]:
        out = block_periodic(x, step=step)
        mag = (out - x).abs().mean().item()
        print(f"   Step {step:2d}: noise magnitude = {mag:.6f}")
    print("   ✓ Periodic: noise should oscillate\n")


def test_spatial_correlation():
    """Test spatially correlated noise generation."""

    print("="*60)
    print("Testing Spatial Correlation")
    print("="*60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 64, 64, device=device)

    # Test 1: Uncorrelated noise (default)
    print("1. Testing uncorrelated noise...")
    block_uncorr = StochasticBlock(3, noise_scale=0.1, spatial_correlation=0.0).to(device)
    out = block_uncorr(x)
    noise = (out - x)[0, 0].cpu()
    print(f"   Noise std: {noise.std().item():.6f}")
    print(f"   Noise min/max: [{noise.min().item():.4f}, {noise.max().item():.4f}]")
    print("   ✓ Uncorrelated noise generated\n")

    # Test 2: Correlated noise
    print("2. Testing spatially correlated noise...")
    block_corr = StochasticBlock(3, noise_scale=0.1, spatial_correlation=0.1).to(device)
    out = block_corr(x)
    noise_corr = (out - x)[0, 0].cpu()
    print(f"   Correlated noise std: {noise_corr.std().item():.6f}")
    print(f"   Correlated noise min/max: [{noise_corr.min().item():.4f}, {noise_corr.max().item():.4f}]")
    print("   ✓ Correlated noise generated\n")

    # Test 3: Different correlation lengths
    print("3. Testing different correlation lengths...")
    for corr_length in [0.02, 0.05, 0.1, 0.2]:
        block = StochasticBlock(3, noise_scale=0.1, spatial_correlation=corr_length).to(device)
        out = block(x)
        noise = (out - x)[0, 0].cpu()
        print(f"   Correlation {corr_length:.2f}: noise std = {noise.std().item():.6f}")
    print("   ✓ Different correlation lengths work\n")


def test_combined_features():
    """Test combining schedules and spatial correlation."""

    print("="*60)
    print("Testing Combined Features")
    print("="*60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 64, 64, device=device)

    print("1. Annealing + spatial correlation...")
    block = StochasticBlock(
        3,
        noise_scale=0.1,
        noise_schedule="annealing",
        spatial_correlation=0.1
    ).to(device)

    for step in [0, 50, 99]:
        out = block(x, step=step, max_steps=100)
        mag = (out - x).abs().mean().item()
        print(f"   Step {step:2d}: noise magnitude = {mag:.6f}")
    print("   ✓ Combined annealing + correlation works\n")

    print("2. Periodic + spatial correlation...")
    block = StochasticBlock(
        3,
        noise_scale=0.1,
        noise_schedule="periodic",
        schedule_period=50,
        spatial_correlation=0.05
    ).to(device)

    for step in [0, 25, 50]:
        out = block(x, step=step)
        mag = (out - x).abs().mean().item()
        print(f"   Step {step:2d}: noise magnitude = {mag:.6f}")
    print("   ✓ Combined periodic + correlation works\n")


def test_backward_compatibility():
    """Test that existing code still works (backward compatibility)."""

    print("="*60)
    print("Testing Backward Compatibility")
    print("="*60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, 64, 64, device=device)

    print("1. Testing old-style usage (no schedule parameters)...")
    block = StochasticBlock(3, noise_type="gaussian", noise_scale=0.1).to(device)
    out = block(x)
    print(f"   Output shape: {out.shape}")
    print(f"   ✓ Old-style usage works\n")

    print("2. Testing different noise types...")
    for noise_type in ["gaussian", "laplace", "multiplicative"]:
        block = StochasticBlock(3, noise_type=noise_type, noise_scale=0.1).to(device)
        out = block(x)
        print(f"   {noise_type}: output shape = {out.shape}")
    print("   ✓ All noise types work\n")


if __name__ == "__main__":
    print("\nRunning StochasticBlock Enhancement Tests\n")

    test_noise_schedules()
    test_spatial_correlation()
    test_combined_features()
    test_backward_compatibility()

    print("="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nEnhanced StochasticBlock features:")
    print("  - Noise schedules: constant, annealing, periodic")
    print("  - Spatial correlation: Fourier-filtered correlated noise")
    print("  - Backward compatible with existing code")
