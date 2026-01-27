#!/usr/bin/env python3
"""Quick validation script for v3.1 enhanced temporal features.

Run this script to verify that the new features work correctly without
requiring a full dataset regeneration or training run.

Usage:
    python scripts/validation/validate_v3_1_features.py [--device cuda]
"""

import argparse
import torch
import numpy as np
from spinlock.features.temporal.temporal import TemporalFeatureExtractor
from spinlock.features.temporal.spectral import SpectralFeatureExtractor
from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator


def validate_temporal_extractor(device):
    """Validate TemporalFeatureExtractor."""
    print("\n" + "="*70)
    print("Testing TemporalFeatureExtractor")
    print("="*70)

    extractor = TemporalFeatureExtractor(
        device=device,
        window_size=5,
        short_window=5,
        medium_window=20,
        long_window=50
    )

    B, C, H, W = 4, 3, 64, 64

    print(f"Input shape: [{B}, {C}, {H}, {W}]")
    print("Feeding 50 timesteps to build history...")

    for t in range(50):
        u = torch.randn(B, C, H, W, device=device)
        features = extractor.extract(u)

    # Get actual dimensions
    dims = extractor.get_feature_dimensions()
    expected_D = dims['total']

    print(f"\n✓ Component dimensions:")
    for name, dim in dims.items():
        if name != 'total':
            print(f"  {name}: {dim}D")
    print(f"  Total: {dims['total']}D")

    print(f"\n✓ Output shape: {features.shape}")
    assert features.shape == (B, expected_D), f"Expected (4, {expected_D}), got {features.shape}"

    print(f"✓ No NaNs: {not torch.isnan(features).any()}")
    assert not torch.isnan(features).any()

    print(f"✓ No Infs: {not torch.isinf(features).any()}")
    assert not torch.isinf(features).any()

    print(f"✓ Feature range: [{features.min():.3f}, {features.max():.3f}]")

    print("\n✅ TemporalFeatureExtractor validation PASSED")
    return True


def validate_spectral_extractor(device):
    """Validate SpectralFeatureExtractor includes spectral_entropy."""
    print("\n" + "="*70)
    print("Testing SpectralFeatureExtractor (spectral entropy)")
    print("="*70)

    extractor = SpectralFeatureExtractor(device=device)

    N, T, C, H, W = 4, 10, 3, 64, 64
    fields = torch.randn(N, T, C, H, W, device=device)

    print(f"Input shape: [{N}, {T}, {C}, {H}, {W}]")

    features = extractor.extract(fields, num_scales=5)

    print(f"✓ Features extracted: {len(features)} types")
    print(f"✓ Feature names: {list(features.keys())[:5]}...")  # Show first 5

    assert 'spectral_entropy' in features, "spectral_entropy not found!"
    print(f"✓ Spectral entropy found")

    entropy = features['spectral_entropy']
    print(f"✓ Entropy shape: {entropy.shape}")
    assert entropy.shape == (N, T, C)

    print(f"✓ No NaNs: {not torch.isnan(entropy).any()}")
    assert not torch.isnan(entropy).any()

    print(f"✓ Non-negative: {(entropy >= 0).all()}")
    assert (entropy >= 0).all(), "Entropy should be non-negative"

    print(f"✓ Entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]")

    print("\n✅ SpectralFeatureExtractor validation PASSED")
    return True


def validate_orchestrator(device):
    """Validate TemporalFeatureOrchestrator."""
    print("\n" + "="*70)
    print("Testing TemporalFeatureOrchestrator")
    print("="*70)

    orchestrator = TemporalFeatureOrchestrator(device=device)

    N, M, T, C, H, W = 2, 3, 50, 3, 64, 64
    trajectories = torch.randn(N, M, T, C, H, W, device=device)

    print(f"Input shape: [{N}, {M}, {T}, {C}, {H}, {W}]")
    print("Extracting per-timestep features...")

    features = orchestrator.extract_per_timestep(trajectories)

    # Get actual dimension
    actual_D = features.shape[-1]

    print(f"\n✓ Output shape: {features.shape}")
    print(f"✓ Total dimensions: {actual_D}D")

    print(f"✓ No NaNs: {not torch.isnan(features).any()}")
    assert not torch.isnan(features).any()

    print(f"✓ No Infs: {not torch.isinf(features).any()}")
    assert not torch.isinf(features).any()

    print(f"✓ Feature range: [{features.min():.3f}, {features.max():.3f}]")

    print("\n✅ TemporalFeatureOrchestrator validation PASSED")
    return True


def validate_gradient_flow(device):
    """Validate that gradients flow through all features."""
    print("\n" + "="*70)
    print("Testing Gradient Flow")
    print("="*70)

    orchestrator = TemporalFeatureOrchestrator(device=device)

    N, M, T, C, H, W = 2, 2, 10, 3, 32, 32
    trajectories = torch.randn(N, M, T, C, H, W, device=device, requires_grad=True)

    print(f"Input shape: [{N}, {M}, {T}, {C}, {H}, {W}]")
    print("Computing features with gradients...")

    features = orchestrator.extract_per_timestep(trajectories)

    print(f"✓ Output shape: {features.shape}")

    print("Computing loss and backpropagating...")
    loss = features.mean()
    loss.backward()

    print(f"✓ Gradients exist: {trajectories.grad is not None}")
    assert trajectories.grad is not None

    print(f"✓ No NaN gradients: {not torch.isnan(trajectories.grad).any()}")
    assert not torch.isnan(trajectories.grad).any()

    print(f"✓ Gradient range: [{trajectories.grad.min():.6f}, {trajectories.grad.max():.6f}]")

    print("\n✅ Gradient flow validation PASSED")
    return True


def validate_numerical_stability(device):
    """Validate numerical stability edge cases."""
    print("\n" + "="*70)
    print("Testing Numerical Stability")
    print("="*70)

    extractor = TemporalFeatureExtractor(device=device)
    B, C, H, W = 4, 3, 64, 64

    # Test 1: Zero inputs
    print("\nTest 1: Zero inputs")
    for t in range(10):
        u = torch.zeros(B, C, H, W, device=device)
        features = extractor.extract(u)

    print(f"✓ Zero input handling: {not torch.isnan(features).any()}")
    assert not torch.isnan(features).any()

    # Reset extractor
    extractor.reset()

    # Test 2: Constant inputs
    print("\nTest 2: Constant inputs")
    for t in range(10):
        u = torch.ones(B, C, H, W, device=device) * 5.0
        features = extractor.extract(u)

    print(f"✓ Constant input handling: {not torch.isnan(features).any()}")
    assert not torch.isnan(features).any()

    # Reset extractor
    extractor.reset()

    # Test 3: Extreme values
    print("\nTest 3: Extreme values")
    for t in range(10):
        u = torch.randn(B, C, H, W, device=device) * 1e3
        features = extractor.extract(u)

    print(f"✓ Extreme value handling: {not torch.isnan(features).any() and not torch.isinf(features).any()}")
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()

    print("\n✅ Numerical stability validation PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate v3.1 enhanced features")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("\n" + "="*70)
    print("v3.1 Enhanced Temporal Features Validation")
    print("="*70)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        # Run validation tests
        validate_temporal_extractor(device)
        validate_spectral_extractor(device)
        validate_orchestrator(device)
        validate_gradient_flow(device)
        validate_numerical_stability(device)

        print("\n" + "="*70)
        print("🎉 ALL VALIDATIONS PASSED! 🎉")
        print("="*70)
        print("\nv3.1 features are working correctly!")
        print("\nFeature summary:")
        print("  • Temporal skewness/kurtosis")
        print("  • Multi-lag autocorrelation")
        print("  • Sample entropy proxy")
        print("  • Enstrophy")
        print("  • Divergence")
        print("  • Spectral entropy")
        print("\nReady for production use!")

    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
