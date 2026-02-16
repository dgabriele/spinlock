#!/usr/bin/env python3
"""
Validation script for enriched feature extraction (v3.2).

Tests:
1. Parameter-sensitive features (Phase 1)
2. Orthogonal spatial features (Phase 2)
3. Orthogonal spectral features (Phase 2)
4. Orthogonal temporal complexity features (Phase 2)
5. Feature name generation
6. Backward compatibility (parameters=None should work)
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add spinlock to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator


def test_parameter_features():
    """Test Phase 1: Parameter-sensitive features."""
    print("\n" + "="*80)
    print("TEST 1: Parameter-Sensitive Features (+35D)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orchestrator = TemporalFeatureOrchestrator(device=device, use_batch_mode=True)

    # Create test data
    N, M, T, C, H, W = 10, 3, 50, 2, 64, 64
    param_dim = 14

    trajectories = torch.randn(N, M, T, C, H, W, device=device)
    parameters = torch.randn(N, param_dim, device=device)
    sensitivities = torch.randn(N, param_dim, device=device)

    # Extract with parameters
    features = orchestrator.extract_per_timestep(
        trajectories,
        parameters=parameters,
        parameter_sensitivities=sensitivities
    )

    print(f"✓ Input shape: {trajectories.shape}")
    print(f"✓ Parameters shape: {parameters.shape}")
    print(f"✓ Output features shape: {features.shape}")

    # Check feature variance
    variance = features.var(dim=(0, 1))
    zero_var_count = (variance < 1e-6).sum().item()

    print(f"✓ Features with zero variance: {zero_var_count}/{features.shape[-1]}")
    print(f"✓ Mean feature variance: {variance.mean().item():.6f}")

    # Extract parameter features specifically
    param_encoding = orchestrator._extract_parameter_encoding(parameters, T)
    regime_indicators = orchestrator._extract_regime_indicators(
        trajectories.mean(dim=1),  # [N, T, C, H, W]
        parameters
    )
    param_gradients = orchestrator._extract_parameter_gradients(sensitivities, T)

    print(f"✓ Parameter encoding: {param_encoding.shape} (expected: [{N}, {T}, {param_dim}])")
    print(f"✓ Regime indicators: {regime_indicators.shape} (expected: [{N}, {T}, 6])")
    print(f"✓ Parameter gradients: {param_gradients.shape} (expected: [{N}, {T}, {param_dim}])")

    # Verify parameter features have non-zero variance
    param_var = param_encoding.var(dim=(0, 1)).mean()
    print(f"✓ Parameter feature variance: {param_var.item():.6f} (should be > 0)")

    assert param_var > 1e-8, "Parameter features should have non-zero variance!"

    print("\n✅ PHASE 1 TESTS PASSED")
    return features.shape[-1]


def test_orthogonal_spatial_features():
    """Test Phase 2: Orthogonal spatial features."""
    print("\n" + "="*80)
    print("TEST 2: Orthogonal Spatial Features (+15D)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from spinlock.features.temporal.spatial import SpatialFeatureExtractor
    extractor = SpatialFeatureExtractor(device=device)

    # Create test data
    NT, C, H, W = 100, 2, 64, 64
    fields_flat = torch.randn(NT, C, H, W, device=device)

    # Extract orthogonal features
    orthogonal = extractor._extract_orthogonal_spatial(fields_flat)

    print(f"✓ Input shape: {fields_flat.shape}")
    print(f"✓ Orthogonal spatial features:")
    for name, feat in sorted(orthogonal.items()):
        print(f"    {name}: {feat.shape}")

    total_features = sum(f.shape[-1] for f in orthogonal.values())
    print(f"✓ Total orthogonal spatial features: {total_features}D")

    # Check feature names
    expected_features = [
        'curl_mean', 'curl_std', 'curl_max',
        'hessian_lambda1_mean', 'hessian_lambda1_std',
        'hessian_lambda2_mean', 'hessian_lambda2_std',
        'hessian_anisotropy_mean', 'hessian_anisotropy_std',
        'structure_coherence_mean', 'structure_coherence_std',
        'structure_orientation_mean', 'structure_orientation_std',
        'structure_strength_mean', 'structure_strength_std',
    ]

    for name in expected_features:
        assert name in orthogonal, f"Missing feature: {name}"

    print(f"✓ All {len(expected_features)} expected features present")
    print("\n✅ ORTHOGONAL SPATIAL TESTS PASSED")
    return total_features


def test_orthogonal_spectral_features():
    """Test Phase 2: Orthogonal spectral (wavelet) features."""
    print("\n" + "="*80)
    print("TEST 3: Orthogonal Spectral Features (+15D)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from spinlock.features.temporal.spectral import SpectralFeatureExtractor
    extractor = SpectralFeatureExtractor(device=device)

    # Create test data
    NT, C, H, W = 100, 2, 64, 64
    fields_flat = torch.randn(NT, C, H, W, device=device)

    # Extract wavelet features
    wavelet = extractor._extract_wavelet_features(fields_flat)

    print(f"✓ Input shape: {fields_flat.shape}")
    print(f"✓ Wavelet features:")
    for name, feat in sorted(wavelet.items()):
        print(f"    {name}: {feat.shape}")

    total_features = sum(f.shape[-1] for f in wavelet.values())
    print(f"✓ Total wavelet features: {total_features}D")

    # Check feature names
    expected_features = [
        'wavelet_approx_mean', 'wavelet_approx_std', 'wavelet_approx_energy',
        'wavelet_horizontal_mean', 'wavelet_horizontal_std', 'wavelet_horizontal_energy',
        'wavelet_vertical_mean', 'wavelet_vertical_std', 'wavelet_vertical_energy',
        'wavelet_diagonal_mean', 'wavelet_diagonal_std', 'wavelet_diagonal_energy',
        'wavelet_entropy_horizontal', 'wavelet_entropy_vertical', 'wavelet_entropy_diagonal',
    ]

    for name in expected_features:
        assert name in wavelet, f"Missing feature: {name}"

    print(f"✓ All {len(expected_features)} expected features present")
    print("\n✅ ORTHOGONAL SPECTRAL TESTS PASSED")
    return total_features


def test_orthogonal_temporal_features():
    """Test Phase 2: Orthogonal temporal (complexity) features."""
    print("\n" + "="*80)
    print("TEST 4: Orthogonal Temporal Features (+10D)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from spinlock.features.temporal.temporal_batch import BatchParallelTemporalExtractor
    extractor = BatchParallelTemporalExtractor(device=device)

    # Create test data
    N, T, C, H, W = 10, 50, 2, 64, 64
    fields = torch.randn(N, T, C, H, W, device=device)

    # Extract complexity features
    complexity = extractor._extract_complexity_features_batch(fields)

    print(f"✓ Input shape: {fields.shape}")
    print(f"✓ Complexity features shape: {complexity.shape} (expected: [{N}, {T}, 10])")

    # Verify shape
    assert complexity.shape == (N, T, 10), f"Expected shape ({N}, {T}, 10), got {complexity.shape}"

    # Check variance (should be non-zero for most features)
    variance = complexity.var(dim=(0, 1))
    non_zero_var = (variance > 1e-8).sum().item()
    print(f"✓ Features with non-zero variance: {non_zero_var}/10")

    print("\n✅ ORTHOGONAL TEMPORAL TESTS PASSED")
    return complexity.shape[-1]


def test_feature_names():
    """Test feature name generation with parameters."""
    print("\n" + "="*80)
    print("TEST 5: Feature Name Generation")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orchestrator = TemporalFeatureOrchestrator(device=device, use_batch_mode=True)

    # Test without parameters
    names_no_params = orchestrator.get_all_feature_names(num_channels=2)
    print(f"✓ Feature names without parameters: {len(names_no_params)}")

    # Test with parameters
    param_dim = 14
    names_with_params = orchestrator.get_all_feature_names(
        num_channels=2,
        param_dim=param_dim,
        include_param_gradients=True
    )
    print(f"✓ Feature names with parameters: {len(names_with_params)}")

    # Check parameter feature names
    param_names = orchestrator.get_parameter_feature_names(
        param_dim=param_dim,
        include_gradients=True
    )
    print(f"✓ Parameter feature names: {len(param_names)}")
    print(f"    First 5: {param_names[:5]}")
    print(f"    Last 5: {param_names[-5:]}")

    expected_param_features = param_dim + 6 + param_dim  # encoding + regime + gradients
    assert len(param_names) == expected_param_features, \
        f"Expected {expected_param_features} param features, got {len(param_names)}"

    print(f"✓ Expected {expected_param_features} parameter features, got {len(param_names)}")
    print("\n✅ FEATURE NAME TESTS PASSED")


def test_backward_compatibility():
    """Test backward compatibility (parameters=None should work)."""
    print("\n" + "="*80)
    print("TEST 6: Backward Compatibility")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orchestrator = TemporalFeatureOrchestrator(device=device, use_batch_mode=True)

    # Create test data WITHOUT parameters
    N, M, T, C, H, W = 10, 3, 50, 2, 64, 64
    trajectories = torch.randn(N, M, T, C, H, W, device=device)

    # Extract without parameters (old API)
    features_old = orchestrator.extract_per_timestep(trajectories)

    # Extract with parameters=None (new API)
    features_new = orchestrator.extract_per_timestep(trajectories, parameters=None)

    print(f"✓ Old API shape: {features_old.shape}")
    print(f"✓ New API shape (parameters=None): {features_new.shape}")

    # Shapes should match
    assert features_old.shape == features_new.shape, \
        f"Backward compatibility broken! Old: {features_old.shape}, New: {features_new.shape}"

    # Values should be close (allowing for numerical differences)
    max_diff = (features_old - features_new).abs().max().item()
    print(f"✓ Max difference: {max_diff:.10f}")

    assert max_diff < 1e-5, f"Features differ too much: {max_diff}"

    print("\n✅ BACKWARD COMPATIBILITY TESTS PASSED")


def test_integration():
    """Test full pipeline with all features enabled."""
    print("\n" + "="*80)
    print("TEST 7: Full Integration Test")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orchestrator = TemporalFeatureOrchestrator(device=device, use_batch_mode=True)

    # Create test data
    N, M, T, C, H, W = 10, 3, 50, 2, 64, 64
    param_dim = 14

    trajectories = torch.randn(N, M, T, C, H, W, device=device)
    parameters = torch.randn(N, param_dim, device=device)
    sensitivities = torch.randn(N, param_dim, device=device)

    # Extract all features
    features_full = orchestrator.extract_per_timestep(
        trajectories,
        parameters=parameters,
        parameter_sensitivities=sensitivities
    )

    features_no_grad = orchestrator.extract_per_timestep(
        trajectories,
        parameters=parameters,
        parameter_sensitivities=None  # No gradients
    )

    features_no_params = orchestrator.extract_per_timestep(
        trajectories,
        parameters=None
    )

    print(f"✓ Full features (params + gradients): {features_full.shape[-1]}D")
    print(f"✓ Features (params, no gradients): {features_no_grad.shape[-1]}D")
    print(f"✓ Features (no params): {features_no_params.shape[-1]}D")

    # Expected differences
    expected_diff_grad = param_dim  # 14D parameter gradients
    expected_diff_params = param_dim + 6 + param_dim  # 14D encoding + 6D regime + 14D gradients

    diff_grad = features_full.shape[-1] - features_no_grad.shape[-1]
    diff_params = features_full.shape[-1] - features_no_params.shape[-1]

    print(f"✓ Dimension increase with gradients: {diff_grad}D (expected: {expected_diff_grad}D)")
    print(f"✓ Dimension increase with params: {diff_params}D (expected: {expected_diff_params}D)")

    assert diff_grad == expected_diff_grad, \
        f"Gradient dimension mismatch: {diff_grad} != {expected_diff_grad}"
    assert diff_params == expected_diff_params, \
        f"Parameter dimension mismatch: {diff_params} != {expected_diff_params}"

    print("\n✅ INTEGRATION TESTS PASSED")


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("ENRICHED FEATURE EXTRACTION VALIDATION (v3.2)")
    print("="*80)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    try:
        # Phase 1 tests
        dim_param = test_parameter_features()

        # Phase 2 tests
        dim_spatial = test_orthogonal_spatial_features()
        dim_spectral = test_orthogonal_spectral_features()
        dim_temporal = test_orthogonal_temporal_features()

        # Other tests
        test_feature_names()
        test_backward_compatibility()
        test_integration()

        # Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"✅ Phase 1 (Parameter-sensitive): +{14 + 6 + 14}D")
        print(f"✅ Phase 2 (Orthogonal spatial): +{dim_spatial}D")
        print(f"✅ Phase 2 (Orthogonal spectral): +{dim_spectral}D")
        print(f"✅ Phase 2 (Orthogonal temporal): +{dim_temporal}D")
        print(f"✅ Total enrichment: +{34 + dim_spatial + dim_spectral + dim_temporal}D")
        print(f"✅ Feature name generation: PASSED")
        print(f"✅ Backward compatibility: PASSED")
        print(f"✅ Integration: PASSED")
        print("\n" + "="*80)
        print("ALL VALIDATION TESTS PASSED ✅")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
