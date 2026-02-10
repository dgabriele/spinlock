"""Numerical validation tests for quantum features."""

import pytest
import torch
import numpy as np
from spinlock.features.quantum import QuantumFeatureExtractor, QuantumConfig


@pytest.fixture
def device():
    """Get compute device (prefer CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_normalized_gaussian_wavepacket(
    N: int, T: int, H: int, W: int, device: torch.device
) -> torch.Tensor:
    """Create normalized Gaussian wavepacket.

    Args:
        N: Number of samples
        T: Number of timesteps
        H: Grid height
        W: Grid width
        device: Compute device

    Returns:
        [N, T, 2, H, W] normalized wavefunction (Re, Im channels)
    """
    # Create random wavepacket
    psi = torch.randn(N, T, 2, H, W, device=device)

    # Normalize: ∫|ψ|² dx = 1
    norm = torch.sqrt((psi ** 2).sum(dim=(2, 3, 4), keepdim=True))
    psi = psi / (norm + 1e-10)

    return psi


def test_purity_pure_state(device):
    """Purity should be ≈1 for normalized pure states."""
    config = QuantumConfig(
        include_purity=True,
        include_entropy=False,
        include_coherence=False,
        include_variance=False,
        include_decoherence=False,
    )
    extractor = QuantumFeatureExtractor(device=device, config=config, grid_size=64)

    # Create normalized Gaussian wavepacket
    N, T, H, W = 4, 10, 64, 64
    psi = create_normalized_gaussian_wavepacket(N, T, H, W, device)

    features = extractor.extract(psi)

    # Purity is first feature (index 0)
    purity = features[:, :, 0]

    # Check that purity ≈ 1 (pure state)
    assert torch.all(purity > 0.99), (
        f"Purity should be ≈1 for pure states. "
        f"Got: min={purity.min():.6f}, mean={purity.mean():.6f}, max={purity.max():.6f}"
    )

    # Check that purity ≤ 1 (physical constraint)
    assert torch.all(purity <= 1.01), (
        f"Purity must be ≤1. Got max={purity.max():.6f}"
    )

    print(f"✓ Purity test passed: mean={purity.mean():.6f}, std={purity.std():.6f}")


def test_linear_entropy_pure_state(device):
    """Linear entropy should be ≈0 for pure states."""
    config = QuantumConfig(
        include_purity=True,
        include_entropy=False,
        include_coherence=False,
        include_variance=False,
    )
    extractor = QuantumFeatureExtractor(device=device, config=config, grid_size=64)

    N, T, H, W = 4, 10, 64, 64
    psi = create_normalized_gaussian_wavepacket(N, T, H, W, device)

    features = extractor.extract(psi)

    # Linear entropy is second feature (index 1)
    lin_ent = features[:, :, 1]

    # For pure states (purity ≈ 1), linear entropy ≈ 0
    assert torch.all(lin_ent < 0.05), (
        f"Linear entropy should be ≈0 for pure states. "
        f"Got: min={lin_ent.min():.6f}, mean={lin_ent.mean():.6f}, max={lin_ent.max():.6f}"
    )

    # Non-negative
    assert torch.all(lin_ent >= -1e-6), (
        f"Linear entropy must be non-negative. Got min={lin_ent.min():.6f}"
    )

    print(f"✓ Linear entropy test passed: mean={lin_ent.mean():.6f}, std={lin_ent.std():.6f}")


def test_uncertainty_principle(device):
    """Check Δx·Δp ≥ ℏ/2."""
    config = QuantumConfig(
        include_purity=False,
        include_entropy=False,
        include_coherence=False,
        include_variance=True,
        hbar=1.0,
    )
    extractor = QuantumFeatureExtractor(device=device, config=config, grid_size=64)

    # Generate test wavefunction
    N, T = 4, 10
    psi = create_normalized_gaussian_wavepacket(N, T, 64, 64, device)

    features = extractor.extract(psi)

    # Extract uncertainty products
    # Feature order: [Δx, Δy, Δpx, Δpy, prod_x, prod_y]
    prod_x = features[:, :, -2]  # Δx·Δpx
    prod_y = features[:, :, -1]  # Δy·Δpy

    hbar_over_2 = config.hbar / 2

    # Allow 10% tolerance due to grid discretization
    min_threshold = hbar_over_2 * 0.9

    assert torch.all(prod_x >= min_threshold), (
        f"Uncertainty principle violated: min(Δx·Δpx) = {prod_x.min():.3f} < 0.9·ℏ/2 = {min_threshold:.3f}"
    )

    assert torch.all(prod_y >= min_threshold), (
        f"Uncertainty principle violated: min(Δy·Δpy) = {prod_y.min():.3f} < 0.9·ℏ/2 = {min_threshold:.3f}"
    )

    print(
        f"✓ Uncertainty principle test passed:\n"
        f"  Δx·Δpx: min={prod_x.min():.3f}, mean={prod_x.mean():.3f}, ℏ/2={hbar_over_2:.3f}\n"
        f"  Δy·Δpy: min={prod_y.min():.3f}, mean={prod_y.mean():.3f}, ℏ/2={hbar_over_2:.3f}"
    )


def test_entropy_bounds(device):
    """Entropy should be ≥0 and bounded."""
    config = QuantumConfig(
        include_purity=False,
        include_entropy=True,
        include_coherence=False,
        include_variance=False,
    )
    extractor = QuantumFeatureExtractor(device=device, config=config, grid_size=64)

    N, T = 4, 10
    psi = create_normalized_gaussian_wavepacket(N, T, 64, 64, device)

    features = extractor.extract(psi)

    # von Neumann entropy (index 0 since only entropy enabled)
    vn_ent = features[:, :, 0]

    # Non-negative
    assert torch.all(vn_ent >= 0), (
        f"Entropy must be non-negative. Got min={vn_ent.min():.6f}"
    )

    # Bounded by log(dimension) = log(64*64) ≈ 8.5
    max_entropy = np.log(64 * 64)
    assert torch.all(vn_ent <= max_entropy), (
        f"Entropy exceeds maximum log(D)={max_entropy:.2f}. Got max={vn_ent.max():.2f}"
    )

    print(
        f"✓ Entropy bounds test passed: "
        f"min={vn_ent.min():.3f}, mean={vn_ent.mean():.3f}, max={vn_ent.max():.3f}, "
        f"max_allowed={max_entropy:.3f}"
    )


def test_coherence_positive(device):
    """Coherence should be non-negative."""
    config = QuantumConfig(
        include_purity=False,
        include_entropy=False,
        include_coherence=True,
        include_variance=False,
    )
    extractor = QuantumFeatureExtractor(device=device, config=config, grid_size=64)

    N, T = 4, 10
    psi = create_normalized_gaussian_wavepacket(N, T, 64, 64, device)

    features = extractor.extract(psi)

    # Coherence (index 0 since only coherence enabled)
    coh = features[:, :, 0]

    # Non-negative
    assert torch.all(coh >= 0), (
        f"Coherence must be non-negative. Got min={coh.min():.6f}"
    )

    # Should be positive for coherent states
    assert torch.all(coh > 0), (
        f"Coherence should be positive for coherent states. Got min={coh.min():.6f}"
    )

    print(f"✓ Coherence test passed: min={coh.min():.3f}, mean={coh.mean():.3f}, max={coh.max():.3f}")


def test_all_features_enabled(device):
    """Test with all features enabled."""
    config = QuantumConfig(
        include_purity=True,
        include_entropy=True,
        include_coherence=True,
        include_variance=True,
        include_decoherence=False,  # Skip for now (requires time series)
    )
    extractor = QuantumFeatureExtractor(device=device, config=config, grid_size=64)

    N, T = 4, 10
    psi = create_normalized_gaussian_wavepacket(N, T, 64, 64, device)

    features = extractor.extract(psi)

    # Expected features: purity(1) + lin_ent(1) + vn_ent(1) + coh(1) + variance(6) = 10
    expected_dims = 10
    assert features.shape == (N, T, expected_dims), (
        f"Expected shape ({N}, {T}, {expected_dims}), got {features.shape}"
    )

    # Check no NaN/Inf
    assert not torch.isnan(features).any(), "Features contain NaN"
    assert not torch.isinf(features).any(), "Features contain Inf"

    print(
        f"✓ All features test passed: shape={features.shape}\n"
        f"  Feature stats: min={features.min():.3f}, mean={features.mean():.3f}, max={features.max():.3f}"
    )


def test_decoherence_rate_estimation(device):
    """Test decoherence rate estimation (requires coherence enabled)."""
    config = QuantumConfig(
        include_purity=False,
        include_entropy=False,
        include_coherence=True,
        include_variance=False,
        include_decoherence=True,
    )
    extractor = QuantumFeatureExtractor(device=device, config=config, grid_size=64)

    N, T = 4, 50  # Need longer time series for rate estimation
    psi = create_normalized_gaussian_wavepacket(N, T, 64, 64, device)

    features = extractor.extract(psi)

    # Coherence (index 0), decoherence_rate (index 1)
    coh = features[:, :, 0]
    gamma = features[:, :, 1]

    # Decoherence rate should be constant across time (summary statistic)
    gamma_first = gamma[:, 0]
    gamma_last = gamma[:, -1]
    assert torch.allclose(gamma_first, gamma_last, atol=1e-6), (
        "Decoherence rate should be constant across time"
    )

    # For random wavefunctions, rate can be positive or negative (no actual decay)
    # Just check that it's within reasonable bounds
    assert torch.all(torch.abs(gamma) < 1.0), (
        f"Decoherence rate should be bounded for random data. Got range=[{gamma.min():.6f}, {gamma.max():.6f}]"
    )

    print(
        f"✓ Decoherence rate test passed: "
        f"gamma_mean={gamma[:, 0].mean():.6f}, gamma_std={gamma[:, 0].std():.6f}, "
        f"gamma_range=[{gamma.min():.6f}, {gamma.max():.6f}]"
    )


if __name__ == "__main__":
    """Run all tests."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running quantum feature tests on device: {device}\n")

    test_purity_pure_state(device)
    test_linear_entropy_pure_state(device)
    test_uncertainty_principle(device)
    test_entropy_bounds(device)
    test_coherence_positive(device)
    test_all_features_enabled(device)
    test_decoherence_rate_estimation(device)

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
