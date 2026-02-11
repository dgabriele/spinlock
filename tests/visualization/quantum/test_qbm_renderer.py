"""Unit tests for QBM renderers."""
import torch
import pytest
from spinlock.visualization.quantum.renderer import QBMRenderer
from spinlock.visualization.quantum.wigner_renderer import WignerRenderer


def test_qbm_renderer_probability_input():
    """Test rendering from probability density input."""
    renderer = QBMRenderer(colormap='viridis', device=torch.device('cpu'))

    # [B, H, W] probability density
    prob_density = torch.rand(4, 64, 64)

    rendered = renderer.render(prob_density)

    assert rendered.shape == (4, 64, 64, 3)
    assert rendered.dtype == torch.float32
    assert rendered.min() >= 0.0 and rendered.max() <= 1.0


def test_qbm_renderer_complex_input():
    """Test rendering from complex [real, imag] input."""
    renderer = QBMRenderer(colormap='viridis', device=torch.device('cpu'))

    # [B, 2, H, W] complex wavefunction
    real = torch.randn(4, 1, 64, 64)
    imag = torch.randn(4, 1, 64, 64)
    complex_data = torch.cat([real, imag], dim=1)

    rendered = renderer.render(complex_data)

    assert rendered.shape == (4, 64, 64, 3)
    # Verify probability density was computed correctly
    expected_prob = (real[:, 0]**2 + imag[:, 0]**2).mean()
    assert torch.isfinite(expected_prob)


def test_qbm_renderer_4d_single_channel():
    """Test rendering from [B, 1, H, W] input."""
    renderer = QBMRenderer(colormap='viridis', device=torch.device('cpu'))

    # [B, 1, H, W] probability density
    prob_density = torch.rand(4, 1, 64, 64)

    rendered = renderer.render(prob_density)

    assert rendered.shape == (4, 64, 64, 3)


def test_wigner_renderer():
    """Test Wigner function computation and rendering."""
    renderer = WignerRenderer(colormap='RdBu_r', device=torch.device('cpu'))

    # [B, 2, H, W] complex wavefunction
    real = torch.randn(2, 1, 64, 64)
    imag = torch.randn(2, 1, 64, 64)
    complex_data = torch.cat([real, imag], dim=1)

    rendered = renderer.render(complex_data)

    assert rendered.shape == (2, 64, 64, 3)
    # Wigner can be negative, but rendered output normalized to [0,1]
    assert rendered.min() >= 0.0 and rendered.max() <= 1.0


def test_wigner_renderer_requires_two_channels():
    """Test that Wigner renderer requires 2 channels."""
    renderer = WignerRenderer(colormap='RdBu_r', device=torch.device('cpu'))

    # Single channel should fail
    single_channel = torch.rand(2, 1, 64, 64)

    with pytest.raises(ValueError, match="requires 2 channels"):
        renderer.render(single_channel)


def test_global_normalization():
    """Test consistent normalization across timesteps."""
    renderer = QBMRenderer(normalize_mode='global', device=torch.device('cpu'))

    # Full rollout [N, M, T, 2, H, W]
    rollout = torch.randn(2, 3, 10, 2, 64, 64)

    # Compute global stats
    global_stats = renderer.compute_global_stats(rollout)

    assert 'vmin' in global_stats and 'vmax' in global_stats
    assert global_stats['vmin'] < global_stats['vmax']

    # Render frames with consistent normalization
    for t in range(rollout.shape[2]):
        frame = rollout[:, :, t].reshape(-1, 2, 64, 64)  # [N*M, 2, H, W]
        rendered = renderer.render(frame, global_stats=global_stats)
        assert rendered.shape == (frame.shape[0], 64, 64, 3)


def test_qbm_renderer_supports_channels():
    """Test supports_channels method."""
    renderer = QBMRenderer(device=torch.device('cpu'))

    assert renderer.supports_channels(1) is True
    assert renderer.supports_channels(2) is True
    assert renderer.supports_channels(3) is False


def test_wigner_renderer_supports_channels():
    """Test supports_channels method for Wigner."""
    renderer = WignerRenderer(device=torch.device('cpu'))

    assert renderer.supports_channels(1) is False
    assert renderer.supports_channels(2) is True
    assert renderer.supports_channels(3) is False
