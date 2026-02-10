"""Position and momentum uncertainty computation"""

import torch
from typing import Tuple


def compute_position_variance(
    psi: torch.Tensor,  # [N, T, 2, H, W]
    x_grid: torch.Tensor,  # [H, W] position grid
    y_grid: torch.Tensor,  # [H, W] position grid
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute position uncertainties Δx, Δy.

    Var(x) = ⟨x²⟩ - ⟨x⟩²

    where ⟨x⟩ = ∫ x |ψ(x)|² dx

    Args:
        psi: Wavefunction [N, T, 2, H, W] (Re, Im channels)
        x_grid: [H, W] x-coordinates
        y_grid: [H, W] y-coordinates

    Returns:
        (delta_x, delta_y) each [N, T]
    """
    N, T, _, H, W = psi.shape

    # Probability density |ψ|²
    prob = psi[:, :, 0] ** 2 + psi[:, :, 1] ** 2  # [N, T, H, W]

    # Reshape grids for broadcasting
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Mean position: ⟨x⟩ = Σ x |ψ(x)|² dx
    x_mean = (prob * x_grid).sum(dim=(-2, -1))  # [N, T]
    y_mean = (prob * y_grid).sum(dim=(-2, -1))  # [N, T]

    # Mean square: ⟨x²⟩ = Σ x² |ψ(x)|² dx
    x2_mean = (prob * x_grid ** 2).sum(dim=(-2, -1))  # [N, T]
    y2_mean = (prob * y_grid ** 2).sum(dim=(-2, -1))  # [N, T]

    # Variance: Var(x) = ⟨x²⟩ - ⟨x⟩²
    var_x = x2_mean - x_mean ** 2
    var_y = y2_mean - y_mean ** 2

    # Standard deviation
    delta_x = torch.sqrt(torch.clamp(var_x, min=0))
    delta_y = torch.sqrt(torch.clamp(var_y, min=0))

    return delta_x, delta_y


def compute_momentum_variance(
    psi: torch.Tensor,  # [N, T, 2, H, W]
    px_grid: torch.Tensor,  # [H, W] momentum grid (FFT frequencies)
    py_grid: torch.Tensor,  # [H, W] momentum grid
    hbar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute momentum uncertainties Δpx, Δpy.

    Via Fourier transform: ψ̃(p) = FFT[ψ(x)]

    Then: Var(p) = ⟨p²⟩ - ⟨p⟩²

    Args:
        psi: Wavefunction [N, T, 2, H, W] (Re, Im channels)
        px_grid: [H, W] momentum grid (ℏ·2π·kx)
        py_grid: [H, W] momentum grid (ℏ·2π·ky)
        hbar: Reduced Planck constant (default: 1.0)

    Returns:
        (delta_px, delta_py) each [N, T]
    """
    N, T, _, H, W = psi.shape

    # Convert to complex
    psi_complex = torch.complex(psi[:, :, 0], psi[:, :, 1])  # [N, T, H, W]

    # Fourier transform
    psi_k = torch.fft.fft2(psi_complex, dim=(-2, -1))  # [N, T, H, W]

    # Momentum space probability density |ψ̃(p)|²
    prob_k = torch.abs(psi_k) ** 2  # [N, T, H, W]

    # Normalize
    norm_k = prob_k.sum(dim=(-2, -1), keepdim=True)
    prob_k = prob_k / (norm_k + 1e-10)

    # Reshape grids
    px_grid = px_grid.unsqueeze(0).unsqueeze(0)
    py_grid = py_grid.unsqueeze(0).unsqueeze(0)

    # Mean momentum
    px_mean = (prob_k * px_grid).sum(dim=(-2, -1))
    py_mean = (prob_k * py_grid).sum(dim=(-2, -1))

    # Mean square
    px2_mean = (prob_k * px_grid ** 2).sum(dim=(-2, -1))
    py2_mean = (prob_k * py_grid ** 2).sum(dim=(-2, -1))

    # Variance
    var_px = px2_mean - px_mean ** 2
    var_py = py2_mean - py_mean ** 2

    delta_px = torch.sqrt(torch.clamp(var_px, min=0))
    delta_py = torch.sqrt(torch.clamp(var_py, min=0))

    return delta_px, delta_py
