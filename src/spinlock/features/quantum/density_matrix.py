"""Construct reduced density matrix from wavefunction."""

import torch
from typing import Tuple


def wavefunction_to_density_matrix(
    psi: torch.Tensor  # [N, T, 2, H, W] (Re, Im channels)
) -> torch.Tensor:
    """Convert wavefunction to flattened complex representation.

    For pure state ψ, ρ = |ψ⟩⟨ψ| is rank-1.
    We work in position basis, so ρ(x, x') = ψ(x) ψ*(x').

    Note: Full matrix is O(N⁴) memory! This function returns the flattened
    wavefunction for efficient computation of density matrix properties.

    Args:
        psi: Wavefunction [N, T, 2, H, W] where channel 0 is Re, 1 is Im

    Returns:
        Flattened complex wavefunction [N, T, D] where D = H*W
    """
    N, T, _, H, W = psi.shape
    D = H * W  # Hilbert space dimension

    # Convert to complex
    psi_complex = torch.complex(psi[:, :, 0], psi[:, :, 1])  # [N, T, H, W]

    # Flatten spatial dimensions: [N, T, H, W] → [N, T, D]
    psi_flat = psi_complex.reshape(N, T, D)

    return psi_flat


def compute_diagonal_elements(psi_flat: torch.Tensor) -> torch.Tensor:
    """Compute diagonal elements of density matrix.

    For pure state: ρ_ii = |ψ_i|² (probability at position i)

    Args:
        psi_flat: [N, T, D] complex wavefunction

    Returns:
        [N, T, D] real diagonal elements (probabilities)
    """
    return torch.abs(psi_flat) ** 2


def compute_off_diagonal_sum(psi_flat: torch.Tensor) -> torch.Tensor:
    """Compute sum of absolute off-diagonal elements.

    This is the coherence measure C = Σ|ρ_ij| for i≠j.

    For pure state ρ_ij = ψ_i ψ*_j:
    C = Σ_{i≠j} |ψ_i ψ*_j| = Σ_{i≠j} |ψ_i| |ψ_j|

    Efficiently computed as:
    Σ_{i≠j} |ψ_i||ψ_j| = (Σ|ψ_i|)² - Σ|ψ_i|²

    Args:
        psi_flat: [N, T, D] complex wavefunction

    Returns:
        [N, T] coherence measure
    """
    N, T, D = psi_flat.shape

    # |ψ_i|
    abs_psi = torch.abs(psi_flat)  # [N, T, D]

    # Sum of all pairwise products: (Σ|ψ_i|)² = Σ|ψ_i|² + 2·Σ_{i<j}|ψ_i||ψ_j|
    # Therefore: Σ_{i≠j}|ψ_i||ψ_j| = (Σ|ψ_i|)² - Σ|ψ_i|²

    sum_abs = abs_psi.sum(dim=-1)  # [N, T]
    sum_squared = (abs_psi ** 2).sum(dim=-1)  # [N, T]

    coherence = sum_abs ** 2 - sum_squared  # = 2·Σ_{i<j}|ψ_i||ψ_j|

    return coherence
