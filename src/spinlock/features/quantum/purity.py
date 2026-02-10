"""Purity computation: Tr(ρ²)"""

import torch


def compute_purity(psi_flat: torch.Tensor) -> torch.Tensor:
    """Compute purity Tr(ρ²) for pure states.

    For pure state |ψ⟩: ρ = |ψ⟩⟨ψ|
    Tr(ρ²) = Tr(|ψ⟩⟨ψ|ψ⟩⟨ψ|) = ⟨ψ|ψ⟩² = 1 (if normalized)

    For pure states, we use the identity:
    Tr(ρ²) = (Σ|ψ_i|²)² = 1 for normalized states

    In practice, numerical errors may cause slight deviations from 1.
    This serves as a validation check.

    For mixed states (e.g., after thermal decoherence), purity < 1.

    Args:
        psi_flat: [N, T, D] complex wavefunction

    Returns:
        [N, T] purity values (should be ≈1 for pure states)
    """
    # |ψ|²
    prob = torch.abs(psi_flat) ** 2  # [N, T, D]

    # For pure states: Tr(ρ²) = (Σ|ψ_i|²)² = ⟨ψ|ψ⟩²
    norm_squared = prob.sum(dim=-1)  # [N, T], should be ≈1
    purity = norm_squared ** 2  # For pure states

    return purity


def compute_linear_entropy(purity: torch.Tensor, dimension: int) -> torch.Tensor:
    """Compute linear entropy from purity.

    S_lin = (1 - Tr(ρ²)) / (d - 1)

    where d is Hilbert space dimension (H*W for 2D grid).

    Linear entropy is a fast approximation to von Neumann entropy.
    - S_lin = 0 for pure states (purity = 1)
    - S_lin = 1 for maximally mixed states (purity = 1/d)

    Args:
        purity: [N, T] purity values
        dimension: Hilbert space dimension (H*W)

    Returns:
        [N, T] linear entropy
    """
    return (1 - purity) / (dimension - 1)
