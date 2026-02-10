"""Coherence measure: l1-norm of off-diagonal elements"""

import torch


def compute_coherence_measure(psi_flat: torch.Tensor) -> torch.Tensor:
    """Compute coherence measure C = Σ|ρ_ij| for i≠j.

    This quantifies quantum coherence via off-diagonal density matrix elements.
    As decoherence proceeds, off-diagonals decay → C decreases.

    For a fully decohered (diagonal) state, C = 0.
    For maximally coherent states, C is large.

    This measure is basis-dependent. We compute in the position basis,
    where decoherence suppresses coherence between spatially separated points.

    Args:
        psi_flat: [N, T, D] complex wavefunction

    Returns:
        [N, T] coherence measure
    """
    from .density_matrix import compute_off_diagonal_sum

    return compute_off_diagonal_sum(psi_flat)
