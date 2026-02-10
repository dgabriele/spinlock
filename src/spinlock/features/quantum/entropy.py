"""Entropy measures: von Neumann and linear"""

import torch


def compute_von_neumann_entropy_approximate(
    prob: torch.Tensor,  # [N, T, D] diagonal elements |ψ_i|²
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Approximate von Neumann entropy S = -Tr(ρ log ρ).

    For pure state, exact computation requires diagonalizing ρ (O(D³)).

    Approximation: Assume ρ is approximately diagonal (valid in position basis
    after decoherence). Then:

    S ≈ -Σ_i p_i log p_i  (Shannon entropy of diagonal elements)

    where p_i = |ψ_i|² are probabilities.

    This approximation is exact for diagonal ρ and provides a good estimate
    for weakly coherent states.

    Args:
        prob: [N, T, D] probability distribution |ψ(x)|²
        epsilon: Small constant to avoid log(0)

    Returns:
        [N, T] approximate von Neumann entropy
    """
    # Add epsilon to avoid log(0)
    prob_safe = torch.clamp(prob, min=epsilon)

    # -Σ p_i log p_i
    entropy = -(prob_safe * torch.log(prob_safe)).sum(dim=-1)  # [N, T]

    return entropy
