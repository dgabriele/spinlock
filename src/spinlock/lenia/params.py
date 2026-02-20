"""Lenia CA parameter definition and Sobol vector mapping."""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class LeniaParams:
    """Per-sample Lenia simulation parameters.

    Sobol vector layout for n_channels=C (total: C²+2C+1 dims):
        [0..C-1]         kernel_radius_i   [1.5, 35.0]
        [C..2C-1]        growth_mu_i       [0.01, 0.6]
        [2C..3C-1]       growth_sigma_i    [0.003, 0.25]
        [3C]             dt                [0.01, 0.4]
        [3C+1..3C+C(C-1)] coupling_ij      [-1.0, 1.5]  (row-major, skip diagonal)
    """

    n_channels: int
    kernel_radius: List[float]       # per-channel, len=n_channels
    growth_mu: List[float]           # per-channel
    growth_sigma: List[float]        # per-channel
    dt: float
    coupling: List[List[float]]      # [C, C] matrix; diagonal=1.0, off-diag sampled
    kernel_type: str = "gaussian"    # "gaussian" | "polynomial" | "exponential"

    def __post_init__(self):
        C = self.n_channels
        assert len(self.kernel_radius) == C
        assert len(self.growth_mu) == C
        assert len(self.growth_sigma) == C
        assert len(self.coupling) == C
        for row in self.coupling:
            assert len(row) == C


def sobol_to_lenia_params(
    unit_vec: np.ndarray,
    n_channels: int,
    kernel_type: str = "gaussian",
) -> LeniaParams:
    """Map [0,1]^(C²+2C+1) Sobol unit vector to LeniaParams.

    Coupling matrix: diagonal=1.0, off-diagonal entries from dims 3C+1..end
    stored in row-major order, skipping diagonal entries.
    """
    C = n_channels
    expected_dim = C * C + 2 * C + 1
    if len(unit_vec) != expected_dim:
        raise ValueError(
            f"Expected unit_vec of length {expected_dim} for C={C}, got {len(unit_vec)}"
        )

    # Per-channel params
    kernel_radius = [1.5 + u * (35.0 - 1.5) for u in unit_vec[0:C]]
    growth_mu = [0.01 + u * (0.6 - 0.01) for u in unit_vec[C:2*C]]
    growth_sigma = [0.003 + u * (0.25 - 0.003) for u in unit_vec[2*C:3*C]]

    # Shared dt
    dt = 0.01 + unit_vec[3*C] * (0.4 - 0.01)

    # Coupling matrix: diagonal=1.0, off-diagonal from remaining dims
    # Row-major: for i in 0..C-1, for j in 0..C-1, if i != j
    coupling = [[1.0 if i == j else 0.0 for j in range(C)] for i in range(C)]
    off_diag_vals = unit_vec[3*C + 1:]
    idx = 0
    for i in range(C):
        for j in range(C):
            if i != j:
                coupling[i][j] = -1.0 + off_diag_vals[idx] * (1.5 - (-1.0))
                idx += 1

    return LeniaParams(
        n_channels=n_channels,
        kernel_radius=kernel_radius,
        growth_mu=growth_mu,
        growth_sigma=growth_sigma,
        dt=dt,
        coupling=coupling,
        kernel_type=kernel_type,
    )
