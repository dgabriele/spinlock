"""Lenia CA parameter definition and Sobol vector mapping.

Literature-informed parameter ranges with log-uniform sigma, T reparameterization,
multi-ring kernels (variable B), and categorical kernel/growth types.

Sobol vector layout (C=3, 34 dims):
    Dims 0-2:   kernel_radius[3]      [8, 200]     integer
    Dims 3-5:   growth_mu[3]          [0.12, 0.38] linear
    Dims 6-8:   growth_sigma[3]       [0.01, 0.1]  log-uniform
    Dim  9:     T (time_scale)        [4, 400]     integer → dt=1/T
    Dims 10-15: coupling[6]           [-0.5, 1.0]  linear
    Dim  16:    kernel_rank (B)       [1, 5]       integer
    Dims 17-31: beta[3ch × 5rings]    [0, 1]       linear, per-ch normalized
    Dim  32:    kernel_type           {0,1,2}       categorical
    Dim  33:    growth_type           {0,1,2}       categorical
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# Parameter range specifications
# =============================================================================


@dataclass
class LeniaParamRanges:
    """Defines how Sobol unit vectors map to physical Lenia parameters.

    Controls ranges, transforms (log-uniform, integer), and which optional
    parameter dimensions (multi-ring kernels, categorical types) are active.
    """

    radius_bounds: Tuple[float, float]
    radius_integer: bool
    mu_bounds: Tuple[float, float]
    sigma_bounds: Tuple[float, float]
    sigma_log_scale: bool
    coupling_bounds: Tuple[float, float]
    # T-mode: dt = 1/T with integer T; None = direct dt sampling
    time_scale_bounds: Optional[Tuple[int, int]] = None
    dt_bounds: Optional[Tuple[float, float]] = None
    # Phase 2: multi-ring kernels
    max_rings: int = 1
    kernel_rank_bounds: Optional[Tuple[int, int]] = None
    # Phase 3: categorical types
    kernel_types: Optional[List[str]] = None
    growth_types: Optional[List[str]] = None


DEFAULT_RANGES = LeniaParamRanges(
    radius_bounds=(8.0, 200.0),
    radius_integer=True,
    mu_bounds=(0.12, 0.38),
    sigma_bounds=(0.01, 0.1),
    sigma_log_scale=True,
    coupling_bounds=(-0.5, 1.0),
    time_scale_bounds=(4, 400),
    max_rings=5,
    kernel_rank_bounds=(1, 5),
    kernel_types=["gaussian", "polynomial", "step"],
    growth_types=["gaussian", "polynomial", "step"],
)


def sobol_expected_dims(n_channels: int, ranges: LeniaParamRanges) -> int:
    """Compute expected Sobol vector dimensionality for given ranges."""
    C = n_channels
    # Base: radius[C] + mu[C] + sigma[C] + dt/T[1] + coupling[C*(C-1)]
    base = C * C + 2 * C + 1
    extra = 0
    if ranges.kernel_rank_bounds is not None:
        extra += 1
    if ranges.max_rings > 1:
        extra += C * ranges.max_rings
    if ranges.kernel_types is not None:
        extra += 1
    if ranges.growth_types is not None:
        extra += 1
    return base + extra


# =============================================================================
# Parameter dataclasses
# =============================================================================


@dataclass
class LeniaParams:
    """Per-sample Lenia simulation parameters.

    Extended for V3 with multi-ring kernels and categorical types.
    V2-era callers can ignore the new fields (defaults are backward-compatible).
    """

    n_channels: int
    kernel_radius: List[float]
    growth_mu: List[float]
    growth_sigma: List[float]
    dt: float
    coupling: List[List[float]]
    kernel_type: str = "gaussian"
    kernel_rank: int = 1
    beta: Optional[List[List[float]]] = None  # [C][max_rings], normalized per-channel
    growth_type: str = "gaussian"

    def __post_init__(self):
        C = self.n_channels
        assert len(self.kernel_radius) == C
        assert len(self.growth_mu) == C
        assert len(self.growth_sigma) == C
        assert len(self.coupling) == C
        for row in self.coupling:
            assert len(row) == C
        if self.beta is not None:
            assert len(self.beta) == C
            for row in self.beta:
                assert len(row) >= self.kernel_rank


@dataclass
class LeniaBatchTensors:
    """Batched parameter tensors extracted from Sobol vectors.

    The core 5 fields (radii through coupling) are always present.
    Phase 2+3 fields are None when using V2 ranges.
    """

    radii: torch.Tensor           # [B, C]
    growth_mu: torch.Tensor       # [B, C]
    growth_sigma: torch.Tensor    # [B, C]
    dt: torch.Tensor              # [B]
    coupling: torch.Tensor        # [B, C, C]
    # Phase 2: multi-ring kernels
    kernel_rank: Optional[torch.Tensor] = None   # [B] long
    beta: Optional[torch.Tensor] = None          # [B, C, max_rings]
    # Phase 3: categorical types
    kernel_type: Optional[torch.Tensor] = None   # [B] long
    growth_type: Optional[torch.Tensor] = None   # [B] long


# =============================================================================
# Per-sample Sobol → LeniaParams (used in retry path)
# =============================================================================


def sobol_to_lenia_params(
    unit_vec: np.ndarray,
    n_channels: int,
    kernel_type: str = "gaussian",
    ranges: Optional[LeniaParamRanges] = None,
) -> LeniaParams:
    """Map [0,1]^D Sobol unit vector to LeniaParams.

    Args:
        unit_vec: Sobol unit vector in [0,1]^D.
        n_channels: Number of Lenia channels (C).
        kernel_type: Fallback kernel type (overridden by ranges.kernel_types).
        ranges: Parameter range spec (default: DEFAULT_RANGES for backward compat).

    Returns:
        LeniaParams with all fields populated.
    """
    if ranges is None:
        ranges = DEFAULT_RANGES

    C = n_channels
    expected_dim = sobol_expected_dims(C, ranges)
    if len(unit_vec) != expected_dim:
        raise ValueError(
            f"Expected unit_vec of length {expected_dim} for C={C} with "
            f"given ranges, got {len(unit_vec)}"
        )

    # --- Radii [0..C) ---
    lo, hi = ranges.radius_bounds
    kernel_radius = [lo + u * (hi - lo) for u in unit_vec[0:C]]
    if ranges.radius_integer:
        kernel_radius = [float(round(r)) for r in kernel_radius]

    # --- Growth mu [C..2C) ---
    lo, hi = ranges.mu_bounds
    growth_mu = [lo + u * (hi - lo) for u in unit_vec[C:2*C]]

    # --- Growth sigma [2C..3C) ---
    lo, hi = ranges.sigma_bounds
    if ranges.sigma_log_scale:
        log_lo, log_hi = math.log10(lo), math.log10(hi)
        growth_sigma = [
            10.0 ** (log_lo + u * (log_hi - log_lo))
            for u in unit_vec[2*C:3*C]
        ]
    else:
        growth_sigma = [lo + u * (hi - lo) for u in unit_vec[2*C:3*C]]

    # --- dt/T [3C] ---
    if ranges.time_scale_bounds is not None:
        T_lo, T_hi = ranges.time_scale_bounds
        T = round(T_lo + unit_vec[3*C] * (T_hi - T_lo))
        T = max(T_lo, min(T_hi, T))  # clamp
        dt = 1.0 / T
    else:
        lo, hi = ranges.dt_bounds
        dt = lo + unit_vec[3*C] * (hi - lo)

    # --- Coupling [3C+1 .. 3C+C*(C-1)) ---
    coupling = [[1.0 if i == j else 0.0 for j in range(C)] for i in range(C)]
    c_lo, c_hi = ranges.coupling_bounds
    off_diag_vals = unit_vec[3*C + 1: 3*C + 1 + C*(C-1)]
    idx = 0
    for i in range(C):
        for j in range(C):
            if i != j:
                coupling[i][j] = c_lo + off_diag_vals[idx] * (c_hi - c_lo)
                idx += 1

    dim_cursor = 3*C + 1 + C*(C-1)

    # --- Phase 2: kernel_rank and beta ---
    kr = 1
    beta = None
    if ranges.kernel_rank_bounds is not None:
        kr_lo, kr_hi = ranges.kernel_rank_bounds
        kr = round(kr_lo + unit_vec[dim_cursor] * (kr_hi - kr_lo))
        kr = max(kr_lo, min(kr_hi, kr))
        dim_cursor += 1

    if ranges.max_rings > 1:
        MR = ranges.max_rings
        beta = []
        for c in range(C):
            raw = list(unit_vec[dim_cursor + c*MR: dim_cursor + (c+1)*MR])
            # Zero out unused rings
            for r in range(MR):
                if r >= kr:
                    raw[r] = 0.0
            # Normalize active rings to sum=1
            total = sum(raw[:kr]) if kr > 0 else 1e-12
            if total < 1e-12:
                total = 1e-12
            beta.append([v / total if r < kr else 0.0 for r, v in enumerate(raw)])
        dim_cursor += C * MR

    # --- Phase 3: kernel_type ---
    resolved_kernel_type = kernel_type
    if ranges.kernel_types is not None:
        n_types = len(ranges.kernel_types)
        type_idx = min(int(unit_vec[dim_cursor] * n_types), n_types - 1)
        resolved_kernel_type = ranges.kernel_types[type_idx]
        dim_cursor += 1

    # --- Phase 3: growth_type ---
    growth_type = "gaussian"
    if ranges.growth_types is not None:
        n_types = len(ranges.growth_types)
        type_idx = min(int(unit_vec[dim_cursor] * n_types), n_types - 1)
        growth_type = ranges.growth_types[type_idx]
        dim_cursor += 1

    return LeniaParams(
        n_channels=n_channels,
        kernel_radius=kernel_radius,
        growth_mu=growth_mu,
        growth_sigma=growth_sigma,
        dt=dt,
        coupling=coupling,
        kernel_type=resolved_kernel_type,
        kernel_rank=kr,
        beta=beta,
        growth_type=growth_type,
    )


# =============================================================================
# Batched Sobol → tensors (GPU-vectorized fast path)
# =============================================================================


def sobol_batch_to_tensors(
    unit_vecs: np.ndarray,
    n_channels: int,
    device: torch.device,
    ranges: Optional[LeniaParamRanges] = None,
) -> LeniaBatchTensors:
    """Vectorized Sobol→tensor conversion for a batch of unit vectors.

    Converts B Sobol vectors directly to GPU tensors without constructing
    intermediate LeniaParams dataclasses.

    Args:
        unit_vecs: [B, D] Sobol unit vectors in [0,1].
        n_channels: Number of Lenia channels (C).
        device: Target device for output tensors.
        ranges: Parameter range spec (default: DEFAULT_RANGES for backward compat).

    Returns:
        LeniaBatchTensors with all applicable fields populated.
    """
    if ranges is None:
        ranges = DEFAULT_RANGES

    C = n_channels
    uv = torch.as_tensor(unit_vecs, dtype=torch.float32, device=device)
    B = uv.shape[0]

    # --- Radii [0..C) ---
    lo, hi = ranges.radius_bounds
    radii = lo + uv[:, :C] * (hi - lo)
    if ranges.radius_integer:
        radii = radii.round()

    # --- Growth mu [C..2C) ---
    lo, hi = ranges.mu_bounds
    growth_mu = lo + uv[:, C:2*C] * (hi - lo)

    # --- Growth sigma [2C..3C) — log-uniform when configured ---
    lo, hi = ranges.sigma_bounds
    if ranges.sigma_log_scale:
        log_lo = math.log10(lo)
        log_hi = math.log10(hi)
        growth_sigma = 10.0 ** (log_lo + uv[:, 2*C:3*C] * (log_hi - log_lo))
    else:
        growth_sigma = lo + uv[:, 2*C:3*C] * (hi - lo)

    # --- dt/T [3C] ---
    if ranges.time_scale_bounds is not None:
        T_lo, T_hi = ranges.time_scale_bounds
        T = (T_lo + uv[:, 3*C] * (T_hi - T_lo)).round().clamp(min=T_lo, max=T_hi)
        dt = 1.0 / T
    else:
        lo, hi = ranges.dt_bounds
        dt = lo + uv[:, 3*C] * (hi - lo)

    # --- Coupling [3C+1 .. 3C+1+C*(C-1)) ---
    coupling = (
        torch.eye(C, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(B, -1, -1)
        .clone()
    )
    c_lo, c_hi = ranges.coupling_bounds
    off_diag_raw = uv[:, 3*C + 1: 3*C + 1 + C*(C-1)]
    off_diag_vals = c_lo + off_diag_raw * (c_hi - c_lo)
    mask = ~torch.eye(C, dtype=torch.bool, device=device)
    coupling[:, mask] = off_diag_vals

    dim_cursor = 3*C + 1 + C*(C-1)

    # --- Phase 2: kernel_rank and beta ---
    kernel_rank = None
    beta = None

    if ranges.kernel_rank_bounds is not None:
        kr_lo, kr_hi = ranges.kernel_rank_bounds
        kernel_rank = (
            (kr_lo + uv[:, dim_cursor] * (kr_hi - kr_lo))
            .round()
            .clamp(min=kr_lo, max=kr_hi)
            .long()
        )
        dim_cursor += 1

    if ranges.max_rings > 1:
        MR = ranges.max_rings
        beta_raw = uv[:, dim_cursor: dim_cursor + C * MR].reshape(B, C, MR)
        dim_cursor += C * MR

        # Zero out unused rings: positions >= kernel_rank
        if kernel_rank is not None:
            ring_indices = torch.arange(MR, device=device)  # [MR]
            # kernel_rank [B] → [B, 1, 1] for broadcast over [B, C, MR]
            kr_expanded = kernel_rank.unsqueeze(1).unsqueeze(2)
            ring_mask = ring_indices.unsqueeze(0).unsqueeze(0) < kr_expanded
            beta_raw = beta_raw * ring_mask.float()

        # Normalize per-channel to sum=1
        beta_sum = beta_raw.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        beta = beta_raw / beta_sum

    # --- Phase 3: kernel_type ---
    kernel_type_ids = None
    if ranges.kernel_types is not None:
        n_types = len(ranges.kernel_types)
        kernel_type_ids = (
            (uv[:, dim_cursor] * n_types)
            .floor()
            .long()
            .clamp(max=n_types - 1)
        )
        dim_cursor += 1

    # --- Phase 3: growth_type ---
    growth_type_ids = None
    if ranges.growth_types is not None:
        n_types = len(ranges.growth_types)
        growth_type_ids = (
            (uv[:, dim_cursor] * n_types)
            .floor()
            .long()
            .clamp(max=n_types - 1)
        )
        dim_cursor += 1

    return LeniaBatchTensors(
        radii=radii,
        growth_mu=growth_mu,
        growth_sigma=growth_sigma,
        dt=dt,
        coupling=coupling,
        kernel_rank=kernel_rank,
        beta=beta,
        kernel_type=kernel_type_ids,
        growth_type=growth_type_ids,
    )
