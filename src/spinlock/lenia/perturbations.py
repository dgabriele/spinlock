"""Mid-simulation perturbation probing for Lenia dynamics diversity.

Perturbs Lenia states mid-trajectory to reveal latent dynamics. A fixed-point
attractor that recovers identically from perturbation is genuinely information-
poor. One that oscillates, shifts, or shows traveling waves has richer dynamics
encoded in its parameters — the temporal features just can't see it from the
unperturbed trajectory alone.

All perturbations scale with kernel_radius so they operate at the system's
natural length scale. GPU-batched: each function takes [B, C, H, W] and
returns [B, C, H, W].
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from pydantic import BaseModel, Field
from torch import Generator, Tensor


class PerturbationConfig(BaseModel):
    """Configuration for mid-simulation perturbation probing."""

    enabled: bool = Field(
        default=False,
        description="Enable perturbation probing (adds P perturbed realizations).",
    )
    num_perturbed_realizations: int = Field(
        default=3, ge=1, le=20,
        description="Number of perturbed realizations per parameter config.",
    )
    injection_fraction: float = Field(
        default=0.25, gt=0.0, lt=1.0,
        description="Inject perturbation at T * fraction of the simulation.",
    )
    types: Dict[str, float] = Field(
        default_factory=lambda: {
            "gaussian_bump": 0.3,
            "channel_swap": 0.2,
            "local_reset": 0.3,
            "global_noise": 0.2,
        },
        description="Perturbation type weights (normalized to probabilities).",
    )
    bump_amplitude: float = Field(
        default=0.3, gt=0.0, le=1.0,
        description="Amplitude of Gaussian bump perturbation.",
    )
    noise_amplitude: float = Field(
        default=0.1, gt=0.0, le=1.0,
        description="Amplitude of global noise perturbation.",
    )


def _sample_perturbation_type(
    config: PerturbationConfig,
    rng: Generator,
    device: torch.device,
) -> str:
    """Sample a perturbation type from weighted distribution."""
    names = list(config.types.keys())
    weights = torch.tensor(
        [config.types[n] for n in names],
        device=device,
        dtype=torch.float32,
    )
    weights = weights / weights.sum()
    idx = torch.multinomial(weights, 1, generator=rng).item()
    return names[idx]


def _effective_radius(kernel_radii: Tensor) -> Tensor:
    """Mean kernel radius per sample → perturbation spatial scale.

    Args:
        kernel_radii: [B, C] per-channel radii.
    Returns:
        [B] effective radius (mean across channels).
    """
    return kernel_radii.mean(dim=1)


def _gaussian_bump(
    state: Tensor,
    kernel_radii: Tensor,
    config: PerturbationConfig,
    rng: Generator,
) -> Tensor:
    """Add a localized Gaussian bump to one channel at a random position.

    The bump radius is proportional to the mean kernel radius, ensuring the
    perturbation operates at the system's natural length scale.

    Reveals: response to local energy injection — propagation, dissipation, waves.
    """
    B, C, H, W = state.shape
    device = state.device
    eff_r = _effective_radius(kernel_radii)  # [B]

    # Random center positions [B]
    cx = torch.empty(B, device=device).uniform_(0, H, generator=rng).long()
    cy = torch.empty(B, device=device).uniform_(0, W, generator=rng).long()

    # Random channel to perturb [B]
    ch = torch.randint(0, C, (B,), device=device, generator=rng)

    # Build coordinate grids
    yy = torch.arange(H, device=device, dtype=torch.float32)
    xx = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")  # [H, W]

    # Toroidal distance from center per sample
    # dx, dy with periodic wrapping
    dx = (grid_x.unsqueeze(0) - cx.float().view(B, 1, 1))  # [B, H, W]
    dy = (grid_y.unsqueeze(0) - cy.float().view(B, 1, 1))  # [B, H, W]
    # Periodic wrap
    dx = dx - H * torch.round(dx / H)
    dy = dy - W * torch.round(dy / W)
    dist_sq = dx ** 2 + dy ** 2  # [B, H, W]

    # Gaussian envelope: sigma = eff_r / 3 (99.7% energy within radius)
    sigma = (eff_r / 3.0).clamp(min=1.0)  # [B]
    sigma_sq = (sigma ** 2).view(B, 1, 1)  # [B, 1, 1]
    bump = config.bump_amplitude * torch.exp(-dist_sq / (2 * sigma_sq))  # [B, H, W]

    # Apply to selected channel only
    result = state.clone()
    for b in range(B):
        result[b, ch[b]] = (result[b, ch[b]] + bump[b]).clamp(0.0, 1.0)

    return result


def _channel_swap(
    state: Tensor,
    kernel_radii: Tensor,
    config: PerturbationConfig,
    rng: Generator,
) -> Tensor:
    """Swap two channels in a local circular patch.

    Patch radius is proportional to the mean kernel radius.

    Reveals: cross-channel coupling sensitivity — does the system restore
    channel identity or reorganize?
    """
    B, C, H, W = state.shape
    device = state.device

    if C < 2:
        return state.clone()

    eff_r = _effective_radius(kernel_radii)  # [B]

    # Random center
    cx = torch.empty(B, device=device).uniform_(0, H, generator=rng).long()
    cy = torch.empty(B, device=device).uniform_(0, W, generator=rng).long()

    # Pick two distinct channels per sample
    ch1 = torch.randint(0, C, (B,), device=device, generator=rng)
    ch2 = torch.randint(0, C - 1, (B,), device=device, generator=rng)
    ch2 = ch2 + (ch2 >= ch1).long()  # shift to avoid self-swap

    # Circular mask
    yy = torch.arange(H, device=device, dtype=torch.float32)
    xx = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    dx = grid_x.unsqueeze(0) - cx.float().view(B, 1, 1)
    dy = grid_y.unsqueeze(0) - cy.float().view(B, 1, 1)
    dx = dx - H * torch.round(dx / H)
    dy = dy - W * torch.round(dy / W)
    dist = (dx ** 2 + dy ** 2).sqrt()  # [B, H, W]

    mask = dist < eff_r.view(B, 1, 1)  # [B, H, W] bool

    result = state.clone()
    for b in range(B):
        m = mask[b]  # [H, W]
        c1, c2 = ch1[b].item(), ch2[b].item()
        tmp = result[b, c1, m].clone()
        result[b, c1, m] = result[b, c2, m]
        result[b, c2, m] = tmp

    return result


def _local_reset(
    state: Tensor,
    kernel_radii: Tensor,
    config: PerturbationConfig,
    rng: Generator,
) -> Tensor:
    """Reset a local patch to random [0, 1] values.

    Patch radius is 2× kernel radius (tests recovery from larger damage).

    Reveals: recovery from damage — heal, fragment, or reorganize?
    """
    B, C, H, W = state.shape
    device = state.device
    eff_r = _effective_radius(kernel_radii)  # [B]

    # Random center
    cx = torch.empty(B, device=device).uniform_(0, H, generator=rng).long()
    cy = torch.empty(B, device=device).uniform_(0, W, generator=rng).long()

    # Circular mask at 2× radius
    yy = torch.arange(H, device=device, dtype=torch.float32)
    xx = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    dx = grid_x.unsqueeze(0) - cx.float().view(B, 1, 1)
    dy = grid_y.unsqueeze(0) - cy.float().view(B, 1, 1)
    dx = dx - H * torch.round(dx / H)
    dy = dy - W * torch.round(dy / W)
    dist = (dx ** 2 + dy ** 2).sqrt()

    mask = dist < (2.0 * eff_r).view(B, 1, 1)  # [B, H, W]
    mask_4d = mask.unsqueeze(1).expand_as(state)  # [B, C, H, W]

    result = state.clone()
    noise = torch.empty_like(state).uniform_(0.0, 1.0, generator=rng)
    result[mask_4d] = noise[mask_4d]

    return result


def _global_noise(
    state: Tensor,
    kernel_radii: Tensor,
    config: PerturbationConfig,
    rng: Generator,
) -> Tensor:
    """Add uniform noise to entire grid.

    Reveals: stability — does it amplify perturbations (chaotic) or damp them (stable)?
    """
    noise = torch.empty_like(state).uniform_(
        -config.noise_amplitude, config.noise_amplitude, generator=rng,
    )
    return (state + noise).clamp(0.0, 1.0)


# Dispatch table: type name → implementation
_PERTURBATION_FNS = {
    "gaussian_bump": _gaussian_bump,
    "channel_swap": _channel_swap,
    "local_reset": _local_reset,
    "global_noise": _global_noise,
}


def apply_perturbation(
    state: Tensor,
    perturbation_type: str,
    kernel_radii: Tensor,
    config: PerturbationConfig,
    rng: Generator,
) -> Tensor:
    """Apply a perturbation to a batch of Lenia states.

    Args:
        state: [B, C, H, W] current simulation state.
        perturbation_type: One of 'gaussian_bump', 'channel_swap',
            'local_reset', 'global_noise'.
        kernel_radii: [B, C] kernel radii for length-scale coupling.
        config: Perturbation configuration.
        rng: PyTorch Generator for reproducibility.

    Returns:
        [B, C, H, W] perturbed state (clamped to [0, 1]).
    """
    fn = _PERTURBATION_FNS.get(perturbation_type)
    if fn is None:
        raise ValueError(
            f"Unknown perturbation type '{perturbation_type}'. "
            f"Available: {list(_PERTURBATION_FNS.keys())}"
        )
    return fn(state, kernel_radii, config, rng)
