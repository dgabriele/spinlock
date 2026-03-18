"""Theta-coherent Fourier IC generator for Lenia.

Generates initial conditions as sums of oriented cosine waves whose spatial
frequencies are derived from kernel_radii (theta parameter).  This couples
IC structure to simulation dynamics, ensuring similar parameters produce
similar ICs — a smooth VQ landscape with wide basins.

The IC manifold dimensionality is 3C·K (C channels × K modes × 3 free
params per mode: amplitude, orientation, phase), compared to C·H·W for
random pixel ICs.  For C=3, K=4, H=W=128: 36 vs 49,152 dimensions.

Formula per channel c:
    IC_c(x, y) = dc_offset + (1/√K) · Σ_{k=1..K}
        A_k · cos(2π f_k (x·cos(θ_k) + y·sin(θ_k)) / G + φ_k)

where f_k = base_freq_scale · G / kernel_radius_c · harmonic_ratios[k].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass
class FourierICConfig:
    """Configuration for Fourier IC generation."""

    num_modes: int = 4
    base_frequency_scale: float = 1.0
    harmonic_ratios: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    amplitude_range: Tuple[float, float] = (0.2, 1.0)
    dc_offset: float = 0.5


class FourierICGenerator:
    """Theta-coherent Fourier IC generator for Lenia.

    Spatial frequencies are derived from kernel_radii (theta parameter),
    ensuring similar parameters produce similar ICs and a smooth VQ landscape.
    Amplitudes, orientations, and phases are free parameters (seeded).
    """

    def __init__(self, config: FourierICConfig = FourierICConfig()):
        self.config = config
        self.last_types: Optional[list[str]] = None
        # Cached coordinate grids (lazily initialized per device+grid_size)
        self._coord_cache: dict[tuple, Tensor] = {}

    def _get_coords(self, grid_size: int, device: torch.device) -> Tensor:
        """Get or create cached normalized coordinate grid [H, W]."""
        key = (grid_size, device)
        if key not in self._coord_cache:
            coords = torch.arange(grid_size, device=device, dtype=torch.float32)
            self._coord_cache[key] = coords
        return self._coord_cache[key]

    def generate_batch(
        self,
        batch_size: int,
        n_channels: int,
        grid_size: int = 128,
        seed: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        kernel_radii: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Generate [B, C, H, W] Fourier ICs.

        Args:
            batch_size: Number of samples.
            n_channels: Number of Lenia channels.
            grid_size: Spatial resolution (G).
            seed: Random seed for reproducibility.
            device: Target device.
            kernel_radii: [B, C] kernel radii from LeniaBatchTensors.radii.
                If None, falls back to random frequencies (backward compat).

        Returns:
            [B, C, H, W] float32 in [0, 1].
        """
        cfg = self.config
        K = cfg.num_modes
        G = grid_size
        B = batch_size
        C = n_channels

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)

        # Harmonic ratios [K] — truncate or pad if config doesn't match num_modes
        harmonics = torch.tensor(
            cfg.harmonic_ratios[:K], device=device, dtype=torch.float32,
        )
        if len(harmonics) < K:
            # Extend with integer harmonics beyond what's configured
            extra = torch.arange(
                len(harmonics) + 1, K + 1, device=device, dtype=torch.float32,
            )
            harmonics = torch.cat([harmonics, extra])

        # ── Compute spatial frequencies [B, C, K] ──
        if kernel_radii is not None:
            # Theta-coherent: f_k = scale * G / radius * harmonic_k
            radii = kernel_radii.to(device=device, dtype=torch.float32)  # [B, C]
            # Clamp radii to avoid division issues
            radii = radii.clamp(min=1.0)
            # [B, C, 1] * [K] → [B, C, K]
            freqs = (
                cfg.base_frequency_scale
                * G
                / radii.unsqueeze(-1)
                * harmonics.unsqueeze(0).unsqueeze(0)
            )
        else:
            # Fallback: random frequencies (no theta coupling)
            freqs = torch.empty(B, C, K, device=device, dtype=torch.float32)
            # Random frequencies in reasonable range [0.5, 8.0] cycles across grid
            freqs.uniform_(0.5, 8.0, generator=rng)

        # ── Sample free parameters ──
        # Amplitudes [B, C, K] in [amp_lo, amp_hi]
        amp_lo, amp_hi = cfg.amplitude_range
        amplitudes = torch.empty(B, C, K, device=device, dtype=torch.float32)
        amplitudes.uniform_(0.0, 1.0, generator=rng)
        amplitudes = amp_lo + amplitudes * (amp_hi - amp_lo)

        # Orientations [B, C, K] in [0, 2π)
        orientations = torch.empty(B, C, K, device=device, dtype=torch.float32)
        orientations.uniform_(0.0, 2.0 * math.pi, generator=rng)

        # Phases [B, C, K] in [0, 2π)
        phases = torch.empty(B, C, K, device=device, dtype=torch.float32)
        phases.uniform_(0.0, 2.0 * math.pi, generator=rng)

        # ── Build IC field vectorized over B, C, K ──
        coords = self._get_coords(G, device)  # [G]
        # Coordinate grids [H, W]
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # [H, W]

        # Project spatial coords onto wave direction:
        #   proj = x * cos(θ) + y * sin(θ)
        # Reshape for broadcasting: orientations [B, C, K, 1, 1]
        cos_th = orientations.unsqueeze(-1).unsqueeze(-1).cos()  # [B, C, K, 1, 1]
        sin_th = orientations.unsqueeze(-1).unsqueeze(-1).sin()  # [B, C, K, 1, 1]
        # xx, yy [H, W] → [1, 1, 1, H, W]
        xx_5d = xx.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        yy_5d = yy.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        proj = xx_5d * cos_th + yy_5d * sin_th  # [B, C, K, H, W]

        # Cosine waves: A_k * cos(2π f_k * proj / G + φ_k)
        freqs_5d = freqs.unsqueeze(-1).unsqueeze(-1)        # [B, C, K, 1, 1]
        phases_5d = phases.unsqueeze(-1).unsqueeze(-1)       # [B, C, K, 1, 1]
        amplitudes_5d = amplitudes.unsqueeze(-1).unsqueeze(-1)  # [B, C, K, 1, 1]

        waves = amplitudes_5d * torch.cos(
            2.0 * math.pi * freqs_5d * proj / G + phases_5d
        )  # [B, C, K, H, W]

        # Sum modes with 1/√K normalization + dc_offset
        ic = cfg.dc_offset + waves.sum(dim=2) / math.sqrt(K)  # [B, C, H, W]

        # Clamp to [0, 1]
        ic = ic.clamp(0.0, 1.0)

        self.last_types = ["fourier"] * B
        return ic
