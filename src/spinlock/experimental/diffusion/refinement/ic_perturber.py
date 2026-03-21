"""Sobol-based IC perturbation in Fourier parameter space.

Extracts the (amplitude, orientation, phase) free parameters from decoded IC
grids via 2D FFT peak analysis, normalizes them to [0,1]^D, then uses a
scrambled Sobol sequence to generate quasi-random perturbations around that
center.  Reconstructs new IC grids using frequencies derived from the
perturbed theta's kernel radii.

Why Sobol instead of Gaussian:
  In 36 dimensions, Gaussian noise concentrates in a thin shell near ||x||=6.
  Sobol low-discrepancy offsets fill the local region uniformly, giving far
  better coverage per sample.  For P=4 perturbations this matters.

Parameter space (C=3, K=4 = 36 free dims):
  - 12 amplitudes  in [amp_lo, amp_hi]  (FourierICConfig.amplitude_range)
  - 12 orientations in [0, 2pi)
  - 12 phases       in [0, 2pi)
  Frequencies are NOT free --- derived from theta's kernel_radii.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

from spinlock.lenia.fourier_ic import FourierICConfig
from spinlock.sampling.sobol import StratifiedSobolSampler


class FourierICPerturber:
    """Perturb decoded IC grids in Fourier parameter space via Sobol offsets.

    Pipeline:
      1. FFT extract (A, theta, phi) from decoded grids     [R', C, K] each
      2. Flatten + normalize to [0,1]^(3*C*K)               [R', D]
      3. Sobol quasi-random offsets centered on each point   [R'*P, D]
      4. De-normalize back to physical ranges                [R'*P, C, K] each
      5. Reconstruct with FourierICGenerator's formula       [R'*P, C, H, W]
    """

    def __init__(
        self,
        ic_config: FourierICConfig = FourierICConfig(),
    ) -> None:
        self.ic_config = ic_config
        self.K = ic_config.num_modes
        self.dc_offset = ic_config.dc_offset
        self.base_frequency_scale = ic_config.base_frequency_scale
        self.harmonic_ratios = ic_config.harmonic_ratios

        self.amp_lo, self.amp_hi = ic_config.amplitude_range

    def perturb_from_decoded(
        self,
        decoded_ics: Tensor,      # [R', C, H, W]
        perturbed_radii: Tensor,  # [R'*P, C]
        sigma: float,
        n_per_center: int,        # P
        seed: int = 0,
    ) -> Tensor:                  # [R'*P, C, H, W]
        """Extract Fourier params, Sobol-perturb in [0,1]^D, reconstruct.

        Args:
            decoded_ics: Decoded IC grids from D3PM's best tokens.
            perturbed_radii: Kernel radii from perturbed theta [R'*P, C].
            sigma: Perturbation radius in [0,1] space (same scale as theta).
            n_per_center: Number of perturbations per decoded IC.
            seed: Sobol scrambling seed (should vary per cycle/round).

        Returns:
            Perturbed IC grids [R'*P, C, H, W] in [0, 1].
        """
        R, C, H, W = decoded_ics.shape
        P = n_per_center
        K = self.K
        device = decoded_ics.device
        N = R * P
        D = 3 * C * K  # total free IC params

        # 1. Extract Fourier params from decoded ICs
        amplitudes, orientations, phases = self._extract_params(decoded_ics)
        # Each is [R', C, K]

        # 2. Normalize to [0,1] and flatten to [R', D]
        centers = self._params_to_unit(amplitudes, orientations, phases)

        # 3. Sobol perturbation around each center
        perturbed_unit = self._sobol_perturb(centers, sigma, P, device, seed)
        # [R'*P, D]

        # 4. De-normalize back to physical ranges [N, C, K] each
        amp_p, orient_p, phase_p = self._unit_to_params(perturbed_unit, C, K)

        # 5. Compute frequencies from perturbed theta's radii
        harmonic_t = torch.tensor(
            self.harmonic_ratios[:K], device=device, dtype=torch.float32,
        )
        radii_clamped = perturbed_radii.clamp(min=1.0)
        frequencies = (
            self.base_frequency_scale * H
            / radii_clamped.unsqueeze(-1)
            * harmonic_t[None, None, :]
        )  # [N, C, K]

        # 6. Reconstruct grids
        return self._reconstruct(frequencies, amp_p, orient_p, phase_p, H)

    # -- Normalization to/from [0,1] -----------------------------------------

    def _params_to_unit(
        self,
        amplitudes: Tensor,    # [B, C, K]
        orientations: Tensor,  # [B, C, K]
        phases: Tensor,        # [B, C, K]
    ) -> Tensor:               # [B, D]
        """Flatten and normalize (A, theta, phi) to [0,1]^D."""
        B = amplitudes.shape[0]
        # Normalize each param type to [0,1]
        amp_unit = (amplitudes - self.amp_lo) / (self.amp_hi - self.amp_lo)
        amp_unit = amp_unit.clamp(0.0, 1.0)
        orient_unit = orientations / (2 * math.pi)
        orient_unit = orient_unit.clamp(0.0, 1.0)
        phase_unit = phases / (2 * math.pi)
        phase_unit = phase_unit.clamp(0.0, 1.0)
        # Flatten: [B, C, K] -> [B, C*K], then concat 3 types -> [B, 3*C*K]
        return torch.cat([
            amp_unit.reshape(B, -1),
            orient_unit.reshape(B, -1),
            phase_unit.reshape(B, -1),
        ], dim=1)

    def _unit_to_params(
        self,
        unit: Tensor,  # [N, D]
        C: int,
        K: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """De-normalize [0,1]^D back to physical (A, theta, phi) [N,C,K]."""
        N = unit.shape[0]
        CK = C * K
        amp_unit = unit[:, :CK].reshape(N, C, K)
        orient_unit = unit[:, CK:2 * CK].reshape(N, C, K)
        phase_unit = unit[:, 2 * CK:].reshape(N, C, K)

        amplitudes = self.amp_lo + amp_unit * (self.amp_hi - self.amp_lo)
        orientations = orient_unit * (2 * math.pi)
        phases = phase_unit * (2 * math.pi)

        return amplitudes, orientations, phases

    # -- Sobol local perturbation ---------------------------------------------

    def _sobol_perturb(
        self,
        centers: Tensor,  # [R', D] in [0,1]
        sigma: float,
        P: int,
        device: torch.device,
        seed: int = 0,
    ) -> Tensor:           # [R'*P, D] in [0,1]
        """Quasi-random perturbation: Sobol offsets scaled by sigma."""
        R, D = centers.shape
        N = R * P

        # Generate Sobol points in [0,1]^D (seed varies per cycle/round)
        sampler = StratifiedSobolSampler(
            dimensionality=D, scramble=True, seed=seed,
        )
        sobol_raw = sampler.sample(N)  # [N, D] numpy, in [0,1]
        sobol_t = torch.from_numpy(sobol_raw).float().to(device)

        # Map to symmetric offsets: [0,1] -> [-sigma, +sigma]
        offsets = (sobol_t - 0.5) * 2.0 * sigma  # [N, D]

        # Expand centers: [R', D] -> [R'*P, D]
        centers_expanded = (
            centers.unsqueeze(1).expand(R, P, D).reshape(N, D)
        )

        # Perturb and clamp
        perturbed = (centers_expanded + offsets).clamp(0.0, 1.0)

        # Wrap angular dimensions (orientations, phases) instead of clamping:
        # Indices CK..3CK are orientation and phase, which are periodic.
        CK = D // 3
        perturbed[:, CK:] = (centers_expanded[:, CK:] + offsets[:, CK:]) % 1.0

        return perturbed

    # -- FFT extraction -------------------------------------------------------

    def _extract_params(
        self, grids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract top-K Fourier modes from IC grids via 2D FFT.

        For each (batch, channel), finds the K highest-energy peaks in the
        positive half-plane of the frequency spectrum (avoiding conjugate
        duplicates), then extracts amplitude, orientation, and phase.

        Args:
            grids: [B, C, H, W] IC grids.

        Returns:
            (amplitudes [B,C,K], orientations [B,C,K], phases [B,C,K])
        """
        B, C, H, W = grids.shape
        K = self.K
        device = grids.device

        # Remove DC per (b, c) and undo 1/sqrt(K) normalization
        centered = grids - grids.mean(dim=(-2, -1), keepdim=True)
        centered = centered * math.sqrt(K)

        # Batch FFT [B, C, H, W]
        fft_full = torch.fft.fft2(centered)
        magnitude = fft_full.abs()

        # Mask conjugate duplicates: keep positive half-plane only
        mask = torch.ones(H, W, device=device, dtype=torch.bool)
        mask[H // 2 + 1 :, :] = False        # negative ky (first axis)
        mask[0, 0] = False                    # DC
        mask[0, W // 2 + 1 :] = False        # ky=0, negative kx
        if H % 2 == 0:
            mask[H // 2, W // 2 + 1 :] = False  # Nyquist ambiguity

        magnitude = magnitude * mask[None, None]

        # Find top-K peaks per (b, c)
        mag_flat = magnitude.reshape(B * C, H * W)
        _, peak_indices = mag_flat.topk(K, dim=1)  # [B*C, K]

        peak_k1 = peak_indices // W
        peak_k2 = peak_indices % W

        # Center frequency coordinates (map indices > N//2 to negatives)
        k1_c = torch.where(
            peak_k1 > H // 2, peak_k1.float() - H, peak_k1.float()
        )
        k2_c = torch.where(
            peak_k2 > W // 2, peak_k2.float() - W, peak_k2.float()
        )

        # Gather FFT values at peak positions
        fft_flat = fft_full.reshape(B * C, H * W)
        fft_at_peaks = fft_flat.gather(1, peak_indices)  # [B*C, K]

        # Amplitude: 2*|DFT|/(H*W) because we only look at one conjugate half
        amplitudes = 2 * fft_at_peaks.abs() / (H * W)

        # Orientation: atan2(k2, k1) matching FourierICGenerator convention
        orientations = torch.atan2(k2_c, k1_c) % (2 * math.pi)

        # Phase: angle(DFT) at the peak
        phases = fft_at_peaks.angle() % (2 * math.pi)

        return (
            amplitudes.reshape(B, C, K),
            orientations.reshape(B, C, K),
            phases.reshape(B, C, K),
        )

    # -- Reconstruction -------------------------------------------------------

    def _reconstruct(
        self,
        frequencies: Tensor,    # [B, C, K]
        amplitudes: Tensor,     # [B, C, K]
        orientations: Tensor,   # [B, C, K]
        phases: Tensor,         # [B, C, K]
        grid_size: int,
    ) -> Tensor:
        """Reconstruct IC grids from Fourier parameters.

        Matches FourierICGenerator's formula exactly:
            IC = dc_offset + (1/sqrt(K)) * sum_k A_k * cos(2pi*f_k*proj/G + phi_k)

        Returns: [B, C, H, W] clamped to [0, 1].
        """
        B, C, K = frequencies.shape
        G = grid_size
        device = frequencies.device

        x = torch.arange(G, device=device, dtype=torch.float32)
        gx, gy = torch.meshgrid(x, x, indexing="ij")
        gx = gx[None, None, None]  # [1, 1, 1, G, G]
        gy = gy[None, None, None]

        f = frequencies[..., None, None]       # [B, C, K, 1, 1]
        A = amplitudes[..., None, None]
        theta = orientations[..., None, None]
        phi = phases[..., None, None]

        proj = gx * torch.cos(theta) + gy * torch.sin(theta)
        waves = A * torch.cos(2 * math.pi * f * proj / G + phi)

        field = waves.sum(dim=2) / math.sqrt(K) + self.dc_offset
        return field.clamp(0, 1)
