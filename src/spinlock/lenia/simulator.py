"""Batched GPU Lenia CA simulator via FFT convolution."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from .params import LeniaParams


# =============================================================================
# Kernel builders (abstract factory)
# =============================================================================


class LeniaKernelBuilder(ABC):
    """Abstract kernel factory — one subclass per kernel type."""

    @abstractmethod
    def build_kernel_fft(
        self,
        radius: float,
        grid_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build kernel in rfft2 format.

        Returns:
            Complex tensor [H, W//2+1] in torch.rfft2 layout.
        """
        ...


class GaussianKernelBuilder(LeniaKernelBuilder):
    """K(r) = exp(-r² / (2*(R/3)²)), normalized to sum=1.

    The standard deviation R/3 means the kernel drops to ~1% at radius R,
    providing a smooth spatial coupling over the full support radius.
    """

    def build_kernel_fft(
        self,
        radius: float,
        grid_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        H = W = grid_size
        sigma = radius / 3.0

        # Distance grid with periodic wrapping (toroidal topology)
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        # Wrap distances to [-H/2, H/2] and [-W/2, W/2]
        ys = torch.where(ys > H / 2, ys - H, ys)
        xs = torch.where(xs > W / 2, xs - W, xs)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt(yy ** 2 + xx ** 2)

        # Gaussian kernel, clamp outside radius
        kernel = torch.exp(-0.5 * (r / sigma) ** 2)
        kernel = torch.where(r <= radius, kernel, torch.zeros_like(kernel))

        # Normalize so convolution output is in [0,1] when input is in [0,1]
        total = kernel.sum()
        if total > 0:
            kernel = kernel / total

        return torch.fft.rfft2(kernel)  # [H, W//2+1] complex


class PolynomialKernelBuilder(LeniaKernelBuilder):
    """K(r) = (1 - (r/R)^α)^β for r ≤ R, else 0. STUB."""

    def build_kernel_fft(
        self,
        radius: float,
        grid_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError("Polynomial kernel: stub — not yet implemented")


class ExponentialKernelBuilder(LeniaKernelBuilder):
    """K(r) = exp(-r/λ). STUB."""

    def build_kernel_fft(
        self,
        radius: float,
        grid_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError("Exponential kernel: stub — not yet implemented")


KERNEL_BUILDERS: Dict[str, Type[LeniaKernelBuilder]] = {
    "gaussian": GaussianKernelBuilder,
    "polynomial": PolynomialKernelBuilder,
    "exponential": ExponentialKernelBuilder,
}


# =============================================================================
# Main simulator
# =============================================================================


class LeniaSimulator:
    """Batched GPU Lenia simulation.

    All samples in a batch are stepped in parallel. Kernel FFTs are
    precomputed once per sample per batch (one per channel), then reused
    across all T timesteps.
    """

    def __init__(self, grid_size: int = 64, device: str = "cuda"):
        self.grid_size = grid_size
        self.device = torch.device(device)

    @torch.no_grad()
    def rollout_batch(
        self,
        ics: torch.Tensor,              # [B, C, H, W]
        params_list: List[LeniaParams],
        num_timesteps: int = 256,
    ) -> torch.Tensor:
        """Run Lenia for all B samples in parallel.

        Returns:
            Trajectories [B, T, C, H, W] float32 ∈ [0,1].
            T = num_timesteps (the initial condition is NOT included).
        """
        B, C, H, W = ics.shape
        assert len(params_list) == B, f"Expected {B} param sets, got {len(params_list)}"
        assert H == self.grid_size and W == self.grid_size

        state = ics.to(self.device).float()

        # Precompute kernel FFTs: [B, C, H, W//2+1] complex
        kernel_ffts = self._build_kernel_ffts(params_list, C, H, W)

        # Stack coupling, growth_mu, growth_sigma into batched tensors
        coupling = torch.tensor(
            [p.coupling for p in params_list],
            device=self.device, dtype=torch.float32
        )  # [B, C, C]
        growth_mu = torch.tensor(
            [p.growth_mu for p in params_list],
            device=self.device, dtype=torch.float32
        )  # [B, C]
        growth_sigma = torch.tensor(
            [p.growth_sigma for p in params_list],
            device=self.device, dtype=torch.float32
        )  # [B, C]
        dt = torch.tensor(
            [p.dt for p in params_list],
            device=self.device, dtype=torch.float32
        )  # [B]

        # Collect trajectory frames
        frames: List[torch.Tensor] = []
        for _ in range(num_timesteps):
            state = self._step(state, kernel_ffts, coupling, growth_mu, growth_sigma, dt)
            frames.append(state)

        return torch.stack(frames, dim=1)  # [B, T, C, H, W]

    def _build_kernel_ffts(
        self,
        params_list: List[LeniaParams],
        C: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Precompute per-sample per-channel kernel FFTs.

        Returns:
            [B, C, H, W//2+1] complex64
        """
        B = len(params_list)
        ffts = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.complex64, device=self.device)
        for b, params in enumerate(params_list):
            builder = KERNEL_BUILDERS[params.kernel_type]()
            for c in range(C):
                ffts[b, c] = builder.build_kernel_fft(
                    radius=params.kernel_radius[c],
                    grid_size=self.grid_size,
                    device=self.device,
                )
        return ffts

    def _step(
        self,
        state: torch.Tensor,            # [B, C, H, W]
        kernel_fft: torch.Tensor,       # [B, C, H, W//2+1] complex
        coupling: torch.Tensor,         # [B, C, C]
        growth_mu: torch.Tensor,        # [B, C]
        growth_sigma: torch.Tensor,     # [B, C]
        dt: torch.Tensor,               # [B]
    ) -> torch.Tensor:
        """Single Lenia time step (fully batched).

        Algorithm:
            1. FFT each channel of state
            2. Multiply pointwise with kernel FFT (circular convolution)
            3. IFFT to get per-channel neighborhood sums U_i ∈ [0,1]
            4. Mix channels via coupling matrix: U = coupling @ U_flat
            5. Apply growth function G = 2*exp(-((U-μ)/σ)²) - 1  ∈ [-1, 1]
            6. Euler step: new_state = clamp(state + dt * G, 0, 1)
        """
        B, C, H, W = state.shape

        # Step 1+2+3: convolution per channel via FFT
        state_fft = torch.fft.rfft2(state)          # [B, C, H, W//2+1]
        conv_fft = state_fft * kernel_fft           # [B, C, H, W//2+1]
        U = torch.fft.irfft2(conv_fft, s=(H, W))   # [B, C, H, W]

        # Step 4: cross-channel coupling  [B, C, H, W]
        # coupling: [B, C_out, C_in]  U: [B, C_in, H, W]
        # → U_mixed[b, c_out, h, w] = sum_c_in coupling[b, c_out, c_in] * U[b, c_in, h, w]
        U_flat = U.view(B, C, -1)                   # [B, C, H*W]
        U_mixed = torch.bmm(coupling, U_flat)        # [B, C, H*W]
        U_mixed = U_mixed.view(B, C, H, W)          # [B, C, H, W]

        # Step 5: growth function G ∈ [-1, 1]
        # mu, sigma: [B, C] → reshape for broadcast to [B, C, 1, 1]
        mu = growth_mu[:, :, None, None]
        sigma = growth_sigma[:, :, None, None]
        G = 2.0 * torch.exp(-((U_mixed - mu) / sigma) ** 2) - 1.0  # [B, C, H, W]

        # Step 6: Euler update with dt [B] → [B, 1, 1, 1]
        dt_view = dt[:, None, None, None]
        return torch.clamp(state + dt_view * G, 0.0, 1.0)
