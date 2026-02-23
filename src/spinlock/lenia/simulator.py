"""Batched GPU Lenia CA simulator via FFT convolution."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import torch

from .params import LeniaParams

logger = logging.getLogger(__name__)


# =============================================================================
# Kernel builders (abstract factory — kept for non-Gaussian kernel types)
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
# Batched kernel builder (Gaussian-only, vectorized)
# =============================================================================


def build_kernel_ffts_batched(
    radii: torch.Tensor,
    grid_size: int,
    dist_grid: torch.Tensor,
) -> torch.Tensor:
    """Build Gaussian kernel FFTs for an entire batch in one shot.

    Replaces the B×C Python loop with a single batched computation:
    distance grid broadcast against per-sample sigma, one exp(), one rfft2().

    Args:
        radii:     [B, C] kernel radii per sample per channel.
        grid_size: Spatial grid size (H = W = grid_size).
        dist_grid: [H, W] precomputed periodic distance grid.

    Returns:
        [B, C, H, W//2+1] complex64 — kernel FFTs ready for convolution.
    """
    B, C = radii.shape
    sigma = radii / 3.0  # [B, C]

    # Broadcast: sigma [B, C, 1, 1] vs dist_grid [1, 1, H, W]
    sigma_4d = sigma[:, :, None, None]
    r = dist_grid[None, None, :, :]  # [1, 1, H, W]

    # Gaussian kernel: exp(-0.5 * (r/sigma)^2), zeroed outside radius
    kernel = torch.exp(-0.5 * (r / sigma_4d) ** 2)  # [B, C, H, W]
    radius_4d = radii[:, :, None, None]
    kernel = torch.where(r <= radius_4d, kernel, torch.zeros_like(kernel))

    # Normalize each kernel to sum=1
    total = kernel.sum(dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
    total = total.clamp(min=1e-12)  # avoid division by zero
    kernel = kernel / total

    return torch.fft.rfft2(kernel)  # [B, C, H, W//2+1]


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
        self._dist_grid: Optional[torch.Tensor] = None

        # Try to compile the hot path; fall back to eager on failure.
        # Suppress the "complex operators" warning — Lenia uses FFTs which are
        # inherently complex-valued; TorchInductor falls back to eager for those
        # ops while still accelerating the real-valued portions of the graph.
        # The filter must persist (no context manager) because torch.compile uses
        # lazy compilation: lowering happens on first call, not at compile() time.
        self._compiled_step = self._step
        self._compiled_kernel_builder = build_kernel_ffts_batched
        if torch.cuda.is_available():
            try:
                warnings.filterwarnings(
                    "ignore",
                    message=".*complex operators.*",
                    category=UserWarning,
                    module=r"torch\._inductor",
                )
                self._compiled_step = torch.compile(self._step)
                self._compiled_kernel_builder = torch.compile(build_kernel_ffts_batched)
                logger.info("LeniaSimulator: torch.compile enabled for _step and kernel builder")
            except Exception as e:
                logger.info(f"LeniaSimulator: torch.compile unavailable ({e}), using eager mode")

    def _get_dist_grid(self) -> torch.Tensor:
        """Lazy-cached periodic distance grid [H, W].

        Computed once, reused across all batches. Uses toroidal wrapping
        so distances respect the periodic boundary conditions.
        """
        if self._dist_grid is not None:
            return self._dist_grid

        H = W = self.grid_size
        ys = torch.arange(H, device=self.device, dtype=torch.float32)
        xs = torch.arange(W, device=self.device, dtype=torch.float32)
        ys = torch.where(ys > H / 2, ys - H, ys)
        xs = torch.where(xs > W / 2, xs - W, xs)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self._dist_grid = torch.sqrt(yy ** 2 + xx ** 2)
        return self._dist_grid

    @torch.no_grad()
    def rollout_batch(
        self,
        ics: torch.Tensor,              # [B, C, H, W]
        params_list: List[LeniaParams],
        num_timesteps: int = 256,
    ) -> torch.Tensor:
        """Run Lenia for all B samples in parallel.

        Backward-compatible entry point. Extracts tensors from params_list,
        builds kernel FFTs, and delegates to rollout_batch_from_tensors.

        Returns:
            Trajectories [B, T, C, H, W] float32 ∈ [0,1].
            T = num_timesteps (the initial condition is NOT included).
        """
        B, C, H, W = ics.shape
        assert len(params_list) == B, f"Expected {B} param sets, got {len(params_list)}"
        assert H == self.grid_size and W == self.grid_size

        radii, growth_mu, growth_sigma, dt, coupling = self._params_list_to_tensors(params_list)

        # Check if all kernels are Gaussian (can use batched path)
        all_gaussian = all(p.kernel_type == "gaussian" for p in params_list)
        if all_gaussian:
            dist_grid = self._get_dist_grid()
            kernel_ffts = self._compiled_kernel_builder(radii, self.grid_size, dist_grid)
        else:
            kernel_ffts = self._build_kernel_ffts_legacy(params_list, C, H, W)

        return self.rollout_batch_from_tensors(
            ics, kernel_ffts, coupling, growth_mu, growth_sigma, dt, num_timesteps
        )

    @torch.no_grad()
    def rollout_batch_from_tensors(
        self,
        ics: torch.Tensor,             # [B, C, H, W]
        kernel_ffts: torch.Tensor,     # [B, C, H, W//2+1] complex
        coupling: torch.Tensor,        # [B, C, C]
        growth_mu: torch.Tensor,       # [B, C]
        growth_sigma: torch.Tensor,    # [B, C]
        dt: torch.Tensor,              # [B]
        num_timesteps: int = 256,
    ) -> torch.Tensor:
        """Run Lenia with precomputed kernel FFTs and pre-extracted tensors.

        This is the fast path: no Python loops for param extraction, no
        kernel rebuilding. All tensors are pre-shaped before the timestep
        loop to eliminate per-step reshape overhead.

        Returns:
            Trajectories [B, T, C, H, W] float32 ∈ [0,1].
        """
        B, C, H, W = ics.shape
        state = ics.to(self.device).float()

        # Pre-shape for broadcast (done once, not 256 times)
        mu = growth_mu[:, :, None, None]        # [B, C, 1, 1]
        sigma = growth_sigma[:, :, None, None]  # [B, C, 1, 1]
        dt_view = dt[:, None, None, None]       # [B, 1, 1, 1]

        # Pre-allocate trajectory tensor (avoids list appends + stack copy)
        traj = torch.empty(B, num_timesteps, C, H, W, device=self.device, dtype=torch.float32)

        step_fn = self._compiled_step
        for t in range(num_timesteps):
            state = step_fn(state, kernel_ffts, coupling, mu, sigma, dt_view)
            traj[:, t] = state

        return traj

    def _params_list_to_tensors(
        self,
        params_list: List[LeniaParams],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract LeniaParams list into batched tensors on device.

        Returns:
            radii [B,C], growth_mu [B,C], growth_sigma [B,C], dt [B], coupling [B,C,C]
        """
        radii = torch.tensor(
            [p.kernel_radius for p in params_list],
            device=self.device, dtype=torch.float32,
        )
        growth_mu = torch.tensor(
            [p.growth_mu for p in params_list],
            device=self.device, dtype=torch.float32,
        )
        growth_sigma = torch.tensor(
            [p.growth_sigma for p in params_list],
            device=self.device, dtype=torch.float32,
        )
        dt = torch.tensor(
            [p.dt for p in params_list],
            device=self.device, dtype=torch.float32,
        )
        coupling = torch.tensor(
            [p.coupling for p in params_list],
            device=self.device, dtype=torch.float32,
        )
        return radii, growth_mu, growth_sigma, dt, coupling

    def _build_kernel_ffts_legacy(
        self,
        params_list: List[LeniaParams],
        C: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Legacy per-sample per-channel kernel FFT builder.

        Used as fallback for non-Gaussian kernel types.

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

    @staticmethod
    def _step(
        state: torch.Tensor,            # [B, C, H, W]
        kernel_fft: torch.Tensor,       # [B, C, H, W//2+1] complex
        coupling: torch.Tensor,         # [B, C, C]
        growth_mu: torch.Tensor,        # [B, C, 1, 1]  (pre-shaped)
        growth_sigma: torch.Tensor,     # [B, C, 1, 1]  (pre-shaped)
        dt: torch.Tensor,               # [B, 1, 1, 1]  (pre-shaped)
    ) -> torch.Tensor:
        """Single Lenia time step (fully batched).

        Accepts pre-shaped mu/sigma/dt tensors to avoid per-step reshape overhead.

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
        U_flat = U.view(B, C, -1)                   # [B, C, H*W]
        U_mixed = torch.bmm(coupling, U_flat)        # [B, C, H*W]
        U_mixed = U_mixed.view(B, C, H, W)          # [B, C, H, W]

        # Step 5: growth function G ∈ [-1, 1]
        G = 2.0 * torch.exp(-((U_mixed - growth_mu) / growth_sigma) ** 2) - 1.0

        # Step 6: Euler update
        return torch.clamp(state + dt * G, 0.0, 1.0)
