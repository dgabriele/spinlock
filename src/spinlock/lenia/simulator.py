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
# Multi-ring kernel builder (V3: variable B rings, multiple shell types)
# =============================================================================


def _kernel_shell(
    type_name: str,
    x: torch.Tensor,
    peak: torch.Tensor,
    width: torch.Tensor,
) -> torch.Tensor:
    """Compute kernel shell value for a given type.

    Args:
        type_name: "gaussian", "polynomial", or "step".
        x: Normalized distance tensor [B, C, H, W].
        peak: Ring center in normalized space [B, 1, 1, 1].
        width: Ring width [B, 1, 1, 1].

    Returns:
        Shell values [B, C, H, W].
    """
    if type_name == "gaussian":
        return torch.exp(-((x - peak) ** 2) / (2 * width ** 2))
    elif type_name == "polynomial":
        # Compact support, quartic smoothness: max(0, 1 - ((x-peak)/w)²)^4
        return torch.clamp(1.0 - ((x - peak).abs() / width) ** 2, min=0.0) ** 4
    elif type_name == "step":
        # Rectangular rings
        return ((x - peak).abs() < width).float()
    else:
        raise ValueError(f"Unknown kernel shell type: {type_name}")


def build_multiring_kernel_ffts_batched(
    radii: torch.Tensor,
    grid_size: int,
    dist_grid: torch.Tensor,
    kernel_rank: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    kernel_type_ids: Optional[torch.Tensor] = None,
    kernel_types_list: Optional[List[str]] = None,
) -> torch.Tensor:
    """Build multi-ring kernel FFTs for a batch.

    Generalizes build_kernel_ffts_batched to support:
    - Variable number of concentric rings (kernel_rank B ∈ [1, max_rings])
    - Per-channel beta weights for ring mixing
    - Multiple kernel shell types (gaussian, polynomial, step)

    When kernel_rank is None or beta is None, falls back to the single-ring
    Gaussian builder for backward compatibility.

    Args:
        radii:            [B, C] kernel radii per sample per channel.
        grid_size:        Spatial grid size (H = W).
        dist_grid:        [H, W] precomputed periodic distance grid.
        kernel_rank:      [B] integer, number of active rings per sample.
        beta:             [B, C, max_rings] normalized ring weights.
        kernel_type_ids:  [B] integer kernel type indices (0=gauss, 1=poly, 2=step).
        kernel_types_list: List mapping type index to name.

    Returns:
        [B, C, H, W//2+1] complex64 — kernel FFTs.
    """
    # Fast path: no multi-ring params → delegate to single-Gaussian builder
    if kernel_rank is None or beta is None:
        return build_kernel_ffts_batched(radii, grid_size, dist_grid)

    B, C = radii.shape
    H = W = grid_size
    max_rings = beta.shape[2]
    device = radii.device

    # Normalized distance: x = dist / R for each sample and channel
    x = dist_grid[None, None] / radii[:, :, None, None]  # [B, C, H, W]
    kr = kernel_rank[:, None, None, None].float()  # [B, 1, 1, 1]

    kernel = torch.zeros(B, C, H, W, device=device)

    # Determine if batch is homogeneous in kernel type
    all_same_type = (
        kernel_type_ids is None or kernel_type_ids.unique().numel() == 1
    )

    if all_same_type:
        # Homogeneous batch — one shell function for all samples
        type_name = "gaussian"
        if kernel_type_ids is not None and kernel_types_list is not None:
            type_name = kernel_types_list[kernel_type_ids[0].item()]

        for ring in range(max_rings):
            peak = (ring + 0.5) / kr
            width = 1.0 / (3.0 * kr)
            shell = _kernel_shell(type_name, x, peak, width)
            kernel = kernel + beta[:, :, ring, None, None] * shell
    else:
        # Heterogeneous batch — sort and partition by kernel type
        for type_idx, type_name in enumerate(kernel_types_list or ["gaussian"]):
            type_mask = kernel_type_ids == type_idx
            if not type_mask.any():
                continue
            idx = type_mask.nonzero(as_tuple=True)[0]
            sub_x = x[idx]
            sub_kr = kr[idx]
            sub_beta = beta[idx]
            sub_kernel = torch.zeros_like(sub_x)

            for ring in range(max_rings):
                peak = (ring + 0.5) / sub_kr
                width = 1.0 / (3.0 * sub_kr)
                shell = _kernel_shell(type_name, sub_x, peak, width)
                sub_kernel = sub_kernel + sub_beta[:, :, ring, None, None] * shell

            kernel[idx] = sub_kernel

    # Zero outside radius, normalize, FFT
    kernel = torch.where(x <= 1.0, kernel, torch.zeros_like(kernel))
    total = kernel.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    kernel = kernel / total

    return torch.fft.rfft2(kernel)


# =============================================================================
# Growth function constants
# =============================================================================

GROWTH_GAUSSIAN = 0
GROWTH_POLYNOMIAL = 1
GROWTH_STEP = 2


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
            # Enable TF32 for faster float32 matmul on Ampere+ GPUs
            torch.set_float32_matmul_precision("high")
            try:
                warnings.filterwarnings(
                    "ignore",
                    message=".*complex operators.*",
                    category=UserWarning,
                    module=r"torch\._inductor",
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*TensorFloat32.*",
                    category=UserWarning,
                )
                # Suppress inductor warnings about SM count / autotune
                logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
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
        growth_type: Optional[torch.Tensor] = None,  # [B] long or None
    ) -> torch.Tensor:
        """Run Lenia with precomputed kernel FFTs and pre-extracted tensors.

        This is the fast path: no Python loops for param extraction, no
        kernel rebuilding. All tensors are pre-shaped before the timestep
        loop to eliminate per-step reshape overhead.

        Args:
            growth_type: Per-sample growth function index (0=gaussian,
                1=polynomial, 2=step). None = all gaussian (V2 fast path).

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
            state = step_fn(state, kernel_ffts, coupling, mu, sigma, dt_view, growth_type)
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
        growth_type: Optional[torch.Tensor] = None,  # [B] long or None
    ) -> torch.Tensor:
        """Single Lenia time step (fully batched).

        Accepts pre-shaped mu/sigma/dt tensors to avoid per-step reshape overhead.

        Algorithm:
            1. FFT each channel of state
            2. Multiply pointwise with kernel FFT (circular convolution)
            3. IFFT to get per-channel neighborhood sums U_i ∈ [0,1]
            4. Mix channels via coupling matrix: U = coupling @ U_flat
            5. Apply growth function G ∈ [-1, 1] (type-dispatched for V3)
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
        if growth_type is None:
            # V2 fast path: all Gaussian
            G = 2.0 * torch.exp(-((U_mixed - growth_mu) / growth_sigma) ** 2) - 1.0
        else:
            # V3: compute all three growth functions and select per-sample.
            # Computing all three avoids dynamic indexing / graph breaks, at
            # ~3x cost on the growth step (negligible vs FFT convolution).
            z = (U_mixed - growth_mu) / growth_sigma
            G_gauss = 2.0 * torch.exp(-(z ** 2)) - 1.0
            G_poly = 2.0 * torch.clamp(1.0 - z ** 2, min=0.0) ** 2 - 1.0
            # Smoothed step: tanh(α·(1−|z|)) with α=5.  Preserves near-binary
            # growth/decay while keeping G' finite (|G'|_max = α/σ), which is
            # essential for CFL-based substep stability.
            G_step = torch.tanh(5.0 * (1.0 - z.abs()))
            G_all = torch.stack([G_gauss, G_poly, G_step], dim=0)  # [3, B, C, H, W]
            gt_idx = growth_type.view(1, -1, 1, 1, 1).expand(1, -1, C, H, W)
            G = G_all.gather(0, gt_idx).squeeze(0)  # [B, C, H, W]

        # Step 6: Euler update
        return torch.clamp(state + dt * G, 0.0, 1.0)
