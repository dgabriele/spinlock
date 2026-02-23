"""LeniaReplayAdapter: VQTokenizer-compatible trajectory replayer for Lenia.

Adapts LeniaSimulator for on-the-fly trajectory generation during VQ tokenizer
training. Implements the same interface as CNOReplayer (rollout, from_config)
plus a batched rollout_batch() that exploits Lenia's shared-simulator property
for O(B) speedup over the per-sample loop.

Key difference from CNOReplayer: CNO constructs a different neural operator per
sample (expensive per-sample cache lookups). Lenia shares ONE simulator — only
parameters differ. This means the entire batch can be vectorized.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml

from .params import sobol_batch_to_tensors
from .simulator import LeniaSimulator

logger = logging.getLogger(__name__)


class LeniaReplayAdapter:
    """VQTokenizer-compatible replayer wrapping LeniaSimulator.

    Provides both the per-sample ``rollout()`` interface (matching CNOReplayer)
    and a fully vectorized ``rollout_batch()`` for throughput-critical paths.

    GPU memory is controlled via ``max_gpu_batch``: trajectories are generated
    in sub-batches and moved to CPU incrementally.
    """

    def __init__(
        self,
        n_channels: int,
        grid_size: int,
        kernel_type: str = "gaussian",
        device: str = "cuda",
        max_gpu_batch: int = 16,
    ):
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.kernel_type = kernel_type
        self.device = torch.device(device)
        self.max_gpu_batch = max_gpu_batch

        self.simulator = LeniaSimulator(grid_size=grid_size, device=device)

    @classmethod
    def from_config(
        cls,
        config_path: str,
        device: str = "cuda",
        cache_size: int = 0,
    ) -> LeniaReplayAdapter:
        """Create adapter from Lenia generation config YAML.

        Args:
            config_path: Path to config YAML used for dataset generation.
            device: Computation device.
            cache_size: Ignored (kept for interface parity with CNOReplayer).

        Returns:
            Configured LeniaReplayAdapter.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        sim_cfg = config.get("simulation", {})
        lenia_cfg = sim_cfg.get("lenia", {})

        return cls(
            n_channels=sim_cfg.get("n_channels", 3),
            grid_size=sim_cfg.get("grid_size", 64),
            kernel_type=lenia_cfg.get("kernel_type", "gaussian"),
            device=device,
        )

    def _to_numpy_batch(
        self,
        params_vector: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Normalize params to [B, D] numpy array."""
        if isinstance(params_vector, torch.Tensor):
            params_vector = params_vector.cpu().numpy()
        if params_vector.ndim == 1:
            params_vector = params_vector[None, :]
        return params_vector

    def _build_lenia_tensors(
        self,
        params_batch: np.ndarray,
    ) -> tuple:
        """Convert Sobol batch → kernel FFTs + physics tensors on device."""
        radii, growth_mu, growth_sigma, dt, coupling = sobol_batch_to_tensors(
            params_batch, self.n_channels, self.device,
        )
        dist_grid = self.simulator._get_dist_grid()
        kernel_ffts = self.simulator._compiled_kernel_builder(
            radii, self.grid_size, dist_grid,
        )
        return kernel_ffts, coupling, growth_mu, growth_sigma, dt

    def _simulate(
        self,
        ics: torch.Tensor,
        kernel_ffts: torch.Tensor,
        coupling: torch.Tensor,
        growth_mu: torch.Tensor,
        growth_sigma: torch.Tensor,
        dt: torch.Tensor,
        timesteps: int,
        return_all_steps: bool,
    ) -> torch.Tensor:
        """Run simulator and format output."""
        traj = self.simulator.rollout_batch_from_tensors(
            ics, kernel_ffts, coupling, growth_mu, growth_sigma, dt, timesteps,
        )  # [B, T, C, H, W]

        if return_all_steps:
            return torch.cat([ics.unsqueeze(1), traj], dim=1)  # [B, T+1, C, H, W]
        return traj[:, -1]  # [B, C, H, W]

    # ── Per-sample interface (CNOReplayer compat) ─────────────

    def rollout(
        self,
        params_vector: Union[np.ndarray, torch.Tensor],
        ic: torch.Tensor,
        timesteps: int,
        num_realizations: int = 1,
        seed: Optional[int] = None,
        return_all_steps: bool = True,
    ) -> torch.Tensor:
        """Single-sample rollout matching CNOReplayer.rollout() interface.

        Args:
            params_vector: [D,] Sobol parameter vector.
            ic: [C, H, W] or [1, C, H, W] initial condition.
            timesteps: Number of simulation steps.
            num_realizations: Ignored (Lenia is deterministic given IC+params).
            seed: Ignored (deterministic).
            return_all_steps: If True, prepend IC → [1, T+1, C, H, W].

        Returns:
            Trajectory tensor on adapter device.
        """
        if ic.dim() == 3:
            ic = ic.unsqueeze(0)
        ic = ic.to(self.device)

        params_np = self._to_numpy_batch(params_vector)
        kernel_ffts, coupling, growth_mu, growth_sigma, dt = self._build_lenia_tensors(
            params_np,
        )
        return self._simulate(
            ic, kernel_ffts, coupling, growth_mu, growth_sigma, dt,
            timesteps, return_all_steps,
        )

    # ── Batched interface (throughput path) ───────────────────

    def rollout_batch(
        self,
        params_batch: Union[np.ndarray, torch.Tensor],
        ics: torch.Tensor,
        timesteps: int,
        return_all_steps: bool = True,
    ) -> torch.Tensor:
        """Fully vectorized batch rollout with GPU memory management.

        Processes in sub-batches of ``max_gpu_batch`` to avoid OOM on large
        batches (128 samples × 256 timesteps × 3ch × 64² ≈ 12 GB).

        Args:
            params_batch: [B, D] Sobol parameters.
            ics: [B, C, H, W] initial conditions.
            timesteps: Number of simulation steps.
            return_all_steps: If True, prepend IC → [B, T+1, C, H, W].

        Returns:
            Trajectories [B, T+1, C, H, W] on CPU.
        """
        params_np = self._to_numpy_batch(params_batch)
        B = params_np.shape[0]

        if B <= self.max_gpu_batch:
            # Single batch — no sub-batching needed
            ics_dev = ics.to(self.device)
            kernel_ffts, coupling, growth_mu, growth_sigma, dt = (
                self._build_lenia_tensors(params_np)
            )
            result = self._simulate(
                ics_dev, kernel_ffts, coupling, growth_mu, growth_sigma, dt,
                timesteps, return_all_steps,
            )
            return result.cpu()

        # Sub-batch to control GPU memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        chunks = []
        for start in range(0, B, self.max_gpu_batch):
            end = min(start + self.max_gpu_batch, B)
            sub_ics = ics[start:end].to(self.device)
            sub_kfft, sub_coup, sub_gmu, sub_gsig, sub_dt = (
                self._build_lenia_tensors(params_np[start:end])
            )
            sub_result = self._simulate(
                sub_ics, sub_kfft, sub_coup, sub_gmu, sub_gsig, sub_dt,
                timesteps, return_all_steps,
            )
            chunks.append(sub_result.cpu())

        return torch.cat(chunks, dim=0)
