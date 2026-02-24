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
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml

from .params import (
    DEFAULT_RANGES,
    LeniaBatchTensors,
    LeniaParamRanges,
    sobol_batch_to_tensors,
)
from .simulator import (
    LeniaSimulator,
    build_multiring_kernel_ffts_batched,
)

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
        substeps: int = 1,
        param_ranges: Optional[LeniaParamRanges] = None,
    ):
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.kernel_type = kernel_type
        self.device = torch.device(device)
        self.max_gpu_batch = max_gpu_batch
        self.substeps = substeps
        self.param_ranges = param_ranges

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

        grid_size = sim_cfg.get("grid_size", 64)
        # Scale max_gpu_batch inversely with pixel count (baseline: 16 @ 64²)
        max_gpu_batch = max(1, 16 * (64 * 64) // (grid_size * grid_size))

        param_ranges = DEFAULT_RANGES

        return cls(
            n_channels=sim_cfg.get("n_channels", 3),
            grid_size=grid_size,
            kernel_type=lenia_cfg.get("kernel_type", "gaussian"),
            device=device,
            max_gpu_batch=max_gpu_batch,
            substeps=lenia_cfg.get("substeps", 1),
            param_ranges=param_ranges,
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
    ) -> LeniaBatchTensors:
        """Convert Sobol batch → LeniaBatchTensors on device."""
        return sobol_batch_to_tensors(
            params_batch, self.n_channels, self.device,
            ranges=self.param_ranges,
        )

    def _build_kernel_ffts(
        self,
        tensors: LeniaBatchTensors,
    ) -> torch.Tensor:
        """Build kernel FFTs from batch tensors.

        Delegates to multi-ring builder when kernel_rank/beta are present,
        otherwise uses the single-Gaussian compiled builder.
        """
        dist_grid = self.simulator._get_dist_grid()

        if tensors.kernel_rank is not None and tensors.beta is not None:
            return build_multiring_kernel_ffts_batched(
                radii=tensors.radii,
                grid_size=self.grid_size,
                dist_grid=dist_grid,
                kernel_rank=tensors.kernel_rank,
                beta=tensors.beta,
                kernel_type_ids=tensors.kernel_type,
                kernel_types_list=(
                    self.param_ranges.kernel_types
                    if self.param_ranges else None
                ),
            )
        else:
            return self.simulator._compiled_kernel_builder(
                tensors.radii, self.grid_size, dist_grid,
            )

    def _compute_adaptive_substeps(
        self,
        dt: torch.Tensor,
        growth_sigma: torch.Tensor,
        growth_type: Optional[torch.Tensor] = None,
    ) -> int:
        """Compute substep count K for CFL stability.

        See LeniaReplayer._compute_adaptive_substeps for full rationale.
        Uses growth-type-aware |G'|_max constants and caps at K=32.
        """
        K = self.substeps
        if self.param_ranges is None:
            return K

        sigma_min = growth_sigma.min(dim=1).values.clamp(min=1e-8)

        if growth_type is None:
            g_prime_const = 1.716
        else:
            has_step = (growth_type == 2).any().item()
            has_poly = (growth_type == 1).any().item()
            if has_step:
                g_prime_const = 5.0
            elif has_poly:
                g_prime_const = 3.079
            else:
                g_prime_const = 1.716

        cfl_max = (dt * g_prime_const / sigma_min).max().item()
        K_cfl = math.ceil(cfl_max / 1.0)
        K = max(K, K_cfl)
        K = min(K, 32)
        return K

    def _simulate(
        self,
        ics: torch.Tensor,
        tensors: LeniaBatchTensors,
        kernel_ffts: torch.Tensor,
        timesteps: int,
        return_all_steps: bool,
    ) -> torch.Tensor:
        """Run simulator and format output.

        When ``substeps > 1``, each visible frame is produced by K Euler
        steps at dt/K instead of one step at dt.  The total simulated time
        per visible frame is unchanged, but the integration is more stable
        (eliminates period-2 oscillations from large dt + sharp growth).

        Memory-efficient: only stores one frame per visible timestep.
        If K is capped (at 32), per-sample dt clamping ensures CFL < 1.0.
        """
        K = self._compute_adaptive_substeps(
            tensors.dt, tensors.growth_sigma, tensors.growth_type,
        )

        if K <= 1:
            traj = self.simulator.rollout_batch_from_tensors(
                ics, kernel_ffts, tensors.coupling,
                tensors.growth_mu, tensors.growth_sigma, tensors.dt, timesteps,
                growth_type=tensors.growth_type,
            )
        else:
            sim_dt = tensors.dt / K

            # Per-sample dt safety clamp when K was capped
            if tensors.growth_type is not None:
                sigma_min = tensors.growth_sigma.min(dim=1).values.clamp(min=1e-8)
                g_prime = torch.full_like(tensors.dt, 1.716)
                g_prime[tensors.growth_type == 1] = 3.079
                g_prime[tensors.growth_type == 2] = 5.0
                dt_safe = sigma_min / g_prime
                sim_dt = torch.minimum(sim_dt, dt_safe)

            # Memory-efficient substep loop
            B, C, H, W = ics.shape
            state = ics.float()
            mu = tensors.growth_mu[:, :, None, None]
            sigma = tensors.growth_sigma[:, :, None, None]
            dt_view = sim_dt[:, None, None, None]

            traj = torch.empty(B, timesteps, C, H, W,
                               device=self.device, dtype=torch.float32)
            step_fn = self.simulator._compiled_step

            for t in range(timesteps):
                for _k in range(K):
                    state = step_fn(state, kernel_ffts, tensors.coupling,
                                    mu, sigma, dt_view, tensors.growth_type)
                traj[:, t] = state

        if return_all_steps:
            return torch.cat([ics.unsqueeze(1), traj], dim=1)
        return traj[:, -1]

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
        tensors = self._build_lenia_tensors(params_np)
        kernel_ffts = self._build_kernel_ffts(tensors)
        return self._simulate(ic, tensors, kernel_ffts, timesteps, return_all_steps)

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
            tensors = self._build_lenia_tensors(params_np)
            kernel_ffts = self._build_kernel_ffts(tensors)
            result = self._simulate(
                ics_dev, tensors, kernel_ffts, timesteps, return_all_steps,
            )
            return result.cpu()

        # Sub-batch to control GPU memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        chunks = []
        for start in range(0, B, self.max_gpu_batch):
            end = min(start + self.max_gpu_batch, B)
            sub_ics = ics[start:end].to(self.device)
            sub_tensors = self._build_lenia_tensors(params_np[start:end])
            sub_kfft = self._build_kernel_ffts(sub_tensors)
            sub_result = self._simulate(
                sub_ics, sub_tensors, sub_kfft, timesteps, return_all_steps,
            )
            chunks.append(sub_result.cpu())

        return torch.cat(chunks, dim=0)
