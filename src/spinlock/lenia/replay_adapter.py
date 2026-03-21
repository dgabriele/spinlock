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
from .perturbations import PerturbationConfig, apply_perturbation, _sample_perturbation_type
from .simulator import (
    LeniaSimulator,
    build_multiring_kernel_ffts_batched,
    simulate_with_early_exit,
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
        compile: bool = True,
    ):
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.kernel_type = kernel_type
        self.device = torch.device(device)
        self.max_gpu_batch = max_gpu_batch
        self.substeps = substeps
        self.param_ranges = param_ranges

        self.simulator = LeniaSimulator(grid_size=grid_size, device=device, compile=compile)

        # Cache of last successful sub-batch size per timestep count.
        # Avoids repeated OOM + empty_cache() cycles when GPU memory is
        # stable across batches (the common case during training).
        self._sub_batch_cache: dict[int, int] = {}

    @classmethod
    def from_config(
        cls,
        config_path: str,
        device: str = "cuda",
        cache_size: int = 0,
        compile: bool = True,
    ) -> LeniaReplayAdapter:
        """Create adapter from Lenia generation config YAML.

        Args:
            config_path: Path to config YAML used for dataset generation.
            device: Computation device.
            cache_size: Ignored (kept for interface parity with CNOReplayer).
            compile: Whether to torch.compile the simulator hot path.

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
        n_ch = sim_cfg.get("n_channels", 3)
        # max_gpu_batch is computed dynamically in _effective_max_gpu_batch()
        # based on actual free GPU memory at rollout time. The static value
        # here serves as an upper bound (generous for large-VRAM GPUs).
        max_gpu_batch = 128

        param_ranges = DEFAULT_RANGES

        return cls(
            n_channels=sim_cfg.get("n_channels", 3),
            grid_size=grid_size,
            kernel_type=lenia_cfg.get("kernel_type", "gaussian"),
            device=device,
            max_gpu_batch=max_gpu_batch,
            substeps=lenia_cfg.get("substeps", 1),
            param_ranges=param_ranges,
            compile=compile,
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
            if return_all_steps:
                return torch.cat([ics.unsqueeze(1), traj], dim=1)
            return traj[:, -1]

        # ── Substep path (CFL-adaptive, the common case for Lenia v1) ──
        sim_dt = tensors.dt / K

        # Per-sample dt safety clamp when K was capped
        if tensors.growth_type is not None:
            sigma_min = tensors.growth_sigma.min(dim=1).values.clamp(min=1e-8)
            g_prime = torch.full_like(tensors.dt, 1.716)
            g_prime[tensors.growth_type == 1] = 3.079
            g_prime[tensors.growth_type == 2] = 5.0
            dt_safe = sigma_min / g_prime
            sim_dt = torch.minimum(sim_dt, dt_safe)

        state = ics.float()
        mu = tensors.growth_mu[:, :, None, None]
        sigma = tensors.growth_sigma[:, :, None, None]
        dt_view = sim_dt[:, None, None, None]

        traj = simulate_with_early_exit(
            state, self.simulator._compiled_step, kernel_ffts, tensors.coupling,
            mu, sigma, dt_view, tensors.growth_type, timesteps, substeps=K,
        )
        if return_all_steps:
            return torch.cat([ics.unsqueeze(1), traj], dim=1)
        return traj[:, -1]

    def _simulate_perturbed(
        self,
        ics: torch.Tensor,
        tensors: LeniaBatchTensors,
        kernel_ffts: torch.Tensor,
        timesteps: int,
        return_all_steps: bool,
        perturbation_config: PerturbationConfig,
        seed: int = 0,
    ) -> torch.Tensor:
        """Simulate with mid-trajectory perturbation.

        1. Simulate naturally for injection_step frames
        2. Apply perturbation to the state
        3. Continue simulation for remaining frames
        4. Return full trajectory (pre + post perturbation)
        """
        injection_step = max(1, int(timesteps * perturbation_config.injection_fraction))
        remaining_steps = timesteps - injection_step

        # Phase 1: simulate naturally up to injection point
        pre_traj = self._simulate(
            ics, tensors, kernel_ffts, injection_step, return_all_steps=True,
        )  # [B, injection_step+1, C, H, W] (IC prepended)

        # State at injection point (last frame)
        state = pre_traj[:, -1].clone()

        # Apply perturbation
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        ptype = _sample_perturbation_type(perturbation_config, rng, self.device)
        state = apply_perturbation(state, ptype, tensors.radii, perturbation_config, rng)

        # Phase 2: simulate from perturbed state
        if remaining_steps > 0:
            post_traj = self._simulate(
                state, tensors, kernel_ffts, remaining_steps, return_all_steps=True,
            )  # [B, remaining_steps+1, C, H, W] (perturbed state prepended)

            if return_all_steps:
                # pre_traj: [B, injection_step+1, ...] includes IC
                # post_traj: [B, remaining_steps+1, ...] includes perturbed state
                # Concatenate: pre_traj + post_traj[:, 1:] to avoid duplicating injection frame
                return torch.cat([pre_traj, post_traj[:, 1:]], dim=1)
            return post_traj[:, -1:]
        else:
            if return_all_steps:
                return pre_traj
            return pre_traj[:, -1:]

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

    def _effective_max_gpu_batch(self, timesteps: int) -> int:
        """Compute sub-batch size from current free GPU memory.

        Available memory = (PyTorch reserved − allocated) + CUDA free.
        This avoids calling ``empty_cache()``, which inflates the free
        count by releasing reserved blocks that PyTorch immediately
        re-reserves on the next allocation.

        Uses 40% of available memory for the trajectory tensor, leaving
        60% for kernel FFTs, simulation intermediates (state, FFT
        buffers), and allocator overhead.
        """
        per_sample_bytes = (timesteps + 1) * self.n_channels * self.grid_size * self.grid_size * 4
        if self.device.type == "cuda":
            reserved = torch.cuda.memory_reserved(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            pool_available = reserved - allocated
            cuda_free = torch.cuda.mem_get_info(self.device)[0]
            total_available = pool_available + cuda_free
            budget = int(total_available * 0.4)
        else:
            budget = 2 * 1024 ** 3  # 2 GB fallback for CPU
        dynamic = max(4, budget // per_sample_bytes)
        return min(dynamic, self.max_gpu_batch)

    def _run_sub_batch(
        self,
        params_np: np.ndarray,
        ics: torch.Tensor,
        timesteps: int,
        return_all_steps: bool,
        perturbation_config: Optional[PerturbationConfig] = None,
        perturbation_seed: int = 0,
    ) -> torch.Tensor:
        """Simulate one sub-batch on GPU and return result on CPU."""
        ics_dev = ics.to(self.device)
        tensors = self._build_lenia_tensors(params_np)
        kernel_ffts = self._build_kernel_ffts(tensors)
        if perturbation_config is not None and perturbation_config.enabled:
            result = self._simulate_perturbed(
                ics_dev, tensors, kernel_ffts, timesteps, return_all_steps,
                perturbation_config, seed=perturbation_seed,
            )
        else:
            result = self._simulate(
                ics_dev, tensors, kernel_ffts, timesteps, return_all_steps,
            )
        return result.cpu()

    def rollout_batch(
        self,
        params_batch: Union[np.ndarray, torch.Tensor],
        ics: torch.Tensor,
        timesteps: int,
        return_all_steps: bool = True,
        perturbation_config: Optional[PerturbationConfig] = None,
        perturbation_seed: int = 0,
    ) -> torch.Tensor:
        """Fully vectorized batch rollout with OOM-safe GPU memory management.

        Sub-batch size is computed dynamically from free GPU memory.
        If an OOM occurs, the sub-batch is halved and retried (down to
        a floor of 4 samples).  This handles memory fragmentation,
        variable-length timesteps, and co-resident model/optimizer state
        without manual tuning.

        Args:
            params_batch: [B, D] Sobol parameters.
            ics: [B, C, H, W] initial conditions.
            timesteps: Number of simulation steps.
            return_all_steps: If True, prepend IC → [B, T+1, C, H, W].
            perturbation_config: If provided and enabled, apply perturbation
                mid-trajectory. Used for perturbed realizations during
                VQ training.
            perturbation_seed: Seed for perturbation RNG.

        Returns:
            Trajectories [B, T+1, C, H, W] on CPU.
        """
        params_np = self._to_numpy_batch(params_batch)
        B = params_np.shape[0]

        # Use cached sub-batch size if available, otherwise estimate from
        # free GPU memory.  The cache prevents repeated OOM + empty_cache()
        # cycles that waste ~50-100ms per occurrence.
        if timesteps in self._sub_batch_cache:
            eff_batch = self._sub_batch_cache[timesteps]
        else:
            eff_batch = self._effective_max_gpu_batch(timesteps)

        # Pre-allocate result on CPU to avoid O(num_chunks) intermediate
        # tensors and the expensive torch.cat copy at the end.
        C = ics.shape[1]
        T_out = timesteps + 1 if return_all_steps else timesteps
        result = torch.empty(B, T_out, C, self.grid_size, self.grid_size,
                             dtype=torch.float32)

        start = 0
        while start < B:
            end = min(start + eff_batch, B)
            try:
                result[start:end] = self._run_sub_batch(
                    params_np[start:end], ics[start:end],
                    timesteps, return_all_steps,
                    perturbation_config=perturbation_config,
                    perturbation_seed=perturbation_seed + start,
                )
                start = end  # advance on success
                # Cache the working size for this timestep count
                self._sub_batch_cache[timesteps] = eff_batch
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                old = eff_batch
                eff_batch = max(1, eff_batch // 2)
                logger.warning(
                    "OOM in rollout_batch (sub-batch %d→%d at T=%d), "
                    "halving to %d",
                    start, end, timesteps, eff_batch,
                )
                if eff_batch == old:
                    raise  # floor of 4, can't shrink further

        return result
