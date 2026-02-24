"""LeniaReplayer: maps Sobol parameter vectors to Lenia trajectories.

Matches the QBM-style interface used by DatasetGenerationPipeline:
    rollout_batch(params_batch, num_realizations, num_timesteps, seed)
    → (inputs [B, M, C, H, W], outputs [B, M, T, C, H, W])
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch

from .initial_conditions import LeniaICGenerator
from .params import (
    DEFAULT_RANGES,
    LeniaBatchTensors,
    LeniaParamRanges,
    LeniaParams,
    sobol_batch_to_tensors,
    sobol_to_lenia_params,
)
from .simulator import (
    LeniaSimulator,
    build_kernel_ffts_batched,
    build_multiring_kernel_ffts_batched,
)

logger = logging.getLogger(__name__)


class LeniaReplayer:
    """Maps Sobol parameter vectors → Lenia trajectories.

    For each of M realizations:
        1. Generate fresh Gaussian blob ICs
        2. Run LeniaSimulator.rollout_batch
        3. Check aliveness — retry up to max_retries times if the simulation
           dies (all cells ≤ alive_threshold) or saturates (all cells ≥ saturation_threshold)
        4. If all retries are exhausted, keep the last attempt

    This mirrors the QBMReplayer pattern used by DatasetGenerationPipeline.
    """

    def __init__(
        self,
        n_channels: int = 3,
        grid_size: int = 64,
        kernel_type: str = "gaussian",
        device: str = "cuda",
        alive_threshold: float = 0.01,
        saturation_threshold: float = 0.95,
        max_retries: int = 5,
        ic_generator=None,
        substeps: int = 1,
        param_ranges: Optional[LeniaParamRanges] = None,
    ):
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.kernel_type = kernel_type
        self.device = torch.device(device)
        self.alive_threshold = alive_threshold
        self.saturation_threshold = saturation_threshold
        self.max_retries = max_retries
        self.substeps = substeps
        self.param_ranges = param_ranges  # None → DEFAULT_RANGES in sobol functions

        self.simulator = LeniaSimulator(grid_size=grid_size, device=device)
        self.ic_generator = ic_generator if ic_generator is not None else LeniaICGenerator()

    def generate_ics_only(
        self,
        batch_size: int,
        num_realizations: int = 3,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, list[str]]:
        """Generate initial conditions without running simulation.

        Handles IC type locking/unlocking internally (same protocol as
        rollout_batch) so the caller never needs to touch ic_generator.

        Returns:
            inputs: [B, M, C, H, W] — ICs per realization
            ic_types: list of IC type strings (length B)
        """
        C = self.n_channels
        H = W = self.grid_size
        M = num_realizations

        inputs = torch.zeros(batch_size, M, C, H, W, device=self.device)

        has_locking = hasattr(self.ic_generator, 'lock_types')
        if has_locking:
            types = self.ic_generator.sample_types_for_batch(batch_size)
            self.ic_generator.lock_types(types)

        try:
            for m in range(M):
                ic_seed = None if seed is None else (seed * 1000 + m)
                ic = self.ic_generator.generate_batch(
                    batch_size=batch_size,
                    n_channels=C,
                    grid_size=H,
                    seed=ic_seed,
                    device=self.device,
                )
                inputs[:, m] = ic
        finally:
            if has_locking:
                self.ic_generator.unlock_types()

        # Read IC types from generator (DiverseLeniaICGenerator tracks last_types)
        last_types = getattr(self.ic_generator, 'last_types', None)
        if last_types is not None:
            ic_types = list(last_types)
        else:
            ic_types = ["lenia_gaussian_blobs"] * batch_size

        return inputs, ic_types

    def _build_kernel_ffts(
        self,
        tensors: LeniaBatchTensors,
    ) -> torch.Tensor:
        """Build kernel FFTs from batch tensors, choosing the right builder.

        For V2 (no multi-ring params): uses the fast single-Gaussian builder.
        For V3 (with kernel_rank + beta): uses the multi-ring builder which
        supports variable ring count and multiple shell types.

        Returns:
            [B, C, H, W//2+1] complex64 kernel FFTs.
        """
        dist_grid = self.simulator._get_dist_grid()
        ranges = self.param_ranges

        if tensors.kernel_rank is not None and tensors.beta is not None:
            return build_multiring_kernel_ffts_batched(
                radii=tensors.radii,
                grid_size=self.grid_size,
                dist_grid=dist_grid,
                kernel_rank=tensors.kernel_rank,
                beta=tensors.beta,
                kernel_type_ids=tensors.kernel_type,
                kernel_types_list=ranges.kernel_types if ranges else None,
            )
        else:
            return self.simulator._compiled_kernel_builder(
                tensors.radii, self.grid_size, dist_grid,
            )

    def rollout_batch(
        self,
        params_batch: np.ndarray,           # [B, D]
        num_realizations: int = 3,
        num_timesteps: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate B parameter sets × M realizations.

        Precomputes kernel FFTs once and shares them across all M realizations
        (kernels depend only on params, not on ICs).

        Returns:
            inputs  [B, M, C, H, W]    — initial conditions per realization
            outputs [B, M, T, C, H, W] — full trajectories
        """
        B = params_batch.shape[0]
        C = self.n_channels
        H = W = self.grid_size

        # Vectorized Sobol → tensor conversion (one GPU transfer, no Python loop)
        tensors = sobol_batch_to_tensors(
            params_batch, self.n_channels, self.device,
            ranges=self.param_ranges,
        )

        # Build kernel FFTs ONCE — shared across all M realizations
        kernel_ffts = self._build_kernel_ffts(tensors)

        all_inputs = torch.zeros(B, num_realizations, C, H, W, device=self.device)
        all_outputs = torch.zeros(B, num_realizations, num_timesteps, C, H, W, device=self.device)

        # Lock IC types across realizations for consistency:
        # all M realizations of a given parameter set use the same IC type
        has_locking = hasattr(self.ic_generator, 'lock_types')
        if has_locking:
            types = self.ic_generator.sample_types_for_batch(B)
            self.ic_generator.lock_types(types)

        try:
            for m in range(num_realizations):
                ic_seed = None if seed is None else (seed * 1000 + m)
                ic, traj = self._rollout_one_realization_fast(
                    kernel_ffts, tensors, num_timesteps, ic_seed,
                )
                all_inputs[:, m] = ic
                all_outputs[:, m] = traj
        finally:
            if has_locking:
                # Restore full batch types (retries may have overwritten last_types)
                self.ic_generator.last_types = list(types)
                self.ic_generator.unlock_types()

        return all_inputs, all_outputs

    def _compute_adaptive_substeps(
        self,
        dt: torch.Tensor,
        growth_sigma: torch.Tensor,
        growth_type: Optional[torch.Tensor] = None,
    ) -> int:
        """Compute substep count K for CFL stability.

        The CFL condition for explicit Euler on Lenia's growth update is:
            dt_eff * |G'|_max < 2
        where |G'|_max depends on the growth function type:
            Gaussian:    1.716 / σ_min
            Polynomial:  3.079 / σ_min
            Step (α=5):  5.0   / σ_min  (smoothed tanh approximation)

        We compute K = max(base, ceil(CFL_max / target)) where target = 1.0
        (well below stability limit of 2.0 for safety margin), capped at 32
        to prevent extreme compute cost for pathological parameter corners.

        Args:
            dt:           [B] timestep tensor.
            growth_sigma: [B, C] per-channel growth sigma.
            growth_type:  [B] long or None (None = all Gaussian).

        Returns:
            Integer substep count K ≥ self.substeps.
        """
        K = self.substeps
        if self.param_ranges is None:
            return K

        # Min sigma across channels per sample → most restrictive CFL
        sigma_min = growth_sigma.min(dim=1).values.clamp(min=1e-8)  # [B]

        # Select worst-case G'_max constant based on growth types in batch
        if growth_type is None:
            g_prime_const = 1.716  # all Gaussian
        else:
            has_step = (growth_type == 2).any().item()
            has_poly = (growth_type == 1).any().item()
            if has_step:
                g_prime_const = 5.0
            elif has_poly:
                g_prime_const = 3.079
            else:
                g_prime_const = 1.716

        # CFL = dt * g_prime_const / sigma_min; want CFL_eff = CFL / K < 1.0
        cfl_max = (dt * g_prime_const / sigma_min).max().item()
        K_cfl = math.ceil(cfl_max / 1.0)
        K = max(K, K_cfl)
        K = min(K, 32)  # cap to bound compute cost
        return K

    def _simulate_with_substeps(
        self,
        ics: torch.Tensor,
        kernel_ffts: torch.Tensor,
        coupling: torch.Tensor,
        growth_mu: torch.Tensor,
        growth_sigma: torch.Tensor,
        dt: torch.Tensor,
        num_timesteps: int,
        growth_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run simulation with substep integration for numerical stability.

        When substeps K > 1, each visible frame is produced by K Euler steps
        at dt/K.  Total simulated physical time per frame is unchanged, but
        sharp growth functions (small sigma) no longer cause period-2
        oscillation because the effective CFL number is reduced by K.

        Memory-efficient: only stores one frame per visible timestep (not the
        K intermediate substeps). Memory is O(B·T·C·H·W) regardless of K.

        If K is capped (at 32), a per-sample dt safety clamp ensures the
        effective CFL stays below 1.0 even for extreme parameter combinations.

        Returns:
            traj [B, T, C, H, W]
        """
        K = self._compute_adaptive_substeps(dt, growth_sigma, growth_type)

        if K <= 1:
            return self.simulator.rollout_batch_from_tensors(
                ics, kernel_ffts, coupling, growth_mu, growth_sigma, dt,
                num_timesteps, growth_type=growth_type,
            )

        sim_dt = dt / K

        # Per-sample dt safety clamp: if K was capped, ensure CFL < 1.0
        if growth_type is not None:
            sigma_min = growth_sigma.min(dim=1).values.clamp(min=1e-8)
            g_prime = torch.full_like(dt, 1.716)
            g_prime[growth_type == 1] = 3.079
            g_prime[growth_type == 2] = 5.0
            dt_safe = sigma_min / g_prime
            sim_dt = torch.minimum(sim_dt, dt_safe)

        # Memory-efficient substep loop: K inner steps per visible frame
        B, C, H, W = ics.shape
        state = ics.to(self.simulator.device).float()
        mu = growth_mu[:, :, None, None]
        sigma = growth_sigma[:, :, None, None]
        dt_view = sim_dt[:, None, None, None]

        traj = torch.empty(B, num_timesteps, C, H, W,
                           device=self.simulator.device, dtype=torch.float32)
        step_fn = self.simulator._compiled_step

        for t in range(num_timesteps):
            for _k in range(K):
                state = step_fn(state, kernel_ffts, coupling, mu, sigma,
                                dt_view, growth_type)
            traj[:, t] = state

        return traj

    def _rollout_one_realization_fast(
        self,
        kernel_ffts: torch.Tensor,      # [B, C, H, W//2+1] complex
        tensors: LeniaBatchTensors,      # all batch tensors
        num_timesteps: int,
        seed: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one realization using precomputed kernel FFTs, with retries.

        Returns:
            ic   [B, C, H, W]
            traj [B, T, C, H, W]
        """
        B = tensors.radii.shape[0]
        retry_counts = [0] * B

        # Initial attempt
        ic = self.ic_generator.generate_batch(
            batch_size=B,
            n_channels=self.n_channels,
            grid_size=self.grid_size,
            seed=seed,
            device=self.device,
        )
        batch_ic_types = getattr(self.ic_generator, 'last_types', None)

        traj = self._simulate_with_substeps(
            ic, kernel_ffts, tensors.coupling, tensors.growth_mu,
            tensors.growth_sigma, tensors.dt, num_timesteps,
            growth_type=tensors.growth_type,
        )

        # Retry dead/saturated/oscillating samples
        has_locking = hasattr(self.ic_generator, 'lock_types')
        for _ in range(self.max_retries):
            bad_indices = self._find_bad_samples(traj)
            if not bad_indices:
                break
            for b in bad_indices:
                retry_counts[b] += 1
                retry_seed = None if seed is None else (seed * 10000 + b * 100 + retry_counts[b])
                saved_locked = None
                if has_locking and batch_ic_types is not None:
                    saved_locked = self.ic_generator._locked_types
                    self.ic_generator.lock_types([batch_ic_types[b]])
                new_ic = self.ic_generator.generate_batch(
                    batch_size=1,
                    n_channels=self.n_channels,
                    grid_size=self.grid_size,
                    seed=retry_seed,
                    device=self.device,
                )
                if saved_locked is not None:
                    self.ic_generator._locked_types = saved_locked
                # Slice tensors for single sample (zero-copy views)
                gt_slice = (
                    tensors.growth_type[b:b+1]
                    if tensors.growth_type is not None else None
                )
                new_traj = self._simulate_with_substeps(
                    new_ic,
                    kernel_ffts[b:b+1],
                    tensors.coupling[b:b+1],
                    tensors.growth_mu[b:b+1],
                    tensors.growth_sigma[b:b+1],
                    tensors.dt[b:b+1],
                    num_timesteps,
                    growth_type=gt_slice,
                )
                ic[b] = new_ic[0]
                traj[b] = new_traj[0]

        if any(r > 0 for r in retry_counts):
            total_retried = sum(1 for r in retry_counts if r > 0)
            logger.debug(f"LeniaReplayer: retried {total_retried}/{B} samples")

        return ic, traj

    def _find_bad_samples(self, traj: torch.Tensor) -> list[int]:
        """Identify dead, saturated, or period-2 oscillating samples.

        Checks:
            1. Dead: final-frame mean activation ≤ alive_threshold
            2. Saturated: final-frame mean activation ≥ saturation_threshold
            3. Period-2 oscillation: last 10 frames show alternating-sign
               consecutive differences (Euler instability artifact from
               large dt × sharp growth functions)

        Uses vectorized comparison instead of per-sample .item() calls
        to avoid CUDA synchronization overhead.

        Args:
            traj: [B, T, C, H, W]
        Returns:
            List of bad sample indices.
        """
        final_frame = traj[:, -1]               # [B, C, H, W]
        mean_act = final_frame.mean(dim=(1, 2, 3))  # [B]
        bad_mask = (mean_act <= self.alive_threshold) | (mean_act >= self.saturation_threshold)

        # Detect period-2 oscillation in last 10 frames
        T = traj.shape[1]
        if T >= 10:
            late = traj[:, -10:]                              # [B, 10, C, H, W]
            means = late.mean(dim=(2, 3, 4))                  # [B, 10]
            diffs = means[:, 1:] - means[:, :-1]              # [B, 9]
            sign_alt = (diffs[:, 1:] * diffs[:, :-1]) < 0     # [B, 8]
            oscillating = sign_alt.float().mean(dim=1) > 0.75  # >75% alternation
            amplitude = means.max(dim=1).values - means.min(dim=1).values
            bad_mask = bad_mask | (oscillating & (amplitude > 0.01))

        return bad_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
