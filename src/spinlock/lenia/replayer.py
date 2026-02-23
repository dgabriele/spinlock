"""LeniaReplayer: maps Sobol parameter vectors to Lenia trajectories.

Matches the QBM-style interface used by DatasetGenerationPipeline:
    rollout_batch(params_batch, num_realizations, num_timesteps, seed)
    → (inputs [B, M, C, H, W], outputs [B, M, T, C, H, W])
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch

from .initial_conditions import LeniaICGenerator
from .params import LeniaParams, sobol_to_lenia_params, sobol_batch_to_tensors
from .simulator import LeniaSimulator, build_kernel_ffts_batched

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
    ):
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.kernel_type = kernel_type
        self.device = torch.device(device)
        self.alive_threshold = alive_threshold
        self.saturation_threshold = saturation_threshold
        self.max_retries = max_retries

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

    def rollout_batch(
        self,
        params_batch: np.ndarray,           # [B, C²+2C+1]
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
        radii, growth_mu, growth_sigma, dt, coupling = sobol_batch_to_tensors(
            params_batch, self.n_channels, self.device,
        )

        # Build kernel FFTs ONCE — shared across all M realizations
        dist_grid = self.simulator._get_dist_grid()
        kernel_ffts = self.simulator._compiled_kernel_builder(
            radii, self.grid_size, dist_grid,
        )

        # Keep params_list for backward-compat retry path (builds single-sample params)
        params_list = [
            sobol_to_lenia_params(params_batch[b], self.n_channels, self.kernel_type)
            for b in range(B)
        ]

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
                    kernel_ffts, coupling, growth_mu, growth_sigma, dt,
                    params_list, num_timesteps, ic_seed,
                )
                all_inputs[:, m] = ic
                all_outputs[:, m] = traj
        finally:
            if has_locking:
                # Restore full batch types (retries may have overwritten last_types)
                self.ic_generator.last_types = list(types)
                self.ic_generator.unlock_types()

        return all_inputs, all_outputs

    def _rollout_one_realization_fast(
        self,
        kernel_ffts: torch.Tensor,      # [B, C, H, W//2+1] complex
        coupling: torch.Tensor,          # [B, C, C]
        growth_mu: torch.Tensor,         # [B, C]
        growth_sigma: torch.Tensor,      # [B, C]
        dt: torch.Tensor,                # [B]
        params_list: list[LeniaParams],  # fallback for retries
        num_timesteps: int,
        seed: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one realization using precomputed kernel FFTs, with retries.

        Returns:
            ic   [B, C, H, W]
            traj [B, T, C, H, W]
        """
        B = len(params_list)
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

        traj = self.simulator.rollout_batch_from_tensors(
            ic, kernel_ffts, coupling, growth_mu, growth_sigma, dt, num_timesteps,
        )

        # Retry dead/saturated samples
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
                # Slice kernel FFTs for single sample (zero-copy view)
                new_traj = self.simulator.rollout_batch_from_tensors(
                    new_ic,
                    kernel_ffts[b:b+1],
                    coupling[b:b+1],
                    growth_mu[b:b+1],
                    growth_sigma[b:b+1],
                    dt[b:b+1],
                    num_timesteps,
                )
                ic[b] = new_ic[0]
                traj[b] = new_traj[0]

        if any(r > 0 for r in retry_counts):
            total_retried = sum(1 for r in retry_counts if r > 0)
            logger.debug(f"LeniaReplayer: retried {total_retried}/{B} samples")

        return ic, traj

    def _find_bad_samples(self, traj: torch.Tensor) -> list[int]:
        """Identify samples where the final frame is dead or fully saturated.

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
        return bad_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
