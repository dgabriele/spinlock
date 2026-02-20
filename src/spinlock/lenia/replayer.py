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
from .params import LeniaParams, sobol_to_lenia_params
from .simulator import LeniaSimulator

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
    ):
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.kernel_type = kernel_type
        self.device = torch.device(device)
        self.alive_threshold = alive_threshold
        self.saturation_threshold = saturation_threshold
        self.max_retries = max_retries

        self.simulator = LeniaSimulator(grid_size=grid_size, device=device)
        self.ic_generator = LeniaICGenerator()

    def rollout_batch(
        self,
        params_batch: np.ndarray,           # [B, C²+2C+1]
        num_realizations: int = 3,
        num_timesteps: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate B parameter sets × M realizations.

        Returns:
            inputs  [B, M, C, H, W]    — initial conditions per realization
            outputs [B, M, T, C, H, W] — full trajectories
        """
        B = params_batch.shape[0]
        C = self.n_channels
        H = W = self.grid_size

        # Convert all Sobol vectors to LeniaParams at once
        params_list = [
            sobol_to_lenia_params(params_batch[b], self.n_channels, self.kernel_type)
            for b in range(B)
        ]

        all_inputs = torch.zeros(B, num_realizations, C, H, W, device=self.device)
        all_outputs = torch.zeros(B, num_realizations, num_timesteps, C, H, W, device=self.device)

        for m in range(num_realizations):
            ic_seed = None if seed is None else (seed * 1000 + m)
            ic, traj = self._rollout_one_realization(
                params_list, num_timesteps, ic_seed
            )
            all_inputs[:, m] = ic
            all_outputs[:, m] = traj

        return all_inputs, all_outputs

    def _rollout_one_realization(
        self,
        params_list: list[LeniaParams],
        num_timesteps: int,
        seed: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one realization for all B samples, with alive-check and retries.

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
        traj = self.simulator.rollout_batch(ic, params_list, num_timesteps)

        # Check each sample; replace dead/saturated ones up to max_retries times
        for _ in range(self.max_retries):
            bad_indices = self._find_bad_samples(traj)
            if not bad_indices:
                break
            for b in bad_indices:
                retry_counts[b] += 1
                retry_seed = None if seed is None else (seed * 10000 + b * 100 + retry_counts[b])
                new_ic = self.ic_generator.generate_batch(
                    batch_size=1,
                    n_channels=self.n_channels,
                    grid_size=self.grid_size,
                    seed=retry_seed,
                    device=self.device,
                )
                new_traj = self.simulator.rollout_batch(new_ic, [params_list[b]], num_timesteps)
                ic[b] = new_ic[0]
                traj[b] = new_traj[0]

        if any(r > 0 for r in retry_counts):
            total_retried = sum(1 for r in retry_counts if r > 0)
            logger.debug(f"LeniaReplayer: retried {total_retried}/{B} samples")

        return ic, traj

    def _find_bad_samples(self, traj: torch.Tensor) -> list[int]:
        """Identify samples where the final frame is dead or fully saturated.

        A sample is 'dead' if mean activation ≤ alive_threshold.
        A sample is 'saturated' if mean activation ≥ saturation_threshold.

        Args:
            traj: [B, T, C, H, W]
        Returns:
            List of bad sample indices.
        """
        final_frame = traj[:, -1]               # [B, C, H, W]
        mean_act = final_frame.mean(dim=(1, 2, 3))  # [B]
        bad = []
        for b in range(len(mean_act)):
            v = mean_act[b].item()
            if v <= self.alive_threshold or v >= self.saturation_threshold:
                bad.append(b)
        return bad
