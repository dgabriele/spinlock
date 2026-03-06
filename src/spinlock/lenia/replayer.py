"""LeniaReplayer: maps Sobol parameter vectors to Lenia trajectories.

Matches the QBM-style interface used by DatasetGenerationPipeline:
    rollout_batch(params_batch, num_realizations, num_timesteps, seed)
    → (inputs [B, M, C, H, W], outputs [B, M, T, C, H, W])
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
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
    simulate_with_early_exit,
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalActivityMetrics:
    """Per-sample temporal activity metrics computed from trajectory late half."""
    early_late_mse: torch.Tensor       # [B] MSE(frame[T//8], frame[-1])
    quarter_late_mse: torch.Tensor     # [B] MSE(frame[T//4], frame[-1])
    late_half_mean_var: torch.Tensor   # [B] var of spatial-mean over late half
    late_evolution_rate: torch.Tensor  # [B] mean |frame[t+1] - frame[t]| over late half


class DynamicsClass(str, Enum):
    """Classification of trajectory dynamics (informational, never triggers rejection)."""
    FIXED_POINT = "fixed_point"
    PERIODIC = "periodic"
    APERIODIC = "aperiodic"
    TRANSIENT = "transient"


class _WelfordNormalizer:
    """Online mean/std estimator using Welford's algorithm.

    Tracks running mean and variance without storing all samples,
    enabling normalization of fingerprint vectors in the dedup buffer.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = torch.zeros(dim)
        self.M2 = torch.zeros(dim)

    def update(self, batch: torch.Tensor) -> None:
        """Update stats with a batch of vectors [N, D] (CPU)."""
        for x in batch:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def std(self) -> torch.Tensor:
        """Return current std estimate, clamped to avoid div-by-zero."""
        if self.count < 2:
            return torch.ones(self.dim)
        return (self.M2 / (self.count - 1)).sqrt().clamp(min=1e-8)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize [N, D] tensor to unit variance using running stats."""
        return (x - self.mean) / self.std()


class BehavioralDedupBuffer:
    """Cross-batch behavioral fingerprint deduplication buffer.

    Maintains a CPU buffer of normalized fingerprints and checks new
    samples against it using chunked GPU cdist. Accepts samples whose
    minimum L2 distance to all existing fingerprints exceeds threshold.
    """

    FINGERPRINT_DIM = 8  # 3 channel means + spatial_var + temporal_var + grad_energy + spectral_centroid + inter_channel_corr

    def __init__(self, threshold: float = 0.5, chunk_size: int = 10_000, device: str = "cuda"):
        self._buffer = torch.empty(0, self.FINGERPRINT_DIM)  # CPU
        self._normalizer = _WelfordNormalizer(self.FINGERPRINT_DIM)
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.device = torch.device(device)

    def check_and_add(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """Check fingerprints against buffer and add non-duplicates.

        Args:
            fingerprints: [B, D] raw fingerprint vectors (GPU or CPU).

        Returns:
            [B] bool tensor — True = duplicate (should reject).
        """
        fp_cpu = fingerprints.detach().cpu()
        B = fp_cpu.shape[0]

        # Update normalizer with new batch
        self._normalizer.update(fp_cpu)

        # Normalize all (buffer + new) with current stats
        fp_norm = self._normalizer.normalize(fp_cpu)

        if self._buffer.shape[0] == 0:
            # First batch: accept all, seed buffer
            self._buffer = fp_cpu
            return torch.zeros(B, dtype=torch.bool)

        buf_norm = self._normalizer.normalize(self._buffer)

        # Chunked cdist on GPU for speed
        dup_mask = torch.zeros(B, dtype=torch.bool)
        fp_gpu = fp_norm.to(self.device)

        for start in range(0, buf_norm.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, buf_norm.shape[0])
            chunk_gpu = buf_norm[start:end].to(self.device)
            dists = torch.cdist(fp_gpu, chunk_gpu)  # [B, chunk]
            min_dists = dists.min(dim=1).values      # [B]
            dup_mask |= (min_dists < self.threshold).cpu()

        # Add non-duplicates to buffer
        keep = ~dup_mask
        if keep.any():
            self._buffer = torch.cat([self._buffer, fp_cpu[keep]], dim=0)

        return dup_mask


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
        # Dimension 1: temporal convergence
        min_temporal_activity: float = 0.0,
        min_early_late_mse: float = 0.0,
        # Dimension 2: spatial+temporal complexity
        spatial_var_threshold: float = 0.0,
        gradient_energy_threshold: float = 0.0,
        spectral_flatness_threshold: float = 0.0,
        # Dimension 3: behavioral deduplication
        dedup_enabled: bool = False,
        dedup_threshold: float = 0.5,
        # Dynamics classification
        classify_dynamics: bool = False,
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

        # Dimension 1: temporal convergence thresholds
        self.min_temporal_activity = min_temporal_activity
        self.min_early_late_mse = min_early_late_mse
        # Dimension 2: spatial+temporal complexity thresholds
        self.spatial_var_threshold = spatial_var_threshold
        self.gradient_energy_threshold = gradient_energy_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold
        # Dimension 3: behavioral deduplication
        self.dedup_enabled = dedup_enabled
        self.dedup_buffer: Optional[BehavioralDedupBuffer] = None
        if dedup_enabled:
            self.dedup_buffer = BehavioralDedupBuffer(
                threshold=dedup_threshold, device=device,
            )
        # Dynamics classification
        self.classify_dynamics = classify_dynamics
        self._last_dynamics_classes: Optional[list[str]] = None
        self._last_activity_metrics: Optional[TemporalActivityMetrics] = None

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

        # Classify dynamics on realization 0 (informational, never rejects)
        if self.classify_dynamics:
            self._classify_dynamics(all_outputs[:, 0])

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

        state = ics.to(self.simulator.device).float()
        mu = growth_mu[:, :, None, None]
        sigma = growth_sigma[:, :, None, None]
        dt_view = sim_dt[:, None, None, None]

        return simulate_with_early_exit(
            state, self.simulator._compiled_step, kernel_ffts, coupling,
            mu, sigma, dt_view, growth_type, num_timesteps, substeps=K,
        )

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

    # ── Dimension 1: Temporal convergence metrics ──

    def _compute_temporal_activity(self, traj: torch.Tensor) -> TemporalActivityMetrics:
        """Compute temporal activity metrics from trajectory.

        Memory-efficient: loops over frame pairs for evolution rate instead
        of materializing a full [B, T//2, C, H, W] diff tensor.

        Args:
            traj: [B, T, C, H, W]
        Returns:
            TemporalActivityMetrics with all per-sample [B] tensors.
        """
        B, T, C, H, W = traj.shape
        final = traj[:, -1]

        # MSE between early frames and final frame
        early_idx = max(T // 8, 0)
        quarter_idx = max(T // 4, 0)
        early_late_mse = ((traj[:, early_idx] - final) ** 2).mean(dim=(1, 2, 3))
        quarter_late_mse = ((traj[:, quarter_idx] - final) ** 2).mean(dim=(1, 2, 3))

        # Variance of spatial-mean over late half
        half_T = T // 2
        late_means = traj[:, half_T:].mean(dim=(2, 3, 4))  # [B, T//2]
        late_half_mean_var = late_means.var(dim=1)           # [B]

        # Frame-diff loop (memory-efficient: O(B·C·H·W) temp per step)
        late = traj[:, half_T:]
        accum = torch.zeros(B, device=traj.device)
        num_diffs = late.shape[1] - 1
        for t in range(num_diffs):
            accum += (late[:, t + 1] - late[:, t]).abs().mean(dim=(1, 2, 3))
        late_evolution_rate = accum / max(num_diffs, 1)

        return TemporalActivityMetrics(
            early_late_mse=early_late_mse,
            quarter_late_mse=quarter_late_mse,
            late_half_mean_var=late_half_mean_var,
            late_evolution_rate=late_evolution_rate,
        )

    # ── Dimension 2: Spatial+temporal complexity metrics ──

    def _compute_late_spatial_variance(self, traj: torch.Tensor) -> torch.Tensor:
        """Mean spatial variance over late half. Catches uniform-pulsing grids.

        Args:
            traj: [B, T, C, H, W]
        Returns:
            [B] mean spatial variance.
        """
        T = traj.shape[1]
        late = traj[:, T // 2:]                  # [B, T//2, C, H, W]
        spatial_var = late.var(dim=(-2, -1))      # [B, T//2, C]
        return spatial_var.mean(dim=(1, 2))        # [B]

    def _compute_gradient_energy(self, traj: torch.Tensor, num_frames: int = 4) -> torch.Tensor:
        """Mean |nabla u|^2 on sampled late frames.

        Uses central differences with circular padding (toroidal grid).

        Args:
            traj: [B, T, C, H, W]
            num_frames: number of late frames to sample.
        Returns:
            [B] mean gradient energy.
        """
        B, T, C, H, W = traj.shape
        indices = torch.linspace(T // 2, T - 1, num_frames, dtype=torch.long, device=traj.device)
        frames = traj[:, indices]  # [B, num_frames, C, H, W]

        # Central differences with circular boundary (toroidal grid)
        grad_x = (frames.roll(-1, dims=-1) - frames.roll(1, dims=-1)) * 0.5
        grad_y = (frames.roll(-1, dims=-2) - frames.roll(1, dims=-2)) * 0.5
        return (grad_x.pow(2) + grad_y.pow(2)).mean(dim=(1, 2, 3, 4))  # [B]

    def _compute_spectral_flatness(self, traj: torch.Tensor) -> torch.Tensor:
        """Spectral flatness of spatial-mean temporal signal.

        Generalizes period-2 detection to any periodicity. A flat spectrum
        (flatness ~ 1.0) indicates aperiodic dynamics; a peaked spectrum
        (flatness ~ 0.0) indicates strong periodic orbits.

        Args:
            traj: [B, T, C, H, W]
        Returns:
            [B] spectral flatness in [0, 1].
        """
        B, T, C, H, W = traj.shape
        half_T = T // 2
        signal = traj[:, half_T:].mean(dim=(-2, -1))           # [B, T//2, C]
        signal = signal - signal.mean(dim=1, keepdim=True)       # remove DC

        # Power spectrum (skip DC bin at index 0)
        power = torch.fft.rfft(signal, dim=1).abs().pow(2)[:, 1:]  # [B, F, C]
        log_power = torch.log(power.clamp(min=1e-20))

        # Geometric mean / arithmetic mean per channel
        geom = torch.exp(log_power.mean(dim=1))                   # [B, C]
        arith = power.mean(dim=1)                                  # [B, C]
        flatness = geom / arith.clamp(min=1e-20)                   # [B, C]
        return flatness.mean(dim=1)                                 # [B]

    # ── Core rejection logic ──

    def _find_bad_samples(self, traj: torch.Tensor) -> list[int]:
        """Identify dead, saturated, oscillating, or trivially-converged samples.

        Checks (original):
            1. Dead: final-frame mean activation <= alive_threshold
            2. Saturated: final-frame mean activation >= saturation_threshold
            3. Period-2 oscillation: alternating-sign diffs in last 10 frames

        Checks (Dimension 1 — temporal convergence, when thresholds > 0):
            4. Fast convergence: late evolution rate < min_temporal_activity
            5. Static final state: MSE(frame[T//8], frame[-1]) < min_early_late_mse

        Checks (Dimension 2 — spatial+temporal complexity, when thresholds > 0):
            6. Spatial homogeneity: late spatial variance < spatial_var_threshold
            7. Structureless: gradient energy < gradient_energy_threshold
            8. Periodic orbit: spectral flatness < spectral_flatness_threshold

        All new checks are zero-overhead when threshold = 0.0 (default).

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

        # Dimension 1: temporal convergence
        if self.min_temporal_activity > 0 or self.min_early_late_mse > 0:
            metrics = self._compute_temporal_activity(traj)
            if self.min_temporal_activity > 0:
                bad_mask = bad_mask | (metrics.late_evolution_rate < self.min_temporal_activity)
            if self.min_early_late_mse > 0:
                bad_mask = bad_mask | (metrics.early_late_mse < self.min_early_late_mse)

        # Dimension 2: spatial+temporal complexity
        if self.spatial_var_threshold > 0:
            bad_mask = bad_mask | (self._compute_late_spatial_variance(traj) < self.spatial_var_threshold)
        if self.gradient_energy_threshold > 0:
            bad_mask = bad_mask | (self._compute_gradient_energy(traj) < self.gradient_energy_threshold)
        if self.spectral_flatness_threshold > 0:
            bad_mask = bad_mask | (self._compute_spectral_flatness(traj) < self.spectral_flatness_threshold)

        return bad_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    # ── Dimension 3: Behavioral deduplication ──

    def _compute_fingerprint(self, traj: torch.Tensor) -> torch.Tensor:
        """Compute 8-dim behavioral fingerprint from trajectory late half.

        Components:
            [0:C] per-channel late-half means
            [C]   spatial variance
            [C+1] temporal variance (of spatial mean)
            [C+2] gradient energy
            [C+3] spectral centroid (weighted mean frequency)
            [C+4] inter-channel correlation (mean pairwise)

        For C=3 this yields exactly 8 dimensions.

        Args:
            traj: [B, T, C, H, W]
        Returns:
            [B, 8] fingerprint tensor.
        """
        B, T, C, H, W = traj.shape
        half_T = T // 2
        late = traj[:, half_T:]

        # Per-channel late means
        ch_means = late.mean(dim=(1, 3, 4))  # [B, C]

        # Spatial variance (scalar)
        spatial_var = self._compute_late_spatial_variance(traj).unsqueeze(1)  # [B, 1]

        # Temporal variance of spatial mean
        temporal_signal = late.mean(dim=(2, 3, 4))  # [B, T//2]
        temporal_var = temporal_signal.var(dim=1, keepdim=True)  # [B, 1]

        # Gradient energy (scalar)
        grad_energy = self._compute_gradient_energy(traj).unsqueeze(1)  # [B, 1]

        # Spectral centroid of spatial-mean signal
        signal = late.mean(dim=(-2, -1))  # [B, T//2, C]
        signal = signal - signal.mean(dim=1, keepdim=True)
        power = torch.fft.rfft(signal, dim=1).abs().pow(2)[:, 1:]  # [B, F, C]
        freqs = torch.arange(1, power.shape[1] + 1, device=traj.device, dtype=torch.float32)
        freqs = freqs.unsqueeze(0).unsqueeze(-1)  # [1, F, 1]
        power_sum = power.sum(dim=1).clamp(min=1e-20)  # [B, C]
        spectral_centroid = ((power * freqs).sum(dim=1) / power_sum).mean(dim=1, keepdim=True)  # [B, 1]

        # Inter-channel correlation (mean pairwise correlation of late spatial means)
        if C >= 2:
            ch_signals = late.mean(dim=(-2, -1))  # [B, T//2, C]
            ch_signals = ch_signals - ch_signals.mean(dim=1, keepdim=True)
            # Pairwise correlations
            norms = ch_signals.norm(dim=1).clamp(min=1e-8)  # [B, C]
            corr_sum = torch.zeros(B, device=traj.device)
            n_pairs = 0
            for i in range(C):
                for j in range(i + 1, C):
                    dot = (ch_signals[:, :, i] * ch_signals[:, :, j]).sum(dim=1)
                    corr = dot / (norms[:, i] * norms[:, j])
                    corr_sum += corr
                    n_pairs += 1
            inter_ch_corr = (corr_sum / max(n_pairs, 1)).unsqueeze(1)  # [B, 1]
        else:
            inter_ch_corr = torch.zeros(B, 1, device=traj.device)

        # Concatenate: [B, C + 5] = [B, 8] for C=3
        fp = torch.cat([ch_means, spatial_var, temporal_var, grad_energy,
                         spectral_centroid, inter_ch_corr], dim=1)
        return fp

    def filter_duplicates(self, outputs: torch.Tensor) -> torch.Tensor:
        """Check batch for behavioral near-duplicates against accumulated buffer.

        Uses realization 0 for fingerprinting. NOT called inside _find_bad_samples
        because dedup is cross-batch stateful.

        Args:
            outputs: [B, M, T, C, H, W] trajectory tensor.
        Returns:
            [B] bool tensor — True = duplicate (should reject).
        """
        if self.dedup_buffer is None:
            return torch.zeros(outputs.shape[0], dtype=torch.bool)
        traj = outputs[:, 0]  # realization 0: [B, T, C, H, W]
        fp = self._compute_fingerprint(traj)
        return self.dedup_buffer.check_and_add(fp)

    # ── Dynamics classification (informational) ──

    def _classify_dynamics(self, traj: torch.Tensor) -> list[str]:
        """Classify each sample's dynamics from trajectory metrics.

        Priority: FIXED_POINT > PERIODIC > TRANSIENT > APERIODIC (default).
        Stores result in self._last_dynamics_classes for pipeline retrieval.

        Args:
            traj: [B, T, C, H, W]
        Returns:
            List of dynamics class strings, length B.
        """
        metrics = self._compute_temporal_activity(traj)
        self._last_activity_metrics = metrics

        B = traj.shape[0]
        classes = [DynamicsClass.APERIODIC] * B

        # Thresholds for classification (internal, not config-exposed)
        fixed_thresh = 1e-6
        periodic_flatness_thresh = 0.05
        transient_quarter_thresh = 1e-4

        # Fixed point: essentially no evolution in late half
        is_fixed = metrics.late_evolution_rate < fixed_thresh

        # Periodic: low spectral flatness (strong periodic component)
        flatness = self._compute_spectral_flatness(traj)
        is_periodic = (flatness < periodic_flatness_thresh) & ~is_fixed

        # Transient: significant early-late MSE but low late evolution
        is_transient = (
            (metrics.early_late_mse > transient_quarter_thresh)
            & (metrics.late_evolution_rate < fixed_thresh * 100)
            & ~is_fixed & ~is_periodic
        )

        # Assign by priority
        for i in range(B):
            if is_fixed[i]:
                classes[i] = DynamicsClass.FIXED_POINT
            elif is_periodic[i]:
                classes[i] = DynamicsClass.PERIODIC
            elif is_transient[i]:
                classes[i] = DynamicsClass.TRANSIENT
            # else: APERIODIC (default)

        self._last_dynamics_classes = [c.value for c in classes]
        return self._last_dynamics_classes
