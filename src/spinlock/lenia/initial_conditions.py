"""Initial condition generators for Lenia.

Provides both the original Gaussian blob generator and a diverse IC generator
that bridges InputFieldGenerator (used by CNO pipeline) to Lenia's [0,1] state space.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LeniaICGenerator:
    """Generates Gaussian blob initial conditions for Lenia.

    Each channel is independently populated with 1–5 Gaussian blobs
    placed at random positions with random amplitudes. This gives the
    CA system diverse starting configurations while keeping values ∈ [0,1].
    """

    def generate_batch(
        self,
        batch_size: int,
        n_channels: int,
        grid_size: int = 64,
        n_blobs_range: Tuple[int, int] = (1, 5),
        blob_radius_range: Tuple[float, float] = (4.0, 16.0),
        seed: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate a batch of random Gaussian blob ICs.

        Returns:
            [B, C, H, W] float32 ∈ [0, 1]
        """
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)

        H = W = grid_size
        output = torch.zeros(batch_size, n_channels, H, W, device=device, dtype=torch.float32)

        # Coordinate grids [H, W]
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

        n_blobs_lo, n_blobs_hi = n_blobs_range
        r_lo, r_hi = blob_radius_range

        for b in range(batch_size):
            for c in range(n_channels):
                # Sample number of blobs for this (b, c)
                n_blobs = int(
                    n_blobs_lo + torch.randint(
                        n_blobs_hi - n_blobs_lo + 1, (1,), generator=rng, device=device
                    ).item()
                )
                for _ in range(n_blobs):
                    # Random center position
                    cy = torch.rand(1, generator=rng, device=device).item() * H
                    cx = torch.rand(1, generator=rng, device=device).item() * W

                    # Random blob radius (= sigma of Gaussian)
                    sigma = r_lo + torch.rand(1, generator=rng, device=device).item() * (r_hi - r_lo)

                    # Random peak amplitude ∈ (0, 1]
                    amplitude = 0.3 + torch.rand(1, generator=rng, device=device).item() * 0.7

                    # Gaussian blob (no periodic wrapping — simple, sufficient)
                    dy = yy - cy
                    dx = xx - cx
                    blob = amplitude * torch.exp(-0.5 * (dy ** 2 + dx ** 2) / sigma ** 2)
                    output[b, c] = torch.clamp(output[b, c] + blob, 0.0, 1.0)

        return output


class DiverseLeniaICGenerator:
    """Diverse IC generator for Lenia using InputFieldGenerator.

    Samples IC types from a weighted distribution, generates via InputFieldGenerator,
    then applies per-sample min-max normalization to [0, 1] (Lenia state constraint).
    Supports IC type locking for realization-consistent IC assignment.
    """

    def __init__(
        self,
        grid_size: int,
        n_channels: int,
        device: torch.device,
        ic_type_weights: dict[str, float],
        ic_type_params: dict[str, dict],
    ):
        from spinlock.dataset.generators import InputFieldGenerator

        self.grid_size = grid_size
        self.n_channels = n_channels
        self.device = device
        self.generator = InputFieldGenerator(grid_size, n_channels, device)
        self.ic_type_params = ic_type_params

        # Normalize weights to probabilities
        self._types = list(ic_type_weights.keys())
        total = sum(ic_type_weights.values())
        self._probs = [ic_type_weights[t] / total for t in self._types]

        self._locked_types: Optional[list[str]] = None
        self.last_types: Optional[list[str]] = None

    @staticmethod
    def _get_base_ic_type(ic_type: str) -> str:
        """Strip alias suffixes: gaussian_random_field_v0 → gaussian_random_field."""
        return re.sub(r'_(v\d+|low|mid|high)$', '', ic_type)

    def sample_types_for_batch(self, batch_size: int) -> list[str]:
        """Sample IC types for a batch. Call once before all realizations."""
        return np.random.choice(self._types, size=batch_size, p=self._probs).tolist()

    def lock_types(self, types: list[str]) -> None:
        """Lock IC types so all subsequent generate_batch calls use these types."""
        self._locked_types = types

    def unlock_types(self) -> None:
        """Clear locked types."""
        self._locked_types = None

    def generate_batch(
        self,
        batch_size: int,
        n_channels: int,
        grid_size: int = 64,
        seed: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate a batch of diverse ICs normalized to [0, 1].

        Groups samples by IC type, batch-generates each group via InputFieldGenerator,
        then applies per-sample min-max normalization to satisfy Lenia's state constraint.

        Returns:
            [B, C, H, W] float32 ∈ [0, 1]
        """
        if self._locked_types is not None:
            # Truncate to batch_size (handles retry case where batch_size < locked count)
            types = self._locked_types[:batch_size]
        else:
            types = self.sample_types_for_batch(batch_size)
        self.last_types = list(types)

        output = torch.zeros(batch_size, n_channels, grid_size, grid_size, device=device, dtype=torch.float32)

        # Group samples by IC type for batched generation
        type_to_indices: dict[str, list[int]] = {}
        for i, t in enumerate(types):
            type_to_indices.setdefault(t, []).append(i)

        for ic_type, indices in type_to_indices.items():
            base_type = self._get_base_ic_type(ic_type)
            params = dict(self.ic_type_params.get(ic_type, {}))

            # Compute per-group seed for reproducibility
            group_seed = None
            if seed is not None:
                group_seed = seed + hash(ic_type) % (2**31)

            fields = self.generator.generate_batch(
                batch_size=len(indices),
                field_type=base_type,
                seed=group_seed,
                **params,
            )

            # Per-sample min-max normalization to [0, 1]
            for j, idx in enumerate(indices):
                sample = fields[j]  # [C, H, W]
                s_min = sample.min()
                s_max = sample.max()
                normalized = (sample - s_min) / (s_max - s_min + 1e-8)
                output[idx] = normalized.clamp(0.0, 1.0).to(device)

        return output
