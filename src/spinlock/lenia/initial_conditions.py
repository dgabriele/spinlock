"""Gaussian blob initial condition generator for Lenia."""

from typing import Optional, Tuple

import torch


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
