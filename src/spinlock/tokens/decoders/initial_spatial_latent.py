"""Latent-space decoder for IC roundtrip loss.

Bypasses the pixel-space CNN decode → re-encode cycle by mapping concatenated
IC quantized latents directly to the pre-encoder latent space (the output of
SpatialICEncoder before per-position projection).  This gives the roundtrip
loss a short, high-bandwidth gradient path that avoids the double-CNN
information loss responsible for ~8.5% IC roundtrip accuracy.

Shape flow:
    IC FSQ quantized values [B, encoded_dim]  (e.g. 16 positions × 3 FSQ dims = 48)
    → MLP: encoded_dim → hidden → hidden → output_dim
    → pre-encoder latent [B, output_dim]  (e.g. 16 × 8 spatial_token_dim = 128)
    → split per position → project → FSQ → compare tokens (roundtrip CE)
"""

import torch
import torch.nn as nn


class SpatialICLatentDecoder(nn.Module):
    """Decode FSQ codes to pre-encoder latent space (roundtrip only).

    This is NOT a pixel decoder — it maps to the same representation space
    as ``SpatialICEncoder.forward()`` output, so the roundtrip loss can
    skip the CNN re-encoding step entirely.

    Args:
        encoded_dim: Total input dim (num_positions × fsq_dim, e.g. 48).
        output_dim: Pre-encoder latent dim (num_positions × spatial_token_dim, e.g. 128).
        hidden_dim: MLP hidden layer width.
    """

    def __init__(
        self,
        encoded_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map quantized IC codes to pre-encoder latent space.

        Args:
            x: Concatenated FSQ quantized values [B, encoded_dim].

        Returns:
            Pre-encoder latent [B, output_dim].
        """
        return self.net(x)

    def extra_repr(self) -> str:
        return f"encoded_dim={self.encoded_dim}, output_dim={self.output_dim}"
