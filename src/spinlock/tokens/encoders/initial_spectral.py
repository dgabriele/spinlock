"""Spectral initial condition encoder for periodic-BC operators.

FFT-based encoder: IC grid → low-frequency Fourier features → per-group
learned projection → VQ-friendly embeddings.

The FFT is deterministic and lossless for the K×K low-freq band. The per-group
projections are learned — they compress high-dimensional spectral features into
compact embeddings where VQ codebooks can form meaningful clusters.
"""

import torch
import torch.nn as nn


class SpectralICEncoder(nn.Module):
    """Encode initial conditions via 2D FFT + learned group projection.

    Stage 1 (deterministic): FFT → crop K×K modes → flatten to 2*C*K*K reals.
    Stage 2 (learned): per-group linear projection to group_dim.

    The learned projections solve the VQ capacity problem: raw spectral features
    are 192D/group (with 8 groups), too high for VQ to discretize with ~36 codes.
    Projecting to 32D/group lets codebooks form meaningful clusters.

    Args:
        in_channels: Number of input channels (e.g. 3 for Lenia).
        spatial_size: Spatial grid size H=W.
        num_modes: Fourier modes per spatial dimension (frequency cutoff K).
        num_groups: Number of VQ groups for initial features.
        group_dim: Output dimension per group (should match encoder.embedding_dim).
            If None, no projection is applied (raw spectral features pass through).
    """

    def __init__(
        self,
        in_channels: int,
        spatial_size: int,
        num_modes: int,
        num_groups: int,
        group_dim: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.num_modes = num_modes
        self.num_groups = num_groups
        self.group_dim = group_dim

        if num_modes > spatial_size // 2:
            raise ValueError(
                f"num_modes ({num_modes}) must be <= spatial_size//2 = {spatial_size // 2}"
            )

        # Raw spectral features: real + imaginary parts of K×K modes per channel
        self.raw_dim = 2 * in_channels * num_modes * num_modes
        self.features_per_group = self.raw_dim // num_groups

        if self.raw_dim % num_groups != 0:
            raise ValueError(
                f"raw_dim ({self.raw_dim}) must be divisible by num_groups ({num_groups})"
            )

        # Per-group learned projection (when group_dim is set)
        if group_dim is not None:
            self.group_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.features_per_group, group_dim),
                    nn.LayerNorm(group_dim),
                )
                for _ in range(num_groups)
            ])
            self.output_dim = group_dim * num_groups
        else:
            self.group_projections = None
            self.output_dim = self.raw_dim

    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        """Encode initial condition grids to spectral features.

        Args:
            ic: Initial condition grids [B, C, H, W].

        Returns:
            If group_dim is set: [B, num_groups * group_dim] projected features.
            If group_dim is None: [B, 2*C*K*K] raw spectral features.
        """
        B = ic.shape[0]
        K = self.num_modes

        # 2D FFT → [B, C, H, W] complex
        spectrum = torch.fft.fft2(ic.float())

        # Crop to low-frequency modes → [B, C, K, K] complex
        low_freq = spectrum[:, :, :K, :K]

        # Stack real + imag → [B, 2*C*K*K] real features
        features = torch.cat(
            [low_freq.real.reshape(B, -1), low_freq.imag.reshape(B, -1)],
            dim=1,
        )

        if self.group_projections is None:
            return features

        # Per-group projection
        groups = features.split(self.features_per_group, dim=1)
        projected = [proj(g) for proj, g in zip(self.group_projections, groups)]
        return torch.cat(projected, dim=1)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, spatial_size={self.spatial_size}, "
            f"num_modes={self.num_modes}, num_groups={self.num_groups}, "
            f"group_dim={self.group_dim}, output_dim={self.output_dim}"
        )
