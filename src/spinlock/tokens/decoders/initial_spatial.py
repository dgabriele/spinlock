"""CNN spatial decoder for initial conditions.

Symmetric counterpart of SpatialICEncoder: upsamples from a G×G spatial grid
back to the full H×W initial condition grid via transposed convolutions.

Architecture (for G=4, H=W=128, base_channels=32):
    [B, encoded_dim] → Linear → [B, 256, 4, 4]
    → Upsample+Conv+ResBlock → [B, 256, 8, 8]
    → Upsample+Conv+ResBlock → [B, 128, 16, 16]
    → Upsample+Conv+ResBlock → [B, 64, 32, 32]
    → Upsample+Conv+ResBlock → [B, 32, 64, 64]
    → Upsample+Conv         → [B, C, 128, 128]
    → Sigmoid

Five upsample stages mirror the encoder's five downsample stages (stem + 4 ResBlocks).
Bilinear upsample + conv avoids checkerboard artifacts from transposed convolutions.
"""

import math

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block for decoder (no downsampling)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class SpatialICDecoder(nn.Module):
    """CNN spatial decoder for initial conditions.

    Symmetric counterpart of ``SpatialICEncoder``.  Takes concatenated FSQ
    quantized values and upsamples back to the full spatial grid.

    Args:
        encoded_dim: Total input dim (num_positions × fsq_dim, e.g. 16 × 3 = 48).
        channels: Number of output channels (auto-detected, e.g. 3 for Lenia).
        spatial_size: Target grid size H=W (auto-detected, e.g. 128).
        spatial_token_grid: Starting spatial grid (matches encoder, default 4).
        base_channels: Base CNN channel width (matches encoder, default 32).
    """

    def __init__(
        self,
        encoded_dim: int,
        channels: int,
        spatial_size: int,
        spatial_token_grid: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.channels = channels
        self.spatial_size = spatial_size
        self.spatial_token_grid = spatial_token_grid

        # Mirror encoder channel progression (reversed): 8× → 8× → 4× → 2× → 1×
        ch = [
            base_channels * 8,   # start (matches encoder stage4 output)
            base_channels * 8,   # after upsample 1
            base_channels * 4,   # after upsample 2
            base_channels * 2,   # after upsample 3
            base_channels,       # after upsample 4
        ]

        # Number of upsample stages needed: spatial_token_grid * 2^N = spatial_size
        n_stages = int(math.log2(spatial_size / spatial_token_grid))
        assert spatial_token_grid * (2 ** n_stages) == spatial_size, (
            f"spatial_size ({spatial_size}) must be spatial_token_grid ({spatial_token_grid}) "
            f"× power of 2"
        )
        self._n_stages = n_stages

        # Project flat quantized vector to spatial feature map
        self.project = nn.Sequential(
            nn.Linear(encoded_dim, ch[0] * spatial_token_grid * spatial_token_grid),
            nn.ReLU(inplace=True),
        )
        self._start_ch = ch[0]

        # Refine at starting resolution
        self.res0 = ResBlock(ch[0])

        # Build upsample stages (up to n_stages)
        # Channel schedule: ch[0] → ch[1] → ... → ch[n_stages-1]
        self.up_convs = nn.ModuleList()
        self.up_res = nn.ModuleList()
        prev_ch = ch[0]
        for i in range(n_stages):
            # Use channel schedule, clamping to available entries
            out_ch = ch[min(i + 1, len(ch) - 1)]
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(prev_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            self.up_res.append(ResBlock(out_ch))
            prev_ch = out_ch

        # Final conv to output channels
        self.final_conv = nn.Conv2d(prev_ch, channels, 3, padding=1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode quantized IC features to spatial grids.

        Args:
            encoded: Concatenated FSQ quantized values [B, encoded_dim].

        Returns:
            Reconstructed grids [B, C, H, W] in [0, 1].
        """
        B = encoded.shape[0]
        G = self.spatial_token_grid

        # Project to spatial feature map
        x = self.project(encoded)
        x = x.view(B, self._start_ch, G, G)

        # Refine at starting resolution
        x = self.res0(x)

        # Progressive upsample: G → 2G → 4G → ... → spatial_size
        for up_conv, up_res in zip(self.up_convs, self.up_res):
            x = nn.functional.interpolate(
                x, scale_factor=2, mode='bilinear', align_corners=False,
            )
            x = up_conv(x)
            x = up_res(x)

        # Final conv + sigmoid (ICs are in [0, 1])
        x = self.final_conv(x)
        return torch.sigmoid(x)

    def extra_repr(self) -> str:
        return (
            f"encoded_dim={self.encoded_dim}, channels={self.channels}, "
            f"spatial_size={self.spatial_size}, grid={self.spatial_token_grid}, "
            f"stages={self._n_stages}"
        )
