"""CNN spatial encoder for initial conditions.

ResNet-style encoder that produces spatially-organized features suitable
for per-position FSQ quantization.  Each spatial position in the output
grid becomes an independent VQ group with its own FSQ quantizer.

Architecture:
    128×128×C → Conv3×3/s2 → BN → ReLU
    → ResBlock → /2  (64→32)
    → ResBlock → /2  (32→16)
    → ResBlock → /2  (16→8)
    → ResBlock → /2  (8→4)
    → AdaptiveAvgPool2d(G, G) → [B, D_ch, G, G]
    → per-position Linear(D_ch → spatial_token_dim) → [B, G², spatial_token_dim]
    → flatten → [B, G² × spatial_token_dim]
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Pre-activation residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class SpatialICEncoder(nn.Module):
    """CNN spatial encoder for initial conditions.

    Produces spatially-organized features suitable for per-position FSQ
    quantization.  The output is a flat [B, num_positions * spatial_token_dim]
    tensor; model.py splits it into ``num_positions`` groups of
    ``spatial_token_dim`` features each.

    Args:
        in_channels: Input channels (auto-detected from dataset, e.g. 3).
        spatial_token_grid: Output spatial grid size (default 4 → 16 positions).
        spatial_token_dim: Feature dim per spatial position (default 8).
        base_channels: Base CNN channel width (doubled at each stage).
    """

    def __init__(
        self,
        in_channels: int,
        spatial_token_grid: int = 4,
        spatial_token_dim: int = 8,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_token_grid = spatial_token_grid
        self.spatial_token_dim = spatial_token_dim
        self.num_positions = spatial_token_grid ** 2
        self.output_dim = self.num_positions * spatial_token_dim

        # Channel progression: base → 2× → 4× → 8× → 8×
        ch = [base_channels, base_channels * 2, base_channels * 4,
              base_channels * 8, base_channels * 8]

        # Initial conv: /2 downsample
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )

        # 4 residual stages, each with /2 downsample
        self.stage1 = ResBlock(ch[0], ch[1], stride=2)
        self.stage2 = ResBlock(ch[1], ch[2], stride=2)
        self.stage3 = ResBlock(ch[2], ch[3], stride=2)
        self.stage4 = ResBlock(ch[3], ch[4], stride=2)

        # After stem(/2) + 4 stages(/2 each) = /32 total
        # 128→4 or adaptive pool to target grid
        self.pool = nn.AdaptiveAvgPool2d(spatial_token_grid)

        # Per-position projection: D_channel → spatial_token_dim
        self.d_channel = ch[4]
        self.proj = nn.Linear(ch[4], spatial_token_dim)

    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        """Encode initial condition grids to spatial token features.

        Args:
            ic: Initial condition grids [B, C, H, W].

        Returns:
            Flat features [B, num_positions * spatial_token_dim].
        """
        B = ic.shape[0]

        x = self.stem(ic)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)              # [B, D_ch, H', W']
        x = self.pool(x)                # [B, D_ch, G, G]

        # Reshape to [B, G², D_ch] → project → [B, G², spatial_token_dim]
        x = x.flatten(2).transpose(1, 2)  # [B, G², D_ch]
        x = self.proj(x)                  # [B, G², spatial_token_dim]

        return x.reshape(B, -1)            # [B, G² × spatial_token_dim]

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"grid={self.spatial_token_grid}×{self.spatial_token_grid}, "
            f"token_dim={self.spatial_token_dim}, "
            f"output_dim={self.output_dim}"
        )
