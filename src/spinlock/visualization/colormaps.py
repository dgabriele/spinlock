"""
GPU-resident colormap lookup tables for fast rendering.

Provides matplotlib colormaps converted to GPU tensors for efficient
colormap application without CPU transfers.
"""

import torch
import matplotlib.pyplot as plt
from typing import Optional


class GPUColormap:
    """
    GPU-resident colormap with fast lookup.

    Converts matplotlib colormaps to GPU lookup tables (LUTs) for
    efficient colormap application. All operations stay on GPU.

    Example:
        ```python
        colormap = GPUColormap("viridis", device=torch.device("cuda"))

        # Apply colormap to normalized values [0, 1]
        values = torch.rand(10, 64, 64, device="cuda")  # [B, H, W]
        rgb = colormap.apply(values)  # [B, 3, H, W]
        ```
    """

    def __init__(
        self,
        name: str = "viridis",
        num_entries: int = 256,
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize GPU colormap.

        Args:
            name: Matplotlib colormap name
            num_entries: Number of LUT entries (default: 256)
            device: Torch device
        """
        self.name = name
        self.num_entries = num_entries
        self.device = device

        # Create lookup table
        self.lut = self._create_lut(name, num_entries)

    def _create_lut(self, name: str, num_entries: int) -> torch.Tensor:
        """
        Create GPU-resident colormap lookup table.

        Args:
            name: Matplotlib colormap name
            num_entries: Number of entries

        Returns:
            LUT tensor [num_entries, 3] on device
        """
        try:
            cmap = plt.get_cmap(name)
        except ValueError:
            # Fallback to viridis if colormap not found
            print(f"Warning: Colormap '{name}' not found, using 'viridis'")
            cmap = plt.get_cmap("viridis")

        # Sample colormap
        colors = []
        for i in range(num_entries):
            rgba = cmap(i / (num_entries - 1))  # Normalize to [0, 1]
            colors.append(rgba[:3])  # Drop alpha channel

        # Convert to tensor and move to GPU
        lut = torch.tensor(colors, dtype=torch.float32, device=self.device)

        return lut

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """
        Apply colormap to normalized values.

        Args:
            values: Normalized values in [0, 1], shape [B, H, W] or [H, W]

        Returns:
            RGB tensor [B, 3, H, W] or [3, H, W]
        """
        # Handle both batched and unbatched inputs
        unsqueezed = False
        if values.ndim == 2:
            values = values.unsqueeze(0)  # [H, W] -> [1, H, W]
            unsqueezed = True

        B, H, W = values.shape

        # Clamp values to [0, 1]
        values = torch.clamp(values, 0.0, 1.0)

        # Convert to indices [0, num_entries-1]
        indices = (values * (self.num_entries - 1)).long()

        # Lookup RGB values
        rgb_flat = self.lut[indices.flatten()]  # [B*H*W, 3]
        rgb = rgb_flat.reshape(B, H, W, 3)  # [B, H, W, 3]

        # Permute to [B, 3, H, W]
        rgb = rgb.permute(0, 3, 1, 2)

        # Remove batch dimension if input was unbatched
        if unsqueezed:
            rgb = rgb.squeeze(0)

        return rgb


# Common scientific colormaps
SCIENTIFIC_COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "seismic",
    "RdYlBu",
    "RdBu",
    "Spectral",
]


def create_colormap(
    name: str = "viridis",
    device: Optional[torch.device] = None
) -> GPUColormap:
    """
    Factory function for creating GPU colormaps.

    Args:
        name: Colormap name (default: "viridis")
        device: Torch device (default: cuda if available)

    Returns:
        GPUColormap instance

    Example:
        ```python
        cmap = create_colormap("plasma")
        rgb = cmap.apply(values)
        ```
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return GPUColormap(name, device=device)
