"""
Render strategies for converting multi-channel data to RGB.

Provides polymorphic rendering strategies for different channel counts:
- HeatmapRenderer: C=1 (single channel to RGB via colormap)
- RGBRenderer: C=3 (direct RGB mapping)
- PCARenderer: C≥3 (PCA dimensionality reduction to RGB)
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from ..colormaps import GPUColormap


class RenderStrategy(ABC):
    """
    Abstract base class for channel-to-RGB rendering strategies.

    All renderers implement the Strategy pattern, allowing polymorphic
    rendering of different channel configurations.
    """

    @abstractmethod
    def render(self, data: torch.Tensor) -> torch.Tensor:
        """
        Render multi-channel data to RGB.

        Args:
            data: Input tensor [B, C, H, W]

        Returns:
            RGB tensor [B, 3, H, W] in range [0, 1]
        """
        pass

    @abstractmethod
    def supports_channels(self, num_channels: int) -> bool:
        """Check if strategy supports given channel count."""
        pass

    def _normalize(
        self,
        x: torch.Tensor,
        percentile_clip: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        DRY normalization shared across all renderers.

        Args:
            x: Tensor to normalize
            percentile_clip: Optional (low, high) percentile clipping

        Returns:
            Normalized tensor in [0, 1]
        """
        if percentile_clip:
            low, high = percentile_clip
            vmin = torch.quantile(x.flatten(), low)
            vmax = torch.quantile(x.flatten(), high)
        else:
            vmin, vmax = x.min(), x.max()

        # Avoid division by zero
        if (vmax - vmin).abs() < 1e-8:
            return torch.zeros_like(x)

        return (x - vmin) / (vmax - vmin)


class HeatmapRenderer(RenderStrategy):
    """
    Single-channel to RGB via colormap (C=1).

    Uses GPU-resident colormap lookup tables for efficient rendering.
    Ideal for scalar fields like temperature, pressure, or density.

    Example:
        ```python
        renderer = HeatmapRenderer(colormap="viridis")
        data = torch.randn(10, 1, 64, 64)  # [B, 1, H, W]
        rgb = renderer.render(data)  # [10, 3, 64, 64]
        ```
    """

    def __init__(
        self,
        colormap: str = "viridis",
        device: torch.device = torch.device("cuda"),
        percentile_clip: Optional[Tuple[float, float]] = (0.01, 0.99)
    ):
        """
        Initialize heatmap renderer.

        Args:
            colormap: Matplotlib colormap name
            device: Torch device
            percentile_clip: Percentile clipping for robust normalization
        """
        self.colormap = GPUColormap(colormap, device=device)
        self.device = device
        self.percentile_clip = percentile_clip

    def render(self, data: torch.Tensor) -> torch.Tensor:
        """
        Render single channel to RGB via colormap.

        Args:
            data: [B, 1, H, W]

        Returns:
            [B, 3, H, W]
        """
        if data.shape[1] != 1:
            raise ValueError(f"HeatmapRenderer expects 1 channel, got {data.shape[1]}")

        # Normalize to [0, 1]
        normalized = self._normalize(data, self.percentile_clip)

        # Remove channel dimension: [B, 1, H, W] -> [B, H, W]
        normalized = normalized.squeeze(1)

        # Apply colormap: [B, H, W] -> [B, 3, H, W]
        rgb = self.colormap.apply(normalized)

        return rgb

    def supports_channels(self, num_channels: int) -> bool:
        return num_channels == 1


class RGBRenderer(RenderStrategy):
    """
    Direct RGB mapping (C=3).

    Maps three channels directly to RGB with normalization.
    Ideal for natural images or pre-composed RGB visualizations.

    Example:
        ```python
        renderer = RGBRenderer()
        data = torch.randn(10, 3, 64, 64)  # [B, 3, H, W]
        rgb = renderer.render(data)  # [10, 3, 64, 64]
        ```
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        percentile_clip: Optional[Tuple[float, float]] = (0.01, 0.99)
    ):
        """
        Initialize RGB renderer.

        Args:
            device: Torch device
            percentile_clip: Percentile clipping for robust normalization
        """
        self.device = device
        self.percentile_clip = percentile_clip

    def render(self, data: torch.Tensor) -> torch.Tensor:
        """
        Render RGB channels with normalization.

        Args:
            data: [B, 3, H, W]

        Returns:
            [B, 3, H, W] normalized to [0, 1]
        """
        if data.shape[1] != 3:
            raise ValueError(f"RGBRenderer expects 3 channels, got {data.shape[1]}")

        # Normalize to [0, 1]
        return self._normalize(data, self.percentile_clip)

    def supports_channels(self, num_channels: int) -> bool:
        return num_channels == 3


class PCARenderer(RenderStrategy):
    """
    PCA-based dimensionality reduction to RGB (C≥3).

    Projects C-dimensional data to 3D RGB space using GPU-accelerated PCA.
    Preserves maximum variance in the first three principal components.

    Example:
        ```python
        # Fit PCA on training data
        renderer = PCARenderer()
        renderer.fit(training_data)  # [N, C, H, W]

        # Render test data
        data = torch.randn(10, 5, 64, 64)  # [B, 5, H, W]
        rgb = renderer.render(data)  # [10, 3, 64, 64]
        ```
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        percentile_clip: Optional[Tuple[float, float]] = (0.01, 0.99)
    ):
        """
        Initialize PCA renderer.

        Args:
            device: Torch device
            percentile_clip: Percentile clipping for robust normalization
        """
        self.device = device
        self.percentile_clip = percentile_clip
        self.pca_components = None
        self.pca_mean = None

    def fit(self, data: torch.Tensor, max_samples: int = 10000):
        """
        Fit PCA on data.

        Args:
            data: Training data [N, C, H, W]
            max_samples: Maximum pixels to use for PCA (for efficiency)
        """
        N, C, H, W = data.shape

        # Reshape to [N*H*W, C]
        X = data.permute(0, 2, 3, 1).reshape(-1, C).to(self.device)

        # Subsample if too many pixels
        if X.shape[0] > max_samples:
            indices = torch.randperm(X.shape[0], device=self.device)[:max_samples]
            X = X[indices]

        # Center data
        self.pca_mean = X.mean(dim=0, keepdim=True)
        X_centered = X - self.pca_mean

        # GPU-accelerated SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        # Keep top 3 components
        self.pca_components = Vt[:3, :]  # [3, C]

    def render(self, data: torch.Tensor) -> torch.Tensor:
        """
        Project C channels to RGB via PCA.

        Args:
            data: [B, C, H, W]

        Returns:
            [B, 3, H, W]
        """
        if self.pca_components is None:
            # Auto-fit on input data if not fitted
            self.fit(data)

        B, C, H, W = data.shape

        # Reshape to [B*H*W, C]
        X = data.permute(0, 2, 3, 1).reshape(-1, C).to(self.device)

        # Center
        X_centered = X - self.pca_mean

        # Project to RGB space: [B*H*W, 3]
        rgb_flat = torch.mm(X_centered, self.pca_components.T)

        # Reshape to [B, H, W, 3] -> [B, 3, H, W]
        rgb = rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        return self._normalize(rgb, self.percentile_clip)

    def supports_channels(self, num_channels: int) -> bool:
        return num_channels >= 3


def create_render_strategy(
    num_channels: int,
    strategy: str = "auto",
    colormap: str = "viridis",
    device: Optional[torch.device] = None,
    **kwargs
) -> RenderStrategy:
    """
    Factory function for creating render strategies.

    Args:
        num_channels: Number of input channels
        strategy: "auto", "heatmap", "rgb", or "pca"
        colormap: Colormap for heatmap renderer
        device: Torch device
        **kwargs: Additional args for renderers

    Returns:
        RenderStrategy instance

    Example:
        ```python
        # Auto-select based on channels
        renderer = create_render_strategy(num_channels=5)

        # Explicit selection
        renderer = create_render_strategy(
            num_channels=1,
            strategy="heatmap",
            colormap="plasma"
        )
        ```
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if strategy == "auto":
        if num_channels == 1:
            return HeatmapRenderer(colormap=colormap, device=device, **kwargs)
        elif num_channels == 3:
            return RGBRenderer(device=device, **kwargs)
        else:
            return PCARenderer(device=device, **kwargs)

    elif strategy == "heatmap":
        return HeatmapRenderer(colormap=colormap, device=device, **kwargs)

    elif strategy == "rgb":
        return RGBRenderer(device=device, **kwargs)

    elif strategy == "pca":
        return PCARenderer(device=device, **kwargs)

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Must be 'auto', 'heatmap', 'rgb', or 'pca'"
        )
