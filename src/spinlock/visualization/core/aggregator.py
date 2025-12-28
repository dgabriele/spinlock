"""
Aggregate renderers for ensemble statistics across realizations.

Provides visualization strategies for summarizing multiple stochastic
realizations: mean field, variance maps, etc.
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional
from .renderer import RenderStrategy, HeatmapRenderer


class AggregateRenderer(ABC):
    """
    Abstract base for ensemble aggregation strategies.

    Aggregates M stochastic realizations into summary visualizations
    (e.g., mean, variance, entropy).
    """

    @abstractmethod
    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Aggregate M realizations to single summary.

        Args:
            realizations: [M, C, H, W] - M realizations

        Returns:
            Aggregated result [C, H, W] or [1, H, W]
        """
        pass

    @abstractmethod
    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Aggregate and render to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        pass


class MeanFieldRenderer(AggregateRenderer):
    """
    Mean field across realizations.

    Computes pixel-wise mean across all M realizations,
    visualizing the expected/average state.

    Example:
        ```python
        renderer = MeanFieldRenderer(base_renderer=rgb_renderer)
        realizations = torch.randn(10, 3, 64, 64)  # 10 realizations
        mean_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        base_renderer: RenderStrategy,
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize mean field renderer.

        Args:
            base_renderer: Renderer for the mean field
            device: Torch device
        """
        self.base_renderer = base_renderer
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute mean field.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Mean field [C, H, W]
        """
        return realizations.mean(dim=0)  # [M, C, H, W] -> [C, H, W]

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render mean field to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        # Compute mean
        mean_field = self.aggregate(realizations)  # [C, H, W]

        # Add batch dimension for renderer
        mean_field = mean_field.unsqueeze(0)  # [1, C, H, W]

        # Render
        rgb = self.base_renderer.render(mean_field)  # [1, 3, H, W]

        # Remove batch dimension
        return rgb.squeeze(0)  # [3, H, W]


class VarianceMapRenderer(AggregateRenderer):
    """
    Variance/uncertainty visualization across realizations.

    Computes spatial variance to show where realizations diverge,
    indicating regions of high uncertainty or stochasticity.

    Example:
        ```python
        renderer = VarianceMapRenderer(colormap="hot")
        realizations = torch.randn(10, 3, 64, 64)  # 10 realizations
        var_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        colormap: str = "hot",
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize variance map renderer.

        Args:
            colormap: Colormap for variance visualization
            device: Torch device
        """
        self.heatmap = HeatmapRenderer(colormap=colormap, device=device)
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute variance magnitude across realizations.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Variance magnitude [1, H, W]
        """
        # Compute variance per channel: [M, C, H, W] -> [C, H, W]
        var_per_channel = realizations.var(dim=0)

        # Compute magnitude (L2 norm across channels): [C, H, W] -> [1, H, W]
        variance_magnitude = var_per_channel.norm(dim=0, keepdim=True)

        return variance_magnitude

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render variance map to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        # Compute variance magnitude
        var_map = self.aggregate(realizations)  # [1, H, W]

        # Add batch dimension
        var_map = var_map.unsqueeze(0)  # [1, 1, H, W]

        # Render as heatmap
        rgb = self.heatmap.render(var_map)  # [1, 3, H, W]

        # Remove batch dimension
        return rgb.squeeze(0)  # [3, H, W]


class StdDevMapRenderer(AggregateRenderer):
    """
    Standard deviation map across realizations.

    Similar to variance but uses standard deviation (square root of variance)
    for more interpretable magnitude visualization.

    Example:
        ```python
        renderer = StdDevMapRenderer(colormap="plasma")
        realizations = torch.randn(10, 3, 64, 64)
        std_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        colormap: str = "plasma",
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize standard deviation map renderer.

        Args:
            colormap: Colormap for std dev visualization
            device: Torch device
        """
        self.heatmap = HeatmapRenderer(colormap=colormap, device=device)
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute standard deviation magnitude.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Std dev magnitude [1, H, W]
        """
        # Compute std dev per channel: [M, C, H, W] -> [C, H, W]
        std_per_channel = realizations.std(dim=0)

        # Compute magnitude: [C, H, W] -> [1, H, W]
        std_magnitude = std_per_channel.norm(dim=0, keepdim=True)

        return std_magnitude

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render std dev map to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        # Compute std dev magnitude
        std_map = self.aggregate(realizations)  # [1, H, W]

        # Add batch dimension
        std_map = std_map.unsqueeze(0)  # [1, 1, H, W]

        # Render as heatmap
        rgb = self.heatmap.render(std_map)  # [1, 3, H, W]

        # Remove batch dimension
        return rgb.squeeze(0)  # [3, H, W]


def create_aggregate_renderer(
    aggregate_type: str,
    base_renderer: Optional[RenderStrategy] = None,
    colormap: str = "hot",
    device: Optional[torch.device] = None
) -> AggregateRenderer:
    """
    Factory function for creating aggregate renderers.

    Args:
        aggregate_type: "mean", "variance", or "stddev"
        base_renderer: Base renderer for mean field (required for "mean")
        colormap: Colormap for variance/stddev
        device: Torch device

    Returns:
        AggregateRenderer instance

    Example:
        ```python
        # Mean field with custom renderer
        renderer = create_aggregate_renderer(
            "mean",
            base_renderer=RGBRenderer()
        )

        # Variance map
        renderer = create_aggregate_renderer(
            "variance",
            colormap="hot"
        )
        ```
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if aggregate_type == "mean":
        if base_renderer is None:
            raise ValueError("base_renderer required for 'mean' aggregate type")
        return MeanFieldRenderer(base_renderer=base_renderer, device=device)

    elif aggregate_type == "variance":
        return VarianceMapRenderer(colormap=colormap, device=device)

    elif aggregate_type == "stddev":
        return StdDevMapRenderer(colormap=colormap, device=device)

    else:
        raise ValueError(
            f"Unknown aggregate type: {aggregate_type}. "
            f"Must be 'mean', 'variance', or 'stddev'"
        )
