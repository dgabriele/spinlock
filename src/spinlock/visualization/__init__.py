"""
Visualization system for temporal evolution of stochastic neural operators.

Provides GPU-accelerated rendering with modular strategies for different
channel configurations, ensemble aggregation, and export formats.

Key components:
- Render strategies: HeatmapRenderer, RGBRenderer, PCARenderer
- Aggregate renderers: MeanFieldRenderer, VarianceMapRenderer
- Grid layout manager: VisualizationGrid
- Exporters: VideoExporter, ImageSequenceExporter

Example:
    ```python
    from spinlock.visualization import (
        HeatmapRenderer,
        VisualizationGrid,
        VideoExporter
    )

    # Create renderer
    renderer = HeatmapRenderer(colormap="viridis")

    # Create grid manager
    grid = VisualizationGrid(
        render_strategy=renderer,
        aggregate_renderers=[...],
        grid_size=64
    )

    # Render frames
    frame = grid.create_frame(trajectories, timestep=0)
    ```
"""

from .core.renderer import (
    RenderStrategy,
    HeatmapRenderer,
    RGBRenderer,
    PCARenderer,
    create_render_strategy,
)
from .core.aggregator import (
    AggregateRenderer,
    MeanFieldRenderer,
    VarianceMapRenderer,
    StdDevMapRenderer,
    create_aggregate_renderer,
)
from .core.grid import VisualizationGrid
from .colormaps import GPUColormap, create_colormap, SCIENTIFIC_COLORMAPS
from .exporters.video import VideoExporter
from .exporters.frames import ImageSequenceExporter

__all__ = [
    # Render strategies
    "RenderStrategy",
    "HeatmapRenderer",
    "RGBRenderer",
    "PCARenderer",
    "create_render_strategy",
    # Aggregators
    "AggregateRenderer",
    "MeanFieldRenderer",
    "VarianceMapRenderer",
    "StdDevMapRenderer",
    "create_aggregate_renderer",
    # Layout
    "VisualizationGrid",
    # Utilities
    "GPUColormap",
    "create_colormap",
    "SCIENTIFIC_COLORMAPS",
    # Exporters
    "VideoExporter",
    "ImageSequenceExporter",
]
