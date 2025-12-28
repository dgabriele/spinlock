"""
Image sequence exporter for temporal evolution visualizations.

Exports frame sequences as individual PNG/JPEG images using torchvision.
"""

import torch
from pathlib import Path
from typing import Optional, Literal


class ImageSequenceExporter:
    """
    Export frame sequences as individual image files.

    Uses torchvision.utils.save_image for efficient GPU-to-disk export.
    Each frame is saved as a separate file with sequential numbering.

    Example:
        ```python
        exporter = ImageSequenceExporter(
            format="png",
            prefix="frame",
            padding=4
        )
        frames = torch.rand(100, 3, 256, 256)  # [T, 3, H, W]
        exporter.export(frames, output_dir=Path("frames/"))
        # Creates: frames/frame_0000.png, frame_0001.png, ...
        ```
    """

    def __init__(
        self,
        format: Literal["png", "jpg", "jpeg"] = "png",
        prefix: str = "frame",
        padding: int = 4,
        quality: int = 95
    ):
        """
        Initialize image sequence exporter.

        Args:
            format: Image format ("png", "jpg", "jpeg")
            prefix: Filename prefix for images
            padding: Zero-padding width for frame numbers (e.g., 4 -> "0001")
            quality: JPEG quality (1-100, ignored for PNG)
        """
        self.format = format
        self.prefix = prefix
        self.padding = padding
        self.quality = quality

    def export(
        self,
        frames: torch.Tensor,
        output_dir: Path,
        start_index: int = 0
    ) -> None:
        """
        Export frames to image sequence.

        Args:
            frames: Frame tensor [T, 3, H, W] in range [0, 1]
            output_dir: Output directory for image files
            start_index: Starting index for frame numbering

        Raises:
            ImportError: If torchvision not installed
            ValueError: If frames have wrong shape
        """
        try:
            from torchvision.utils import save_image
        except ImportError:
            raise ImportError(
                "torchvision required for image export. "
                "Install with: pip install torchvision"
            )

        # Validate input shape
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"Expected frames shape [T, 3, H, W], got {frames.shape}"
            )

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export each frame
        T = frames.shape[0]
        for t in range(T):
            frame = frames[t]  # [3, H, W]

            # Generate filename
            frame_idx = start_index + t
            filename = f"{self.prefix}_{frame_idx:0{self.padding}d}.{self.format}"
            output_path = output_dir / filename

            # Save image
            save_image(frame, output_path)

    def export_with_metadata(
        self,
        frames: torch.Tensor,
        output_dir: Path,
        metadata: Optional[dict] = None,
        start_index: int = 0
    ) -> None:
        """
        Export frames with accompanying metadata JSON.

        Args:
            frames: Frame tensor [T, 3, H, W]
            output_dir: Output directory
            metadata: Optional metadata dict (e.g., fps, operator params)
            start_index: Starting frame index

        Example:
            ```python
            exporter.export_with_metadata(
                frames,
                output_dir=Path("frames/"),
                metadata={
                    "fps": 30,
                    "num_frames": 100,
                    "grid_size": 256,
                    "operators": [0, 1, 2],
                    "realizations": 10
                }
            )
            # Creates: frames/frame_*.png + frames/metadata.json
            ```
        """
        import json

        # Export frames
        self.export(frames, output_dir, start_index=start_index)

        # Export metadata if provided
        if metadata is not None:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)


class GridSequenceExporter:
    """
    Export grid visualizations with optional overlays.

    Extends ImageSequenceExporter with grid-specific features like
    cell borders, labels, and annotations.

    Example:
        ```python
        exporter = GridSequenceExporter(
            format="png",
            add_borders=True,
            border_color=(1.0, 1.0, 1.0),
            border_width=2
        )
        frames = torch.rand(100, 3, 768, 768)  # [T, 3, H, W]
        exporter.export(frames, output_dir=Path("grid_frames/"))
        ```
    """

    def __init__(
        self,
        format: Literal["png", "jpg", "jpeg"] = "png",
        prefix: str = "grid",
        padding: int = 4,
        add_borders: bool = False,
        border_color: tuple = (1.0, 1.0, 1.0),
        border_width: int = 2
    ):
        """
        Initialize grid sequence exporter.

        Args:
            format: Image format
            prefix: Filename prefix
            padding: Frame number padding
            add_borders: Add grid cell borders (future feature)
            border_color: RGB border color in [0, 1]
            border_width: Border width in pixels
        """
        self.base_exporter = ImageSequenceExporter(
            format=format,
            prefix=prefix,
            padding=padding
        )
        self.add_borders = add_borders
        self.border_color = torch.tensor(border_color).view(3, 1, 1)
        self.border_width = border_width

    def export(
        self,
        frames: torch.Tensor,
        output_dir: Path,
        grid_info: Optional[dict] = None,
        start_index: int = 0
    ) -> None:
        """
        Export grid frames with optional borders.

        Args:
            frames: Frame tensor [T, 3, H, W]
            output_dir: Output directory
            grid_info: Optional grid layout info (num_rows, num_cols, etc.)
            start_index: Starting frame index
        """
        # Apply borders if requested
        if self.add_borders and grid_info is not None:
            frames = self._add_grid_borders(frames, grid_info)

        # Export using base exporter
        self.base_exporter.export(frames, output_dir, start_index=start_index)

    def _add_grid_borders(
        self,
        frames: torch.Tensor,
        grid_info: dict
    ) -> torch.Tensor:
        """
        Add grid cell borders to frames.

        Args:
            frames: [T, 3, H, W]
            grid_info: Dict with num_rows, num_cols, cell_size

        Returns:
            Frames with borders [T, 3, H, W]
        """
        # TODO: Implement border overlay
        # This would draw lines between grid cells
        # For now, return frames unchanged
        return frames


def create_frame_exporter(
    exporter_type: Literal["sequence", "grid"] = "sequence",
    **kwargs
):
    """
    Factory function for creating frame exporters.

    Args:
        exporter_type: Type of exporter ("sequence" or "grid")
        **kwargs: Additional arguments for exporter

    Returns:
        ImageSequenceExporter or GridSequenceExporter

    Example:
        ```python
        # Basic sequence exporter
        exporter = create_frame_exporter("sequence", format="png")

        # Grid exporter with borders
        exporter = create_frame_exporter(
            "grid",
            format="png",
            add_borders=True
        )
        ```
    """
    if exporter_type == "sequence":
        return ImageSequenceExporter(**kwargs)
    elif exporter_type == "grid":
        return GridSequenceExporter(**kwargs)
    else:
        raise ValueError(
            f"Unknown exporter type: {exporter_type}. "
            f"Must be 'sequence' or 'grid'"
        )
