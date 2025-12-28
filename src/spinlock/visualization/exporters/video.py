"""
Video exporter for temporal evolution visualizations.

Exports frame sequences as MP4 or GIF videos using torchvision.
"""

import torch
from pathlib import Path
from typing import Optional, Literal


class VideoExporter:
    """
    Export frame sequences as video files.

    Uses torchvision.io.write_video for MP4 encoding with configurable
    codec and frame rate. Handles tensor format conversion automatically.

    Example:
        ```python
        exporter = VideoExporter(fps=30, codec="libx264")
        frames = torch.rand(100, 3, 256, 256)  # [T, 3, H, W]
        exporter.export(frames, output_path=Path("evolution.mp4"))
        ```
    """

    def __init__(
        self,
        fps: int = 30,
        codec: str = "libx264",
        video_codec: Optional[str] = None,
        options: Optional[dict] = None
    ):
        """
        Initialize video exporter.

        Args:
            fps: Frames per second for output video
            codec: Video codec (default: "libx264" for H.264)
            video_codec: Deprecated alias for codec (for backward compatibility)
            options: Additional codec options (e.g., {"crf": "23"})
        """
        self.fps = fps
        self.codec = video_codec if video_codec is not None else codec
        self.options = options or {}

    def export(
        self,
        frames: torch.Tensor,
        output_path: Path,
        quality: Optional[int] = None
    ) -> None:
        """
        Export frames to video file.

        Args:
            frames: Frame tensor [T, 3, H, W] in range [0, 1]
            output_path: Output video file path (.mp4, .avi, etc.)
            quality: Optional quality setting (0-51 for H.264, lower is better)

        Raises:
            ImportError: If torchvision not installed
            ValueError: If frames have wrong shape
        """
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")
                from torchvision.io import write_video
        except ImportError:
            raise ImportError(
                "torchvision required for video export. "
                "Install with: pip install torchvision"
            )

        # Validate input shape
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"Expected frames shape [T, 3, H, W], got {frames.shape}"
            )

        # Convert to uint8 [0, 255] and move to CPU
        # torchvision expects [T, H, W, 3] format
        frames_uint8 = (frames * 255.0).clamp(0, 255).to(torch.uint8).cpu()
        frames_uint8 = frames_uint8.permute(0, 2, 3, 1)  # [T, 3, H, W] -> [T, H, W, 3]

        # Prepare codec options
        options = self.options.copy()
        if quality is not None:
            options["crf"] = str(quality)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write video (suppress torchvision deprecation warning)
        import warnings
        with warnings.catch_warnings():
            # Suppress torchvision video deprecation warnings (migrating to TorchCodec in future)
            warnings.filterwarnings("ignore", category=UserWarning, message=".*video.*deprecated.*")
            write_video(
                str(output_path),
                frames_uint8,
                fps=self.fps,
                video_codec=self.codec,
                options=options
            )


class GIFExporter:
    """
    Export frame sequences as GIF animations.

    Uses PIL for GIF creation with configurable loop count and duration.
    Useful for lightweight animations and web sharing.

    Example:
        ```python
        exporter = GIFExporter(fps=10, loop=0)
        frames = torch.rand(50, 3, 128, 128)  # [T, 3, H, W]
        exporter.export(frames, output_path=Path("evolution.gif"))
        ```
    """

    def __init__(
        self,
        fps: int = 10,
        loop: int = 0,
        optimize: bool = True
    ):
        """
        Initialize GIF exporter.

        Args:
            fps: Frames per second
            loop: Number of loops (0 = infinite)
            optimize: Enable GIF optimization (reduces file size)
        """
        self.fps = fps
        self.loop = loop
        self.optimize = optimize

    def export(
        self,
        frames: torch.Tensor,
        output_path: Path
    ) -> None:
        """
        Export frames to GIF file.

        Args:
            frames: Frame tensor [T, 3, H, W] in range [0, 1]
            output_path: Output GIF file path

        Raises:
            ImportError: If PIL not installed
            ValueError: If frames have wrong shape
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow required for GIF export. "
                "Install with: pip install Pillow"
            )

        # Validate input shape
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"Expected frames shape [T, 3, H, W], got {frames.shape}"
            )

        # Convert to uint8 [0, 255] and move to CPU
        frames_uint8 = (frames * 255.0).clamp(0, 255).to(torch.uint8).cpu()
        frames_uint8 = frames_uint8.permute(0, 2, 3, 1)  # [T, 3, H, W] -> [T, H, W, 3]

        # Convert to PIL images
        pil_frames = [
            Image.fromarray(frame.numpy(), mode="RGB")
            for frame in frames_uint8
        ]

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate duration in milliseconds
        duration_ms = int(1000 / self.fps)

        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=self.loop,
            optimize=self.optimize
        )


def create_exporter(
    format: Literal["mp4", "gif"],
    fps: int = 30,
    **kwargs
):
    """
    Factory function for creating frame exporters.

    Args:
        format: Export format ("mp4" or "gif")
        fps: Frames per second
        **kwargs: Additional format-specific arguments

    Returns:
        VideoExporter or GIFExporter instance

    Example:
        ```python
        # Create MP4 exporter
        exporter = create_exporter("mp4", fps=30, codec="libx264")

        # Create GIF exporter
        exporter = create_exporter("gif", fps=10, loop=0)
        ```
    """
    if format == "mp4":
        return VideoExporter(fps=fps, **kwargs)
    elif format == "gif":
        return GIFExporter(fps=fps, **kwargs)
    else:
        raise ValueError(
            f"Unknown format: {format}. Must be 'mp4' or 'gif'"
        )
