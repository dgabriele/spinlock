"""
Video exporter for temporal evolution visualizations.

Exports frame sequences as MP4 or GIF videos using torchvision or PyAV (GPU-accelerated).
"""

import torch
from pathlib import Path
from typing import Optional, Literal
import warnings

# Try to import PyAV for GPU-accelerated encoding
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


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


class NVENCVideoExporter:
    """
    GPU-accelerated video exporter using NVENC hardware encoding.

    Uses PyAV with h264_nvenc or hevc_nvenc codec for GPU-accelerated encoding.
    Falls back to CPU encoding if GPU encoding is unavailable.

    Example:
        ```python
        exporter = NVENCVideoExporter(fps=30, codec="h264_nvenc")
        frames = torch.rand(100, 3, 256, 256)  # [T, 3, H, W]
        exporter.export(frames, output_path=Path("evolution.mp4"))
        ```
    """

    def __init__(
        self,
        fps: int = 30,
        codec: str = "h264_nvenc",
        preset: str = "p4",  # p1 (fastest) to p7 (slowest, best quality)
        bitrate: int = 8_000_000,  # 8 Mbps default
        gpu_device: int = 0
    ):
        """
        Initialize NVENC video exporter.

        Args:
            fps: Frames per second for output video
            codec: Video codec ("h264_nvenc" or "hevc_nvenc")
            preset: NVENC preset (p1-p7, higher is better quality but slower)
            bitrate: Target bitrate in bits/second (default: 8 Mbps)
            gpu_device: GPU device index for encoding (default: 0)
        """
        if not PYAV_AVAILABLE:
            raise ImportError(
                "PyAV required for GPU-accelerated encoding. "
                "Install with: pip install av"
            )

        self.fps = fps
        self.codec = codec
        self.preset = preset
        self.bitrate = bitrate
        self.gpu_device = gpu_device

    def export(
        self,
        frames: torch.Tensor,
        output_path: Path
    ) -> None:
        """
        Export frames to video file using GPU encoding.

        Args:
            frames: Frame tensor [T, 3, H, W] in range [0, 1]
            output_path: Output video file path (.mp4)

        Raises:
            RuntimeError: If GPU encoding fails
            ValueError: If frames have wrong shape
        """
        # Validate input shape
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"Expected frames shape [T, 3, H, W], got {frames.shape}"
            )

        T, C, H, W = frames.shape

        # Convert to uint8 on GPU (minimize CPU transfer)
        frames_uint8 = (frames * 255.0).clamp(0, 255).to(torch.uint8)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open output container
        try:
            container = av.open(str(output_path), mode='w')

            # Add video stream with NVENC codec
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.width = W
            stream.height = H
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = self.bitrate

            # Set NVENC-specific options
            stream.options = {
                'preset': self.preset,
                'gpu': str(self.gpu_device)
            }

            # Encode frames
            for t_idx in range(T):
                # Single frame GPU→CPU (unavoidable for PyAV)
                frame_np = frames_uint8[t_idx].permute(1, 2, 0).cpu().numpy()
                frame_av = av.VideoFrame.from_ndarray(frame_np, format='rgb24')

                # Encode on GPU
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', '.*deprecated.*')
                    for packet in stream.encode(frame_av):
                        container.mux(packet)

            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)

            container.close()

        except Exception as e:
            # If GPU encoding fails, provide helpful error message
            if "h264_nvenc" in str(e) or "nvenc" in str(e).lower():
                raise RuntimeError(
                    f"GPU encoding failed: {e}\n"
                    "NVENC may not be available on this system. "
                    "Falling back to CPU encoding recommended."
                )
            else:
                raise


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


def create_video_exporter_with_gpu_fallback(
    fps: int = 30,
    try_gpu: bool = True,
    verbose: bool = False
):
    """
    Create video exporter with automatic GPU encoding detection and fallback.

    Attempts to use GPU-accelerated encoding (NVENC) if available, falls back
    to CPU encoding (libx264) if GPU encoding is unavailable.

    Args:
        fps: Frames per second
        try_gpu: Whether to attempt GPU encoding (default: True)
        verbose: Print status messages about encoding method

    Returns:
        Video exporter instance (NVENC or CPU-based)

    Example:
        ```python
        exporter = create_video_exporter_with_gpu_fallback(fps=30, verbose=True)
        frames = torch.rand(100, 3, 256, 256)
        exporter.export(frames, Path("output.mp4"))
        ```
    """
    if try_gpu and PYAV_AVAILABLE:
        try:
            # Test if NVENC is available by creating a test encoder
            import tempfile
            test_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            test_path = Path(test_file.name)
            test_file.close()

            try:
                # Try to create NVENC encoder
                container = av.open(str(test_path), mode='w')
                stream = container.add_stream('h264_nvenc', rate=30)
                stream.width = 64
                stream.height = 64
                stream.pix_fmt = 'yuv420p'
                container.close()

                # Success - NVENC is available
                test_path.unlink()  # Clean up test file

                if verbose:
                    print("  Using GPU-accelerated encoding (NVENC)")

                return NVENCVideoExporter(fps=fps)

            except Exception as e:
                # NVENC not available
                test_path.unlink(missing_ok=True)

                if verbose:
                    print(f"  GPU encoding unavailable ({str(e).split(':')[0]}), using CPU encoding")

                return VideoExporter(fps=fps, codec="libx264")

        except Exception:
            # PyAV import succeeded but something else failed
            if verbose:
                print("  GPU encoding test failed, using CPU encoding")
            return VideoExporter(fps=fps, codec="libx264")

    else:
        # GPU encoding disabled or PyAV not available
        if verbose and try_gpu and not PYAV_AVAILABLE:
            print("  PyAV not installed, using CPU encoding (install with: pip install av)")
        elif verbose:
            print("  Using CPU encoding (libx264)")

        return VideoExporter(fps=fps, codec="libx264")


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
