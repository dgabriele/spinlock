"""Creates 4-panel comparison visualizations for trajectory in-painting."""

from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spinlock.visualization.core.renderer import HeatmapRenderer
from spinlock.visualization.exporters.frames import ImageSequenceExporter
from spinlock.visualization.exporters.video import VideoExporter


class ComparisonVisualizer:
    """Creates 4-panel comparison visualizations."""

    def __init__(
        self,
        colormap: str = "seismic",
        device: str = "cuda"
    ):
        """
        Initialize visualizer.

        Args:
            colormap: Matplotlib colormap name
            device: Device for rendering
        """
        self.renderer = HeatmapRenderer(
            colormap=colormap,
            percentile_clip=(0.01, 0.99),
            device=device
        )
        self.frame_exporter = ImageSequenceExporter(format="png", padding=4)
        self.video_exporter = VideoExporter(fps=24, codec="libx264")
        self.device = device

    def create_comparison_frames(
        self,
        original_trajectory: torch.Tensor,
        masked_trajectory: torch.Tensor,
        reconstructed_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create 4-panel comparison frames.

        Args:
            original_trajectory: [T, H, W] CNO field
            masked_trajectory: [T, H, W] with NaN for masked regions
            reconstructed_trajectory: [T, H, W]

        Returns:
            [T, 3, 2H, 2W] RGB frames arranged as:
                [Original      | Masked        ]
                [Reconstructed | Error Map     ]
        """
        T, H, W = original_trajectory.shape

        # Render each panel
        original_rgb = self._render_field(original_trajectory)
        masked_rgb = self._render_masked(masked_trajectory)
        reconstructed_rgb = self._render_field(reconstructed_trajectory)
        error_rgb = self._render_error(original_trajectory, reconstructed_trajectory)

        # Combine into 2x2 grid
        top_row = torch.cat([original_rgb, masked_rgb], dim=3)  # [T, 3, H, 2W]
        bottom_row = torch.cat([reconstructed_rgb, error_rgb], dim=3)
        grid = torch.cat([top_row, bottom_row], dim=2)  # [T, 3, 2H, 2W]

        # Add panel labels
        grid = self._add_panel_labels(
            grid,
            ["Original", "Masked", "Reconstructed", "Error"]
        )

        return grid

    def export_visualization(
        self,
        frames: torch.Tensor,
        output_dir: Path,
        name: str,
        format: str = "both",
        fps: int = 24
    ):
        """
        Export frames as PNG sequence and/or video.

        Args:
            frames: [T, 3, H, W] RGB frames
            output_dir: Output directory
            name: Base name for outputs
            format: "frames", "video", or "both"
            fps: Frames per second for video
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if format in ["frames", "both"]:
            frame_dir = output_dir / f"{name}_frames"
            self.frame_exporter.export(frames, frame_dir)
            print(f"  ✓ Frames exported: {frame_dir}")

        if format in ["video", "both"]:
            video_path = output_dir / f"{name}.mp4"
            self.video_exporter.export(frames, video_path, fps=fps)
            print(f"  ✓ Video exported: {video_path}")

    def _render_field(self, field: torch.Tensor) -> torch.Tensor:
        """
        Render CNO field to RGB via colormap.

        Args:
            field: [T, H, W]

        Returns:
            [T, 3, H, W] RGB
        """
        # Add channel dimension: [T, H, W] → [T, 1, H, W]
        field_4d = field.unsqueeze(1)
        return self.renderer.render(field_4d)

    def _render_masked(self, masked_field: torch.Tensor) -> torch.Tensor:
        """
        Render masked trajectory with grey overlay for NaN regions.

        Args:
            masked_field: [T, H, W] with NaN for masked

        Returns:
            [T, 3, H, W] RGB
        """
        # Replace NaN with 0, then overlay grey mask
        valid_mask = ~torch.isnan(masked_field)
        filled = torch.where(
            valid_mask,
            masked_field,
            torch.zeros_like(masked_field)
        )
        rgb = self._render_field(filled)

        # Grey out masked regions (0.5 = medium grey)
        grey = 0.5 * torch.ones_like(rgb)
        rgb = torch.where(
            valid_mask.unsqueeze(1),  # [T, 1, H, W]
            rgb,
            grey
        )
        return rgb

    def _render_error(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Render absolute error map.

        Args:
            original: [T, H, W]
            reconstructed: [T, H, W]

        Returns:
            [T, 3, H, W] RGB error map
        """
        error = (original - reconstructed).abs()
        return self._render_field(error)

    def _add_panel_labels(
        self,
        grid: torch.Tensor,
        labels: List[str]
    ) -> torch.Tensor:
        """
        Add text labels to each panel.

        Args:
            grid: [T, 3, 2H, 2W] RGB frames
            labels: List of 4 labels [top-left, top-right, bottom-left, bottom-right]

        Returns:
            [T, 3, 2H, 2W] RGB frames with labels
        """
        T, C, H_total, W_total = grid.shape
        H = H_total // 2
        W = W_total // 2

        # Convert to numpy for PIL processing
        grid_np = (grid * 255).byte().cpu().permute(0, 2, 3, 1).numpy()

        # Process each frame
        labeled_frames = []
        for t in range(T):
            # Convert to PIL
            img = Image.fromarray(grid_np[t])
            draw = ImageDraw.Draw(img)

            # Try to load a font (fallback to default)
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Label positions (top-left corner of each panel)
            positions = [
                (10, 10),           # Original
                (W + 10, 10),       # Masked
                (10, H + 10),       # Reconstructed
                (W + 10, H + 10)    # Error
            ]

            # Draw labels with black background for visibility
            for label, pos in zip(labels, positions):
                # Get text bounding box
                bbox = draw.textbbox(pos, label, font=font)
                # Draw background rectangle
                draw.rectangle(bbox, fill='black')
                # Draw text in white
                draw.text(pos, label, fill='white', font=font)

            # Convert back to tensor
            labeled_np = np.array(img)
            labeled_frames.append(labeled_np)

        # Stack and convert back to tensor format
        labeled_frames = np.stack(labeled_frames)  # [T, H, W, 3]
        labeled_tensor = torch.from_numpy(labeled_frames).permute(0, 3, 1, 2).float() / 255.0

        return labeled_tensor

    def generate_error_analysis(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        output_path: Path
    ):
        """
        Generate error analysis plots.

        Args:
            original: [T, H, W]
            reconstructed: [T, H, W]
            output_path: Path to save plot
        """
        import matplotlib.pyplot as plt

        # Compute metrics
        mse_per_frame = ((original - reconstructed) ** 2).mean(dim=(1, 2))
        l2_per_frame = ((original - reconstructed) ** 2).sum(dim=(1, 2)).sqrt()

        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(mse_per_frame.cpu().numpy(), linewidth=2)
        axes[0].set_title("MSE Error Over Time", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Timestep", fontsize=12)
        axes[0].set_ylabel("MSE", fontsize=12)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(l2_per_frame.cpu().numpy(), linewidth=2, color='orange')
        axes[1].set_title("L2 Error Over Time", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Timestep", fontsize=12)
        axes[1].set_ylabel("L2 Norm", fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Error analysis saved: {output_path}")
