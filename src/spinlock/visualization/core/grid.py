"""
Grid layout manager for multi-operator, multi-realization visualizations.

Composes visualization grids showing:
- Rows: N operators
- Columns: M realizations + K aggregate visualizations
"""

import torch
from typing import Dict, List, Optional
from .renderer import RenderStrategy
from .aggregator import AggregateRenderer


class VisualizationGrid:
    """
    Manages grid layout for operator rollout visualizations.

    Creates grids with:
    - Rows: Different operators
    - Columns: Individual realizations + aggregate statistics

    Example:
        ```python
        grid = VisualizationGrid(
            render_strategy=rgb_renderer,
            aggregate_renderers=[mean_renderer, var_renderer],
            grid_size=64,
            device=torch.device("cuda")
        )

        # Create frame at timestep t
        frame = grid.create_frame(trajectories, timestep=10)
        # frame: [3, N*64, (M+K)*64] RGB image
        ```
    """

    def __init__(
        self,
        render_strategy: RenderStrategy,
        aggregate_renderers: List[AggregateRenderer],
        grid_size: int = 64,
        device: torch.device = torch.device("cuda"),
        add_spacing: bool = False,
        spacing_width: int = 2
    ):
        """
        Initialize visualization grid.

        Args:
            render_strategy: Strategy for rendering individual realizations
            aggregate_renderers: List of aggregate renderers
            grid_size: Spatial grid size (H=W)
            device: Torch device
            add_spacing: Add white spacing between cells (optional)
            spacing_width: Width of spacing in pixels
        """
        self.render_strategy = render_strategy
        self.aggregate_renderers = aggregate_renderers
        self.grid_size = grid_size
        self.device = device
        self.add_spacing = add_spacing
        self.spacing_width = spacing_width

    def create_frame(
        self,
        trajectories: Dict[int, torch.Tensor],  # op_idx -> [T, M, C, H, W]
        timestep: int
    ) -> torch.Tensor:
        """
        Create single frame of visualization grid at given timestep.

        Args:
            trajectories: Dict mapping operator_idx -> trajectory [T, M, C, H, W]
            timestep: Which timestep to visualize

        Returns:
            RGB grid [3, grid_H, grid_W]

        Example:
            ```python
            trajectories = {
                0: torch.randn(100, 10, 3, 64, 64),  # Op 0
                1: torch.randn(100, 10, 3, 64, 64),  # Op 1
                2: torch.randn(100, 10, 3, 64, 64),  # Op 2
            }
            frame = grid.create_frame(trajectories, timestep=50)
            ```
        """
        N = len(trajectories)  # Number of operators
        M = trajectories[0].shape[1]  # Number of realizations
        K = len(self.aggregate_renderers)  # Number of aggregates
        H, W = self.grid_size, self.grid_size

        # Calculate grid dimensions with optional spacing
        spacing = self.spacing_width if self.add_spacing else 0
        grid_H = N * H + (N - 1) * spacing if N > 1 else H
        grid_W = (M + K) * W + (M + K - 1) * spacing if (M + K) > 1 else W

        # Create canvas (white background if spacing enabled)
        grid = torch.ones(3, grid_H, grid_W, dtype=torch.float32, device=self.device)
        if not self.add_spacing:
            grid = torch.zeros(3, grid_H, grid_W, dtype=torch.float32, device=self.device)

        # Render each operator (row)
        for row, op_idx in enumerate(sorted(trajectories.keys())):
            trajectory = trajectories[op_idx]  # [T, M, C, H, W]
            realizations_t = trajectory[timestep]  # [M, C, H, W]

            # Calculate row position
            row_start = row * (H + spacing)
            row_end = row_start + H

            # Render individual realizations
            for col in range(M):
                realization = realizations_t[col:col+1]  # [1, C, H, W]
                rgb = self.render_strategy.render(realization)  # [1, 3, H_orig, W_orig]

                # Resize to match grid cell size if needed
                if rgb.shape[-2:] != (H, W):
                    rgb = torch.nn.functional.interpolate(
                        rgb, size=(H, W), mode='bilinear', align_corners=False
                    )

                # Calculate column position
                col_start = col * (W + spacing)
                col_end = col_start + W

                # Place in grid
                grid[:, row_start:row_end, col_start:col_end] = rgb[0]

            # Render aggregates
            for agg_idx, agg_renderer in enumerate(self.aggregate_renderers):
                col = M + agg_idx
                agg_rgb = agg_renderer.render(realizations_t)  # [3, H_orig, W_orig]

                # Resize to match grid cell size if needed
                if agg_rgb.shape[-2:] != (H, W):
                    agg_rgb = torch.nn.functional.interpolate(
                        agg_rgb.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
                    ).squeeze(0)

                # Calculate column position
                col_start = col * (W + spacing)
                col_end = col_start + W

                # Place in grid
                grid[:, row_start:row_end, col_start:col_end] = agg_rgb

        return grid

    def create_single_frame(
        self,
        realizations: Dict[int, torch.Tensor]  # op_idx -> [M, C, H, W]
    ) -> torch.Tensor:
        """
        Create single frame from pre-extracted realizations (for memory-efficient rendering).

        This method is used for frame-by-frame rendering where only one timestep
        is loaded to GPU at a time, avoiding OOM with large datasets.

        Args:
            realizations: Dict mapping operator_idx -> realizations [M, C, H, W]
                         (already extracted at specific timestep)

        Returns:
            RGB grid [3, grid_H, grid_W]

        Example:
            ```python
            # Extract single timestep from trajectories
            realizations_t = {
                op_idx: traj[:, timestep, :, :, :]  # [M, C, H, W]
                for op_idx, traj in trajectories.items()
            }
            frame = grid.create_single_frame(realizations_t)
            ```
        """
        N = len(realizations)  # Number of operators
        first_real = next(iter(realizations.values()))
        M = first_real.shape[0]  # Number of realizations
        K = len(self.aggregate_renderers)  # Number of aggregates
        H, W = self.grid_size, self.grid_size

        # Calculate grid dimensions with optional spacing
        spacing = self.spacing_width if self.add_spacing else 0
        grid_H = N * H + (N - 1) * spacing if N > 1 else H
        grid_W = (M + K) * W + (M + K - 1) * spacing if (M + K) > 1 else W

        # Create canvas (white background if spacing enabled)
        grid = torch.ones(3, grid_H, grid_W, dtype=torch.float32, device=self.device)
        if not self.add_spacing:
            grid = torch.zeros(3, grid_H, grid_W, dtype=torch.float32, device=self.device)

        # Render each operator (row)
        for row, op_idx in enumerate(sorted(realizations.keys())):
            realizations_op = realizations[op_idx]  # [M, C, H, W]

            # Calculate row position
            row_start = row * (H + spacing)
            row_end = row_start + H

            # Render individual realizations
            for col in range(M):
                realization = realizations_op[col:col+1]  # [1, C, H, W]
                rgb = self.render_strategy.render(realization)  # [1, 3, H_orig, W_orig]

                # Resize to match grid cell size if needed
                if rgb.shape[-2:] != (H, W):
                    rgb = torch.nn.functional.interpolate(
                        rgb, size=(H, W), mode='bilinear', align_corners=False
                    )

                # Calculate column position
                col_start = col * (W + spacing)
                col_end = col_start + W

                # Place in grid
                grid[:, row_start:row_end, col_start:col_end] = rgb[0]

            # Render aggregates
            for agg_idx, agg_renderer in enumerate(self.aggregate_renderers):
                col = M + agg_idx
                agg_rgb = agg_renderer.render(realizations_op)  # [3, H_orig, W_orig]

                # Resize to match grid cell size if needed
                if agg_rgb.shape[-2:] != (H, W):
                    agg_rgb = torch.nn.functional.interpolate(
                        agg_rgb.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
                    ).squeeze(0)

                # Calculate column position
                col_start = col * (W + spacing)
                col_end = col_start + W

                # Place in grid
                grid[:, row_start:row_end, col_start:col_end] = agg_rgb

        return grid

    def create_animation_frames(
        self,
        trajectories: Dict[int, torch.Tensor],
        num_timesteps: Optional[int] = None,
        stride: int = 1
    ) -> torch.Tensor:
        """
        Create all frames for animation.

        Args:
            trajectories: Dict of trajectories [T, M, C, H, W] on GPU
            num_timesteps: Number of timesteps (default: all)
            stride: Sample every Nth timestep

        Returns:
            Frames tensor [T, 3, grid_H, grid_W]

        Example:
            ```python
            # Create frames at every 5th timestep
            frames = grid.create_animation_frames(
                trajectories,
                stride=5
            )
            ```
        """
        # Get total timesteps from first trajectory
        first_traj = next(iter(trajectories.values()))
        T = num_timesteps or first_traj.shape[0]

        # Get grid dimensions from first frame
        sample_frame = self.create_frame(trajectories, timestep=0)
        C, H, W = sample_frame.shape

        # Pre-allocate frames tensor (avoids OOM from list accumulation + stack)
        num_frames = len(range(0, T, stride))
        frames = torch.empty(
            (num_frames, C, H, W),
            dtype=sample_frame.dtype,
            device=sample_frame.device
        )

        # Write first frame
        frames[0] = sample_frame

        # Generate remaining frames directly into pre-allocated tensor
        for i, t in enumerate(range(stride, T, stride), start=1):
            frames[i] = self.create_frame(trajectories, timestep=t)

        return frames

    def get_grid_info(self, trajectories: Dict[int, torch.Tensor]) -> Dict[str, int]:
        """
        Get grid layout information.

        Args:
            trajectories: Dict of trajectories

        Returns:
            Dict with grid layout info

        Example:
            ```python
            info = grid.get_grid_info(trajectories)
            # {'num_operators': 3, 'num_realizations': 10,
            #  'num_aggregates': 2, 'grid_height': 192, 'grid_width': 768}
            ```
        """
        N = len(trajectories)
        M = trajectories[0].shape[1]
        K = len(self.aggregate_renderers)
        H, W = self.grid_size, self.grid_size

        spacing = self.spacing_width if self.add_spacing else 0
        grid_H = N * H + (N - 1) * spacing if N > 1 else H
        grid_W = (M + K) * W + (M + K - 1) * spacing if (M + K) > 1 else W

        return {
            "num_operators": N,
            "num_realizations": M,
            "num_aggregates": K,
            "cell_size": H,
            "grid_height": grid_H,
            "grid_width": grid_W,
            "spacing": spacing
        }
