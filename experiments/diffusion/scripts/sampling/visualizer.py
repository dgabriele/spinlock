"""Visualization utilities for samples."""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class SampleVisualizer:
    """Create visualizations for generated samples."""

    def plot_trajectory(
        self,
        trajectory: np.ndarray,
        theta: np.ndarray,
        output_path: Path
    ):
        """Plot single trajectory evolution.

        Args:
            trajectory: [M, T+1, C, H, W]
            theta: [14] parameter vector
            output_path: Output file path
        """
        fig = self._create_trajectory_figure(trajectory, theta)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_parameter_distribution(
        self,
        theta_samples: np.ndarray,
        output_path: Path
    ):
        """Plot parameter distributions.

        Args:
            theta_samples: [N, 14]
            output_path: Output file path
        """
        fig = self._create_param_hist_figure(theta_samples)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_diversity_comparison(
        self,
        trajectories: list,
        output_path: Path,
        num_samples: int = 10
    ):
        """Compare trajectory diversity.

        Args:
            trajectories: List of trajectory arrays
            output_path: Output file path
            num_samples: Number of trajectories to plot
        """
        fig = self._create_diversity_figure(trajectories, num_samples)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _create_trajectory_figure(
        self,
        trajectory: np.ndarray,
        theta: np.ndarray
    ) -> Figure:
        """Create trajectory plot."""
        M, T, C, H, W = trajectory.shape
        num_timesteps = min(8, T)
        timesteps = np.linspace(0, T-1, num_timesteps).astype(int)

        fig, axes = plt.subplots(2, num_timesteps//2, figsize=(16, 6))
        axes = axes.flatten()

        for idx, t in enumerate(timesteps):
            frame = trajectory[0, t, 0]  # First realization, first channel
            im = axes[idx].imshow(frame, cmap='viridis')
            axes[idx].set_title(f't={t}')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)

        plt.suptitle(f'Trajectory Evolution (θ sample)', fontsize=14)
        plt.tight_layout()

        return fig

    def _create_param_hist_figure(self, theta_samples: np.ndarray) -> Figure:
        """Create parameter histogram."""
        N, param_dim = theta_samples.shape

        fig, axes = plt.subplots(4, 4, figsize=(14, 10))
        axes = axes.flatten()

        for i in range(min(param_dim, 16)):
            axes[i].hist(theta_samples[:, i], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(f'θ_{i}')
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'Parameter {i}')
            axes[i].grid(alpha=0.3)

        # Hide extra subplots
        for i in range(param_dim, 16):
            axes[i].axis('off')

        plt.suptitle(f'Parameter Distributions (N={N})', fontsize=14)
        plt.tight_layout()

        return fig

    def _create_diversity_figure(
        self,
        trajectories: list,
        num_samples: int
    ) -> Figure:
        """Create diversity comparison plot."""
        num_samples = min(num_samples, len(trajectories))
        num_timesteps = 5

        fig, axes = plt.subplots(num_samples, num_timesteps, figsize=(15, 3*num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            traj = trajectories[i]
            T = traj.shape[1]
            timesteps = np.linspace(0, T-1, num_timesteps).astype(int)

            for j, t in enumerate(timesteps):
                frame = traj[0, t, 0]  # First realization, first channel
                axes[i, j].imshow(frame, cmap='viridis')
                axes[i, j].axis('off')
                if i == 0:
                    axes[i, j].set_title(f't={t}')

        plt.suptitle(f'Trajectory Diversity (N={num_samples})', fontsize=14)
        plt.tight_layout()

        return fig
