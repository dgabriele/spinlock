"""Interface for using trained MNO models to generate trajectories."""

import torch
from pathlib import Path
from typing import Optional


class TrainedMNO:
    """Interface for using trained MNO models to generate trajectories."""

    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load MNO checkpoint."""
        from spinlock.mno.backbone import MetaOperator

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Load model configuration
        self.config = checkpoint['config']['model']
        self.model = MetaOperator(**self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def generate_trajectory(
        self,
        ic: torch.Tensor,
        params: torch.Tensor,
        timesteps: int = 256
    ) -> torch.Tensor:
        """
        Generate trajectory from initial condition and parameters.

        Args:
            ic: [batch, 1, H, W] initial condition
            params: [batch, 14] parameter vector
            timesteps: Number of timesteps to generate

        Returns:
            trajectory: [batch, timesteps, 1, H, W]
        """
        with torch.no_grad():
            trajectory = self.model(
                ic.to(self.device),
                steps=timesteps,
                return_all_steps=True,
                params=params.to(self.device)
            )
        return trajectory[:, 1:, ...]  # Remove IC from output
