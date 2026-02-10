"""Trajectory generation from decoded tokens."""

from pathlib import Path
from typing import Optional
import torch
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from spinlock.mno.cno_replay import CNOReplayer


class TrajectoryGenerator:
    """Generate PDE trajectories from decoded parameters and ICs.

    Supports:
    - CNOReplayer for ground truth rollouts (default)
    - MNO for learned operator rollouts (future)

    Args:
        cno_config_path: Path to CNO config YAML
        device: Computation device
        mno_checkpoint: Optional MNO checkpoint path (future)
    """

    def __init__(
        self,
        cno_config_path: Path,
        device: str = "cuda",
        mno_checkpoint: Optional[Path] = None
    ):
        self.device = device
        self.mno_checkpoint = mno_checkpoint

        # Initialize CNOReplayer
        self.replayer = CNOReplayer.from_config(str(cno_config_path))
        self.replayer.device = torch.device(device)

        # MNO support (future)
        self.mno = None
        if mno_checkpoint is not None:
            self._load_mno(mno_checkpoint)

    def generate_trajectories(
        self,
        theta: torch.Tensor,
        u0: torch.Tensor,
        num_steps: int = 256,
        num_realizations: int = 1,
        use_mno: bool = False,
        batch_size: int = 8
    ) -> torch.Tensor:
        """Generate trajectories from parameters and ICs.

        Args:
            theta: Operator parameters [B, 14] in [0,1]
            u0: Initial conditions [B, C, H, W]
            num_steps: Rollout length
            num_realizations: Stochastic realizations
            use_mno: Use MNO instead of CNOReplayer
            batch_size: Batch size for processing

        Returns:
            Trajectories [B, M, T+1, C, H, W]
        """
        if use_mno:
            if self.mno is None:
                raise ValueError("MNO not loaded. Provide mno_checkpoint to use MNO.")
            return self._generate_with_mno(
                theta, u0, num_steps, num_realizations, batch_size
            )
        else:
            return self._generate_with_cno(
                theta, u0, num_steps, num_realizations, batch_size
            )

    def _generate_with_cno(
        self,
        theta: torch.Tensor,
        u0: torch.Tensor,
        num_steps: int,
        num_realizations: int,
        batch_size: int
    ) -> torch.Tensor:
        """Generate using CNOReplayer (ground truth)."""
        B = theta.shape[0]
        C, H, W = u0.shape[1:]

        all_trajectories = []

        # Process in batches
        num_batches = (B + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating CNO trajectories"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, B)
            batch_theta = theta[start_idx:end_idx]
            batch_u0 = u0[start_idx:end_idx]

            batch_trajectories = self._rollout_batch_cno(
                batch_theta, batch_u0, num_steps, num_realizations
            )
            all_trajectories.append(batch_trajectories)

        return torch.cat(all_trajectories, dim=0)

    def _rollout_batch_cno(
        self,
        theta_batch: torch.Tensor,
        u0_batch: torch.Tensor,
        num_steps: int,
        num_realizations: int
    ) -> torch.Tensor:
        """Rollout single batch using CNOReplayer."""
        batch_size = theta_batch.shape[0]
        C, H, W = u0_batch.shape[1:]

        batch_trajectories = []

        for i in range(batch_size):
            params_vector = theta_batch[i].cpu().numpy()  # [14]
            ic = u0_batch[i]  # Keep as tensor [C, H, W]

            # Rollout with CNOReplayer (expects torch tensor for ic)
            trajectory = self.replayer.rollout(
                params_vector=params_vector,
                ic=ic,
                timesteps=num_steps,
                num_realizations=num_realizations
            )  # [M, T+1, C, H, W] as torch tensor

            batch_trajectories.append(trajectory.cpu())

        # Stack to tensor [B, M, T+1, C, H, W]
        return torch.stack(batch_trajectories, dim=0)

    def _generate_with_mno(
        self,
        theta: torch.Tensor,
        u0: torch.Tensor,
        num_steps: int,
        num_realizations: int,
        batch_size: int
    ) -> torch.Tensor:
        """Generate using MNO (future implementation)."""
        raise NotImplementedError(
            "MNO trajectory generation not yet implemented. "
            "Use use_mno=False for CNOReplayer rollouts."
        )

    def _load_mno(self, checkpoint_path: Path):
        """Load MNO model (future implementation)."""
        raise NotImplementedError(
            "MNO loading not yet implemented. "
            "Will be added when MNO training is complete."
        )
