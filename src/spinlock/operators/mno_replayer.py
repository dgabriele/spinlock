"""MNO Replayer for generating rollouts from trained meta-operator.

Used in Stage 2 of two-stage training to generate training data for VQ-VAE.
Parallel to CNOReplayer but uses MNO instead of CNO.
"""

import torch
from pathlib import Path


class MNOReplayer:
    """Generate rollouts from trained MNO checkpoint.

    Loads a trained MNO meta-operator (from Stage 1) and generates
    rollouts for VQ-VAE training (Stage 2).

    Parallel interface to CNOReplayer for seamless integration.
    """

    def __init__(self, mno_checkpoint_path: str, device: str = "cuda"):
        """Load trained MNO from checkpoint.

        Args:
            mno_checkpoint_path: Path to Stage 1 meta-operator checkpoint
            device: Device to run on (default: cuda)
        """
        from spinlock.mno import MNOBackbone

        checkpoint = torch.load(mno_checkpoint_path, map_location=device)

        # Validate checkpoint format
        if "config" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint: missing 'config' field.\n"
                f"Expected Stage 1 checkpoint format (from train-meta-operator)."
            )

        if "model" not in checkpoint["config"]:
            raise ValueError(
                f"Invalid checkpoint: missing 'config.model' field.\n"
                f"Expected Stage 1 checkpoint format with model configuration."
            )

        # Reconstruct MNO from checkpoint config
        model_config = checkpoint["config"]["model"]
        self.mno = MNOBackbone(**model_config)
        self.mno.load_state_dict(checkpoint["model_state_dict"])
        self.mno = self.mno.to(device).eval()
        self.device = device

        # Store checkpoint metadata
        self.checkpoint_path = mno_checkpoint_path
        self.checkpoint_epoch = checkpoint.get("epoch", "unknown")
        self.checkpoint_val_loss = checkpoint.get("val_loss", "unknown")

        print(f"MNOReplayer initialized:")
        print(f"  Checkpoint: {mno_checkpoint_path}")
        print(f"  Epoch: {self.checkpoint_epoch}")
        print(f"  Val Loss: {self.checkpoint_val_loss}")
        print(f"  Device: {device}")

    @torch.no_grad()
    def rollout(
        self,
        ic: torch.Tensor,
        timesteps: int,
        num_realizations: int = 1,
        return_all_steps: bool = True,
    ) -> torch.Tensor:
        """Generate MNO trajectory.

        Args:
            ic: Initial condition [B, C, H, W] or [C, H, W]
            timesteps: Number of timesteps to rollout
            num_realizations: Number of realizations (for ensemble, not used by MNO)
            return_all_steps: Return all timesteps (True) or only final (False)

        Returns:
            Trajectory [B, T, C, H, W] if return_all_steps=True (excluding IC)
            Final state [B, C, H, W] if return_all_steps=False
        """
        # Normalize input shape
        if ic.dim() == 3:
            ic = ic.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

        ic = ic.to(self.device)

        # Generate MNO rollout
        # Note: MNO is deterministic, so num_realizations is ignored
        if num_realizations > 1:
            import warnings
            warnings.warn(
                f"MNO is deterministic, ignoring num_realizations={num_realizations}. "
                f"Will return single realization."
            )

        trajectory = self.mno(
            ic,
            steps=timesteps,
            return_all_steps=return_all_steps,
        )

        if return_all_steps:
            # Return only future states (exclude IC at t=0)
            return trajectory[:, 1:, :, :, :]  # [B, T, C, H, W]
        else:
            # Return final state only
            return trajectory  # [B, C, H, W]

    def get_checkpoint_info(self) -> dict:
        """Get checkpoint metadata.

        Returns:
            Dictionary with checkpoint information
        """
        return {
            "checkpoint_path": self.checkpoint_path,
            "epoch": self.checkpoint_epoch,
            "val_loss": self.checkpoint_val_loss,
            "device": str(self.device),
        }

    def __repr__(self) -> str:
        return (
            f"MNOReplayer(checkpoint='{self.checkpoint_path}', "
            f"epoch={self.checkpoint_epoch}, val_loss={self.checkpoint_val_loss})"
        )
