"""Configuration schemas for MNO-VQ-VAE validation."""

from pydantic import BaseModel, ConfigDict, Field


class ValidationConfig(BaseModel):
    """Configuration for MNO-VQ-VAE validation."""

    num_samples: int = Field(
        default=100,
        description="Number of validation samples to test"
    )

    batch_size: int = Field(
        default=8,
        description="Batch size for inference"
    )

    max_reconstruction_ratio: float = Field(
        default=2.0,
        description="Maximum allowed ratio of MNO/CNO reconstruction error"
    )

    device: str = Field(
        default="cuda",
        description="Torch device for inference"
    )

    rollout_steps: int = Field(
        default=256,
        description="Number of timesteps to generate in MNO rollouts"
    )

    model_config = ConfigDict(frozen=True)
