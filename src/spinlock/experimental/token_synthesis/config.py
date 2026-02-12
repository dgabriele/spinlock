"""Configuration for token synthesis self-play pipeline.

Defines the full configuration hierarchy for the explore-refine loop:
- CheckpointPaths: All model checkpoint locations
- GenerationConfig: Token generation parameters
- RolloutConfig: QBM simulation parameters
- SurprisalConfig: Verification scoring parameters
- PriorityConfig: Queue scoring weights and capacity
- SchedulerConfig: Explore/refine mode switching
- RefinementConfig: Diffusion model fine-tuning parameters
- TokenSynthesisConfig: Top-level aggregator
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class CheckpointPaths(BaseModel):
    """Paths to all model checkpoints required by the pipeline.

    theta_inverse_path and initial_inverse_path are optional: when omitted,
    the VQTokenizer uses its built-in inverse heads (trained via roundtrip
    loss during VQ-VAE training). External inverse models should only be
    provided when they were specifically trained for this tokenizer.
    """
    diffusion_checkpoint: Path
    vqvae_checkpoint: Path
    theta_inverse_path: Optional[Path] = None
    initial_inverse_path: Optional[Path] = None
    qbm_substrate_config: Path

    @field_validator(
        "diffusion_checkpoint", "vqvae_checkpoint",
        "qbm_substrate_config",
    )
    @classmethod
    def validate_path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path not found: {v}")
        return v

    @field_validator("theta_inverse_path", "initial_inverse_path")
    @classmethod
    def validate_optional_path_exists(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"Path not found: {v}")
        return v


class GenerationConfig(BaseModel):
    """Token generation parameters for exploration."""
    batch_size: int = Field(default=8, ge=1)
    sampling_mode: Literal["unconditional", "partial_mask"] = "unconditional"
    mask_probability: float = Field(default=0.5, ge=0.0, le=1.0)


class RolloutConfig(BaseModel):
    """QBM simulation parameters for physical grounding."""
    num_realizations: int = Field(default=3, ge=1)
    num_timesteps: int = Field(default=256, ge=1)
    rollout_batch_size: int = Field(default=4, ge=1)


class SurprisalConfig(BaseModel):
    """Verification scoring parameters.

    surprisal = (1 - Jaccard) + lambda_entropy * entropy(retokenized)

    Tokens with surprisal > threshold are considered novel discoveries.
    """
    lambda_entropy: float = Field(default=0.2, ge=0.0)
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    verification_samples: int = Field(default=3, ge=1)
    max_variance: float = Field(default=0.15, ge=0.0)


class PriorityConfig(BaseModel):
    """Priority queue scoring weights and capacity.

    priority = alpha * surprisal + beta * usage_frequency + gamma * rarity

    Higher priority items are refined first.
    """
    alpha: float = Field(default=0.6, ge=0.0)
    beta: float = Field(default=0.25, ge=0.0)
    gamma: float = Field(default=0.15, ge=0.0)
    queue_capacity: int = Field(default=1000, ge=1)
    min_queue_for_refinement: int = Field(default=64, ge=1)


class SchedulerConfig(BaseModel):
    """Explore/refine mode scheduling.

    Fixed mode: alternate explore_steps exploration batches with
    refine_epochs refinement epochs.

    Adaptive mode: additionally check queue fill level to switch
    modes early (high fill → refine, low fill → explore).
    """
    explore_steps: int = Field(default=5, ge=1)
    refine_epochs: int = Field(default=3, ge=1)
    max_cycles: int = Field(default=20, ge=1)
    adaptive: bool = True
    queue_threshold_high: float = Field(default=0.8, ge=0.0, le=1.0)
    queue_threshold_low: float = Field(default=0.2, ge=0.0, le=1.0)


class RefinementConfig(BaseModel):
    """Diffusion model fine-tuning parameters."""
    learning_rate: float = Field(default=1e-5, gt=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    batch_size: int = Field(default=16, ge=1)
    replay_buffer_size: int = Field(default=5000, ge=1)


class TokenSynthesisConfig(BaseModel):
    """Top-level configuration for token synthesis self-play pipeline."""
    checkpoints: CheckpointPaths
    generation: GenerationConfig = GenerationConfig()
    rollout: RolloutConfig = RolloutConfig()
    surprisal: SurprisalConfig = SurprisalConfig()
    priority: PriorityConfig = PriorityConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    refinement: RefinementConfig = RefinementConfig()
    output_dir: Path = Path("experiments/token_synthesis/results")
    device: str = "cuda"
    seed: int = 42

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
