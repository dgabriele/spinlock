"""Pydantic configuration models for discrete diffusion experiments."""

from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum


class MaskingStrategy(str, Enum):
    """Masking strategies for hierarchical tokens."""
    RANDOM = "random"
    COARSE_ONLY = "coarse_only"
    HIERARCHICAL = "hierarchical"
    FAMILY_SELECTIVE = "family_selective"
    MIXED = "mixed"


class DatasetConfig(BaseModel):
    """Dataset configuration for diffusion training."""
    # Pre-tokenized mode (fast)
    use_pretokenized: bool = False
    tokenized_path: Optional[Path] = None

    # Single truncation selection (load only this T from multi-trunc HDF5)
    truncation_length: Optional[int] = None

    # On-the-fly tokenization mode (slow, flexible)
    path: Optional[Path] = None
    tokenizer_checkpoint: Optional[Path] = None
    cache_tokens: bool = False
    max_cache_size: Optional[int] = None
    device: str = "cpu"

    @field_validator("tokenized_path")
    @classmethod
    def validate_pretokenized_path(cls, v, info):
        """Validate pre-tokenized dataset exists if use_pretokenized=True."""
        if info.data.get("use_pretokenized") and v is not None:
            if not v.exists():
                raise ValueError(f"Pre-tokenized dataset not found: {v}")
        return v

    @field_validator("path")
    @classmethod
    def validate_dataset_path(cls, v, info):
        """Validate dataset exists if not using pre-tokenized."""
        if not info.data.get("use_pretokenized") and v is not None:
            if not v.exists():
                raise ValueError(f"Dataset not found: {v}")
        return v

    @field_validator("tokenizer_checkpoint")
    @classmethod
    def validate_tokenizer(cls, v, info):
        """Validate tokenizer exists when provided.

        Required in on-the-fly mode. Optional in pretokenized mode (used
        to extract authoritative vocab sizes from actual codebooks rather
        than inferring from max observed token values).
        """
        if v is not None and not v.exists():
            raise ValueError(f"Tokenizer checkpoint not found: {v}")
        return v

    def model_post_init(self, __context):
        """Validate configuration consistency after initialization."""
        if self.use_pretokenized and self.tokenized_path is None:
            raise ValueError("tokenized_path required when use_pretokenized=True")
        if not self.use_pretokenized and (self.path is None or self.tokenizer_checkpoint is None):
            raise ValueError("path and tokenizer_checkpoint required when use_pretokenized=False")


class MixedStrategyEntry(BaseModel):
    """Single strategy entry for mixed masking."""
    name: str
    weight: float = Field(gt=0.0)


class MaskingConfig(BaseModel):
    """Masking configuration for diffusion training.

    Single strategy:
        strategy: "random"
        mask_probability: 0.5

    Mixed (strategy: "mixed"):
        strategy: "mixed"
        strategies:
          - name: "random"
            weight: 0.5
          - name: "coarse_only"
            weight: 0.3
          - name: "hierarchical"
            weight: 0.2
        mask_probability: 0.5   # passed to the random sub-strategy
    """
    strategy: MaskingStrategy = MaskingStrategy.RANDOM
    mask_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: int = 42
    strategies: Optional[List[MixedStrategyEntry]] = None  # only used when strategy="mixed"


class GradedScheduleConfig(BaseModel):
    """Per-position graded forward process configuration.

    When enabled, maps global diffusion timestep to per-position effective
    timesteps using scale factors. Positions with low scale (theta, IC)
    resolve first during denoising; positions with high scale (long-horizon
    temporal) resolve last — encoding causal hierarchy directly in the
    noise schedule.

    Scale factors map each position key to a multiplier:
        effective_t(key, t) = clamp(round(t * scale[key]), 0, T-1)

    Scale factors can be provided inline (scale_factors dict) or loaded
    from a JSON file (position_scale_factors_path), typically computed by
    compute_position_scales.py from cross-truncation token divergence.
    """
    enabled: bool = False
    scale_factors: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Inline per-position scale factors, e.g. "
            "{'temporal_group_0_L0': 0.72, 'theta_group_0_L0': 0.3, ...}."
        ),
    )
    position_scale_factors_path: Optional[str] = Field(
        default=None,
        description="Path to JSON file with per-position scale factors.",
    )
    non_temporal_scale: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Fallback scale factor for keys not in scale_factors. Low values "
            "resolve causes first, matching the causal DAG: (theta, IC) → temporal."
        ),
    )


class DiffusionConfig(BaseModel):
    """Diffusion process configuration."""
    num_timesteps: int = Field(default=50, ge=1)
    beta_start: float = Field(default=0.0001, gt=0.0)
    beta_end: float = Field(default=0.02, gt=0.0)
    schedule_type: str = Field(default="cosine", pattern="^(linear|cosine|sqrt)$")
    transition_type: str = Field(default="uniform", pattern="^(uniform|absorbing)$")
    beta_scaling: str = Field(default="uniform", pattern="^(uniform|vocab_aware)$")
    graded_schedule: GradedScheduleConfig = Field(
        default_factory=GradedScheduleConfig,
        description="Truncation-graded forward process (per-truncation noise scaling)",
    )

    @field_validator("beta_end")
    @classmethod
    def validate_beta_range(cls, v, info):
        """Ensure beta_end > beta_start."""
        if "beta_start" in info.data and v <= info.data["beta_start"]:
            raise ValueError(f"beta_end ({v}) must be > beta_start ({info.data['beta_start']})")
        return v


class ModelConfig(BaseModel):
    """Denoising network model configuration."""
    hidden_dim: int = Field(default=256, ge=64)
    num_layers: int = Field(default=6, ge=1)
    num_heads: int = Field(default=8, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    use_hierarchical_guidance: bool = True
    hierarchical_guidance_weight: float = Field(default=0.1, ge=0.0)
    hierarchical_guidance_mode: str = Field(
        default="global", pattern="^(global|per_category)$"
    )


class LRSchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""
    type: str = Field(default="cosine", pattern="^(cosine|linear|step)$")
    warmup_epochs: int = Field(default=2, ge=0)
    min_lr: float = Field(default=1e-6, gt=0.0)


class PhysicsLossConfig(BaseModel):
    """Physics-aware auxiliary loss configuration.

    Soft-decodes denoiser logits through the frozen VQTokenizer decode pipeline
    (codebooks → shared decoder → inverse heads) and computes MSE against
    ground-truth decoded physics parameters. This encourages the denoiser to
    produce token distributions that decode to physically consistent values.
    """
    enabled: bool = False
    weight: float = Field(default=0.1, ge=0.0)
    temperature: float = Field(default=1.0, gt=0.0)
    warmup_epochs: int = Field(default=3, ge=0)
    families: Optional[List[str]] = Field(
        default=None,
        description="Families to decode: ['theta', 'initial']. None = all available."
    )
    theta_weight: float = Field(default=1.0, ge=0.0)
    initial_weight: float = Field(default=1.0, ge=0.0)
    timestep_gate: str = Field(
        default="bell",
        pattern="^(none|linear|cosine|bell)$"
    )


class MNOQualityFilterConfig(BaseModel):
    """Quality filter: trust MNO retokenization only when observed positions agree."""
    min_observed_agreement: float = Field(default=0.8, ge=0.0, le=1.0)
    # Fraction of observed-position tokens that must match GT


class FineTuningConfig(BaseModel):
    """Fine-tuning schedule for refinement epochs."""
    learning_rate: float = Field(default=2e-5, gt=0.0)
    num_epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=32, ge=1)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)


class RefinementConfig(BaseModel):
    """Offline hard-target refinement loop configuration.

    Closed loop: D3PM predict -> CVAE decode -> MNO rollout -> retokenize ->
    quality filter -> fine-tune.
    Requires trained D3PM, MNO, VQTokenizer, and CVAE checkpoints.
    """
    # Checkpoints
    d3pm_checkpoint: str           # Path to trained D3PM checkpoint
    mno_checkpoint: str            # Path to trained MNO checkpoint
    tokenizer_checkpoint: str      # Path to VQTokenizer checkpoint
    cvae_checkpoint: Optional[str] = None  # Path to trained CVAE checkpoint

    # Dataset
    dataset_path: str              # HDF5 dataset with (IC, theta, rollouts)
    max_samples: Optional[int] = None
    rollout_steps: int = 512       # MNO rollout length

    # D3PM sampling
    num_refinement_cycles: int = Field(default=3, ge=1)
    mask_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    d3pm_start_step: Optional[int] = None  # Partial-start for conservative corrections

    # Quality filter
    quality_filter: MNOQualityFilterConfig = Field(default_factory=MNOQualityFilterConfig)

    # Fine-tuning
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)

    # Output
    output_dir: str = "experiments/diffusion/results/v7_refinement"
    device: str = "cuda"
    seed: int = 42


class TrainingConfig(BaseModel):
    """Training configuration."""
    batch_size: int = Field(default=32, ge=1)
    num_epochs: int = Field(default=30, ge=1)
    num_workers: int = Field(default=0, ge=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    use_snr_weighting: bool = False
    use_vocab_size_weighting: bool = False
    physics_loss: PhysicsLossConfig = Field(default_factory=PhysicsLossConfig)

    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)

    val_batch_size: int = Field(default=128, ge=1)
    val_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    val_frequency: int = Field(default=1, ge=1)
    val_samples: int = Field(default=5, ge=1)

    checkpoint_frequency: int = Field(default=5, ge=1)
    save_best: bool = True

    log_frequency: int = Field(default=50, ge=1)
    use_wandb: bool = False


class OutputConfig(BaseModel):
    """Output configuration."""
    dir: Path
    checkpoint_prefix: str = "diffusion"

    @field_validator("dir")
    @classmethod
    def create_output_dir(cls, v):
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class DiffusionExperimentConfig(BaseModel):
    """Complete configuration for discrete diffusion experiment."""
    dataset: DatasetConfig
    masking: MaskingConfig
    diffusion: DiffusionConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig

    device: str = Field(default="cuda", pattern="^(cuda|cpu)$")
    seed: int = 42

    class Config:
        """Pydantic config."""
        use_enum_values = True
