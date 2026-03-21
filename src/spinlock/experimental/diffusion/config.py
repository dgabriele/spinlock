"""Pydantic configuration models for discrete diffusion experiments."""

from pydantic import BaseModel, Field, field_validator, model_validator
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

    # Auxiliary truncation lengths for roundtrip consistency loss.
    # Loads extra token stores at these truncation levels for multi-resolution
    # ground truth comparison. Only used when roundtrip_loss.enabled=True.
    aux_truncation_lengths: Optional[List[int]] = Field(
        default=None,
        description=(
            "Extra truncation lengths to load for roundtrip loss, "
            "e.g. [32, 64, 128, 256]. Each adds ~7MB per 30K samples."
        ),
    )

    # Entropy-based token filtering (Otsu threshold, auto-detected)
    entropy_filter: bool = Field(
        default=False,
        description=(
            "Filter low-entropy token positions via Otsu's method. "
            "Positions with near-zero entropy (constants) are frozen and "
            "excluded from D3PM training. Requires use_pretokenized=True."
        ),
    )

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

    # Family-level masking overrides (applied after base strategy)
    always_masked_families: Optional[List[str]] = Field(
        default=None,
        description=(
            "Families whose keys are always masked (target). "
            "E.g., ['initial', 'theta'] for inverse generation training."
        ),
    )
    always_observed_families: Optional[List[str]] = Field(
        default=None,
        description=(
            "Families whose keys are always observed (conditioning). "
            "E.g., ['temporal'] to always condition on dynamics."
        ),
    )

    @model_validator(mode="after")
    def validate_family_overrides(self):
        """Ensure no family appears in both always_masked and always_observed."""
        masked = set(self.always_masked_families or [])
        observed = set(self.always_observed_families or [])
        overlap = masked & observed
        if overlap:
            raise ValueError(
                f"Families cannot be both always_masked and always_observed: {overlap}"
            )
        return self


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
    family_scale_overrides: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Per-family scale overrides, e.g. {'theta': 0.15, 'initial': 0.25}. "
            "Applied to keys whose family matches, unless the key has an explicit "
            "entry in scale_factors (which takes highest priority)."
        ),
    )
    non_temporal_scale: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Fallback scale factor for keys not in scale_factors or "
            "family_scale_overrides. Lowest priority in the 3-tier resolution."
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


class PerturbationConfig(BaseModel):
    """Local parameter perturbation around D3PM proposals."""
    enabled: bool = True
    initial_sigma: float = Field(default=0.03, gt=0.0)
    sigma_growth_factor: float = Field(default=1.5, gt=1.0)
    max_sigma: float = Field(default=0.15, gt=0.0)
    perturbations_per_round: int = Field(default=4, ge=1)
    freeze_categorical_dims: bool = True
    categorical_dim_indices: List[int] = Field(default_factory=lambda: [32, 33])


class AdaptiveBudgetConfig(BaseModel):
    """Difficulty-proportional candidate allocation."""
    min_extra_candidates: int = Field(default=2, ge=1)
    max_extra_candidates: int = Field(default=16, ge=1)
    budget_scaling: str = Field(default="linear", pattern="^(linear|sqrt)$")
    d3pm_fraction: float = Field(default=0.5, gt=0.0, le=1.0)


class DynamicStoppingConfig(BaseModel):
    """Per-sample and per-cycle stopping criteria."""
    max_rounds_per_sample: int = Field(default=4, ge=1)
    min_acceptance_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    acceptance_rate_window: int = Field(default=200, ge=10)
    min_loss_improvement: float = Field(default=0.001, ge=0.0)


class AdaptiveRefinementConfig(BaseModel):
    """Top-level adaptive refinement configuration."""
    initial_d3pm_candidates: int = Field(default=2, ge=1)
    perturbation: PerturbationConfig = Field(default_factory=PerturbationConfig)
    budget: AdaptiveBudgetConfig = Field(default_factory=AdaptiveBudgetConfig)
    stopping: DynamicStoppingConfig = Field(default_factory=DynamicStoppingConfig)


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

    # v11 stabilization: weight anchoring
    anchor_weight: float = Field(
        default=0.05, ge=0.0,
        description="L2 penalty toward initial checkpoint weights. 0.0 = disabled.",
    )

    # v11 stabilization: replay buffer
    replay_fraction: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Fraction of training batch from replay buffer. 0.0 = disabled.",
    )
    max_replay_size: int = Field(
        default=5000, ge=1,
        description="Max replay buffer capacity (reservoir sampling when full).",
    )

    # v11 stabilization: LR scheduling
    use_cosine_schedule: bool = Field(
        default=True,
        description="Cosine LR decay within each fine-tuning cycle.",
    )
    warmup_fraction: float = Field(
        default=0.1, ge=0.0, le=0.5,
        description="Fraction of FT steps for linear warmup.",
    )
    min_lr_fraction: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Min LR as fraction of base at end of cosine cycle.",
    )
    per_cycle_lr_decay: float = Field(
        default=0.9, gt=0.0, le=1.0,
        description="Multiplicative LR decay each cycle. 1.0 = no decay.",
    )

    # v12 surprise-driven training
    surprise_alpha: float = Field(
        default=1.0, ge=0.0,
        description="Priority exponent for replay sampling. 0=uniform, 1=proportional.",
    )
    max_surprise_weight: float = Field(
        default=5.0, ge=1.0,
        description="Cap on per-sample surprise weight to prevent outlier domination.",
    )


class CVAEFineTuningConfig(BaseModel):
    """CVAE fine-tuning on verified ensemble winners.

    After ensemble selection, the winning (theta, IC) pairs have been verified
    through rollout + retokenize to produce high observed-position agreement.
    These verified pairs are ideal fine-tuning targets: they teach the CVAE
    which (theta, IC) solutions actually survive the roundtrip when conditioned
    on D3PM-completed tokens (with their errors).

    Low beta (0.1 vs 0.5 during original training) prioritizes reconstruction
    accuracy over KL regularization, since verified winners are high-quality
    targets — not noisy samples that need regularization.
    """
    enabled: bool = True
    learning_rate: float = Field(default=2e-5, gt=0.0)
    num_epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=64, ge=1)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    beta: float = Field(default=0.1, ge=0.0)
    free_bits: float = Field(default=2.0, ge=0.0)


class RefinementConfig(BaseModel):
    """Offline hard-target refinement loop configuration.

    Closed loop: D3PM inpaint → IntegratedTokenDecoder → rollout → retokenize →
    quality filter → fine-tune.

    The D3PM generates ALL token positions (temporal + initial + theta). Observed
    temporal tokens are fixed; initial + theta positions are inpainted. The
    IntegratedTokenDecoder (codebook lookup + inverse heads) replaces the CVAE.
    Diversity comes from D3PM's stochastic denoising trajectories.

    Rollout source (controlled by ``mno_checkpoint``):
    - ``mno_checkpoint`` set: uses trained V2MNO surrogate
    - ``mno_checkpoint`` null: uses GT simulator (LeniaReplayAdapter, etc.)
    """
    # Checkpoints
    d3pm_checkpoint: str           # Path to trained D3PM checkpoint
    mno_checkpoint: Optional[str] = None  # MNO path, or null for GT simulator
    tokenizer_checkpoint: str      # Path to VQTokenizer checkpoint

    # Dataset
    dataset_path: str              # HDF5 dataset with (IC, theta, rollouts)
    dataset_config_path: Optional[str] = None  # Explicit dataset gen YAML override
    max_samples: Optional[int] = None
    rollout_steps: int = 512       # Rollout length (MNO or GT simulator)

    # D3PM sampling
    num_refinement_cycles: int = Field(default=3, ge=1)
    mask_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    d3pm_start_step: Optional[int] = None  # Partial-start for conservative corrections
    sampling_temperature: float = Field(
        default=1.0, gt=0.0,
        description="Temperature for D3PM sampling. >1 increases diversity, <1 sharpens. "
                    "When != 1.0, the final step (t=0) uses tempered sampling instead of argmax.",
    )

    # Early stopping: stop generating hard targets once enough are accepted.
    max_accepted_targets: Optional[int] = None

    # Mini-batch size for hard-target generation.
    generation_batch_size: int = Field(default=8, ge=1)

    # Quality filter
    quality_filter: MNOQualityFilterConfig = Field(default_factory=MNOQualityFilterConfig)

    # Fine-tuning (D3PM only — no CVAE)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)

    # Adaptive refinement (difficulty-proportional search)
    adaptive: AdaptiveRefinementConfig = Field(default_factory=AdaptiveRefinementConfig)

    # Held-out evaluation: fixed sample set for apples-to-apples comparison
    # across cycles. 0 = disabled.
    eval_samples: int = 0
    eval_frequency: int = Field(default=1, ge=1)

    # v11 stabilization: eval-based early stopping
    early_stopping_patience: int = Field(
        default=3, ge=0,
        description="Stop if eval agreement doesn't improve for N cycles. 0 = disabled.",
    )

    # Output
    output_dir: str = "experiments/diffusion/results/v7_refinement"
    device: str = "cuda"
    seed: int = 42


class DenoisingRoundtripLossConfig(BaseModel):
    """Denoising roundtrip consistency loss configuration.

    At each training step, soft-decodes D3PM logits through the frozen VQ
    pipeline (codebooks → decoder → temporal inverse → pyramid re-encode →
    projector → quantizer distances), producing roundtrip logits that are
    compared against ground-truth tokens at a truncation level matching the
    current noise level.

    Requires auxiliary truncation-level tokens in the dataset (see
    DatasetConfig.aux_truncation_lengths).
    """
    enabled: bool = False
    weight: float = Field(default=0.1, ge=0.0)
    temperature: float = Field(default=1.0, gt=0.0, description="Soft-decode sharpness")
    warmup_epochs: int = Field(default=3, ge=0, description="Skip early epochs")
    timestep_gate: str = Field(
        default="cosine",
        pattern="^(none|linear|cosine)$",
        description="Weight by noise level: cosine (heavy at low noise), linear, or none",
    )
    noise_boundaries: Optional[List[float]] = Field(
        default=None,
        description=(
            "Noise fraction boundaries for truncation level mapping. "
            "Auto-computed as uniform spacing if None. Can be loaded from "
            "calibrate_trajectory.py output for empirical boundaries."
        ),
    )
    roundtrip_metric: str = Field(
        default="ce",
        pattern="^(ce|weighted_hamming)$",
        description=(
            "Per-position roundtrip comparison metric. "
            "'ce': cross-entropy against GT tokens (standard, treats all errors equally). "
            "'weighted_hamming': soft embedding-distance loss that penalizes proportionally "
            "to codebook geometry — near-miss codes cost less than distant codes."
        ),
    )

    # Soft set-level coherence loss (differentiable Jaccard)
    set_coherence_weight: float = Field(
        default=0.0, ge=0.0,
        description=(
            "Weight for soft set-level Jaccard loss term. 0.0 = disabled "
            "(backward compatible). Encourages roundtrip logits to produce "
            "code usage sets matching truncation-level ground truth."
        ),
    )
    set_coherence_temperature: float = Field(
        default=0.5, gt=0.0,
        description="Softmax temperature for soft code usage in Jaccard computation.",
    )

    # Trajectory probe (validation-time monitoring)
    trajectory_probe_frequency: int = Field(
        default=0, ge=0,
        description=(
            "Run trajectory probe every N validations. 0 = disabled. "
            "Samples denoising trajectories and measures per-step agreement "
            "against truncation-level ground truth."
        ),
    )
    trajectory_probe_samples: int = Field(
        default=4, ge=1,
        description="Number of samples for trajectory probe.",
    )


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
    focal_gamma: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Focal loss gamma. 0 = standard CE. γ>0 down-weights easy predictions: "
            "loss_i = -(1-p_i)^γ · log(p_i). γ=2.0 recommended when rollouts share "
            "70-80% of tokens, amplifying gradient on the 20-30% that differ."
        ),
    )
    primary_loss_metric: str = Field(
        default="ce",
        pattern="^(ce|weighted_hamming)$",
        description=(
            "Primary per-position loss metric. "
            "'ce': cross-entropy (standard, treats all wrong predictions equally). "
            "'weighted_hamming': soft embedding-distance loss through frozen codebook — "
            "near-miss codes get small loss, distant codes get large loss. "
            "Requires dataset.tokenizer_checkpoint for codebook access."
        ),
    )
    physics_loss: PhysicsLossConfig = Field(default_factory=PhysicsLossConfig)
    roundtrip_loss: DenoisingRoundtripLossConfig = Field(
        default_factory=DenoisingRoundtripLossConfig,
    )

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


class CurriculumStageConfig(BaseModel):
    """Single stage in a training curriculum.

    Each stage defines a masking strategy, epoch count, and optional
    overrides for learning rate, mask probability, and family-level
    masking constraints.
    """
    name: str
    strategy: MaskingStrategy
    num_epochs: int = Field(ge=1)
    learning_rate: Optional[float] = Field(default=None, gt=0.0)
    mask_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    always_masked_families: Optional[List[str]] = None
    always_observed_families: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_stage_family_overrides(self):
        """Ensure no family appears in both always_masked and always_observed."""
        masked = set(self.always_masked_families or [])
        observed = set(self.always_observed_families or [])
        overlap = masked & observed
        if overlap:
            raise ValueError(
                f"Stage '{self.name}': families cannot be both "
                f"always_masked and always_observed: {overlap}"
            )
        return self


class CurriculumConfig(BaseModel):
    """Multi-stage curriculum learning configuration.

    When enabled, the trainer progresses through stages sequentially,
    each with its own masking strategy and optional LR/family overrides.
    Total training epochs = sum of stage num_epochs.
    """
    enabled: bool = False
    stages: List[CurriculumStageConfig] = Field(default_factory=list)


class DiffusionExperimentConfig(BaseModel):
    """Complete configuration for discrete diffusion experiment."""
    dataset: DatasetConfig
    masking: MaskingConfig
    diffusion: DiffusionConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig

    curriculum: Optional[CurriculumConfig] = None

    device: str = Field(default="cuda", pattern="^(cuda|cpu)$")
    seed: int = 42

    class Config:
        """Pydantic config."""
        use_enum_values = True
