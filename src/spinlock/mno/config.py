"""MNO Training Configuration with Runtime Dimension Inference.

Framework-compliant Pydantic models that distinguish between:
- Data-dependent dimensions (auto-detected at runtime)
- Algorithmic choices (user-specified)

Example:
    ```python
    from spinlock.data import SpinlockDataset
    from spinlock.mno.config import MNOConfig
    import yaml

    # Load config
    with open('configs/qbm/mno/baseline.yaml') as f:
        config_dict = yaml.safe_load(f)

    # Infer dimensions from dataset
    dataset, config_dict = SpinlockDataset.infer_and_update_config(
        config_dict,
        'datasets/qbm_50k.h5',
        verbose=True
    )

    # Apply MNO-specific dimension inference
    mno_dimensions = dataset.infer_mno_dimensions()
    for key, value in mno_dimensions.items():
        if isinstance(value, dict):
            config_dict.setdefault(key, {}).update(value)
        else:
            config_dict[key] = value

    # Validate and create config
    config = MNOConfig(**config_dict)

    # All dimensions now resolved!
    print(f"in_channels: {config.model.in_channels}")
    print(f"param_dim: {config.model.param_dim}")
    ```

Design Principles:
    1. Optional fields for auto-detected dimensions (None defaults)
    2. Required fields for algorithmic choices
    3. Runtime validation that all dimensions are resolved
    4. Clear error messages when dimensions missing
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, Literal


class FiLMConfig(BaseModel):
    """FiLM (Feature-wise Linear Modulation) configuration.

    All parameters are algorithmic choices (no auto-detection).
    """

    enabled: bool = True
    embed_dim: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    init_gamma: float = 1.0
    init_beta: float = 0.0
    post_norm: bool = True
    modulate_encoder: bool = True
    modulate_decoder: bool = True


class MNOModelConfig(BaseModel):
    """MNO backbone configuration with auto-detected dimensions.

    Data-Dependent Dimensions (AUTO-DETECTED):
        - spatial_dim: Spatial resolution (from inputs/fields shape)
        - in_channels: Input channels (from dataset metadata)
        - out_channels: Output channels (from dataset metadata)
        - param_dim: Parameter dimension (from parameters/params shape)

    Algorithmic Choices (USER-SPECIFIED):
        - base_channels: Base channel count for U-Net
        - encoder_levels: Number of encoder/decoder levels
        - modes: Number of Fourier modes for AFNO
        - afno_blocks: Number of AFNO blocks in bottleneck
        - param_conditioning: Whether to condition on parameters
        - param_embed_dim: Dimension of parameter embeddings
        - conditioning_mode: How to condition ("concat", "film", "both")
        - film: FiLM configuration (if conditioning_mode includes "film")
    """

    # ============================================================================
    # DATA-DEPENDENT DIMENSIONS (AUTO-DETECTED AT RUNTIME)
    # ============================================================================

    spatial_dim: Optional[int] = Field(
        default=None,
        description="Spatial resolution (auto-detected from inputs/fields shape)"
    )

    in_channels: Optional[int] = Field(
        default=None,
        description="Input channels (auto-detected from dataset metadata)"
    )

    out_channels: Optional[int] = Field(
        default=None,
        description="Output channels (auto-detected from dataset metadata)"
    )

    param_dim: Optional[int] = Field(
        default=None,
        description="Parameter dimension (auto-detected from parameters/params shape)"
    )

    # ============================================================================
    # ALGORITHMIC CHOICES (USER-SPECIFIED)
    # ============================================================================

    base_channels: int = Field(default=40, gt=0)
    encoder_levels: int = Field(default=3, ge=1)
    modes: int = Field(default=16, gt=0)
    afno_blocks: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)

    # Parameter conditioning
    param_conditioning: bool = True
    param_embed_dim: int = Field(default=128, gt=0)
    conditioning_mode: Literal["concat", "film", "both"] = "film"

    # FiLM configuration (only used when conditioning_mode includes "film")
    film: Optional[FiLMConfig] = None

    # Update mode
    update_mode: Literal["residual", "autoregressive"] = "residual"

    # Gradient checkpointing (memory optimization)
    use_checkpointing: bool = True
    checkpoint_every: int = Field(default=16, gt=0)

    @field_validator("param_conditioning")
    @classmethod
    def validate_param_conditioning(cls, v, info):
        """If param_conditioning is True, param_dim must be set."""
        # Note: We can't validate param_dim here because it might not be set yet
        # Runtime validation happens in validate_all_dimensions()
        return v

    def validate_all_dimensions(self):
        """Validate that all required dimensions are resolved.

        Call this after dimension inference to ensure config is complete.

        Raises:
            ValueError: If any required dimension is None
        """
        missing = []

        if self.in_channels is None:
            missing.append("in_channels")
        if self.out_channels is None:
            missing.append("out_channels")
        if self.param_conditioning and self.param_dim is None:
            missing.append("param_dim (required when param_conditioning=True)")

        if missing:
            raise ValueError(
                f"MNO model config has unresolved dimensions: {', '.join(missing)}.\n"
                f"These dimensions should be auto-detected from the dataset.\n"
                f"Use: dataset, config_dict = SpinlockDataset.infer_and_update_config(...)"
            )


class TrainingConfig(BaseModel):
    """Training configuration with validated dimensions.

    Data-Dependent Validation:
        - timesteps: Validated against dataset max_timesteps (cannot exceed)

    User-Specified:
        - All other training parameters
    """

    # Training data
    n_samples: int = Field(ge=1)
    sampling_strategy: Literal["sequential", "random", "stratified"] = "sequential"

    # Batch and optimization
    batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    clip_grad: Optional[float] = Field(default=None, gt=0.0)

    # Timesteps (validated against dataset)
    timesteps: int = Field(
        description="Rollout timesteps (validated against dataset max_timesteps)"
    )
    bptt_window: Optional[int] = Field(default=32, ge=1)

    # Training optimizations
    use_torch_compile: bool = False
    replayer_cache_size: int = Field(default=16, ge=1)

    # Early stopping
    early_stopping_patience: Optional[int] = Field(default=None, ge=1)

    # FiLM-specific learning rate multiplier
    film_lr_multiplier: float = Field(default=5.0, gt=0.0)

    def validate_timesteps(self, max_timesteps: Optional[int]):
        """Validate timesteps against dataset maximum.

        Args:
            max_timesteps: Maximum timesteps available in dataset

        Raises:
            ValueError: If requested timesteps exceed dataset maximum
        """
        if max_timesteps is not None and self.timesteps > max_timesteps:
            raise ValueError(
                f"Requested timesteps ({self.timesteps}) exceeds dataset maximum ({max_timesteps}).\n"
                f"Reduce training.timesteps in config or regenerate dataset with more timesteps."
            )


class LossConfig(BaseModel):
    """Loss function configuration.

    All parameters are algorithmic choices.
    """

    mode: str
    lambda_traj: float = Field(default=1.0, ge=0.0)
    lambda_ic: float = Field(default=0.3, ge=0.0)
    lambda_param_recon: float = Field(default=0.5, ge=0.0)
    lambda_contrastive: float = Field(default=0.3, ge=0.0)
    lambda_param_sensitivity: float = Field(default=0.2, ge=0.0)

    # Token diversity mode lambdas (VQ coherence)
    lambda_recon: float = Field(default=1.0, ge=0.0)
    lambda_commit: float = Field(default=0.5, ge=0.0)

    # Contrastive loss config
    contrastive: Optional[Dict[str, Any]] = None

    # Parameter reconstruction config
    param_recon: Optional[Dict[str, Any]] = None

    # Sensitivity config
    sensitivity: Optional[Dict[str, Any]] = None


class DataConfig(BaseModel):
    """Data configuration.

    Paths are user-specified, dimensions come from dataset.
    """

    dataset_path: str
    config: str  # Path to replayer config (e.g., configs/qbm/substrate.yaml)
    val_split: float = Field(default=0.2, ge=0.0, le=1.0)
    num_workers: int = Field(default=8, ge=0)


class CheckpointingConfig(BaseModel):
    """Checkpointing configuration."""

    save_dir: str
    save_every: int = Field(default=5, gt=0)
    keep_best: bool = True


class MNOConfig(BaseModel):
    """Complete MNO training configuration.

    Framework-compliant: All data-dependent dimensions auto-detected at runtime.

    Usage:
        1. Load config YAML (with Optional dimension fields as None)
        2. Use SpinlockDataset.infer_and_update_config() to populate dimensions
        3. Apply MNO-specific dimension inference
        4. Validate and create MNOConfig instance
        5. Call config.validate() to ensure all dimensions resolved
    """

    model: MNOModelConfig
    training: TrainingConfig
    loss: LossConfig
    data: DataConfig
    checkpointing: CheckpointingConfig
    device: str = "cuda"
    seed: int = 42

    def validate(self, max_timesteps: Optional[int] = None):
        """Validate complete configuration.

        Args:
            max_timesteps: Maximum timesteps from dataset (for validation)

        Raises:
            ValueError: If configuration is invalid or dimensions unresolved
        """
        # Validate model dimensions
        self.model.validate_all_dimensions()

        # Validate training timesteps
        if max_timesteps is not None:
            self.training.validate_timesteps(max_timesteps)
