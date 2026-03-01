"""V2 MNO configuration schemas — trajectory-first training.

Train the MNO as an accurate trajectory surrogate first (MSE + contrastive),
then separately train a VQTokenizer on MNO outputs and align token spaces.

Reuses MNOModelConfig, DataConfig, CheckpointingConfig from v1 for consistency.
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from spinlock.mno.config import (
    CheckpointingConfig,
    DataConfig,
    MNOModelConfig,
)


class V2ContrastiveConfig(BaseModel):
    """Soft token contrastive loss hyperparameters.

    Uses Jaccard similarity between GT token indicator vectors as soft
    targets instead of identity-based InfoNCE. MoCo queue provides
    additional negatives for small-batch training (B=1).
    """

    tau_pred: float = 0.07
    tau_target: float = 0.1
    embed_dim: int = 128
    hidden_dim: int = 256
    queue_size: int = 64

    @model_validator(mode="before")
    @classmethod
    def _migrate_temperature(cls, data: dict) -> dict:
        """Accept legacy ``temperature`` key as alias for ``tau_pred``."""
        if isinstance(data, dict) and "temperature" in data:
            data.setdefault("tau_pred", data.pop("temperature"))
        return data


class V2LossConfig(BaseModel):
    """Trajectory-first loss weights.

    Primary: trajectory MSE (direct pixel-level supervision).
    Secondary: IC MSE (warmup endpoint accuracy) + contrastive (diversity).
    Optional: feature MSE (only if tokenizer_checkpoint provided).
    """

    lambda_traj: float = 1.0
    lambda_ic: float = 0.3
    lambda_contrastive: float = 0.3
    lambda_feat_mse: float = 0.0
    lambda_token_ce: float = 0.0
    lambda_centroid_mse: float = 0.0
    lambda_token_head: float = 0.0
    token_ce_temperature: float = 1.0
    gate_weight_token_ce: bool = False
    normalize_loss_scales: bool = False
    loss_scale_ema_momentum: float = 0.99
    contrastive: V2ContrastiveConfig = V2ContrastiveConfig()


class V2TrainingConfig(BaseModel):
    """V2 MNO training hyperparameters."""

    n_samples: int
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int
    learning_rate: float = 2e-4
    warmup_optimizer_steps: int = Field(
        default=500,
        description="LR warmup in optimizer steps (not micro-batches or epochs)",
    )
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    timesteps: int
    bptt_window: Optional[int] = 32
    film_lr_multiplier: float = 5.0
    token_head_lr_multiplier: float = 1.0
    bptt_num_windows: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of evenly-spaced BPTT supervision windows. "
            "1 = current behavior (final window only). "
            "3 = recommended (initial + mid + final). "
            "T//W = full coverage (most expensive)."
        ),
    )
    replayer_cache_size: int = 16
    eval_every: int = 5
    eval_samples: int = 50
    use_torch_compile: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_warmup_steps(cls, data: dict) -> dict:
        """Accept legacy ``warmup_steps`` key as alias."""
        if isinstance(data, dict) and "warmup_steps" in data:
            data.setdefault("warmup_optimizer_steps", data.pop("warmup_steps"))
        return data


class V2MNOConfig(BaseModel):
    """Top-level V2 MNO configuration.

    Modes:
    - Standard (default): Train on raw θ/IC from dataset
    - Quantization-aware: Set quantization_aware=true + pretokenized_tokens +
      tokenizer_checkpoint. Decodes VQ tokens → θ̂/IC_hat and trains on
      rollouts from these decoded (quantized) parameters.
    """

    model: MNOModelConfig
    training: V2TrainingConfig
    loss: V2LossConfig = V2LossConfig()
    data: DataConfig
    checkpointing: CheckpointingConfig
    tokenizer_checkpoint: Optional[str] = None
    token_dataset: Optional[str] = None
    token_truncation_length: Optional[int] = None
    quantization_aware: bool = False
    resume_from: Optional[str] = None
    device: str = "cuda"
    seed: int = 42
