"""Configuration for NLTokenizer — per-group hierarchical VAE + LFM.

The NLTokenizer encodes Lenia dynamics via the same PyramidFirstEncoder
and group structure as VQTokenizer, but replaces discrete codebooks with
a per-group hierarchical VAE. Each group independently maps D_group → z
through multi-level (μ, logvar) projections, matching VQ's information
capacity without quantization.
"""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .config import (
    EncoderConfig,
    FeatureCleaningConfig,
    HierarchyConfig,
    NormalizationConfig,
)


# ──────────────────────────────────────────────────────────────────────
# VAE config (per-group hierarchical)
# ──────────────────────────────────────────────────────────────────────

class VAEConfig(BaseModel):
    """Per-group hierarchical VAE configuration.

    Each temporal group's D_group embedding passes through a shared
    HierarchicalVAEHead that projects to per-level (μ, logvar) → z.
    Level dims are computed from D_group × level_ratios.

    Total z_dim = num_groups × sum(D_group × ratio for ratio in level_ratios)
                + theta_z_dim
    """

    level_ratios: List[float] = Field(
        default=[1.0, 0.5],
        description="Per-level latent dim as fraction of D_group. [1.0, 0.5] → L0=D_group, L1=D_group//2",
    )
    group_encoder_hidden_dim: Optional[int] = Field(
        default=None,
        description="Hidden dim for per-group encoder MLP. None = D_group × 2",
    )
    theta_z_dim: int = Field(
        default=32, gt=0,
        description="Latent dim for theta family's VAE head",
    )
    feature_decoder_hidden_dims: List[int] = Field(
        default=[1024, 768],
        description="Hidden dims for z_full → ĥ reconstruction decoder",
    )
    lfm_projection_dim: int = Field(
        default=256, gt=0,
        description="Projection from z_full → z_lfm for LFM adapter input",
    )


# ──────────────────────────────────────────────────────────────────────
# LFM adapter config
# ──────────────────────────────────────────────────────────────────────

class LFMAdapterConfig(BaseModel):
    """Configuration for the LFM generator adapter."""

    latent_dim: int = Field(default=256, gt=0, description="LFM decoder's expected latent dim")
    vocab_size: int = Field(default=8000, gt=0)
    max_output_len: int = Field(default=256, ge=16)
    decoder_hidden_dim: int = Field(default=256, gt=0)
    decoder_num_layers: int = Field(default=2, ge=1)
    decoder_num_heads: int = Field(default=4, ge=1)
    pretrained_decoder_path: Optional[Path] = None
    spm_model_path: Optional[Path] = None
    freeze_decoder: bool = True
    temperature: float = Field(default=1.0, gt=0.0)
    hard_sample: bool = True


# ──────────────────────────────────────────────────────────────────────
# NL Listener config
# ──────────────────────────────────────────────────────────────────────

class NLListenerConfig(BaseModel):
    """Configuration for the NL listener (text → z decoder).

    latent_dim is auto-set to z_full_dim at model creation time.
    """

    vocab_size: int = Field(default=8000, gt=0)
    hidden_dim: int = Field(default=256, gt=0)
    num_heads: int = Field(default=4, ge=1)
    num_layers: int = Field(default=2, ge=1)
    latent_dim: int = Field(default=256, gt=0, description="Auto-set to z_full_dim at runtime")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────
# Inverse decoder config
# ──────────────────────────────────────────────────────────────────────

class NLInverseConfig(BaseModel):
    """Configuration for z → parameter inverse decoders."""

    theta_hidden_dim: int = Field(default=128, gt=0)
    theta_dropout: float = Field(default=0.1, ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────
# Loss config
# ──────────────────────────────────────────────────────────────────────

class NLLossConfig(BaseModel):
    """Loss weights for NLTokenizer training.

    Inverse losses use behavioral equivalence (encoding space, not param space).
    Topographic loss preserves behavioral neighborhoods in z-space.
    """

    reconstruction_weight: float = Field(default=1.0, ge=0.0)
    kl_weight: float = Field(default=0.1, ge=0.0)
    kl_free_bits: float = Field(default=2.0, ge=0.0)
    theta_inverse_weight: float = Field(
        default=1.0, ge=0.0,
        description="Behavioral: ‖encoder(θ_hat) - encoder(θ_true)‖²",
    )
    listener_roundtrip_weight: float = Field(default=1.0, ge=0.0)
    topographic_weight: float = Field(default=0.5, ge=0.0)
    topographic_n_samples: int = Field(default=64, ge=4)


# ──────────────────────────────────────────────────────────────────────
# Training config
# ──────────────────────────────────────────────────────────────────────

class NLTrainingConfig(BaseModel):
    """Training hyperparameters for NLTokenizer."""

    num_epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    optimizer: Literal["adam", "adamw"] = "adam"
    val_split: float = Field(default=0.2, ge=0.0, le=0.5)
    val_every_n_epochs: int = Field(default=5, ge=1)
    shuffle: bool = False
    gradient_clip_norm: Optional[float] = None
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    use_scheduler: bool = False
    scheduler_type: Literal["cosine", "step", "exponential"] = "cosine"
    warmup_epochs: int = Field(default=0, ge=0)
    warmup_batches: int = Field(default=0, ge=0)
    early_stopping_patience: int = Field(default=20, ge=1)
    early_stopping_min_delta: float = Field(default=1e-4, ge=0.0)
    device: Literal["cuda", "cpu", "auto"] = "auto"

    # VAE training stages
    kl_warmup_epochs: int = Field(default=10, ge=0)
    listener_start_epoch: int = Field(default=10, ge=0)
    log_every_n_batches: int = Field(default=50, ge=1)
    checkpoint_every_n_epochs: int = Field(default=10, ge=1)


# ──────────────────────────────────────────────────────────────────────
# Top-level config
# ──────────────────────────────────────────────────────────────────────

class NLTokenizerConfig(BaseModel):
    """Complete NLTokenizer configuration.

    Uses the same encoder + hierarchy pattern as VQTokenizer, but with
    continuous VAE bottleneck instead of discrete codebooks.
    """

    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    hierarchy: HierarchyConfig = Field(
        default_factory=lambda: HierarchyConfig(num_levels=2),
    )
    vae: VAEConfig = Field(default_factory=VAEConfig)
    lfm_adapter: LFMAdapterConfig = Field(default_factory=LFMAdapterConfig)
    listener: NLListenerConfig = Field(default_factory=NLListenerConfig)
    inverse: NLInverseConfig = Field(default_factory=NLInverseConfig)
    loss: NLLossConfig = Field(default_factory=NLLossConfig)
    training: NLTrainingConfig = Field(default_factory=NLTrainingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    feature_cleaning: Optional[FeatureCleaningConfig] = None

    # Dataset / pipeline
    random_seed: Optional[int] = None
    verbose: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    generation_config_path: Optional[str] = None
    generation_timesteps: Optional[int] = Field(default=None, ge=1)
    realization_mode: Literal["single", "mean", "all"] = "single"
    replayer_cache_size: int = Field(default=8, ge=0)
    checkpoint_dir: Optional[str] = None
