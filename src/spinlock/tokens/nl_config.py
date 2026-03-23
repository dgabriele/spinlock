"""Configuration for NLTokenizer — continuous VAE + LFM integration.

The NLTokenizer encodes Lenia dynamics into continuous VAE latent vectors
that project into LFM's frozen autoregressive decoder to generate natural
language expressions. This replaces VQ+D3PM with a continuous system where
perturbation-based sampling in latent space replaces discrete diffusion.
"""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .config import (
    EncoderConfig,
    FeatureCleaningConfig,
    NormalizationConfig,
)


# ──────────────────────────────────────────────────────────────────────
# VAE config
# ──────────────────────────────────────────────────────────────────────

class VAEConfig(BaseModel):
    """VAE bottleneck configuration."""

    latent_dim: int = Field(default=256, gt=0, description="Total latent dimension (coarse + fine)")
    coarse_dim: int = Field(default=64, gt=0, description="Coarse latent dims (behavioral category)")
    encoder_hidden_dims: List[int] = Field(
        default=[512, 384],
        description="Hidden layer dims for VAE encoder MLP",
    )
    decoder_hidden_dims: List[int] = Field(
        default=[384, 512],
        description="Hidden layer dims for feature decoder MLP",
    )


# ──────────────────────────────────────────────────────────────────────
# LFM adapter config
# ──────────────────────────────────────────────────────────────────────

class LFMAdapterConfig(BaseModel):
    """Configuration for the LFM generator adapter."""

    latent_dim: int = Field(default=256, gt=0, description="Must match VAEConfig.latent_dim")
    vocab_size: int = Field(default=8000, gt=0, description="SentencePiece vocabulary size")
    max_output_len: int = Field(default=256, ge=16, description="Max NL tokens to generate")
    decoder_hidden_dim: int = Field(default=256, gt=0)
    decoder_num_layers: int = Field(default=2, ge=1)
    decoder_num_heads: int = Field(default=4, ge=1)
    pretrained_decoder_path: Optional[Path] = Field(
        default=None,
        description="Path to pretrained LFM decoder checkpoint",
    )
    spm_model_path: Optional[Path] = Field(
        default=None,
        description="Path to SentencePiece model for text decoding",
    )
    freeze_decoder: bool = Field(default=True, description="Freeze decoder weights during training")
    temperature: float = Field(default=1.0, gt=0.0, description="Gumbel-Softmax temperature")
    hard_sample: bool = Field(default=True, description="Use hard Gumbel-Softmax samples")


# ──────────────────────────────────────────────────────────────────────
# NL Listener config
# ──────────────────────────────────────────────────────────────────────

class NLListenerConfig(BaseModel):
    """Configuration for the NL listener (text → z decoder)."""

    vocab_size: int = Field(default=8000, gt=0, description="Must match LFMAdapterConfig.vocab_size")
    hidden_dim: int = Field(default=256, gt=0, description="Transformer hidden dim")
    num_heads: int = Field(default=4, ge=1)
    num_layers: int = Field(default=2, ge=1)
    latent_dim: int = Field(default=256, gt=0, description="Must match VAEConfig.latent_dim")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────
# Inverse decoder config
# ──────────────────────────────────────────────────────────────────────

class NLInverseConfig(BaseModel):
    """Configuration for z → parameter inverse decoders."""

    theta_hidden_dim: int = Field(default=128, gt=0)
    theta_dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    ic_hidden_dim: int = Field(default=256, gt=0)
    ic_dropout: float = Field(default=0.1, ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────
# Loss config
# ──────────────────────────────────────────────────────────────────────

class NLLossConfig(BaseModel):
    """Loss weights and settings for NLTokenizer training.

    Inverse losses use **behavioral equivalence**: predicted parameters are
    re-encoded through the same encoder and compared in embedding space,
    not parameter space. This handles the many-to-many mapping from
    (theta, IC) → behavior correctly.
    """

    reconstruction_weight: float = Field(default=1.0, ge=0.0, description="Feature reconstruction MSE")
    kl_weight: float = Field(default=0.1, ge=0.0, description="KL divergence (after warmup)")
    kl_free_bits: float = Field(default=2.0, ge=0.0, description="Free-bits floor per latent dim")
    theta_inverse_weight: float = Field(
        default=1.0, ge=0.0,
        description="Behavioral theta inverse: ‖encoder(θ_hat) - encoder(θ_true)‖²",
    )
    ic_inverse_weight: float = Field(default=0.5, ge=0.0, description="IC inverse L2")
    listener_roundtrip_weight: float = Field(default=1.0, ge=0.0, description="Listener z roundtrip L2")
    topographic_weight: float = Field(
        default=0.5, ge=0.0,
        description="Topographic: preserve behavioral neighborhoods in z-space",
    )
    topographic_n_samples: int = Field(
        default=64, ge=4,
        description="Samples per batch for topographic pairwise distance computation",
    )


# ──────────────────────────────────────────────────────────────────────
# Training config
# ──────────────────────────────────────────────────────────────────────

class NLTrainingConfig(BaseModel):
    """Training hyperparameters for NLTokenizer."""

    num_epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=128, ge=1)
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

    # ── VAE training stages ──
    kl_warmup_epochs: int = Field(
        default=10, ge=0,
        description="Epochs before KL weight ramps to full value",
    )
    listener_start_epoch: int = Field(
        default=10, ge=0,
        description="Epoch to enable listener roundtrip loss",
    )
    log_every_n_batches: int = Field(default=50, ge=1)
    checkpoint_every_n_epochs: int = Field(default=10, ge=1)


# ──────────────────────────────────────────────────────────────────────
# Top-level NLTokenizer config
# ──────────────────────────────────────────────────────────────────────

class NLTokenizerConfig(BaseModel):
    """Complete configuration for NLTokenizer."""

    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    vae: VAEConfig = Field(default_factory=VAEConfig)
    lfm_adapter: LFMAdapterConfig = Field(default_factory=LFMAdapterConfig)
    listener: NLListenerConfig = Field(default_factory=NLListenerConfig)
    inverse: NLInverseConfig = Field(default_factory=NLInverseConfig)
    loss: NLLossConfig = Field(default_factory=NLLossConfig)
    training: NLTrainingConfig = Field(default_factory=NLTrainingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    feature_cleaning: Optional[FeatureCleaningConfig] = None

    # ── Dataset / pipeline settings ──
    feature_source: Literal["manual", "learned"] = Field(
        default="learned",
        description=(
            "Feature extraction mode. 'learned' uses PyramidFirstEncoder on raw "
            "trajectories (production path). 'manual' uses pre-extracted temporal "
            "features (legacy)."
        ),
    )
    random_seed: Optional[int] = None
    verbose: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    generation_config_path: Optional[str] = Field(
        default=None,
        description="Path to Lenia generation config (for replayer + auto-detection)",
    )
    generation_timesteps: Optional[int] = Field(
        default=None, ge=1,
        description="Trajectory timesteps for on-the-fly generation (auto-detected if None)",
    )
    realization_mode: Literal["single", "mean", "all"] = Field(
        default="single",
        description="How to aggregate stochastic realizations per operator",
    )
    replayer_cache_size: int = Field(default=8, ge=0)
    checkpoint_dir: Optional[str] = None
