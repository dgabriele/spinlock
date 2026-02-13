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
from typing import List, Literal, Optional, Tuple

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

    priority = alpha * surprisal + delta * repr_trace
             + epsilon * structural_growth + gamma * rarity

    Higher priority items are refined first. When ``learned=True``, a
    ``nn.Linear(4, 1)`` head replaces the fixed weights after
    ``warmup_cycles`` cycles, trained to predict which items yield the
    greatest refinement improvement (measured by Jaccard delta).

    repr_trace and structural_growth are stubbed (return 0.0) until the
    DenoisingNetwork exposes hidden-state access.
    """
    alpha: float = Field(default=0.6, ge=0.0)       # surprisal
    delta: float = Field(default=0.0, ge=0.0)        # repr_trace (stub)
    epsilon: float = Field(default=0.0, ge=0.0)      # structural_growth (stub)
    gamma: float = Field(default=0.15, ge=0.0)       # rarity
    queue_capacity: int = Field(default=1000, ge=1)
    min_queue_for_refinement: int = Field(default=64, ge=1)
    learned: bool = Field(default=True, description="Use learned priority head after warmup")
    warmup_cycles: int = Field(default=3, ge=0, description="Cycles with fixed weights before learning")
    priority_lr: float = Field(default=0.01, gt=0.0, description="Learning rate for priority head")


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


class CounterfactualConfig(BaseModel):
    """Counterfactual surprisal: novelty via masked token prediction.

    Masks a subset of token keys, forward-diffuses masked positions to
    full noise, and has the denoiser predict them from observed context.
    Cross-entropy on masked positions measures how surprising the token
    combination is to the current denoiser — high CE = genuinely novel.

    Two masking strategies are mixed:
    - Category-level: mask all levels of 1+ random categories (structured)
    - Random: each key independently masked with probability p (unstructured)
    """
    mask_fraction: float = Field(default=0.5, ge=0.0, le=1.0,
        description="Fraction of keys to mask (random strategy)")
    max_categories_to_mask: int = Field(default=3, ge=1,
        description="Max categories to mask (category strategy)")
    category_mask_prob: float = Field(default=0.5, ge=0.0, le=1.0,
        description="P(category-level masking) vs P(random masking)")
    t_probe: Optional[int] = Field(default=None,
        description="Forward-diffusion timestep. None = T (full noise)")


class ResamplingConfig(BaseModel):
    """Physically-guided resampling: fix sampling errors via RePaint inpainting.

    After the first QBM roundtrip identifies mismatched tokens (~15%), uses
    D3PM inpainting to re-generate only mismatched keys conditioned on
    physically-grounded matching keys. Then re-runs QBM for better grounding.

    The correction timestep is adaptive: T_correction = T × mismatch_rate × noise_scale.
    This ensures the noise level matches the correction difficulty — high mismatch
    rates get more noise (exploration), low rates get less (conservative refinement).
    Both T and mismatch_rate are auto-detected from the model and data.
    """
    noise_scale: float = Field(default=1.0, gt=0.0,
        description="Scale factor for adaptive correction timestep. "
                    "T_correction = T × mismatch_rate × noise_scale. "
                    "Values > 1.0 allow more exploration, < 1.0 keep corrections conservative.")
    max_correction_steps: Optional[int] = Field(default=None, ge=1,
        description="Hard cap on correction steps regardless of adaptive formula. "
                    "None = no cap (adaptive only). Useful as a safety valve.")


class AdaptConfig(BaseModel):
    """VQTokenizer online refinement configuration.

    When enabled, the pipeline periodically fine-tunes the VQTokenizer
    on accumulated exploration experiences mixed with original training data.
    After fine-tuning, the priority queue is retokenized with the updated
    codebooks.
    """
    enabled: bool = Field(default=False,
        description="Enable periodic VQTokenizer fine-tuning")
    frequency: int = Field(default=5, ge=1,
        description="Run ADAPT every N synthesis cycles")
    num_epochs: int = Field(default=10, ge=1,
        description="Fine-tuning epochs per ADAPT phase")
    learning_rate: float = Field(default=1e-5, gt=0.0,
        description="Lower LR than initial training for stability")
    original_mix_ratio: float = Field(default=0.3, ge=0.0, le=1.0,
        description="Fraction of original training data mixed in (anti-forgetting)")
    max_experiences: int = Field(default=2000, ge=1,
        description="Max experiences to accumulate in buffer")
    min_experiences: int = Field(default=100, ge=1,
        description="Min experiences needed to trigger ADAPT")
    anchor_cache_size: int = Field(default=5000, ge=100,
        description="Number of original training samples to cache for anti-forgetting. "
                    "Should be 10-20% of original training set size.")
    anchor_refresh_interval: int = Field(default=5, ge=1,
        description="Re-randomize anchor cache every N ADAPT cycles for broader coverage")
    retokenize_replay: bool = Field(default=False,
        description="Whether to retokenize replay buffer after ADAPT. "
                    "Disabled by default: replay items represent the denoiser's "
                    "training history and retokenizing them destabilizes training.")
    original_features_path: Optional[Path] = Field(default=None,
        description="Path to pre-extracted original training features HDF5")

    @field_validator("original_features_path")
    @classmethod
    def validate_features_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"Original features path not found: {v}")
        return v


class TokenSynthesisConfig(BaseModel):
    """Top-level configuration for token synthesis self-play pipeline."""
    checkpoints: CheckpointPaths
    generation: GenerationConfig = GenerationConfig()
    rollout: RolloutConfig = RolloutConfig()
    surprisal: SurprisalConfig = SurprisalConfig()
    priority: PriorityConfig = PriorityConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    refinement: RefinementConfig = RefinementConfig()
    adapt: AdaptConfig = AdaptConfig()
    counterfactual: Optional[CounterfactualConfig] = None
    resampling: Optional[ResamplingConfig] = None
    output_dir: Path = Path("experiments/token_synthesis/results")
    device: str = "cuda"
    seed: int = 42

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
