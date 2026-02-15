"""Configuration for LLM-aligned conversational agent.

Pydantic models for the Rosetta alignment layer between the denoiser's
discrete token language and a pretrained English LLM (frozen + LoRA).
Follows the same patterns as token_synthesis/config.py.
"""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Frozen LLM backbone configuration."""

    model_name: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description=(
            "HuggingFace model ID. Small models preferred for single-GPU "
            "alongside denoiser and tokenizer."
        ),
    )
    lora_rank: int = Field(
        default=16,
        ge=1,
        description="LoRA adapter rank. 16 is good for alignment tasks.",
    )
    lora_alpha: float = Field(default=32.0, gt=0.0)
    lora_target_modules: List[str] = Field(
        default=["q_proj", "v_proj"],
        description="Modules to apply LoRA to",
    )
    max_description_tokens: int = Field(default=128, ge=16)
    device: str = "cuda"


class AlignmentConfig(BaseModel):
    """Contrastive Rosetta alignment training."""

    shared_dim: int = Field(
        default=256,
        ge=64,
        description="Shared embedding dimension for contrastive alignment",
    )
    temperature: float = Field(
        default=0.07,
        gt=0.0,
        description="InfoNCE temperature (CLIP default: 0.07)",
    )
    learning_rate: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=32, ge=1)
    num_negatives: int = Field(
        default=31,
        ge=1,
        description="In-batch negatives for InfoNCE. batch_size-1 by default.",
    )
    description_mode: Literal["template", "llm_zeroshot", "hybrid"] = Field(
        default="template",
        description="How to generate physics descriptions for training pairs",
    )
    accumulation_threshold: int = Field(
        default=256,
        ge=32,
        description="Min paired examples before first alignment training",
    )
    train_epochs: int = Field(default=5, ge=1)


class EvaluatorConfig(BaseModel):
    """RLAIF: LLM-as-evaluator configuration."""

    enabled: bool = Field(default=False, description="Enable RLAIF evaluation loop")
    score_prompt_template: str = Field(
        default=(
            "Rate 0-10 how well this physical system matches '{query}': "
            "{description}\nRespond with just the number."
        ),
        description="Template for LLM scoring prompt",
    )
    reward_threshold: float = Field(
        default=6.0,
        ge=0.0,
        le=10.0,
        description="Min LLM score to count as a match",
    )
    top_k_refinement: int = Field(
        default=8,
        ge=1,
        description="Number of top-scored items to feed back for refinement",
    )


class ContinuousExplorationConfig(BaseModel):
    """Background exploration between user prompts.

    When enabled, the agent continues exploring in a background thread
    after returning the initial response. Each iteration progressively
    refines results (best-so-far become frontier seeds). The loop stops
    cleanly when the next user message arrives.
    """

    enabled: bool = Field(
        default=False,
        description="Enable continuous background exploration",
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        description="Max background batches before auto-stop",
    )
    use_frontier_refinement: bool = Field(
        default=True,
        description="Use best-so-far as frontier seeds after first batch",
    )
    score_batch_size: int = Field(
        default=8,
        ge=1,
        description=(
            "In swap mode, batch this many iterations of results "
            "before loading LLM for scoring"
        ),
    )


class AutonomousTrainingConfig(BaseModel):
    """Configuration for autonomous alignment self-play training.

    Controls the DISCOVER/LEARN loop that bootstraps the Rosetta alignment
    without human interaction. NOT nested inside AgentConfig — this is a
    training-loop concern, passed separately to AutonomousAlignmentTrainer.
    """

    num_cycles: int = Field(default=50, ge=1, description="Number of DISCOVER/LEARN cycles")
    queries_per_cycle: int = Field(default=10, ge=1, description="Queries generated per cycle")
    candidates_per_query: int = Field(
        default=16, ge=1, description="Exploration candidates per query"
    )
    checkpoint_frequency: int = Field(
        default=5, ge=1, description="Save checkpoint every N cycles"
    )
    enabled_categories: Optional[List[str]] = Field(
        default=None, description="Category filter (None = all categories)"
    )
    discovery_history_size: int = Field(
        default=50, ge=1, description="Max discoveries to keep for query context"
    )
    use_hybrid_descriptions: bool = Field(
        default=False,
        description="If True, enrich template descriptions with LLM in Phase 3",
    )
    max_enrichments_per_cycle: int = Field(
        default=8, ge=0, description="Max LLM-enriched descriptions per cycle"
    )
    accumulate_all: bool = Field(
        default=False,
        description=(
            "If True, accumulate ALL pairs regardless of score. "
            "If False (default), only accumulate pairs scoring >= threshold."
        ),
    )
    output_dir: Path = Field(
        default=Path("experiments/alignment/results"),
        description="Output directory for checkpoints and metrics",
    )


class AgentConfig(BaseModel):
    """Top-level conversational agent configuration."""

    llm: LLMConfig = LLMConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    continuous: ContinuousExplorationConfig = ContinuousExplorationConfig()
    memory_mode: Literal["resident", "swap"] = Field(
        default="resident",
        description=(
            "'resident': LLM stays on GPU at all times. "
            "'swap': LLM offloads to CPU during exploration, "
            "loads back for scoring/response."
        ),
    )
    denoiser_hidden_dim: Optional[int] = Field(
        default=None,
        description="Auto-detected from denoiser checkpoint",
    )
    num_token_positions: Optional[int] = Field(
        default=None,
        description="Auto-detected from denoiser (number of token keys, e.g. 84)",
    )
    description_cache_size: int = Field(
        default=10000,
        ge=100,
        description="Max cached (embedding, description) pairs",
    )
    system_prompt: str = Field(
        default=(
            "You are a physics exploration assistant. You help users discover "
            "interesting quantum-inspired dynamical systems by translating "
            "natural language descriptions into token-space exploration targets. "
            "Ground your responses in physical observables (energy, entropy, "
            "Lyapunov exponents, spectral peaks) and explain what you find."
        ),
        description="System prompt for the conversational agent",
    )
