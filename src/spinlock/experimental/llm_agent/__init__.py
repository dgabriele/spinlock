"""LLM-aligned conversational agent for token synthesis.

Rosetta alignment layer between the denoiser's discrete token language
and a pretrained English LLM (frozen + LoRA adapters). Enables
conversational control over token synthesis exploration via natural
language queries.

Core components:
- TokenEmbedder: Extracts embeddings from denoiser hidden states
- TemplateDescriptor: Auto-describes physics from quantitative features
- RosettaAlignment: CLIP-style contrastive alignment (token ↔ text)
- RLAIFEvaluator: LLM-as-judge scoring
- ConversationalAgent: Top-level chat() interface

Requires optional dependencies: poetry install --with llm
"""

from spinlock.experimental.llm_agent.config import (
    AgentConfig,
    AlignmentConfig,
    AutonomousTrainingConfig,
    ContinuousExplorationConfig,
    EvaluatorConfig,
    LLMConfig,
)

__all__ = [
    "AgentConfig",
    "AlignmentConfig",
    "AutonomousTrainingConfig",
    "ContinuousExplorationConfig",
    "EvaluatorConfig",
    "LLMConfig",
]
