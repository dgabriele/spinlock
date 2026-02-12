"""Token synthesis self-play pipeline.

Composes diffusion generation, VQ tokenization, and physics simulation
into an explore-refine loop that discovers novel token sequences and
teaches the diffusion model to generate physically coherent tokens.
"""

from spinlock.experimental.token_synthesis.config import TokenSynthesisConfig
from spinlock.experimental.token_synthesis.pipeline import SynthesisVerificationPipeline
from spinlock.experimental.token_synthesis.verification import SurprisalComputer
from spinlock.experimental.token_synthesis.priority_queue import SurprisalPriorityQueue
from spinlock.experimental.token_synthesis.scheduler import ModeScheduler, Mode

__all__ = [
    "TokenSynthesisConfig",
    "SynthesisVerificationPipeline",
    "SurprisalComputer",
    "SurprisalPriorityQueue",
    "ModeScheduler",
    "Mode",
]
