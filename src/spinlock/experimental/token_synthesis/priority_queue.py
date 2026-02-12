"""Surprisal-scored max-heap priority queue for novel token sequences.

Stores generated/retokenized token pairs scored by surprisal, with token
frequency tracking for computing usage-based priority adjustments.

The priority queue feeds the refinement phase: high-surprisal items
(where generated tokens diverged most from physical grounding) are
popped first for training the diffusion model.
"""

import heapq
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from spinlock.experimental.token_synthesis.config import PriorityConfig

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PriorityItem:
    """A scored token sequence in the priority queue.

    Uses negated priority for max-heap behavior (Python heapq is min-heap).
    """
    priority: float                                           # Negated for max-heap
    index: int = field(compare=False)                         # Insertion order (tiebreaker)
    surprisal: float = field(compare=False)
    generated_tokens: Dict[str, int] = field(compare=False)   # {key: token_idx}
    retokenized_tokens: Dict[str, int] = field(compare=False)
    theta: Optional[np.ndarray] = field(default=None, compare=False)  # [param_dim]


class SurprisalPriorityQueue:
    """Max-heap priority queue for surprisal-scored token sequences.

    Manages a bounded queue of generated/retokenized token pairs. Items
    are scored by a weighted combination of:
    - surprisal: How much the roundtrip changed the tokens
    - usage_frequency: How common this token combination is (inversely weighted)
    - rarity: How rare the individual tokens are across all seen samples

    Args:
        config: PriorityConfig with alpha/beta/gamma weights and capacity
        vocab_sizes: Dict mapping quantizer key → vocab size
    """

    def __init__(self, config: PriorityConfig, vocab_sizes: Dict[str, int]):
        self.config = config
        self.vocab_sizes = vocab_sizes
        self._heap: List[PriorityItem] = []
        self._counter = 0  # Monotonic insertion counter for tiebreaking

        # Token frequency tracking: counts per (key, token_value) pair
        self._token_freq: Counter = Counter()
        self._total_samples = 0

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def fill_fraction(self) -> float:
        return len(self._heap) / self.config.queue_capacity if self.config.queue_capacity > 0 else 0.0

    def compute_priority(self, surprisal: float, tokens: Dict[str, int]) -> float:
        """Compute priority score for a token sequence.

        priority = alpha * surprisal + beta * novelty + gamma * rarity

        Where:
        - novelty = 1 - usage_frequency (how novel this combination is)
        - rarity = mean inverse frequency of individual tokens

        Args:
            surprisal: Surprisal score from verification
            tokens: Generated token dict {key: token_value}

        Returns:
            Priority score (higher = more important for refinement)
        """
        # Novelty: inverse of how often we've seen similar token combos
        if self._total_samples > 0:
            seen_count = sum(
                self._token_freq.get((k, v), 0)
                for k, v in tokens.items()
            )
            avg_freq = seen_count / (len(tokens) * self._total_samples) if tokens else 0.0
            novelty = 1.0 - min(avg_freq, 1.0)
        else:
            novelty = 1.0  # First samples are maximally novel

        # Rarity: mean inverse frequency of individual tokens
        if self._total_samples > 0:
            rarities = []
            for k, v in tokens.items():
                freq = self._token_freq.get((k, v), 0) / self._total_samples
                rarities.append(1.0 - min(freq, 1.0))
            rarity = sum(rarities) / len(rarities) if rarities else 0.0
        else:
            rarity = 1.0

        priority = (
            self.config.alpha * surprisal
            + self.config.beta * novelty
            + self.config.gamma * rarity
        )

        return priority

    def _update_frequency(self, tokens: Dict[str, int]) -> None:
        """Update token frequency counters."""
        self._total_samples += 1
        for k, v in tokens.items():
            self._token_freq[(k, v)] += 1

    def push(
        self,
        gen_tokens: Dict[str, int],
        retok_tokens: Dict[str, int],
        surprisal: float,
        theta: Optional[np.ndarray] = None,
    ) -> bool:
        """Push a single item onto the priority queue.

        If queue is at capacity, only inserts if priority exceeds the
        minimum priority currently in the queue.

        Args:
            gen_tokens: Generated token dict {key: token_value}
            retok_tokens: Retokenized token dict {key: token_value}
            surprisal: Surprisal score
            theta: Optional decoded parameter vector

        Returns:
            True if item was inserted, False if rejected
        """
        priority = self.compute_priority(surprisal, gen_tokens)
        self._update_frequency(gen_tokens)

        item = PriorityItem(
            priority=-priority,  # Negate for max-heap
            index=self._counter,
            surprisal=surprisal,
            generated_tokens=gen_tokens,
            retokenized_tokens=retok_tokens,
            theta=theta,
        )
        self._counter += 1

        if len(self._heap) < self.config.queue_capacity:
            heapq.heappush(self._heap, item)
            return True
        elif -priority < self._heap[0].priority:
            # New item has higher priority than minimum in queue
            heapq.heapreplace(self._heap, item)
            return True
        else:
            return False

    def push_batch(
        self,
        gen_tokens: Dict[str, torch.Tensor],
        retok_tokens: Dict[str, torch.Tensor],
        surprisals: torch.Tensor,
        thetas: Optional[torch.Tensor] = None,
    ) -> int:
        """Push a batch of items. Returns number actually inserted.

        Args:
            gen_tokens: Dict mapping key → [B] token tensors
            retok_tokens: Dict mapping key → [B] token tensors
            surprisals: [B] surprisal scores
            thetas: Optional [B, param_dim] parameter vectors

        Returns:
            Number of items inserted into queue
        """
        batch_size = surprisals.shape[0]
        inserted = 0

        for i in range(batch_size):
            gen_i = {k: v[i].item() for k, v in gen_tokens.items()}
            retok_i = {k: v[i].item() for k, v in retok_tokens.items()}
            theta_i = thetas[i].cpu().numpy() if thetas is not None else None

            if self.push(gen_i, retok_i, surprisals[i].item(), theta_i):
                inserted += 1

        return inserted

    def pop_batch(self, batch_size: int) -> List[PriorityItem]:
        """Pop up to batch_size items with highest priority.

        Args:
            batch_size: Maximum number of items to pop

        Returns:
            List of PriorityItems sorted by priority (highest first)
        """
        items = []
        for _ in range(min(batch_size, len(self._heap))):
            item = heapq.heappop(self._heap)
            item.priority = -item.priority  # Restore original priority
            items.append(item)

        return items
