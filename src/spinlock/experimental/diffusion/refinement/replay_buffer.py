"""Prioritized replay buffer for D3PM refinement.

Priority-proportional sampling: hard targets (low agreement) are replayed
more frequently, focusing compute on the model's weakest points.

Priority: P(i) ∝ (1 - agreement_i)^α + ε

Literature: Prioritized Experience Replay (Schaul et al. 2016),
Prioritized Generative Replay (ICLR 2025).
"""

import random
from typing import Dict, List

import torch


class PrioritizedReplayBuffer:
    """Priority-proportional replay buffer with reservoir sampling.

    Samples with low agreement (high surprise) are replayed more frequently.
    Uses reservoir sampling for capacity management when buffer is full.

    Args:
        max_size: Maximum buffer capacity.
        alpha: Priority exponent. 0 = uniform sampling, 1 = proportional
            to surprise. Higher values concentrate sampling on harder targets.
    """

    def __init__(self, max_size: int, alpha: float = 1.0):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer: List[Dict] = []
        self._count = 0  # total items seen (for reservoir sampling)

    def add(self, targets: List[Dict]):
        """Add targets, using reservoir sampling when buffer is full."""
        for item in targets:
            self._count += 1
            if len(self.buffer) < self.max_size:
                self.buffer.append(item)
            else:
                idx = random.randint(0, self._count - 1)
                if idx < self.max_size:
                    self.buffer[idx] = item

    def sample(self, n: int) -> List[Dict]:
        """Priority-proportional sampling. Hard targets sampled more often.

        When alpha=0, falls back to uniform sampling.
        Uses replacement to allow oversampling of high-priority items.
        """
        if not self.buffer:
            return []
        k = min(n, len(self.buffer))
        if self.alpha == 0.0:
            return random.choices(self.buffer, k=k)
        priorities = torch.tensor([
            (1.0 - item["agreement"]) ** self.alpha + 1e-6
            for item in self.buffer
        ])
        priorities = priorities / priorities.sum()
        indices = torch.multinomial(priorities, k, replacement=True)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def state_dict(self) -> dict:
        return {"buffer": self.buffer, "count": self._count}

    def load_state_dict(self, state: dict):
        self.buffer = state["buffer"]
        self._count = state["count"]
