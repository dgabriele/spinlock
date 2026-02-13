"""Significance-weighted priority queue for token sequences.

Stores generated/retokenized token pairs scored by a 4-term priority:
  priority = α×surprisal + δ×repr_trace + ε×structural_growth + γ×rarity

The priority queue feeds the refinement phase: high-priority items
(where generated tokens are most epistemically valuable) are popped
first for training the diffusion model.

Optionally includes a learned priority head (``nn.Linear(4, 1)``) that
replaces the fixed weights after a warmup period, trained to predict
which items yield the greatest refinement improvement.
"""

import heapq
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from spinlock.experimental.token_synthesis.config import PriorityConfig

logger = logging.getLogger(__name__)


class LearnedPriorityHead(nn.Module):
    """Linear layer mapping 4-dim feature vector to priority score.

    Features: [surprisal, repr_trace, structural_growth, rarity]

    Initialized to match the fixed α/δ/ε/γ weights so the transition
    from warmup (fixed) to learned is smooth — no priority discontinuity.

    After training, the 4 weights are directly interpretable as learned
    importance factors for each significance term.
    """

    def __init__(
        self,
        init_weights: Tuple[float, float, float, float] = (0.6, 0.0, 0.0, 0.15),
    ):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([list(init_weights)]))
            self.linear.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map feature vectors to priority scores.

        Args:
            features: [N, 4] tensor of
                [surprisal, repr_trace, structural_growth, rarity]

        Returns:
            [N] tensor of priority scores
        """
        return self.linear(features).squeeze(-1)

    @property
    def weights(self) -> Tuple[float, float, float, float]:
        """Return current learned (alpha, delta, epsilon, gamma)."""
        w = self.linear.weight.detach().squeeze()
        return (w[0].item(), w[1].item(), w[2].item(), w[3].item())

    @property
    def bias(self) -> float:
        return self.linear.bias.detach().item()


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
    features: Optional[np.ndarray] = field(default=None, compare=False)  # [4]: sur, rt, sg, rar
    raw_features: Optional[Dict[str, np.ndarray]] = field(default=None, compare=False)  # For ADAPT retokenization


class SurprisalPriorityQueue:
    """Max-heap priority queue for significance-weighted token sequences.

    Manages a bounded queue of generated/retokenized token pairs. Items
    are scored by a 4-term weighted combination of:
    - surprisal: How much the roundtrip changed the tokens
    - repr_trace: Hidden-state shift on probes (stub, returns 0.0)
    - structural_growth: Effective dimensionality growth (stub, returns 0.0)
    - rarity: How rare the individual tokens are across all seen samples

    When ``config.learned`` is True, a ``LearnedPriorityHead`` replaces
    the fixed weights after ``config.warmup_cycles`` cycles. The head is
    trained via MSE on improvement deltas (Jaccard change between cycles).

    Args:
        config: PriorityConfig with α/δ/ε/γ weights and capacity
        vocab_sizes: Dict mapping quantizer key -> vocab size
    """

    def __init__(self, config: PriorityConfig, vocab_sizes: Dict[str, int]):
        self.config = config
        self.vocab_sizes = vocab_sizes
        self._heap: List[PriorityItem] = []
        self._counter = 0  # Monotonic insertion counter for tiebreaking

        # Token frequency tracking: counts per (key, token_value) pair
        self._token_freq: Counter = Counter()
        self._total_samples = 0

        # Learned priority head (optional)
        self._head: Optional[LearnedPriorityHead] = None
        self._head_optimizer: Optional[torch.optim.SGD] = None
        self._head_active = False  # True after warmup

        # Training data for the head: list of (features[N,4], labels[N]) tensors
        # Labels are per-item: normalized_loss * max(delta, 0)
        self._training_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Per-item (features, normalized_loss) pairs for the CURRENT cycle
        self._current_cycle_data: List[Tuple[np.ndarray, float]] = []

        if config.learned:
            self._head = LearnedPriorityHead(
                init_weights=(config.alpha, config.delta,
                              config.epsilon, config.gamma),
            )
            self._head_optimizer = torch.optim.SGD(
                self._head.parameters(), lr=config.priority_lr,
            )

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def fill_fraction(self) -> float:
        return len(self._heap) / self.config.queue_capacity if self.config.queue_capacity > 0 else 0.0

    def _compute_features(
        self, surprisal: float, tokens: Dict[str, int],
    ) -> np.ndarray:
        """Compute the 4-dim feature vector for priority scoring.

        Features: [surprisal, repr_trace, structural_growth, rarity]

        - surprisal: passed in from verification
        - repr_trace: stub (0.0) — needs DenoisingNetwork hidden-state access
        - structural_growth: stub (0.0) — needs DenoisingNetwork hidden-state access
        - rarity: mean inverse frequency of individual tokens

        Returns:
            np.ndarray of shape [4]
        """
        # Rarity: mean inverse frequency of individual tokens
        if self._total_samples > 0:
            rarities = []
            for k, v in tokens.items():
                freq = self._token_freq.get((k, v), 0) / self._total_samples
                rarities.append(1.0 - min(freq, 1.0))
            rarity = sum(rarities) / len(rarities) if rarities else 0.0
        else:
            rarity = 1.0

        return np.array([
            surprisal,
            0.0,                        # repr_trace (stub)
            0.0,                        # structural_growth (stub)
            rarity,
        ], dtype=np.float32)

    def compute_priority(self, surprisal: float, tokens: Dict[str, int]) -> Tuple[float, np.ndarray]:
        """Compute priority score for a token sequence.

        When the learned head is active (post-warmup), uses the neural
        network. Otherwise uses the fixed weighted sum.

        Args:
            surprisal: Surprisal score from verification
            tokens: Generated token dict {key: token_value}

        Returns:
            Tuple of (priority_score, features_array[4])
        """
        features = self._compute_features(surprisal, tokens)

        if self._head_active and self._head is not None:
            with torch.no_grad():
                feat_tensor = torch.tensor(features).unsqueeze(0)  # [1, 4]
                priority = self._head(feat_tensor).item()
        else:
            priority = (
                self.config.alpha * features[0]      # surprisal
                + self.config.delta * features[1]    # repr_trace (stub)
                + self.config.epsilon * features[2]  # structural_growth (stub)
                + self.config.gamma * features[3]    # rarity
            )

        return priority, features

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
        raw_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> bool:
        """Push a single item onto the priority queue.

        If queue is at capacity, only inserts if priority exceeds the
        minimum priority currently in the queue.

        Args:
            gen_tokens: Generated token dict {key: token_value}
            retok_tokens: Retokenized token dict {key: token_value}
            surprisal: Surprisal score
            theta: Optional decoded parameter vector
            raw_features: Optional raw feature dicts for ADAPT retokenization

        Returns:
            True if item was inserted, False if rejected
        """
        priority, features = self.compute_priority(surprisal, gen_tokens)
        self._update_frequency(gen_tokens)

        item = PriorityItem(
            priority=-priority,  # Negate for max-heap
            index=self._counter,
            surprisal=surprisal,
            generated_tokens=gen_tokens,
            retokenized_tokens=retok_tokens,
            theta=theta,
            features=features,
            raw_features=raw_features,
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
        raw_features_list: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> int:
        """Push a batch of items. Returns number actually inserted.

        Args:
            gen_tokens: Dict mapping key -> [B] token tensors
            retok_tokens: Dict mapping key -> [B] token tensors
            surprisals: [B] surprisal scores
            thetas: Optional [B, param_dim] parameter vectors
            raw_features_list: Optional list of per-item raw feature dicts
                for ADAPT retokenization

        Returns:
            Number of items inserted into queue
        """
        batch_size = surprisals.shape[0]
        inserted = 0

        for i in range(batch_size):
            gen_i = {k: v[i].item() for k, v in gen_tokens.items()}
            retok_i = {k: v[i].item() for k, v in retok_tokens.items()}
            theta_i = thetas[i].cpu().numpy() if thetas is not None else None
            raw_i = raw_features_list[i] if raw_features_list is not None else None

            if self.push(gen_i, retok_i, surprisals[i].item(), theta_i, raw_features=raw_i):
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

    def get_all_items(self) -> List[PriorityItem]:
        """Return all items in the queue (without removing them).

        Items are returned with their original (non-negated) priority.
        Useful for retokenization during ADAPT phase.
        """
        items = []
        for item in self._heap:
            # Create a copy with restored priority
            restored = PriorityItem(
                priority=-item.priority,
                index=item.index,
                surprisal=item.surprisal,
                generated_tokens=item.generated_tokens,
                retokenized_tokens=item.retokenized_tokens,
                theta=item.theta,
                features=item.features,
                raw_features=item.raw_features,
            )
            items.append(restored)
        return items

    def update_items_in_place(self, items: List[PriorityItem]) -> None:
        """Replace the queue contents with updated items.

        Used after ADAPT retokenization to refresh the queue with
        updated token values. Re-heapifies with negated priorities.

        Args:
            items: Items with updated retokenized_tokens
        """
        self._heap = []
        for item in items:
            heap_item = PriorityItem(
                priority=-item.priority,
                index=item.index,
                surprisal=item.surprisal,
                generated_tokens=item.generated_tokens,
                retokenized_tokens=item.retokenized_tokens,
                theta=item.theta,
                features=item.features,
                raw_features=item.raw_features,
            )
            heapq.heappush(self._heap, heap_item)

    # ─── Learned Priority Head ─────────────────────────────────────

    def record_refined_features(
        self,
        items: List[PriorityItem],
        per_item_losses: List[float],
    ) -> None:
        """Record feature vectors and losses of items being refined.

        Called by the pipeline during each refine epoch. Data accumulates
        until ``train_priority_head`` is called at the cycle boundary,
        which combines per-item losses with the cycle-level improvement
        delta to produce per-item training labels.

        Args:
            items: Items being refined (must have .features set)
            per_item_losses: Detached per-item refinement losses, one per item
        """
        for item, loss_val in zip(items, per_item_losses):
            if item.features is not None:
                self._current_cycle_data.append((item.features, loss_val))

    def train_priority_head(self, delta: float, cycle: int) -> Optional[float]:
        """Train the learned priority head on accumulated data.

        Converts per-item data into training labels using the two-level
        scheme: ``label_i = normalized_loss_i * max(delta, 0)``.

        - Per-item losses provide **attribution**: which items were hard
        - Cycle delta provides **validation**: did refinement actually help

        Items in failed cycles (negative delta) all get label 0 — they
        were hard but unhelpful. Items in successful cycles get labels
        proportional to their difficulty, rewarding the head for
        prioritizing the items the model learned most from.

        Args:
            delta: Jaccard improvement for this cycle (curr - prev)
            cycle: Current cycle number (for warmup check)

        Returns:
            Training loss if head was trained, None otherwise
        """
        if self._head is None:
            return None

        # Compute per-item labels and store training data
        if self._current_cycle_data:
            features_list = [f for f, _ in self._current_cycle_data]
            losses_list = [l for _, l in self._current_cycle_data]

            features_tensor = torch.tensor(
                np.stack(features_list), dtype=torch.float32,
            )
            raw_losses = torch.tensor(losses_list, dtype=torch.float32)

            # Normalize losses to [0, 1] within this cycle's batch
            loss_min = raw_losses.min()
            loss_max = raw_losses.max()
            if loss_max > loss_min:
                normalized_losses = (raw_losses - loss_min) / (loss_max - loss_min)
            else:
                normalized_losses = torch.ones_like(raw_losses)

            # Two-level label: per-item attribution × cycle-level validation
            clamped_delta = max(delta, 0.0)
            labels = normalized_losses * clamped_delta

            self._training_buffer.append((features_tensor, labels))
            self._current_cycle_data = []
        else:
            self._current_cycle_data = []
            return None

        # Activate head after warmup
        if cycle >= self.config.warmup_cycles:
            if not self._head_active:
                self._head_active = True
        else:
            return None  # Still in warmup, don't train yet

        # Retrain on full history
        self._head.train()
        total_loss = 0.0
        num_batches = 0

        for features, labels in self._training_buffer:
            predicted = self._head(features)  # [N]
            loss = nn.functional.mse_loss(predicted, labels)

            self._head_optimizer.zero_grad()
            loss.backward()
            self._head_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self._head.eval()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def get_learned_weights(self) -> Optional[Tuple[float, float, float, float, float]]:
        """Return current learned weights for logging.

        Returns:
            (alpha, delta, epsilon, gamma, bias) tuple, or None if no learned head
        """
        if self._head is None:
            return None
        w = self._head.weights
        return (w[0], w[1], w[2], w[3], self._head.bias)
