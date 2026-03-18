"""Mixed masking strategy that samples from multiple strategies per batch."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .hierarchical_masking import HierarchicalMaskGenerator, MaskingStrategy

logger = logging.getLogger(__name__)


class MixedMaskGenerator:
    """Generate masks by randomly sampling from multiple strategies.

    Useful for training a single model that handles diverse masking patterns.
    Each batch samples a strategy according to provided weights.

    Args:
        strategies: List of (MaskingStrategy, weight) tuples
        vocab_sizes: Dict mapping "family_category_Ll" → vocab_size
        category_level_info: Dict mapping key → {family, category, level}
        seed: Random seed

    Example:
        >>> mixed_gen = MixedMaskGenerator(
        ...     strategies=[
        ...         (MaskingStrategy.RANDOM, 0.5),      # 50% of batches
        ...         (MaskingStrategy.COARSE_ONLY, 0.3), # 30% of batches
        ...         (MaskingStrategy.HIERARCHICAL, 0.2), # 20% of batches
        ...     ],
        ...     vocab_sizes=vocab_sizes,
        ...     category_level_info=category_level_info,
        ... )
    """

    def __init__(
        self,
        strategies: List[Tuple[MaskingStrategy, float]],
        vocab_sizes: Dict[str, int],
        category_level_info: Dict[str, Dict[str, any]],
        mask_probability: float = 0.5,
        seed: int = 42,
        always_masked_families: Optional[List[str]] = None,
        always_observed_families: Optional[List[str]] = None,
    ):
        # Validate weights sum to 1
        total_weight = sum(weight for _, weight in strategies)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Strategy weights must sum to 1.0, got {total_weight}")

        self.strategies = strategies
        self.vocab_sizes = vocab_sizes
        self.category_level_info = category_level_info
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Create generator for each strategy (propagate family overrides)
        self.generators = {}
        for strategy, _ in strategies:
            self.generators[strategy] = HierarchicalMaskGenerator(
                strategy=strategy,
                vocab_sizes=vocab_sizes,
                category_level_info=category_level_info,
                mask_probability=mask_probability,
                seed=seed,
                always_masked_families=always_masked_families,
                always_observed_families=always_observed_families,
            )

        logger.info(
            f"MixedMaskGenerator initialized: {len(strategies)} strategies, "
            f"weights={[w for _, w in strategies]}"
        )

    def generate_dict_mask(
        self, batch_size: int
    ) -> Tuple[Dict[str, torch.BoolTensor], Dict[str, torch.BoolTensor]]:
        """Generate mask by randomly sampling a strategy.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (observed_dict, target_dict)
        """
        # Sample strategy by index to avoid numpy truncating str-enum values
        strategy_list = [s for s, _ in self.strategies]
        weights = [w for _, w in self.strategies]
        idx = self.rng.choice(len(strategy_list), p=weights)
        chosen_strategy = strategy_list[idx]

        # Generate mask with chosen strategy
        generator = self.generators[chosen_strategy]
        return generator.generate_dict_mask(batch_size)

    def get_strategy_distribution(self, num_batches: int = 1000) -> Dict[str, float]:
        """Get empirical distribution of strategies over many batches.

        Useful for verifying that sampling matches specified weights.

        Args:
            num_batches: Number of batches to simulate

        Returns:
            Dict mapping strategy name → empirical probability
        """
        counts = {s: 0 for s, _ in self.strategies}
        strategy_list = [s for s, _ in self.strategies]
        weights = [w for _, w in self.strategies]

        for _ in range(num_batches):
            idx = self.rng.choice(len(strategy_list), p=weights)
            counts[strategy_list[idx]] += 1

        return {
            str(strategy): count / num_batches
            for strategy, count in counts.items()
        }
