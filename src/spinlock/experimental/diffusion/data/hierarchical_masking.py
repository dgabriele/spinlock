"""Hierarchical masking strategies for dict-format VQTokenizer v2 tokens.

Extends temporal masking concepts to hierarchical dict tokens where:
- Tokens: Dict[str, Tensor[B]] (set-like, no temporal ordering)
- Keys: "family_category_Ll" format
- Strategies respect hierarchical structure and family relationships
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MaskingStrategy(str, Enum):
    """Masking strategies for hierarchical dict tokens.

    Unlike temporal masking, these operate on category-level sets
    rather than sequential tokens.
    """
    RANDOM = "random"                         # Randomly mask each category-level
    COARSE_ONLY = "coarse_only"               # Keep only L0, mask L1+L2
    HIERARCHICAL = "hierarchical"             # Keep L0+L1, mask L2
    FAMILY_SELECTIVE = "family_selective"     # Mask entire families
    MIXED = "mixed"                           # Sample from multiple strategies per batch (MixedMaskGenerator)


class HierarchicalMaskGenerator:
    """Generate dict masks for hierarchical VQTokenizer v2 tokens.

    Produces masks for training and evaluation that respect the
    hierarchical structure of dict tokens.

    Args:
        strategy: Masking strategy to use
        vocab_sizes: Dict mapping "family_category_Ll" → vocab_size
        category_level_info: Dict mapping key → {family, category, level}
        mask_probability: Probability of masking (for RANDOM strategy)
        seed: Random seed for reproducibility

    Example:
        >>> vocab_sizes = {"temporal_group_1_L0": 28, "temporal_group_1_L1": 14}
        >>> mask_gen = HierarchicalMaskGenerator(
        ...     strategy=MaskingStrategy.RANDOM,
        ...     vocab_sizes=vocab_sizes,
        ...     category_level_info=category_level_info,
        ...     mask_probability=0.5
        ... )
        >>> observed_dict, target_dict = mask_gen.generate_dict_mask(batch_size=32)
    """

    def __init__(
        self,
        strategy: MaskingStrategy,
        vocab_sizes: Dict[str, int],
        category_level_info: Dict[str, Dict[str, any]],
        mask_probability: float = 0.5,
        seed: int = 42,
        always_masked_families: Optional[List[str]] = None,
        always_observed_families: Optional[List[str]] = None,
    ):
        self.strategy = strategy
        self.vocab_sizes = vocab_sizes
        self.category_level_info = category_level_info
        self.mask_probability = mask_probability
        self.rng = np.random.RandomState(seed)

        # Family-level masking overrides
        self.always_masked_families = set(always_masked_families or [])
        self.always_observed_families = set(always_observed_families or [])

        # Build key → family mapping for fast override lookups
        self._family_for_key: Dict[str, str] = {
            key: info["family"]
            for key, info in category_level_info.items()
        }

        # Extract category structure
        self.families, self.categories_by_level = self._extract_categories()

        logger.info(
            f"HierarchicalMaskGenerator initialized: "
            f"strategy={strategy}, families={list(self.families.keys())}, "
            f"mask_prob={mask_probability}"
            + (f", always_masked={list(self.always_masked_families)}" if self.always_masked_families else "")
            + (f", always_observed={list(self.always_observed_families)}" if self.always_observed_families else "")
        )

    def _extract_categories(self) -> Tuple[Dict[str, set], Dict[int, list]]:
        """Extract family and level groupings from category_level_info.

        Returns:
            Tuple of (families_dict, categories_by_level_dict):
            - families: Dict mapping family → set of keys
            - categories_by_level: Dict mapping level → list of keys
        """
        families = {}
        categories_by_level = {}

        for key, info in self.category_level_info.items():
            family = info['family']
            level = info['level']

            if family not in families:
                families[family] = set()
            families[family].add(key)

            if level not in categories_by_level:
                categories_by_level[level] = []
            categories_by_level[level].append(key)

        return families, categories_by_level

    def generate_dict_mask(
        self, batch_size: int
    ) -> Tuple[Dict[str, torch.BoolTensor], Dict[str, torch.BoolTensor]]:
        """Generate mask dicts for a batch.

        After the base strategy produces masks, family-level overrides are
        applied: keys in always_masked_families are forced to target,
        keys in always_observed_families are forced to observed.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Tuple of (observed_dict, target_dict):
            - observed_dict: Dict mapping key → bool tensor [B] (True = observed)
            - target_dict: Dict mapping key → bool tensor [B] (True = predict)
            Both are complementary: target_dict[key] = ~observed_dict[key]
        """
        if self.strategy == MaskingStrategy.RANDOM:
            observed_dict, target_dict = self._generate_random_mask(batch_size)
        elif self.strategy == MaskingStrategy.COARSE_ONLY:
            observed_dict, target_dict = self._generate_coarse_only_mask(batch_size)
        elif self.strategy == MaskingStrategy.HIERARCHICAL:
            observed_dict, target_dict = self._generate_hierarchical_mask(batch_size)
        elif self.strategy == MaskingStrategy.FAMILY_SELECTIVE:
            observed_dict, target_dict = self._generate_family_selective_mask(batch_size)
        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")

        # Apply family-level overrides
        if self.always_masked_families or self.always_observed_families:
            self._apply_family_overrides(observed_dict, target_dict, batch_size)

        return observed_dict, target_dict

    def _apply_family_overrides(
        self,
        observed_dict: Dict[str, torch.BoolTensor],
        target_dict: Dict[str, torch.BoolTensor],
        batch_size: int,
    ) -> None:
        """Post-process masks to enforce family-level overrides in-place.

        Keys in always_masked_families → forced to target (masked).
        Keys in always_observed_families → forced to observed.
        """
        for key in self.vocab_sizes:
            family = self._family_for_key.get(key)
            if family is None:
                continue

            if family in self.always_masked_families:
                observed_dict[key] = torch.zeros(batch_size, dtype=torch.bool)
                target_dict[key] = torch.ones(batch_size, dtype=torch.bool)
            elif family in self.always_observed_families:
                observed_dict[key] = torch.ones(batch_size, dtype=torch.bool)
                target_dict[key] = torch.zeros(batch_size, dtype=torch.bool)

    def _generate_random_mask(
        self, batch_size: int
    ) -> Tuple[Dict[str, torch.BoolTensor], Dict[str, torch.BoolTensor]]:
        """Randomly mask each category-level with probability p.

        Primary training strategy for general-purpose token prediction.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (observed_dict, target_dict)
        """
        observed_dict = {}
        target_dict = {}

        for key in self.vocab_sizes.keys():
            # Random mask per category-level
            mask = self.rng.rand(batch_size) > self.mask_probability
            observed_dict[key] = torch.from_numpy(mask)
            target_dict[key] = ~observed_dict[key]

        return observed_dict, target_dict

    def _generate_coarse_only_mask(
        self, batch_size: int
    ) -> Tuple[Dict[str, torch.BoolTensor], Dict[str, torch.BoolTensor]]:
        """Keep only L0 (coarse), mask L1+L2.

        Tests hierarchical inference: can model predict fine details
        from coarse structure alone?

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (observed_dict, target_dict)
        """
        observed_dict = {}
        target_dict = {}

        for key, info in self.category_level_info.items():
            level = info['level']
            if level == 0:
                # Keep L0 (observed)
                observed_dict[key] = torch.ones(batch_size, dtype=torch.bool)
                target_dict[key] = torch.zeros(batch_size, dtype=torch.bool)
            else:
                # Mask L1, L2 (target)
                observed_dict[key] = torch.zeros(batch_size, dtype=torch.bool)
                target_dict[key] = torch.ones(batch_size, dtype=torch.bool)

        return observed_dict, target_dict

    def _generate_hierarchical_mask(
        self, batch_size: int
    ) -> Tuple[Dict[str, torch.BoolTensor], Dict[str, torch.BoolTensor]]:
        """Keep L0+L1, mask L2 (fine detail prediction).

        Tests fine detail inference from coarse+medium structure.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (observed_dict, target_dict)
        """
        observed_dict = {}
        target_dict = {}

        for key, info in self.category_level_info.items():
            level = info['level']
            if level <= 1:
                # Keep L0, L1 (observed)
                observed_dict[key] = torch.ones(batch_size, dtype=torch.bool)
                target_dict[key] = torch.zeros(batch_size, dtype=torch.bool)
            else:
                # Mask L2 (target)
                observed_dict[key] = torch.zeros(batch_size, dtype=torch.bool)
                target_dict[key] = torch.ones(batch_size, dtype=torch.bool)

        return observed_dict, target_dict

    def _generate_family_selective_mask(
        self, batch_size: int
    ) -> Tuple[Dict[str, torch.BoolTensor], Dict[str, torch.BoolTensor]]:
        """Mask entire families (e.g., all initial or all temporal).

        Tests cross-family relationships: can temporal predict initial?

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (observed_dict, target_dict)
        """
        observed_dict = {}
        target_dict = {}

        # Randomly select one family to mask (uniform over families)
        family_names = list(self.families.keys())
        if len(family_names) == 0:
            raise ValueError("No families found in category_level_info")

        masked_family = self.rng.choice(family_names)

        for key, info in self.category_level_info.items():
            family = info['family']
            if family == masked_family:
                # Mask this family (target)
                observed_dict[key] = torch.zeros(batch_size, dtype=torch.bool)
                target_dict[key] = torch.ones(batch_size, dtype=torch.bool)
            else:
                # Keep other families (observed)
                observed_dict[key] = torch.ones(batch_size, dtype=torch.bool)
                target_dict[key] = torch.zeros(batch_size, dtype=torch.bool)

        return observed_dict, target_dict
