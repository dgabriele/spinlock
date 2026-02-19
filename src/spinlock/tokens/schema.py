"""Runtime-discovered token schema from VQTokenizer or pretokenized dataset.

Encapsulates vocab_sizes, category-level info, and family discovery.
Reusable by diffusion training, token synthesis, and any future consumer.

This extracts logic that was previously duplicated in:
- experiments/diffusion/scripts/train.py (extract_vocab_sizes_and_info)
- experiments/diffusion/scripts/train.py (extract_vocab_sizes_from_pretokenized)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Matches keys with a temporal-resolution truncation suffix.
# e.g. "temporal_group_1_trunc_T032_L0" → groups: ("temporal_group_1", "032", "L0")
_TRUNC_RE = re.compile(r'^(.+)_trunc_T(\d+)_(L\d+)$')


def strip_trunc_suffix(key: str) -> str:
    """Strip _trunc_T{TTT} from a temporal-resolution key.

    "temporal_group_1_trunc_T032_L0" → "temporal_group_1_L0"
    "temporal_group_1_L0"            → "temporal_group_1_L0"  (no-op)
    """
    m = _TRUNC_RE.match(key)
    return f"{m.group(1)}_{m.group(3)}" if m else key


@dataclass
class CategoryLevelInfo:
    """Parsed metadata for a single quantizer key."""
    family: str                    # "temporal", "initial", "theta", etc.
    category: str                  # "group_1", "group_2", etc. — always WITHOUT trunc suffix
    level: int                     # 0, 1, 2
    trunc_len: Optional[int] = None  # None for base keys; int (e.g. 32) for trunc keys

    @property
    def base_key(self) -> str:
        """Canonical base key, stripping any truncation suffix."""
        return f"{self.family}_{self.category}_L{self.level}"


class TokenSchema:
    """Runtime-discovered token schema from VQTokenizer or pretokenized dataset.

    Encapsulates vocab_sizes, category-level info, and family discovery.
    Reusable by diffusion training, token synthesis, and any future consumer.

    All families and keys are discovered dynamically — no hardcoded assumptions
    about what families exist (temporal, initial, theta, etc.).

    Example:
        >>> schema = TokenSchema.from_pretokenized("datasets/tokens_50k.h5")
        >>> schema.families   # ['initial', 'temporal', 'theta']
        >>> schema.vocab_sizes  # {'temporal_group_1_L0': 28, ...}
        >>> schema.keys_for_family('theta')  # ['theta_group_1_L0', ...]
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        category_level_info: Dict[str, CategoryLevelInfo],
    ):
        self.vocab_sizes = vocab_sizes
        self.category_level_info = category_level_info
        self.families = sorted({info.family for info in category_level_info.values()})
        self.keys = sorted(vocab_sizes.keys())

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> 'TokenSchema':
        """Discover schema from loaded VQTokenizer model's quantizers.

        Args:
            tokenizer: VQTokenizer instance (with model loaded)

        Returns:
            TokenSchema with vocab sizes and category-level info
        """
        if tokenizer.model is None:
            raise ValueError("Tokenizer model not loaded. Use from_checkpoint() first.")

        vocab_sizes = {}
        category_level_info = {}

        for quantizer_key, quantizer in tokenizer.model.quantizers.items():
            vocab_size = quantizer.embedding.weight.shape[0]
            vocab_sizes[quantizer_key] = vocab_size
            category_level_info[quantizer_key] = cls.parse_key(quantizer_key)

        logger.info(
            f"TokenSchema from tokenizer: {len(vocab_sizes)} category-levels, "
            f"sizes={sorted(set(vocab_sizes.values()))}"
        )

        return cls(vocab_sizes, category_level_info)

    @classmethod
    def from_pretokenized(cls, tokenized_path: Path) -> 'TokenSchema':
        """Discover schema from pretokenized HDF5 dataset.

        Infers vocab_size from max token value + 1 per key.

        Args:
            tokenized_path: Path to pretokenized HDF5 file with /tokens group

        Returns:
            TokenSchema with vocab sizes and category-level info
        """
        import h5py

        vocab_sizes = {}
        category_level_info = {}

        with h5py.File(tokenized_path, 'r') as f:
            tokens_group = f['tokens']

            for key in tokens_group.keys():
                tokens = tokens_group[key][:]
                vocab_size = int(tokens.max()) + 1
                vocab_sizes[key] = vocab_size
                category_level_info[key] = cls.parse_key(key)

        logger.info(
            f"TokenSchema from pretokenized: {len(vocab_sizes)} category-levels, "
            f"sizes={sorted(set(vocab_sizes.values()))}"
        )

        return cls(vocab_sizes, category_level_info)

    @classmethod
    def from_pretokenized_store(cls, store: "PretokenizedTokenStore") -> 'TokenSchema':
        """Infer schema from a loaded PretokenizedTokenStore.

        Vocab size = max(token_values) + 1 per key.

        Args:
            store: Loaded PretokenizedTokenStore instance

        Returns:
            TokenSchema with inferred vocab sizes
        """
        vocab_sizes = {}
        category_level_info = {}

        for key in store.keys:
            vocab_sizes[key] = int(store.tokens[key].max().item()) + 1
            category_level_info[key] = cls.parse_key(key)

        return cls(vocab_sizes, category_level_info)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> 'TokenSchema':
        """Load schema from diffusion checkpoint metadata.

        Diffusion checkpoints store vocab_sizes and category_level_info
        in their metadata for reproducibility.

        Args:
            checkpoint_path: Path to diffusion checkpoint

        Returns:
            TokenSchema reconstructed from checkpoint metadata
        """
        import torch

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Try metadata first (diffusion checkpoints)
        if 'vocab_sizes' in checkpoint:
            vocab_sizes = checkpoint['vocab_sizes']
            raw_info = checkpoint.get('category_level_info', {})
            category_level_info = {}
            for key, info in raw_info.items():
                if isinstance(info, CategoryLevelInfo):
                    category_level_info[key] = info
                else:
                    category_level_info[key] = CategoryLevelInfo(
                        family=info['family'],
                        category=info['category'],
                        level=info['level'],
                        trunc_len=info.get('trunc_len'),
                    )
            return cls(vocab_sizes, category_level_info)

        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain vocab_sizes metadata. "
            f"Keys found: {list(checkpoint.keys())}"
        )

    def keys_for_family(self, family: str) -> List[str]:
        """Get all quantizer keys belonging to a family.

        Args:
            family: Family name (e.g., 'temporal', 'theta', 'initial')

        Returns:
            Sorted list of keys for that family
        """
        return sorted(
            k for k, info in self.category_level_info.items()
            if info.family == family
        )

    def unique_categories(self) -> List[Tuple[str, str]]:
        """Unique (family, category) tuples across all keys.

        Returns:
            Sorted list of (family, category) pairs
        """
        cats = {(info.family, info.category) for info in self.category_level_info.values()}
        return sorted(cats)

    def keys_for_category(self, family: str, category: str) -> List[str]:
        """All keys for a (family, category) pair across all levels.

        Args:
            family: Family name (e.g., 'temporal')
            category: Category name (e.g., 'group_1')

        Returns:
            Sorted list of keys matching this family+category
        """
        return sorted(
            k for k, info in self.category_level_info.items()
            if info.family == family and info.category == category
        )

    def vocab_sizes_dict(self) -> Dict[str, int]:
        """Dict format for DiscreteD3PM constructor."""
        return dict(self.vocab_sizes)

    def category_level_info_dict(self) -> Dict[str, Dict[str, Any]]:
        """Dict format for DenoisingNetwork constructor.

        Returns dict of {key: {'family': ..., 'category': ..., 'level': ...}}
        matching the format expected by existing diffusion code.
        """
        return {
            key: {
                'family': info.family,
                'category': info.category,
                'level': info.level,
            }
            for key, info in self.category_level_info.items()
        }

    @staticmethod
    def parse_key(key: str) -> CategoryLevelInfo:
        """Parse token key → CategoryLevelInfo.

        Handles both formats:
            "temporal_group_1_L0"             → category="group_1", trunc_len=None
            "temporal_group_1_trunc_T032_L0"  → category="group_1", trunc_len=32
            "initial_manual_features_L2"      → category="manual_features", trunc_len=None
            "theta_group_1_L0"                → family="theta", category="group_1"
        """
        m = _TRUNC_RE.match(key)
        if m:
            family_cat = m.group(1)          # "temporal_group_1"
            trunc_len = int(m.group(2))
            level = int(m.group(3)[1:])      # "L0" → 0
        else:
            # Base format: split on rightmost "_L"
            head, level_str = key.rsplit('_L', 1)
            family_cat = head
            level = int(level_str)
            trunc_len = None

        family, category = family_cat.split('_', 1)
        return CategoryLevelInfo(family=family, category=category,
                                 level=level, trunc_len=trunc_len)

    def __repr__(self) -> str:
        return (
            f"TokenSchema(families={self.families}, "
            f"num_keys={len(self.keys)}, "
            f"vocab_sizes={sorted(set(self.vocab_sizes.values()))})"
        )


# Module-level alias so callers can do `from spinlock.tokens.schema import parse_key`
parse_key = TokenSchema.parse_key
