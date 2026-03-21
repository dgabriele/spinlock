"""Dataset for pre-tokenized diffusion training.

Loads pre-computed tokens from HDF5 for instant training without on-the-fly tokenization.
Uses the shared PretokenizedTokenStore for HDF5 loading and indexing.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from spinlock.tokens.pretokenized_store import PretokenizedTokenStore

from .hierarchical_masking import HierarchicalMaskGenerator

logger = logging.getLogger(__name__)


class PretokenizedDiffusionDataset(Dataset):
    """Dataset for diffusion training with pre-tokenized data.

    Loads tokens directly from pre-tokenized HDF5 file, eliminating the
    tokenization bottleneck during training. Delegates token loading to
    the shared PretokenizedTokenStore.

    Args:
        tokenized_dataset_path: Path to pre-tokenized HDF5 file
        mask_generator: HierarchicalMaskGenerator instance
        truncation_length: If set, select only this truncation from a
            multi-truncation HDF5 and remap keys to base form.
        aux_truncation_lengths: Optional list of additional truncation lengths
            to load for roundtrip consistency loss. Each creates a separate
            PretokenizedTokenStore filtered to that truncation.

    Example:
        >>> dataset = PretokenizedDiffusionDataset(
        ...     tokenized_dataset_path=Path("datasets/50k_baseline_tokenized.h5"),
        ...     mask_generator=mask_gen,
        ...     aux_truncation_lengths=[32, 64, 128, 256],
        ... )
        >>> batch = dataset[0]
        >>> # batch['aux_trunc_tokens'][32] = Dict[str, Tensor]  # tokens at T=32
    """

    def __init__(
        self,
        tokenized_dataset_path: Path,
        mask_generator: HierarchicalMaskGenerator,
        truncation_length: Optional[int] = None,
        token_filter=None,
        aux_truncation_lengths: Optional[list] = None,
    ):
        self.tokenized_dataset_path = tokenized_dataset_path
        self.mask_generator = mask_generator
        self.token_filter = token_filter

        # Load tokens via shared store
        self.store = PretokenizedTokenStore(
            tokenized_dataset_path, truncation_length=truncation_length
        )
        self.num_samples = self.store.num_samples

        # Auxiliary truncation stores for roundtrip loss
        self.aux_stores: Dict[int, PretokenizedTokenStore] = {}
        if aux_truncation_lengths:
            for tl in aux_truncation_lengths:
                self.aux_stores[tl] = PretokenizedTokenStore(
                    tokenized_dataset_path, truncation_length=tl
                )
            logger.info(
                f"Loaded {len(self.aux_stores)} auxiliary truncation stores: "
                f"T={sorted(self.aux_stores.keys())}"
            )

        logger.info(
            f"PretokenizedDiffusionDataset initialized: "
            f"N={self.num_samples}, category_levels={len(self.category_levels)}"
            + (f", entropy_filter={len(token_filter.active_keys)} active" if token_filter else "")
        )

    @property
    def category_levels(self):
        """Sorted quantizer keys (delegated to store)."""
        return self.store.keys

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get a single masked example.

        Args:
            idx: Sample index

        Returns:
            Dict with keys:
            - 'tokens': Dict[str, Tensor] - full tokens
            - 'observed': Dict[str, BoolTensor] - observed mask
            - 'target': Dict[str, BoolTensor] - target mask
        """
        # Get tokens for this sample
        tokens_dict = self.store.get_sample(idx)

        # Contract to active keys if entropy filter is set
        if self.token_filter is not None:
            tokens_dict = self.token_filter.contract(tokens_dict)

        # Generate mask (batch_size=1 for single sample)
        observed_dict, target_dict = self.mask_generator.generate_dict_mask(batch_size=1)

        # Squeeze batch dimension from masks
        observed_dict = {key: mask.squeeze(0) for key, mask in observed_dict.items()}
        target_dict = {key: mask.squeeze(0) for key, mask in target_dict.items()}

        result = {
            'tokens': tokens_dict,
            'observed': observed_dict,
            'target': target_dict,
        }

        # Include auxiliary truncation tokens (temporal-only, base-key form)
        if self.aux_stores:
            result['aux_trunc_tokens'] = {
                tl: store.get_sample(idx)
                for tl, store in self.aux_stores.items()
            }

        return result
