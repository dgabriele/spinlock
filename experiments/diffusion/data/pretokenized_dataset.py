"""Dataset for pre-tokenized diffusion training.

Loads pre-computed tokens from HDF5 for instant training without on-the-fly tokenization.
"""

import logging
from pathlib import Path
from typing import Dict

import h5py
import torch
from torch.utils.data import Dataset

from .hierarchical_masking import HierarchicalMaskGenerator

logger = logging.getLogger(__name__)


class PretokenizedDiffusionDataset(Dataset):
    """Dataset for diffusion training with pre-tokenized data.

    Loads tokens directly from pre-tokenized HDF5 file, eliminating the
    tokenization bottleneck during training.

    Args:
        tokenized_dataset_path: Path to pre-tokenized HDF5 file
        mask_generator: HierarchicalMaskGenerator instance

    Example:
        >>> dataset = PretokenizedDiffusionDataset(
        ...     tokenized_dataset_path=Path("datasets/50k_baseline_tokenized.h5"),
        ...     mask_generator=mask_gen
        ... )
        >>> batch = dataset[0]
        >>> # batch = {
        >>> #   'tokens': Dict[str, Tensor],
        >>> #   'observed': Dict[str, BoolTensor],
        >>> #   'target': Dict[str, BoolTensor]
        >>> # }
    """

    def __init__(
        self,
        tokenized_dataset_path: Path,
        mask_generator: HierarchicalMaskGenerator,
    ):
        self.tokenized_dataset_path = tokenized_dataset_path
        self.mask_generator = mask_generator

        # Load tokens
        logger.info(f"Loading pre-tokenized dataset from {tokenized_dataset_path}")
        self._load_tokens()

        logger.info(
            f"PretokenizedDiffusionDataset initialized: "
            f"N={self.num_samples}, category_levels={len(self.category_levels)}"
        )

    def _load_tokens(self):
        """Load all tokens into memory."""
        with h5py.File(self.tokenized_dataset_path, 'r') as f:
            tokens_group = f['tokens']

            # Get category-levels
            self.category_levels = sorted(tokens_group.keys())

            # Load all tokens into tensors
            self.tokens = {}
            for key in self.category_levels:
                self.tokens[key] = torch.from_numpy(tokens_group[key][:]).long()

            # Get num samples
            self.num_samples = self.tokens[self.category_levels[0]].shape[0]

        logger.info(f"✓ Loaded {self.num_samples:,} pre-tokenized samples")
        logger.info(f"  Category-levels: {len(self.category_levels)}")

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
        tokens_dict = {key: self.tokens[key][idx] for key in self.category_levels}

        # Generate mask (batch_size=1 for single sample)
        observed_dict, target_dict = self.mask_generator.generate_dict_mask(batch_size=1)

        # Squeeze batch dimension from masks
        observed_dict = {key: mask.squeeze(0) for key, mask in observed_dict.items()}
        target_dict = {key: mask.squeeze(0) for key, mask in target_dict.items()}

        return {
            'tokens': tokens_dict,
            'observed': observed_dict,
            'target': target_dict,
        }
