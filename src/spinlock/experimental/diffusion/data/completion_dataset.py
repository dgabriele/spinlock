"""Dataset for diffusion-based token completion.

Loads features, tokenizes with VQTokenizer v2, and generates
masked examples for training/evaluation.
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from spinlock.data import SpinlockDataset
from spinlock.tokens.tokenizer import VQTokenizer

from .hierarchical_masking import HierarchicalMaskGenerator

logger = logging.getLogger(__name__)


class DiffusionCompletionDataset(Dataset):
    """Dataset for diffusion-based token completion.

    Loads trajectories, tokenizes them, and generates masked examples
    using HierarchicalMaskGenerator.

    Args:
        dataset_path: Path to SpinlockDataset HDF5 file
        tokenizer_checkpoint: Path to VQTokenizer v2 checkpoint
        mask_generator: HierarchicalMaskGenerator instance
        cache_tokens: Whether to cache tokenized data in memory
        device: Device for tokenization (cpu or cuda)

    Example:
        >>> dataset = DiffusionCompletionDataset(
        ...     dataset_path=Path("datasets/cno_100k.h5"),
        ...     tokenizer_checkpoint=Path("checkpoints/tokenizer_v2/best.pt"),
        ...     mask_generator=mask_gen
        ... )
        >>> batch = dataset[0]
        >>> # batch = {
        >>> #   'tokens': Dict[str, Tensor[B]],
        >>> #   'observed': Dict[str, BoolTensor[B]],
        >>> #   'target': Dict[str, BoolTensor[B]]
        >>> # }
    """

    def __init__(
        self,
        dataset_path: Path,
        tokenizer_checkpoint: Path,
        mask_generator: HierarchicalMaskGenerator,
        cache_tokens: bool = True,
        max_cache_size: Optional[int] = None,  # LRU cache size (None = cache all)
        device: str = "cpu",
    ):
        self.dataset_path = dataset_path
        self.mask_generator = mask_generator
        self.device = device
        self.cache_tokens = cache_tokens
        self.max_cache_size = max_cache_size

        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_checkpoint}")
        self.tokenizer = VQTokenizer.from_checkpoint(tokenizer_checkpoint)
        self.tokenizer.model.to(device)
        self.tokenizer.model.eval()

        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        self.dataset = SpinlockDataset.from_file(str(dataset_path))

        # Extract and tokenize features
        logger.info("Extracting and tokenizing features")
        self.features = self._extract_features()

        if cache_tokens and max_cache_size is None:
            # Full caching (all samples)
            logger.info("Caching all tokenized data")
            self.cached_tokens = self._tokenize_all()
            self.lru_cache = None
        elif cache_tokens and max_cache_size is not None:
            # LRU caching (limited number of samples)
            logger.info(f"Using LRU cache with max_size={max_cache_size}")
            self.cached_tokens = None
            self.lru_cache = OrderedDict()  # LRU cache: {idx: tokens_dict}
        else:
            # No caching
            logger.info("No caching - tokenizing on-the-fly")
            self.cached_tokens = None
            self.lru_cache = None

        logger.info(
            f"DiffusionCompletionDataset initialized: "
            f"N={len(self)}, cache_tokens={cache_tokens}"
        )

    def _extract_features(self) -> Dict[str, Optional[torch.Tensor]]:
        """Extract features from dataset.

        Returns:
            Dict with keys:
            - temporal: [N, T, D_t] or None
            - initial_manual: [N, D_i] or None
            - initial_raw: [N, C, H, W] or None
            - temporal_mask: [N, T] or None
            - temporal_lengths: [N] or None
        """
        features = {}

        with self.dataset.open():
            # Temporal features
            if hasattr(self.dataset.features, 'temporal'):
                temporal = self.dataset.features.temporal.load_all()
                features['temporal'] = torch.from_numpy(temporal).float()

                # Variable-length support (assume all valid for now)
                N, T, _ = temporal.shape
                features['temporal_mask'] = torch.ones(N, T, dtype=torch.bool)
                features['temporal_lengths'] = torch.full((N,), T, dtype=torch.long)
            else:
                features['temporal'] = None
                features['temporal_mask'] = None
                features['temporal_lengths'] = None

            # Initial manual features
            if hasattr(self.dataset.features, 'initial'):
                initial_manual = self.dataset.features.initial.load_all()
                features['initial_manual'] = torch.from_numpy(initial_manual).float()
            else:
                features['initial_manual'] = None

            # Initial raw features (ICs)
            if self.dataset.inputs is not None:
                ics = self.dataset.inputs.load_all()

                # Aggregate over realizations if needed
                if ics.ndim == 5:  # [N, M, C, H, W]
                    ics = ics.mean(axis=1)  # [N, C, H, W]
                elif ics.ndim == 3:  # [N, H, W]
                    ics = ics[:, None, :, :]  # [N, 1, H, W]

                features['initial_raw'] = torch.from_numpy(ics).float()
            else:
                features['initial_raw'] = None

        return features

    def _tokenize_all(self) -> Dict[str, torch.Tensor]:
        """Tokenize all features in dataset.

        Returns:
            Dict mapping "family_category_Ll" → token indices [N]
        """
        with torch.no_grad():
            # Move features to device
            temporal = self.features['temporal']
            initial_manual = self.features['initial_manual']
            initial_raw = self.features['initial_raw']
            temporal_mask = self.features['temporal_mask']
            temporal_lengths = self.features['temporal_lengths']

            if temporal is not None:
                temporal = temporal.to(self.device)
            if initial_manual is not None:
                initial_manual = initial_manual.to(self.device)
            if initial_raw is not None:
                initial_raw = initial_raw.to(self.device)
            if temporal_mask is not None:
                temporal_mask = temporal_mask.to(self.device)
            if temporal_lengths is not None:
                temporal_lengths = temporal_lengths.to(self.device)

            # Tokenize
            tokens_dict = self.tokenizer.tokenize(
                temporal_features=temporal,
                initial_manual=initial_manual,
                initial_raw=initial_raw,
                temporal_mask=temporal_mask,
                temporal_lengths=temporal_lengths,
            )

            # Move back to CPU if caching
            tokens_dict = {
                key: tokens.cpu() for key, tokens in tokens_dict.items()
            }

        return tokens_dict

    def __len__(self) -> int:
        """Number of samples in dataset."""
        if self.features['temporal'] is not None:
            return self.features['temporal'].shape[0]
        elif self.features['initial_manual'] is not None:
            return self.features['initial_manual'].shape[0]
        elif self.features['initial_raw'] is not None:
            return self.features['initial_raw'].shape[0]
        else:
            raise ValueError("No features loaded")

    def _tokenize_single(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenize a single sample.

        Args:
            idx: Sample index

        Returns:
            tokens_dict: Dict[str, Tensor] of token indices (without batch dim)
        """
        with torch.no_grad():
            temporal = self.features['temporal'][idx:idx+1] if self.features['temporal'] is not None else None
            initial_manual = self.features['initial_manual'][idx:idx+1] if self.features['initial_manual'] is not None else None
            initial_raw = self.features['initial_raw'][idx:idx+1] if self.features['initial_raw'] is not None else None
            temporal_mask = self.features['temporal_mask'][idx:idx+1] if self.features['temporal_mask'] is not None else None
            temporal_lengths = self.features['temporal_lengths'][idx:idx+1] if self.features['temporal_lengths'] is not None else None

            if temporal is not None:
                temporal = temporal.to(self.device)
            if initial_manual is not None:
                initial_manual = initial_manual.to(self.device)
            if initial_raw is not None:
                initial_raw = initial_raw.to(self.device)
            if temporal_mask is not None:
                temporal_mask = temporal_mask.to(self.device)
            if temporal_lengths is not None:
                temporal_lengths = temporal_lengths.to(self.device)

            tokens_dict = self.tokenizer.tokenize(
                temporal_features=temporal,
                initial_manual=initial_manual,
                initial_raw=initial_raw,
                temporal_mask=temporal_mask,
                temporal_lengths=temporal_lengths,
            )

            # Squeeze batch dimension and move to CPU
            tokens_dict = {
                key: tokens.squeeze(0).cpu() for key, tokens in tokens_dict.items()
            }

        return tokens_dict

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
        # Get tokens (with LRU caching if enabled)
        if self.cached_tokens is not None:
            # Use full cache
            tokens_dict = {
                key: tokens[idx] for key, tokens in self.cached_tokens.items()
            }
        elif self.lru_cache is not None:
            # Use LRU cache
            if idx in self.lru_cache:
                # Cache hit - move to end (most recently used)
                self.lru_cache.move_to_end(idx)
                tokens_dict = self.lru_cache[idx]
            else:
                # Cache miss - tokenize and add to cache
                tokens_dict = self._tokenize_single(idx)
                self.lru_cache[idx] = tokens_dict
                self.lru_cache.move_to_end(idx)

                # Evict oldest if cache is full
                if len(self.lru_cache) > self.max_cache_size:
                    self.lru_cache.popitem(last=False)  # Remove oldest (first) item
        else:
            # No caching - tokenize on-the-fly
            tokens_dict = self._tokenize_single(idx)

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


def collate_dict_batch(batch: list) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collate batch of dict examples.

    Args:
        batch: List of dicts from DiffusionCompletionDataset.__getitem__

    Returns:
        Dict with keys:
        - 'tokens': Dict[str, Tensor[B]] - batched tokens
        - 'observed': Dict[str, BoolTensor[B]] - batched observed masks
        - 'target': Dict[str, BoolTensor[B]] - batched target masks
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    # Get keys from first example
    token_keys = list(batch[0]['tokens'].keys())

    # Stack each key across batch
    tokens_batched = {}
    observed_batched = {}
    target_batched = {}

    for key in token_keys:
        # Stack tokens
        tokens_batched[key] = torch.stack([
            example['tokens'][key] for example in batch
        ])

        # Stack observed masks
        observed_batched[key] = torch.stack([
            example['observed'][key] for example in batch
        ])

        # Stack target masks
        target_batched[key] = torch.stack([
            example['target'][key] for example in batch
        ])

    result = {
        'tokens': tokens_batched,
        'observed': observed_batched,
        'target': target_batched,
    }

    # Collate auxiliary truncation tokens if present (for roundtrip loss)
    if 'aux_trunc_tokens' in batch[0]:
        aux_trunc_batched = {}
        for tl in batch[0]['aux_trunc_tokens']:
            trunc_keys = list(batch[0]['aux_trunc_tokens'][tl].keys())
            aux_trunc_batched[tl] = {
                key: torch.stack([
                    example['aux_trunc_tokens'][tl][key] for example in batch
                ])
                for key in trunc_keys
            }
        result['aux_trunc_tokens'] = aux_trunc_batched

    return result
