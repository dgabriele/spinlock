"""Generate masks for trajectory completion experiments."""

import torch
import numpy as np
from typing import Tuple
from enum import Enum


class MaskingStrategy(str, Enum):
    """Trajectory masking strategies."""
    START_END = "start_end"           # Keep start% + end%
    COARSE_ONLY = "coarse_only"       # Keep only coarse (L0) tokens
    RANDOM_WINDOWS = "random_windows" # Random temporal windows
    HIERARCHICAL = "hierarchical"     # Keep coarse, mask fine


class TemporalMaskGenerator:
    """Generate masks for trajectory completion experiments."""

    def __init__(
        self,
        strategy: MaskingStrategy,
        start_percent: float = 0.3,
        end_percent: float = 0.2,
        seed: int = 42
    ):
        self.strategy = strategy
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.rng = np.random.RandomState(seed)

    def generate_token_mask(
        self,
        num_categories: int,
        num_levels: int = 3
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        Generate mask for token sequence.

        Args:
            num_categories: Number of VQ-VAE categories (N)
            num_levels: Number of hierarchical levels (L)

        Returns:
            observed_mask: [N×L] bool tensor (True = observed, False = masked)
            target_mask: [N×L] bool tensor (True = predict, False = given)
        """
        total_tokens = num_categories * num_levels
        observed_mask = torch.zeros(total_tokens, dtype=torch.bool)

        if self.strategy == MaskingStrategy.START_END:
            # Keep start_percent + end_percent of tokens
            n_start = int(total_tokens * self.start_percent)
            n_end = int(total_tokens * self.end_percent)
            observed_mask[:n_start] = True
            observed_mask[-n_end:] = True

        elif self.strategy == MaskingStrategy.COARSE_ONLY:
            # Keep only level 0 (coarse) tokens for each category
            for cat_idx in range(num_categories):
                token_idx = cat_idx * num_levels  # First level of each category
                observed_mask[token_idx] = True

        elif self.strategy == MaskingStrategy.HIERARCHICAL:
            # Keep coarse (L0) + medium (L1), mask fine (L2)
            for cat_idx in range(num_categories):
                base_idx = cat_idx * num_levels
                observed_mask[base_idx:base_idx+2] = True  # L0, L1

        elif self.strategy == MaskingStrategy.RANDOM_WINDOWS:
            # Random contiguous windows
            window_size = int(total_tokens * (self.start_percent + self.end_percent))
            start_idx = self.rng.randint(0, total_tokens - window_size + 1)
            observed_mask[start_idx:start_idx+window_size] = True

        target_mask = ~observed_mask
        return observed_mask, target_mask

    def apply_temporal_mask(
        self,
        temporal_features: torch.Tensor,
        mask_percent: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask temporal features (per-timestep).

        Args:
            temporal_features: [batch, T, D_temporal]
            mask_percent: Fraction of timesteps to mask

        Returns:
            masked_features: Features with masked timesteps set to 0
            timestep_mask: [T] bool tensor (True = observed)
        """
        T = temporal_features.shape[1]
        n_masked = int(T * mask_percent)

        # Generate mask
        if self.strategy == MaskingStrategy.START_END:
            n_start = int(T * self.start_percent)
            n_end = int(T * self.end_percent)
            timestep_mask = torch.zeros(T, dtype=torch.bool)
            timestep_mask[:n_start] = True
            timestep_mask[-n_end:] = True
        else:
            # Random masking
            timestep_mask = torch.ones(T, dtype=torch.bool)
            masked_indices = self.rng.choice(T, n_masked, replace=False)
            timestep_mask[masked_indices] = False

        # Apply mask
        masked_features = temporal_features.clone()
        masked_features[:, ~timestep_mask, :] = 0

        return masked_features, timestep_mask
