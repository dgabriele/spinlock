"""Masking pattern generators for trajectory completion."""

from typing import Dict, List

import torch


class MaskingStrategy:
    """Generate different masking patterns for trajectory completion."""

    @staticmethod
    def temporal_mask(
        num_tokens: int,
        mask_ratio: float = 0.5,
        keep_edges: bool = True
    ) -> torch.BoolTensor:
        """
        Mask middle portion of trajectory (temporal dimension).

        Args:
            num_tokens: Total number of tokens
            mask_ratio: Fraction to mask (default: 0.5 = middle 50%)
            keep_edges: If True, keep first/last 25% (mask middle 50%)

        Returns:
            Boolean mask [num_tokens] (True = observed, False = masked)
        """
        mask = torch.ones(num_tokens, dtype=torch.bool)

        if keep_edges:
            # Mask middle 50%
            start_idx = int(num_tokens * 0.25)
            end_idx = int(num_tokens * 0.75)
            mask[start_idx:end_idx] = False
        else:
            # Mask random consecutive segment
            mask_len = int(num_tokens * mask_ratio)
            start_idx = torch.randint(0, num_tokens - mask_len, (1,)).item()
            mask[start_idx:start_idx + mask_len] = False

        return mask

    @staticmethod
    def random_mask(
        num_tokens: int,
        mask_ratio: float = 0.5
    ) -> torch.BoolTensor:
        """
        Random sparse masking.

        Args:
            num_tokens: Total number of tokens
            mask_ratio: Fraction to mask

        Returns:
            Boolean mask [num_tokens] (True = observed, False = masked)
        """
        mask = torch.rand(num_tokens) > mask_ratio
        return mask

    @staticmethod
    def keyframe_mask(
        num_tokens: int,
        keyframe_interval: int = 5
    ) -> torch.BoolTensor:
        """
        Keep only keyframes (e.g., every 5th token).

        Args:
            num_tokens: Total number of tokens
            keyframe_interval: Spacing between keyframes

        Returns:
            Boolean mask [num_tokens] (True = observed, False = masked)
        """
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        mask[::keyframe_interval] = True
        return mask

    @staticmethod
    def create_token_mask_dict(
        token_dict: Dict[str, torch.Tensor],
        strategy: str = "temporal",
        **kwargs
    ) -> Dict[str, torch.BoolTensor]:
        """
        Create masking pattern for all category-levels in token dict.

        For scalar tokens (most common), we randomly mask a fraction of category-levels.
        For sequence tokens, we mask temporal positions.

        Args:
            token_dict: Dictionary of tokens from VQTokenizer
            strategy: "temporal", "random", or "keyframe"
            **kwargs: Additional arguments (mask_ratio, etc.)

        Returns:
            Dict mapping category-level → boolean mask (True = observed, False = masked)
        """
        mask_dict = {}
        mask_ratio = kwargs.get('mask_ratio', 0.5)

        # Separate scalar and sequence tokens
        scalar_keys = []
        sequence_keys = []

        for key, tokens in token_dict.items():
            if tokens.ndim == 1 or (tokens.ndim == 2 and tokens.shape[1] == 1):
                scalar_keys.append(key)
            else:
                sequence_keys.append(key)

        # For scalar tokens: randomly mask some category-levels
        if scalar_keys:
            num_to_mask = int(len(scalar_keys) * mask_ratio)
            # Random permutation to select which to mask
            perm = torch.randperm(len(scalar_keys))
            masked_indices = set(perm[:num_to_mask].tolist())

            for i, key in enumerate(scalar_keys):
                # True = observed, False = masked
                mask_dict[key] = torch.tensor([i not in masked_indices], dtype=torch.bool)

        # For sequence tokens: use temporal masking
        for key in sequence_keys:
            num_tokens = token_dict[key].shape[1]
            if strategy == "temporal":
                mask_dict[key] = MaskingStrategy.temporal_mask(num_tokens, **kwargs)
            elif strategy == "random":
                mask_dict[key] = MaskingStrategy.random_mask(num_tokens, **kwargs)
            elif strategy == "keyframe":
                mask_dict[key] = MaskingStrategy.keyframe_mask(num_tokens, **kwargs)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return mask_dict
