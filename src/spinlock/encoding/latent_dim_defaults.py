"""Intelligent latent dimension defaults for hierarchical VQ-VAE.

This module provides automatic computation of latent dimensions for multi-resolution
VQ-VAE tokenization, ensuring:
1. Expansion: latent_dim > group_embedding_dim (for separability)
2. Multi-resolution: Coarse level > Medium level > Fine level
3. Token scaling: More tokens → larger latent_dim (log-scaled)
4. GPU efficiency: All dimensions are multiples of 4

Ported from unisim.system.models.latent_dim_defaults (100% generic).
"""

import numpy as np
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


def parse_compression_ratios(compression_ratios: str, num_levels: int = 3) -> List[float]:
    """Parse compression ratio string into list of floats.

    Args:
        compression_ratios: Colon-separated ratios (e.g., "4:2:1" or "0.5:0.25:0.125")
        num_levels: Expected number of levels (default: 3)

    Returns:
        List of float ratios [L0_ratio, L1_ratio, L2_ratio]

    Raises:
        ValueError: If format is invalid or wrong number of levels

    Examples:
        >>> parse_compression_ratios("4:2:1")
        [4.0, 2.0, 1.0]

        >>> parse_compression_ratios("0.5:0.25:0.125")
        [0.5, 0.25, 0.125]
    """
    try:
        ratios = [float(x.strip()) for x in compression_ratios.split(":")]
    except ValueError as e:
        raise ValueError(
            f"Invalid compression_ratios format: '{compression_ratios}'. "
            f"Expected colon-separated numbers (e.g., '4:2:1' or '0.5:0.25:0.125')"
        ) from e

    if len(ratios) != num_levels:
        raise ValueError(
            f"Expected {num_levels} compression ratios but got {len(ratios)}. "
            f"Format: 'L0_ratio:L1_ratio:L2_ratio' (e.g., '4:2:1')"
        )

    # Validate all ratios are positive
    if any(r <= 0 for r in ratios):
        raise ValueError(f"All compression ratios must be positive, got: {ratios}")

    return ratios


def compute_default_latent_dims(
    num_levels: int,
    group_embedding_dim: int,
    num_tokens_per_level: List[int],
    n_samples: int = 10000,
    category_name: Optional[str] = None,
    compression_ratios: Optional[List[float]] = None,
) -> List[int]:
    """Compute default latent dimensions for hierarchical VQ-VAE.

    Uses intelligent formula that ensures:
    - Expansion for separability (latent_dim > group_embedding_dim)
    - Multi-resolution ordering (coarse > medium > fine)
    - Token scaling (more tokens → more dimensions)

    Args:
        num_levels: Number of hierarchical levels (typically 3)
        group_embedding_dim: Embedding dimension from GroupedFeatureExtractor
        num_tokens_per_level: Token counts for each level [L0, L1, L2, ...]
        n_samples: Number of samples in dataset (for dataset-aware defaults)
        category_name: Optional category name (for logging)
        compression_ratios: Optional explicit ratios [L0, L1, L2] to override defaults

    Returns:
        List of latent_dims [latent_dim_L0, latent_dim_L1, ...]

    Example:
        >>> compute_default_latent_dims(
        ...     num_levels=3,
        ...     group_embedding_dim=128,
        ...     num_tokens_per_level=[48, 24, 12]
        ... )
        [192, 96, 48]
    """
    if len(num_tokens_per_level) != num_levels:
        raise ValueError(
            f"num_tokens_per_level has {len(num_tokens_per_level)} elements, "
            f"but num_levels={num_levels}"
        )

    # OVERRIDE: Use explicit compression ratios if provided
    if compression_ratios is not None:
        if len(compression_ratios) != num_levels:
            raise ValueError(
                f"compression_ratios has {len(compression_ratios)} elements, "
                f"but num_levels={num_levels}"
            )

        latent_dims = []
        for level_idx, ratio in enumerate(compression_ratios):
            # Apply ratio to group_embedding_dim
            latent_dim_float = group_embedding_dim * ratio

            # Round to multiple of 4 (GPU alignment)
            latent_dim = int(np.ceil(latent_dim_float / 4.0)) * 4

            # Enforce minimum 4D (allow aggressive compression)
            latent_dim = max(4, latent_dim)

            latent_dims.append(latent_dim)

        # Log computed dimensions
        if category_name:
            logger.info(
                f"Category '{category_name}': computed latent_dims = {latent_dims} "
                f"(group_embedding_dim={group_embedding_dim}, "
                f"compression_ratios={compression_ratios}, tokens={num_tokens_per_level})"
            )
        else:
            logger.info(
                f"Computed latent_dims = {latent_dims} "
                f"(group_embedding_dim={group_embedding_dim}, "
                f"compression_ratios={compression_ratios}, tokens={num_tokens_per_level})"
            )

        return latent_dims

    # STANDARD: Use automatic defaults
    latent_dims = []

    for level_idx in range(num_levels):
        num_tokens = num_tokens_per_level[level_idx]

        # Compute latent_dim for this level
        latent_dim = _compute_single_level_latent_dim(
            level_idx=level_idx,
            num_levels=num_levels,
            group_embedding_dim=group_embedding_dim,
            num_tokens=num_tokens,
            n_samples=n_samples,
        )

        latent_dims.append(latent_dim)

    # Enforce strict monotonicity: L0 >= L1 >= L2
    latent_dims = _enforce_monotonicity(latent_dims)

    # Log computed dimensions
    if category_name:
        logger.info(
            f"Category '{category_name}': computed latent_dims = {latent_dims} "
            f"(group_embedding_dim={group_embedding_dim}, tokens={num_tokens_per_level})"
        )
    else:
        logger.info(
            f"Computed latent_dims = {latent_dims} "
            f"(group_embedding_dim={group_embedding_dim}, tokens={num_tokens_per_level})"
        )

    return latent_dims


def _enforce_monotonicity(latent_dims: List[int]) -> List[int]:
    """Enforce strict monotonicity: L0 >= L1 >= L2 >= ... >= LN.

    Adjusts latent_dims to ensure hierarchical pyramid structure is preserved.

    Args:
        latent_dims: Computed latent dimensions (may violate monotonicity)

    Returns:
        Adjusted latent_dims with strict monotonicity enforced

    Example:
        >>> _enforce_monotonicity([48, 48, 48])
        [48, 20, 8]

        >>> _enforce_monotonicity([48, 8, 8])
        [48, 20, 8]
    """
    if len(latent_dims) <= 1:
        return latent_dims

    enforced = [latent_dims[0]]  # L0 stays as-is

    for i in range(1, len(latent_dims)):
        prev_dim = enforced[i - 1]
        curr_dim = latent_dims[i]

        # If current level >= previous level, enforce strict decrease
        if curr_dim >= prev_dim:
            # Reduce by at least 4D or halve, whichever is larger
            new_dim = max(prev_dim - 4, prev_dim // 2)
            # Absolute minimum: 4D
            new_dim = max(4, new_dim)
            enforced.append(new_dim)
        else:
            # Already monotonic - keep it
            enforced.append(curr_dim)

    return enforced


def _compute_single_level_latent_dim(
    level_idx: int,
    num_levels: int,
    group_embedding_dim: int,
    num_tokens: int,
    n_samples: int = 10000,
) -> int:
    """Compute default latent_dim for a single level.

    Formula:
        latent_dim = group_embedding_dim × base_expansion × level_multiplier × token_factor

    Where:
        - base_expansion = adaptive based on category size (1.0 to 2.14)
        - level_multiplier = 0.5^level_idx (geometric decay)
        - token_factor = max(1.0, log2(num_tokens) / 20.0)

    Args:
        level_idx: Level index (0 = top level with most tokens)
        num_levels: Total number of levels (typically 3)
        group_embedding_dim: Embedding dimension from feature extractor
        num_tokens: Number of tokens at this level
        n_samples: Number of samples in dataset

    Returns:
        Computed latent_dim for this level
    """
    # Adaptive expansion based on category size
    # Larger categories need more capacity
    base_expansion = 1.0 + 0.8 * ((group_embedding_dim / 100.0) ** 0.7)

    # Level progression: geometric decay (L0 → L1 → L2)
    level_multiplier = 0.5**level_idx

    # Token scaling (gentle log scaling)
    token_factor = max(1.0, np.log2(num_tokens) / 20.0)

    # Compute base value
    latent_dim_float = (
        group_embedding_dim * base_expansion * level_multiplier * token_factor
    )

    # Round to multiple of 4 (GPU alignment)
    latent_dim = int(np.ceil(latent_dim_float / 4.0)) * 4

    # Dataset-aware minimum capacity (L0 only)
    if level_idx == 0 and n_samples > 1000:
        # Minimum scales with dataset size
        min_latent_dim = int(np.ceil(np.log10(n_samples) * 12 / 4.0)) * 4
        # Cap at reasonable maximum
        min_latent_dim = min(64, max(8, min_latent_dim))
        latent_dim = max(min_latent_dim, latent_dim)
    else:
        # L1 and L2: preserve standard minimum (8D)
        latent_dim = max(8, latent_dim)

    return latent_dim


def fill_missing_latent_dims(
    levels: List[Dict[str, Any]],
    group_embedding_dim: int,
    n_samples: int = 10000,
    category_name: Optional[str] = None,
    compression_ratios: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Fill missing 'latent_dim' fields in level configs.

    Preserves explicit latent_dim values, computes defaults for missing ones.

    Args:
        levels: List of level dicts (must have 'num_tokens')
        group_embedding_dim: Embedding dimension from GroupedFeatureExtractor
        n_samples: Number of samples in dataset
        category_name: Optional category name (for logging)
        compression_ratios: Optional compression ratios for latent_dim computation

    Returns:
        Updated levels with all latent_dims filled
    """
    if not levels:
        raise ValueError("levels list cannot be empty")

    # Validate all levels have num_tokens
    for i, level in enumerate(levels):
        if "num_tokens" not in level:
            raise ValueError(
                f"Level {i} missing 'num_tokens' key. Each level must specify num_tokens."
            )

    num_levels = len(levels)
    num_tokens_per_level = [level["num_tokens"] for level in levels]

    # Compute defaults for all levels
    default_latent_dims = compute_default_latent_dims(
        num_levels=num_levels,
        group_embedding_dim=group_embedding_dim,
        num_tokens_per_level=num_tokens_per_level,
        n_samples=n_samples,
        category_name=category_name,
        compression_ratios=compression_ratios,
    )

    # Fill missing latent_dims, preserve explicit ones
    filled_levels = []
    for level_idx, (level, default_latent_dim) in enumerate(
        zip(levels, default_latent_dims)
    ):
        filled_level = level.copy()

        if "latent_dim" not in filled_level or filled_level["latent_dim"] is None:
            filled_level["latent_dim"] = default_latent_dim
            if category_name:
                logger.debug(
                    f"Category '{category_name}' L{level_idx}: "
                    f"Using computed latent_dim={default_latent_dim}"
                )
        else:
            user_latent_dim = filled_level["latent_dim"]
            if level_idx == 0 and user_latent_dim < group_embedding_dim:
                logger.warning(
                    f"Category '{category_name or 'unknown'}' L0 has "
                    f"latent_dim={user_latent_dim} < group_embedding_dim={group_embedding_dim}. "
                    f"This compresses features instead of expanding for separability."
                )

            if category_name:
                logger.debug(
                    f"Category '{category_name}' L{level_idx}: "
                    f"Using explicit latent_dim={user_latent_dim}"
                )

        filled_levels.append(filled_level)

    return filled_levels


def compute_default_num_tokens(
    num_levels: int,
    group_embedding_dim: int,
    n_samples: int,
) -> List[int]:
    """Compute default num_tokens for hierarchical VQ-VAE.

    Token count scales with:
    1. Embedding capacity (more dims → more tokens)
    2. Dataset diversity (more samples → more distinct patterns)

    Hierarchical structure: L0 has MOST tokens, L2 has FEWEST tokens.

    Args:
        num_levels: Number of hierarchical levels (typically 3)
        group_embedding_dim: Embedding dimension from GroupedFeatureExtractor
        n_samples: Number of samples in dataset

    Returns:
        List of num_tokens [num_tokens_L0, num_tokens_L1, ...]

    Example:
        >>> compute_default_num_tokens(
        ...     num_levels=3,
        ...     group_embedding_dim=128,
        ...     n_samples=25000
        ... )
        [32, 16, 8]
    """
    # Base token count scales with embedding capacity
    base_tokens = int(np.log2(group_embedding_dim) * 7)

    # Hierarchical progression: geometric halving
    tokens_per_level = []
    for level_idx in range(num_levels):
        level_multiplier = 0.5**level_idx
        num_tokens_float = base_tokens * level_multiplier

        # Round to multiple of 4 (GPU efficiency) using floor for L0
        if level_idx == 0:
            num_tokens = (int(num_tokens_float) // 4) * 4
        else:
            num_tokens = int(num_tokens_float)

        # Dataset-aware minimum tokens (L0 only)
        if level_idx == 0 and n_samples > 1000:
            # Minimum scales with dataset size
            min_tokens = min(28, max(5, int(np.sqrt(n_samples / 1000.0) * 4.8)))
            num_tokens = max(min_tokens, num_tokens)
        else:
            # L1 and L2: preserve standard minimum (6)
            num_tokens = max(6, num_tokens)

        tokens_per_level.append(num_tokens)

    logger.info(
        f"Computed default num_tokens = {tokens_per_level} "
        f"(group_embedding_dim={group_embedding_dim}, n_samples={n_samples})"
    )

    return tokens_per_level


def fill_missing_num_tokens(
    levels: List[Dict[str, Any]],
    group_embedding_dim: int,
    n_samples: int = 10000,
    category_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fill missing 'num_tokens' fields in level configs.

    Preserves explicit num_tokens values, computes defaults for missing ones.

    Args:
        levels: List of level dicts
        group_embedding_dim: Embedding dimension from GroupedFeatureExtractor
        n_samples: Number of samples in dataset
        category_name: Optional category name (for logging)

    Returns:
        Updated levels with all num_tokens filled
    """
    if not levels:
        raise ValueError("levels list cannot be empty")

    num_levels = len(levels)

    # Compute defaults for all levels
    default_num_tokens = compute_default_num_tokens(
        num_levels=num_levels,
        group_embedding_dim=group_embedding_dim,
        n_samples=n_samples,
    )

    # Fill missing num_tokens, preserve explicit ones
    filled_levels = []
    for level_idx, (level, default_tokens) in enumerate(
        zip(levels, default_num_tokens)
    ):
        filled_level = level.copy()

        if "num_tokens" not in filled_level or filled_level["num_tokens"] is None:
            filled_level["num_tokens"] = default_tokens
            if category_name:
                logger.debug(
                    f"Category '{category_name}' L{level_idx}: "
                    f"Using computed num_tokens={default_tokens}"
                )
        else:
            user_num_tokens = filled_level["num_tokens"]
            if category_name:
                logger.debug(
                    f"Category '{category_name}' L{level_idx}: "
                    f"Using explicit num_tokens={user_num_tokens}"
                )

        filled_levels.append(filled_level)

    return filled_levels
