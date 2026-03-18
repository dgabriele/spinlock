"""TokenFilter: entropy-based filtering of low-information token positions.

Computes per-position entropy from pretokenized data, applies Otsu's method
to find the optimal threshold separating constant/near-constant positions
from informative ones, and provides expand/contract mappings between the
full token dict and the active-only subset.

The D3PM trains only on active positions (higher entropy), reducing wasted
capacity on tokens that carry no information. At inference, frozen positions
are filled with their mode values.
"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


class TokenFilter:
    """Filter low-entropy token positions from D3PM training.

    Args:
        active_keys: Keys to keep (high entropy).
        frozen_modes: Dict mapping frozen key → mode token value.
        threshold: Entropy threshold used (for logging/serialization).

    Example:
        >>> tf = TokenFilter.from_pretokenized("tokens.h5", truncation_length=512)
        >>> small_dict = tf.contract(full_tokens_dict)      # 160 → 102 keys
        >>> full_dict  = tf.expand(small_dict)               # 102 → 160 keys
    """

    def __init__(
        self,
        active_keys: List[str],
        frozen_modes: Dict[str, int],
        threshold: float,
    ):
        self.active_keys = sorted(active_keys)
        self.frozen_modes = frozen_modes
        self.threshold = threshold
        self._active_set = set(active_keys)

    @classmethod
    def from_pretokenized(
        cls,
        tokenized_path: str,
        truncation_length: Optional[int] = 512,
        threshold: Optional[float] = None,
    ) -> "TokenFilter":
        """Build filter from pretokenized HDF5 using Otsu's method.

        Args:
            tokenized_path: Path to pretokenized HDF5.
            truncation_length: If set, select this T from multi-trunc keys.
            threshold: Manual override. None = auto (Otsu).

        Returns:
            TokenFilter instance.
        """
        with h5py.File(tokenized_path, "r") as f:
            all_keys = sorted(f["tokens"].keys())

            # Select truncation-specific keys
            keys = _select_truncation_keys(all_keys, truncation_length)

            # Compute per-key entropy and mode
            key_stats = {}
            for key in keys:
                vals = f["tokens"][key][:]
                counts = Counter(vals.tolist())
                total = len(vals)
                probs = np.array([c / total for c in counts.values()])
                entropy = max(0.0, -np.sum(probs * np.log2(probs + 1e-15)))
                mode_val = counts.most_common(1)[0][0]
                key_stats[key] = {"entropy": entropy, "mode": int(mode_val)}

        # Determine threshold
        entropies = np.array([s["entropy"] for s in key_stats.values()])
        if threshold is None:
            threshold = _otsu_threshold(entropies)

        # Split
        active_keys = []
        frozen_modes = {}
        for key, stats in key_stats.items():
            if stats["entropy"] > threshold:
                active_keys.append(key)
            else:
                frozen_modes[key] = stats["mode"]

        logger.info(
            f"TokenFilter: Otsu threshold={threshold:.4f} bits, "
            f"{len(active_keys)} active, {len(frozen_modes)} frozen "
            f"(total {len(key_stats)})"
        )

        # Log per-family breakdown
        family_counts = {"active": {}, "frozen": {}}
        for key in active_keys:
            fam = _family_from_key(key)
            family_counts["active"][fam] = family_counts["active"].get(fam, 0) + 1
        for key in frozen_modes:
            fam = _family_from_key(key)
            family_counts["frozen"][fam] = family_counts["frozen"].get(fam, 0) + 1
        for fam in sorted(set(list(family_counts["active"]) + list(family_counts["frozen"]))):
            a = family_counts["active"].get(fam, 0)
            fr = family_counts["frozen"].get(fam, 0)
            logger.info(f"  {fam}: {a} active, {fr} frozen")

        return cls(active_keys, frozen_modes, threshold)

    def contract(
        self, tokens_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Remove frozen keys from a token dict.

        Args:
            tokens_dict: Full token dict (160 keys).

        Returns:
            Contracted dict with only active keys.
        """
        return {k: v for k, v in tokens_dict.items() if k in self._active_set}

    def expand(
        self,
        active_dict: Dict[str, torch.Tensor],
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Restore frozen keys with mode values.

        Args:
            active_dict: Dict with only active keys.
            batch_size: Batch size for frozen tensors. Auto-detected if None.
            device: Device for frozen tensors. Auto-detected if None.

        Returns:
            Full token dict with all keys.
        """
        if batch_size is None:
            first_val = next(iter(active_dict.values()))
            batch_size = first_val.shape[0]
        if device is None:
            first_val = next(iter(active_dict.values()))
            device = first_val.device

        full_dict = dict(active_dict)
        for key, mode_val in self.frozen_modes.items():
            full_dict[key] = torch.full(
                (batch_size,), mode_val, dtype=torch.long, device=device,
            )
        return full_dict

    def contract_masks(
        self,
        observed: Dict[str, torch.BoolTensor],
        target: Dict[str, torch.BoolTensor],
    ) -> Tuple[Dict[str, torch.BoolTensor], Dict[str, torch.BoolTensor]]:
        """Contract mask dicts to active keys only."""
        return (
            {k: v for k, v in observed.items() if k in self._active_set},
            {k: v for k, v in target.items() if k in self._active_set},
        )

    def filter_vocab_sizes(
        self, vocab_sizes: Dict[str, int]
    ) -> Dict[str, int]:
        """Return vocab sizes for active keys only."""
        return {k: v for k, v in vocab_sizes.items() if k in self._active_set}

    def filter_category_level_info(
        self, cat_info: Dict[str, dict]
    ) -> Dict[str, dict]:
        """Return category_level_info for active keys only."""
        return {k: v for k, v in cat_info.items() if k in self._active_set}

    def state_dict(self) -> dict:
        """Serialize for checkpoint storage."""
        return {
            "active_keys": self.active_keys,
            "frozen_modes": self.frozen_modes,
            "threshold": self.threshold,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "TokenFilter":
        """Restore from checkpoint."""
        return cls(
            active_keys=state["active_keys"],
            frozen_modes=state["frozen_modes"],
            threshold=state["threshold"],
        )


def _select_truncation_keys(
    all_keys: List[str], truncation_length: Optional[int]
) -> List[str]:
    """Select keys for a specific truncation length from multi-trunc HDF5."""
    if truncation_length is None:
        return all_keys

    trunc_tag = f"_trunc_T{truncation_length:03d}_"
    selected = []
    for k in all_keys:
        if "_trunc_T" in k:
            if trunc_tag in k:
                selected.append(k)
        else:
            selected.append(k)
    return selected


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's method: find threshold maximizing between-class variance."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    best_thresh = 0.0
    best_var = 0.0

    for i in range(1, n):
        lo = sorted_vals[:i]
        hi = sorted_vals[i:]
        w0 = len(lo) / n
        w1 = len(hi) / n
        var_between = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = (sorted_vals[i - 1] + sorted_vals[i]) / 2

    return best_thresh


def _family_from_key(key: str) -> str:
    """Extract family name from token key."""
    if key.startswith("temporal"):
        return "temporal"
    elif key.startswith("initial"):
        return "initial"
    elif key.startswith("theta"):
        return "theta"
    return "unknown"
