"""Compute per-position noise scale factors from cross-truncation token divergence.

One-time analysis script. Reads a multi-truncation pretokenized HDF5,
compares token agreement between shortest and longest truncation for
each position (base key), and outputs per-position scale factors as JSON.

Positions where tokens are stable across truncations (theta, IC, short-horizon
temporal) get low scale factors (resolve early during denoising). Positions
where tokens diverge (long-horizon temporal) get high scale factors (resolve
late).

Usage:
    python experiments/diffusion/scripts/compute_position_scales.py \
        --tokenized-path datasets/lenia_1.5M_temporal.h5 \
        --output experiments/diffusion/configs/position_scales_v6.json
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_TRUNC_RE = re.compile(r"^(.+)_trunc_T(\d+)_(L\d+)$")


def strip_trunc_suffix(key: str) -> str:
    """Strip _trunc_T{TTT} from a temporal-resolution key."""
    m = _TRUNC_RE.match(key)
    return f"{m.group(1)}_{m.group(3)}" if m else key


def compute_position_scales(
    tokenized_path: str,
    output_path: str,
    non_temporal_scale: float = 0.3,
    min_temporal_scale: float = 0.5,
    max_temporal_scale: float = 1.0,
) -> Dict[str, float]:
    """Compute per-position scale factors from cross-truncation divergence.

    For each base key that appears at multiple truncation lengths:
    1. Load tokens at the shortest and longest truncation
    2. Compute agreement rate = mean(tokens_short == tokens_long)
    3. divergence = 1 - agreement_rate
    4. Normalize to [min_temporal_scale, max_temporal_scale]

    Non-temporal keys get non_temporal_scale.

    Args:
        tokenized_path: Path to multi-truncation pretokenized HDF5.
        output_path: Path to write JSON output.
        non_temporal_scale: Scale for non-temporal keys (theta, initial).
        min_temporal_scale: Floor for temporal scale factors.
        max_temporal_scale: Ceiling for temporal scale factors.

    Returns:
        Dict mapping base key → scale factor.
    """
    logger.info(f"Opening {tokenized_path}")

    with h5py.File(tokenized_path, "r") as f:
        if "tokens" not in f:
            raise ValueError(f"No 'tokens' group in {tokenized_path}")

        token_group = f["tokens"]
        all_keys = list(token_group.keys())
        logger.info(f"Found {len(all_keys)} token keys")

        # Group keys by base key → {trunc_len: full_key}
        base_to_trunc: Dict[str, Dict[int, str]] = defaultdict(dict)
        non_trunc_keys = []

        for key in all_keys:
            m = _TRUNC_RE.match(key)
            if m:
                base = f"{m.group(1)}_{m.group(3)}"
                t_len = int(m.group(2))
                base_to_trunc[base][t_len] = key
            else:
                non_trunc_keys.append(key)

        logger.info(
            f"Temporal base keys: {len(base_to_trunc)}, "
            f"non-temporal keys: {len(non_trunc_keys)}"
        )

        if not base_to_trunc:
            logger.warning(
                "No truncation keys found. Writing non_temporal_scale for all keys."
            )
            result = {key: non_temporal_scale for key in all_keys}
            _write_json(result, output_path)
            return result

        # Compute divergence for each base key
        divergences: Dict[str, float] = {}

        for base, trunc_dict in base_to_trunc.items():
            sorted_truncs = sorted(trunc_dict.keys())
            if len(sorted_truncs) < 2:
                # Only one truncation → no divergence data, assume mid-range
                divergences[base] = 0.5
                continue

            t_min = sorted_truncs[0]
            t_max = sorted_truncs[-1]

            tokens_short = token_group[trunc_dict[t_min]][:]  # [N]
            tokens_long = token_group[trunc_dict[t_max]][:]  # [N]

            agreement = np.mean(tokens_short == tokens_long)
            divergences[base] = 1.0 - agreement

        # Log divergence statistics
        divs = np.array(list(divergences.values()))
        logger.info(
            f"Divergence stats: min={divs.min():.4f}, max={divs.max():.4f}, "
            f"mean={divs.mean():.4f}, median={np.median(divs):.4f}"
        )

        # Normalize divergences to [min_temporal_scale, max_temporal_scale]
        d_min, d_max = divs.min(), divs.max()
        scale_range = max_temporal_scale - min_temporal_scale

        result: Dict[str, float] = {}

        for base, div in divergences.items():
            if d_max > d_min:
                normalized = (div - d_min) / (d_max - d_min)
            else:
                normalized = 0.5  # all same divergence
            scale = min_temporal_scale + normalized * scale_range
            result[base] = round(scale, 4)

        # Non-temporal keys
        for key in non_trunc_keys:
            result[key] = non_temporal_scale

        # Log scale factor distribution
        temporal_scales = [v for k, v in result.items() if k not in non_trunc_keys]
        if temporal_scales:
            ts = np.array(temporal_scales)
            logger.info(
                f"Temporal scale factors: min={ts.min():.4f}, max={ts.max():.4f}, "
                f"mean={ts.mean():.4f}"
            )

    _write_json(result, output_path)
    return result


def _write_json(data: Dict[str, float], path: str):
    """Write scale factors to JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    logger.info(f"Wrote {len(data)} scale factors to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-position noise scale factors from cross-truncation token divergence"
    )
    parser.add_argument(
        "--tokenized-path",
        type=str,
        required=True,
        help="Path to multi-truncation pretokenized HDF5",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write JSON output",
    )
    parser.add_argument(
        "--non-temporal-scale",
        type=float,
        default=0.3,
        help="Scale factor for non-temporal keys (default: 0.3)",
    )
    parser.add_argument(
        "--min-temporal-scale",
        type=float,
        default=0.5,
        help="Floor for temporal scale factors (default: 0.5)",
    )
    parser.add_argument(
        "--max-temporal-scale",
        type=float,
        default=1.0,
        help="Ceiling for temporal scale factors (default: 1.0)",
    )

    args = parser.parse_args()
    compute_position_scales(
        tokenized_path=args.tokenized_path,
        output_path=args.output,
        non_temporal_scale=args.non_temporal_scale,
        min_temporal_scale=args.min_temporal_scale,
        max_temporal_scale=args.max_temporal_scale,
    )


if __name__ == "__main__":
    main()
