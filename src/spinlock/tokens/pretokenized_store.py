"""Shared pretokenized token store for indexed access to HDF5 token datasets.

Provides a simple in-memory token store that loads all tokens from an HDF5 file
and supports fast batch/sample indexing. Used by:
- PretokenizedDiffusionDataset (D3PM training)
- RoundtripConsistencyLoss (MNO roundtrip loss, Mode A)

HDF5 format: /tokens/{quantizer_key} -> [N] int32
Keys follow the convention: "{family}_{category}_L{level}"

When the HDF5 contains multi-truncation keys (e.g. temporal_group_0_trunc_T256_L0),
pass ``truncation_length`` to filter to a single resolution and remap to base keys
(e.g. temporal_group_0_L0) that downstream consumers expect.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import torch

from spinlock.tokens.schema import _TRUNC_RE, strip_trunc_suffix

logger = logging.getLogger(__name__)


class PretokenizedTokenStore:
    """Load pretokenized tokens from HDF5 and provide indexed access.

    All tokens are loaded into CPU memory for fast batch indexing.
    Typical memory footprint: ~50MB for 50K samples x 30 quantizers x 4 bytes.

    Args:
        path: Path to HDF5 file with /tokens group
        device: Device for returned tensors (default: "cpu")
        truncation_length: If set, only load keys matching this truncation
            length (e.g. 256 → ``_trunc_T256_``) and remap them to base keys
            (strip the truncation suffix). Keys without a truncation suffix
            are always included.
    """

    def __init__(
        self,
        path: Path,
        device: str = "cpu",
        truncation_length: Optional[int] = None,
    ):
        self.tokens: Dict[str, torch.Tensor] = {}
        self.keys: List[str] = []
        self.num_samples: int = 0
        self._device = device
        self._truncation_length = truncation_length
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        """Read /tokens/ group from HDF5, converting each key to long tensor.

        When ``truncation_length`` is set, only keys matching
        ``_trunc_T{truncation_length:03d}_`` are loaded (plus any non-truncated
        keys). Matched keys are remapped to their base form via
        ``strip_trunc_suffix()``.
        """
        trunc_tag = (
            f"_trunc_T{self._truncation_length:03d}_"
            if self._truncation_length is not None
            else None
        )

        with h5py.File(path, "r") as f:
            tokens_group = f["tokens"]
            raw_keys = sorted(tokens_group.keys())

            for key in raw_keys:
                # Filtering: when truncation_length is set, skip keys that
                # have a truncation suffix for a *different* length.
                if trunc_tag is not None and _TRUNC_RE.match(key):
                    if trunc_tag not in key:
                        continue

                # Remap truncation-suffixed keys to base form
                base_key = strip_trunc_suffix(key)
                self.tokens[base_key] = torch.from_numpy(
                    tokens_group[key][:]
                ).long()

            self.keys = sorted(self.tokens.keys())
            self.num_samples = self.tokens[self.keys[0]].shape[0]

        logger.info(
            "PretokenizedTokenStore: %d samples, %d quantizer keys from %s%s",
            self.num_samples,
            len(self.keys),
            path,
            f" (truncation_length={self._truncation_length})"
            if self._truncation_length is not None
            else "",
        )

        self._build_indicators()

    def get_batch(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Index tokens for a batch.

        Args:
            indices: [B] integer indices into the token arrays

        Returns:
            Dict mapping quantizer key -> [B] long tensor
        """
        return {key: self.tokens[key][indices] for key in self.keys}

    def get_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Single sample access.

        Args:
            idx: Sample index

        Returns:
            Dict mapping quantizer key -> scalar long tensor
        """
        return {key: self.tokens[key][idx] for key in self.keys}

    def _build_indicators(self) -> None:
        """Pre-compute binary bag-of-tokens indicator vectors [N, K*max_code].

        Each sample gets a binary vector where position ``k * max_code + code``
        is 1 if temporal quantizer k assigned that code. Used for Jaccard
        similarity computation in SoftTokenContrastiveLoss.
        """
        temporal_keys = sorted(k for k in self.keys if k.startswith("temporal_"))
        if not temporal_keys:
            self._indicators = None
            self._indicator_keys: list = []
            self._max_code = 0
            return

        max_code = max(self.tokens[k].max().item() + 1 for k in temporal_keys)
        N = self.num_samples
        K = len(temporal_keys)

        indicators = torch.zeros(N, K * max_code, dtype=torch.bool)
        for ki, key in enumerate(temporal_keys):
            codes = self.tokens[key]  # [N]
            flat_idx = ki * max_code + codes  # [N]
            indicators[torch.arange(N), flat_idx] = True

        self._indicators = indicators
        self._indicator_keys = temporal_keys
        self._max_code = max_code
        logger.info(
            "  Indicator matrix: [%d, %d] (%d temporal keys × %d max codes, %.1fMB)",
            N, indicators.shape[1], K, max_code,
            indicators.numel() / 8 / 1e6,
        )

    def get_indicators(self, indices: torch.Tensor) -> torch.Tensor:
        """Return binary indicator vectors for a batch.

        Args:
            indices: [B] integer indices into the token arrays.

        Returns:
            [B, indicator_dim] bool tensor of binary indicators.

        Raises:
            RuntimeError: If no temporal keys were found during loading.
        """
        if self._indicators is None:
            raise RuntimeError(
                "No indicator vectors available — token store has no temporal keys"
            )
        return self._indicators[indices]

    @property
    def indicator_dim(self) -> int:
        """Dimension of indicator vectors (n_temporal_keys * max_code)."""
        if self._indicators is None:
            return 0
        return self._indicators.shape[1]

    @property
    def schema(self) -> "TokenSchema":
        """Infer TokenSchema from loaded tokens (max+1 per key for vocab size)."""
        from spinlock.tokens.schema import TokenSchema

        return TokenSchema.from_pretokenized_store(self)
