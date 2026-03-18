"""Finite Scalar Quantization (FSQ) — Mentzer et al., 2023.

Rounds each dimension to L uniformly-spaced levels in [-1, 1].
Implicit codebook = Cartesian product of per-dim levels.
100% utilization by construction — no dead codes, no commitment loss.

Interface matches VectorQuantizer for drop-in use in the quantizer loop:
    forward(x) → (quantized, one_hot_encodings, {"loss": 0})

References:
    Mentzer et al. "Finite Scalar Quantization: VQ-VAE Made Simple" (2023)
    https://arxiv.org/abs/2309.15505
"""

from math import prod
from typing import Dict, Tuple

import torch
import torch.nn as nn


class FiniteScalarQuantizer(nn.Module):
    """Finite Scalar Quantization — implicit codebook via per-dim rounding.

    Each input dimension is independently rounded to one of L uniformly-spaced
    levels in [-1, 1].  The implicit codebook is the Cartesian product of all
    per-dim level sets, giving ``prod(levels)`` total codes with guaranteed
    100% utilization (every code is reachable by construction).

    Args:
        levels: Number of quantization levels per dimension, e.g. [8, 8, 8].
            ``len(levels)`` determines the input/output dimensionality.
            ``prod(levels)`` determines the implicit codebook size.
    """

    def __init__(self, levels: list[int]):
        super().__init__()
        if not levels or any(l < 2 for l in levels):
            raise ValueError(f"Each level must be >= 2, got {levels}")

        self.levels = levels
        self.codebook_size = prod(levels)   # For D3PM vocab
        self.num_embeddings = self.codebook_size  # Alias for VQ compat
        self.embedding_dim = len(levels)    # FSQ operates in len(levels) dims

        # Register level counts as buffer for device movement
        self.register_buffer(
            "_levels", torch.tensor(levels, dtype=torch.long)
        )
        # Pre-compute basis for mixed-radix encoding (for values_to_indices)
        basis = torch.ones(len(levels), dtype=torch.long)
        for i in range(len(levels) - 2, -1, -1):
            basis[i] = basis[i + 1] * levels[i + 1]
        self.register_buffer("_basis", basis)

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Round each dim to nearest level using STE (straight-through estimator).

        Maps continuous values to the nearest of L uniformly-spaced levels in
        [-1, 1].  Levels for L=8 are: {-1, -5/7, -3/7, -1/7, 1/7, 3/7, 5/7, 1}.

        Gradient flows through via STE: grad(quantized) = grad(input).
        """
        # Bound input to [-1, 1] with tanh
        x = torch.tanh(x)

        # Per-dim quantization: scale to [0, L-1], round, scale back to [-1, 1]
        half_levels = (self._levels.float() - 1) / 2  # e.g. 3.5 for L=8

        # Map [-1, 1] → [0, L-1]
        scaled = (x + 1) * half_levels  # [0, L-1]
        rounded = scaled.round()         # discrete [0, L-1]
        # Map back to [-1, 1]
        quantized = rounded / half_levels - 1

        # STE: forward uses quantized, backward uses identity
        return x + (quantized - x).detach()

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize inputs via per-dim rounding.

        Interface matches VectorQuantizer.forward().

        Args:
            inputs: Continuous features [..., D] where D = len(levels).

        Returns:
            quantized: Quantized values [..., D] (same shape as input).
            encodings: One-hot encodings [..., codebook_size] (for perplexity).
            losses: {"loss": tensor(0)} — FSQ has no training loss.
        """
        assert inputs.shape[-1] == self.embedding_dim, (
            f"Expected last dim {self.embedding_dim}, got {inputs.shape[-1]}"
        )

        quantized = self._quantize(inputs)

        # Compute one-hot encodings for perplexity measurement
        indices = self.values_to_indices(quantized)
        flat_indices = indices.reshape(-1)
        encodings = torch.zeros(
            flat_indices.shape[0],
            self.codebook_size,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        encodings.scatter_(1, flat_indices.unsqueeze(1), 1.0)
        encodings = encodings.reshape(*indices.shape, self.codebook_size)

        # No loss — FSQ needs no commitment or codebook loss
        zero_loss = torch.tensor(0.0, device=inputs.device, requires_grad=False)

        return quantized, encodings, {"loss": zero_loss}

    def values_to_indices(self, quantized: torch.Tensor) -> torch.Tensor:
        """Convert quantized values to codebook indices via mixed-radix encoding.

        Args:
            quantized: Quantized values [..., D] in [-1, 1].

        Returns:
            Indices [...] as long tensor in [0, codebook_size).
        """
        half_levels = (self._levels.float() - 1) / 2
        # Map [-1, 1] → [0, L-1] integer
        per_dim = ((quantized + 1) * half_levels).round().long()
        per_dim = per_dim.clamp(min=0)
        for i, lev in enumerate(self.levels):
            per_dim[..., i] = per_dim[..., i].clamp(max=lev - 1)

        # Mixed-radix: index = sum(per_dim_i * basis_i)
        return (per_dim * self._basis).sum(dim=-1)

    def indices_to_values(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert codebook indices back to quantized values.

        Args:
            indices: Integer indices [...] in [0, codebook_size).

        Returns:
            Quantized values [..., D] in [-1, 1].
        """
        half_levels = (self._levels.float() - 1) / 2
        per_dim = []
        remainder = indices
        for i, lev in enumerate(self.levels):
            per_dim.append(remainder // self._basis[i])
            remainder = remainder % self._basis[i]

        per_dim = torch.stack(per_dim, dim=-1).float()
        return per_dim / half_levels - 1

    def extra_repr(self) -> str:
        return f"levels={self.levels}, codebook_size={self.codebook_size}"
