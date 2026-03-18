"""Integrated token decoder: VQ tokens → (theta, IC) via codebook lookup + inverse heads.

Drop-in replacement for CVAE in the refinement pipeline. Deterministic: each
token set maps to exactly one (theta, IC) pair. Diversity comes from D3PM's
stochastic denoising (different denoising trajectories → different candidates),
not from z-sampling.
"""

import logging
from typing import Dict

import torch

from .fsq import FiniteScalarQuantizer

logger = logging.getLogger(__name__)


class IntegratedTokenDecoder:
    """Decode VQ tokens to continuous (theta, IC) via codebook lookup + inverse heads.

    Replaces CVAE in the refinement pipeline. Deterministic: each token set maps
    to exactly one (theta, IC) pair.

    Interface matches CVAE.sample() for drop-in replacement.

    Args:
        tokenizer: VQTokenizer with trained model (including inverse heads).
    """

    def __init__(self, tokenizer):
        from .tokenizer import VQTokenizer
        self.tokenizer = tokenizer
        self.model = tokenizer.model

    @torch.no_grad()
    def decode(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode token indices to continuous values.

        Args:
            tokens: Dict mapping "family_category_Ll" → token indices [B].

        Returns:
            Dict with optional keys:
                "theta": [B, param_dim] decoded parameters
                "grids": [B, C, H, W] decoded initial conditions
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # 1. Look up embeddings from codebooks (in sorted key order)
        quantized_parts = []
        for key in sorted(self.model.quantizers.keys()):
            if key not in tokens:
                continue
            quantizer = self.model.quantizers[key]
            indices = tokens[key].to(device)
            if isinstance(quantizer, FiniteScalarQuantizer):
                emb = quantizer.indices_to_values(indices)  # [B, len(levels)]
            else:
                emb = quantizer.embedding(indices)  # [B, latent_dim]
            quantized_parts.append(emb)

        if not quantized_parts:
            raise ValueError("No matching quantizer keys found in tokens")

        all_quantized = torch.cat(quantized_parts, dim=1)  # [B, total_latent_dim]

        # 2. Decode through shared decoder → split by family
        reconstructed = self.model.decoder(all_quantized)

        # Build a dummy all_encoded for shape reference
        # (the decoder output has same shape as total_encoded_dim)
        reconstructed_split = self.model._split_reconstructed(
            reconstructed, reconstructed
        )

        # 3. Apply inverse heads
        result = {}

        # Theta: bypass decoder (direct mode) or inverse MLP
        if self.model.theta_bypass_decoder is not None:
            theta_q_parts = []
            for key in sorted(self.model.quantizers.keys()):
                if key.startswith("theta_") and key in tokens:
                    quantizer = self.model.quantizers[key]
                    indices = tokens[key].to(device)
                    if isinstance(quantizer, FiniteScalarQuantizer):
                        emb = quantizer.indices_to_values(indices)
                    else:
                        emb = quantizer.embedding(indices)
                    theta_q_parts.append(emb)
            if theta_q_parts:
                theta_q_cat = torch.cat(theta_q_parts, dim=1)
                result["theta"] = self.model.theta_bypass_decoder(theta_q_cat)
        elif self.model.theta_inverse is not None and "theta" in reconstructed_split:
            result["theta"] = self.model.theta_inverse(reconstructed_split["theta"])

        # Initial: bypass decoder (spectral, from quantized latents) or shared decoder path
        if self.model.initial_bypass_inverse is not None:
            initial_q_parts = []
            for key in sorted(self.model.quantizers.keys()):
                if key.startswith("initial_") and key in tokens:
                    quantizer = self.model.quantizers[key]
                    indices = tokens[key].to(device)
                    if isinstance(quantizer, FiniteScalarQuantizer):
                        emb = quantizer.indices_to_values(indices)
                    else:
                        emb = quantizer.embedding(indices)
                    initial_q_parts.append(emb)
            if initial_q_parts:
                initial_q_cat = torch.cat(initial_q_parts, dim=1)
                result["grids"] = self.model.initial_bypass_inverse(initial_q_cat)
        elif self.model.initial_inverse is not None and "initial" in reconstructed_split:
            result["grids"] = self.model.initial_inverse(reconstructed_split["initial"])

        return result
