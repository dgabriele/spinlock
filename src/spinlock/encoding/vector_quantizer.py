"""Vector quantization layer for VQ-VAE tokenization.

This module implements the core vector quantization layer used in VQ-VAE
for converting continuous latent codes into discrete tokens. Features:

- Learned codebook with nearest-neighbor quantization
- Exponential moving average (EMA) for stable codebook learning
- Straight-through gradient estimator for backpropagation
- Dead code reset mechanism to prevent codebook collapse
- Codebook utilization and perplexity metrics

Ported from unisim.pipeline.encoding.vqvae (100% generic, no domain-specific code).

References:
    van den Oord et al. "Neural Discrete Representation Learning" (2017)
    https://arxiv.org/abs/1711.00937
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class VectorQuantizer(nn.Module):
    """Vector quantization layer with learned codebook.

    This layer quantizes continuous latent codes by finding the nearest
    codebook vector. Gradients flow through quantization via straight-through
    estimator. Codebook can be updated via EMA or gradient descent.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_ema: bool = True,
    ):
        """Initialize vector quantizer.

        Args:
            num_embeddings: Codebook size (K)
            embedding_dim: Dimension of codebook vectors (D)
            commitment_cost: Weight for commitment loss
            decay: EMA decay rate for codebook updates
            epsilon: Small constant for numerical stability
            use_ema: Whether to use EMA for codebook updates
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema

        # Codebook: K x D embedding matrix
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # Usage counter: accumulates assignment counts between dead code resets.
        # Used by both EMA and gradient modes for dead code detection.
        self.register_buffer("_usage_count", torch.zeros(num_embeddings))

        if use_ema:
            # EMA statistics for codebook updates
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize inputs using codebook.

        Args:
            inputs: Continuous latent codes [B, D] or [B, T, D]

        Returns:
            quantized: Quantized codes (same shape as inputs)
            encodings: One-hot encodings of nearest codebook indices [B, K] or [B, T, K]
            losses: Dictionary with 'loss', 'codebook_loss', 'commitment_loss'
        """
        # Handle both 2D and 3D inputs
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)  # [N, D]

        # Calculate distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)  # ||z||^2
            + torch.sum(self.embedding.weight**2, dim=1)  # ||e||^2
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())  # 2*z*e
        )  # [N, K]

        # Find nearest codebook vectors
        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # [N, K]

        # Quantize: look up embeddings
        quantized = self.embedding(encoding_indices)  # [N, D]

        if self.training:
            # Accumulate assignment counts for dead code detection
            self._usage_count.add_(encodings.sum(0))

        if self.training and self.use_ema:
            # EMA codebook updates (only in training)
            self.ema_cluster_size.data.mul_(self.decay).add_(  # type: ignore
                encodings.sum(0), alpha=1 - self.decay
            )

            # Laplace smoothing for cluster sizes
            n = self.ema_cluster_size.sum()  # type: ignore
            self.ema_cluster_size.data.add_(self.epsilon).div_(  # type: ignore
                n + self.num_embeddings * self.epsilon  # type: ignore
            ).mul_(n)  # type: ignore

            # Update embeddings
            dw = torch.matmul(encodings.t(), flat_input)  # [K, D]
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)  # type: ignore

            self.embedding.weight.data.copy_(
                self.ema_w / self.ema_cluster_size.unsqueeze(1)  # type: ignore
            )

        # Compute losses
        # Codebook loss: move codebook towards encoder outputs
        codebook_loss = F.mse_loss(quantized.detach(), flat_input)

        # Commitment loss: move encoder outputs towards codebook
        commitment_loss = F.mse_loss(quantized, flat_input.detach())

        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from decoder to encoder
        quantized = flat_input + (quantized - flat_input).detach()

        # Reshape to original shape
        quantized = quantized.view(input_shape)
        try:
            encodings = encodings.view(*input_shape[:-1], self.num_embeddings)
        except RuntimeError as e:
            # Add debugging info for dimension mismatches
            expected_shape = list(input_shape[:-1]) + [self.num_embeddings]
            actual_elements = encodings.numel()
            expected_elements = int(np.prod(expected_shape))
            raise RuntimeError(
                f"VectorQuantizer dimension mismatch:\n"
                f"  Input shape: {input_shape}\n"
                f"  embedding_dim (self): {self.embedding_dim}\n"
                f"  num_embeddings (self): {self.num_embeddings}\n"
                f"  flat_input shape: {flat_input.shape}\n"
                f"  encodings shape: {encodings.shape}\n"
                f"  encodings.numel(): {actual_elements}\n"
                f"  trying to view as: {expected_shape}\n"
                f"  expected elements: {expected_elements}\n"
                f"Original error: {e}"
            ) from e

        losses = {
            "loss": loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }

        return quantized, encodings, losses

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codebook vectors for given indices.

        Args:
            indices: Codebook indices [B] or [B, T]

        Returns:
            Codebook vectors [B, D] or [B, T, D]
        """
        return self.embedding(indices)

    def reset_dead_codes(
        self,
        data_tensor: torch.Tensor,
        percentile_threshold: float = 10.0,
        max_reset_fraction: float = 0.25,
    ) -> int:
        """Reset underused codebook entries by reinitializing from encoder outputs.

        Identifies "dead codes" (entries below a usage percentile) and replaces
        them with random samples from the current batch.  Works in both EMA
        mode (uses ``ema_cluster_size``) and gradient mode (uses accumulated
        ``_usage_count`` since last reset).

        Args:
            data_tensor: Encoder output tensor [N, D] for reinitialization
            percentile_threshold: Percentile (0-100) below which codes are
                considered dead.  Default 10.0 = bottom 10%.
            max_reset_fraction: Max fraction of codebook to reset per call.

        Returns:
            Number of codes that were reset
        """
        with torch.no_grad():
            # Select usage signal: EMA cluster sizes or accumulated counts
            usage = (
                self.ema_cluster_size.clone()
                if self.use_ema
                else self._usage_count.clone()
            )

            # Identify dead codes (at or below percentile threshold)
            threshold = torch.quantile(usage, percentile_threshold / 100.0)
            dead_codes = (usage <= threshold).nonzero(as_tuple=True)[0]

            # Cap resets to avoid training disruption
            max_reset = int(self.num_embeddings * max_reset_fraction)
            if len(dead_codes) > max_reset:
                sorted_idx = torch.argsort(usage[dead_codes])
                dead_codes = dead_codes[sorted_idx[:max_reset]]

            if len(dead_codes) > 0:
                flat_input = data_tensor.view(-1, self.embedding_dim)
                rand_idx = torch.randint(
                    0, len(flat_input), (len(dead_codes),),
                    device=data_tensor.device,
                )
                new_embeddings = flat_input[rand_idx]

                # Reinitialize codebook entries
                self.embedding.weight.data[dead_codes] = new_embeddings

                # Update mode-specific bookkeeping
                median_usage = usage.median()
                if self.use_ema:
                    self.ema_w[dead_codes] = new_embeddings
                    self.ema_cluster_size[dead_codes] = median_usage
                self._usage_count[dead_codes] = median_usage

            # Reset accumulated counts for next interval
            self._usage_count.zero_()

            return int(len(dead_codes))


def compute_codebook_metrics(
    encodings: torch.Tensor, num_embeddings: int
) -> Dict[str, float]:
    """Compute codebook quality metrics.

    Args:
        encodings: One-hot encodings [B, K] or [B, T, K]
        num_embeddings: Codebook size K

    Returns:
        Dictionary with:
            - utilization: Fraction of codebook used (0-1)
            - perplexity: Effective codebook size
            - avg_usage: Average usage per active code
    """
    # Flatten to [N, K]
    if encodings.dim() == 3:
        encodings = encodings.view(-1, num_embeddings)

    # Average usage per codebook vector
    avg_probs = encodings.mean(0)  # [K]

    # Codebook utilization: fraction of codes used
    utilization = (avg_probs > 0).float().mean().item()

    # Perplexity: exp(entropy)
    # Higher perplexity = more codes used uniformly
    epsilon = 1e-10
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + epsilon))).item()

    # Average usage per active code
    active_codes = avg_probs > 0
    avg_usage = avg_probs[active_codes].mean().item() if active_codes.any() else 0.0

    return {
        "utilization": utilization,
        "perplexity": perplexity,
        "avg_usage": avg_usage,
    }
