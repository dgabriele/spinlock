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
        encodings = encodings.view(*input_shape[:-1], self.num_embeddings)

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
        """Reset codes below Nth percentile of EMA cluster sizes with reset cap.

        This method identifies "dead codes" (codebook entries with EMA cluster
        sizes below a percentile threshold) and reinitializes them with random
        data points. Uses EMA statistics for more stable detection compared to
        batch-based counting.

        Args:
            data_tensor: Input data tensor [N, D] for reinitialization samples
            percentile_threshold: Percentile threshold (0-100). Codes with EMA
                cluster sizes below this percentile are reset.
                Default 10.0 = reset bottom 10% of codes.
            max_reset_fraction: Maximum fraction of codebook to reset at once
                to avoid training disruption. Default 0.25 = max 25% reset.

        Returns:
            Number of codes that were reset
        """
        if not self.use_ema:
            return 0  # Only applies to EMA-based training

        with torch.no_grad():
            # Use EMA cluster sizes (more stable than batch counting)
            ema_sizes = self.ema_cluster_size.clone()

            # Compute percentile threshold from EMA distribution
            threshold_value = torch.quantile(ema_sizes, percentile_threshold / 100.0)

            # Find dead codes (at or below percentile threshold)
            # Use <= to handle cases where many codes have identical EMA sizes
            dead_mask = ema_sizes <= threshold_value
            dead_codes_all = dead_mask.nonzero(as_tuple=True)[0]

            # Cap number of resets to avoid training disruption
            max_reset = int(self.num_embeddings * max_reset_fraction)
            if len(dead_codes_all) > max_reset:
                # Sort by EMA size and reset the lowest ones first
                sorted_indices = torch.argsort(ema_sizes[dead_codes_all])
                dead_codes = dead_codes_all[sorted_indices[:max_reset]]
            else:
                dead_codes = dead_codes_all

            if len(dead_codes) > 0:
                # Get random samples from current batch for reinitialization
                flat_input = data_tensor.view(-1, self.embedding_dim)
                random_indices = torch.randint(
                    0, len(flat_input), (len(dead_codes),), device=data_tensor.device
                )
                random_embeddings = flat_input[random_indices]

                # Update codebook and EMA buffers
                self.embedding.weight.data[dead_codes] = random_embeddings
                self.ema_w[dead_codes] = random_embeddings

                # Reset to median cluster size (not 1.0 - prevents immediate re-death)
                median_size = ema_sizes.median()
                self.ema_cluster_size[dead_codes] = median_size

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
