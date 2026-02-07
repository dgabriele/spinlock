"""Loss functions for VQ-VAE training.

Implements the 5-component loss used in spinlock V2:
1. Reconstruction loss - MSE between input and reconstructed features
2. VQ loss - Vector quantization commitment loss (from quantizers)
3. Orthogonality loss - Minimize correlation between category representations
4. Informativeness loss - Maximize variance within each category
5. Topographic loss - Optional spatial organization of codebook
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import LossConfig

logger = logging.getLogger(__name__)


def compute_reconstruction_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute reconstruction loss between original and reconstructed features.

    Args:
        original: Original encoded features [B, D]
        reconstructed: Reconstructed features [B, D]
        normalize: If True, normalize by feature dimension

    Returns:
        Scalar reconstruction loss
    """
    mse = F.mse_loss(reconstructed, original, reduction='mean')

    if normalize:
        # Normalize by feature dimension to make loss scale-invariant
        mse = mse / original.shape[1]

    return mse


def compute_orthogonality_loss(
    category_embeddings: Dict[str, torch.Tensor],
    target_correlation: float = 0.0,
) -> torch.Tensor:
    """Compute orthogonality loss between category representations.

    Encourages different categories to learn decorrelated representations
    by penalizing high correlations between category embeddings.

    Args:
        category_embeddings: Dict mapping category_name → embeddings [B, D_cat]
        target_correlation: Target correlation (default: 0.0 for full orthogonality)

    Returns:
        Scalar orthogonality loss
    """
    if len(category_embeddings) <= 1:
        return torch.tensor(0.0, device=next(iter(category_embeddings.values())).device)

    categories = sorted(category_embeddings.keys())
    correlations = []

    for i, cat_i in enumerate(categories):
        for cat_j in categories[i + 1:]:
            emb_i = category_embeddings[cat_i]  # [B, D_i]
            emb_j = category_embeddings[cat_j]  # [B, D_j]

            # Normalize embeddings
            emb_i_norm = F.normalize(emb_i, p=2, dim=1)
            emb_j_norm = F.normalize(emb_j, p=2, dim=1)

            # Compute correlation (cosine similarity)
            corr = (emb_i_norm * emb_j_norm).sum(dim=1).abs().mean()
            correlations.append(corr)

    if not correlations:
        return torch.tensor(0.0, device=emb_i.device)

    # Penalize deviation from target correlation
    correlations = torch.stack(correlations)
    loss = (correlations - target_correlation).pow(2).mean()

    return loss


def compute_informativeness_loss(
    category_embeddings: Dict[str, torch.Tensor],
    min_variance: float = 0.01,
) -> torch.Tensor:
    """Compute informativeness loss to encourage high variance within categories.

    Penalizes low variance in category embeddings to prevent collapse
    to constant representations.

    Args:
        category_embeddings: Dict mapping category_name → embeddings [B, D_cat]
        min_variance: Minimum target variance threshold

    Returns:
        Scalar informativeness loss (lower variance = higher loss)
    """
    variances = []

    for cat_name, embeddings in category_embeddings.items():
        # Compute variance along batch dimension for each feature
        var = embeddings.var(dim=0, unbiased=False).mean()  # Average across features
        variances.append(var)

    if not variances:
        return torch.tensor(0.0, device=next(iter(category_embeddings.values())).device)

    variances = torch.stack(variances)

    # Penalize variance below threshold (ReLU ensures only low variance is penalized)
    loss = F.relu(min_variance - variances).mean()

    return loss


def compute_topographic_loss(
    quantized_vectors: torch.Tensor,
    token_indices: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Compute topographic loss for spatial organization of codebook.

    Encourages similar latent vectors to map to nearby codebook entries,
    creating a smooth topographic organization.

    Args:
        quantized_vectors: Quantized latent vectors [B, D]
        token_indices: Selected codebook indices [B]
        codebook: Codebook embeddings [K, D]

    Returns:
        Scalar topographic loss
    """
    batch_size = quantized_vectors.shape[0]
    device = quantized_vectors.device

    # For each sample, compute distance in latent space vs codebook index space
    latent_dists = torch.cdist(quantized_vectors, quantized_vectors, p=2)  # [B, B]

    # Codebook index distances (L1 distance between indices)
    index_dists = (
        token_indices.unsqueeze(1) - token_indices.unsqueeze(0)
    ).abs().float()  # [B, B]

    # Normalize both distances to [0, 1]
    latent_dists = latent_dists / (latent_dists.max() + 1e-8)
    index_dists = index_dists / (index_dists.max() + 1e-8)

    # Penalize mismatch between latent and index distances
    loss = F.mse_loss(latent_dists, index_dists)

    return loss


class VQVAELoss:
    """Combined loss function for VQ-VAE training.

    Computes weighted sum of 5 loss components:
    1. Reconstruction loss (MSE)
    2. VQ loss (from quantizers)
    3. Orthogonality loss
    4. Informativeness loss
    5. Topographic loss (optional)

    Args:
        config: Loss configuration with weights
    """

    def __init__(self, config: LossConfig):
        self.config = config

    def __call__(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        vq_loss: torch.Tensor,
        category_embeddings: Dict[str, torch.Tensor],
        quantized_vectors: Optional[Dict[str, torch.Tensor]] = None,
        token_indices: Optional[Dict[str, torch.Tensor]] = None,
        codebooks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total VQ-VAE loss.

        Args:
            original: Original encoded features [B, D]
            reconstructed: Reconstructed features [B, D]
            vq_loss: VQ commitment loss from quantizers (scalar)
            category_embeddings: Dict mapping category → embeddings [B, D_cat]
            quantized_vectors: Optional dict of quantized vectors per category
            token_indices: Optional dict of token indices per category
            codebooks: Optional dict of codebook tensors per category

        Returns:
            Dict with keys:
                - total: Total weighted loss
                - reconstruction: Reconstruction loss component
                - vq: VQ loss component
                - orthogonality: Orthogonality loss component
                - informativeness: Informativeness loss component
                - topographic: Topographic loss component (if enabled)
        """
        # 1. Reconstruction loss
        recon_loss = compute_reconstruction_loss(
            original, reconstructed, normalize=self.config.normalize_reconstruction
        )

        # 2. VQ loss (already computed by quantizers)
        # This includes commitment cost from VectorQuantizer

        # 3. Orthogonality loss
        ortho_loss = compute_orthogonality_loss(category_embeddings)

        # 4. Informativeness loss
        info_loss = compute_informativeness_loss(category_embeddings)

        # 5. Topographic loss (optional)
        topo_loss = torch.tensor(0.0, device=original.device)
        if self.config.topographic_weight > 0:
            if quantized_vectors and token_indices and codebooks:
                topo_losses = []
                for cat in quantized_vectors.keys():
                    topo_losses.append(
                        compute_topographic_loss(
                            quantized_vectors[cat],
                            token_indices[cat],
                            codebooks[cat],
                        )
                    )
                if topo_losses:
                    topo_loss = torch.stack(topo_losses).mean()

        # Weighted combination
        total_loss = (
            self.config.reconstruction_weight * recon_loss
            + vq_loss  # VQ loss already weighted by commitment_cost in quantizer
            + self.config.orthogonality_weight * ortho_loss
            + self.config.informativeness_weight * info_loss
            + self.config.topographic_weight * topo_loss
        )

        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "vq": vq_loss,
            "orthogonality": ortho_loss,
            "informativeness": info_loss,
            "topographic": topo_loss,
        }
