"""Loss functions for VQ-VAE training.

5-component loss function:
1. Reconstruction (MSE)
2. Commitment (from VectorQuantizer)
3. Orthogonality (codebook diversity)
4. Informativeness (partial decoder quality)
5. Topographic similarity (topology preservation)

Ported from unisim.system.training.losses (simplified, removed multimodal support).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any


def reconstruction_loss(outputs: Dict[str, Any], targets: Dict[str, torch.Tensor]):
    """Compute reconstruction loss.

    Args:
        outputs: Model outputs with 'reconstruction' dict containing 'features'
        targets: Target dict with 'features'

    Returns:
        Reconstruction loss (MSE)
    """
    reconstruction = outputs["reconstruction"]
    loss = F.mse_loss(reconstruction["features"], targets["features"])
    return loss


def orthogonality_loss(model, max_samples: int = 64):
    """Compute orthogonality loss for codebook diversity.

    Encourages codebook vectors to be orthogonal (diverse).
    Uses aggressive sampling for large codebooks to avoid OOM.

    Args:
        model: CategoricalHierarchicalVQVAE model
        max_samples: Maximum number of codebook vectors to sample

    Returns:
        Orthogonality loss
    """
    loss = 0.0
    for quantizer in model.quantizers:
        # Get codebook
        codebook = quantizer.embedding.weight  # [num_embeddings, embedding_dim]
        num_embeddings = codebook.size(0)

        # For large codebooks, sample a subset to avoid OOM
        if num_embeddings > max_samples:
            # Random sampling without replacement
            indices = torch.randperm(num_embeddings, device=codebook.device)[:max_samples]
            codebook_sample = codebook[indices]
        else:
            codebook_sample = codebook

        # Normalize
        codebook_norm = F.normalize(codebook_sample, p=2, dim=1)

        # Compute pairwise cosine similarities in chunks to save memory
        n = codebook_norm.size(0)
        chunk_size = 32
        ortho_sum = 0.0
        count = 0

        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            chunk = codebook_norm[i:end_i]

            # Compute similarities for this chunk
            sim_chunk = chunk @ codebook_norm.t()  # [chunk_size, n]

            # Only penalize off-diagonal elements
            for local_idx, global_idx in enumerate(range(i, end_i)):
                # Zero out the diagonal element
                sim_chunk[local_idx, global_idx] = 0.0

            # Accumulate squared off-diagonal similarities
            ortho_sum += (sim_chunk**2).sum().item()
            count += sim_chunk.numel() - (end_i - i)  # Total minus diagonal

            # Free memory
            del sim_chunk
            del chunk

        # Average loss for this quantizer
        loss += torch.tensor(
            ortho_sum / count if count > 0 else 0.0, device=codebook.device
        )

    return loss / len(model.quantizers)


def informativeness_loss(outputs: Dict[str, Any], targets: Dict[str, torch.Tensor]):
    """Compute informativeness loss from partial decoders.

    Encourages each level to independently reconstruct the input.

    Args:
        outputs: Model outputs with 'partial_reconstructions' key
        targets: Target dict with 'features'

    Returns:
        Informativeness loss (average MSE across partial decoders)
    """
    partial_recons = outputs["partial_reconstructions"]
    target = targets["features"]

    # Average MSE across all partial decoders
    loss = sum(F.mse_loss(partial, target) for partial in partial_recons) / len(
        partial_recons
    )

    return loss


def topographic_similarity_loss(
    outputs: Dict[str, Any], targets: Dict[str, torch.Tensor], n_samples: int = 64
):
    """Compute topographic similarity loss.

    Preserves topology: nearby inputs should have nearby latent representations.

    Args:
        outputs: Model outputs with 'latents' key
        targets: Target dict with 'features' key
        n_samples: Number of samples to use for pairwise distances

    Returns:
        Topographic similarity loss
    """
    latents = outputs["latents"]  # List of [batch, latent_dim]
    features = targets["features"]  # [batch, input_dim]

    batch_size = features.size(0)
    if batch_size < n_samples:
        n_samples = batch_size

    # Sample random pairs
    indices = torch.randperm(batch_size, device=features.device)[:n_samples]
    sampled_features = features[indices]

    # Compute pairwise distances in input space
    input_dists = torch.cdist(sampled_features, sampled_features)  # [n_samples, n_samples]

    # Compute pairwise distances in latent space (average across levels)
    latent_dists = torch.zeros_like(input_dists)
    for latent in latents:
        sampled_latent = latent[indices]
        latent_dists += torch.cdist(sampled_latent, sampled_latent)
        del sampled_latent  # Free memory

    latent_dists /= len(latents)

    # Loss: correlation between input and latent distances
    # Pearson correlation approximation
    input_dists_flat = input_dists.view(-1)
    latent_dists_flat = latent_dists.view(-1)

    input_mean = input_dists_flat.mean()
    latent_mean = latent_dists_flat.mean()

    input_centered = input_dists_flat - input_mean
    latent_centered = latent_dists_flat - latent_mean

    correlation = (input_centered * latent_centered).sum() / (
        input_centered.norm() * latent_centered.norm() + 1e-8
    )

    # Clean up large temporaries
    del input_dists, latent_dists, input_dists_flat, latent_dists_flat
    del input_centered, latent_centered

    # Maximize correlation = minimize (1 - correlation)
    return 1.0 - correlation


def compute_total_loss(
    outputs: Dict[str, Any],
    targets: Dict[str, torch.Tensor],
    model,
    orthogonality_weight: float = 0.1,
    informativeness_weight: float = 0.1,
    topo_weight: float = 0.02,
    topo_samples: int = 64,
) -> Dict[str, torch.Tensor]:
    """Compute total loss with all components.

    Args:
        outputs: Model outputs
        targets: Target dict with 'features'
        model: CategoricalHierarchicalVQVAE model
        orthogonality_weight: Weight for orthogonality loss
        informativeness_weight: Weight for informativeness loss
        topo_weight: Weight for topographic loss
        topo_samples: Number of samples for topographic loss

    Returns:
        Dict with 'total' and individual loss components
    """
    # 1. Reconstruction loss
    recon_loss = reconstruction_loss(outputs, targets)

    # 2. VQ losses (commitment + codebook, already computed in forward pass)
    vq_loss = sum(outputs["vq_losses"])

    # 3. Orthogonality loss (codebook diversity)
    ortho_loss = orthogonality_loss(model)

    # 4. Informativeness loss (partial decoders)
    info_loss = informativeness_loss(outputs, targets)

    # 5. Topographic similarity loss
    topo_loss = topographic_similarity_loss(outputs, targets, topo_samples)

    # Total loss
    total = (
        recon_loss
        + vq_loss
        + orthogonality_weight * ortho_loss
        + informativeness_weight * info_loss
        + topo_weight * topo_loss
    )

    return {
        "total": total,
        "reconstruction": recon_loss,
        "vq": vq_loss,
        "orthogonality": ortho_loss,
        "informativeness": info_loss,
        "topographic": topo_loss,
    }
