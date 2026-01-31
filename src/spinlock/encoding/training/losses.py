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
from typing import Dict, Any, Optional


def reconstruction_loss(
    outputs: Dict[str, Any],
    targets: Dict[str, torch.Tensor],
    feature_weights: Optional[torch.Tensor] = None,
    mask_info: Optional[Dict[str, Any]] = None,
):
    """Compute reconstruction loss with optional per-feature weighting and masking.

    Args:
        outputs: Model outputs with 'reconstruction' dict containing 'features'
        targets: Target dict with 'features'
        feature_weights: Optional per-feature importance weights [D]
                        If None, uses uniform weighting (standard MSE)
        mask_info: Optional mask information dict from encoder.
                  If provided, applies sample weighting based on valid fraction.
                  Expected keys:
                      - num_valid_timesteps: [B] number of valid timesteps per sample

    Returns:
        Reconstruction loss (weighted MSE with optional sample weighting)

    Notes:
        Sample weighting prevents short trajectories from dominating gradient updates:
        - Short trajectory (100/500 valid) gets weight 100/500 = 0.2
        - Long trajectory (500/500 valid) gets weight 500/500 = 1.0

        This ensures loss contribution is proportional to information content.
    """
    reconstruction = outputs["reconstruction"]
    recon_features = reconstruction["features"]  # [batch, D]
    target_features = targets["features"]  # [batch, D]

    # Compute squared errors
    squared_errors = (recon_features - target_features) ** 2  # [batch, D]

    # Apply per-feature weights
    if feature_weights is not None:
        # feature_weights: [D] -> [1, D] for broadcasting
        squared_errors = squared_errors * feature_weights.unsqueeze(0)  # [batch, D]

    # Apply sample weighting based on valid fraction (variable-length support)
    if mask_info is not None and "num_valid_timesteps" in mask_info:
        num_valid = mask_info["num_valid_timesteps"]  # [B]

        # Normalize by maximum valid count in batch
        # This prevents short trajectories from dominating
        max_valid = num_valid.max().float()

        if max_valid > 0:
            # Sample weights: fraction of valid timesteps
            # Shape: [B] -> [B, 1] for broadcasting over features
            sample_weights = (num_valid.float() / max_valid).unsqueeze(1)

            # Weight squared errors by sample importance
            weighted_errors = squared_errors * sample_weights  # [B, D]
            loss = weighted_errors.mean()
        else:
            # No valid timesteps (edge case), fallback to uniform
            loss = squared_errors.mean()
    else:
        # Standard mode: uniform weighting across samples
        loss = squared_errors.mean()

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
    """Compute topographic similarity loss (PRE and POST quantization).

    Preserves topology at TWO stages:
    1. PRE-quantization: Input → Latent (encoder quality)
    2. POST-quantization: Latent → Code (VQ quality)

    The total loss optimizes both, with emphasis on post-quantization since
    that's what the downstream NOA will use.

    Args:
        outputs: Model outputs with 'latents' and 'quantized' keys
        targets: Target dict with 'features' key
        n_samples: Number of samples to use for pairwise distances

    Returns:
        Tuple of (total_loss, metrics_dict) where metrics_dict contains:
            - topo_pre: Pre-quantization correlation
            - topo_post: Post-quantization correlation
    """
    latents = outputs["latents"]  # List of [batch, latent_dim] - PRE-quantization
    features = targets["features"]  # [batch, input_dim]

    batch_size = features.size(0)
    if batch_size < n_samples:
        n_samples = batch_size

    # Sample random pairs
    indices = torch.randperm(batch_size, device=features.device)[:n_samples]
    sampled_features = features[indices]

    # Compute pairwise distances in input space
    input_dists = torch.cdist(sampled_features, sampled_features)  # [n_samples, n_samples]

    # Compute pairwise distances in PRE-quantization latent space
    latent_dists = torch.zeros_like(input_dists)
    for latent in latents:
        sampled_latent = latent[indices]
        latent_dists += torch.cdist(sampled_latent, sampled_latent)
        del sampled_latent

    latent_dists /= len(latents)

    # Compute PRE-quantization correlation (input → latent)
    input_flat = input_dists.view(-1)
    latent_flat = latent_dists.view(-1)

    input_mean = input_flat.mean()
    latent_mean = latent_flat.mean()

    input_centered = input_flat - input_mean
    latent_centered = latent_flat - latent_mean

    pre_correlation = (input_centered * latent_centered).sum() / (
        input_centered.norm() * latent_centered.norm() + 1e-8
    )

    # Compute POST-quantization correlation (latent → code)
    post_correlation = torch.tensor(0.0, device=features.device)

    if "quantized" in outputs:
        # Use quantized code embeddings
        quantized = outputs["quantized"]  # List of [batch, embed_dim] - POST-quantization
        code_dists = torch.zeros_like(input_dists)

        for q in quantized:
            sampled_q = q[indices]
            code_dists += torch.cdist(sampled_q, sampled_q)
            del sampled_q

        code_dists /= len(quantized)

        # Correlation: latent → code (VQ discretization quality)
        code_flat = code_dists.view(-1)
        code_mean = code_flat.mean()
        code_centered = code_flat - code_mean

        post_correlation = (latent_centered * code_centered).sum() / (
            latent_centered.norm() * code_centered.norm() + 1e-8
        )

        del code_dists, code_flat, code_centered

    # Clean up
    del input_dists, latent_dists, input_flat, latent_flat
    del input_centered, latent_centered

    # Combined loss: optimize both pre and post quantization
    # Weight post-quantization more since that's what NOA uses
    pre_loss = 1.0 - pre_correlation
    post_loss = 1.0 - post_correlation

    # 30% pre, 70% post weighting (post is more important for NOA)
    total_loss = 0.3 * pre_loss + 0.7 * post_loss

    return total_loss, {
        "topo_pre": pre_correlation.item(),
        "topo_post": post_correlation.item(),
    }


def reference_regularization_loss(
    outputs: Dict[str, Any],
    targets: Dict[str, torch.Tensor],
    reference_features: Optional[torch.Tensor] = None,
    is_interpolated: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Regularize MNO SUMMARY features against reference solver SUMMARY features.

    Compares MNO's raw SUMMARY features (330D) against reference solver (CNO) SUMMARY
    features (330D) to guide VQ-VAE toward learning physics-consistent representations.

    This is NOT a reconstruction loss - it measures how close MNO's features are to
    the reference solver's features in the raw SUMMARY space (before VQ-VAE encoding).

    The loss is applied selectively:
    - Interpolated points: MNO generalizing between training points (regularized)
    - Exact training points: MNO learned these during training (not regularized)

    Args:
        outputs: Model outputs (not used)
        targets: Target dict containing 'raw_summary' [B, 330] (MNO SUMMARY features)
        reference_features: Reference solver SUMMARY features [B, 330] from CNO
        is_interpolated: Boolean mask [B] for interpolated samples

    Returns:
        Scalar loss: MSE between MNO SUMMARY and reference SUMMARY (330D vs 330D)
    """
    # No regularization if reference features not provided
    if reference_features is None:
        return torch.tensor(0.0, device=targets["features"].device)

    # Extract MNO raw SUMMARY features
    if "raw_summary" not in targets:
        # No raw_summary available, skip regularization
        return torch.tensor(0.0, device=targets["features"].device)

    mno_summary = targets["raw_summary"]  # [B, 330]

    # Apply only to interpolated samples if mask provided
    if is_interpolated is not None:
        if not is_interpolated.any():
            return torch.tensor(0.0, device=targets["features"].device)

        mno_summary = mno_summary[is_interpolated]
        reference_summary = reference_features[is_interpolated]
    else:
        reference_summary = reference_features

    # Verify dimensions match (reference features should be pre-cleaned)
    if reference_summary.shape[1] != mno_summary.shape[1]:
        raise ValueError(
            f"Dimension mismatch: MNO summary={mno_summary.shape[1]}D, "
            f"reference={reference_summary.shape[1]}D. "
            f"Reference features should be pre-cleaned using FeaturePreprocessor "
            f"to remove operator_sensitivity features before training."
        )

    # MSE between MNO SUMMARY and reference (CNO) SUMMARY in raw space
    loss = F.mse_loss(mno_summary, reference_summary)

    return loss


def entropy_regularization_loss(outputs: Dict[str, Any], device: str = "cuda") -> torch.Tensor:
    """Compute entropy regularization loss to encourage uniform codebook usage.

    Encourages uniform distribution across codebook entries by maximizing
    the entropy of code usage. Higher entropy = better utilization.

    Formula:
        H = -sum(p_i * log(p_i))  (actual entropy)
        H_max = log(K)             (maximum possible entropy for K codes)
        Loss = H_max - H           (distance from uniform, always positive)

    Args:
        outputs: Model outputs with 'encodings' key containing one-hot code assignments
        device: Device for tensor creation (default: "cuda")

    Returns:
        Entropy regularization loss (positive, 0 = perfect uniform usage)
    """
    if "encodings" not in outputs or not outputs["encodings"]:
        return torch.tensor(0.0, device=device)

    encodings_list = outputs["encodings"]  # List of [batch, K] one-hot encodings

    if len(encodings_list) == 0:
        return torch.tensor(0.0, device=device)

    entropy_losses = []

    for encodings in encodings_list:
        # Compute code usage distribution across batch
        # encodings: [batch, K] one-hot vectors
        # avg_probs: [K] probability of each code being used
        avg_probs = encodings.mean(0)  # Average over batch dimension
        K = encodings.shape[-1]  # Number of codebook entries

        # Compute actual entropy: H = -sum(p * log(p))
        epsilon = 1e-10
        actual_entropy = -torch.sum(avg_probs * torch.log(avg_probs + epsilon))

        # Maximum possible entropy (uniform distribution): H_max = log(K)
        max_entropy = torch.log(torch.tensor(K, dtype=torch.float32, device=device))

        # Loss = distance from maximum entropy (positive, 0 = perfect)
        entropy_loss = max_entropy - actual_entropy
        entropy_losses.append(entropy_loss)

    # Average across all quantizers
    return torch.stack(entropy_losses).mean()


def compute_total_loss(
    outputs: Dict[str, Any],
    targets: Dict[str, torch.Tensor],
    model,
    orthogonality_weight: float = 0.1,
    informativeness_weight: float = 0.1,
    topo_weight: float = 0.02,
    topo_samples: int = 64,
    reference_reg_weight: float = 0.0,
    reference_features: Optional[torch.Tensor] = None,
    is_interpolated: Optional[torch.Tensor] = None,
    feature_weights: Optional[torch.Tensor] = None,
    entropy_weight: float = 0.0,
    mask_info: Optional[Dict[str, Any]] = None,
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
        reference_reg_weight: Weight for reference feature regularization loss
                             0.0 = disabled (default)
        reference_features: Reference solver features [B, D] from high-fidelity solver
                           If None, regularization disabled
        is_interpolated: Boolean mask [B] for interpolated samples
                        If None, regularization applied to all samples
        feature_weights: Optional per-feature importance weights [D]
                        If None, uses uniform weighting (standard MSE)
        entropy_weight: Weight for entropy regularization (encourages uniform codebook usage)
                       0.0 = disabled (default). Recommended: 0.01-0.1
        mask_info: Optional mask information dict from encoder (variable-length support).
                  If provided, applies sample weighting in reconstruction loss.
                  Expected keys:
                      - num_valid_timesteps: [B] number of valid timesteps per sample

    Returns:
        Dict with 'total' and individual loss components including:
            - topo_pre: Pre-quantization topographic similarity (correlation)
            - topo_post: Post-quantization topographic similarity (correlation)
            - reference_regularization: Reference feature regularization loss
            - entropy: Entropy regularization loss (positive, 0 = perfect uniform usage)
    """
    # 1. Reconstruction loss (with optional feature weighting and masking)
    recon_loss = reconstruction_loss(
        outputs, targets, feature_weights=feature_weights, mask_info=mask_info
    )

    # 2. VQ losses (commitment + codebook, already computed in forward pass)
    vq_loss = sum(outputs["vq_losses"])

    # 3. Orthogonality loss (codebook diversity)
    ortho_loss = orthogonality_loss(model)

    # 4. Informativeness loss (partial decoders)
    info_loss = informativeness_loss(outputs, targets)

    # 5. Topographic similarity loss (PRE + POST quantization)
    topo_loss, topo_metrics = topographic_similarity_loss(outputs, targets, topo_samples)

    # 6. Reference feature regularization loss (optional)
    if reference_reg_weight > 0 and reference_features is not None:
        ref_reg_loss = reference_regularization_loss(
            outputs, targets, reference_features, is_interpolated
        )
    else:
        # Skip computation if disabled (weight = 0) or no reference features
        ref_reg_loss = torch.tensor(0.0, device=targets["features"].device)

    # 7. Entropy regularization loss (optional)
    if entropy_weight > 0:
        ent_loss = entropy_regularization_loss(outputs, device=targets["features"].device)
    else:
        # Skip computation if disabled (weight = 0)
        ent_loss = torch.tensor(0.0, device=targets["features"].device)

    # Total loss
    total = (
        recon_loss
        + vq_loss
        + orthogonality_weight * ortho_loss
        + informativeness_weight * info_loss
        + topo_weight * topo_loss
        + reference_reg_weight * ref_reg_loss
        + entropy_weight * ent_loss
    )

    return {
        "total": total,
        "reconstruction": recon_loss,
        "vq": vq_loss,
        "orthogonality": ortho_loss,
        "informativeness": info_loss,
        "topographic": topo_loss,
        "topo_pre": topo_metrics["topo_pre"],  # Pre-quantization correlation
        "topo_post": topo_metrics["topo_post"],  # Post-quantization correlation
        "reference_regularization": ref_reg_loss,  # Reference feature regularization
        "entropy": ent_loss,  # Entropy regularization (uniform codebook usage)
    }
