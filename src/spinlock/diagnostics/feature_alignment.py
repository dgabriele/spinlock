"""
Feature extraction alignment for MNO diagnostics.

Provides utilities to extract features from rollouts in a way that exactly matches
how VQ-VAE was trained. This ensures that tokenization evaluation is performed on
correctly formatted and normalized features.

Supports both 2-family (initial + temporal) and 3-family (initial + summary + temporal) checkpoints.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from spinlock.mno.vqvae_alignment import AlignedFeatureExtractor


def extract_features_from_rollouts(
    rollouts: torch.Tensor,
    ics: torch.Tensor,
    vqvae_checkpoint_path: Path,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract features from rollouts using AlignedFeatureExtractor.

    For fair comparison between MNO (1-channel) and CNO (3-channel), we extract
    features from ONLY the first channel (density) of all rollouts.

    Args:
        rollouts: Rollouts tensor [N, T+1, C, H, W]
        ics: Initial conditions [N, C, H, W]
        vqvae_checkpoint_path: Path to VQ-VAE checkpoint directory or file
        device: Computation device

    Returns:
        features: Extracted and normalized features [N, D] where D depends on checkpoint type
    """
    # Extract only first channel for fair comparison
    # MNO outputs 1 channel, CNO outputs 3 channels - we compare on density only
    rollouts_ch0 = rollouts[:, :, :1, :, :]  # [N, T+1, 1, H, W]
    ics_ch0 = ics[:, :1, :, :]  # [N, 1, H, W]

    # Use AlignedFeatureExtractor which handles both 2-family and 3-family checkpoints
    extractor = AlignedFeatureExtractor.from_checkpoint(
        checkpoint_path=str(vqvae_checkpoint_path),
        device=device
    )

    with torch.no_grad():
        # Extract features - rollouts are [N, T+1, C, H, W] where T+1 includes IC
        # AlignedFeatureExtractor expects trajectory [N, T, C, H, W] (without IC)
        # and ic [N, C, H, W] separately
        features, _ = extractor(
            trajectory=rollouts_ch0[:, 1:].to(device),  # Remove IC from trajectory
            ic=ics_ch0.to(device)
        )

    return features


def tokenize_features(
    rollouts: torch.Tensor,
    ics: torch.Tensor,
    vqvae_checkpoint_path: Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Tokenize rollouts using VQVAEAlignmentLoss infrastructure.

    Uses the existing DRY infrastructure (VQVAEAlignmentLoss) which handles:
    - Feature extraction via AlignedFeatureExtractor
    - Initial field encoding
    - Feature masking and grouping
    - VQ-VAE tokenization and reconstruction

    Args:
        rollouts: Rollouts tensor [N, T+1, C, H, W]
        ics: Initial conditions [N, C, H, W]
        vqvae_checkpoint_path: Path to VQ-VAE checkpoint directory or file
        device: Computation device

    Returns:
        Dictionary containing:
            - tokens: Hierarchical tokens from VQ-VAE
            - reconstructed_features: Reconstructed features
            - reconstruction_mse: Overall reconstruction MSE (float)
            - perplexity: Token perplexity (float)
            - per_category_mse: Per-category reconstruction MSE (dict)
    """
    from spinlock.mno.vqvae_alignment import VQVAEAlignmentLoss

    # Resolve checkpoint path
    checkpoint_path = Path(vqvae_checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best_model.pt"

    # Extract only first channel for fair comparison
    rollouts_ch0 = rollouts[:, :, :1, :, :].to(device)  # [N, T+1, 1, H, W]
    ics_ch0 = ics[:, :1, :, :].to(device)  # [N, 1, H, W]

    # Create VQVAEAlignmentLoss which handles the full pipeline
    alignment_loss = VQVAEAlignmentLoss.from_checkpoint(
        vqvae_path=str(checkpoint_path),
        device=device
    )

    # Extract features and tokenize
    with torch.no_grad():
        # Extract features using the feature extractor
        pred_result = alignment_loss.feature_extractor(rollouts_ch0[:, 1:], ic=ics_ch0)  # [N, T, C, H, W]

        if isinstance(pred_result, tuple):
            features, raw_ics = pred_result
        else:
            features = pred_result
            raw_ics = ics_ch0

        # Apply normalization if stats are available
        if hasattr(alignment_loss, '_normalize_features'):
            features_norm = alignment_loss._normalize_features(features)
        else:
            # Features are already normalized by UnifiedFeaturePipeline
            features_norm = features

        # Pass through VQ-VAE encode -> quantize -> decode pipeline
        vqvae = alignment_loss.vqvae

        # Encode: features -> latents (handles all preprocessing internally)
        if hasattr(vqvae, 'encode'):
            z_list = vqvae.encode(features_norm, raw_ics=raw_ics)
        else:
            z_list = vqvae.vqvae.encode(features_norm)

        # Quantize: latents -> quantized latents + tokens + losses + encodings
        z_q_list, tokens, vq_losses, _ = vqvae.quantize(z_list)

        # Decode: quantized latents -> reconstructed features
        reconstructed = vqvae.decode(z_q_list)

        # Compute reconstruction error in latent space (commitment loss)
        # This measures how much information is lost during quantization
        z_cat = torch.cat(z_list, dim=1)
        z_q_cat = torch.cat(z_q_list, dim=1)
        reconstruction_mse = torch.nn.functional.mse_loss(z_cat, z_q_cat).item()

    # Compute perplexity from tokens
    token_counts = {}

    # Handle different token formats
    if isinstance(tokens, torch.Tensor):
        for token in tokens.flatten():
            token_val = token.item()
            token_counts[token_val] = token_counts.get(token_val, 0) + 1
    elif isinstance(tokens, (list, tuple)):
        for token_tensor in tokens:
            for token in token_tensor.flatten():
                token_val = token.item()
                token_counts[token_val] = token_counts.get(token_val, 0) + 1

    total_tokens = sum(token_counts.values())
    if total_tokens > 0:
        probs = torch.tensor([count / total_tokens for count in token_counts.values()])
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        perplexity = torch.exp(entropy).item()
    else:
        perplexity = 0.0

    return {
        'tokens': tokens,
        'reconstructed_features': reconstructed,
        'reconstruction_mse': reconstruction_mse,
        'perplexity': perplexity,
        'per_category_mse': {},  # TODO: Compute per-category if needed
    }


def get_token_distribution(
    tokens: torch.Tensor,
    num_embeddings: int,
) -> np.ndarray:
    """Compute token distribution from a set of tokens.

    Args:
        tokens: Discrete tokens [N, L]
        num_embeddings: Codebook size per level

    Returns:
        Token distribution [num_embeddings] (normalized to sum to 1)
    """
    # Flatten all tokens
    tokens_flat = tokens.flatten().cpu().numpy()

    # Compute histogram
    token_counts = np.bincount(tokens_flat, minlength=num_embeddings)

    # Normalize to probability distribution
    token_dist = token_counts.astype(float) / token_counts.sum()

    return token_dist


def get_per_level_distributions(
    tokens: torch.Tensor,
    num_embeddings: int,
) -> Dict[int, np.ndarray]:
    """Compute per-level token distributions.

    Args:
        tokens: Discrete tokens [N, L]
        num_embeddings: Codebook size per level

    Returns:
        Dictionary mapping level_idx -> token distribution [num_embeddings]
    """
    N, L = tokens.shape

    distributions = {}

    for level in range(L):
        level_tokens = tokens[:, level].cpu().numpy()
        token_counts = np.bincount(level_tokens, minlength=num_embeddings)
        token_dist = token_counts.astype(float) / token_counts.sum()
        distributions[level] = token_dist

    return distributions
