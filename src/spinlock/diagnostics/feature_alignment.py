"""
Feature extraction alignment for MNO diagnostics.

Provides utilities to extract features from rollouts in a way that exactly matches
how VQ-VAE was trained. This ensures that tokenization evaluation is performed on
correctly formatted and normalized features.

The key challenge is that VQ-VAE was trained on features extracted from CNO rollouts.
To evaluate MNO outputs, we must extract features from MNO rollouts using the identical
extraction and normalization pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from spinlock.noa.vqvae_alignment import AlignedFeatureExtractor


def extract_features_from_rollouts(
    rollouts: torch.Tensor,
    vqvae_checkpoint_path: Path,
    device: str = "cuda",
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract features from rollouts matching VQ-VAE training format.

    For the 50K baseline VQ-VAE which uses initial + temporal features,
    this function extracts spatial statistics and temporal dynamics features.

    Args:
        rollouts: Rollouts tensor [N, T+1, C, H, W]
        vqvae_checkpoint_path: Path to VQ-VAE checkpoint directory or file
        device: Computation device
        batch_size: Batch size for processing

    Returns:
        features: Extracted and normalized features [N, D]
                 where D matches VQ-VAE input dimension
    """
    from spinlock.features.extractors.initial import extract_all_initial_features
    from spinlock.features.extractors.temporal import extract_all_temporal_features

    # Load VQ-VAE checkpoint to get feature names and normalization
    checkpoint_path = Path(vqvae_checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_file = checkpoint_path / "best_model.pt"
    else:
        checkpoint_file = checkpoint_path

    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    feature_names = checkpoint['feature_names']
    normalization_stats = checkpoint['normalization_stats']

    N = rollouts.shape[0]
    feature_list = []

    # Process in batches
    with torch.no_grad():
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_rollouts = rollouts[batch_start:batch_end]  # [B, T+1, C, H, W]

            batch_features = []

            for i in range(batch_rollouts.shape[0]):
                rollout = batch_rollouts[i]  # [T+1, C, H, W]

                # Extract initial features (from IC at t=0)
                ic = rollout[0].cpu().numpy()  # [C, H, W]
                initial_feats = extract_all_initial_features(ic)

                # Extract temporal features (from full rollout)
                rollout_np = rollout.cpu().numpy()  # [T+1, C, H, W]
                temporal_feats = extract_all_temporal_features(rollout_np)

                # Combine features in the order they appear in feature_names
                all_feats = {**initial_feats, **temporal_feats}

                # Select only the features used by VQ-VAE
                selected_feats = []
                for name in feature_names:
                    if name in all_feats:
                        selected_feats.append(all_feats[name])
                    else:
                        # Feature not found, use 0.0
                        print(f"Warning: Feature {name} not found, using 0.0")
                        selected_feats.append(0.0)

                batch_features.append(selected_feats)

            # Stack batch features
            batch_tensor = torch.tensor(batch_features, dtype=torch.float32)

            # Normalize using VQ-VAE's stats
            mean = torch.tensor(normalization_stats['mean'], dtype=torch.float32)
            std = torch.tensor(normalization_stats['std'], dtype=torch.float32)
            batch_tensor = (batch_tensor - mean) / (std + 1e-8)

            feature_list.append(batch_tensor)

    # Concatenate all batches
    all_features = torch.cat(feature_list, dim=0)

    return all_features


def load_vqvae_for_tokenization(
    vqvae_checkpoint_path: Path,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load VQ-VAE model for tokenization evaluation.

    Args:
        vqvae_checkpoint_path: Path to VQ-VAE checkpoint directory or file
        device: Computation device

    Returns:
        Tuple of (vqvae_model, checkpoint_metadata):
            - vqvae_model: Loaded VQ-VAE model in eval mode
            - checkpoint_metadata: Dict with metadata (group_indices, config, etc.)
    """
    from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig

    path = Path(vqvae_checkpoint_path)

    # Determine checkpoint file
    if path.is_dir():
        checkpoint_path = path / "best_model.pt"
    else:
        checkpoint_path = path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract metadata
    model_config = checkpoint.get('model_config', {})
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']

    # Get group_indices from model_config (authoritative)
    group_indices = model_config.get('group_indices', config.get('group_indices', {}))

    # Get input_dim
    input_dim = model_config.get('input_dim', config.get('input_dim', 270))

    # Detect hybrid model
    is_hybrid_model = any('initial_encoder' in k for k in state_dict.keys())

    # Create VQ-VAE model
    if is_hybrid_model:
        # For hybrid models, use the loading logic from VQVAEAlignmentLoss
        # This is complex, so we'll import it
        from spinlock.noa.vqvae_alignment import VQVAEAlignmentLoss

        # Use the from_checkpoint method to handle all the complexity
        alignment = VQVAEAlignmentLoss.from_checkpoint(
            vqvae_path=str(vqvae_checkpoint_path),
            device=device,
            use_aligned_extractor=False,  # We don't need the feature extractor
        )
        vqvae = alignment.vqvae
    else:
        # Standard VQ-VAE model
        vqvae_config = CategoricalVQVAEConfig(
            input_dim=input_dim,
            group_indices=group_indices,
            group_embedding_dim=config.get('group_embedding_dim', 256),
            group_hidden_dim=config.get('group_hidden_dim', 512),
            levels=config.get('levels'),
        )

        vqvae = CategoricalHierarchicalVQVAE(vqvae_config)

        # Handle different checkpoint formats
        has_orig_mod = any(k.startswith('_orig_mod.') for k in state_dict.keys())
        has_vqvae_prefix = any(k.startswith('vqvae.') for k in state_dict.keys())

        if has_orig_mod and has_vqvae_prefix:
            # _orig_mod.vqvae. prefix
            vqvae_state = {k.replace('_orig_mod.vqvae.', ''): v
                          for k, v in state_dict.items()
                          if k.startswith('_orig_mod.vqvae.')}
        elif has_vqvae_prefix:
            # vqvae. prefix
            vqvae_state = {k.replace('vqvae.', ''): v
                          for k, v in state_dict.items()
                          if k.startswith('vqvae.')}
        elif has_orig_mod:
            # _orig_mod. prefix
            vqvae_state = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        else:
            # No prefix
            vqvae_state = state_dict

        vqvae.load_state_dict(vqvae_state)

    # Set to eval mode and freeze
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False

    # Prepare metadata
    metadata = {
        'group_indices': group_indices,
        'input_dim': input_dim,
        'config': config,
        'model_config': model_config,
        'is_hybrid': is_hybrid_model,
        'checkpoint_path': str(checkpoint_path),
    }

    return vqvae.to(device), metadata


def tokenize_features(
    features: torch.Tensor,
    vqvae: nn.Module,
    device: str = "cuda",
    batch_size: int = 256,
    return_reconstructed: bool = True,
) -> Dict[str, torch.Tensor]:
    """Tokenize features using VQ-VAE model.

    Args:
        features: Normalized features [N, D]
        vqvae: VQ-VAE model in eval mode
        device: Computation device
        batch_size: Batch size for processing
        return_reconstructed: If True, also return reconstructed features

    Returns:
        Dictionary containing:
            - tokens: Discrete tokens [N, L] where L is number of VQ levels
            - reconstructed: Reconstructed features [N, D] (if return_reconstructed=True)
            - quantized: Quantized latents (pre-decoding)
    """
    N = features.shape[0]
    tokens_list = []
    reconstructed_list = []
    quantized_list = []

    vqvae.eval()

    with torch.no_grad():
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_features = features[batch_start:batch_end].to(device)

            # Encode to pre-quantization latents
            z_list = vqvae.encode(batch_features)

            # Quantize to get tokens and quantized latents
            z_q_list, token_list, _ = vqvae.quantize(z_list)

            # Stack tokens from all levels [B, L]
            batch_tokens = torch.stack(token_list, dim=1)
            tokens_list.append(batch_tokens.cpu())

            # Concatenate quantized latents
            z_q = torch.cat(z_q_list, dim=1)
            quantized_list.append(z_q.cpu())

            if return_reconstructed:
                # Decode to reconstructed features
                reconstructed = vqvae.decode(z_q_list)
                reconstructed_list.append(reconstructed.cpu())

    # Concatenate all batches
    result = {
        'tokens': torch.cat(tokens_list, dim=0),
        'quantized': torch.cat(quantized_list, dim=0),
    }

    if return_reconstructed:
        result['reconstructed'] = torch.cat(reconstructed_list, dim=0)

    return result


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
