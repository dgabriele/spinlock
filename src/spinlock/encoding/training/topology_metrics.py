"""Post-Quantization Topology Metrics for VQ-VAE.

This module measures topology preservation from continuous latent embeddings
to discrete codebook assignments. Complements the existing pre-quantization
topographic similarity metric.

Key Metrics:
1. Pre-Quantization: Corr(D_input, D_latent)
2. Post-Quantization: Corr(D_latent, D_code)
3. End-to-End: Corr(D_input, D_code)
4. Quantization Degradation: Pre - Post

Ported from unisim.system.utils.topology_metrics (2025-12-30).
"""

from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def collect_samples_from_dataloader(
    model,
    dataloader: DataLoader,
    device: str,
    n_samples: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Collect features, latents, and tokens from dataloader.

    Args:
        model: Trained VQ-VAE model
        dataloader: Data loader
        device: Computation device
        n_samples: Number of samples to collect

    Returns:
        features: [n_samples, feature_dim]
        latents: List of [n_samples, latent_dim] per level
        tokens: [n_samples, num_levels]
    """
    model.eval()

    all_features = []
    all_latents = []
    all_tokens = []

    with torch.no_grad():
        for batch in dataloader:
            # Extract features from batch
            if isinstance(batch, dict):
                features = batch["features"].to(device)
            else:
                # Legacy single tensor format
                features = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

            # Forward pass
            outputs = model(features, return_encodings=True)

            # Initialize on first batch
            if not all_latents:
                num_levels = len(outputs["latents"])
                all_latents = [[] for _ in range(num_levels)]

            # Collect samples
            all_features.append(features.cpu())
            for i, latent in enumerate(outputs["latents"]):
                all_latents[i].append(latent.cpu())
            all_tokens.append(outputs["tokens"].cpu())

            # Check if we have enough samples
            total_samples = sum(f.shape[0] for f in all_features)
            if total_samples >= n_samples:
                break

    # Concatenate and trim to exactly n_samples
    features = torch.cat(all_features, dim=0)[:n_samples]
    latents = [torch.cat(level, dim=0)[:n_samples] for level in all_latents]
    tokens = torch.cat(all_tokens, dim=0)[:n_samples]

    return features, latents, tokens


def compute_discrete_topographic_similarity(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    n_samples: int = 1000,
    per_level: bool = False,
) -> Union[float, Dict[int, float]]:
    """Measure latent → code topology preservation (POST-quantization).

    Computes Pearson correlation between:
    - Pre-quantization latent pairwise distances (continuous space)
    - Post-quantization code embedding distances (in codebook space)

    Args:
        model: Trained categorical hierarchical VQ-VAE
        dataloader: Evaluation data loader
        device: Computation device
        n_samples: Number of samples for distance computation
        per_level: If True, return per-level correlations instead of average

    Returns:
        Average correlation across levels (float)
        OR per-level correlations (Dict[int, float]) if per_level=True
    """
    model.eval()

    # Collect samples
    features, latents, tokens = collect_samples_from_dataloader(
        model, dataloader, device, n_samples
    )

    # Move to device for distance computation
    device_obj = torch.device(device)

    # Compute per-level topology preservation
    level_correlations = {}

    for level_idx, (latent, vq_layer) in enumerate(zip(latents, model.vq_layers)):
        # Pre-quantization latent distances
        latent = latent.to(device_obj)
        latent_dists = torch.cdist(latent, latent).detach().cpu().numpy().flatten()

        # Post-quantization: Get code embeddings for assigned tokens
        level_tokens = tokens[:, level_idx].to(device_obj)  # [n_samples]
        code_embeddings = vq_layer.embedding(level_tokens)  # [n_samples, embedding_dim]

        # Code embedding distances
        code_dists = torch.cdist(code_embeddings, code_embeddings).detach().cpu().numpy().flatten()

        # Pearson correlation
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(latent_dists, code_dists)
        level_correlations[level_idx] = correlation

    if per_level:
        return level_correlations
    else:
        return float(np.mean(list(level_correlations.values())))


def compute_end_to_end_topographic_similarity(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    n_samples: int = 1000,
) -> float:
    """Measure input → code topology preservation (END-TO-END).

    Computes Pearson correlation between:
    - Input feature pairwise distances
    - Code embedding distances (discrete codebook space)

    Args:
        model: Trained VQ-VAE model
        dataloader: Evaluation data loader
        device: Computation device
        n_samples: Number of samples for distance computation

    Returns:
        Correlation coefficient
    """
    model.eval()

    # Collect samples
    features, latents, tokens = collect_samples_from_dataloader(
        model, dataloader, device, n_samples
    )

    device_obj = torch.device(device)
    features = features.to(device_obj)

    # Input feature distances
    input_dists = torch.cdist(features, features).detach().cpu().numpy().flatten()

    # Aggregate code embeddings across all levels
    all_code_embeddings = []
    for level_idx, vq_layer in enumerate(model.vq_layers):
        level_tokens = tokens[:, level_idx].to(device_obj)
        code_embeddings = vq_layer.embedding(level_tokens)
        all_code_embeddings.append(code_embeddings)

    # Concatenate embeddings from all levels
    aggregated_codes = torch.cat(all_code_embeddings, dim=1)  # [n_samples, total_embedding_dim]

    # Code embedding distances
    code_dists = torch.cdist(aggregated_codes, aggregated_codes).detach().cpu().numpy().flatten()

    # Pearson correlation
    from scipy.stats import pearsonr
    correlation, _ = pearsonr(input_dists, code_dists)

    return float(correlation)


def compute_pre_quantization_topographic_similarity(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    n_samples: int = 1000,
) -> float:
    """Measure input → latent topology preservation (PRE-quantization).

    Computes Pearson correlation between:
    - Input feature pairwise distances
    - Latent embedding distances (continuous space)

    Args:
        model: Trained VQ-VAE model
        dataloader: Evaluation data loader
        device: Computation device
        n_samples: Number of samples for distance computation

    Returns:
        Correlation coefficient
    """
    model.eval()

    # Collect samples
    features, latents, tokens = collect_samples_from_dataloader(
        model, dataloader, device, n_samples
    )

    device_obj = torch.device(device)
    features = features.to(device_obj)

    # Input feature distances
    input_dists = torch.cdist(features, features).detach().cpu().numpy().flatten()

    # Aggregate latent embeddings across all levels
    all_latents = []
    for latent in latents:
        all_latents.append(latent.to(device_obj))

    # Concatenate latents from all levels
    aggregated_latents = torch.cat(all_latents, dim=1)  # [n_samples, total_latent_dim]

    # Latent distances
    latent_dists = torch.cdist(aggregated_latents, aggregated_latents).detach().cpu().numpy().flatten()

    # Pearson correlation
    from scipy.stats import pearsonr
    correlation, _ = pearsonr(input_dists, latent_dists)

    return float(correlation)


def compute_topology_breakdown(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    n_samples: int = 1000,
) -> Dict[str, float]:
    """Comprehensive topology analysis across all stages.

    Measures topology preservation at three stages:
    1. Pre-quantization (input → latent)
    2. Post-quantization (latent → code)
    3. End-to-end (input → code)

    Also computes quantization degradation (pre - post).

    Args:
        model: Trained VQ-VAE model
        dataloader: Evaluation data loader
        device: Computation device
        n_samples: Number of samples for distance computation

    Returns:
        Dict with keys:
            - pre_quantization: Input → Latent correlation
            - post_quantization: Latent → Code correlation
            - end_to_end: Input → Code correlation
            - quantization_degradation: Pre - Post
    """
    # Collect samples once for all metrics
    features, latents, tokens = collect_samples_from_dataloader(
        model, dataloader, device, n_samples
    )

    device_obj = torch.device(device)
    features_dev = features.to(device_obj)

    # Compute input distances once
    input_dists = torch.cdist(features_dev, features_dev).detach().cpu().numpy().flatten()

    # Aggregate latents across levels
    aggregated_latents = torch.cat([l.to(device_obj) for l in latents], dim=1)
    latent_dists = torch.cdist(aggregated_latents, aggregated_latents).detach().cpu().numpy().flatten()

    # Aggregate code embeddings across levels
    all_code_embeddings = []
    for level_idx, vq_layer in enumerate(model.vq_layers):
        level_tokens = tokens[:, level_idx].to(device_obj)
        code_embeddings = vq_layer.embedding(level_tokens)
        all_code_embeddings.append(code_embeddings)

    aggregated_codes = torch.cat(all_code_embeddings, dim=1)
    code_dists = torch.cdist(aggregated_codes, aggregated_codes).detach().cpu().numpy().flatten()

    # Compute correlations
    from scipy.stats import pearsonr

    pre_quantization, _ = pearsonr(input_dists, latent_dists)
    post_quantization, _ = pearsonr(latent_dists, code_dists)
    end_to_end, _ = pearsonr(input_dists, code_dists)

    quantization_degradation = pre_quantization - post_quantization

    return {
        "pre_quantization": float(pre_quantization),
        "post_quantization": float(post_quantization),
        "end_to_end": float(end_to_end),
        "quantization_degradation": float(quantization_degradation),
    }
