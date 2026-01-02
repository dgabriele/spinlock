"""Per-category metrics for categorical VQ-VAE evaluation.

Ported from unisim.system.utils.metrics (2025-12-30).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader


def compute_reconstruction_error(
    model,
    dataloader: DataLoader,
    device: str = "cuda"
) -> float:
    """Compute reconstruction error on dataset.

    Args:
        model: CategoricalHierarchicalVQVAE model
        dataloader: DataLoader for evaluation
        device: Device to use

    Returns:
        Mean reconstruction error (MSE)
    """
    model.eval()
    total_error = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Extract features from batch (handle both dict and tuple formats)
            if isinstance(batch, dict):
                features = batch["features"].to(device)
            else:
                features = batch[0].to(device)

            # Forward pass
            outputs = model(features)

            # Features reconstruction error
            # outputs["reconstruction"] is a dict with "features" key
            reconstruction = outputs["reconstruction"]
            if isinstance(reconstruction, dict):
                reconstruction = reconstruction["features"]
            error = F.mse_loss(reconstruction, features)

            total_error += error.item() * features.size(0)
            n_samples += features.size(0)

    return total_error / n_samples


def compute_quality_score(reconstruction_error: float, max_error: float = 1.0) -> float:
    """Compute quality score from reconstruction error.

    Quality = 1 - (error / max_error), clamped to [0, 1].
    Higher is better (1.0 = perfect reconstruction, 0.0 = worst).

    Args:
        reconstruction_error: MSE reconstruction error
        max_error: Maximum error for normalization (default 1.0)

    Returns:
        Quality score in [0, 1]
    """
    return max(0.0, min(1.0, 1.0 - (reconstruction_error / max_error)))


def compute_per_category_metrics(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    max_batches: int = 10,
) -> Dict[str, float]:
    """Compute reconstruction error and utilization per category.

    Args:
        model: CategoricalHierarchicalVQVAE model
        dataloader: DataLoader for evaluation
        device: Device to use
        max_batches: Maximum batches to process (default 10)

    Returns:
        Dict with keys like:
            "{category_name}/reconstruction_mse": float
            "{category_name}/level_{level_idx}/utilization": float
    """
    from spinlock.encoding import CategoricalHierarchicalVQVAE

    if not isinstance(model, CategoricalHierarchicalVQVAE):
        # Fallback for non-categorical models
        return {}

    model.eval()
    metrics = {}

    # Get category names from config
    category_names = sorted(model.config.group_indices.keys())

    # Accumulate per-category reconstruction errors
    category_recon_errors = {cat: [] for cat in category_names}
    all_tokens = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Extract features from batch (handle both dict and tuple formats)
            if isinstance(batch, dict):
                features = batch["features"].to(device)
            else:
                features = batch[0].to(device)

            # Forward pass
            output = model(features)

            # Per-category partial reconstructions
            partial_recons = output["partial_reconstructions"]
            tokens = output["tokens"]

            # Compute per-category MSE
            # partial_recons is list of [batch, input_dim] tensors (one per category-level)
            idx = 0
            for cat_name in category_names:
                # Get number of levels for this category
                cat_levels = model.config.get_category_levels(cat_name)
                num_levels = len(cat_levels)

                # Average across all levels for this category
                cat_recon_mse = 0.0
                for level_idx in range(num_levels):
                    partial_recon = partial_recons[idx]
                    cat_recon_mse += F.mse_loss(partial_recon, features).item()
                    idx += 1

                cat_recon_mse /= num_levels
                category_recon_errors[cat_name].append(cat_recon_mse)

            # Collect tokens for utilization computation
            all_tokens.append(tokens.cpu().numpy())

    # Aggregate per-category reconstruction errors
    for cat_name in category_names:
        if category_recon_errors[cat_name]:
            metrics[f"{cat_name}/reconstruction_mse"] = np.mean(
                category_recon_errors[cat_name]
            )

    # Compute per-level utilization
    all_tokens = np.concatenate(all_tokens, axis=0)  # [N, num_tokens]

    idx = 0
    for cat_name in category_names:
        cat_levels = model.config.get_category_levels(cat_name)
        for level_idx, level_config in enumerate(cat_levels):
            # Extract tokens for this category-level
            level_tokens = all_tokens[:, idx]

            # Compute utilization
            num_unique = len(np.unique(level_tokens))
            num_total = level_config["num_tokens"]
            utilization = num_unique / num_total

            metrics[f"{cat_name}/level_{level_idx}/utilization"] = utilization
            idx += 1

    return metrics


def compute_category_correlation(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    max_batches: int = 10,
) -> Dict[str, any]:
    """Compute pairwise correlation matrix between category latent vectors.

    Args:
        model: CategoricalHierarchicalVQVAE model
        dataloader: DataLoader for evaluation
        device: Device to use
        max_batches: Maximum batches to process (default 10)

    Returns:
        Dict with keys:
            "correlation_matrix": np.ndarray [N_categories, N_categories]
            "max_off_diagonal": float
            "mean_off_diagonal": float
            "category_names": List[str]
    """
    from spinlock.encoding import CategoricalHierarchicalVQVAE

    if not isinstance(model, CategoricalHierarchicalVQVAE):
        # Fallback for non-categorical models
        return {
            "correlation_matrix": np.array([[1.0]]),
            "max_off_diagonal": 0.0,
            "mean_off_diagonal": 0.0,
            "category_names": [],
        }

    model.eval()
    category_names = sorted(model.config.group_indices.keys())

    # Collect latent activations per category
    # Average across levels within each category
    latent_activations = {cat: [] for cat in category_names}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Extract features from batch (handle both dict and tuple formats)
            if isinstance(batch, dict):
                features = batch["features"].to(device)
            else:
                features = batch[0].to(device)

            # Forward pass
            output = model(features)
            latents = output["latents"]  # List of latent vectors

            # Collect latent vectors per category (average across levels)
            idx = 0
            for cat_name in category_names:
                cat_levels = model.config.get_category_levels(cat_name)
                num_levels = len(cat_levels)

                # Concatenate across levels within this category (handles variable latent dims)
                cat_latents = []
                for level_idx in range(num_levels):
                    cat_latents.append(latents[idx])
                    idx += 1

                # Concatenate along feature dim: [batch, latent_dim_0 + latent_dim_1 + ...]
                cat_concat = torch.cat(cat_latents, dim=1)

                # Mean pool across batch: [batch, concat_dim] -> [concat_dim]
                cat_avg_batch = cat_concat.mean(dim=0)
                latent_activations[cat_name].append(cat_avg_batch.cpu().numpy())

    # Compute correlation matrix
    # We correlate the flattened latent activation patterns across all batches
    N = len(category_names)
    corr_matrix = np.zeros((N, N))

    # Flatten all batch activations into single vectors per category
    category_vectors = {}
    for cat_name in category_names:
        acts = np.stack(latent_activations[cat_name])  # [batches, latent_dim]
        # Flatten to [batches * latent_dim]
        category_vectors[cat_name] = acts.flatten()

    for i, cat_i in enumerate(category_names):
        for j, cat_j in enumerate(category_names):
            vec_i = category_vectors[cat_i]
            vec_j = category_vectors[cat_j]

            # Check for zero variance (causes NaN in corrcoef)
            if vec_i.std() < 1e-10 or vec_j.std() < 1e-10:
                # Zero variance - set diagonal to 1.0, off-diagonal to 0.0
                corr = 1.0 if i == j else 0.0
            else:
                # Compute correlation between the flattened latent patterns
                # Need to handle different vector lengths - use corrcoef on reshaped arrays
                # Reshape to 2D: [2, min_length] for proper correlation
                min_len = min(len(vec_i), len(vec_j))
                vec_i_trunc = vec_i[:min_len]
                vec_j_trunc = vec_j[:min_len]

                corr = np.corrcoef(vec_i_trunc, vec_j_trunc)[0, 1]

            # Handle NaN (shouldn't happen with variance check, but be safe)
            if np.isnan(corr):
                corr = 1.0 if i == j else 0.0

            corr_matrix[i, j] = corr

    # Extract off-diagonal statistics
    mask = ~np.eye(N, dtype=bool)
    off_diagonal = corr_matrix[mask]

    return {
        "correlation_matrix": corr_matrix,
        "max_off_diagonal": (
            float(np.max(np.abs(off_diagonal))) if len(off_diagonal) > 0 else 0.0
        ),
        "mean_off_diagonal": (
            float(np.mean(np.abs(off_diagonal))) if len(off_diagonal) > 0 else 0.0
        ),
        "category_names": category_names,
    }
