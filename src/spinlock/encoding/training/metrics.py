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
        model: CategoricalHierarchicalVQVAE or VQVAEWithInitial model
        dataloader: DataLoader for evaluation
        device: Device to use

    Returns:
        Mean reconstruction error (MSE)
    """
    model.eval()
    total_error = 0.0
    n_samples = 0

    # Check if model is a hybrid model that needs raw_ics
    is_hybrid = hasattr(model, 'initial_encoder')

    with torch.no_grad():
        for batch in dataloader:
            # Extract features from batch (handle both dict and tuple formats)
            if isinstance(batch, dict):
                features = batch["features"].to(device)
                raw_ics = batch.get("raw_ics")
                if raw_ics is not None:
                    raw_ics = raw_ics.to(device)
            else:
                features = batch[0].to(device)
                raw_ics = None

            # Forward pass (pass raw_ics for hybrid models)
            if is_hybrid and raw_ics is not None:
                outputs = model(features, raw_ics=raw_ics)
            else:
                outputs = model(features)

            # Features reconstruction error
            # outputs["reconstruction"] is a dict with "features" key
            reconstruction = outputs["reconstruction"]
            if isinstance(reconstruction, dict):
                reconstruction = reconstruction["features"]

            # For hybrid models, use input_features (expanded) as target
            if "input_features" in outputs:
                target = outputs["input_features"]
            else:
                target = features

            error = F.mse_loss(reconstruction, target)

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


def compute_category_balance(per_category_errors: Dict[str, float]) -> float:
    """Compute Gini coefficient of per-category reconstruction errors.

    Lower Gini = more balanced errors across categories.

    Args:
        per_category_errors: Dict mapping category names to reconstruction MSE

    Returns:
        Gini coefficient in [0, 1]: 0 = perfect balance, 1 = complete inequality
    """
    import numpy as np

    if not per_category_errors:
        return 0.0

    errors = np.array(list(per_category_errors.values()))
    n = len(errors)

    if n == 0:
        return 0.0

    # Sort errors
    sorted_errors = np.sort(errors)

    # Compute Gini: (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_errors)) / (n * np.sum(sorted_errors) + 1e-8) - (n + 1) / n

    return float(max(0.0, min(1.0, gini)))


def compute_quality_score_3factor(
    reconstruction_error: float,
    utilization: float,
    category_balance: float,
    max_error: float = 1.0
) -> Dict[str, float]:
    """Compute comprehensive 3-factor quality score (inspired by unisim).

    Combines three factors using geometric mean:
    1. Reconstruction quality (lower error is better)
    2. Utilization quality (healthy codebook usage, optimal range 30-70%)
    3. Balance quality (low variance across categories)

    Args:
        reconstruction_error: MSE reconstruction error
        utilization: Average codebook utilization [0-1]
        category_balance: Gini coefficient of per-category MSE [0-1]
        max_error: Maximum error for normalization (default: 1.0)

    Returns:
        Dict with:
            - reconstruction_quality: 1 - error/max_error [0-1]
            - utilization_quality: sigmoid-weighted utilization [0-1]
            - balance_quality: 1 - Gini coefficient [0-1]
            - composite_quality: geometric mean of 3 factors [0-1]
            - target_met: bool (composite >= 0.90)
    """
    import numpy as np

    # Factor 1: Reconstruction quality (existing metric)
    recon_quality = max(0.0, min(1.0, 1.0 - (reconstruction_error / max_error)))

    # Factor 2: Utilization quality (sigmoid penalty for low utilization)
    # Optimal range: 30-70% (avoid overfitting to few codes or dead codes)
    # Sigmoid centered at 25% utilization with steep penalty below
    util_quality = 1.0 / (1.0 + np.exp(-20 * (utilization - 0.25)))

    # Factor 3: Category balance (avoid catastrophic failure in any category)
    # Lower Gini = more balanced errors across categories = better
    balance_quality = 1.0 - category_balance

    # Composite: Geometric mean (penalizes weakest factor)
    # Cannot compensate low utilization with high reconstruction
    composite = (recon_quality * util_quality * balance_quality) ** (1/3)

    return {
        "reconstruction_quality": float(recon_quality),
        "utilization_quality": float(util_quality),
        "balance_quality": float(balance_quality),
        "composite_quality": float(composite),
        "target_met": composite >= 0.90
    }


def compute_per_category_metrics(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    max_batches: int = 10,
) -> Dict[str, float]:
    """Compute reconstruction error and utilization per category.

    Args:
        model: CategoricalHierarchicalVQVAE or VQVAEWithInitial model
        dataloader: DataLoader for evaluation
        device: Device to use
        max_batches: Maximum batches to process (default 10)

    Returns:
        Dict with keys like:
            "{category_name}/reconstruction_mse": float
            "{category_name}/level_{level_idx}/utilization": float
    """
    from spinlock.encoding import CategoricalHierarchicalVQVAE

    # Handle VQVAEWithInitial wrapper - use underlying vqvae for type checks
    model_for_check = model
    if hasattr(model, 'vqvae'):
        model_for_check = model.vqvae

    if not isinstance(model_for_check, CategoricalHierarchicalVQVAE):
        # Fallback for non-categorical models
        return {}

    model.eval()
    metrics = {}

    # Get category names from config (use the model's config, not model_for_check)
    category_names = sorted(model.config.group_indices.keys())

    # Check if model is a hybrid model that needs raw_ics
    is_hybrid = hasattr(model, 'initial_encoder')

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
                raw_ics = batch.get("raw_ics")
                if raw_ics is not None:
                    raw_ics = raw_ics.to(device)
            else:
                features = batch[0].to(device)
                raw_ics = None

            # Forward pass (pass raw_ics for hybrid models)
            if is_hybrid and raw_ics is not None:
                output = model(features, raw_ics=raw_ics)
                # Use expanded input_features for per-category comparisons
                features_for_comparison = output["input_features"]
            else:
                output = model(features)
                features_for_comparison = features

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

                # Get the feature indices for this category
                cat_indices = model.config.group_indices[cat_name]

                # Extract only this category's features from the full feature vector
                # Use features_for_comparison which is expanded for hybrid models
                cat_features = features_for_comparison[:, cat_indices]

                # Average across all levels for this category
                cat_recon_mse = 0.0
                for level_idx in range(num_levels):
                    partial_recon = partial_recons[idx]
                    # CRITICAL FIX: partial_recon is a FULL 298-feature reconstruction
                    # Extract category-specific features from BOTH tensors before comparing
                    cat_partial_recon = partial_recon[:, cat_indices]
                    # Now compare like-to-like (both are category-specific)
                    cat_recon_mse += F.mse_loss(cat_partial_recon, cat_features).item()
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
        model: CategoricalHierarchicalVQVAE or VQVAEWithInitial model
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

    # Handle VQVAEWithInitial wrapper - use underlying vqvae for type checks
    model_for_check = model
    if hasattr(model, 'vqvae'):
        model_for_check = model.vqvae

    if not isinstance(model_for_check, CategoricalHierarchicalVQVAE):
        # Fallback for non-categorical models
        return {
            "correlation_matrix": np.array([[1.0]]),
            "max_off_diagonal": 0.0,
            "mean_off_diagonal": 0.0,
            "category_names": [],
        }

    model.eval()
    category_names = sorted(model.config.group_indices.keys())

    # Check if model is a hybrid model that needs raw_ics
    is_hybrid = hasattr(model, 'initial_encoder')

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
                raw_ics = batch.get("raw_ics")
                if raw_ics is not None:
                    raw_ics = raw_ics.to(device)
            else:
                features = batch[0].to(device)
                raw_ics = None

            # Forward pass (pass raw_ics for hybrid models)
            if is_hybrid and raw_ics is not None:
                output = model(features, raw_ics=raw_ics)
            else:
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
