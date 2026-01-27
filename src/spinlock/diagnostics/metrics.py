"""
Metric computation functions for MNO diagnostics.

Provides functions for computing physics fidelity metrics, tokenization quality metrics,
and distribution comparison metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
from scipy.spatial.distance import jensenshannon


def compute_mse(pred: torch.Tensor, target: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """Compute mean squared error.

    Args:
        pred: Predictions [...]
        target: Targets [...]
        reduce: Reduction mode ('mean', 'none')

    Returns:
        MSE value(s)
    """
    mse = (pred - target) ** 2
    if reduce == "mean":
        return mse.mean()
    elif reduce == "none":
        return mse
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")


def compute_relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute relative L2 error.

    Args:
        pred: Predictions [B, ...]
        target: Targets [B, ...]
        eps: Small constant for numerical stability

    Returns:
        Relative L2 error per sample [B]
    """
    batch_size = pred.shape[0]
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    numerator = torch.norm(pred_flat - target_flat, p=2, dim=1)
    denominator = torch.norm(target_flat, p=2, dim=1) + eps

    return numerator / denominator


def compute_nrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute normalized root mean squared error.

    Args:
        pred: Predictions [B, ...]
        target: Targets [B, ...]
        eps: Small constant for numerical stability

    Returns:
        NRMSE per sample [B]
    """
    batch_size = pred.shape[0]
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    mse = ((pred_flat - target_flat) ** 2).mean(dim=1)
    rmse = torch.sqrt(mse)

    target_range = target_flat.max(dim=1)[0] - target_flat.min(dim=1)[0] + eps

    return rmse / target_range


def compute_energy_norm_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE in energy norm (squared L2 norm).

    Args:
        pred: Predictions [B, ...]
        target: Targets [B, ...]

    Returns:
        Energy norm MSE per sample [B]
    """
    batch_size = pred.shape[0]
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    pred_energy = (pred_flat ** 2).sum(dim=1)
    target_energy = (target_flat ** 2).sum(dim=1)

    return (pred_energy - target_energy) ** 2


def compute_physics_metrics(
    pred_rollout: torch.Tensor,
    target_rollout: torch.Tensor,
    per_timestep: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute comprehensive physics fidelity metrics.

    Args:
        pred_rollout: Predicted rollout [B, T+1, C, H, W]
        target_rollout: Target rollout [B, T+1, C, H, W]
        per_timestep: If True, compute metrics per timestep

    Returns:
        Dictionary of metrics:
            - mse: Overall MSE or per-timestep MSE [T+1]
            - relative_l2: Mean relative L2 or per-timestep [T+1]
            - nrmse: Mean NRMSE or per-timestep [T+1]
            - energy_norm_mse: Mean energy norm MSE or per-timestep [T+1]
            - max_abs_error: Maximum absolute error or per-timestep [T+1]
    """
    B, T_plus_1, C, H, W = pred_rollout.shape

    if per_timestep:
        metrics = {
            "mse": [],
            "relative_l2": [],
            "nrmse": [],
            "energy_norm_mse": [],
            "max_abs_error": [],
        }

        for t in range(T_plus_1):
            pred_t = pred_rollout[:, t]  # [B, C, H, W]
            target_t = target_rollout[:, t]

            metrics["mse"].append(compute_mse(pred_t, target_t).item())
            metrics["relative_l2"].append(compute_relative_l2(pred_t, target_t).mean().item())
            metrics["nrmse"].append(compute_nrmse(pred_t, target_t).mean().item())
            metrics["energy_norm_mse"].append(compute_energy_norm_mse(pred_t, target_t).mean().item())
            metrics["max_abs_error"].append((pred_t - target_t).abs().max().item())

        # Convert lists to tensors
        for key in metrics:
            metrics[key] = torch.tensor(metrics[key])
    else:
        # Aggregate metrics across all timesteps
        metrics = {
            "mse": compute_mse(pred_rollout, target_rollout),
            "relative_l2": compute_relative_l2(pred_rollout, target_rollout).mean(),
            "nrmse": compute_nrmse(pred_rollout, target_rollout).mean(),
            "energy_norm_mse": compute_energy_norm_mse(pred_rollout, target_rollout).mean(),
            "max_abs_error": (pred_rollout - target_rollout).abs().max(),
        }

    return metrics


def compute_error_growth_stats(per_timestep_errors: torch.Tensor) -> Dict[str, float]:
    """Compute error growth statistics.

    Args:
        per_timestep_errors: Per-timestep errors [T+1]

    Returns:
        Dictionary with:
            - growth_rate: Linear growth rate (slope)
            - growth_type: 'linear' or 'exponential'
            - r_squared: Fit quality
    """
    timesteps = np.arange(len(per_timestep_errors))
    errors = per_timestep_errors.cpu().numpy()

    # Fit linear model
    linear_slope, linear_intercept, r_value, _, _ = scipy_stats.linregress(timesteps, errors)
    r_squared_linear = r_value ** 2

    # Fit exponential model (log-linear)
    # Avoid log of very small/zero values
    errors_safe = np.maximum(errors, 1e-10)
    try:
        exp_slope, exp_intercept, exp_r_value, _, _ = scipy_stats.linregress(timesteps, np.log(errors_safe))
        r_squared_exp = exp_r_value ** 2
    except:
        r_squared_exp = 0.0
        exp_slope = 0.0

    # Determine growth type based on better fit
    if r_squared_exp > r_squared_linear + 0.1:  # Exponential significantly better
        growth_type = "exponential"
        growth_rate = float(exp_slope)
        r_squared = r_squared_exp
    else:
        growth_type = "linear"
        growth_rate = float(linear_slope)
        r_squared = r_squared_linear

    return {
        "growth_rate": growth_rate,
        "growth_type": growth_type,
        "r_squared": r_squared,
    }


def compute_tokenization_metrics(
    features: torch.Tensor,
    reconstructed: torch.Tensor,
    tokens: torch.Tensor,
    num_embeddings: int,
    per_category: Optional[List[slice]] = None,
) -> Dict[str, float]:
    """Compute tokenization quality metrics.

    Args:
        features: Original features [N, D]
        reconstructed: Reconstructed features [N, D]
        tokens: Discrete tokens [N, L] where L is number of levels
        num_embeddings: Total codebook size
        per_category: Optional list of slices for per-category metrics

    Returns:
        Dictionary of metrics:
            - reconstruction_mse: Overall reconstruction MSE
            - perplexity: Token perplexity (codebook utilization)
            - token_entropy: Shannon entropy of token distribution
            - unique_tokens: Number of unique tokens used
            - per_category_mse: Dict of MSE per category (if provided)
    """
    metrics = {}

    # Reconstruction MSE
    metrics["reconstruction_mse"] = compute_mse(reconstructed, features).item()

    # Per-category reconstruction MSE
    if per_category is not None:
        per_cat_mse = {}
        for i, cat_slice in enumerate(per_category):
            cat_features = features[:, cat_slice]
            cat_reconstructed = reconstructed[:, cat_slice]
            per_cat_mse[f"category_{i}"] = compute_mse(cat_reconstructed, cat_features).item()
        metrics["per_category_mse"] = per_cat_mse

    # Token statistics
    tokens_flat = tokens.flatten().cpu().numpy()

    # Perplexity
    token_counts = np.bincount(tokens_flat, minlength=num_embeddings)
    token_probs = token_counts / token_counts.sum()
    token_probs = token_probs[token_probs > 0]  # Remove zeros
    perplexity = np.exp(-np.sum(token_probs * np.log(token_probs)))
    metrics["perplexity"] = float(perplexity)

    # Entropy
    entropy = -np.sum(token_probs * np.log2(token_probs + 1e-10))
    metrics["token_entropy"] = float(entropy)

    # Unique tokens
    metrics["unique_tokens"] = int(np.sum(token_counts > 0))

    return metrics


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL divergence KL(P||Q).

    Args:
        p: Probability distribution P
        q: Probability distribution Q
        eps: Small constant for numerical stability

    Returns:
        KL divergence value
    """
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence.

    Args:
        p: Probability distribution P
        q: Probability distribution Q

    Returns:
        JS divergence value (square root of JS distance)
    """
    p = np.asarray(p)
    q = np.asarray(q)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    return float(jensenshannon(p, q))


def compute_wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute 1D Wasserstein distance (Earth Mover's Distance).

    Args:
        p: Probability distribution P
        q: Probability distribution Q

    Returns:
        Wasserstein-1 distance
    """
    p = np.asarray(p)
    q = np.asarray(q)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Compute cumulative distributions
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)

    return float(np.sum(np.abs(cdf_p - cdf_q)))


def compute_distribution_metrics(
    tokens_a: np.ndarray,
    tokens_b: np.ndarray,
    num_embeddings: int,
    name_a: str = "A",
    name_b: str = "B",
) -> Dict[str, float]:
    """Compute distribution comparison metrics between two token sets.

    Args:
        tokens_a: Tokens from distribution A [N_a, L]
        tokens_b: Tokens from distribution B [N_b, L]
        num_embeddings: Codebook size
        name_a: Name for distribution A
        name_b: Name for distribution B

    Returns:
        Dictionary of metrics:
            - kl_divergence: KL(A||B)
            - js_divergence: JS divergence
            - wasserstein_distance: Wasserstein-1 distance
    """
    # Flatten tokens
    tokens_a_flat = tokens_a.flatten()
    tokens_b_flat = tokens_b.flatten()

    # Compute distributions
    dist_a = np.bincount(tokens_a_flat, minlength=num_embeddings).astype(float)
    dist_b = np.bincount(tokens_b_flat, minlength=num_embeddings).astype(float)

    # Normalize
    dist_a = dist_a / dist_a.sum()
    dist_b = dist_b / dist_b.sum()

    metrics = {
        f"kl_{name_a}_to_{name_b}": compute_kl_divergence(dist_a, dist_b),
        f"kl_{name_b}_to_{name_a}": compute_kl_divergence(dist_b, dist_a),
        f"js_divergence": compute_js_divergence(dist_a, dist_b),
        f"wasserstein_distance": compute_wasserstein_distance(dist_a, dist_b),
    }

    return metrics
