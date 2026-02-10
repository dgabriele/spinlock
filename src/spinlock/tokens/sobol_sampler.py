"""Sobol sampling for Token-to-Rollout VAE latent space.

This module provides utilities for generating Sobol-sampled variations
in the VAE latent space, enabling diverse rollout generation with low-discrepancy
sampling.
"""

from typing import Dict

import numpy as np
import torch
from scipy.stats import norm

from spinlock.sampling.sobol import StratifiedSobolSampler
from spinlock.tokens.rollout_vae import TokenToRolloutVAE


def generate_sobol_variations(
    model: TokenToRolloutVAE,
    tokens: Dict[str, torch.Tensor],
    n_variations: int = 100,
    sobol_seed: int = 42,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Generate Sobol-sampled variations in VAE latent space.

    This function:
    1. Encodes tokens to latent distribution (mu, logvar)
    2. Generates Sobol samples in unit hypercube [0,1]^latent_dim
    3. Transforms via inverse CDF to Gaussian space
    4. Decodes to (theta, initial_grids)

    Args:
        model: Trained TokenToRolloutVAE
        tokens: Dict mapping token keys to indices [1]
        n_variations: Number of variations to generate
        sobol_seed: Seed for Sobol sequence
        device: Device for computation

    Returns:
        Dictionary containing:
        - theta: [n_variations, theta_dim] parameter variations
        - grids: [n_variations, C, H, W] initial grid variations
        - z: [n_variations, latent_dim] latent codes
        - mu: [latent_dim] mean latent
        - std: [latent_dim] std latent
    """
    model.eval()
    model.to(device)

    # Move tokens to device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Get latent distribution
    with torch.no_grad():
        mu, logvar = model.encode(tokens)  # [1, latent_dim]
        std = torch.exp(0.5 * logvar)  # [1, latent_dim]

        # Squeeze to 1D
        mu = mu.squeeze(0)  # [latent_dim]
        std = std.squeeze(0)  # [latent_dim]

    # Create Sobol sampler
    sobol_sampler = StratifiedSobolSampler(
        dimensionality=model.latent_dim,
        scramble=True,
        seed=sobol_seed,
    )

    # Generate Sobol samples in [0,1]^latent_dim
    sobol_samples = sobol_sampler.sample(n_variations)  # [n_variations, latent_dim]

    # Transform to Gaussian space via inverse CDF
    # Note: Handle edge cases (0 and 1) to avoid inf in norm.ppf
    sobol_samples = np.clip(sobol_samples, 1e-10, 1 - 1e-10)
    gaussian_samples = norm.ppf(sobol_samples)  # [n_variations, latent_dim]

    # Convert to torch and apply to distribution
    gaussian_samples = torch.from_numpy(gaussian_samples).float().to(device)
    z = mu.unsqueeze(0) + gaussian_samples * std.unsqueeze(0)  # [n_variations, latent_dim]

    # Decode variations
    with torch.no_grad():
        theta, grids = model.decode(z)

    return {
        "theta": theta.cpu().numpy(),
        "grids": grids.cpu().numpy(),
        "z": z.cpu().numpy(),
        "mu": mu.cpu().numpy(),
        "std": std.cpu().numpy(),
    }


def validate_sobol_coverage(
    samples: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Validate Sobol sampling coverage quality.

    Computes metrics to verify that Sobol samples uniformly cover
    the latent space.

    Args:
        samples: Sobol samples [n_samples, latent_dim]
        n_bins: Number of bins for coverage check

    Returns:
        Dictionary containing:
        - discrepancy: Star discrepancy (lower is better, target < 0.01)
        - min_pairwise_dist: Minimum pairwise distance (higher is better)
        - coverage_uniformity: Uniformity of bin coverage (1.0 = perfect)
    """
    from spinlock.sampling.metrics import compute_star_discrepancy

    n_samples, latent_dim = samples.shape

    # Compute star discrepancy
    discrepancy = compute_star_discrepancy(samples)

    # Compute min pairwise distance (normalized)
    from scipy.spatial.distance import pdist

    pairwise_dists = pdist(samples)
    min_pairwise_dist = pairwise_dists.min() if len(pairwise_dists) > 0 else 0.0

    # Check coverage uniformity (project to 1D and bin)
    # Use first dimension as representative
    first_dim = samples[:, 0]
    hist, _ = np.histogram(first_dim, bins=n_bins, range=(first_dim.min(), first_dim.max()))
    expected_per_bin = n_samples / n_bins
    coverage_uniformity = 1.0 - np.std(hist) / expected_per_bin if expected_per_bin > 0 else 0.0

    return {
        "discrepancy": float(discrepancy),
        "min_pairwise_dist": float(min_pairwise_dist),
        "coverage_uniformity": float(coverage_uniformity),
    }


def generate_and_validate_variations(
    model: TokenToRolloutVAE,
    tokens: Dict[str, torch.Tensor],
    n_variations: int = 100,
    sobol_seed: int = 42,
    device: str = "cuda",
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Generate Sobol variations and validate coverage.

    Convenience function that combines generation and validation.

    Args:
        model: Trained TokenToRolloutVAE
        tokens: Dict mapping token keys to indices [1]
        n_variations: Number of variations to generate
        sobol_seed: Seed for Sobol sequence
        device: Device for computation

    Returns:
        Tuple of (variations dict, metrics dict)
    """
    # Generate variations
    variations = generate_sobol_variations(
        model, tokens, n_variations, sobol_seed, device
    )

    # Validate coverage
    metrics = validate_sobol_coverage(variations["z"])

    return variations, metrics
