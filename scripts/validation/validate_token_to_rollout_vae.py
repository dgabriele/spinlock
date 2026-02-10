"""Validation script for Token-to-Rollout VAE.

This script provides comprehensive validation metrics:
1. Reconstruction quality (theta MAE, grid MSE)
2. Latent space quality (KL divergence, latent statistics)
3. Sobol sampling diversity (discrepancy, coverage)
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from spinlock.tokens.rollout_dataset import TokenToRolloutDataset, collate_token_rollout_batch
from spinlock.tokens.rollout_vae import TokenToRolloutVAE
from spinlock.tokens.sobol_sampler import generate_sobol_variations, validate_sobol_coverage


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cuda") -> TokenToRolloutVAE:
    """Load trained VAE from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device for model

    Returns:
        Loaded TokenToRolloutVAE model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract dimensions
    theta_dim = checkpoint["theta_dim"]
    grid_shape = checkpoint["grid_shape"]
    config = checkpoint["config"]

    # Resolve paths (they may be stored as strings in checkpoint)
    vq_checkpoint = Path(config["data"]["vq_checkpoint"])

    # Create model
    model = TokenToRolloutVAE(
        vq_checkpoint=vq_checkpoint,
        theta_dim=theta_dim,
        grid_shape=grid_shape,
        latent_dim=config["model"]["latent_dim"],
        encoder_hidden_dims=config["model"]["encoder"]["hidden_dims"],
        param_decoder_hidden_dims=config["model"]["param_decoder"]["hidden_dims"],
        grid_decoder_hidden_channels=config["model"]["grid_decoder"]["hidden_channels"],
        dropout=config["model"]["encoder"]["dropout"],
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def compute_reconstruction_metrics(
    model: TokenToRolloutVAE,
    dataset: TokenToRolloutDataset,
    n_samples: int = 100,
    device: str = "cuda",
) -> dict:
    """Compute reconstruction quality metrics.

    Args:
        model: Trained VAE model
        dataset: Test dataset
        n_samples: Number of samples to evaluate
        device: Device for computation

    Returns:
        Dictionary of metrics
    """
    model.eval()
    theta_errors = []
    grid_errors = []
    kl_divs = []

    # TODO: Sample random subset
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing reconstruction metrics"):
            sample = dataset[int(idx)]

            # Move to device
            tokens = {k: v.unsqueeze(0).to(device) for k, v in sample["tokens"].items()}
            theta_gt = sample["theta"].unsqueeze(0).to(device)
            grids_gt = sample["grids"].unsqueeze(0).to(device)

            # Forward pass
            outputs = model(tokens)

            # Compute errors
            theta_error = torch.abs(outputs["theta"] - theta_gt).mean().item()
            grid_error = torch.nn.functional.mse_loss(outputs["grids"], grids_gt).item()

            # KL divergence
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()

            theta_errors.append(theta_error)
            grid_errors.append(grid_error)
            kl_divs.append(kl_div)

    return {
        "theta_mae_mean": float(np.mean(theta_errors)),
        "theta_mae_std": float(np.std(theta_errors)),
        "grid_mse_mean": float(np.mean(grid_errors)),
        "grid_mse_std": float(np.std(grid_errors)),
        "kl_div_mean": float(np.mean(kl_divs)),
        "kl_div_std": float(np.std(kl_divs)),
    }


def compute_latent_statistics(
    model: TokenToRolloutVAE,
    dataset: TokenToRolloutDataset,
    n_samples: int = 100,
    device: str = "cuda",
) -> dict:
    """Compute latent space statistics.

    Args:
        model: Trained VAE model
        dataset: Test dataset
        n_samples: Number of samples to evaluate
        device: Device for computation

    Returns:
        Dictionary of latent statistics
    """
    model.eval()
    mus = []
    stds = []

    # TODO: Sample random subset
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing latent statistics"):
            sample = dataset[int(idx)]

            # Move to device
            tokens = {k: v.unsqueeze(0).to(device) for k, v in sample["tokens"].items()}

            # Encode
            mu, logvar = model.encode(tokens)
            std = torch.exp(0.5 * logvar)

            mus.append(mu.squeeze(0).cpu().numpy())
            stds.append(std.squeeze(0).cpu().numpy())

    mus = np.array(mus)  # [n_samples, latent_dim]
    stds = np.array(stds)  # [n_samples, latent_dim]

    return {
        "mu_mean": float(mus.mean()),
        "mu_std": float(mus.std()),
        "std_mean": float(stds.mean()),
        "std_std": float(stds.std()),
        "latent_dim": mus.shape[1],
    }


def validate_sobol_sampling(
    model: TokenToRolloutVAE,
    dataset: TokenToRolloutDataset,
    n_variations: int = 100,
    n_test_samples: int = 10,
    device: str = "cuda",
) -> dict:
    """Validate Sobol sampling diversity.

    Args:
        model: Trained VAE model
        dataset: Test dataset
        n_variations: Number of variations per sample
        n_test_samples: Number of test samples
        device: Device for computation

    Returns:
        Dictionary of Sobol sampling metrics
    """
    model.eval()
    all_discrepancies = []
    all_min_dists = []
    all_coverage_uniformities = []

    # TODO: Sample random test samples
    indices = np.random.choice(len(dataset), size=n_test_samples, replace=False)

    for idx in tqdm(indices, desc="Validating Sobol sampling"):
        sample = dataset[int(idx)]

        # Move to device
        tokens = {k: v.unsqueeze(0).to(device) for k, v in sample["tokens"].items()}

        # Generate variations
        variations = generate_sobol_variations(
            model, tokens, n_variations=n_variations, device=device
        )

        # Validate coverage
        coverage_metrics = validate_sobol_coverage(variations["z"])

        all_discrepancies.append(coverage_metrics["discrepancy"])
        all_min_dists.append(coverage_metrics["min_pairwise_dist"])
        all_coverage_uniformities.append(coverage_metrics["coverage_uniformity"])

    return {
        "discrepancy_mean": float(np.mean(all_discrepancies)),
        "discrepancy_std": float(np.std(all_discrepancies)),
        "min_pairwise_dist_mean": float(np.mean(all_min_dists)),
        "min_pairwise_dist_std": float(np.std(all_min_dists)),
        "coverage_uniformity_mean": float(np.mean(all_coverage_uniformities)),
        "coverage_uniformity_std": float(np.std(all_coverage_uniformities)),
    }


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate Token-to-Rollout VAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        required=True,
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument(
        "--cno-dataset",
        type=Path,
        required=True,
        help="Path to CNO dataset",
    )
    parser.add_argument(
        "--tokenized-dataset",
        type=Path,
        required=True,
        help="Path to tokenized dataset",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for validation",
    )
    parser.add_argument(
        "--n-variations",
        type=int,
        default=100,
        help="Number of Sobol variations per sample",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for computation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="validation_results.json",
        help="Output path for validation results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Token-to-Rollout VAE Validation")
    print("=" * 80)

    # Load model
    print("\nLoading model from checkpoint...")
    model = load_model_from_checkpoint(args.vae_checkpoint, device=args.device)
    print(f"  ✓ Model loaded (latent_dim={model.latent_dim})")

    # Load test dataset (use validation split)
    print("\nLoading test dataset...")
    _, test_dataset = TokenToRolloutDataset.create_splits(
        args.cno_dataset,
        args.tokenized_dataset,
        train_split=0.9,
        seed=42,
    )
    print(f"  ✓ Test dataset loaded (size={len(test_dataset)})")

    # Run validation
    results = {}

    print("\n" + "=" * 80)
    print("1. Reconstruction Quality")
    print("=" * 80)
    recon_metrics = compute_reconstruction_metrics(
        model, test_dataset, n_samples=args.n_samples, device=args.device
    )
    results["reconstruction"] = recon_metrics
    print(f"  Theta MAE: {recon_metrics['theta_mae_mean']:.6f} ± {recon_metrics['theta_mae_std']:.6f}")
    print(f"  Grid MSE:  {recon_metrics['grid_mse_mean']:.6f} ± {recon_metrics['grid_mse_std']:.6f}")
    print(f"  KL Div:    {recon_metrics['kl_div_mean']:.6f} ± {recon_metrics['kl_div_std']:.6f}")

    print("\n" + "=" * 80)
    print("2. Latent Space Quality")
    print("=" * 80)
    latent_stats = compute_latent_statistics(
        model, test_dataset, n_samples=args.n_samples, device=args.device
    )
    results["latent_statistics"] = latent_stats
    print(f"  Latent dim: {latent_stats['latent_dim']}")
    print(f"  Mu:  {latent_stats['mu_mean']:.6f} ± {latent_stats['mu_std']:.6f}")
    print(f"  Std: {latent_stats['std_mean']:.6f} ± {latent_stats['std_std']:.6f}")

    print("\n" + "=" * 80)
    print("3. Sobol Sampling Diversity")
    print("=" * 80)
    sobol_metrics = validate_sobol_sampling(
        model, test_dataset, n_variations=args.n_variations, device=args.device
    )
    results["sobol_sampling"] = sobol_metrics
    print(f"  Discrepancy:       {sobol_metrics['discrepancy_mean']:.6f} ± {sobol_metrics['discrepancy_std']:.6f}")
    print(f"  Min pairwise dist: {sobol_metrics['min_pairwise_dist_mean']:.6f} ± {sobol_metrics['min_pairwise_dist_std']:.6f}")
    print(f"  Coverage:          {sobol_metrics['coverage_uniformity_mean']:.6f} ± {sobol_metrics['coverage_uniformity_std']:.6f}")

    # Save results
    print("\n" + "=" * 80)
    print(f"Saving results to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
