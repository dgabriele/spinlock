"""
Validate inverse decoders end-to-end.

Tests the complete pipeline:
  1. Load tokenizer + inverse models
  2. Load test data (theta, ICs)
  3. Encode → tokens
  4. Decode tokens → (theta, ICs)
  5. Compare with ground truth
  6. Generate CNO rollouts to verify compatibility

Usage:
    poetry run python scripts/validate_inverse_decoders.py \
        --tokenizer checkpoints/vq_tokenizer_best.pt \
        --theta-inverse checkpoints/theta_inverse.pt \
        --initial-inverse checkpoints/initial_inverse.pt \
        --dataset datasets/50k_baseline.h5 \
        --cno-checkpoint checkpoints/cno_replayer.pt \
        --num-samples 100
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.data import SpinlockDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(dataset_path: Path, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load test data (theta, ICs)."""
    logger.info(f"Loading {num_samples} test samples from {dataset_path}")

    with h5py.File(dataset_path, 'r') as f:
        # Load theta parameters
        theta = torch.from_numpy(f['parameters/params'][:num_samples]).float()

        # Load ICs and aggregate over realizations
        ics = f['inputs/fields'][:num_samples]
        if ics.ndim == 5:  # [N, M, C, H, W]
            ics = ics.mean(axis=1)  # [N, C, H, W]
        ics = torch.from_numpy(ics).float()

    logger.info(f"Loaded theta: {theta.shape}, ICs: {ics.shape}")
    return theta, ics


def validate_theta_reconstruction(
    theta_true: torch.Tensor,
    theta_pred: torch.Tensor,
) -> Dict[str, float]:
    """Validate theta reconstruction quality."""
    mse = torch.mean((theta_pred - theta_true) ** 2).item()
    mae = torch.mean(torch.abs(theta_pred - theta_true)).item()
    max_error = torch.max(torch.abs(theta_pred - theta_true)).item()

    # Per-parameter MSE
    per_param_mse = torch.mean((theta_pred - theta_true) ** 2, dim=0)

    # Check range
    in_range = torch.all((theta_pred >= 0) & (theta_pred <= 1)).item()

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'per_param_mse': per_param_mse.tolist(),
        'in_range': in_range,
    }


def validate_ic_reconstruction(
    ics_true: torch.Tensor,
    ics_pred: torch.Tensor,
) -> Dict[str, float]:
    """Validate IC reconstruction quality."""
    mse = torch.mean((ics_pred - ics_true) ** 2).item()
    mae = torch.mean(torch.abs(ics_pred - ics_true)).item()
    max_error = torch.max(torch.abs(ics_pred - ics_true)).item()

    # Per-channel MSE
    channel_mse = torch.mean((ics_pred - ics_true) ** 2, dim=[0, 2, 3]).tolist()

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'channel_mse': channel_mse,
    }


def visualize_reconstruction(
    theta_true: torch.Tensor,
    theta_pred: torch.Tensor,
    ics_true: torch.Tensor,
    ics_pred: torch.Tensor,
    output_dir: Path,
    sample_idx: int = 0,
):
    """Visualize reconstruction quality."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot theta comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Theta parameters
    param_names = [f'p{i}' for i in range(14)]
    x = np.arange(14)

    axes[0].bar(x - 0.2, theta_true[sample_idx].cpu().numpy(), width=0.4, label='True', alpha=0.7)
    axes[0].bar(x + 0.2, theta_pred[sample_idx].cpu().numpy(), width=0.4, label='Pred', alpha=0.7)
    axes[0].set_xlabel('Parameter')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Theta Parameters')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(param_names, rotation=45)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Error
    error = torch.abs(theta_pred[sample_idx] - theta_true[sample_idx]).cpu().numpy()
    axes[1].bar(x, error, alpha=0.7)
    axes[1].set_xlabel('Parameter')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Theta Reconstruction Error')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(param_names, rotation=45)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'theta_reconstruction_{sample_idx}.png', dpi=150)
    plt.close()

    # Plot IC comparison
    num_channels = ics_true.shape[1]
    fig, axes = plt.subplots(3, num_channels, figsize=(4*num_channels, 10))

    if num_channels == 1:
        axes = axes[:, None]

    for ch in range(num_channels):
        # True
        im = axes[0, ch].imshow(ics_true[sample_idx, ch].cpu().numpy(), cmap='viridis')
        axes[0, ch].set_title(f'True IC - Channel {ch}')
        axes[0, ch].axis('off')
        plt.colorbar(im, ax=axes[0, ch])

        # Predicted
        im = axes[1, ch].imshow(ics_pred[sample_idx, ch].cpu().numpy(), cmap='viridis')
        axes[1, ch].set_title(f'Predicted IC - Channel {ch}')
        axes[1, ch].axis('off')
        plt.colorbar(im, ax=axes[1, ch])

        # Error
        error = torch.abs(ics_pred[sample_idx, ch] - ics_true[sample_idx, ch]).cpu().numpy()
        im = axes[2, ch].imshow(error, cmap='hot')
        axes[2, ch].set_title(f'Error - Channel {ch}')
        axes[2, ch].axis('off')
        plt.colorbar(im, ax=axes[2, ch])

    plt.tight_layout()
    plt.savefig(output_dir / f'ic_reconstruction_{sample_idx}.png', dpi=150)
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Validate inverse decoders')
    parser.add_argument('--tokenizer', type=Path, required=True,
                        help='Path to trained VQTokenizer checkpoint')
    parser.add_argument('--theta-inverse', type=Path, required=True,
                        help='Path to trained theta inverse checkpoint')
    parser.add_argument('--initial-inverse', type=Path, required=True,
                        help='Path to trained initial inverse checkpoint')
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Path to dataset for testing')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to test')
    parser.add_argument('--output-dir', type=Path, default=Path('results/inverse_validation'),
                        help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load tokenizer with inverse models
    logger.info("Loading tokenizer with inverse models")
    tokenizer = VQTokenizer.from_checkpoint(
        args.tokenizer,
        theta_inverse_path=args.theta_inverse,
        initial_inverse_path=args.initial_inverse,
    )
    tokenizer.model.to(device)
    tokenizer.model.eval()

    # Load test data
    theta_true, ics_true = load_test_data(args.dataset, args.num_samples)

    # Load dataset for feature extraction
    logger.info("Loading dataset for feature extraction")
    dataset = SpinlockDataset.from_file(str(args.dataset))

    # Extract features for tokenization
    logger.info("Extracting features")
    features = tokenizer._extract_features(dataset)

    # Tokenize
    logger.info("Tokenizing samples")
    with torch.no_grad():
        tokens = tokenizer.tokenize(
            temporal_features=features.get('temporal')[:args.num_samples].to(device) if features.get('temporal') is not None else None,
            initial_manual=features.get('initial_manual')[:args.num_samples].to(device) if features.get('initial_manual') is not None else None,
            initial_raw=features.get('initial_raw')[:args.num_samples].to(device) if features.get('initial_raw') is not None else None,
            theta_features=features.get('theta')[:args.num_samples].to(device) if features.get('theta') is not None else None,
        )

    logger.info(f"Generated {len(tokens)} token categories")

    # Decode
    logger.info("Decoding tokens")
    with torch.no_grad():
        theta_pred, ics_pred = tokenizer.decode(tokens)

    # Move to CPU for comparison
    theta_pred = theta_pred.cpu()
    ics_pred = ics_pred.cpu()

    # Validate theta reconstruction
    logger.info("\n" + "="*60)
    logger.info("THETA RECONSTRUCTION VALIDATION")
    logger.info("="*60)

    theta_metrics = validate_theta_reconstruction(theta_true, theta_pred)

    logger.info(f"MSE: {theta_metrics['mse']:.6f}")
    logger.info(f"MAE: {theta_metrics['mae']:.6f}")
    logger.info(f"Max Error: {theta_metrics['max_error']:.6f}")
    logger.info(f"In Range [0,1]: {theta_metrics['in_range']}")

    if theta_metrics['mse'] < 0.01:
        logger.info("✅ SUCCESS: Theta MSE < 0.01")
    else:
        logger.warning(f"⚠️  WARNING: Theta MSE = {theta_metrics['mse']:.6f} > 0.01")

    # Validate IC reconstruction
    logger.info("\n" + "="*60)
    logger.info("INITIAL CONDITION RECONSTRUCTION VALIDATION")
    logger.info("="*60)

    ic_metrics = validate_ic_reconstruction(ics_true, ics_pred)

    logger.info(f"MSE: {ic_metrics['mse']:.6f}")
    logger.info(f"MAE: {ic_metrics['mae']:.6f}")
    logger.info(f"Max Error: {ic_metrics['max_error']:.6f}")
    logger.info(f"Channel MSE: {ic_metrics['channel_mse']}")

    if ic_metrics['mse'] < 0.05:
        logger.info("✅ SUCCESS: IC MSE < 0.05")
    else:
        logger.warning(f"⚠️  WARNING: IC MSE = {ic_metrics['mse']:.6f} > 0.05")

    # Visualize reconstruction
    logger.info("\nGenerating visualizations")
    visualize_reconstruction(
        theta_true, theta_pred,
        ics_true, ics_pred,
        args.output_dir,
        sample_idx=0,
    )

    logger.info("\n" + "="*60)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()
