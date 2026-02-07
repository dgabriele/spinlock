#!/usr/bin/env python3
"""Analyze per-timestep error accumulation in MNO rollouts.

This script loads a trained MNO checkpoint and evaluates how error
accumulates over the 32-step rollout. Helps diagnose if the bottleneck
is early prediction or late-timestep compounding.
"""

import argparse
import h5py
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from spinlock.mno.backbone import NOABackbone
from spinlock.mno.cno_replay import CNOReplayer


def load_checkpoint(checkpoint_path: str, config_path: str, device: torch.device):
    """Load MNO from checkpoint.

    Returns:
        model: Loaded NOABackbone model
        config: Full experiment config
        uses_tokens: Whether this model uses token conditioning
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load checkpoint to CPU first to avoid OOM during loading
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Detect if model uses token conditioning by checking for token_embedding in weights
    uses_tokens = any(k.startswith('token_embedding.') for k in state_dict.keys())

    # Create model
    model_config = config['model']

    if uses_tokens:
        # Extract codebook sizes from token_embedding weights in checkpoint
        codebook_sizes = []
        idx = 0
        while f'token_embedding.embeddings.{idx}.weight' in state_dict:
            emb_weight = state_dict[f'token_embedding.embeddings.{idx}.weight']
            codebook_size = emb_weight.shape[0]  # [codebook_size, embed_dim]
            codebook_sizes.append(codebook_size)
            idx += 1
        num_tokens = len(codebook_sizes)

        print(f"  Detected token conditioning: {num_tokens} tokens")

        model = NOABackbone(
            spatial_dim=model_config['spatial_dim'],
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            base_channels=model_config['base_channels'],
            encoder_levels=model_config['encoder_levels'],
            modes=model_config['modes'],
            afno_blocks=model_config['afno_blocks'],
            token_conditioning=True,
            token_embed_dim=model_config.get('token_embed_dim', 64),
            num_tokens=num_tokens,
            codebook_sizes=codebook_sizes,
        )
    else:
        print(f"  No token conditioning detected")

        model = NOABackbone(
            spatial_dim=model_config['spatial_dim'],
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            base_channels=model_config['base_channels'],
            encoder_levels=model_config['encoder_levels'],
            modes=model_config['modes'],
            afno_blocks=model_config['afno_blocks'],
            token_conditioning=False,
        )

    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config, uses_tokens


def compute_per_timestep_error(
    model: NOABackbone,
    cno_replayer: CNOReplayer,
    dataset_path: str,
    n_samples: int = 100,
    timesteps: int = 32,
    device: torch.device = torch.device('cuda'),
    uses_tokens: bool = False,
    oracle_tokens: torch.Tensor = None,
):
    """Compute MSE at each timestep of the rollout.

    Args:
        model: Trained NOABackbone model
        cno_replayer: CNO replayer for ground truth
        dataset_path: Path to dataset HDF5
        n_samples: Number of validation samples to evaluate
        timesteps: Number of rollout timesteps
        device: Computation device
        uses_tokens: Whether model uses token conditioning
        oracle_tokens: Pre-computed oracle tokens (required if uses_tokens=True)

    Returns:
        per_timestep_mse: [timesteps] array of MSE at each timestep
        per_sample_errors: [n_samples, timesteps] array of errors per sample
    """
    # Load dataset
    dataset = h5py.File(dataset_path, 'r')

    # If using oracle tokens, they correspond to first N samples of dataset
    # Use those samples instead of trying to use validation split
    if uses_tokens and oracle_tokens is not None:
        n_available = len(oracle_tokens)
        # Use last 10% of available tokens as validation
        val_start = int(0.9 * n_available)
        val_indices = np.arange(val_start, min(val_start + n_samples, n_available))
        print(f"  Using oracle token indices {val_start} to {val_indices[-1]} as validation")
    else:
        # Use validation split from full dataset (last 10%)
        n_total = len(dataset['inputs/fields'])
        val_start = int(0.9 * n_total)
        val_indices = np.arange(val_start, min(val_start + n_samples, n_total))

    per_sample_errors = []

    with torch.no_grad():
        for idx in val_indices:
            # Load IC and parameters
            ic = torch.tensor(
                dataset['inputs/fields'][idx],  # [M, H, W]
                device=device,
                dtype=torch.float32,
            )
            # Take first channel only
            ic = ic[0:1].unsqueeze(0)  # [1, 1, H, W]

            # Load parameters as 1D vector
            params_vector = np.array(dataset['parameters/params'][idx], dtype=np.float64)

            # Generate ground truth trajectory
            target_traj = cno_replayer.rollout(
                params_vector=params_vector,
                ic=ic,
                timesteps=timesteps,
                num_realizations=1,
                return_all_steps=True,
            )  # [1, T+1, 1, H, W]

            # Generate predicted trajectory
            if uses_tokens:
                if oracle_tokens is None:
                    raise ValueError("Model uses token conditioning but no oracle_tokens provided")
                tokens = oracle_tokens[idx:idx+1].to(device)

                # Debug: print first sample
                if len(per_sample_errors) == 0:
                    print(f"\n  Debug first sample:")
                    print(f"    IC shape: {ic.shape}")
                    print(f"    Tokens shape: {tokens.shape}")
                    print(f"    Tokens values: {tokens[0].tolist()}")
                    print(f"    model.token_conditioning: {model.token_conditioning}")

                pred_traj = model.rollout(
                    ic,
                    steps=timesteps,
                    return_all_steps=True,
                    tokens=tokens,
                )  # [1, T+1, 1, H, W]
            else:
                pred_traj = model.rollout(
                    ic,
                    steps=timesteps,
                    return_all_steps=True,
                )  # [1, T+1, 1, H, W]

            # Compute per-timestep MSE (skip IC at t=0)
            sample_errors = []
            for t in range(1, timesteps + 1):
                mse = torch.mean((pred_traj[:, t] - target_traj[:, t]) ** 2).item()
                sample_errors.append(mse)

            per_sample_errors.append(sample_errors)

    dataset.close()

    per_sample_errors = np.array(per_sample_errors)  # [n_samples, timesteps]
    per_timestep_mse = np.mean(per_sample_errors, axis=0)  # [timesteps]

    return per_timestep_mse, per_sample_errors


def plot_error_accumulation(
    per_timestep_mse: np.ndarray,
    per_sample_errors: np.ndarray,
    output_path: str,
    title: str = "Rollout Error Accumulation",
):
    """Plot error accumulation over timesteps."""
    timesteps = len(per_timestep_mse)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean error with std bands
    mean_error = per_timestep_mse
    std_error = np.std(per_sample_errors, axis=0)

    ax1.plot(range(1, timesteps + 1), mean_error, 'b-', linewidth=2, label='Mean MSE')
    ax1.fill_between(
        range(1, timesteps + 1),
        mean_error - std_error,
        mean_error + std_error,
        alpha=0.3,
        label='±1 std'
    )
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title(f'{title}\nMean Error ± Std', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Individual sample trajectories (first 20)
    for i in range(min(20, len(per_sample_errors))):
        ax2.plot(range(1, timesteps + 1), per_sample_errors[i], alpha=0.3, linewidth=1)
    ax2.plot(range(1, timesteps + 1), mean_error, 'r-', linewidth=3, label='Mean')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title(f'{title}\nIndividual Samples', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")
    plt.close()


def analyze_growth_rate(per_timestep_mse: np.ndarray):
    """Analyze whether error grows linearly, exponentially, or plateaus."""
    timesteps = len(per_timestep_mse)
    t = np.arange(1, timesteps + 1)

    # Fit linear model: error = a * t + b
    linear_fit = np.polyfit(t, per_timestep_mse, 1)
    linear_pred = np.polyval(linear_fit, t)
    linear_r2 = 1 - np.sum((per_timestep_mse - linear_pred) ** 2) / np.sum((per_timestep_mse - np.mean(per_timestep_mse)) ** 2)

    # Fit exponential model: error = a * exp(b * t)
    log_error = np.log(per_timestep_mse + 1e-8)
    exp_fit = np.polyfit(t, log_error, 1)
    exp_pred = np.exp(np.polyval(exp_fit, t))
    exp_r2 = 1 - np.sum((per_timestep_mse - exp_pred) ** 2) / np.sum((per_timestep_mse - np.mean(per_timestep_mse)) ** 2)

    # Determine growth type
    if exp_r2 > linear_r2 + 0.05:
        growth_type = "EXPONENTIAL"
    elif linear_r2 > 0.8:
        growth_type = "LINEAR"
    else:
        growth_type = "IRREGULAR"

    return {
        'growth_type': growth_type,
        'linear_r2': linear_r2,
        'exp_r2': exp_r2,
        'linear_slope': linear_fit[0],
        'exp_rate': exp_fit[0],
    }


def compute_frequency_spectrum(
    model: NOABackbone,
    cno_replayer: CNOReplayer,
    dataset_path: str,
    n_samples: int = 100,
    timesteps: int = 256,
    device: torch.device = torch.device('cuda'),
    uses_tokens: bool = False,
    oracle_tokens: torch.Tensor = None,
):
    """Compute frequency-domain residual spectrum.

    Returns:
        magnitude_spectrum: [H//2+1, W] array (2D FFT magnitude)
    """
    dataset = h5py.File(dataset_path, 'r')

    # Use same validation split logic as compute_per_timestep_error
    if uses_tokens and oracle_tokens is not None:
        n_available = len(oracle_tokens)
        val_start = int(0.9 * n_available)
        val_indices = np.arange(val_start, min(val_start + n_samples, n_available))
    else:
        n_total = len(dataset['inputs/fields'])
        val_start = int(0.9 * n_total)
        val_indices = np.arange(val_start, min(val_start + n_samples, n_total))

    spectrum_accumulator = None

    with torch.no_grad():
        for idx in val_indices:
            # Load IC and parameters
            ic = torch.tensor(
                dataset['inputs/fields'][idx],
                device=device,
                dtype=torch.float32,
            )
            ic = ic[0:1].unsqueeze(0)  # [1, 1, H, W]

            params_vector = np.array(dataset['parameters/params'][idx], dtype=np.float64)

            # Generate trajectories
            if uses_tokens:
                if oracle_tokens is None:
                    raise ValueError("Model uses token conditioning but no oracle_tokens provided")
                tokens = oracle_tokens[idx:idx+1].to(device)
                pred_traj = model.rollout(ic, steps=timesteps, return_all_steps=True, tokens=tokens)
            else:
                pred_traj = model.rollout(ic, steps=timesteps, return_all_steps=True)

            target_traj = cno_replayer.rollout(
                params_vector=params_vector,
                ic=ic,
                timesteps=timesteps,
                num_realizations=1,
                return_all_steps=True,
            )

            # Compute residuals (skip IC at t=0)
            residuals = pred_traj[:, 1:] - target_traj[:, 1:]  # [1, T, C, H, W]

            # 2D FFT over spatial dimensions
            fft_residuals = torch.fft.rfft2(residuals, dim=(-2, -1))
            magnitude = torch.abs(fft_residuals).mean(dim=(0, 1, 2))  # [H//2+1, W]

            if spectrum_accumulator is None:
                spectrum_accumulator = magnitude.cpu().numpy()
            else:
                spectrum_accumulator += magnitude.cpu().numpy()

    dataset.close()
    return spectrum_accumulator / len(val_indices)


def plot_frequency_spectrum(
    spectrum: np.ndarray,
    output_path: str,
    title: str = "Frequency-Domain Residual Spectrum",
):
    """Plot frequency spectrum as log-scale heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(
        np.log10(spectrum + 1e-10),
        aspect='auto',
        cmap='hot',
        origin='lower',
    )
    plt.colorbar(label='log10(Magnitude)')
    plt.xlabel('Frequency (width)')
    plt.ylabel('Frequency (height)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved frequency spectrum: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config')
    parser.add_argument('--dataset', type=str, default='datasets/100k_full_features.h5', help='Dataset path')
    parser.add_argument('--substrate-config', type=str, default='configs/experiments/local_100k_optimized.yaml', help='Substrate config')
    parser.add_argument('--oracle-tokens', type=str, help='Optional oracle tokens path')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of validation samples')
    parser.add_argument('--timesteps', type=int, default=32, help='Number of rollout timesteps')
    parser.add_argument('--output', type=str, default='/tmp/rollout_error_analysis.png', help='Output plot path')
    parser.add_argument('--frequency-analysis', action='store_true', help='Compute frequency-domain residual spectrum')
    parser.add_argument('--output-frequency', type=str, default='/tmp/frequency_spectrum.png', help='Frequency spectrum plot path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, config, uses_tokens = load_checkpoint(args.checkpoint, args.config, device)
    print(f"  ✓ Model loaded")

    # Load substrate replayer
    print(f"\nLoading substrate replayer: {args.substrate_config}")
    replayer = CNOReplayer.from_config(args.substrate_config, device=str(device))
    print(f"  ✓ Substrate replayer loaded")

    # Load oracle tokens if model uses them
    oracle_tokens = None
    if uses_tokens:
        if args.oracle_tokens:
            print(f"\nLoading oracle tokens: {args.oracle_tokens}")
            token_file = h5py.File(args.oracle_tokens, 'r')
            oracle_tokens = torch.tensor(token_file['tokens'][:], dtype=torch.long)
            token_file.close()
            print(f"  ✓ Loaded {len(oracle_tokens)} oracle tokens")
        else:
            print(f"\n⚠ WARNING: Model uses token conditioning but no oracle_tokens provided")
            print(f"  Analysis will fail. Please provide --oracle-tokens argument.")
            return
    elif args.oracle_tokens:
        print(f"\n⚠ Note: Oracle tokens provided but model doesn't use token conditioning")

    # Compute per-timestep errors
    print(f"\nComputing per-timestep errors on {args.n_samples} samples...")
    per_timestep_mse, per_sample_errors = compute_per_timestep_error(
        model=model,
        cno_replayer=replayer,
        dataset_path=args.dataset,
        n_samples=args.n_samples,
        timesteps=args.timesteps,
        device=device,
        uses_tokens=uses_tokens,
        oracle_tokens=oracle_tokens,
    )

    # Print statistics
    print(f"\n{'='*70}")
    print(f"Rollout Error Analysis")
    print(f"{'='*70}")
    print(f"\nPer-Timestep MSE:")
    print(f"  t=1:    {per_timestep_mse[0]:.6f}")
    print(f"  t=8:    {per_timestep_mse[7]:.6f} ({per_timestep_mse[7]/per_timestep_mse[0]:.2f}x)")
    print(f"  t=16:   {per_timestep_mse[15]:.6f} ({per_timestep_mse[15]/per_timestep_mse[0]:.2f}x)")
    print(f"  t=24:   {per_timestep_mse[23]:.6f} ({per_timestep_mse[23]/per_timestep_mse[0]:.2f}x)")
    print(f"  t=32:   {per_timestep_mse[31]:.6f} ({per_timestep_mse[31]/per_timestep_mse[0]:.2f}x)")
    print(f"\nMean over all timesteps: {np.mean(per_timestep_mse):.6f}")

    # Analyze growth rate
    growth_stats = analyze_growth_rate(per_timestep_mse)
    print(f"\nError Growth Analysis:")
    print(f"  Growth Type: {growth_stats['growth_type']}")
    print(f"  Linear R²:   {growth_stats['linear_r2']:.4f}")
    print(f"  Exponential R²: {growth_stats['exp_r2']:.4f}")
    if growth_stats['growth_type'] == 'LINEAR':
        print(f"  Linear Slope: {growth_stats['linear_slope']:.6f} (error/timestep)")
    elif growth_stats['growth_type'] == 'EXPONENTIAL':
        print(f"  Exponential Rate: {growth_stats['exp_rate']:.6f} (growth factor)")

    # Plot
    print(f"\nGenerating plots...")
    model_name = Path(args.checkpoint).parent.name
    title_suffix = " (with tokens)" if uses_tokens else " (no tokens)"
    plot_error_accumulation(
        per_timestep_mse=per_timestep_mse,
        per_sample_errors=per_sample_errors,
        output_path=args.output,
        title=model_name + title_suffix,
    )

    # Frequency spectrum analysis (if requested)
    if args.frequency_analysis:
        print(f"\nComputing frequency-domain residual spectrum on {args.n_samples} samples...")
        frequency_spectrum = compute_frequency_spectrum(
            model=model,
            cno_replayer=replayer,
            dataset_path=args.dataset,
            n_samples=args.n_samples,
            timesteps=args.timesteps,
            device=device,
            uses_tokens=uses_tokens,
            oracle_tokens=oracle_tokens,
        )

        # Save raw spectrum data
        spectrum_data_path = args.output_frequency.replace('.png', '.npy')
        np.save(spectrum_data_path, frequency_spectrum)
        print(f"  Saved raw spectrum data: {spectrum_data_path}")

        # Plot frequency spectrum
        plot_frequency_spectrum(
            spectrum=frequency_spectrum,
            output_path=args.output_frequency,
            title=f"Frequency Spectrum - {model_name}{title_suffix}",
        )

        # Print spectrum statistics
        print(f"\nFrequency Spectrum Statistics:")
        print(f"  Mean magnitude: {frequency_spectrum.mean():.4e}")
        print(f"  Max magnitude:  {frequency_spectrum.max():.4e}")
        print(f"  Min magnitude:  {frequency_spectrum.min():.4e}")
        print(f"  Std magnitude:  {frequency_spectrum.std():.4e}")

    print(f"\n✓ Analysis complete")


if __name__ == '__main__':
    main()
