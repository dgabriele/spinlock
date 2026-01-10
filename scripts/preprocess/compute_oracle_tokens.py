#!/usr/bin/env python3
"""Precompute oracle VQ tokens for NOA training dataset.

Given a dataset of CNO parameters and initial conditions, this script:
1. Generates ground truth trajectories using CNO replayer
2. Extracts multi-modal features from trajectories
3. Computes VQ tokens from features using frozen VQ-VAE
4. Saves tokens to HDF5 file for efficient loading during training

Usage:
    python scripts/preprocess/compute_oracle_tokens.py \
        --dataset datasets/100k_full_features.h5 \
        --cno-config configs/experiments/local_100k_optimized.yaml \
        --vqvae-checkpoint checkpoints/production/100k_full_features/vqvae_best.pt \
        --output datasets/100k_oracle_tokens.h5 \
        --batch-size 16
"""

import argparse
import h5py
import torch
import yaml
from tqdm import tqdm
from pathlib import Path

from spinlock.operators.cno import CNOReplayer
from spinlock.noa.feature_extraction import AlignedFeatureExtractor
from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE


def load_cno_replayer(config_path: str, device: torch.device) -> CNOReplayer:
    """Load CNO replayer from config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    replayer = CNOReplayer.from_config(config)
    replayer = replayer.to(device)
    replayer.eval()
    return replayer


def main():
    parser = argparse.ArgumentParser(description="Precompute oracle VQ tokens for NOA training")
    parser.add_argument("--dataset", type=str, required=True, help="Path to NOA dataset HDF5")
    parser.add_argument("--cno-config", type=str, required=True, help="CNO configuration YAML")
    parser.add_argument("--vqvae-checkpoint", type=str, required=True, help="VQ-VAE checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path for tokens")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples to process (default: all)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = h5py.File(args.dataset, 'r')
    n_samples_total = len(dataset["initial_conditions"])
    n_samples = args.n_samples if args.n_samples is not None else n_samples_total
    n_samples = min(n_samples, n_samples_total)
    print(f"  {n_samples} samples to process (out of {n_samples_total} total)")

    # Load CNO replayer
    print(f"Loading CNO replayer: {args.cno_config}")
    replayer = load_cno_replayer(args.cno_config, device)

    # Load VQ-VAE and feature extractor
    print(f"Loading VQ-VAE: {args.vqvae_checkpoint}")
    vqvae = CategoricalHierarchicalVQVAE.from_checkpoint(args.vqvae_checkpoint)
    vqvae = vqvae.to(device)
    vqvae.eval()

    print(f"Loading feature extractor from VQ-VAE checkpoint")
    feature_extractor = AlignedFeatureExtractor.from_checkpoint(args.vqvae_checkpoint)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Determine token shape
    print("Determining token shape...")
    with torch.no_grad():
        dummy_ic = torch.randn(1, 1, 64, 64, device=device)
        dummy_params_dict = {}
        for key in dataset["parameters"].keys():
            param_data = dataset["parameters"][key]
            # Handle scalar vs array parameters
            if param_data.shape == (n_samples_total,):
                dummy_params_dict[key] = torch.tensor([param_data[0]], device=device, dtype=torch.float32)
            else:
                dummy_params_dict[key] = torch.tensor([param_data[0]], device=device, dtype=torch.float32)

        dummy_traj = replayer.rollout(dummy_ic, dummy_params_dict)
        dummy_features = feature_extractor(dummy_traj, ic=dummy_ic)
        dummy_tokens = vqvae.get_tokens(dummy_features)
        num_tokens = dummy_tokens.shape[1]

    print(f"Token shape: [{num_tokens}] per sample")

    # Prepare output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = h5py.File(output_path, 'w')
    tokens_dataset = output_file.create_dataset(
        "tokens",
        shape=(n_samples, num_tokens),
        dtype='int32',
        compression='gzip',
    )

    # Process in batches
    print(f"Computing tokens for {n_samples} samples...")
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_samples, args.batch_size)):
            end_idx = min(start_idx + args.batch_size, n_samples)
            batch_size = end_idx - start_idx

            # Load batch of initial conditions
            ics = torch.tensor(
                dataset["initial_conditions"][start_idx:end_idx],
                device=device,
                dtype=torch.float32,
            )  # [B, 1, 64, 64]

            # Load batch of parameters
            params_dict = {}
            for key in dataset["parameters"].keys():
                param_data = dataset["parameters"][key][start_idx:end_idx]
                params_dict[key] = torch.tensor(param_data, device=device, dtype=torch.float32)

            # Generate trajectories using CNO
            try:
                trajectories = replayer.rollout(ics, params_dict)  # [B, T, 1, 64, 64]
            except Exception as e:
                print(f"\nError generating trajectory for batch {start_idx}-{end_idx}: {e}")
                print(f"IC shape: {ics.shape}")
                print(f"Params: {list(params_dict.keys())}")
                raise

            # Extract features
            try:
                features = feature_extractor(trajectories, ic=ics)  # [B, feature_dim]
            except Exception as e:
                print(f"\nError extracting features for batch {start_idx}-{end_idx}: {e}")
                print(f"Trajectory shape: {trajectories.shape}")
                raise

            # Get tokens
            try:
                tokens = vqvae.get_tokens(features)  # [B, num_tokens]
            except Exception as e:
                print(f"\nError computing tokens for batch {start_idx}-{end_idx}: {e}")
                print(f"Features shape: {features.shape}")
                raise

            # Save to HDF5
            tokens_dataset[start_idx:end_idx] = tokens.cpu().numpy()

            # Clear GPU cache periodically
            if (end_idx % (args.batch_size * 10)) == 0:
                torch.cuda.empty_cache()

    # Save metadata
    output_file.attrs['num_samples'] = n_samples
    output_file.attrs['num_tokens'] = num_tokens
    output_file.attrs['vqvae_checkpoint'] = args.vqvae_checkpoint
    output_file.attrs['dataset_source'] = args.dataset

    output_file.close()
    dataset.close()

    print(f"\n✓ Tokens saved to: {output_path}")
    print(f"  Shape: [{n_samples}, {num_tokens}]")
    print(f"  Compression: gzip")


if __name__ == "__main__":
    main()
