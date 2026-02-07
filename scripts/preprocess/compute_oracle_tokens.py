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
from tqdm import tqdm
from pathlib import Path

from spinlock.mno import CNOReplayer, AlignedFeatureExtractor
from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE
from spinlock.encoding import CategoricalVQVAEConfig


def load_vqvae(checkpoint_path: str, device: torch.device):
    """Load VQ-VAE from checkpoint.

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint file
        device: Device to load on

    Returns:
        Tuple of (vqvae_model, config)
    """
    from pathlib import Path

    path = Path(checkpoint_path)
    if path.is_dir():
        ckpt_file = path / "best_model.pt"
    else:
        ckpt_file = path

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    # Handle torch.compile prefix if present
    has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if has_orig_mod:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Check if wrapped with VQVAEWithInitial
    is_wrapped = any(k.startswith("initial_encoder.") for k in state_dict.keys())

    # Create VQ-VAE config
    # Handle both old format (config) and new format (model_config)
    model_config = checkpoint.get("model_config", config)
    vqvae_config = CategoricalVQVAEConfig(
        input_dim=model_config["input_dim"],
        group_indices=model_config["group_indices"],
        group_embedding_dim=model_config.get("group_embedding_dim", config.get("group_embedding_dim")),
        group_hidden_dim=model_config.get("group_hidden_dim", config.get("group_hidden_dim")),
        levels=model_config.get("levels", config.get("levels")),
    )

    if is_wrapped:
        # Extract inner VQ-VAE weights
        vqvae_state_dict = {
            k.replace("vqvae.", ""): v
            for k, v in state_dict.items()
            if k.startswith("vqvae.")
        }
        model = CategoricalHierarchicalVQVAE(vqvae_config)
        model.load_state_dict(vqvae_state_dict)
    else:
        # Load raw CategoricalHierarchicalVQVAE
        model = CategoricalHierarchicalVQVAE(vqvae_config)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Return model_config for consistency with new checkpoint format
    return model, model_config


def load_cno_replayer(config_path: str, device: torch.device) -> CNOReplayer:
    """Load CNO replayer from config."""
    replayer = CNOReplayer.from_config(
        config_path=config_path,
        device=str(device),
        cache_size=8,
    )
    return replayer


def main():
    parser = argparse.ArgumentParser(description="Precompute oracle VQ tokens for NOA training")
    parser.add_argument("--dataset", type=str, required=True, help="Path to NOA dataset HDF5")
    parser.add_argument("--config", type=str, required=True, help="Substrate configuration YAML")
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

    # Determine dataset structure
    if "inputs" in dataset and "fields" in dataset["inputs"]:
        # Production dataset format (used for VQ-VAE and NOA training)
        ic_key = "inputs/fields"
        params_key = "parameters/params"
    elif "initial_conditions" in dataset:
        # Alternative format (not commonly used but supported)
        ic_key = "initial_conditions"
        params_key = "parameters"
    else:
        available_keys = list(dataset.keys())
        raise ValueError(f"Unknown dataset format. Available keys: {available_keys}")

    n_samples_total = len(dataset[ic_key])
    n_samples = args.n_samples if args.n_samples is not None else n_samples_total
    n_samples = min(n_samples, n_samples_total)
    print(f"  {n_samples} samples to process (out of {n_samples_total} total)")

    # Load substrate replayer
    print(f"Loading substrate replayer: {args.config}")
    replayer = load_cno_replayer(args.config, device)

    # Load VQ-VAE and feature extractor
    print(f"Loading VQ-VAE: {args.vqvae_checkpoint}")
    vqvae, vqvae_config = load_vqvae(args.vqvae_checkpoint, device)
    print(f"  ✓ VQ-VAE loaded (input_dim={vqvae_config['input_dim']})")

    print(f"Loading feature extractor from VQ-VAE checkpoint")
    feature_extractor = AlignedFeatureExtractor.from_checkpoint(
        args.vqvae_checkpoint,
        device=str(device)
    )
    print(f"  ✓ Feature extractor loaded")

    # Determine token shape and timesteps
    print("Determining token shape...")
    timesteps = 32  # Default from training config
    with torch.no_grad():
        # Load first IC and parameter vector
        ic_data = dataset[ic_key][0]
        if ic_data.ndim == 3:
            # Format: [M, H, W] - take first realization
            dummy_ic = torch.tensor(ic_data[0, :, :], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        else:
            # Format: [H, W]
            dummy_ic = torch.tensor(ic_data, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Load first parameter vector: [d,]
        dummy_params_numpy = dataset[params_key][0]

        # Generate test trajectory
        dummy_traj = replayer.rollout(
            params_vector=dummy_params_numpy,
            ic=dummy_ic,
            timesteps=timesteps,
            num_realizations=1,
            return_all_steps=True,
        )
        # AlignedFeatureExtractor returns (features, raw_ics) tuple
        dummy_features, _ = feature_extractor(dummy_traj, ic=dummy_ic)
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
            ic_data = dataset[ic_key][start_idx:end_idx]
            # Handle different IC formats
            if ic_data.ndim == 4:
                # Format: [B, M, H, W] - take first realization
                ics = torch.tensor(ic_data[:, 0, :, :], device=device, dtype=torch.float32).unsqueeze(1)  # [B, 1, H, W]
            elif ic_data.ndim == 3:
                # Format: [B, H, W]
                ics = torch.tensor(ic_data, device=device, dtype=torch.float32).unsqueeze(1)  # [B, 1, H, W]
            else:
                raise ValueError(f"Unexpected IC shape: {ic_data.shape}")

            # Load batch of parameter vectors: [B, d] numpy array
            params_numpy = dataset[params_key][start_idx:end_idx]

            # Generate trajectories using CNO (one at a time like in training)
            try:
                trajectories = []
                for b in range(batch_size):
                    traj = replayer.rollout(
                        params_vector=params_numpy[b],
                        ic=ics[b:b+1],
                        timesteps=timesteps,
                        num_realizations=1,
                        return_all_steps=True,
                    )  # [1, T, 1, 64, 64]
                    trajectories.append(traj)
                trajectories = torch.cat(trajectories, dim=0)  # [B, T, 1, 64, 64]
            except Exception as e:
                print(f"\nError generating trajectory for batch {start_idx}-{end_idx}: {e}")
                print(f"IC shape: {ics.shape}")
                print(f"Params shape: {params_numpy.shape}")
                raise

            # Extract features
            try:
                # AlignedFeatureExtractor returns (features, raw_ics) tuple
                features, _ = feature_extractor(trajectories, ic=ics)  # [B, feature_dim]
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
