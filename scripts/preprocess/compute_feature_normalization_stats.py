#!/usr/bin/env python3
"""Compute feature normalization statistics for VQ-led training.

Given a trained Stage 1 NOA checkpoint and dataset, this script:
1. Generates NOA rollouts for a sample of trajectories
2. Extracts features using AlignedFeatureExtractor
3. Computes per-group normalization stats (mean, std)
4. Saves stats to a file for use in Stage 2 training

Usage:
    python scripts/preprocess/compute_feature_normalization_stats.py \
        --noa-checkpoint checkpoints/experiments/phase2/exp2f_256step_tbptt/meta_operator_best.pt \
        --vqvae-checkpoint checkpoints/production/100k_3family_v1/best_model.pt \
        --dataset datasets/100k_full_features.h5 \
        --output checkpoints/production/100k_3family_v1/feature_normalization_stats.pt \
        --n-samples 1000
"""

import argparse
import h5py
import torch
from pathlib import Path
from tqdm import tqdm

from spinlock.mno import NOABackbone
from spinlock.mno.vqvae_alignment import AlignedFeatureExtractor


def compute_stats(features_list, group_indices):
    """Compute per-group mean and std.

    Args:
        features_list: List of [B, feature_dim] tensors
        group_indices: Dict mapping group names to feature indices

    Returns:
        Dict with {group_name}_mean and {group_name}_std
    """
    all_features = torch.cat(features_list, dim=0)  # [N, feature_dim]

    stats = {}
    for group_name, indices in group_indices.items():
        group_features = all_features[:, indices]  # [N, group_size]

        mean = group_features.mean(dim=0)  # [group_size]
        std = group_features.std(dim=0)    # [group_size]

        # Replace zero std with 1.0 (no normalization for constant features)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)

        stats[f"{group_name}_mean"] = mean.cpu().numpy().tolist()
        stats[f"{group_name}_std"] = std.cpu().numpy().tolist()

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noa-checkpoint", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VQ-VAE checkpoint first (need codebook_sizes for token-conditioned NOA)
    vqvae_checkpoint = torch.load(args.vqvae_checkpoint, map_location='cpu', weights_only=False)
    vqvae_config = vqvae_checkpoint.get('config', {})
    state_dict = vqvae_checkpoint.get('model_state_dict', {})

    # Extract codebook sizes using multiple fallback methods
    codebook_sizes = []

    # Method 1: Extract from config['levels'] structure (hierarchical VQ-VAE)
    if "levels" in vqvae_config and isinstance(vqvae_config["levels"], dict):
        # Hierarchical structure: {category_name: [level1, level2, ...]}
        for category_name, category_levels in vqvae_config["levels"].items():
            for level in category_levels:
                num_tokens = level.get("num_tokens", level.get("num_embeddings", 64))
                codebook_sizes.append(num_tokens)
    # Method 2: Extract from categories structure (alternative format)
    elif "categories" in vqvae_config:
        for category in vqvae_config["categories"]:
            for level in category.get("levels", []):
                num_embeddings = level.get("num_embeddings", 64)
                codebook_sizes.append(num_embeddings)
    # Method 3: Fallback to state dict inspection
    else:
        # Look for vq_layers embedding weights in state dict
        vq_embedding_keys = [k for k in state_dict.keys() if k.endswith(".codebook") or k.endswith(".embedding.weight")]
        # Sort to ensure consistent ordering
        vq_embedding_keys_sorted = sorted(
            vq_embedding_keys,
            key=lambda k: int(k.split(".")[1]) if k.split(".")[1].isdigit() else 0
        )

        for key in vq_embedding_keys_sorted:
            num_embeddings = state_dict[key].shape[0]
            codebook_sizes.append(num_embeddings)

    if not codebook_sizes:
        raise ValueError(
            f"Could not extract codebook sizes from VQ-VAE checkpoint.\n"
            f"Expected 'categories' in config or 'vq_layers.*.codebook' in state_dict"
        )

    # Load NOA checkpoint
    print(f"Loading NOA checkpoint: {args.noa_checkpoint}")
    noa_checkpoint = torch.load(args.noa_checkpoint, map_location=device, weights_only=False)
    noa_config = noa_checkpoint['config']['model']

    # Create NOA with proper token conditioning setup
    token_conditioning = noa_config.get('token_conditioning', False)
    noa_kwargs = {
        'spatial_dim': noa_config['spatial_dim'],
        'in_channels': noa_config['in_channels'],
        'base_channels': noa_config['base_channels'],
        'encoder_levels': noa_config['encoder_levels'],
        'modes': noa_config['modes'],
        'afno_blocks': noa_config['afno_blocks'],
    }

    if token_conditioning:
        # Use codebook_sizes from NOA config if available (more reliable than VQ-VAE extraction)
        noa_codebook_sizes = noa_config.get('codebook_sizes', codebook_sizes)
        num_tokens = noa_config.get('num_tokens', len(noa_codebook_sizes))
        token_embed_dim = noa_config.get('token_embed_dim', 64)

        print(f"  Token conditioning parameters:")
        print(f"    num_tokens: {num_tokens}")
        print(f"    codebook_sizes: {noa_codebook_sizes[:5]}... (first 5)")
        print(f"    token_embed_dim: {token_embed_dim}")

        noa_kwargs.update({
            'token_conditioning': True,
            'token_embed_dim': token_embed_dim,
            'num_tokens': num_tokens,
            'codebook_sizes': noa_codebook_sizes,
        })

    noa = NOABackbone(**noa_kwargs)
    noa.load_state_dict(noa_checkpoint['model_state_dict'])
    noa = noa.to(device)
    noa.eval()
    print(f"  ✓ NOA loaded")

    # Load feature extractor
    print(f"Loading feature extractor: {args.vqvae_checkpoint}")
    feature_extractor = AlignedFeatureExtractor.from_checkpoint(
        str(args.vqvae_checkpoint),
        device=str(device)
    )
    feature_extractor.eval()
    print(f"  ✓ Feature extractor loaded (input_dim={feature_extractor.input_dim})")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = h5py.File(args.dataset, 'r')

    # Determine dataset structure
    if "inputs" in dataset and "fields" in dataset["inputs"]:
        ic_key = "inputs/fields"
    elif "initial_conditions" in dataset:
        ic_key = "initial_conditions"
    else:
        raise ValueError(f"Unknown dataset format: {list(dataset.keys())}")

    n_total = len(dataset[ic_key])
    n_samples = min(args.n_samples, n_total)
    print(f"  ✓ Will process {n_samples}/{n_total} samples")

    # Extract features from NOA rollouts
    print(f"Extracting features from NOA rollouts...")
    features_list = []

    with torch.no_grad():
        for start_idx in tqdm(range(0, n_samples, args.batch_size)):
            end_idx = min(start_idx + args.batch_size, n_samples)

            # Load batch of initial conditions
            ic_batch = dataset[ic_key][start_idx:end_idx]

            # Handle 3D or 2D ICs
            if ic_batch.ndim == 4:  # [B, C, H, W]
                ic = torch.tensor(ic_batch, device=device, dtype=torch.float32)
                # Extract only first channel if model expects 1 channel
                if ic.shape[1] > 1:
                    ic = ic[:, 0:1, :, :]  # Take only first channel [B, 1, H, W]
            elif ic_batch.ndim == 3:  # [B, H, W]
                ic = torch.tensor(ic_batch, device=device, dtype=torch.float32).unsqueeze(1)
            else:
                raise ValueError(f"Unexpected IC shape: {ic_batch.shape}")

            # Generate NOA rollout
            # NOTE: We don't pass tokens in Stage 2, so tokens=None
            pred_trajectory = noa(ic, steps=256, tokens=None)  # [B, 256, 1, H, W]

            # Extract features
            features, _ = feature_extractor(pred_trajectory, ic=ic)  # [B, feature_dim]

            features_list.append(features.cpu())

    dataset.close()

    # Compute per-group statistics
    print(f"Computing normalization statistics...")
    group_indices = feature_extractor.group_indices
    stats = compute_stats(features_list, group_indices)

    # Add metadata
    stats['_metadata'] = {
        'n_samples': n_samples,
        'input_dim': feature_extractor.input_dim,
        'groups': list(group_indices.keys()),
        'noa_checkpoint': str(args.noa_checkpoint),
        'vqvae_checkpoint': str(args.vqvae_checkpoint),
    }

    # Save to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.output)
    print(f"✓ Normalization stats saved to: {args.output}")

    # Print summary
    print(f"\nStatistics Summary:")
    for group_name in sorted(group_indices.keys()):
        mean_key = f"{group_name}_mean"
        std_key = f"{group_name}_std"
        if mean_key in stats:
            mean_vals = stats[mean_key]
            std_vals = stats[std_key]
            print(f"  {group_name}:")
            print(f"    Mean range: [{min(mean_vals):.2f}, {max(mean_vals):.2f}]")
            print(f"    Std range:  [{min(std_vals):.2f}, {max(std_vals):.2f}]")


if __name__ == "__main__":
    main()
