#!/usr/bin/env python3
"""Generate dummy oracle tokens for testing token-conditioned MNO.

This creates a simple token file with random (but valid) token indices
to test the token conditioning infrastructure without needing full VQ-VAE pipeline.

Usage:
    python scripts/preprocess/generate_dummy_tokens.py \
        --vqvae-checkpoint checkpoints/production/100k_full_features/best_model.pt \
        --output datasets/100k_oracle_tokens_1k_dummy.h5 \
        --n-samples 1000
"""

import argparse
import h5py
import torch
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate dummy oracle tokens for testing")
    parser.add_argument("--vqvae-checkpoint", type=str, required=True, help="VQ-VAE checkpoint to extract structure")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading VQ-VAE structure from: {args.vqvae_checkpoint}")
    ckpt = torch.load(args.vqvae_checkpoint, map_location='cpu', weights_only=False)

    # Extract codebook sizes from config or state dict
    codebook_sizes = []

    # Method 1: From config['levels']
    if 'config' in ckpt and 'levels' in ckpt['config']:
        levels_dict = ckpt['config']['levels']
        if isinstance(levels_dict, dict):
            for category_name, category_levels in levels_dict.items():
                for level in category_levels:
                    num_tokens = level.get('num_tokens', level.get('num_embeddings', 64))
                    codebook_sizes.append(num_tokens)

    # Method 2: From state dict
    if not codebook_sizes:
        import re
        state_dict = ckpt['model_state_dict']
        vq_keys = sorted(
            [k for k in state_dict.keys() if 'vq_layers' in k and 'embedding.weight' in k],
            key=lambda x: int(re.search(r'vq_layers\.(\d+)\.', x).group(1))
        )
        for key in vq_keys:
            codebook_sizes.append(state_dict[key].shape[0])

    if not codebook_sizes:
        raise ValueError("Could not extract codebook sizes from VQ-VAE checkpoint")

    num_tokens = len(codebook_sizes)
    print(f"  Detected {num_tokens} tokens with sizes: {codebook_sizes[:10]}...")

    # Generate random token indices (valid within each codebook)
    print(f"Generating {args.n_samples} random token samples...")
    tokens = np.zeros((args.n_samples, num_tokens), dtype=np.int32)
    for i, codebook_size in enumerate(codebook_sizes):
        tokens[:, i] = np.random.randint(0, codebook_size, size=args.n_samples)

    # Save to HDF5
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_path}")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('tokens', data=tokens, compression='gzip')
        f.attrs['num_samples'] = args.n_samples
        f.attrs['num_tokens'] = num_tokens
        f.attrs['codebook_sizes'] = codebook_sizes
        f.attrs['vqvae_checkpoint'] = args.vqvae_checkpoint
        f.attrs['note'] = 'Dummy random tokens for testing token conditioning infrastructure'

    print(f"✓ Generated dummy tokens: {tokens.shape}")
    print(f"  Saved to: {output_path}")
    print(f"\nThese are RANDOM tokens for testing infrastructure only.")
    print(f"For actual experiments, use compute_oracle_tokens.py with real VQ-VAE.")


if __name__ == "__main__":
    main()
