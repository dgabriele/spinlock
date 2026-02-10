"""
Test roundtrip consistency of inverse decoders.

Tests the property: encode(decode(tokens)) == tokens

This is the correct property for VQ tokenization - decoded values should
form a self-consistent equivalence class that re-encodes to the same tokens.

Usage:
    poetry run python scripts/test_roundtrip_consistency.py \
        --tokenizer checkpoints/v2/vqvae/vq_tokenizer_best.pt \
        --theta-inverse checkpoints/theta_inverse.pt \
        --initial-inverse checkpoints/initial_inverse.pt \
        --tokenized-dataset datasets/50k_tokenized_v2_cleaned.h5 \
        --raw-dataset datasets/50k_baseline.h5 \
        --num-samples 1000
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import h5py
import torch
import numpy as np
from tqdm import tqdm

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.data import SpinlockDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_batch_tokens(
    tokenized_dataset_path: Path,
    start_idx: int,
    end_idx: int,
) -> Dict[str, torch.Tensor]:
    """Load a batch of tokens from pretokenized dataset."""
    tokens = {}
    with h5py.File(tokenized_dataset_path, 'r') as f:
        for key in f['tokens'].keys():
            tokens[key] = torch.from_numpy(f[f'tokens/{key}'][start_idx:end_idx]).long()
    return tokens


def test_theta_roundtrip(
    tokenizer: VQTokenizer,
    tokens: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """
    Test theta roundtrip consistency.

    Flow: theta_tokens → decode → theta_recon → encode → theta_tokens_rt
    Compare: theta_tokens vs theta_tokens_rt
    """
    tokenizer.model.eval()

    # Move tokens to device
    tokens_device = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        # Extract embeddings and decode
        quantized = tokenizer._extract_embeddings(tokens_device)
        reconstructed = tokenizer.model.decoder(quantized)

        # Extract theta encoded features
        offset = 0
        if "temporal" in tokenizer.model.families:
            offset += tokenizer.model.temporal_dim
        if "initial" in tokenizer.model.families:
            offset += tokenizer.model.initial_dim

        theta_encoded = reconstructed[:, offset:offset+tokenizer.model.theta_dim]

        # Decode to actual parameters
        theta_recon = tokenizer.model.theta_inverse(theta_encoded)

        # Re-encode through theta encoder
        theta_encoded_rt = tokenizer.model.theta_encoder(theta_recon)

        # Project to hierarchical latents and quantize
        theta_tokens_rt = {}

        for family_cat, indices in tokenizer.group_indices.items():
            if not family_cat.startswith('theta_'):
                continue

            # Extract category features
            cat_features = theta_encoded_rt[:, indices]

            # Project to hierarchical latents
            projector = tokenizer.model.projectors[family_cat]
            latents = projector(cat_features)

            # Quantize each level
            for level_idx, latent in enumerate(latents):
                quantizer_key = f"{family_cat}_L{level_idx}"
                quantizer = tokenizer.model.quantizers[quantizer_key]

                # Find nearest codebook entry
                distances = torch.cdist(latent, quantizer.embedding.weight, p=2.0)
                token_indices = distances.argmin(dim=1)

                theta_tokens_rt[quantizer_key] = token_indices.cpu()

    # Compare original vs roundtrip tokens
    results = {}
    total_matches = 0
    total_tokens = 0

    for key in theta_tokens_rt.keys():
        original = tokens[key]
        roundtrip = theta_tokens_rt[key]

        matches = (original == roundtrip).sum().item()
        total = original.numel()
        match_rate = matches / total

        results[key] = match_rate
        total_matches += matches
        total_tokens += total

    results['overall'] = total_matches / total_tokens if total_tokens > 0 else 0.0

    return results


def test_initial_roundtrip(
    tokenizer: VQTokenizer,
    tokens: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """
    Test initial roundtrip consistency.

    Flow: initial_tokens → decode → IC_recon → encode → initial_tokens_rt
    Compare: initial_tokens vs initial_tokens_rt
    """
    tokenizer.model.eval()

    # Move tokens to device
    tokens_device = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        # Extract embeddings and decode
        quantized = tokenizer._extract_embeddings(tokens_device)
        reconstructed = tokenizer.model.decoder(quantized)

        # Extract initial encoded features
        offset = 0
        if "temporal" in tokenizer.model.families:
            offset += tokenizer.model.temporal_dim

        initial_encoded = reconstructed[:, offset:offset+tokenizer.model.initial_dim]

        # Decode to spatial grids
        IC_recon = tokenizer.model.initial_inverse(initial_encoded)

        # Re-encode through initial encoder
        # Note: Initial encoder might need both manual features and raw grids
        # For now, we'll just use the raw grids through the CNN part
        if hasattr(tokenizer.model.initial_encoder, 'cnn_encoder'):
            # Hybrid encoder - use CNN part only
            initial_encoded_rt = tokenizer.model.initial_encoder.cnn_encoder(IC_recon)
        else:
            # CNN-only encoder
            initial_encoded_rt = tokenizer.model.initial_encoder(IC_recon)

        # Project to hierarchical latents and quantize
        initial_tokens_rt = {}

        for family_cat, indices in tokenizer.group_indices.items():
            if not family_cat.startswith('initial_'):
                continue

            # For initial, we need to map the CNN output back to the full encoded space
            # This is tricky because the encoder combines CNN + manual features
            # For now, skip this and just note the limitation
            logger.warning(
                f"Skipping {family_cat} - initial roundtrip requires full feature reconstruction"
            )
            continue

    # Since we're skipping, return empty results
    results = {'overall': 0.0, 'note': 'initial roundtrip not fully implemented yet'}

    return results


def main():
    parser = argparse.ArgumentParser(description='Test roundtrip consistency')
    parser.add_argument('--tokenizer', type=Path, required=True,
                        help='Path to trained VQTokenizer checkpoint')
    parser.add_argument('--theta-inverse', type=Path, required=True,
                        help='Path to trained theta inverse checkpoint')
    parser.add_argument('--initial-inverse', type=Path, required=True,
                        help='Path to trained initial inverse checkpoint')
    parser.add_argument('--tokenized-dataset', type=Path, required=True,
                        help='Path to pretokenized dataset')
    parser.add_argument('--raw-dataset', type=Path, required=True,
                        help='Path to raw dataset')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to test')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for processing')
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

    # Test roundtrip consistency
    logger.info(f"\nTesting roundtrip consistency on {args.num_samples} samples")
    logger.info("="*70)

    theta_results_all = []

    for start_idx in tqdm(range(0, args.num_samples, args.batch_size), desc="Testing batches"):
        end_idx = min(start_idx + args.batch_size, args.num_samples)

        # Load batch of tokens
        tokens = load_batch_tokens(args.tokenized_dataset, start_idx, end_idx)

        # Test theta roundtrip
        theta_results = test_theta_roundtrip(tokenizer, tokens, device)
        theta_results_all.append(theta_results)

    # Aggregate results
    logger.info("\n" + "="*70)
    logger.info("THETA ROUNDTRIP CONSISTENCY RESULTS")
    logger.info("="*70)

    # Get all keys from first result
    all_keys = [k for k in theta_results_all[0].keys() if k != 'overall']

    # Compute average for each token category
    for key in sorted(all_keys):
        avg_match = np.mean([r[key] for r in theta_results_all])
        logger.info(f"{key}: {avg_match*100:.2f}% token match")

    # Overall
    overall_match = np.mean([r['overall'] for r in theta_results_all])
    logger.info(f"\nOverall theta token match: {overall_match*100:.2f}%")

    # Interpretation
    logger.info("\n" + "="*70)
    logger.info("INTERPRETATION")
    logger.info("="*70)

    if overall_match > 0.95:
        logger.info("✅ EXCELLENT: >95% roundtrip consistency")
        logger.info("   Inverse models are self-consistent!")
        logger.info("   VQ tokens properly represent equivalence classes.")
    elif overall_match > 0.80:
        logger.info("⚠️  MODERATE: 80-95% roundtrip consistency")
        logger.info("   Inverse models are partially consistent.")
        logger.info("   May benefit from roundtrip loss retraining.")
    else:
        logger.info("❌ POOR: <80% roundtrip consistency")
        logger.info("   Inverse models are not self-consistent.")
        logger.info("   Need to retrain with explicit roundtrip loss.")

    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Tested: {args.num_samples} samples")
    logger.info(f"Theta roundtrip: {overall_match*100:.2f}% token match")
    logger.info(f"Target: >95% for self-consistency")

    # Save detailed results
    results_file = Path("results/roundtrip_consistency_results.txt")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        f.write("Roundtrip Consistency Test Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Overall theta token match: {overall_match*100:.2f}%\n\n")
        f.write("Per-category results:\n")
        for key in sorted(all_keys):
            avg_match = np.mean([r[key] for r in theta_results_all])
            f.write(f"  {key}: {avg_match*100:.2f}%\n")

    logger.info(f"\nDetailed results saved to: {results_file}")


if __name__ == '__main__':
    main()
