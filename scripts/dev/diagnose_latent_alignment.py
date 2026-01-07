#!/usr/bin/env python3
"""
Diagnostic script for evaluating L_latent alignment quality.

Evaluates:
1. VQ reconstruction quality (MSE after tokenization)
2. Token diversity (codebook utilization)
3. Alignment correlation (cosine similarity between NOA and VQ latents)
4. Per-category breakdown (INITIAL, SUMMARY, TEMPORAL)
5. Temporal consistency (alignment stability across trajectory)

Usage:
    python scripts/dev/diagnose_latent_alignment.py \
        --noa-checkpoint checkpoints/noa/latest.pt \
        --vqvae-path checkpoints/production/100k_3family_v1 \
        --dataset datasets/100k_full_features.h5 \
        --n-samples 100 \
        --timesteps 256 \
        --n-realizations 5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.noa import NOABackbone, CNOReplayer, VQVAEAlignmentLoss
from spinlock.features.storage import HDF5FeatureDataset


def load_models(
    noa_checkpoint: str,
    vqvae_path: str,
    config_path: str,
    device: str = "cuda",
) -> Tuple[NOABackbone, VQVAEAlignmentLoss, CNOReplayer]:
    """Load trained NOA and VQ-VAE models."""

    print("Loading models...")

    # Load CNO replayer
    replayer = CNOReplayer.from_config(config_path)
    print(f"  ✓ CNO replayer loaded")

    # Load NOA checkpoint
    checkpoint = torch.load(noa_checkpoint, map_location=device)

    # Create NOA backbone (infer architecture from checkpoint)
    noa_config = checkpoint.get('config', {})
    noa = NOABackbone(
        in_channels=1,
        out_channels=1,
        base_channels=noa_config.get('base_channels', 32),
        encoder_levels=noa_config.get('encoder_levels', 3),
        modes=noa_config.get('modes', 16),
        afno_blocks=noa_config.get('afno_blocks', 4),
        use_checkpointing=False,
    ).to(device)

    noa.load_state_dict(checkpoint['model_state_dict'])
    noa.eval()
    print(f"  ✓ NOA loaded from {noa_checkpoint}")
    print(f"    Parameters: {sum(p.numel() for p in noa.parameters()):,}")

    # Load VQ-VAE alignment
    alignment = VQVAEAlignmentLoss.from_checkpoint(
        vqvae_path=vqvae_path,
        device=device,
        noa=noa,
        enable_latent_loss=True,
        latent_sample_steps=-1,  # Sample all timesteps for thorough diagnostics
    )

    # Load projector state if available
    if 'alignment_state' in checkpoint and alignment.latent_projector is not None:
        alignment.latent_projector.load_state_dict(checkpoint['alignment_state'])
        print(f"  ✓ Latent projector loaded")

    print(f"  ✓ VQ-VAE loaded from {vqvae_path}")
    print(f"    VQ latent dim: {alignment.latent_projector.vq_latent_dim if alignment.latent_projector else 'N/A'}")
    print(f"    NOA latent dim: {alignment.latent_projector.noa_latent_dim if alignment.latent_projector else 'N/A'}")

    return noa, alignment, replayer


def compute_vq_reconstruction_quality(
    alignment: VQVAEAlignmentLoss,
    trajectories: torch.Tensor,
    ic: torch.Tensor,
) -> Dict[str, float]:
    """Compute VQ reconstruction quality after tokenization.

    Args:
        alignment: VQ-VAE alignment module
        trajectories: [B, T, C, H, W] predicted trajectories
        ic: [B, C, H, W] initial conditions

    Returns:
        Dictionary with reconstruction metrics
    """
    B, T = trajectories.shape[:2]

    # Extract features from trajectories
    result = alignment.feature_extractor(trajectories, ic=ic)
    if isinstance(result, tuple):
        features, raw_ics = result
    else:
        features = result
        raw_ics = ic

    # Normalize features
    features_norm = alignment._normalize_features(features)

    # Encode to latents
    if alignment._is_hybrid_model and raw_ics is not None:
        z_list = alignment.vqvae.encode(features_norm, raw_ics=raw_ics)
    else:
        z_list = alignment.vqvae.encode(features_norm)

    # Quantize
    z_q_list, _, indices_list = alignment.vqvae.quantize(z_list)

    # Decode
    if alignment._is_hybrid_model and raw_ics is not None:
        reconstructed = alignment.vqvae.decode(z_q_list, raw_ics=raw_ics)
    else:
        reconstructed = alignment.vqvae.decode(z_q_list)

    # Compute reconstruction error
    recon_mse = F.mse_loss(features_norm, reconstructed).item()

    # Per-dimension error
    per_dim_error = torch.mean((features_norm - reconstructed) ** 2, dim=0)

    # Compute per-category reconstruction if available
    category_errors = {}
    if hasattr(alignment, 'group_indices'):
        for group_name, indices in alignment.group_indices.items():
            if len(indices) > 0:
                group_features = features_norm[:, indices]
                group_recon = reconstructed[:, indices]
                category_errors[group_name] = F.mse_loss(group_features, group_recon).item()

    return {
        'total_mse': recon_mse,
        'per_dim_mean': per_dim_error.mean().item(),
        'per_dim_max': per_dim_error.max().item(),
        'category_errors': category_errors,
    }


def compute_token_diversity(
    alignment: VQVAEAlignmentLoss,
    trajectories: torch.Tensor,
    ic: torch.Tensor,
) -> Dict[str, float]:
    """Compute codebook utilization statistics.

    Args:
        alignment: VQ-VAE alignment module
        trajectories: [B, T, C, H, W] predicted trajectories
        ic: [B, C, H, W] initial conditions

    Returns:
        Dictionary with diversity metrics
    """
    # Extract features
    result = alignment.feature_extractor(trajectories, ic=ic)
    if isinstance(result, tuple):
        features, raw_ics = result
    else:
        features = result
        raw_ics = ic

    features_norm = alignment._normalize_features(features)

    # Encode and quantize
    if alignment._is_hybrid_model and raw_ics is not None:
        z_list = alignment.vqvae.encode(features_norm, raw_ics=raw_ics)
    else:
        z_list = alignment.vqvae.encode(features_norm)

    _, _, indices_list = alignment.vqvae.quantize(z_list)

    # Compute utilization per category
    total_codes_used = 0
    total_codes_available = 0
    category_utilization = {}

    for i, (indices, group_name) in enumerate(zip(indices_list, alignment.group_indices.keys())):
        # indices: [B, num_levels]
        unique_codes = torch.unique(indices).numel()

        # Get total codes for this category
        if hasattr(alignment.vqvae, 'group_quantizers'):
            num_levels = indices.shape[1]
            codes_per_level = alignment.vqvae.group_quantizers[group_name].num_embeddings
            total_codes = codes_per_level ** num_levels
        else:
            total_codes = alignment.vqvae.num_embeddings if hasattr(alignment.vqvae, 'num_embeddings') else 512

        utilization = 100 * unique_codes / total_codes if total_codes > 0 else 0
        category_utilization[group_name] = {
            'used': unique_codes,
            'total': total_codes,
            'utilization_pct': utilization,
        }

        total_codes_used += unique_codes
        total_codes_available += total_codes

    # Entropy of token distribution (higher = more diverse)
    all_indices = torch.cat([idx.flatten() for idx in indices_list])
    token_counts = torch.bincount(all_indices)
    token_probs = token_counts.float() / token_counts.sum()
    token_probs = token_probs[token_probs > 0]  # Remove zeros
    entropy = -(token_probs * torch.log(token_probs)).sum().item()

    return {
        'total_utilization_pct': 100 * total_codes_used / total_codes_available if total_codes_available > 0 else 0,
        'total_codes_used': total_codes_used,
        'total_codes_available': total_codes_available,
        'entropy': entropy,
        'category_utilization': category_utilization,
    }


def compute_alignment_correlation(
    noa: NOABackbone,
    alignment: VQVAEAlignmentLoss,
    trajectories: torch.Tensor,
    ic: torch.Tensor,
) -> Dict[str, float]:
    """Compute correlation between NOA and VQ latents.

    Args:
        noa: NOA backbone
        alignment: VQ-VAE alignment module
        trajectories: [B, T, C, H, W] predicted trajectories
        ic: [B, C, H, W] initial conditions

    Returns:
        Dictionary with correlation metrics
    """
    B, T = trajectories.shape[:2]

    # Extract features for VQ latents
    result = alignment.feature_extractor(trajectories, ic=ic)
    if isinstance(result, tuple):
        features, raw_ics = result
    else:
        features = result
        raw_ics = ic

    features_norm = alignment._normalize_features(features)

    # Get VQ latents
    if alignment._is_hybrid_model and raw_ics is not None:
        z_list = alignment.vqvae.encode(features_norm, raw_ics=raw_ics)
    else:
        z_list = alignment.vqvae.encode(features_norm)
    vq_latents = torch.cat(z_list, dim=1)  # [B, D_vq]

    # Extract NOA latents from trajectory (sample evenly across time)
    sample_indices = [int(i * (T - 1) / 9) for i in range(10)]  # Sample 10 timesteps
    noa_latents_list = []

    with torch.no_grad():
        for t in sample_indices:
            state_t = trajectories[:, t, :, :, :]
            features_t = noa.get_intermediate_features(state_t, extract_from="bottleneck")
            bottleneck = features_t['bottleneck']
            proj_latent = alignment.latent_projector(bottleneck)
            noa_latents_list.append(proj_latent)

    # Average across sampled timesteps
    noa_latents = torch.stack(noa_latents_list, dim=1).mean(dim=1)  # [B, D_vq]

    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(noa_latents, vq_latents, dim=1)  # [B]

    # Compute MSE (L_latent equivalent)
    mse = F.mse_loss(noa_latents, vq_latents).item()

    # Compute per-dimension correlation
    noa_std = noa_latents.std(dim=0)
    vq_std = vq_latents.std(dim=0)

    return {
        'cosine_similarity_mean': cosine_sim.mean().item(),
        'cosine_similarity_std': cosine_sim.std().item(),
        'cosine_similarity_min': cosine_sim.min().item(),
        'cosine_similarity_max': cosine_sim.max().item(),
        'mse': mse,
        'noa_latent_std_mean': noa_std.mean().item(),
        'vq_latent_std_mean': vq_std.mean().item(),
    }


def compute_temporal_consistency(
    noa: NOABackbone,
    alignment: VQVAEAlignmentLoss,
    trajectories: torch.Tensor,
) -> Dict[str, float]:
    """Compute how alignment varies across trajectory timesteps.

    Args:
        noa: NOA backbone
        alignment: VQ-VAE alignment module
        trajectories: [B, T, C, H, W] predicted trajectories

    Returns:
        Dictionary with temporal consistency metrics
    """
    B, T = trajectories.shape[:2]

    # Sample 10 evenly-spaced timesteps
    sample_indices = [int(i * (T - 1) / 9) for i in range(10)]

    latent_norms = []

    with torch.no_grad():
        for t in sample_indices:
            state_t = trajectories[:, t, :, :, :]
            features_t = noa.get_intermediate_features(state_t, extract_from="bottleneck")
            bottleneck = features_t['bottleneck']
            proj_latent = alignment.latent_projector(bottleneck)

            # Compute L2 norm of projected latents
            latent_norm = torch.norm(proj_latent, dim=1).mean().item()
            latent_norms.append(latent_norm)

    latent_norms = np.array(latent_norms)

    return {
        'latent_norm_mean': latent_norms.mean(),
        'latent_norm_std': latent_norms.std(),
        'latent_norm_cv': latent_norms.std() / latent_norms.mean() if latent_norms.mean() > 0 else 0,
        'temporal_variation': latent_norms.max() - latent_norms.min(),
    }


def run_diagnostics(args):
    """Run full diagnostic suite."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("L_latent Alignment Diagnostics")
    print("=" * 60)

    # Load models
    noa, alignment, replayer = load_models(
        noa_checkpoint=args.noa_checkpoint,
        vqvae_path=args.vqvae_path,
        config_path=args.config,
        device=device,
    )

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = HDF5FeatureDataset(args.dataset)

    # Sample subset for diagnostics
    indices = torch.randperm(len(dataset))[:args.n_samples].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    print(f"  Test samples: {args.n_samples}")
    print(f"  Batch size: {args.batch_size}")

    # Collect results
    all_trajectories = []
    all_ics = []

    print(f"\nGenerating test trajectories...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ic = batch['ic'].to(device)
            params = batch['params'].to(device)

            # Replay CNO operator
            operator = replayer.replay(params)

            # Generate trajectory with NOA
            trajectory = noa.rollout(
                ic,
                steps=args.timesteps,
                operator=operator,
                return_all_steps=True,
            )  # [B, T+1, C, H, W]

            all_trajectories.append(trajectory)
            all_ics.append(ic)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Generated {(batch_idx + 1) * args.batch_size} trajectories...")

    # Concatenate all batches
    trajectories = torch.cat(all_trajectories, dim=0)[:args.n_samples]
    ics = torch.cat(all_ics, dim=0)[:args.n_samples]

    print(f"  ✓ Generated {len(trajectories)} trajectories")

    # Run diagnostics
    print("\n" + "=" * 60)
    print("1. VQ Reconstruction Quality")
    print("=" * 60)
    recon_metrics = compute_vq_reconstruction_quality(alignment, trajectories, ics)
    print(f"Total MSE: {recon_metrics['total_mse']:.4f}")
    print(f"Per-dimension mean error: {recon_metrics['per_dim_mean']:.6f}")
    print(f"Per-dimension max error: {recon_metrics['per_dim_max']:.6f}")

    if recon_metrics['category_errors']:
        print("\nPer-category reconstruction errors:")
        for cat, err in recon_metrics['category_errors'].items():
            print(f"  {cat:12s}: {err:.4f}")

    # Quality assessment
    if recon_metrics['total_mse'] < 0.1:
        quality = "Excellent"
    elif recon_metrics['total_mse'] < 0.5:
        quality = "Good"
    elif recon_metrics['total_mse'] < 1.0:
        quality = "Moderate"
    else:
        quality = "Poor"
    print(f"\n➜ Quality Assessment: {quality}")

    print("\n" + "=" * 60)
    print("2. Token Diversity")
    print("=" * 60)
    diversity_metrics = compute_token_diversity(alignment, trajectories, ics)
    print(f"Overall utilization: {diversity_metrics['total_utilization_pct']:.1f}% "
          f"({diversity_metrics['total_codes_used']}/{diversity_metrics['total_codes_available']} codes)")
    print(f"Token entropy: {diversity_metrics['entropy']:.2f}")

    if diversity_metrics['category_utilization']:
        print("\nPer-category utilization:")
        for cat, stats in diversity_metrics['category_utilization'].items():
            print(f"  {cat:12s}: {stats['utilization_pct']:5.1f}% "
                  f"({stats['used']}/{stats['total']} codes)")

    # Diversity assessment
    if diversity_metrics['total_utilization_pct'] > 70:
        diversity_quality = "Excellent"
    elif diversity_metrics['total_utilization_pct'] > 50:
        diversity_quality = "Good"
    elif diversity_metrics['total_utilization_pct'] > 30:
        diversity_quality = "Moderate"
    else:
        diversity_quality = "Poor"
    print(f"\n➜ Diversity Assessment: {diversity_quality}")

    print("\n" + "=" * 60)
    print("3. Alignment Correlation")
    print("=" * 60)
    corr_metrics = compute_alignment_correlation(noa, alignment, trajectories, ics)
    print(f"Cosine similarity: {corr_metrics['cosine_similarity_mean']:.3f} ± {corr_metrics['cosine_similarity_std']:.3f}")
    print(f"  Range: [{corr_metrics['cosine_similarity_min']:.3f}, {corr_metrics['cosine_similarity_max']:.3f}]")
    print(f"L_latent (MSE): {corr_metrics['mse']:.4f}")
    print(f"NOA latent std: {corr_metrics['noa_latent_std_mean']:.4f}")
    print(f"VQ latent std: {corr_metrics['vq_latent_std_mean']:.4f}")

    # Correlation assessment
    cos_sim = corr_metrics['cosine_similarity_mean']
    if cos_sim > 0.7:
        corr_quality = "Strong"
    elif cos_sim > 0.5:
        corr_quality = "Moderate"
    elif cos_sim > 0.3:
        corr_quality = "Weak"
    else:
        corr_quality = "Poor"
    print(f"\n➜ Correlation Assessment: {corr_quality}")

    print("\n" + "=" * 60)
    print("4. Temporal Consistency")
    print("=" * 60)
    temporal_metrics = compute_temporal_consistency(noa, alignment, trajectories)
    print(f"Latent norm: {temporal_metrics['latent_norm_mean']:.3f} ± {temporal_metrics['latent_norm_std']:.3f}")
    print(f"Coefficient of variation: {temporal_metrics['latent_norm_cv']:.3f}")
    print(f"Temporal variation: {temporal_metrics['temporal_variation']:.3f}")

    # Consistency assessment
    cv = temporal_metrics['latent_norm_cv']
    if cv < 0.05:
        consistency_quality = "Excellent"
    elif cv < 0.1:
        consistency_quality = "Good"
    elif cv < 0.2:
        consistency_quality = "Moderate"
    else:
        consistency_quality = "Poor"
    print(f"\n➜ Consistency Assessment: {consistency_quality}")

    # Overall summary
    print("\n" + "=" * 60)
    print("Overall Summary")
    print("=" * 60)
    print(f"VQ Reconstruction:      {quality}")
    print(f"Token Diversity:        {diversity_quality}")
    print(f"Alignment Correlation:  {corr_quality}")
    print(f"Temporal Consistency:   {consistency_quality}")

    # Final verdict
    quality_scores = {
        'Excellent': 4,
        'Good': 3,
        'Moderate': 2,
        'Strong': 3,
        'Weak': 1,
        'Poor': 0,
    }

    avg_score = np.mean([
        quality_scores[quality],
        quality_scores[diversity_quality],
        quality_scores[corr_quality],
        quality_scores[consistency_quality],
    ])

    if avg_score >= 3.5:
        verdict = "EXCELLENT - L_latent alignment is working very well!"
    elif avg_score >= 2.5:
        verdict = "GOOD - L_latent provides meaningful alignment"
    elif avg_score >= 1.5:
        verdict = "MODERATE - Some alignment achieved, could be improved"
    else:
        verdict = "POOR - L_latent may need adjustment"

    print(f"\n➜ Final Verdict: {verdict}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Diagnose L_latent alignment quality")

    # Model paths
    parser.add_argument(
        "--noa-checkpoint", type=str, required=True,
        help="Path to trained NOA checkpoint"
    )
    parser.add_argument(
        "--vqvae-path", type=str, required=True,
        help="Path to VQ-VAE checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiments/local_100k_optimized.yaml",
        help="Path to CNO config (default: local_100k_optimized.yaml)"
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to HDF5 dataset"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of test samples (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for trajectory generation (default: 8)"
    )

    # Trajectory parameters
    parser.add_argument(
        "--timesteps", type=int, default=256,
        help="Number of timesteps to generate (default: 256)"
    )
    parser.add_argument(
        "--n-realizations", type=int, default=1,
        help="Number of realizations per operator (default: 1)"
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.noa_checkpoint).exists():
        print(f"Error: NOA checkpoint not found: {args.noa_checkpoint}")
        return 1

    if not Path(args.vqvae_path).exists():
        print(f"Error: VQ-VAE path not found: {args.vqvae_path}")
        return 1

    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found: {args.dataset}")
        return 1

    run_diagnostics(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
