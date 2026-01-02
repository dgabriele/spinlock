#!/usr/bin/env python3
"""
Validate Latent Dimension Bug Fix

Compares VQ-VAE performance before and after fixing the categorical_vqvae.py:91 bug
that used feature count instead of embedding_dim for latent dimension computation.

Usage:
    # Extract baseline metrics from V7 (buggy)
    python scripts/dev/validate_latent_dim_bugfix.py \
        --mode baseline \
        --checkpoint checkpoints/production/10k_arch_summary_400epochs_v2/best_model.pt \
        --output results/bugfix_validation/v7_baseline.json

    # Validate fixed version (V8)
    python scripts/dev/validate_latent_dim_bugfix.py \
        --mode validate \
        --checkpoint checkpoints/production/10k_v8_bugfix/best_model.pt \
        --output results/bugfix_validation/v8_fixed.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
from spinlock.encoding.training.metrics import (
    compute_per_category_metrics,
    compute_reconstruction_error,
    compute_quality_score,
)


def load_checkpoint_with_model(checkpoint_path: Path) -> Tuple[Dict, CategoricalHierarchicalVQVAE]:
    """Load checkpoint and reconstruct model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract model config
    if "model_config" not in checkpoint:
        raise ValueError("Checkpoint missing model_config")

    model_config_dict = checkpoint["model_config"]

    # Reconstruct config object from dict
    model_config = CategoricalVQVAEConfig(**model_config_dict)

    # Reconstruct model
    model = CategoricalHierarchicalVQVAE(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint, model


def extract_latent_dimensions(model: CategoricalHierarchicalVQVAE) -> Dict[str, List[int]]:
    """Extract per-category latent dimensions from model config."""
    print("Extracting latent dimensions...")

    latent_dims = {}

    if hasattr(model.config, 'category_levels') and model.config.category_levels is not None:
        for category, levels in model.config.category_levels.items():
            latent_dims[category] = [level['latent_dim'] for level in levels]
    elif hasattr(model, 'categories') and hasattr(model, 'category_vqs'):
        # Extract from actual VQ modules
        for category in model.categories:
            if category in model.category_vqs:
                vq_module = model.category_vqs[category]
                if hasattr(vq_module, 'vqs'):
                    latent_dims[category] = [vq.latent_dim for vq in vq_module.vqs]
    else:
        print("Warning: Could not extract latent dimensions from model")

    return latent_dims


def compute_detailed_metrics(model: CategoricalHierarchicalVQVAE, dataloader: DataLoader, device: str = "cuda") -> Dict[str, Any]:
    """Compute all validation metrics per category."""
    print(f"Computing detailed metrics on {device}...")

    model = model.to(device)
    model.eval()

    # Per-category metrics
    category_metrics = compute_per_category_metrics(
        model, dataloader, device, max_batches=None  # Full validation
    )

    # Overall reconstruction error
    overall_recon_error = compute_reconstruction_error(model, dataloader, device)
    overall_quality = compute_quality_score(overall_recon_error)

    # Aggregate utilization
    util_metrics = [v for k, v in category_metrics.items() if 'utilization' in k and '/level_' in k]
    avg_utilization = sum(util_metrics) / len(util_metrics) if util_metrics else 0.0

    return {
        'per_category': category_metrics,
        'overall': {
            'utilization': avg_utilization,
            'reconstruction_error': overall_recon_error,
            'quality': overall_quality
        }
    }


def identify_problematic_categories(metrics: Dict, threshold: float = 0.20) -> List[Tuple[str, float]]:
    """Identify categories with low L0 utilization."""
    problematic = []

    for key, value in metrics['per_category'].items():
        if 'level_0/utilization' in key and value < threshold:
            category = key.split('/')[0]
            problematic.append((category, value))

    return sorted(problematic, key=lambda x: x[1])


def validate_latent_dims_correctness(latent_dims: Dict[str, List[int]], min_l0_dim: int = 100) -> Dict[str, bool]:
    """Validate that latent dimensions are correctly sized."""
    results = {}

    for category, dims in latent_dims.items():
        if len(dims) == 0:
            results[category] = False
            continue

        l0_dim = dims[0]
        results[category] = l0_dim >= min_l0_dim

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate latent dimension bug fix")
    parser.add_argument("--mode", choices=["baseline", "validate"], required=True,
                      help="baseline: extract V7 metrics, validate: check V8 improvements")
    parser.add_argument("--checkpoint", type=Path, required=True,
                      help="Path to checkpoint file")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/baseline_10k.h5"),
                      help="Path to validation dataset")
    parser.add_argument("--output", type=Path, required=True,
                      help="Path to output JSON file")
    parser.add_argument("--device", default="cuda",
                      help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=512,
                      help="Batch size for validation")

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("VQ-VAE LATENT DIMENSION BUG FIX VALIDATION")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print("="*70)

    # Load checkpoint and model
    checkpoint, model = load_checkpoint_with_model(args.checkpoint)

    # Extract latent dimensions
    latent_dims = extract_latent_dimensions(model)

    print("\nLatent Dimensions by Category:")
    print("-" * 70)
    for category, dims in sorted(latent_dims.items()):
        print(f"  {category:15s}: {dims}")

    # Validate correctness
    print("\nLatent Dimension Validation (L0 >= 100D):")
    print("-" * 70)
    dim_validation = validate_latent_dims_correctness(latent_dims, min_l0_dim=100)
    for category, is_correct in sorted(dim_validation.items()):
        status = "✅ PASS" if is_correct else "❌ FAIL"
        l0_dim = latent_dims[category][0] if latent_dims[category] else 0
        print(f"  {category:15s}: {status} (L0={l0_dim}D)")

    # Load dataset for metric computation
    print(f"\nLoading dataset: {args.dataset}")
    import h5py
    with h5py.File(args.dataset, 'r') as f:
        # Load features from grouped structure
        features_dict = {}
        if 'features' in f and isinstance(f['features'], h5py.Group):
            print(f"Feature groups: {list(f['features'].keys())}")
            for group_name in f['features'].keys():
                # Access the aggregated features dataset
                try:
                    group = f['features'][group_name]
                    if 'aggregated' in group and 'features' in group['aggregated']:
                        dataset = group['aggregated']['features']
                        features_dict[group_name] = torch.from_numpy(dataset[:])
                        print(f"  Loaded {group_name}: shape={features_dict[group_name].shape}")
                except KeyError as e:
                    print(f"  Skipping {group_name}: {e}")
                    continue

        # Concatenate features - use architecture + summary (match V7 training config)
        # NOTE: V7 was trained on architecture + summary ONLY (not initial features)
        feature_list = []
        for group_name in ['architecture', 'summary']:
            if group_name in features_dict:
                feature_list.append(features_dict[group_name])
                print(f"  Including {group_name} in concat")

        if feature_list:
            features = torch.cat(feature_list, dim=1)
        else:
            raise ValueError(f"Could not load required features from {args.dataset}")

        print(f"Total features shape: {features.shape}")

    # Create val split (last 10%)
    split_idx = int(len(features) * 0.9)
    val_features = features[split_idx:]

    # Create dataloader
    from torch.utils.data import TensorDataset
    val_dataset = TensorDataset(val_features)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Compute metrics
    metrics = compute_detailed_metrics(model, val_loader, args.device)

    # Identify problematic categories
    problematic = identify_problematic_categories(metrics, threshold=0.20)

    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    print(f"Utilization: {metrics['overall']['utilization']:.1%}")
    print(f"Reconstruction Error: {metrics['overall']['reconstruction_error']:.4f}")
    print(f"Quality Score: {metrics['overall']['quality']:.4f}")

    print("\n" + "="*70)
    print(f"PROBLEMATIC CATEGORIES (L0 utilization < 20%): {len(problematic)}")
    print("="*70)
    for category, util in problematic:
        feature_count = len(model.config.group_indices[category])
        mse_key = f"{category}/reconstruction_mse"
        mse = metrics['per_category'].get(mse_key, 0.0)
        print(f"  {category:15s}: {util:.1%} utilization, {mse:.4f} MSE ({feature_count} features)")

    # Prepare output
    output_data = {
        "checkpoint_path": str(args.checkpoint),
        "mode": args.mode,
        "latent_dimensions": latent_dims,
        "latent_dim_validation": dim_validation,
        "metrics": {
            "overall": metrics['overall'],
            "per_category": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metrics['per_category'].items()
            }
        },
        "problematic_categories": [
            {"name": cat, "l0_utilization": float(util)}
            for cat, util in problematic
        ],
        "validation_summary": {
            "all_latent_dims_correct": all(dim_validation.values()),
            "num_problematic_categories": len(problematic),
            "overall_utilization": float(metrics['overall']['utilization']),
            "overall_quality": float(metrics['overall']['quality']),
        }
    }

    # Save output
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Print validation summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"All latent dims correct (≥100D): {output_data['validation_summary']['all_latent_dims_correct']}")
    print(f"Number of problematic categories: {output_data['validation_summary']['num_problematic_categories']}")
    print(f"Overall utilization: {output_data['validation_summary']['overall_utilization']:.1%}")
    print(f"Overall quality: {output_data['validation_summary']['overall_quality']:.4f}")

    if args.mode == "validate":
        print("\n" + "="*70)
        print("BUG FIX VALIDATION")
        print("="*70)
        if output_data['validation_summary']['all_latent_dims_correct']:
            print("✅ PASS: All categories have L0 latent_dim ≥ 100D")
        else:
            print("❌ FAIL: Some categories have undersized latent dimensions")

        if output_data['validation_summary']['overall_quality'] >= 0.85:
            print(f"✅ PASS: Quality {output_data['validation_summary']['overall_quality']:.1%} ≥ 85% target")
        else:
            print(f"⚠️  PARTIAL: Quality {output_data['validation_summary']['overall_quality']:.1%} < 85% target")

        if output_data['validation_summary']['num_problematic_categories'] <= 2:
            print(f"✅ PASS: Only {output_data['validation_summary']['num_problematic_categories']} problematic categories")
        else:
            print(f"⚠️  PARTIAL: {output_data['validation_summary']['num_problematic_categories']} problematic categories remain")

    print("="*70)


if __name__ == "__main__":
    main()
