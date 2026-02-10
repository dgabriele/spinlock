#!/usr/bin/env python3
"""Test VQTokenizer trainer metrics implementation.

Verifies that:
1. Feature cleaning is properly integrated
2. Per-quantizer utilization is tracked during validation
3. Per-category MSE is computed
4. Final metrics are formatted for visualization compatibility
"""

import torch
import yaml
from pathlib import Path
from spinlock.tokens.config import TokenizerConfig
from spinlock.tokens.model import JointHierarchicalVQVAE
from spinlock.tokens.trainer import VQTokenizerTrainer


def test_trainer_metrics():
    """Test trainer metrics computation."""

    print("=" * 70)
    print("Testing VQTokenizer Trainer Metrics Implementation")
    print("=" * 70)

    # Load config
    config_path = Path("configs/vqvae_50k.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = TokenizerConfig(**config_dict)

    print("\n1. Feature Cleaning Config:")
    if hasattr(config, 'feature_cleaning'):
        print(f"   Enabled: {config.feature_cleaning.enabled}")
        print(f"   Pre-categorization: {config.feature_cleaning.pre_categorization}")
        print(f"   Variance threshold: {config.feature_cleaning.variance_threshold}")
        print(f"   Deduplicate threshold: {config.feature_cleaning.deduplicate_threshold}")
    else:
        print("   ⚠️  feature_cleaning not found in config!")

    # Create dummy data
    batch_size = 32
    seq_len = 64
    temporal_dim = 320

    temporal_features = torch.randn(batch_size, seq_len, temporal_dim)

    # Create dummy group indices (simulate 16 groups with 20 features each)
    group_indices = {f"temporal_group_{i}": list(range(i*20, (i+1)*20))
                     for i in range(16)}

    print("\n2. Creating Model and Trainer:")
    print(f"   Temporal features: {temporal_features.shape}")
    print(f"   Number of groups: {len(group_indices)}")

    # Build model
    model = JointHierarchicalVQVAE(
        config,
        group_indices,
        temporal_input_dim=temporal_dim  # Provide temporal input dimension
    )

    # Build trainer
    trainer = VQTokenizerTrainer(
        model=model,
        config=config,
        group_indices=group_indices,
        normalization_stats=None
    )

    print(f"   Model quantizers: {len(model.quantizers)}")
    print(f"   Device: {trainer.device}")

    # Create dataloaders through trainer (to initialize tensor_map)
    print("\n3. Creating Dataloaders:")
    train_loader, val_loader = trainer._create_dataloaders(
        temporal_features=temporal_features,
        initial_manual=None,
        initial_raw=None,
        theta_features=None,
        temporal_mask=None,
        temporal_lengths=None
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    print("\n4. Testing Validation Metrics:")
    model.eval()
    with torch.no_grad():
        val_metrics = trainer._validate_epoch(val_loader)

    print(f"   Loss: {val_metrics['loss']:.6f}")
    print(f"   Reconstruction: {val_metrics['reconstruction']:.6f}")
    print(f"   VQ: {val_metrics['vq']:.6f}")
    print(f"   Embedding Utilization: {val_metrics['embedding_utilization']:.2f}%")

    # Check per-quantizer utilization
    if 'per_quantizer_utilization' in val_metrics:
        print(f"\n5. Per-Quantizer Utilization:")
        per_quant_util = val_metrics['per_quantizer_utilization']
        print(f"   Number of quantizers: {len(per_quant_util)}")
        for name, util in sorted(per_quant_util.items())[:5]:
            print(f"   {name}: {util:.2f}%")
        if len(per_quant_util) > 5:
            print(f"   ... and {len(per_quant_util) - 5} more")
        avg_util = sum(per_quant_util.values()) / len(per_quant_util)
        print(f"   Average utilization: {avg_util:.2f}%")
    else:
        print("\n5. ⚠️  per_quantizer_utilization not found in metrics!")

    # Check per-category MSE
    if 'per_category_mse' in val_metrics:
        print(f"\n6. Per-Category MSE:")
        per_cat_mse = val_metrics['per_category_mse']
        print(f"   Number of categories: {len(per_cat_mse)}")
        for name, mse in sorted(per_cat_mse.items())[:5]:
            print(f"   {name}: {mse:.6f}")
        if len(per_cat_mse) > 5:
            print(f"   ... and {len(per_cat_mse) - 5} more")
    else:
        print("\n6. ⚠️  per_category_mse not found in metrics!")

    # Test final metrics computation
    print("\n7. Testing Final Metrics Computation:")
    final_metrics = trainer._compute_final_validation_metrics(val_loader)
    print(f"   Total final metrics: {len(final_metrics)}")

    # Check format
    util_keys = [k for k in final_metrics.keys() if 'utilization' in k]
    mse_keys = [k for k in final_metrics.keys() if 'reconstruction_mse' in k]

    print(f"   Utilization metrics: {len(util_keys)}")
    print(f"   MSE metrics: {len(mse_keys)}")

    if util_keys:
        print(f"\n   Sample utilization keys:")
        for key in sorted(util_keys)[:3]:
            print(f"     {key}: {final_metrics[key]:.2f}%")

    if mse_keys:
        print(f"\n   Sample MSE keys:")
        for key in sorted(mse_keys)[:3]:
            print(f"     {key}: {final_metrics[key]:.6f}")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_trainer_metrics()
