"""Quick verification script for diffusion baseline setup.

Checks that all components are working before starting full training.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.experimental.diffusion.models import DiscreteD3PM, DiffusionSchedule, DenoisingNetwork
from spinlock.experimental.diffusion.data import (
    HierarchicalMaskGenerator,
    MaskingStrategy,
    DiffusionCompletionDataset,
    collate_dict_batch,
)

logger = logging.getLogger(__name__)


def verify_tokenizer(config: dict):
    """Verify tokenizer loads and extracts vocab sizes correctly."""
    logger.info("\n" + "=" * 70)
    logger.info("1. Verifying VQTokenizer v2")
    logger.info("=" * 70)

    tokenizer_path = Path(config['dataset']['tokenizer_checkpoint'])
    logger.info(f"Loading from: {tokenizer_path}")

    # Load tokenizer
    tokenizer = VQTokenizer.from_checkpoint(tokenizer_path)
    logger.info("✓ Tokenizer loaded successfully")

    # Extract vocab sizes
    vocab_sizes = {}
    category_level_info = {}

    for quantizer_key, quantizer in tokenizer.model.quantizers.items():
        vocab_size = quantizer.embedding.weight.shape[0]
        vocab_sizes[quantizer_key] = vocab_size

        parts = quantizer_key.split('_')
        family = parts[0]
        level = int(parts[-1][1:])
        category = '_'.join(parts[1:-1])

        category_level_info[quantizer_key] = {
            'family': family,
            'category': category,
            'level': level,
        }

    logger.info(f"✓ Extracted {len(vocab_sizes)} category-levels")
    logger.info(f"  Vocab sizes: {set(vocab_sizes.values())}")
    logger.info(f"  Families: {set(info['family'] for info in category_level_info.values())}")

    # Group by level
    levels = {}
    for key, info in category_level_info.items():
        level = info['level']
        if level not in levels:
            levels[level] = []
        levels[level].append(key)

    for level in sorted(levels.keys()):
        logger.info(f"  L{level}: {len(levels[level])} category-levels")

    return vocab_sizes, category_level_info


def verify_dataset(config: dict, vocab_sizes: dict, category_level_info: dict):
    """Verify dataset loads and produces correct format."""
    logger.info("\n" + "=" * 70)
    logger.info("2. Verifying Dataset")
    logger.info("=" * 70)

    dataset_path = Path(config['dataset']['path'])
    logger.info(f"Loading from: {dataset_path}")

    # Create mask generator
    mask_gen = HierarchicalMaskGenerator(
        strategy=MaskingStrategy(config['masking']['strategy']),
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        mask_probability=config['masking'].get('mask_probability', 0.5),
        seed=config['masking'].get('seed', 42),
    )
    logger.info(f"✓ Mask generator created: {config['masking']['strategy']}")

    # Create dataset (without caching for quick check)
    logger.info("Creating dataset (no caching for quick check)...")
    dataset = DiffusionCompletionDataset(
        dataset_path=dataset_path,
        tokenizer_checkpoint=Path(config['dataset']['tokenizer_checkpoint']),
        mask_generator=mask_gen,
        cache_tokens=False,  # Don't cache for quick verification
        device='cpu',
    )
    logger.info(f"✓ Dataset created: {len(dataset)} samples")

    # Test single sample
    logger.info("Testing single sample...")
    sample = dataset[0]

    logger.info(f"✓ Sample structure:")
    logger.info(f"  tokens: {len(sample['tokens'])} category-levels")
    logger.info(f"  observed: {len(sample['observed'])} masks")
    logger.info(f"  target: {len(sample['target'])} masks")

    # Check shapes
    for key in list(sample['tokens'].keys())[:3]:
        logger.info(f"  {key}: shape={sample['tokens'][key].shape}")

    # Test batch collation
    logger.info("Testing batch collation...")
    batch = collate_dict_batch([dataset[i] for i in range(4)])
    logger.info(f"✓ Batch collation works:")
    for key in list(batch['tokens'].keys())[:3]:
        logger.info(f"  {key}: shape={batch['tokens'][key].shape}")

    return True


def verify_models(config: dict, vocab_sizes: dict, category_level_info: dict):
    """Verify models initialize and forward pass works."""
    logger.info("\n" + "=" * 70)
    logger.info("3. Verifying Models")
    logger.info("=" * 70)

    # Create diffusion model
    logger.info("Creating DiscreteD3PM...")
    diffusion_schedule = DiffusionSchedule(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule_type=config['diffusion']['schedule_type'],
    )
    diffusion = DiscreteD3PM(vocab_sizes, diffusion_schedule, category_level_info)
    logger.info(f"✓ DiscreteD3PM created: T={diffusion.schedule.num_timesteps}")

    # Create denoising network
    logger.info("Creating DenoisingNetwork...")
    model_config = config['model']
    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_hierarchical_guidance=model_config['use_hierarchical_guidance'],
        hierarchical_guidance_weight=model_config.get('hierarchical_guidance_weight', 0.1),
    )

    num_params = sum(p.numel() for p in denoiser.parameters())
    logger.info(f"✓ DenoisingNetwork created: {num_params:,} parameters")

    # Test forward pass
    logger.info("Testing forward pass...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion.to(device)
    denoiser.to(device)

    # Create dummy batch
    batch_size = 4
    tokens_dict = {}
    for key, vocab_size in vocab_sizes.items():
        tokens_dict[key] = torch.randint(0, vocab_size, (batch_size,), device=device)

    # Sample random timesteps
    t = torch.randint(0, diffusion.schedule.num_timesteps, (batch_size,), device=device)

    # Forward diffusion
    noisy_dict, _ = diffusion.forward_process(tokens_dict, t[0].item())
    logger.info("✓ Forward diffusion works")

    # Denoising
    predicted_logits = denoiser(noisy_dict, t)
    logger.info("✓ Denoising network works")

    # Check output shapes
    for key in list(predicted_logits.keys())[:3]:
        expected_shape = (batch_size, vocab_sizes[key])
        actual_shape = predicted_logits[key].shape
        assert actual_shape == expected_shape, f"Shape mismatch: {actual_shape} != {expected_shape}"
        logger.info(f"  {key}: {actual_shape} ✓")

    logger.info(f"✓ All forward passes work on {device}")

    return True


def verify_training_step(config: dict, vocab_sizes: dict, category_level_info: dict):
    """Verify a single training step works."""
    logger.info("\n" + "=" * 70)
    logger.info("4. Verifying Training Step")
    logger.info("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create models
    diffusion_schedule = DiffusionSchedule(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule_type=config['diffusion']['schedule_type'],
    )
    diffusion = DiscreteD3PM(vocab_sizes, diffusion_schedule, category_level_info).to(device)

    model_config = config['model']
    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        **model_config
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-4)

    # Create dummy batch
    batch_size = 4
    tokens_dict = {}
    observed_dict = {}
    target_dict = {}

    for key, vocab_size in vocab_sizes.items():
        tokens_dict[key] = torch.randint(0, vocab_size, (batch_size,), device=device)
        # 50% observed, 50% target
        mask = torch.rand(batch_size) > 0.5
        observed_dict[key] = mask.to(device)
        target_dict[key] = (~mask).to(device)

    # Training step
    logger.info("Running training step...")
    t = torch.randint(0, diffusion.schedule.num_timesteps, (batch_size,), device=device)

    # Forward
    noisy_dict, _ = diffusion.forward_process(tokens_dict, t[0].item(), mask_dict=target_dict)
    predicted_logits = denoiser(noisy_dict, t, observed_dict=observed_dict)

    # Compute loss
    total_loss = 0.0
    total_count = 0
    for key in predicted_logits.keys():
        logits = predicted_logits[key]
        targets = tokens_dict[key]
        mask = target_dict[key]
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        masked_loss = loss * mask.float()
        total_loss += masked_loss.sum()
        total_count += mask.sum()

    loss = total_loss / total_count if total_count > 0 else total_loss

    logger.info(f"  Loss: {loss.item():.4f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.info("✓ Training step works (forward + backward + optimizer)")

    return True


def main(args):
    """Main verification entry point."""
    logger.info("=" * 70)
    logger.info("Diffusion Baseline Verification")
    logger.info("=" * 70)

    # Load config
    config_path = Path(args.config)
    logger.info(f"Loading config: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run verification steps
    try:
        vocab_sizes, category_level_info = verify_tokenizer(config)
        verify_dataset(config, vocab_sizes, category_level_info)
        verify_models(config, vocab_sizes, category_level_info)
        verify_training_step(config, vocab_sizes, category_level_info)

        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL VERIFICATIONS PASSED")
        logger.info("=" * 70)
        logger.info("\nReady to start training:")
        logger.info(f"  python experiments/diffusion/scripts/train.py --config {config_path}")

    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error("❌ VERIFICATION FAILED")
        logger.error("=" * 70)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify diffusion baseline setup")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/diffusion/configs/baseline_diffusion.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()
    exit(main(args))
