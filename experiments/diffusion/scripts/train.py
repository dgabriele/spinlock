"""Training script for discrete diffusion token completion."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.experimental.common.config.loader import load_experiment_config
from spinlock.experimental.diffusion.config import DiffusionExperimentConfig
from spinlock.experimental.diffusion.models import DiscreteD3PM, DiffusionSchedule, DenoisingNetwork
from spinlock.experimental.diffusion.data import (
    HierarchicalMaskGenerator,
    MaskingStrategy,
    DiffusionCompletionDataset,
    PretokenizedDiffusionDataset,
    collate_dict_batch,
)
from spinlock.experimental.diffusion.training import DiffusionTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Config loading now handled by load_experiment_config from common.config.loader


def extract_vocab_sizes_and_info(tokenizer_path: Path) -> tuple[dict, dict]:
    """Extract vocab sizes and category-level info from tokenizer.

    Args:
        tokenizer_path: Path to VQTokenizer v2 checkpoint

    Returns:
        Tuple of (vocab_sizes, category_level_info)
    """
    tokenizer = VQTokenizer.from_checkpoint(tokenizer_path)

    vocab_sizes = {}
    category_level_info = {}

    for quantizer_key, quantizer in tokenizer.model.quantizers.items():
        vocab_size = quantizer.embedding.weight.shape[0]
        vocab_sizes[quantizer_key] = vocab_size

        # Parse key: "family_category_Ll" → {family, category, level}
        parts = quantizer_key.split('_')
        family = parts[0]  # "temporal" or "initial"
        level = int(parts[-1][1:])  # "L0" → 0
        category = '_'.join(parts[1:-1])  # "group_1"

        category_level_info[quantizer_key] = {
            'family': family,
            'category': category,
            'level': level,
        }

    logger.info(
        f"Extracted vocab sizes: {len(vocab_sizes)} category-levels, "
        f"sizes={list(set(vocab_sizes.values()))}"
    )

    return vocab_sizes, category_level_info


def extract_vocab_sizes_from_pretokenized(tokenized_path: Path) -> tuple[dict, dict]:
    """Extract vocab sizes from pre-tokenized dataset.

    Args:
        tokenized_path: Path to pre-tokenized HDF5 file

    Returns:
        Tuple of (vocab_sizes, category_level_info)
    """
    import h5py

    vocab_sizes = {}
    category_level_info = {}

    with h5py.File(tokenized_path, 'r') as f:
        tokens_group = f['tokens']

        for key in tokens_group.keys():
            # Get vocab size from max token value + 1
            tokens = tokens_group[key][:]
            vocab_size = int(tokens.max()) + 1
            vocab_sizes[key] = vocab_size

            # Parse key: "family_category_Ll" → {family, category, level}
            parts = key.split('_')
            family = parts[0]  # "temporal" or "initial"
            level = int(parts[-1][1:])  # "L0" → 0
            category = '_'.join(parts[1:-1])  # "group_1"

            category_level_info[key] = {
                'family': family,
                'category': category,
                'level': level,
            }

    logger.info(
        f"Extracted vocab sizes from pre-tokenized dataset: {len(vocab_sizes)} category-levels, "
        f"sizes={list(set(vocab_sizes.values()))}"
    )

    return vocab_sizes, category_level_info


def create_datasets(config: DiffusionExperimentConfig, mask_generator: HierarchicalMaskGenerator):
    """Create train and validation datasets."""
    # Check if using pre-tokenized data
    if config.dataset.use_pretokenized:
        logger.info("Using pre-tokenized dataset (fast mode)")
        full_dataset = PretokenizedDiffusionDataset(
            tokenized_dataset_path=config.dataset.tokenized_path,
            mask_generator=mask_generator,
        )
    else:
        logger.info("Using on-the-fly tokenization (slow mode)")
        full_dataset = DiffusionCompletionDataset(
            dataset_path=config.dataset.path,
            tokenizer_checkpoint=config.dataset.tokenizer_checkpoint,
            mask_generator=mask_generator,
            cache_tokens=config.dataset.cache_tokens,
            max_cache_size=config.dataset.max_cache_size,
            device=config.dataset.device,
        )

    # Split into train/val
    val_size = int(len(full_dataset) * config.training.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    logger.info(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}")

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: DiffusionExperimentConfig):
    """Create train and validation dataloaders."""
    num_workers = config.training.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_dict_batch,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,  # Only pin memory with workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.val_batch_size,
        shuffle=False,
        collate_fn=collate_dict_batch,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )

    return train_loader, val_loader


def main(args):
    """Main training entry point."""
    # Load and validate config
    logger.info(f"Loading config from {args.config}")
    config = load_experiment_config(args.config, DiffusionExperimentConfig)

    # Set seed
    torch.manual_seed(config.seed)

    # Extract vocab sizes
    if config.dataset.use_pretokenized:
        logger.info("Extracting vocab sizes from pre-tokenized dataset")
        vocab_sizes, category_level_info = extract_vocab_sizes_from_pretokenized(
            config.dataset.tokenized_path
        )
    else:
        logger.info("Extracting vocab sizes from tokenizer")
        vocab_sizes, category_level_info = extract_vocab_sizes_and_info(
            config.dataset.tokenizer_checkpoint
        )

    # Create mask generator
    logger.info(f"Creating mask generator: strategy={config.masking.strategy}")
    mask_generator = HierarchicalMaskGenerator(
        strategy=MaskingStrategy(config.masking.strategy),
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        mask_probability=config.masking.mask_probability,
        seed=config.masking.seed,
    )

    # Create datasets
    logger.info("Creating datasets")
    train_dataset, val_dataset = create_datasets(config, mask_generator)

    # Create dataloaders
    logger.info("Creating dataloaders")
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    # Create diffusion model
    logger.info("Creating discrete D3PM diffusion model")
    diffusion_schedule = DiffusionSchedule(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule_type=config.diffusion.schedule_type,
    )
    diffusion = DiscreteD3PM(vocab_sizes, diffusion_schedule, category_level_info)

    # Create denoising network
    logger.info("Creating denoising network")
    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        use_hierarchical_guidance=config.model.use_hierarchical_guidance,
        hierarchical_guidance_weight=config.model.hierarchical_guidance_weight,
    )

    # Log model size
    num_params = sum(p.numel() for p in denoiser.parameters())
    logger.info(f"Denoising network: {num_params:,} parameters")

    # Create trainer
    logger.info("Creating trainer")
    trainer = DiffusionTrainer(
        diffusion_model=diffusion,
        denoising_network=denoiser,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config.output.dir,
        device=config.device,
    )

    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Train
    logger.info(f"Starting training for {config.training.num_epochs} epochs")
    history = trainer.train(num_epochs=config.training.num_epochs)

    # Save final checkpoint
    trainer.save_checkpoint(is_best=False)

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train discrete diffusion for token completion")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()
    main(args)
