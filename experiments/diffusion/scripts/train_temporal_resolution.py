"""Training script for temporal resolution D3PM.

Extends standard diffusion training to support multi-truncation tokenizations
where tokens are generated at multiple trajectory truncation points.

Key differences from train.py:
- Loads truncation lengths from temporal resolution dataset
- Uses TemporalResolutionDenoisingNetwork when enabled
- Validates dataset is in temporal resolution format

Usage:
    python experiments/diffusion/scripts/train_temporal_resolution.py \
        --config experiments/diffusion/configs/temporal_resolution.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

import h5py
import torch
from torch.utils.data import DataLoader, random_split

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.tokens.schema import TokenSchema
from spinlock.experimental.common.config.loader import load_experiment_config
from spinlock.experimental.diffusion.config import DiffusionExperimentConfig
from spinlock.experimental.diffusion.models import DiscreteD3PM, DiffusionSchedule, DenoisingNetwork
from spinlock.experimental.diffusion.data import (
    HierarchicalMaskGenerator,
    MaskingStrategy,
    PretokenizedDiffusionDataset,
    collate_dict_batch,
)
from spinlock.experimental.diffusion.training import DiffusionTrainer

# Import temporal resolution denoising network
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from temporal_resolution_denoising_network import TemporalResolutionDenoisingNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_truncation_lengths_from_dataset(dataset_path: Path) -> Optional[List[int]]:
    """Load truncation lengths from temporal resolution dataset.

    Args:
        dataset_path: Path to pre-tokenized HDF5 with temporal resolution

    Returns:
        Sorted list of truncation lengths or None if not a temporal resolution dataset
    """
    try:
        with h5py.File(dataset_path, 'r') as f:
            if "temporal_resolution_mode" not in f.attrs or not f.attrs["temporal_resolution_mode"]:
                logger.warning(
                    f"Dataset {dataset_path} is not in temporal resolution format. "
                    f"Did you forget --temporal-resolution flag during pretokenization?"
                )
                return None

            truncation_lengths = list(f.attrs["truncation_lengths"])
            logger.info(f"✓ Loaded truncation lengths from dataset: {truncation_lengths}")
            return truncation_lengths

    except Exception as e:
        logger.error(f"Failed to load truncation lengths from dataset: {e}")
        return None


def extract_vocab_sizes_and_info(tokenizer_path: Path) -> tuple[dict, dict]:
    """Extract vocab sizes and category-level info from tokenizer.

    Args:
        tokenizer_path: Path to VQTokenizer checkpoint

    Returns:
        Tuple of (vocab_sizes, category_level_info)
    """
    tokenizer = VQTokenizer.from_checkpoint(tokenizer_path)
    schema = TokenSchema.from_tokenizer(tokenizer)
    return schema.vocab_sizes_dict(), schema.category_level_info_dict()


def extract_vocab_sizes_from_pretokenized(tokenized_path: Path) -> tuple[dict, dict]:
    """Extract vocab sizes from pre-tokenized dataset.

    Args:
        tokenized_path: Path to pre-tokenized HDF5

    Returns:
        Tuple of (vocab_sizes, category_level_info)
    """
    schema = TokenSchema.from_pretokenized(tokenized_path)
    return schema.vocab_sizes_dict(), schema.category_level_info_dict()


def create_datasets(config: DiffusionExperimentConfig, mask_generator: HierarchicalMaskGenerator):
    """Create train and validation datasets."""
    if not config.dataset.use_pretokenized:
        raise ValueError(
            "Temporal resolution training requires pre-tokenized dataset. "
            "Please run: spinlock pretokenize-dataset --temporal-resolution"
        )

    logger.info("Loading pre-tokenized temporal resolution dataset")
    full_dataset = PretokenizedDiffusionDataset(
        tokenized_dataset_path=config.dataset.tokenized_path,
        mask_generator=mask_generator,
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
        pin_memory=True if num_workers > 0 else False,
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

    # Check temporal resolution mode is enabled
    if not config.model.temporal_resolution.enabled:
        logger.error(
            "Temporal resolution mode not enabled in config! "
            "Set model.temporal_resolution.enabled: true"
        )
        return 1

    # Set seed
    torch.manual_seed(config.seed)

    # Load truncation lengths from dataset
    truncation_lengths = load_truncation_lengths_from_dataset(config.dataset.tokenized_path)
    if truncation_lengths is None:
        logger.error("Failed to load truncation lengths from dataset")
        return 1

    # Extract vocab sizes from tokenizer (authoritative)
    if config.dataset.tokenizer_checkpoint is not None:
        logger.info("Extracting vocab sizes from tokenizer (authoritative)")
        vocab_sizes, category_level_info = extract_vocab_sizes_and_info(
            config.dataset.tokenizer_checkpoint
        )
    else:
        logger.warning(
            "No tokenizer_checkpoint provided — inferring vocab sizes from pretokenized data"
        )
        vocab_sizes, category_level_info = extract_vocab_sizes_from_pretokenized(
            config.dataset.tokenized_path
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
    diffusion = DiscreteD3PM(
        vocab_sizes,
        diffusion_schedule,
        category_level_info,
        transition_type=config.diffusion.transition_type,
        beta_scaling=config.diffusion.beta_scaling,
    )

    # Create temporal resolution denoising network
    logger.info("Creating temporal resolution denoising network")
    tr_config = config.model.temporal_resolution
    denoiser = TemporalResolutionDenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        truncation_lengths=truncation_lengths,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        use_hierarchical_guidance=config.model.use_hierarchical_guidance,
        hierarchical_guidance_weight=config.model.hierarchical_guidance_weight,
        guidance_mode=config.model.hierarchical_guidance_mode,
        transition_type=config.diffusion.transition_type,
        use_temporal_bias=tr_config.use_temporal_bias,
        temporal_bias_init=tr_config.temporal_bias_init,
        temporal_bias_strength=tr_config.temporal_bias_strength,
        enforce_causality=tr_config.enforce_causality,
    )

    # Log model size
    num_params = sum(p.numel() for p in denoiser.parameters())
    logger.info(f"Temporal Resolution Denoising Network: {num_params:,} parameters")

    # Log temporal bias matrix initialization
    if tr_config.use_temporal_bias:
        bias_matrix = denoiser.get_temporal_bias_matrix()
        logger.info(f"Temporal attention bias matrix [{bias_matrix.shape[0]}, {bias_matrix.shape[1]}]:")
        logger.info(f"\n{bias_matrix}")

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
    logger.info(f"Expected training time: ~{config.training.num_epochs * 0.32:.1f} hours (4x baseline)")
    history = trainer.train(num_epochs=config.training.num_epochs)

    # Save final checkpoint
    trainer.save_checkpoint(is_best=False)

    # Log final temporal bias matrix
    if tr_config.use_temporal_bias:
        final_bias = denoiser.get_temporal_bias_matrix()
        logger.info(f"\nFinal temporal attention bias matrix:")
        logger.info(f"\n{final_bias}")

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train temporal resolution D3PM with multi-truncation diffusion"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to temporal resolution config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()
    sys.exit(main(args))
