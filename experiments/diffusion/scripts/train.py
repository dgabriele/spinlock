"""Training script for discrete diffusion token completion."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.tokens.schema import TokenSchema
from spinlock.experimental.common.config.loader import load_experiment_config
from spinlock.experimental.diffusion.config import DiffusionExperimentConfig
from spinlock.experimental.diffusion.models import (
    DiscreteD3PM,
    DiffusionSchedule,
    DenoisingNetwork,
    TemporalResolutionDenoisingNetwork,
)
from spinlock.experimental.diffusion.data import (
    HierarchicalMaskGenerator,
    MixedMaskGenerator,
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

    Delegates to TokenSchema.from_tokenizer() for the actual parsing.

    Args:
        tokenizer_path: Path to VQTokenizer v2 checkpoint

    Returns:
        Tuple of (vocab_sizes, category_level_info) in dict format
    """
    tokenizer = VQTokenizer.from_checkpoint(tokenizer_path)
    schema = TokenSchema.from_tokenizer(tokenizer)
    return schema.vocab_sizes_dict(), schema.category_level_info_dict()


def extract_vocab_sizes_from_pretokenized(tokenized_path: Path) -> tuple[dict, dict]:
    """Extract vocab sizes from pre-tokenized dataset.

    Delegates to TokenSchema.from_pretokenized() for the actual parsing.

    Args:
        tokenized_path: Path to pre-tokenized HDF5 file

    Returns:
        Tuple of (vocab_sizes, category_level_info) in dict format
    """
    schema = TokenSchema.from_pretokenized(tokenized_path)
    return schema.vocab_sizes_dict(), schema.category_level_info_dict()


def load_truncation_lengths_from_dataset(dataset_path: Path) -> list[int] | None:
    """Load truncation lengths from temporal resolution dataset.

    Args:
        dataset_path: Path to pre-tokenized HDF5 with temporal resolution

    Returns:
        Sorted list of truncation lengths or None if not a temporal resolution dataset
    """
    import h5py

    try:
        with h5py.File(dataset_path, 'r') as f:
            if "temporal_resolution_mode" not in f.attrs or not f.attrs["temporal_resolution_mode"]:
                logger.warning(
                    f"Dataset {dataset_path} is not in temporal resolution format. "
                    f"Did you forget --temporal-resolution flag during pretokenization?"
                )
                return None

            truncation_lengths = list(f.attrs["truncation_lengths"])
            logger.info(f"Loaded truncation lengths from dataset: {truncation_lengths}")
            return truncation_lengths

    except Exception as e:
        logger.error(f"Failed to load truncation lengths from dataset: {e}")
        return None


def create_datasets(
    config: DiffusionExperimentConfig,
    mask_generator: HierarchicalMaskGenerator,
    max_samples: int = None,
):
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

    # Truncate dataset if max_samples specified
    if max_samples is not None and max_samples < len(full_dataset):
        logger.info(f"Limiting dataset to {max_samples}/{len(full_dataset)} samples")
        full_dataset = Subset(full_dataset, list(range(max_samples)))

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

    # Extract vocab sizes — prefer tokenizer codebook sizes (authoritative)
    # over pretokenized max+1 (which underestimates for underutilized codes).
    # Exception: temporal-resolution pretokenized datasets have _trunc_T* keys
    # not present in the tokenizer schema; must use pretokenized schema in that case.
    temporal_res_mode = False
    if config.dataset.use_pretokenized and config.dataset.tokenized_path:
        import h5py as _h5py
        with _h5py.File(config.dataset.tokenized_path, "r") as _f:
            temporal_res_mode = bool(_f.attrs.get("temporal_resolution_mode", False))

    if config.dataset.tokenizer_checkpoint is not None and not temporal_res_mode:
        logger.info(
            "Extracting vocab sizes from tokenizer (authoritative codebook sizes)"
        )
        vocab_sizes, category_level_info = extract_vocab_sizes_and_info(
            config.dataset.tokenizer_checkpoint
        )
    elif config.dataset.use_pretokenized:
        if temporal_res_mode:
            logger.info(
                "Temporal-resolution pretokenized dataset detected — "
                "using pretokenized schema for vocab sizes (includes _trunc_T* keys)"
            )
        else:
            logger.warning(
                "No tokenizer_checkpoint provided — inferring vocab sizes from "
                "pretokenized data (max+1). This may underestimate vocab sizes "
                "for codebooks with unused entries."
            )
        vocab_sizes, category_level_info = extract_vocab_sizes_from_pretokenized(
            config.dataset.tokenized_path
        )
    else:
        raise ValueError(
            "Either tokenizer_checkpoint or use_pretokenized must be set"
        )

    # Create mask generator
    logger.info(f"Creating mask generator: strategy={config.masking.strategy}")
    if config.masking.strategy == MaskingStrategy.MIXED:
        if not config.masking.strategies:
            raise ValueError("masking.strategies list required when strategy='mixed'")
        total_weight = sum(e.weight for e in config.masking.strategies)
        if abs(total_weight - 1.0) > 1e-4:
            raise ValueError(f"masking.strategies weights must sum to 1.0, got {total_weight:.4f}")
        mask_generator = MixedMaskGenerator(
            strategies=[
                (MaskingStrategy(e.name), e.weight)
                for e in config.masking.strategies
            ],
            vocab_sizes=vocab_sizes,
            category_level_info=category_level_info,
            mask_probability=config.masking.mask_probability,
            seed=config.masking.seed,
        )
    else:
        mask_generator = HierarchicalMaskGenerator(
            strategy=MaskingStrategy(config.masking.strategy),
            vocab_sizes=vocab_sizes,
            category_level_info=category_level_info,
            mask_probability=config.masking.mask_probability,
            seed=config.masking.seed,
        )

    # Create datasets
    logger.info("Creating datasets")
    max_samples = getattr(args, 'max_samples', None)
    train_dataset, val_dataset = create_datasets(config, mask_generator, max_samples=max_samples)

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
    graded_cfg = config.diffusion.graded_schedule
    diffusion = DiscreteD3PM(
        vocab_sizes,
        diffusion_schedule,
        category_level_info,
        transition_type=config.diffusion.transition_type,
        beta_scaling=config.diffusion.beta_scaling,
        graded_schedule_enabled=graded_cfg.enabled,
        graded_scale_factors=graded_cfg.scale_factors,
        graded_min_scale=graded_cfg.min_scale,
        non_temporal_scale=graded_cfg.non_temporal_scale,
    )

    # Create denoising network
    if temporal_res_mode and config.model.temporal_resolution.enabled:
        # Load truncation lengths from dataset
        truncation_lengths = load_truncation_lengths_from_dataset(
            config.dataset.tokenized_path
        )
        if truncation_lengths is None:
            logger.error("Failed to load truncation lengths")
            return

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
        if tr_config.use_temporal_bias:
            bias_matrix = denoiser.get_temporal_bias_matrix()
            logger.info(
                f"Temporal attention bias [{bias_matrix.shape[0]}x{bias_matrix.shape[1]}]:\n"
                f"{bias_matrix}"
            )
    else:
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
            guidance_mode=config.model.hierarchical_guidance_mode,
            transition_type=config.diffusion.transition_type,
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset to first N samples (for smoke tests)",
    )

    args = parser.parse_args()
    main(args)
