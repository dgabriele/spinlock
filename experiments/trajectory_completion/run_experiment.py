#!/usr/bin/env python
"""
Trajectory Completion Experiment Runner

Usage:
    python -m experiments.trajectory_completion.run_experiment \
        --config experiments/trajectory_completion/baseline_50k/experiments/baseline.yaml
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from pydantic import BaseModel

from experiments.common.config.loader import load_experiment_config
from experiments.common.config.base import BaseExperimentConfig
from experiments.common.models.trained_vqvae import TrainedVQVAE
from experiments.trajectory_completion.data.masking import TemporalMaskGenerator, MaskingStrategy
from experiments.trajectory_completion.data.completion_dataset import TrajectoryCompletionDataset
from experiments.trajectory_completion.models.completion_model import TrajectoryCompletionModel
from experiments.trajectory_completion.training.trainer import CompletionTrainer


# Config schema for trajectory completion experiment
class MaskingConfig(BaseModel):
    strategy: MaskingStrategy
    start_percent: float = 0.3
    end_percent: float = 0.2


class ModelConfig(BaseModel):
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    hierarchical_guidance_weight: float = 0.1


class CompletionExperimentConfig(BaseExperimentConfig):
    masking: MaskingConfig
    model: ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True, help='Config YAML path')
    args = parser.parse_args()

    # Load config
    config = load_experiment_config(args.config, CompletionExperimentConfig)
    print(f"Experiment: {config.metadata.name}")
    print(f"Output: {config.output_dir}")

    # Set seed
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    # Load VQ-VAE
    print("Loading VQ-VAE...")
    vqvae = TrainedVQVAE(
        checkpoint_path=config.checkpoints.vqvae_path,
        device=config.training.device
    )

    # Create mask generator
    mask_generator = TemporalMaskGenerator(
        strategy=config.masking.strategy,
        start_percent=config.masking.start_percent,
        end_percent=config.masking.end_percent,
        seed=config.training.seed
    )

    # Create dataset
    print("Creating dataset...")
    full_dataset = TrajectoryCompletionDataset(
        dataset_path=config.data.dataset_path,
        vqvae=vqvae,
        mask_generator=mask_generator
    )

    # Train/val split
    num_samples = len(full_dataset)
    num_val = int(num_samples * config.data.val_split)
    num_train = num_samples - num_val

    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("Creating model...")
    # Get codebook sizes from VQ-VAE
    num_tokens_per_level = vqvae.codebook_sizes

    model = TrajectoryCompletionModel(
        num_tokens_per_level=num_tokens_per_level,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        hierarchical_guidance_weight=config.model.hierarchical_guidance_weight
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = CompletionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config.output_dir,
        vqvae=vqvae
    )

    # Train
    print("Starting training...")
    history = trainer.train(epochs=config.training.epochs)

    print("Training complete!")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Results saved to: {config.output_dir}")


if __name__ == '__main__':
    main()
