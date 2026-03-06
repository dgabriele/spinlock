"""Training loop for Token-Conditioned CVAE.

This module provides the trainer class for TokenConditionedCVAE with:
- CVAE loss (reconstruction + KL divergence)
- KL annealing schedule (prevents posterior collapse)
- Checkpointing and logging

Key difference from TokenToRolloutVAETrainer: the CVAE forward pass provides
both tokens AND targets (theta, grids) to the model, so the recognition
network can learn q(z | theta, IC, tokens).
"""

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from spinlock.tokens.cvae import TokenConditionedCVAE
from spinlock.tokens.cvae_config import TokenConditionedCVAEConfig
from spinlock.tokens.cvae_dataset import CVAEDataset, collate_cvae_batch


def cvae_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute CVAE loss: reconstruction + KL divergence.

    Args:
        outputs: Model outputs containing theta, grids, mu, logvar
        targets: Ground truth containing theta, grids
        beta: Weight for KL divergence term (for annealing)

    Returns:
        Dictionary containing:
        - total_loss: Combined loss
        - theta_loss: MSE on parameters
        - grid_loss: MSE on grids
        - kl_loss: KL divergence
    """
    theta_loss = nn.functional.mse_loss(outputs["theta"], targets["theta"])
    grid_loss = nn.functional.mse_loss(outputs["grids"], targets["grids"])

    # KL(q(z|x,c) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()

    total_loss = theta_loss + grid_loss + beta * kl_loss

    return {
        "total_loss": total_loss,
        "theta_loss": theta_loss,
        "grid_loss": grid_loss,
        "kl_loss": kl_loss,
    }


class CVAETrainer:
    """Trainer for TokenConditionedCVAE.

    Args:
        config: Training configuration
        theta_dim: Dimensionality of theta parameters (from dataset)
        grid_shape: Shape of initial grids (from dataset)
    """

    def __init__(
        self,
        config: TokenConditionedCVAEConfig,
        theta_dim: int,
        grid_shape: tuple[int, int, int],
    ):
        self.config = config
        self.theta_dim = theta_dim
        self.grid_shape = grid_shape
        self.device = torch.device(config.device)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.to_yaml(self.output_dir / "config.yaml")

        # Set random seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # Initialize model
        model_cfg = config.model
        cond_cfg = model_cfg.condition
        tgt_cfg = model_cfg.target_encoder

        self.model = TokenConditionedCVAE(
            vq_checkpoint=config.data.vq_checkpoint,
            theta_dim=theta_dim,
            grid_shape=grid_shape,
            latent_dim=model_cfg.latent_dim,
            group_mlp_hidden_dim=cond_cfg.group_mlp_hidden_dim,
            group_mlp_output_dim=cond_cfg.group_mlp_output_dim,
            theta_hidden_dim=tgt_cfg.theta_hidden_dim,
            ic_hidden_dim=tgt_cfg.ic_hidden_dim,
            ic_channels=tgt_cfg.ic_channels,
            encoder_hidden_dims=model_cfg.encoder.hidden_dims,
            param_decoder_hidden_dims=model_cfg.param_decoder.hidden_dims,
            grid_decoder_hidden_channels=model_cfg.grid_decoder.hidden_channels,
            dropout=model_cfg.encoder.dropout,
        ).to(self.device)

        # Initialize optimizer
        if config.training.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        elif config.training.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

        # Initialize scheduler
        if config.training.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.training.num_epochs
            )
        elif config.training.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.5
            )
        else:
            self.scheduler = None

        # Tracking
        self.best_val_loss = float("inf")
        self.epoch = 0
        self.train_history = []
        self.val_history = []

    def get_beta(self, epoch: int) -> float:
        """Get KL divergence weight for current epoch (annealing schedule).

        Args:
            epoch: Current epoch number

        Returns:
            Beta value in [0, beta_max]
        """
        if self.config.training.beta_schedule == "constant":
            return self.config.training.beta_max

        warmup_epochs = self.config.training.beta_warmup_epochs
        if epoch < warmup_epochs:
            return self.config.training.beta_max * (epoch / warmup_epochs)
        return self.config.training.beta_max

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Key difference from standard VAE: forward pass provides tokens AND targets.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        total_losses = {"total_loss": 0.0, "theta_loss": 0.0, "grid_loss": 0.0, "kl_loss": 0.0}
        n_batches = 0

        beta = self.get_beta(self.epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch+1}/{self.config.training.num_epochs}")
        for batch in pbar:
            # Move to device
            tokens = {k: v.to(self.device) for k, v in batch["tokens"].items()}
            theta = batch["theta"].to(self.device)
            grids = batch["grids"].to(self.device)

            # CVAE forward: encoder sees BOTH tokens AND targets
            outputs = self.model(tokens, theta, grids)

            # Compute loss
            losses = cvae_loss(
                outputs,
                {"theta": theta, "grids": grids},
                beta=beta,
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses["total_loss"].backward()

            # Gradient clipping
            if self.config.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_norm
                )

            self.optimizer.step()

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            n_batches += 1

            pbar.set_postfix({
                "loss": losses["total_loss"].item(),
                "theta": losses["theta_loss"].item(),
                "grid": losses["grid_loss"].item(),
                "kl": losses["kl_loss"].item(),
                "beta": beta,
            })

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of average losses
        """
        self.model.eval()
        total_losses = {"total_loss": 0.0, "theta_loss": 0.0, "grid_loss": 0.0, "kl_loss": 0.0}
        n_batches = 0

        beta = self.get_beta(self.epoch)

        for batch in tqdm(val_loader, desc="Validating"):
            tokens = {k: v.to(self.device) for k, v in batch["tokens"].items()}
            theta = batch["theta"].to(self.device)
            grids = batch["grids"].to(self.device)

            outputs = self.model(tokens, theta, grids)

            losses = cvae_loss(
                outputs,
                {"theta": theta, "grids": grids},
                beta=beta,
            )

            for key in total_losses:
                total_losses[key] += losses[key].item()
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> None:
        """Save training checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "config": self.config.model_dump(),
            "theta_dim": self.theta_dim,
            "grid_shape": self.grid_shape,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_history = checkpoint["train_history"]
        self.val_history = checkpoint["val_history"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def train(self) -> None:
        """Run full training loop."""
        print("Training TokenConditionedCVAE")
        print(f"  Model dimensions:")
        print(f"    - theta_dim: {self.theta_dim}")
        print(f"    - grid_shape: {self.grid_shape}")
        print(f"    - latent_dim: {self.config.model.latent_dim}")
        print(f"    - condition_dim: {self.config.model.condition.group_mlp_output_dim}")
        print(f"    - temporal_keys: {self.model.conditioner.num_temporal_keys}")
        print(f"  Training config:")
        print(f"    - num_epochs: {self.config.training.num_epochs}")
        print(f"    - batch_size: {self.config.training.batch_size}")
        print(f"    - learning_rate: {self.config.training.learning_rate}")
        print(f"    - beta_schedule: {self.config.training.beta_schedule}")
        print(f"  Output: {self.output_dir}")

        # Load datasets
        print("\nLoading datasets...")
        train_dataset, val_dataset = CVAEDataset.create_splits(
            self.config.data.dataset,
            self.config.data.tokenized_dataset,
            temporal_keys_only=self.config.data.temporal_keys_only,
            train_split=self.config.data.train_split,
            seed=self.config.seed,
        )

        print(f"  Train size: {len(train_dataset)}")
        print(f"  Val size: {len(val_dataset)}")
        print(f"  Temporal token keys: {train_dataset.num_temporal_keys}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_cvae_batch,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=collate_cvae_batch,
            num_workers=4,
            pin_memory=True,
        )

        # Training loop
        print("\nStarting training...")
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch

            train_losses = self.train_epoch(train_loader)
            self.train_history.append(train_losses)

            if (epoch + 1) % self.config.validation.freq_epochs == 0:
                val_losses = self.validate(val_loader)
                self.val_history.append(val_losses)

                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train - Total: {train_losses['total_loss']:.6f}, "
                      f"Theta: {train_losses['theta_loss']:.6f}, "
                      f"Grid: {train_losses['grid_loss']:.6f}, "
                      f"KL: {train_losses['kl_loss']:.6f}")
                print(f"  Val   - Total: {val_losses['total_loss']:.6f}, "
                      f"Theta: {val_losses['theta_loss']:.6f}, "
                      f"Grid: {val_losses['grid_loss']:.6f}, "
                      f"KL: {val_losses['kl_loss']:.6f}")

                if val_losses["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total_loss"]
                    self.save_checkpoint("best.pt")
                    print(f"  -> Saved best model (val_loss: {self.best_val_loss:.6f})")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model
        self.save_checkpoint("final.pt")

        # Save training history
        with open(self.output_dir / "train_history.json", "w") as f:
            json.dump(
                {"train": self.train_history, "val": self.val_history},
                f,
                indent=2,
            )

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.6f}")
        print(f"Checkpoints saved to: {self.output_dir}")
