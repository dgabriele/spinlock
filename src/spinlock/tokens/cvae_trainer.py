"""Training loop for Token-Conditioned CVAE.

This module provides the trainer class for TokenConditionedCVAE with:
- CVAE loss (reconstruction + KL divergence with free bits)
- KL annealing schedule (prevents posterior collapse)
- Topological preservation metrics (trustworthiness, effective dim)
- Checkpointing and logging

Key difference from TokenToRolloutVAETrainer: the CVAE forward pass provides
both tokens AND targets (theta, grids) to the model, so the recognition
network can learn q(z | theta, IC, tokens).
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
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
    free_bits: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute CVAE loss: reconstruction + KL divergence with free bits.

    Free bits (Kingma et al. 2016): each latent dimension is allowed to carry
    at least `free_bits` nats of information before the KL penalty applies.
    This prevents posterior collapse by zeroing gradients on dimensions whose
    KL is already below the threshold.

    Args:
        outputs: Model outputs containing theta, grids, mu, logvar
        targets: Ground truth containing theta, grids
        beta: Weight for KL divergence term (for annealing)
        free_bits: Per-dimension KL floor in nats (0 = disabled)

    Returns:
        Dictionary containing:
        - total_loss: Combined loss
        - theta_loss: MSE on parameters
        - grid_loss: MSE on grids
        - kl_loss: KL divergence (raw, before free bits)
        - kl_loss_effective: KL after free bits clamping (used in total_loss)
    """
    theta_loss = nn.functional.mse_loss(outputs["theta"], targets["theta"])
    grid_loss = nn.functional.mse_loss(outputs["grids"], targets["grids"])

    # Per-dimension KL: [B, D]
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Raw KL for logging (sum over dims, mean over batch)
    kl_loss = kl_per_dim.sum(dim=1).mean()

    # Free bits: clamp each dimension's KL to at least free_bits.
    # Dimensions below the threshold get zero gradient (no collapse pressure).
    if free_bits > 0:
        kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
        kl_effective = kl_per_dim_clamped.sum(dim=1).mean()
    else:
        kl_effective = kl_loss

    total_loss = theta_loss + grid_loss + beta * kl_effective

    return {
        "total_loss": total_loss,
        "theta_loss": theta_loss,
        "grid_loss": grid_loss,
        "kl_loss": kl_loss,
        "kl_loss_effective": kl_effective,
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

        grid_cfg = model_cfg.grid_decoder
        self.model = TokenConditionedCVAE(
            vq_checkpoint=config.data.vq_checkpoint,
            theta_dim=theta_dim,
            grid_shape=grid_shape,
            latent_dim=model_cfg.latent_dim,
            group_mlp_hidden_dim=cond_cfg.group_mlp_hidden_dim,
            group_mlp_output_dim=cond_cfg.group_mlp_output_dim,
            pooling=cond_cfg.pooling,
            theta_hidden_dim=tgt_cfg.theta_hidden_dim,
            ic_hidden_dim=tgt_cfg.ic_hidden_dim,
            ic_channels=tgt_cfg.ic_channels,
            encoder_hidden_dims=model_cfg.encoder.hidden_dims,
            param_decoder_hidden_dims=model_cfg.param_decoder.hidden_dims,
            grid_decoder_hidden_channels=grid_cfg.hidden_channels,
            grid_decoder_type=grid_cfg.type,
            grid_decoder_num_modes=grid_cfg.num_modes,
            grid_decoder_spectral_hidden_dims=grid_cfg.spectral_hidden_dims,
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
        total_losses = {
            "total_loss": 0.0, "theta_loss": 0.0, "grid_loss": 0.0,
            "kl_loss": 0.0, "kl_loss_effective": 0.0,
        }
        n_batches = 0

        beta = self.get_beta(self.epoch)
        free_bits = self.config.training.free_bits

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
                free_bits=free_bits,
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
                "kl_eff": losses["kl_loss_effective"].item(),
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
        total_losses = {
            "total_loss": 0.0, "theta_loss": 0.0, "grid_loss": 0.0,
            "kl_loss": 0.0, "kl_loss_effective": 0.0,
        }
        n_batches = 0

        beta = self.get_beta(self.epoch)
        free_bits = self.config.training.free_bits

        for batch in tqdm(val_loader, desc="Validating"):
            tokens = {k: v.to(self.device) for k, v in batch["tokens"].items()}
            theta = batch["theta"].to(self.device)
            grids = batch["grids"].to(self.device)

            outputs = self.model(tokens, theta, grids)

            losses = cvae_loss(
                outputs,
                {"theta": theta, "grids": grids},
                beta=beta,
                free_bits=free_bits,
            )

            for key in total_losses:
                total_losses[key] += losses[key].item()
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def compute_topo_metrics(
        self, val_loader: DataLoader, n_samples: int = 1024, k: int = 10
    ) -> Dict[str, float]:
        """Compute topological preservation metrics on validation data.

        Measures whether the CVAE latent space preserves the neighborhood
        structure of the input parameter space:
        - Trustworthiness: are theta-space neighbors also z-space neighbors?
        - Effective dimensionality: how many latent dims carry information?
        - Mean pairwise distance: is the latent space collapsed or spread out?

        Args:
            val_loader: Validation data loader
            n_samples: Number of samples to evaluate (subset for speed)
            k: Number of neighbors for trustworthiness

        Returns:
            Dictionary of topological metrics
        """
        from sklearn.manifold import trustworthiness
        from sklearn.decomposition import PCA

        self.model.eval()
        mus = []
        thetas_true = []

        for batch in val_loader:
            tokens = {k_: v.to(self.device) for k_, v in batch["tokens"].items()}
            theta = batch["theta"].to(self.device)
            grids = batch["grids"].to(self.device)

            outputs = self.model(tokens, theta, grids)
            mus.append(outputs["mu"].cpu())
            thetas_true.append(theta.cpu())

            if sum(m.shape[0] for m in mus) >= n_samples:
                break

        mu = torch.cat(mus)[:n_samples].numpy()
        theta_true = torch.cat(thetas_true)[:n_samples].numpy()

        metrics = {}

        # 1. Trustworthiness: theta-space neighbors preserved in latent space?
        # Score in [0, 1], higher = better preservation of local structure.
        # Requires n_samples > 2*k; skip if too few samples.
        effective_k = min(k, mu.shape[0] // 2 - 1)
        if effective_k >= 2:
            metrics["trust_theta_to_z"] = float(
                trustworthiness(theta_true, mu, n_neighbors=effective_k)
            )
        else:
            metrics["trust_theta_to_z"] = float("nan")

        # 2. Effective dimensionality: PCA on mu, count dims for 95% variance
        pca = PCA().fit(mu)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        metrics["eff_dim_95"] = int(np.searchsorted(cumvar, 0.95)) + 1
        metrics["eff_dim_99"] = int(np.searchsorted(cumvar, 0.99)) + 1

        # 3. Mean pairwise L2 distance in latent space (collapse detector)
        mu_t = torch.from_numpy(mu)
        # Sample pairs to avoid O(n^2) memory for large n
        n = min(mu_t.shape[0], 1024)
        dists = torch.cdist(mu_t[:n], mu_t[:n])
        triu_idx = torch.triu_indices(n, n, offset=1)
        metrics["latent_mean_l2"] = float(dists[triu_idx[0], triu_idx[1]].mean())
        metrics["latent_std_l2"] = float(dists[triu_idx[0], triu_idx[1]].std())

        return metrics

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
        print(f"    - pooling: {self.config.model.condition.pooling}")
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
            truncation_length=self.config.data.truncation_length,
            max_samples=self.config.data.max_samples,
            train_split=self.config.data.train_split,
            seed=self.config.seed,
        )

        print(f"  Train size: {len(train_dataset)}")
        print(f"  Val size: {len(val_dataset)}")
        print(f"  Temporal token keys: {train_dataset.num_temporal_keys}")

        # Create data loaders
        # Grids are preloaded as uint8 in memory (shared via fork COW),
        # so workers just do tensor slicing — fast enough for high GPU utilization.
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_cvae_batch,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=collate_cvae_batch,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
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

                print(f"\nEpoch {epoch+1} Summary (beta={self.get_beta(epoch):.4f}):")
                print(f"  Train - Total: {train_losses['total_loss']:.6f}, "
                      f"Theta: {train_losses['theta_loss']:.6f}, "
                      f"Grid: {train_losses['grid_loss']:.6f}, "
                      f"KL: {train_losses['kl_loss']:.1f}, "
                      f"KL_eff: {train_losses['kl_loss_effective']:.1f}")
                print(f"  Val   - Total: {val_losses['total_loss']:.6f}, "
                      f"Theta: {val_losses['theta_loss']:.6f}, "
                      f"Grid: {val_losses['grid_loss']:.6f}, "
                      f"KL: {val_losses['kl_loss']:.1f}, "
                      f"KL_eff: {val_losses['kl_loss_effective']:.1f}")

                # Topological preservation metrics
                topo = self.compute_topo_metrics(val_loader)
                print(f"  Topo  - Trust(θ→z): {topo['trust_theta_to_z']:.4f}, "
                      f"EffDim95: {topo['eff_dim_95']}, "
                      f"EffDim99: {topo['eff_dim_99']}, "
                      f"LatentL2: {topo['latent_mean_l2']:.2f}±{topo['latent_std_l2']:.2f}")

                # Store topo metrics in val history
                val_losses.update(topo)

                # Checkpoint on reconstruction loss (theta + grid), not total_loss.
                # With free bits, total_loss is dominated by beta * kl_floor;
                # reconstruction quality is what matters for generation.
                val_recon = val_losses["theta_loss"] + val_losses["grid_loss"]
                if val_recon < self.best_val_loss:
                    self.best_val_loss = val_recon
                    self.save_checkpoint("best.pt")
                    print(f"  -> Saved best model (val_recon: {self.best_val_loss:.6f})")

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
