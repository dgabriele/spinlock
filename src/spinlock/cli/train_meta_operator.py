"""
Train Meta-Operator command for Spinlock CLI.

Trains NOA as a precision physics meta-operator using pure trajectory matching.
Stage 1 of two-stage training approach.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import sys
import warnings
import yaml
import time

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Suppress torch._inductor warning about SMs
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

from .base import CLICommand


class TrainMetaOperatorCommand(CLICommand):
    """
    Command to train NOA as a precision physics meta-operator.

    Stage 1 of two-stage training: Train NOA purely on physics (trajectory matching)
    without any VQ involvement. The trained checkpoint can then be used in Stage 2
    to train VQ-VAE on NOA's distribution.
    """

    @property
    def name(self) -> str:
        return "train-meta-operator"

    @property
    def help(self) -> str:
        return "Train NOA as precision physics meta-operator (Stage 1)"

    @property
    def description(self) -> str:
        return """
Train NOA as a precision physics meta-operator using pure trajectory matching.

This is Stage 1 of the two-stage training approach:
- Stage 1 (this command): Train NOA on pure physics (MSE vs CNO rollouts)
- Stage 2 (train-vqvae): Train VQ-VAE on NOA's distribution

Training:
  Uses MSE-led loss with no VQ involvement (L_total = L_traj).
  Optional instrumental regularizers can be added (smoothness, energy conservation).

Output Checkpoint Format:
  Checkpoint includes full model config for Stage 2 compatibility.
  Structure:
    - model_state_dict: NOA weights
    - optimizer_state_dict: Optimizer state (for resuming)
    - scheduler_state_dict: LR scheduler state
    - config: Full config (for reproducibility in Stage 2)
    - val_loss: Validation loss
    - train_loss: Training loss
    - epoch: Current epoch
    - timestamp: Training timestamp

Examples:
  # Train with config file
  spinlock train-meta-operator --config configs/noa/train_meta_operator.yaml

  # Override parameters
  spinlock train-meta-operator \\
      --config configs/noa/train_meta_operator.yaml \\
      --n-samples 1000 \\
      --epochs 20 \\
      --learning-rate 1e-4

  # Resume from checkpoint
  spinlock train-meta-operator \\
      --config configs/noa/train_meta_operator.yaml \\
      --resume-from checkpoints/meta_operator/meta_operator_epoch10.pt

Output:
  The checkpoint directory will contain:
    - meta_operator_epochN.pt:  Epoch checkpoints
    - meta_operator_best.pt:    Best checkpoint (lowest validation loss)
    - training_log.txt:         Training metrics log
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add train-meta-operator command arguments."""
        # Required arguments
        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to meta-operator training config YAML",
        )

        # Configuration overrides
        override_group = parser.add_argument_group("configuration overrides")

        override_group.add_argument(
            "--n-samples",
            type=int,
            metavar="N",
            help="Override number of training samples",
        )

        override_group.add_argument(
            "--epochs",
            type=int,
            metavar="N",
            help="Override number of training epochs",
        )

        override_group.add_argument(
            "--batch-size",
            type=int,
            metavar="N",
            help="Override training batch size",
        )

        override_group.add_argument(
            "--learning-rate",
            type=float,
            metavar="LR",
            help="Override learning rate",
        )

        override_group.add_argument(
            "--timesteps",
            type=int,
            metavar="T",
            help="Override number of rollout timesteps",
        )

        override_group.add_argument(
            "--val-split",
            type=float,
            metavar="FRAC",
            help="Override validation split fraction",
        )

        override_group.add_argument(
            "--resume-from",
            type=Path,
            metavar="PATH",
            help="Resume training from checkpoint",
        )

        # Execution options
        exec_group = parser.add_argument_group("execution options")

        exec_group.add_argument(
            "--device",
            type=str,
            choices=["cuda", "cpu"],
            default="cuda",
            metavar="DEVICE",
            help="Device for computation (default: cuda)",
        )

        exec_group.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate configuration without training",
        )

        exec_group.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed progress information",
        )

        exec_group.add_argument(
            "--log-every",
            type=int,
            default=10,
            metavar="N",
            help="Log metrics every N batches (default: 10)",
        )

    def execute(self, args: Namespace) -> int:
        """Execute meta-operator training."""
        # Validate config exists
        if not self.validate_file_exists(args.config, "Config"):
            return 1

        # Load config
        try:
            config = self._load_config(args.config)
        except Exception as e:
            return self.error(f"Failed to load config: {e}")

        # Apply CLI overrides
        config = self._apply_cli_overrides(config, args)

        # Validate configuration
        try:
            self._validate_config(config)
        except ValueError as e:
            return self.error(f"Invalid configuration: {e}")

        # Validate dataset exists
        dataset_path = config["data"]["dataset_path"]
        if not self.validate_file_exists(Path(dataset_path), "Dataset"):
            return 1

        # Print configuration summary
        if args.verbose or args.dry_run:
            self._print_config_summary(config, args)

        # Dry run: validate and exit
        if args.dry_run:
            print("\n✓ Configuration valid (dry-run mode, no training performed)")
            return 0

        # Execute training
        try:
            return self._run_training(config, args)
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            import traceback
            print(f"\nError during training: {e}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 1

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load meta-operator training config from YAML."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty config file: {config_path}")

        return config

    def _apply_cli_overrides(self, config: Dict[str, Any], args: Namespace) -> Dict[str, Any]:
        """Apply CLI argument overrides to config."""
        if args.n_samples:
            config["training"]["n_samples"] = args.n_samples
        if args.epochs:
            config["training"]["epochs"] = args.epochs
        if args.batch_size:
            config["training"]["batch_size"] = args.batch_size
        if args.learning_rate:
            config["training"]["learning_rate"] = args.learning_rate
        if args.timesteps:
            config["training"]["timesteps"] = args.timesteps
        if args.val_split:
            config["data"]["val_split"] = args.val_split
        if args.resume_from:
            config["resume_from"] = str(args.resume_from)
        if args.device:
            config["device"] = args.device

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration has required fields."""
        # Validate model section
        if "model" not in config:
            raise ValueError(
                "Missing required section: 'model'\n\n"
                "Example:\n"
                "  model:\n"
                "    spatial_dim: 64\n"
                "    in_channels: 1\n"
                "    out_channels: 1\n"
                "    base_channels: 32\n"
                "    encoder_levels: 3\n"
                "    modes: 16\n"
                "    afno_blocks: 4"
            )

        required_model_fields = ["spatial_dim", "in_channels", "out_channels"]
        for field in required_model_fields:
            if field not in config["model"]:
                raise ValueError(f"Model section missing required field: '{field}'")

        # Validate training section
        if "training" not in config:
            raise ValueError(
                "Missing required section: 'training'\n\n"
                "Example:\n"
                "  training:\n"
                "    n_samples: 10000\n"
                "    batch_size: 8\n"
                "    epochs: 50\n"
                "    learning_rate: 1.0e-4\n"
                "    timesteps: 32"
            )

        required_training_fields = ["n_samples", "batch_size", "epochs", "learning_rate", "timesteps"]
        for field in required_training_fields:
            if field not in config["training"]:
                raise ValueError(f"Training section missing required field: '{field}'")

        # Validate data section
        if "data" not in config:
            raise ValueError(
                "Missing required section: 'data'\n\n"
                "Example:\n"
                "  data:\n"
                "    dataset_path: 'data/production/100k_3family_v1'\n"
                "    val_split: 0.1\n"
                "    num_workers: 4"
            )

        if "dataset_path" not in config["data"]:
            raise ValueError("Data section missing required field: 'dataset_path'")

        # Validate checkpointing section
        if "checkpointing" not in config:
            raise ValueError(
                "Missing required section: 'checkpointing'\n\n"
                "Example:\n"
                "  checkpointing:\n"
                "    save_dir: 'checkpoints/meta_operator'\n"
                "    save_every: 5\n"
                "    keep_best: true"
            )

        if "save_dir" not in config["checkpointing"]:
            raise ValueError("Checkpointing section missing required field: 'save_dir'")

    def _print_config_summary(self, config: Dict[str, Any], args: Namespace) -> None:
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("Meta-Operator Training Configuration")
        print("=" * 70)

        print("\nModel:")
        for key, value in config["model"].items():
            print(f"  {key}: {value}")

        print("\nTraining:")
        for key, value in config["training"].items():
            print(f"  {key}: {value}")

        print("\nData:")
        for key, value in config["data"].items():
            print(f"  {key}: {value}")

        print("\nCheckpointing:")
        for key, value in config["checkpointing"].items():
            print(f"  {key}: {value}")

        if "loss" in config:
            print("\nLoss:")
            for key, value in config["loss"].items():
                print(f"  {key}: {value}")

        print(f"\nDevice: {config.get('device', 'cuda')}")
        print(f"Seed: {config.get('seed', 42)}")

        if args.resume_from:
            print(f"\nResuming from: {args.resume_from}")

        print("=" * 70 + "\n")

    def _run_training(self, config: Dict[str, Any], args: Namespace) -> int:
        """Execute training pipeline."""
        from spinlock.noa import NOABackbone, CNOReplayer
        from spinlock.noa.losses import MSELedLoss
        from spinlock.operators.state_dataset import NOAStateDataset

        device = config.get("device", "cuda")
        seed = config.get("seed", 42)

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Initializing training...")

        # Create model
        print("Creating NOA backbone...")
        noa = NOABackbone(**config["model"])
        noa = noa.to(device)
        print(f"  ✓ NOA created ({sum(p.numel() for p in noa.parameters()):,} parameters)")

        # Create loss function (pure physics, no VQ)
        print("Creating loss function...")
        loss_fn = MSELedLoss(
            lambda_traj=config["loss"].get("lambda_traj", 1.0),
            lambda_commit=0.0,  # No VQ alignment
            lambda_latent=0.0,  # No VQ alignment
            vqvae_alignment=None,  # Critical: No VQ-VAE
        )
        print(f"  ✓ Pure physics loss (L_traj = {config['loss'].get('lambda_traj', 1.0)})")

        # Create CNO replayer
        print("Loading CNO replayer...")
        cno_config = config["data"].get("cno_config")

        if not cno_config:
            return self.error(
                "Data section missing CNO configuration.\n"
                "Required field: cno_config (path to CNO config YAML)"
            )

        replayer = CNOReplayer.from_config(
            config_path=cno_config,
            device=device,
            cache_size=8,
        )
        print(f"  ✓ CNO replayer loaded")

        # Create dataset and dataloaders
        print("Loading dataset...")
        dataset = NOAStateDataset(
            dataset_path=config["data"]["dataset_path"],
            max_samples=config["training"]["n_samples"],
        )

        val_split = config["data"].get("val_split", 0.1)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data"].get("num_workers", 4),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"].get("num_workers", 4),
        )

        print(f"  ✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")

        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            noa.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 1e-6),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        best_val_loss = float('inf')

        if "resume_from" in config:
            print(f"Resuming from checkpoint: {config['resume_from']}")
            checkpoint = torch.load(config["resume_from"], map_location=device)
            noa.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("val_loss", float('inf'))
            print(f"  ✓ Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.6f}")

        # Create checkpoint directory
        save_dir = Path(config["checkpointing"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        # Open training log
        log_file = save_dir / "training_log.txt"
        log_fp = open(log_file, "a" if start_epoch > 0 else "w")

        if start_epoch == 0:
            log_fp.write("# Meta-Operator Training Log\n")
            log_fp.write(f"# Started: {datetime.now().isoformat()}\n")
            log_fp.write("# Epoch,TrainLoss,ValLoss,LR,Time(s)\n")

        print(f"\n{'='*70}")
        print("Training")
        print(f"{'='*70}\n")

        # Training loop
        for epoch in range(start_epoch, config["training"]["epochs"]):
            print(f"Epoch {epoch + 1}/{config['training']['epochs']}")

            # Train
            train_metrics = self._train_epoch(
                noa=noa,
                loss_fn=loss_fn,
                replayer=replayer,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                timesteps=config["training"]["timesteps"],
                clip_grad=config["training"].get("clip_grad", 1.0),
                log_every=args.log_every,
            )

            # Validate
            val_metrics = self._validate_epoch(
                noa=noa,
                loss_fn=loss_fn,
                replayer=replayer,
                dataloader=val_loader,
                device=device,
                timesteps=config["training"]["timesteps"],
            )

            # Update scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Print epoch summary
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss:   {val_metrics['loss']:.6f}")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Time:       {train_metrics['time']:.2f}s")

            # Log to file
            log_fp.write(f"{epoch+1},{train_metrics['loss']:.6f},{val_metrics['loss']:.6f},{current_lr:.2e},{train_metrics['time']:.2f}\n")
            log_fp.flush()

            # Save checkpoint
            save_every = config["checkpointing"].get("save_every", 5)
            if (epoch + 1) % save_every == 0 or (epoch + 1) == config["training"]["epochs"]:
                checkpoint_path = save_dir / f"meta_operator_epoch{epoch+1}.pt"
                self._save_checkpoint(
                    path=checkpoint_path,
                    noa=noa,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    config=config,
                )
                print(f"  Saved checkpoint: {checkpoint_path}")

            # Save best checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_checkpoint_path = save_dir / "meta_operator_best.pt"
                self._save_checkpoint(
                    path=best_checkpoint_path,
                    noa=noa,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    config=config,
                )
                print(f"  New best checkpoint: {best_checkpoint_path} (val_loss={best_val_loss:.6f})")

            print()

        log_fp.close()

        print(f"{'='*70}")
        print("Training Complete")
        print(f"{'='*70}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Checkpoints saved to: {save_dir}")
        print(f"Training log: {log_file}")
        print()

        return 0

    def _train_epoch(
        self,
        noa,
        loss_fn,
        replayer,
        dataloader,
        optimizer,
        device,
        timesteps,
        clip_grad=1.0,
        log_every=10,
    ) -> dict:
        """Train for one epoch."""
        noa.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            ic = batch["ic"].to(device)
            params = batch["params"]
            B = ic.shape[0]

            # Generate NOA rollout
            pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True)

            # Generate CNO targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            target_trajectories = []
            skip_batch = False

            for b in range(B):
                try:
                    target_traj = replayer.rollout(
                        params_vector=params[b].numpy(),
                        ic=ic[b:b+1],
                        timesteps=timesteps,
                        num_realizations=1,
                        return_all_steps=True,
                    )
                    target_trajectories.append(target_traj)
                except (ValueError, RuntimeError) as e:
                    print(f"  Warning: CNO rollout failed for sample {b} in batch {batch_idx}: {e}")
                    skip_batch = True
                    break

            if skip_batch:
                continue

            target_trajectory = torch.cat(target_trajectories, dim=0)

            # Extract states (skip IC at t=0)
            pred_states = pred_trajectory[:, 1:, :, :, :]
            target_states = target_trajectory[:, 1:, :, :, :]

            # Compute loss
            try:
                loss_output = loss_fn.compute(
                    pred_trajectory=pred_states,
                    target_trajectory=target_states,
                    ic=ic,
                    noa=noa,
                )
            except Exception as e:
                print(f"  Warning: Loss computation failed: {e}")
                continue

            if torch.isnan(loss_output.total) or torch.isinf(loss_output.total):
                print(f"  Warning: NaN/Inf loss at batch {batch_idx}")
                continue

            # Optimize
            optimizer.zero_grad()
            loss_output.total.backward()
            torch.nn.utils.clip_grad_norm_(noa.parameters(), clip_grad)
            optimizer.step()

            # Accumulate metrics
            total_loss += loss_output.total.item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % log_every == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={avg_loss:.6f}")

        elapsed = time.time() - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        return {
            "loss": avg_loss,
            "time": elapsed,
        }

    def _validate_epoch(
        self,
        noa,
        loss_fn,
        replayer,
        dataloader,
        device,
        timesteps,
    ) -> dict:
        """Validate for one epoch."""
        noa.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                ic = batch["ic"].to(device)
                params = batch["params"]
                B = ic.shape[0]

                # Generate NOA rollout
                pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True)

                # Generate CNO targets
                target_trajectories = []
                skip_batch = False

                for b in range(B):
                    try:
                        target_traj = replayer.rollout(
                            params_vector=params[b].numpy(),
                            ic=ic[b:b+1],
                            timesteps=timesteps,
                            num_realizations=1,
                            return_all_steps=True,
                        )
                        target_trajectories.append(target_traj)
                    except (ValueError, RuntimeError):
                        skip_batch = True
                        break

                if skip_batch:
                    continue

                target_trajectory = torch.cat(target_trajectories, dim=0)

                # Extract states
                pred_states = pred_trajectory[:, 1:, :, :, :]
                target_states = target_trajectory[:, 1:, :, :, :]

                # Compute loss
                try:
                    loss_output = loss_fn.compute(
                        pred_trajectory=pred_states,
                        target_trajectory=target_states,
                        ic=ic,
                        noa=noa,
                    )
                except Exception:
                    continue

                if torch.isnan(loss_output.total) or torch.isinf(loss_output.total):
                    continue

                total_loss += loss_output.total.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        return {
            "loss": avg_loss,
        }

    def _save_checkpoint(
        self,
        path: Path,
        noa,
        optimizer,
        scheduler,
        epoch,
        train_loss,
        val_loss,
        config,
    ) -> None:
        """Save checkpoint in Stage 2 compatible format."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': noa.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'config': config,  # Full config for reproducibility
            'timestamp': datetime.now().isoformat(),
        }

        torch.save(checkpoint, path)
