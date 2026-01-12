"""
Train Meta-Operator command for Spinlock CLI.

Trains NOA as a precision physics meta-operator using pure trajectory matching.
Supports both MSE-led (Stage 1) and VQ-led (Stage 2) training paradigms.

Documentation:
    - Training guide: docs/noa-training-guide.md
    - Two-stage curriculum: docs/two-stage-curriculum-architecture.md
    - NOA architecture: docs/noa-architecture.md
    - MNO architecture spec: docs/MNO_ARCHITECTURE.md
    - Truncated BPTT: docs/truncated-bptt-integration.md
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

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

Prerequisites:
  For token-conditioned training (model.token_conditioning=true):
    1. Train a VQ-VAE on trajectory features (spinlock train-vqvae)
    2. Generate ground-truth tokens for dataset (spinlock compute-oracle-tokens)
    3. Specify ground_truth_token_path in config: data.ground_truth_token_path

  Without token conditioning:
    No additional prerequisites beyond dataset generation.

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

        # Handle token conditioning setup
        token_conditioning = config["model"].get("token_conditioning", False)
        if token_conditioning:
            print("Setting up token conditioning...")

            # Load VQ-VAE checkpoint to extract codebook sizes
            vqvae_checkpoint = config["model"].get("vqvae_checkpoint")
            if not vqvae_checkpoint:
                return self.error(
                    "Token conditioning enabled but vqvae_checkpoint not specified in model config"
                )

            if not Path(vqvae_checkpoint).exists():
                return self.error(f"VQ-VAE checkpoint not found: {vqvae_checkpoint}")

            print(f"  Loading VQ-VAE config from: {vqvae_checkpoint}")
            vqvae_ckpt = torch.load(vqvae_checkpoint, map_location='cpu', weights_only=False)

            # Extract codebook sizes from VQ-VAE checkpoint
            vqvae_config = vqvae_ckpt.get("config", {})
            state_dict = vqvae_ckpt.get("model_state_dict", {})
            codebook_sizes = []

            # Method 1: Extract from config['levels'] structure (hierarchical VQ-VAE)
            if "levels" in vqvae_config and isinstance(vqvae_config["levels"], dict):
                # levels is a dict like: {'cluster_1': [{'num_tokens': 24, ...}, ...], ...}
                for category_name, category_levels in vqvae_config["levels"].items():
                    for level in category_levels:
                        num_tokens = level.get("num_tokens", level.get("num_embeddings", 64))
                        codebook_sizes.append(num_tokens)
            # Method 2: Extract from categories structure (alternative format)
            elif "categories" in vqvae_config:
                for category in vqvae_config["categories"]:
                    for level in category.get("levels", []):
                        num_embeddings = level.get("num_embeddings", 64)
                        codebook_sizes.append(num_embeddings)
            # Method 3: Fallback to state dict inspection
            else:
                # Look for vq_layers embedding weights in state dict
                import re
                vq_embedding_keys = [k for k in state_dict.keys()
                                     if "vq_layers" in k and "embedding.weight" in k]

                # Sort by layer index
                def extract_layer_idx(key):
                    match = re.search(r'vq_layers\.(\d+)\.', key)
                    return int(match.group(1)) if match else 0

                vq_embedding_keys_sorted = sorted(vq_embedding_keys, key=extract_layer_idx)

                for key in vq_embedding_keys_sorted:
                    num_embeddings = state_dict[key].shape[0]
                    codebook_sizes.append(num_embeddings)

            if not codebook_sizes:
                return self.error(
                    f"Could not extract codebook sizes from VQ-VAE checkpoint.\n"
                    f"Expected 'categories' in config or 'vq_layers.*.codebook' in state_dict"
                )

            print(f"  ✓ Extracted {len(codebook_sizes)} codebook sizes: {codebook_sizes[:5]}...")

            # Auto-determine token conditioning parameters from VQ-VAE
            num_tokens = len(codebook_sizes)

            # Override config values with runtime-determined values
            config["model"]["codebook_sizes"] = codebook_sizes
            config["model"]["num_tokens"] = num_tokens

            # Use config defaults if not specified, but prefer runtime values
            if "token_embed_dim" not in config["model"]:
                config["model"]["token_embed_dim"] = 64  # Default

            print(f"  ✓ Token conditioning setup: {num_tokens} tokens, embed_dim={config['model']['token_embed_dim']}")

        # Create model
        print("Creating NOA backbone...")
        noa = NOABackbone(**config["model"])
        noa = noa.to(device)
        print(f"  ✓ NOA created ({sum(p.numel() for p in noa.parameters()):,} parameters)")

        # Wrap with truncated BPTT if configured
        timesteps = config["training"]["timesteps"]
        bptt_window = config["training"].get("bptt_window")

        from spinlock.noa import TruncatedBPTT

        if bptt_window is not None and bptt_window < timesteps:
            print(f"  ✓ Using truncated BPTT: {timesteps} steps, backprop window={bptt_window}")
            noa_rollout = TruncatedBPTT(noa, timesteps=timesteps, bptt_window=bptt_window)
        else:
            print(f"  ✓ Using full backprop: {timesteps} steps")
            # Create pass-through wrapper for uniform interface
            class FullBPTTWrapper:
                def __init__(self, model, timesteps):
                    self.model = model
                    self.timesteps = timesteps

                def rollout(self, ic, tokens=None):
                    return self.model.rollout(ic, steps=self.timesteps, return_all_steps=True, tokens=tokens)

                def align_for_loss(self, pred_traj, target_traj, skip_ic=True):
                    if skip_ic:
                        return pred_traj[:, 1:, :, :, :], target_traj[:, 1:, :, :, :]
                    else:
                        return pred_traj, target_traj

            noa_rollout = FullBPTTWrapper(noa, timesteps)

        # Create loss function
        print("Creating loss function...")
        loss_mode = config["loss"].get("mode", "mse_led")  # Default: MSE-led (Stage 1)

        # Check if VQ-led mode (Stage 2)
        if loss_mode == "vq_led":
            # VQ-led requires VQ-VAE alignment
            vqvae_config = config.get("vqvae")
            if not vqvae_config:
                return self.error(
                    "VQ-led mode requires 'vqvae' section in config.\n"
                    "Required: vqvae.checkpoint"
                )

            vqvae_checkpoint = vqvae_config.get("checkpoint")
            if not vqvae_checkpoint:
                return self.error("VQ-led mode requires vqvae.checkpoint path")

            print(f"  Loading VQ-VAE alignment from: {vqvae_checkpoint}")
            from spinlock.noa.vqvae_alignment import VQVAEAlignmentLoss

            vqvae_alignment = VQVAEAlignmentLoss.from_checkpoint(
                vqvae_path=vqvae_checkpoint,
                device=device,
                use_aligned_extractor=vqvae_config.get("use_aligned_extractor", True),
                enable_latent_loss=False,  # Not used in Stage 2
                normalization_stats_file=vqvae_config.get("normalization_stats_file"),
            )
            print(f"  ✓ VQ-VAE alignment loaded")

            # Create VQ-led loss
            from spinlock.noa.losses.vq_led import VQLedLoss

            loss_fn = VQLedLoss(
                lambda_recon=config["loss"].get("lambda_recon", 1.0),
                lambda_commit=config["loss"].get("lambda_commit", 0.5),
                lambda_traj=config["loss"].get("lambda_traj", 0.3),
                vqvae_alignment=vqvae_alignment,
            )
            print(f"  ✓ VQ-led loss (L_recon={config['loss'].get('lambda_recon', 1.0)}, "
                  f"L_commit={config['loss'].get('lambda_commit', 0.5)}, "
                  f"L_traj={config['loss'].get('lambda_traj', 0.3)})")
        else:
            # MSE-led (pure physics, no VQ)
            loss_fn = MSELedLoss(
                lambda_traj=config["loss"].get("lambda_traj", 1.0),
                lambda_ic=config["loss"].get("lambda_ic", 0.0),
                lambda_commit=0.0,  # No VQ alignment
                lambda_latent=0.0,  # No VQ alignment
                vqvae_alignment=None,  # Critical: No VQ-VAE
            )
            lambda_ic = config['loss'].get('lambda_ic', 0.0)
            if lambda_ic > 0:
                print(f"  ✓ Pure physics loss (L_traj={config['loss'].get('lambda_traj', 1.0)}, L_ic={lambda_ic})")
            else:
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

        # Load ground-truth tokens if token conditioning is enabled
        ground_truth_tokens = None
        train_token_indices = None
        val_token_indices = None

        if token_conditioning:
            ground_truth_token_path = config["data"].get("ground_truth_token_path")
            # Stage 2 (VQ-led): token_conditioning architecture needed for checkpoint loading,
            # but tokens not used during training
            if not ground_truth_token_path and loss_mode != "vq_led":
                return self.error(
                    "Token conditioning enabled but ground_truth_token_path not specified in data config.\n"
                    "For Stage 2 (VQ-led), this is expected - tokens won't be used."
                )

            if ground_truth_token_path and not Path(ground_truth_token_path).exists():
                return self.error(f"Ground-truth token file not found: {ground_truth_token_path}")

            # Load ground-truth tokens if provided (Stage 1)
            # In Stage 2 (VQ-led), ground_truth_token_path is None - model self-regulates
            if ground_truth_token_path:
                print(f"Loading ground-truth tokens from: {ground_truth_token_path}")
                import h5py
                token_file = h5py.File(ground_truth_token_path, 'r')
                ground_truth_tokens_full = torch.tensor(token_file["tokens"][:], dtype=torch.long)
                token_file.close()

                # Get the indices used by train/val split
                train_indices = train_dataset.indices
                val_indices = val_dataset.indices

                # ground_truth_tokens_full is [N, num_tokens] for full dataset
                # We need to select only the tokens for train/val splits
                ground_truth_tokens = ground_truth_tokens_full  # Keep full array
                train_token_indices = train_indices  # Indices to select from ground_truth_tokens
                val_token_indices = val_indices

                print(f"  ✓ Loaded {len(ground_truth_tokens)} ground-truth tokens ({ground_truth_tokens.shape[1]} tokens per sample)")
                print(f"  Train indices: {len(train_indices)}, Val indices: {len(val_indices)}")
            else:
                print(f"  Stage 2 (VQ-led): No ground-truth tokens - model will self-regulate")

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

        # Create LR scheduler with optional warmup
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        warmup_steps = config["training"].get("warmup_steps", 0)
        if warmup_steps > 0:
            # Warmup: LR ramps from 0.1x to 1.0x over warmup_steps
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
            # Cosine: LR decays from 1.0x to 0 over remaining epochs
            total_epochs = config["training"]["epochs"]
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_steps if warmup_steps < total_epochs else 1,
            )
            # Sequential: warmup first, then cosine
            scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
            print(f"  ✓ LR schedule: {warmup_steps}-step warmup + cosine decay")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["training"]["epochs"],
            )
            print(f"  ✓ LR schedule: cosine decay (no warmup)")

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

        # Early stopping setup
        early_stopping_patience = config["training"].get("early_stopping_patience", 0)
        epochs_without_improvement = 0

        # Training loop
        for epoch in range(start_epoch, config["training"]["epochs"]):
            print(f"Epoch {epoch + 1}/{config['training']['epochs']}")

            # Train
            train_metrics = self._train_epoch(
                noa=noa,
                noa_rollout=noa_rollout,
                loss_fn=loss_fn,
                replayer=replayer,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                timesteps=config["training"]["timesteps"],
                clip_grad=config["training"].get("clip_grad", 1.0),
                log_every=args.log_every,
                accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
                ground_truth_tokens=ground_truth_tokens,
            )

            # Validate
            val_metrics = self._validate_epoch(
                noa=noa,
                noa_rollout=noa_rollout,
                loss_fn=loss_fn,
                replayer=replayer,
                dataloader=val_loader,
                device=device,
                timesteps=config["training"]["timesteps"],
                ground_truth_tokens=ground_truth_tokens,
            )

            # Update scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Print epoch summary
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss:   {val_metrics['loss']:.6f}")

            # Print validation loss components if available
            if 'components' in val_metrics and val_metrics['components']:
                components_str = ", ".join([
                    f"{name}={value:.4f}"
                    for name, value in sorted(val_metrics['components'].items())
                ])
                print(f"  Val Components: {components_str}")

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

            # Save best checkpoint and check early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0  # Reset counter
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
            else:
                epochs_without_improvement += 1

                if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                    print(f"\n⚠ Early stopping triggered: no improvement for {early_stopping_patience} epochs")
                    print(f"  Best validation loss: {best_val_loss:.6f} (epoch {epoch + 1 - early_stopping_patience})")
                    break  # Exit training loop

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
        noa_rollout,
        loss_fn,
        replayer,
        dataloader,
        optimizer,
        device,
        timesteps,
        clip_grad=1.0,
        log_every=10,
        accumulation_steps=1,
        ground_truth_tokens=None,
        token_indices=None,
    ) -> dict:
        """Train for one epoch with gradient accumulation."""
        noa.train()
        total_loss = 0.0
        component_losses = {}  # Track individual loss components
        num_batches = 0
        start_time = time.time()

        # Initialize gradients outside loop for accumulation
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            ic = batch["ic"].to(device)
            params = batch["params"]
            indices = batch.get("sample_idx")  # Dataset indices for this batch
            B = ic.shape[0]

            # Get ground-truth tokens for this batch if token conditioning is enabled
            batch_tokens = None
            if ground_truth_tokens is not None and indices is not None:
                # indices contains the original dataset indices (from sample_idx)
                # ground_truth_tokens is indexed by original dataset index
                batch_tokens = ground_truth_tokens[indices].to(device)

            # Generate NOA rollout (with truncated BPTT if configured)
            pred_trajectory = noa_rollout.rollout(ic, tokens=batch_tokens)

            # Generate CNO targets
            # Only clear cache if memory usage > 90% (avoid unnecessary serialization)
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device)
                max_allocated = torch.cuda.max_memory_allocated(device)
                if max_allocated > 0 and allocated / max_allocated > 0.9:
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

            # Align predicted and target states for loss computation
            # (handles truncated BPTT windowing automatically)
            pred_states, target_states = noa_rollout.align_for_loss(
                pred_trajectory,
                target_trajectory,
                skip_ic=True,
            )

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

            # Gradient accumulation: scale loss and accumulate gradients
            scaled_loss = loss_output.total / accumulation_steps
            scaled_loss.backward()

            # Only step optimizer every N batches
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(noa.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()

            # Accumulate metrics
            total_loss += loss_output.total.item()
            num_batches += 1

            # Accumulate component losses
            for component_name, component_value in loss_output.components.items():
                if component_name not in component_losses:
                    component_losses[component_name] = 0.0
                component_losses[component_name] += component_value.item()

            # Log progress every N batches
            if (batch_idx + 1) % log_every == 0:
                avg_loss = total_loss / num_batches
                batch_total_time = time.time() - batch_start_time

                # Format component losses
                component_str = ", ".join([
                    f"{name}={component_losses[name] / num_batches:.4f}"
                    for name in sorted(component_losses.keys())
                ])

                print(f"  Batch {batch_idx + 1}/{len(dataloader)}: total={avg_loss:.6f}, {component_str}, time={batch_total_time:.2f}s")

        # Handle leftover gradients at end of epoch
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(noa.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        return {
            "loss": avg_loss,
            "time": elapsed,
        }

    def _validate_epoch(
        self,
        noa,
        noa_rollout,
        loss_fn,
        replayer,
        dataloader,
        device,
        timesteps,
        ground_truth_tokens=None,
        token_indices=None,
    ) -> dict:
        """Validate for one epoch."""
        noa.eval()
        total_loss = 0.0
        component_losses = {}  # Track individual loss components
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                ic = batch["ic"].to(device)
                params = batch["params"]
                indices = batch.get("sample_idx")  # Dataset indices for this batch
                B = ic.shape[0]

                # Get ground-truth tokens for this batch if token conditioning is enabled
                batch_tokens = None
                if ground_truth_tokens is not None and indices is not None:
                    # indices contains the original dataset indices (from sample_idx)
                    # ground_truth_tokens is indexed by original dataset index
                    batch_tokens = ground_truth_tokens[indices].to(device)

                # Generate NOA rollout (with truncated BPTT if configured)
                pred_trajectory = noa_rollout.rollout(ic, tokens=batch_tokens)

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

                # Align predicted and target states for loss computation
                # (handles truncated BPTT windowing automatically)
                pred_states, target_states = noa_rollout.align_for_loss(
                    pred_trajectory,
                    target_trajectory,
                    skip_ic=True,
                )

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

                # Accumulate component losses
                for component_name, component_value in loss_output.components.items():
                    if component_name not in component_losses:
                        component_losses[component_name] = 0.0
                    component_losses[component_name] += component_value.item()

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Compute average component losses
        avg_components = {
            name: value / num_batches if num_batches > 0 else float('inf')
            for name, value in component_losses.items()
        }

        return {
            "loss": avg_loss,
            "components": avg_components,
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
