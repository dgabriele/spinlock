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
import itertools
import re

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

        # Distributed training options (internal use by launcher)
        dist_group = parser.add_argument_group("distributed training (internal)")

        dist_group.add_argument(
            "--distributed-rank",
            type=int,
            default=None,
            metavar="RANK",
            help="Global rank for distributed training (set by launcher)",
        )

        dist_group.add_argument(
            "--distributed-world-size",
            type=int,
            default=None,
            metavar="SIZE",
            help="World size for distributed training (set by launcher)",
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

        # Check for distributed training
        distributed_config = config.get("distributed", {})
        is_distributed = distributed_config.get("enabled", False)
        is_worker_process = args.distributed_rank is not None

        if is_distributed and not is_worker_process:
            # This is the launcher process - launch distributed training
            return self._launch_distributed_training(config, args)
        else:
            # Execute training (either single-GPU or as distributed worker)
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

    def _launch_distributed_training(self, config: Dict[str, Any], args: Namespace) -> int:
        """Launch distributed training across multiple nodes."""
        from spinlock.distributed import DistributedConfig, launch_distributed_training

        print("\n" + "="*70)
        print("Distributed Training Configuration")
        print("="*70)

        # Parse distributed config
        try:
            dist_config = DistributedConfig.from_dict(config["distributed"])
            dist_config.validate()
        except Exception as e:
            return self.error(f"Invalid distributed configuration: {e}")

        # Print distributed setup
        print(f"\nBackend: {dist_config.backend}")
        print(f"World size: {dist_config.world_size}")
        print(f"Master: {dist_config.get_master_addr()}:{dist_config.master_port}")
        print(f"\nNodes:")
        for i, node in enumerate(dist_config.nodes):
            print(f"  [{i}] {node.host}: {len(node.gpus)} GPU(s) {node.gpus}")

        print("\n" + "="*70 + "\n")

        # Build script arguments
        script_path = "train-meta-operator"  # CLI command name
        script_args = ["--config", str(args.config)]

        # Add CLI overrides
        if args.n_samples:
            script_args.extend(["--n-samples", str(args.n_samples)])
        if args.epochs:
            script_args.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            script_args.extend(["--batch-size", str(args.batch_size)])
        if args.learning_rate:
            script_args.extend(["--learning-rate", str(args.learning_rate)])
        if args.timesteps:
            script_args.extend(["--timesteps", str(args.timesteps)])
        if args.val_split:
            script_args.extend(["--val-split", str(args.val_split)])
        if args.resume_from:
            script_args.extend(["--resume-from", str(args.resume_from)])
        if args.verbose:
            script_args.append("--verbose")
        if args.log_every != 10:
            script_args.extend(["--log-every", str(args.log_every)])

        # Launch distributed training
        try:
            launch_distributed_training(dist_config, script_path, script_args)
            return 0
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            import traceback
            print(f"\nError launching distributed training: {e}", file=sys.stderr)
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

        rank_id = args.distributed_rank if hasattr(args, 'distributed_rank') and args.distributed_rank is not None else "master"
        print(f"[{rank_id}] Initializing training...")
        sys.stdout.flush()

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

        # Setup distributed training if needed
        print(f"[{rank_id}] Checking distributed training setup (rank={args.distributed_rank})...")
        sys.stdout.flush()

        is_distributed = args.distributed_rank is not None
        if is_distributed:
            from spinlock.distributed import setup_process_group, get_rank, is_main_process

            rank = args.distributed_rank
            world_size = args.distributed_world_size

            print(f"[Rank {rank}] Starting distributed initialization...")
            sys.stdout.flush()

            # Parse distributed config
            from spinlock.distributed import DistributedConfig
            dist_config = DistributedConfig.from_dict(config.get("distributed", {}))

            print(f"[Rank {rank}] Master: {dist_config.get_master_addr()}:{dist_config.master_port}")
            sys.stdout.flush()

            # Initialize process group
            print(f"[Rank {rank}] Initializing process group with {dist_config.backend} backend...")
            sys.stdout.flush()

            setup_process_group(rank, world_size, dist_config)

            print(f"[Rank {rank}] Process group initialized successfully")
            sys.stdout.flush()

            # Update device to local GPU
            local_rank = rank % torch.cuda.device_count()
            device = f"cuda:{local_rank}"

            if is_main_process():
                print(f"\n  ✓ Distributed training initialized: rank {rank}/{world_size}")
        else:
            rank = 0
            world_size = 1

        # Create model
        print("Creating NOA backbone..." if rank == 0 else "")
        noa = NOABackbone(**config["model"])
        noa = noa.to(device)
        if rank == 0:
            print(f"  ✓ NOA created ({sum(p.numel() for p in noa.parameters()):,} parameters)")

        # Wrap model in DDP if distributed
        if is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            noa = DDP(noa, device_ids=[local_rank], output_device=local_rank)
            if rank == 0:
                print(f"  ✓ Model wrapped in DistributedDataParallel")
            # Get underlying model for rollout (DDP adds .module attribute)
            noa_base = noa.module
        else:
            noa_base = noa

        # Wrap with truncated BPTT if configured
        timesteps = config["training"]["timesteps"]
        bptt_window = config["training"].get("bptt_window")

        from spinlock.noa import TruncatedBPTT

        if bptt_window is not None and bptt_window < timesteps:
            if rank == 0:
                print(f"  ✓ Using truncated BPTT: {timesteps} steps, backprop window={bptt_window}")
            noa_rollout = TruncatedBPTT(noa_base, timesteps=timesteps, bptt_window=bptt_window)
        else:
            if rank == 0:
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

            noa_rollout = FullBPTTWrapper(noa_base, timesteps)

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
            sampling_strategy=config["training"].get("sampling_strategy", "stratified"),
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

        # Determine if we should shuffle based on sampling strategy
        # Sequential (Sobol): NO shuffle - preserves low-discrepancy property + enables resumption
        # Other strategies: shuffle for better training, but resumption less precise
        sampling_strategy = config["training"].get("sampling_strategy", "stratified")
        use_shuffle = sampling_strategy != "sequential" and not is_distributed

        if not use_shuffle and rank == 0:
            print(f"  ✓ No shuffle (preserves {sampling_strategy} order + enables exact resumption)")

        # Use DistributedSampler for multi-GPU training
        train_sampler = None
        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=use_shuffle,
                seed=seed,
            )
            if rank == 0:
                print(f"  ✓ Using DistributedSampler (world_size={world_size})")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=use_shuffle if train_sampler is None else False,
            sampler=train_sampler,
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
        resume_batch_counter = 0

        if "resume_from" in config:
            print(f"Resuming from checkpoint: {config['resume_from']}")
            checkpoint = torch.load(config["resume_from"], map_location=device)
            noa.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Check if this is a mid-epoch checkpoint
            saved_batch_counter = checkpoint.get("global_batch_counter", None)
            saved_epoch = checkpoint["epoch"]

            # Backward compatibility: parse batch number from filename if not in checkpoint
            if saved_batch_counter is None:
                checkpoint_filename = Path(config["resume_from"]).name
                batch_match = re.search(r'_batch(\d+)\.pt$', checkpoint_filename)
                if batch_match:
                    saved_batch_counter = int(batch_match.group(1))
                    print(f"  ℹ Old checkpoint format detected, inferred batch {saved_batch_counter} from filename")

            if saved_batch_counter is not None:
                # Mid-epoch checkpoint: continue same epoch from saved batch
                start_epoch = saved_epoch
                resume_batch_counter = saved_batch_counter
                print(f"  ✓ Resuming epoch {saved_epoch + 1} from batch {saved_batch_counter}")
            else:
                # End-of-epoch checkpoint: start next epoch
                start_epoch = saved_epoch + 1
                resume_batch_counter = 0
                print(f"  ✓ Resuming from end of epoch {saved_epoch + 1}, starting epoch {start_epoch + 1}")

            best_val_loss = checkpoint.get("val_loss", float('inf'))
            print(f"  ✓ Best val loss: {best_val_loss:.6f}")

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

        # Mid-epoch checkpoint setup
        save_every_batches = config["checkpointing"].get("save_every_batches", None)
        global_batch_counter = resume_batch_counter  # Start from resumed position

        # Training loop
        for epoch in range(start_epoch, config["training"]["epochs"]):
            print(f"Epoch {epoch + 1}/{config['training']['epochs']}")

            # Train (with mid-epoch validation/checkpointing if configured)
            train_metrics, global_batch_counter, best_val_loss = self._train_epoch(
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
                # Mid-epoch validation/checkpointing
                val_loader=val_loader if save_every_batches else None,
                save_every_batches=save_every_batches,
                global_batch_counter=global_batch_counter,
                best_val_loss=best_val_loss,
                epoch=epoch,
                config=config,
                save_dir=save_dir,
                scheduler=scheduler,
                # Distributed training
                rank=rank,
                noa_base=noa_base,
            )

            # Validate
            # Optional: Limit validation batches for faster epochs
            max_val_batches = config["training"].get("max_val_batches", None)
            val_metrics = self._validate_epoch(
                noa=noa,
                noa_rollout=noa_rollout,
                loss_fn=loss_fn,
                replayer=replayer,
                dataloader=val_loader,
                device=device,
                timesteps=config["training"]["timesteps"],
                ground_truth_tokens=ground_truth_tokens,
                max_batches=max_val_batches,
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

            # Save checkpoint (only on rank 0)
            if rank == 0:
                save_every = config["checkpointing"].get("save_every", 5)
                if (epoch + 1) % save_every == 0 or (epoch + 1) == config["training"]["epochs"]:
                    checkpoint_path = save_dir / f"meta_operator_epoch{epoch+1}.pt"
                    self._save_checkpoint(
                        path=checkpoint_path,
                        noa=noa_base,  # Save unwrapped model
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        train_loss=train_metrics['loss'],
                        val_loss=val_metrics['loss'],
                        config=config,
                        global_batch_counter=None,  # End-of-epoch checkpoint
                    )
                    print(f"  Saved checkpoint: {checkpoint_path}")

            # Save best checkpoint and check early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0  # Reset counter
                if rank == 0:
                    best_checkpoint_path = save_dir / "meta_operator_best.pt"
                    self._save_checkpoint(
                        path=best_checkpoint_path,
                        noa=noa_base,  # Save unwrapped model
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        train_loss=train_metrics['loss'],
                        val_loss=val_metrics['loss'],
                        config=config,
                        global_batch_counter=global_batch_counter,  # Include batch position
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

        if rank == 0:
            print(f"{'='*70}")
            print("Training Complete")
            print(f"{'='*70}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Checkpoints saved to: {save_dir}")
            print(f"Training log: {log_file}")
            print()

        # Cleanup distributed training
        if is_distributed:
            from spinlock.distributed import cleanup_process_group
            cleanup_process_group()

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
        # Mid-epoch validation/checkpointing
        val_loader=None,
        save_every_batches=None,
        global_batch_counter=0,
        best_val_loss=float('inf'),
        epoch=0,
        config=None,
        save_dir=None,
        scheduler=None,
        # Distributed training
        rank=0,
        noa_base=None,  # Unwrapped model for checkpointing
    ) -> tuple:
        """Train for one epoch with gradient accumulation."""
        noa.train()
        total_loss = 0.0
        component_losses = {}  # Track individual loss components
        num_batches = 0
        start_time = time.time()

        # Calculate how many batches to skip if resuming mid-epoch
        batches_per_epoch = len(dataloader)
        batches_processed_in_epoch = global_batch_counter % batches_per_epoch if batches_per_epoch > 0 else 0

        # Initialize gradients outside loop for accumulation
        optimizer.zero_grad()

        # Create iterator and fast-forward if resuming mid-epoch
        dataloader_iter = iter(dataloader)
        if batches_processed_in_epoch > 0:
            print(f"  ⏩ Resuming mid-epoch: fast-forwarding past {batches_processed_in_epoch} batches...")
            # Consume (skip) the first N batches without loading their data
            for _ in itertools.islice(dataloader_iter, batches_processed_in_epoch):
                pass
            print(f"  ✓ Skipped {batches_processed_in_epoch} batches, resuming from batch {batches_processed_in_epoch + 1}")

        for batch_idx, batch in enumerate(dataloader_iter, start=batches_processed_in_epoch):
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

            # Mid-epoch validation and checkpointing
            global_batch_counter += 1
            if save_every_batches and (global_batch_counter % save_every_batches == 0):
                print(f"\n  >>> Mid-epoch checkpoint at batch {global_batch_counter} (epoch {epoch+1}, batch {batch_idx+1})")

                # Run validation
                noa.eval()
                val_metrics = self._validate_epoch(
                    noa=noa,
                    noa_rollout=noa_rollout,
                    loss_fn=loss_fn,
                    replayer=replayer,
                    dataloader=val_loader,
                    device=device,
                    timesteps=timesteps,
                    ground_truth_tokens=ground_truth_tokens,
                )
                noa.train()  # Back to training mode

                print(f"  Val Loss: {val_metrics['loss']:.6f}")

                # Save checkpoint if this is the best model so far (only on rank 0)
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    if rank == 0:
                        model_to_save = noa_base if noa_base is not None else noa
                        best_checkpoint_path = save_dir / "meta_operator_best.pt"
                        self._save_checkpoint(
                            path=best_checkpoint_path,
                            noa=model_to_save,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            train_loss=total_loss / num_batches,
                            val_loss=val_metrics['loss'],
                            config=config,
                            global_batch_counter=global_batch_counter,  # Mid-epoch position
                        )
                        print(f"  New best checkpoint: {best_checkpoint_path} (val_loss={best_val_loss:.6f})")

                # Always save periodic checkpoint (only on rank 0)
                if rank == 0:
                    model_to_save = noa_base if noa_base is not None else noa
                    checkpoint_path = save_dir / f"meta_operator_epoch{epoch+1}_batch{global_batch_counter}.pt"
                    self._save_checkpoint(
                        path=checkpoint_path,
                        noa=model_to_save,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        train_loss=total_loss / num_batches,
                        val_loss=val_metrics['loss'],
                        config=config,
                        global_batch_counter=global_batch_counter,  # Mid-epoch position
                    )
                    print(f"  Saved periodic checkpoint: {checkpoint_path}\n")

        # Handle leftover gradients at end of epoch
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(noa.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        train_metrics = {
            "loss": avg_loss,
            "time": elapsed,
        }

        return train_metrics, global_batch_counter, best_val_loss

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
        max_batches=None,
    ) -> dict:
        """Validate for one epoch."""
        noa.eval()
        total_loss = 0.0
        component_losses = {}  # Track individual loss components
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Optional early stopping for faster validation
                if max_batches is not None and batch_idx >= max_batches:
                    break
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
        global_batch_counter=None,
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
            'global_batch_counter': global_batch_counter,  # Save batch position
        }

        torch.save(checkpoint, path)
