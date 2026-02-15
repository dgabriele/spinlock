"""
Train Meta-Operator command for Spinlock CLI.

Trains MNO as a precision physics meta-operator using pure trajectory matching.
Supports both MSE-led (Stage 1) and VQ-led (Stage 2) training paradigms.

Documentation:
    - Training guide: docs/mno-training-guide.md
    - Two-stage curriculum: docs/two-stage-curriculum-architecture.md
    - MNO architecture: docs/mno-architecture.md
    - MNO architecture spec: docs/MNO_ARCHITECTURE.md
    - Truncated BPTT: docs/truncated-bptt-integration.md
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import sys
import os
import warnings
import yaml
import time
import itertools
import re

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Suppress PyTorch 2.7+ scheduler deprecation warning (internal to SequentialLR)
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Suppress torch._inductor warning about SMs
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

from .base import CLICommand


class TrainMetaOperatorCommand(CLICommand):
    """
    Command to train MNO as a precision physics meta-operator.

    Stage 1 of two-stage training: Train MNO purely on physics (trajectory matching)
    without any VQ involvement. The trained checkpoint can then be used in Stage 2
    to train VQ-VAE on MNO's distribution.
    """

    @property
    def name(self) -> str:
        return "train-meta-operator"

    @property
    def help(self) -> str:
        return "Train MNO as precision physics meta-operator (Stage 1)"

    @property
    def description(self) -> str:
        return """
Train MNO as a precision physics meta-operator using pure trajectory matching.

This is Stage 1 of the two-stage training approach:
- Stage 1 (this command): Train MNO on pure physics (MSE vs CNO rollouts)
- Stage 2 (train-vqvae): Train VQ-VAE on MNO's distribution

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
    - model_state_dict: MNO weights
    - optimizer_state_dict: Optimizer state (for resuming)
    - scheduler_state_dict: LR scheduler state
    - config: Full config (for reproducibility in Stage 2)
    - val_loss: Validation loss
    - train_loss: Training loss
    - epoch: Current epoch
    - timestamp: Training timestamp

Examples:
  # Train with config file
  spinlock train-meta-operator --config configs/mno/train_meta_operator.yaml

  # Override parameters
  spinlock train-meta-operator \\
      --config configs/mno/train_meta_operator.yaml \\
      --n-samples 1000 \\
      --epochs 20 \\
      --learning-rate 1e-4

  # Resume from checkpoint
  spinlock train-meta-operator \\
      --config configs/mno/train_meta_operator.yaml \\
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
        from spinlock.distributed import DistributedConfig

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

        if dist_config.backend == "ssh":
            print(f"Master: {dist_config.get_master_addr()}:{dist_config.master_port}")
            print(f"\nNodes:")
            for i, node in enumerate(dist_config.nodes):
                print(f"  [{i}] {node.host}: {len(node.gpus)} GPU(s) {node.gpus}")
        elif dist_config.backend == "salad":
            print(f"Organization: {dist_config.salad.organization}")
            print(f"Project: {dist_config.salad.project}")
            print(f"GPU: {dist_config.salad.resources.gpu_class}")
            print(f"Replicas: {dist_config.salad.resources.num_replicas}")

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

        # Launch based on backend type
        try:
            if dist_config.backend == "salad":
                # Launch on Salad.com
                from spinlock.distributed.salad.launcher import launch_salad_training
                launch_salad_training(config, script_path, script_args)
            else:
                # Launch via SSH (existing launcher)
                from spinlock.distributed import launch_distributed_training
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
        """Validate configuration has required fields.

        Note: Data-dependent dimensions (spatial_dim, in_channels, out_channels, param_dim)
        are auto-detected at runtime and validated later during training initialization.
        Only algorithmic choices are validated here.
        """
        # Validate model section
        if "model" not in config:
            raise ValueError(
                "Missing required section: 'model'\n\n"
                "Example:\n"
                "  model:\n"
                "    # Data-dependent dimensions (AUTO-DETECTED from dataset):\n"
                "    # spatial_dim: auto-detected from inputs/fields shape\n"
                "    # in_channels: auto-detected from dataset metadata\n"
                "    # out_channels: auto-detected from dataset metadata\n"
                "    # param_dim: auto-detected from parameters/params shape\n"
                "    \n"
                "    # Algorithmic choices (USER-SPECIFIED):\n"
                "    base_channels: 32\n"
                "    encoder_levels: 3\n"
                "    modes: 16\n"
                "    afno_blocks: 4"
            )

        # Validate only algorithmic choices (data-dependent fields are auto-detected)
        required_model_fields = ["base_channels", "encoder_levels", "modes", "afno_blocks"]
        for field in required_model_fields:
            if field not in config["model"]:
                raise ValueError(
                    f"Model section missing required algorithmic choice: '{field}'\n"
                    f"Note: Data-dependent dimensions (spatial_dim, in_channels, etc.) "
                    f"are auto-detected from the dataset."
                )

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

    def _sync_data_from_cloud(self, config: Dict[str, Any]) -> None:
        """Sync dataset and checkpoints from cloud storage if on Salad."""
        # Check if running on Salad (environment variable set by container)
        if not os.environ.get("SALAD_MACHINE_ID"):
            return  # Not on Salad, skip sync

        print("[Salad] Detected Salad environment, syncing data from cloud...")

        from spinlock.distributed.salad.storage import create_storage_backend, StorageManager

        salad_config = config.get("distributed", {}).get("salad", {})
        if not salad_config:
            print("[Salad] No Salad config found, skipping sync")
            return

        # Initialize storage
        storage_backend = create_storage_backend(salad_config["storage"])
        storage = StorageManager(storage_backend)

        # Download dataset
        dataset_path = Path(config["data"]["dataset_path"])
        remote_dataset = salad_config["storage"]["dataset_path"]
        storage.sync_dataset(remote_dataset, dataset_path)

        print("[Salad] ✓ Data sync complete")

    def _sync_checkpoint_to_cloud(self, config: Dict[str, Any], checkpoint_path: Path) -> None:
        """Upload checkpoint to cloud storage if on Salad."""
        if not os.environ.get("SALAD_MACHINE_ID"):
            return

        print("[Salad] Uploading checkpoint to cloud...")

        from spinlock.distributed.salad.storage import create_storage_backend, StorageManager

        salad_config = config.get("distributed", {}).get("salad", {})
        if not salad_config:
            return

        storage_backend = create_storage_backend(salad_config["storage"])
        storage = StorageManager(storage_backend)

        remote_path = f"{salad_config['storage']['checkpoint_path']}/{checkpoint_path.name}"
        storage.sync_checkpoint(checkpoint_path, remote_path)

        print("[Salad] ✓ Checkpoint uploaded")

    def _run_training(self, config: Dict[str, Any], args: Namespace) -> int:
        """Execute training pipeline."""
        from spinlock.mno import MNOBackbone, CNOReplayer
        from spinlock.mno.losses import MSELedLoss
        from spinlock.operators.state_dataset import NOAStateDataset

        # Sync data from cloud if on Salad
        self._sync_data_from_cloud(config)

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

        # Use unified dimension inference pattern (framework-compliant)
        from spinlock.data import SpinlockDataset
        from pathlib import Path

        rank_id = args.distributed_rank if hasattr(args, 'distributed_rank') and args.distributed_rank is not None else "master"
        if rank == 0:
            print(f"[{rank_id}] Loading dataset and inferring dimensions...")
            sys.stdout.flush()

        # Load dataset and infer dimensions (single pass, efficient)
        dataset_path = Path(config["data"]["dataset_path"])
        spinlock_dataset, config = SpinlockDataset.infer_and_update_config(
            config_dict=config,
            dataset_path=dataset_path,
            verbose=(rank == 0)
        )

        # Apply MNO-specific dimension inference
        mno_dimensions = spinlock_dataset.infer_mno_dimensions()
        for key, value in mno_dimensions.items():
            if isinstance(value, dict):
                config.setdefault(key, {}).update(value)
            else:
                config[key] = value

        # Validate timesteps
        max_timesteps = config.get('training', {}).get('max_timesteps')
        requested_timesteps = config["training"]["timesteps"]
        if max_timesteps and requested_timesteps > max_timesteps:
            raise ValueError(
                f"Requested timesteps ({requested_timesteps}) exceeds dataset maximum ({max_timesteps}). "
                f"Reduce training.timesteps in config."
            )

        if rank == 0:
            print(f"✓ Auto-detected dimensions:")
            print(f"  in_channels: {config['model']['in_channels']}")
            print(f"  out_channels: {config['model']['out_channels']}")
            print(f"  param_dim: {config['model'].get('param_dim', 'N/A')}")
            print(f"  spatial_dim: {config['model'].get('spatial_dim', 'N/A')}")
            print(f"  max_timesteps: {max_timesteps}")

        # Create model
        if rank == 0:
            print("Creating MNO backbone...")

        # Prepare model config
        model_config = dict(config["model"])

        # Transform 'film' section to 'film_config' for MNOBackbone constructor
        if "film" in model_config:
            model_config["film_config"] = model_config.pop("film")

        mno = MNOBackbone(**model_config)
        mno = mno.to(device)
        if rank == 0:
            print(f"  ✓ MNO created ({sum(p.numel() for p in mno.parameters()):,} parameters)")

            # Log FiLM configuration if enabled
            conditioning_mode = model_config.get("conditioning_mode", "concat")
            if conditioning_mode in ("film", "both"):
                print(f"  ✓ FiLM conditioning enabled (mode: {conditioning_mode})")
                if hasattr(mno, 'operator') and hasattr(mno.operator, 'get_film_parameter_count'):
                    film_params = mno.operator.get_film_parameter_count()
                    total_params = sum(p.numel() for p in mno.parameters())
                    overhead_pct = 100 * film_params / total_params if total_params > 0 else 0
                    print(f"    FiLM parameters: {film_params:,} ({overhead_pct:.1f}% overhead)")

        # Apply torch.compile for speedup (before DDP wrapping)
        use_torch_compile = config.get("training", {}).get("use_torch_compile", False)
        if use_torch_compile and "cuda" in str(device) and not is_distributed:
            if rank == 0:
                print("  Compiling model with torch.compile...")
            try:
                mno = torch.compile(mno, mode="default")
                if rank == 0:
                    print("  ✓ torch.compile enabled (expect 2-3× speedup after warmup)")
            except Exception as e:
                if rank == 0:
                    print(f"  ⚠ torch.compile failed: {e}, continuing without compilation")

        # Wrap model in DDP if distributed
        if is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            mno = DDP(mno, device_ids=[local_rank], output_device=local_rank)
            if rank == 0:
                print(f"  ✓ Model wrapped in DistributedDataParallel")
            # Get underlying model for rollout (DDP adds .module attribute)
            noa_base = mno.module
        else:
            noa_base = mno

        # Wrap with truncated BPTT if configured
        timesteps = config["training"]["timesteps"]
        bptt_window = config["training"].get("bptt_window")

        from spinlock.mno import TruncatedBPTT

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

                def rollout(self, ic, params=None, tokens=None):
                    return self.model.rollout(ic, steps=self.timesteps, return_all_steps=True, params=params, tokens=tokens)

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
            from spinlock.mno.vqvae_alignment import VQVAEAlignmentLoss

            vqvae_alignment = VQVAEAlignmentLoss.from_checkpoint(
                vqvae_path=vqvae_checkpoint,
                device=device,
                use_aligned_extractor=vqvae_config.get("use_aligned_extractor", True),
                enable_latent_loss=False,  # Not used in Stage 2
                normalization_stats_file=vqvae_config.get("normalization_stats_file"),
            )
            print(f"  ✓ VQ-VAE alignment loaded")

            # Create VQ-led loss
            from spinlock.mno.losses.vq_led import VQLedLoss

            loss_fn = VQLedLoss(
                lambda_recon=config["loss"].get("lambda_recon", 1.0),
                lambda_commit=config["loss"].get("lambda_commit", 0.5),
                lambda_traj=config["loss"].get("lambda_traj", 0.3),
                vqvae_alignment=vqvae_alignment,
            )
            print(f"  ✓ VQ-led loss (L_recon={config['loss'].get('lambda_recon', 1.0)}, "
                  f"L_commit={config['loss'].get('lambda_commit', 0.5)}, "
                  f"L_traj={config['loss'].get('lambda_traj', 0.3)})")
        elif loss_mode == "token_diversity":
            # Token diversity: contrastive primary, VQ coherence optional
            from spinlock.mno.losses import TokenDiversityLoss

            loss_config = config["loss"]
            lambda_recon = loss_config.get("lambda_recon", 0.0)
            lambda_commit = loss_config.get("lambda_commit", 0.0)

            # Load VQ coherence adapter only if VQ losses are enabled
            vq_adapter = None
            contrastive_feature_dim = None
            if lambda_recon > 0 or lambda_commit > 0:
                vqvae_config = config.get("vqvae")
                if not vqvae_config:
                    return self.error(
                        "token_diversity mode with lambda_recon > 0 or lambda_commit > 0 "
                        "requires 'vqvae' section in config.\n"
                        "Either add vqvae.checkpoint or set lambda_recon=0, lambda_commit=0."
                    )

                vqvae_checkpoint = vqvae_config.get("checkpoint")
                if not vqvae_checkpoint:
                    return self.error("VQ losses require vqvae.checkpoint path")

                print(f"  Loading VQ coherence adapter from: {vqvae_checkpoint}")
                from spinlock.mno.vq_coherence import VQCoherenceAdapter

                vq_adapter = VQCoherenceAdapter.from_checkpoint(
                    checkpoint_path=vqvae_checkpoint,
                    device=device,
                )
                contrastive_feature_dim = vq_adapter.cleaned_feature_dim
                print(f"  ✓ VQ coherence adapter loaded (feature_dim={contrastive_feature_dim})")

            contrastive_config = loss_config.get("contrastive", {})
            loss_fn = TokenDiversityLoss(
                lambda_contrastive=loss_config.get("lambda_contrastive", 1.0),
                lambda_recon=lambda_recon,
                lambda_commit=lambda_commit,
                lambda_traj=loss_config.get("lambda_traj", 0.0),
                lambda_ic=loss_config.get("lambda_ic", 0.0),
                vq_adapter=vq_adapter,
                param_dim=config["model"]["param_dim"],
                contrastive_temperature=contrastive_config.get("temperature", 0.1),
                contrastive_queue_size=contrastive_config.get("queue_size", 0),
                contrastive_feature_dim=contrastive_feature_dim,
            )
            loss_fn = loss_fn.to(device)
            print(f"  ✓ Token diversity loss:")
            print(f"    L_contrastive={loss_config.get('lambda_contrastive', 1.0)}, "
                  f"L_recon={lambda_recon}, "
                  f"L_commit={lambda_commit}")
            print(f"    L_traj={loss_config.get('lambda_traj', 0.0)}, "
                  f"L_ic={loss_config.get('lambda_ic', 0.0)}")
            if vq_adapter is None:
                print(f"    VQ coherence: disabled (lambda_recon=0, lambda_commit=0)")
            else:
                print(f"    Contrastive space: VQ feature-space ({contrastive_feature_dim}D)")
            queue_size = contrastive_config.get("queue_size", 0)
            if queue_size > 0:
                print(f"    Contrastive queue: {queue_size} (MoCo-style memory bank)")
            if not loss_fn.needs_target_trajectory:
                print(f"    ⚡ Replayer-skip mode: QBM rollout will be skipped (faster training)")

        elif loss_mode == "mse_led_param_sensitive":
            # MSE-led + parameter sensitivity losses
            from spinlock.mno.losses import ParameterSensitiveLoss

            loss_config = config["loss"]
            loss_fn = ParameterSensitiveLoss(
                lambda_traj=loss_config.get("lambda_traj", 1.0),
                lambda_ic=loss_config.get("lambda_ic", 0.3),
                lambda_param_recon=loss_config.get("lambda_param_recon", 0.5),
                lambda_contrastive=loss_config.get("lambda_contrastive", 0.3),
                lambda_sensitivity=loss_config.get("lambda_param_sensitivity", 0.2),
                # Parameter dimension (auto-detected from dataset)
                param_dim=config["model"]["param_dim"],
                # Parameter reconstruction config
                param_recon_hidden_dim=loss_config.get("param_recon", {}).get("hidden_dim", 128),
                param_recon_num_layers=loss_config.get("param_recon", {}).get("num_layers", 2),
                # Contrastive config
                contrastive_temperature=loss_config.get("contrastive", {}).get("temperature", 0.1),
                # Sensitivity config
                sensitivity_epsilon=loss_config.get("sensitivity", {}).get("epsilon", 0.01),
                sensitivity_target_ratio=loss_config.get("sensitivity", {}).get("target_ratio", 0.1),
            )
            # Move loss to device (contains learnable parameters)
            loss_fn = loss_fn.to(device)
            print(f"  ✓ Parameter-sensitive loss:")
            print(f"    L_traj={loss_config.get('lambda_traj', 1.0)}, "
                  f"L_ic={loss_config.get('lambda_ic', 0.3)}")
            print(f"    L_param_recon={loss_config.get('lambda_param_recon', 0.5)}, "
                  f"L_contrastive={loss_config.get('lambda_contrastive', 0.3)}, "
                  f"L_sensitivity={loss_config.get('lambda_param_sensitivity', 0.2)}")
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

        # Create substrate replayer (CNO or QBM based on substrate config structure)
        print("Loading substrate replayer...")
        substrate_config_path = config["data"].get("config")

        if not substrate_config_path:
            return self.error(
                "Data section missing substrate configuration.\n"
                "Required field: config (path to substrate config YAML)"
            )

        # Detect replayer type from substrate config structure
        # CNO: has 'operators' section
        # QBM: has 'parameter_space' section
        import yaml
        with open(substrate_config_path, 'r') as f:
            substrate_config = yaml.safe_load(f)

        if 'parameter_space' in substrate_config:
            # QBM substrate - use QBMReplayer
            from spinlock.qbm import QBMReplayer
            replayer = QBMReplayer.from_config(
                config_path=substrate_config_path,
                device=device
            )
            print(f"  ✓ QBM replayer loaded")
        elif 'operators' in substrate_config:
            # CNO substrate - use CNOReplayer
            cache_size = config.get("training", {}).get("replayer_cache_size", 512)
            replayer = CNOReplayer.from_config(
                config_path=substrate_config_path,
                device=device,
                cache_size=cache_size,
            )
            print(f"  ✓ CNO replayer loaded (cache_size={cache_size})")
        else:
            return self.error(
                f"Unknown substrate config format: {substrate_config_path}\n"
                f"Expected either 'operators' (CNO) or 'parameter_space' (QBM) section"
            )

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

        num_workers = config["data"].get("num_workers", 4)
        use_cuda = torch.cuda.is_available() and "cuda" in str(device)
        pin_memory = config["data"].get("pin_memory", use_cuda) and num_workers > 0
        persistent_workers = config["data"].get("persistent_workers", True) and num_workers > 0
        prefetch_factor = config["data"].get("prefetch_factor", 2) if num_workers > 0 else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=use_shuffle if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        print(f"  ✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")

        # Create optimizer and scheduler
        # Check if FiLM learning rate multiplier is configured
        film_lr_multiplier = config["training"].get("film_lr_multiplier", 1.0)
        base_lr = config["training"]["learning_rate"]
        weight_decay = config["training"].get("weight_decay", 1e-6)

        if film_lr_multiplier != 1.0:
            # Separate parameter groups for FiLM vs other parameters
            film_params = []
            other_params = []

            for name, param in mno.named_parameters():
                if 'film' in name.lower():
                    film_params.append(param)
                else:
                    other_params.append(param)

            optimizer = torch.optim.AdamW([
                {'params': other_params, 'lr': base_lr},
                {'params': film_params, 'lr': base_lr * film_lr_multiplier},
            ], weight_decay=weight_decay)

            print(f"  ✓ FiLM parameters get {film_lr_multiplier}x learning rate ({base_lr * film_lr_multiplier:.2e})")
        else:
            optimizer = torch.optim.AdamW(
                mno.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
            )

        # Create LR scheduler with optional warmup
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        # Calculate total training steps (scheduler steps per batch, not per epoch)
        batches_per_epoch = len(train_loader)
        total_epochs = config["training"]["epochs"]
        total_steps = batches_per_epoch * total_epochs

        warmup_steps = config["training"].get("warmup_steps", 0)
        if warmup_steps > 0:
            # Warmup: LR ramps from 0.1x to 1.0x over warmup_steps (batches)
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
            # Cosine: LR decays from 1.0x to 0 over remaining steps (batches)
            cosine_steps = max(1, total_steps - warmup_steps)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
            )
            # Sequential: warmup first, then cosine
            scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
            print(f"  ✓ LR schedule: {warmup_steps}-batch warmup + {cosine_steps}-batch cosine decay")
            print(f"    Total: {total_steps} batches ({batches_per_epoch} batches/epoch × {total_epochs} epochs)")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
            )
            print(f"  ✓ LR schedule: {total_steps}-batch cosine decay (no warmup)")

        # Resume from checkpoint if specified
        start_epoch = 0
        best_val_loss = float('inf')
        resume_batch_counter = 0

        if "resume_from" in config:
            print(f"Resuming from checkpoint: {config['resume_from']}")
            checkpoint = torch.load(config["resume_from"], map_location=device)
            mno.load_state_dict(checkpoint["model_state_dict"])
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

        # AMP (Automatic Mixed Precision) setup
        # Default: disabled when gradient checkpointing is enabled (incompatible
        # with torch.utils.checkpoint — recomputed tensor graph doesn't match
        # under autocast). Enable manually with use_amp: true + use_checkpointing: false.
        use_checkpointing = config.get("model", {}).get("use_checkpointing", False)
        amp_default = not use_checkpointing
        use_amp = config["training"].get("use_amp", amp_default) and use_cuda
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
        if use_amp:
            print(f"  ✓ AMP enabled (dtype={amp_dtype})")
        else:
            reason = " (gradient checkpointing active)" if use_checkpointing else ""
            print(f"  ✗ AMP disabled{reason}")

        # Training loop
        for epoch in range(start_epoch, config["training"]["epochs"]):
            print(f"Epoch {epoch + 1}/{config['training']['epochs']}")

            # Train (with mid-epoch validation/checkpointing if configured)
            train_metrics, global_batch_counter, best_val_loss = self._train_epoch(
                mno=mno,
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
                # AMP
                scaler=scaler,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            # Validate
            # Optional: Limit validation batches for faster epochs
            max_val_batches = config["training"].get("max_val_batches", None)
            val_metrics = self._validate_epoch(
                mno=mno,
                noa_rollout=noa_rollout,
                loss_fn=loss_fn,
                replayer=replayer,
                dataloader=val_loader,
                device=device,
                timesteps=config["training"]["timesteps"],
                ground_truth_tokens=ground_truth_tokens,
                max_batches=max_val_batches,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            # Note: Scheduler is now stepped per-batch (inside _train_epoch)
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

            # Print normalized metrics if available
            if 'normalized' in val_metrics and val_metrics['normalized']:
                normalized_str = ", ".join([
                    f"{name}={value:.4f}"
                    for name, value in sorted(val_metrics['normalized'].items())
                ])
                print(f"  Val Normalized: {normalized_str}")

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
                        mno=noa_base,  # Save unwrapped model
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
                        mno=noa_base,  # Save unwrapped model
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
        mno,
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
        # AMP
        scaler=None,
        use_amp=False,
        amp_dtype=torch.float32,
    ) -> tuple:
        """Train for one epoch with gradient accumulation."""
        if scaler is None:
            scaler = torch.amp.GradScaler("cuda", enabled=False)
        mno.train()
        total_loss = 0.0
        component_losses = {}  # Track individual loss components
        normalized_metrics = {}  # Track normalized/relative metrics
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
            params = batch["params"].to(device)  # Move params to device for conditioning
            indices = batch.get("sample_idx")  # Dataset indices for this batch
            B = ic.shape[0]

            # Get ground-truth tokens for this batch if token conditioning is enabled
            batch_tokens = None
            if ground_truth_tokens is not None and indices is not None:
                # indices contains the original dataset indices (from sample_idx)
                # ground_truth_tokens is indexed by original dataset index
                batch_tokens = ground_truth_tokens[indices].to(device)

            # Generate MNO rollout under AMP autocast
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred_trajectory = noa_rollout.rollout(ic, params=params, tokens=batch_tokens)

            # Check if loss needs target trajectories (replayer-skip optimization)
            needs_target = not hasattr(loss_fn, 'needs_target_trajectory') or loss_fn.needs_target_trajectory

            if needs_target:
                # Generate CNO/QBM targets (outside autocast — replayer is not a neural net)
                # Only clear cache if memory usage > 90% (avoid unnecessary serialization)
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device)
                    max_allocated = torch.cuda.max_memory_allocated(device)
                    if max_allocated > 0 and allocated / max_allocated > 0.9:
                        torch.cuda.empty_cache()

                # Generate ground truth trajectories using batch rollout (parallel on GPU)
                # Check if replayer supports batch rollout (QBMReplayer does, CNOReplayer might not)
                try:
                    if hasattr(replayer, 'rollout_batch'):
                        # Batch rollout: run all simulations in parallel on GPU
                        target_trajectory = replayer.rollout_batch(
                            params_batch=params.cpu().numpy(),  # [B, param_dim]
                            num_realizations=1,
                            num_timesteps=timesteps,
                            timesteps=timesteps,  # Alias for compatibility
                            return_all_steps=True,
                        )
                        # rollout_batch returns [B, M, T, C, H, W], squeeze M=1 -> [B, T, C, H, W]
                        target_trajectory = target_trajectory.squeeze(1)
                    else:
                        # Fallback: sequential rollout (for CNOReplayer compatibility)
                        target_trajectories = []
                        skip_batch = False
                        for b in range(B):
                            try:
                                target_traj = replayer.rollout(
                                    params_vector=params[b].cpu().numpy(),
                                    ic=ic[b:b+1],  # [1, C, H, W]
                                    timesteps=timesteps,
                                    num_realizations=1,
                                    return_all_steps=True,
                                )
                                target_trajectories.append(target_traj)
                            except (ValueError, RuntimeError) as e:
                                print(f"  Warning: Replayer rollout failed for sample {b}: {e}")
                                skip_batch = True
                                break

                        if skip_batch:
                            continue
                        target_trajectory = torch.cat(target_trajectories, dim=0)
                except Exception as e:
                    print(f"  Warning: Batch rollout failed: {e}, skipping batch")
                    continue

                # Align predicted and target states for loss computation
                # (handles truncated BPTT windowing automatically)
                pred_states, target_states = noa_rollout.align_for_loss(
                    pred_trajectory,
                    target_trajectory,
                    skip_ic=True,
                )
            else:
                # No target needed — skip expensive replayer rollout
                pred_states = pred_trajectory[:, 1:, :, :, :]  # Skip IC step
                target_states = None

            # Compute loss in FP32 — feature extraction includes ops that don't
            # support reduced precision (linalg_eigh, FFT, bincount)
            try:
                loss_output = loss_fn.compute(
                    pred_trajectory=pred_states.float() if pred_states is not None else None,
                    target_trajectory=target_states,
                    ic=ic,
                    mno=mno,
                    params=params,  # For parameter-sensitive losses
                )
            except Exception as e:
                print(f"  Warning: Loss computation failed: {e}")
                continue

            if torch.isnan(loss_output.total) or torch.isinf(loss_output.total):
                print(f"  Warning: NaN/Inf loss at batch {batch_idx}")
                continue

            # Gradient accumulation: scale loss and accumulate gradients
            scaled_loss = loss_output.total / accumulation_steps
            scaler.scale(scaled_loss).backward()

            # Only step optimizer every N batches
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mno.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # Step scheduler after optimizer step (per effective batch)
                if scheduler is not None:
                    scheduler.step()

            # Accumulate metrics
            total_loss += loss_output.total.item()
            num_batches += 1

            # Accumulate component losses
            for component_name, component_value in loss_output.components.items():
                if component_name not in component_losses:
                    component_losses[component_name] = 0.0
                component_losses[component_name] += component_value.item()

            # Accumulate normalized metrics
            for metric_name in ['nrmse', 'relative_l2', 'energy_norm_mse']:
                if metric_name in loss_output.metrics:
                    if metric_name not in normalized_metrics:
                        normalized_metrics[metric_name] = 0.0
                    normalized_metrics[metric_name] += loss_output.metrics[metric_name]

            # Log progress every N batches
            if (batch_idx + 1) % log_every == 0:
                avg_loss = total_loss / num_batches
                batch_total_time = time.time() - batch_start_time

                # Format component losses
                component_str = ", ".join([
                    f"{name}={component_losses[name] / num_batches:.4f}"
                    for name in sorted(component_losses.keys())
                ])

                # Format normalized metrics
                if normalized_metrics:
                    metrics_str = ", ".join([
                        f"{name}={normalized_metrics[name] / num_batches:.4f}"
                        for name in sorted(normalized_metrics.keys())
                    ])
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)}: total={avg_loss:.6f}, {component_str}, {metrics_str}, time={batch_total_time:.2f}s")
                else:
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)}: total={avg_loss:.6f}, {component_str}, time={batch_total_time:.2f}s")

            # Mid-epoch validation and checkpointing
            global_batch_counter += 1
            if save_every_batches and (global_batch_counter % save_every_batches == 0):
                print(f"\n  >>> Mid-epoch checkpoint at batch {global_batch_counter} (epoch {epoch+1}, batch {batch_idx+1})")

                # Run validation
                mno.eval()
                val_metrics = self._validate_epoch(
                    mno=mno,
                    noa_rollout=noa_rollout,
                    loss_fn=loss_fn,
                    replayer=replayer,
                    dataloader=val_loader,
                    device=device,
                    timesteps=timesteps,
                    ground_truth_tokens=ground_truth_tokens,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                mno.train()  # Back to training mode

                print(f"  Val Loss: {val_metrics['loss']:.6f}")

                # Print normalized metrics if available
                if 'normalized' in val_metrics and val_metrics['normalized']:
                    normalized_str = ", ".join([
                        f"{name}={value:.4f}"
                        for name, value in sorted(val_metrics['normalized'].items())
                    ])
                    print(f"  Val Normalized: {normalized_str}")

                # Save checkpoint if this is the best model so far (only on rank 0)
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    if rank == 0:
                        model_to_save = noa_base if noa_base is not None else mno
                        best_checkpoint_path = save_dir / "meta_operator_best.pt"
                        self._save_checkpoint(
                            path=best_checkpoint_path,
                            mno=model_to_save,
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
                    model_to_save = noa_base if noa_base is not None else mno
                    checkpoint_path = save_dir / f"meta_operator_epoch{epoch+1}_batch{global_batch_counter}.pt"
                    self._save_checkpoint(
                        path=checkpoint_path,
                        mno=model_to_save,
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mno.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
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
        mno,
        noa_rollout,
        loss_fn,
        replayer,
        dataloader,
        device,
        timesteps,
        ground_truth_tokens=None,
        token_indices=None,
        max_batches=None,
        use_amp=False,
        amp_dtype=torch.float32,
    ) -> dict:
        """Validate for one epoch."""
        mno.eval()
        total_loss = 0.0
        component_losses = {}  # Track individual loss components
        normalized_metrics = {}  # Track normalized/relative metrics
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Optional early stopping for faster validation
                if max_batches is not None and batch_idx >= max_batches:
                    break
                ic = batch["ic"].to(device)
                params = batch["params"].to(device)  # Move params to device for conditioning
                indices = batch.get("sample_idx")  # Dataset indices for this batch
                B = ic.shape[0]

                # Get ground-truth tokens for this batch if token conditioning is enabled
                batch_tokens = None
                if ground_truth_tokens is not None and indices is not None:
                    # indices contains the original dataset indices (from sample_idx)
                    # ground_truth_tokens is indexed by original dataset index
                    batch_tokens = ground_truth_tokens[indices].to(device)

                # Generate MNO rollout under AMP
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    pred_trajectory = noa_rollout.rollout(ic, params=params, tokens=batch_tokens)

                # Check if loss needs target trajectories (replayer-skip optimization)
                needs_target = not hasattr(loss_fn, 'needs_target_trajectory') or loss_fn.needs_target_trajectory

                if needs_target:
                    # Generate ground truth trajectories using batch rollout (parallel on GPU)
                    try:
                        if hasattr(replayer, 'rollout_batch'):
                            # Batch rollout: run all simulations in parallel on GPU
                            target_trajectory = replayer.rollout_batch(
                                params_batch=params.cpu().numpy(),  # [B, param_dim]
                                num_realizations=1,
                                num_timesteps=timesteps,
                                timesteps=timesteps,  # Alias for compatibility
                                return_all_steps=True,
                            )
                            # rollout_batch returns [B, M, T, C, H, W], squeeze M=1 -> [B, T, C, H, W]
                            target_trajectory = target_trajectory.squeeze(1)
                        else:
                            # Fallback: sequential rollout (for CNOReplayer compatibility)
                            target_trajectories = []
                            skip_batch = False
                            for b in range(B):
                                try:
                                    target_traj = replayer.rollout(
                                        params_vector=params[b].cpu().numpy(),
                                        ic=ic[b:b+1],  # [1, C, H, W]
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
                    except Exception as e:
                        print(f"  Warning: Batch rollout failed: {e}, skipping batch")
                        continue

                    # Align predicted and target states for loss computation
                    # (handles truncated BPTT windowing automatically)
                    pred_states, target_states = noa_rollout.align_for_loss(
                        pred_trajectory,
                        target_trajectory,
                        skip_ic=True,
                    )
                else:
                    # No target needed — skip expensive replayer rollout
                    pred_states = pred_trajectory[:, 1:, :, :, :]  # Skip IC step
                    target_states = None

                # Compute loss in FP32 (feature extraction needs full precision)
                try:
                    loss_output = loss_fn.compute(
                        pred_trajectory=pred_states.float() if pred_states is not None else None,
                        target_trajectory=target_states,
                        ic=ic,
                        mno=mno,
                        params=params,  # For parameter-sensitive losses
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

                # Accumulate normalized metrics
                for metric_name in ['nrmse', 'relative_l2', 'energy_norm_mse']:
                    if metric_name in loss_output.metrics:
                        if metric_name not in normalized_metrics:
                            normalized_metrics[metric_name] = 0.0
                        normalized_metrics[metric_name] += loss_output.metrics[metric_name]

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Compute average component losses
        avg_components = {
            name: value / num_batches if num_batches > 0 else float('inf')
            for name, value in component_losses.items()
        }

        # Compute average normalized metrics
        avg_normalized = {
            name: value / num_batches if num_batches > 0 else float('inf')
            for name, value in normalized_metrics.items()
        }

        return {
            "loss": avg_loss,
            "components": avg_components,
            "normalized": avg_normalized,
        }

    def _save_checkpoint(
        self,
        path: Path,
        mno,
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
            'model_state_dict': mno.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'config': config,  # Full config for reproducibility
            'timestamp': datetime.now().isoformat(),
            'global_batch_counter': global_batch_counter,  # Save batch position
        }

        torch.save(checkpoint, path)

        # Upload to cloud storage if on Salad
        self._sync_checkpoint_to_cloud(config, path)
