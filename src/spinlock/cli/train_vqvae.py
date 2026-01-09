"""
Train VQ-VAE command for Spinlock CLI.

Trains categorical hierarchical VQ-VAE for operator feature tokenization.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys
import warnings
import yaml
import numpy as np

# Suppress torch._inductor warning about SMs
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

from .base import CLICommand


class TrainVQVAECommand(CLICommand):
    """
    Command to train VQ-VAE tokenizer on extracted features.

    Implements categorical hierarchical VQ-VAE with auto-discovered categories
    and 3-level hierarchical quantization. Supports progressive training with
    resumable checkpoints.
    """

    @property
    def name(self) -> str:
        return "train-vqvae"

    @property
    def help(self) -> str:
        return "Train VQ-VAE tokenizer on operator features"

    @property
    def description(self) -> str:
        return """
Train categorical hierarchical VQ-VAE for operator feature tokenization.

The VQ-VAE learns discrete token representations of operator behaviors through:
1. Auto-discovered feature categories via hierarchical clustering
2. Per-category MLP encoders/decoders
3. 3-level hierarchical vector quantization (coarse → medium → fine)
4. 5-component loss function (reconstruction, VQ, orthogonality, informativeness, topology)

Production Configuration:
  Uses proven hyperparameters from unisim's agent_training_v1 pipeline.
  Single-stage training with high quality and utilization targets.

Feature Cleaning (enabled by default):
  Before training, features are automatically cleaned using a 4-step pipeline:
  1. Remove zero-variance features (std < 1e-8)
  2. Deduplicate highly correlated features (|corr| > 0.99)
  3. Replace NaN values with feature median
  4. Cap outliers using MAD (Median Absolute Deviation)

  This ensures numerical stability and clean categorical clustering.
  Disable with: clean_features: false in config.

Examples:
  # Train with production config
  spinlock train-vqvae --config configs/vqvae/production.json

  # Train on custom dataset
  spinlock train-vqvae \
      --config configs/vqvae/production.json \
      --input datasets/benchmark_50k.h5 \
      --output checkpoints/vqvae/production_50k

  # Resume from checkpoint
  spinlock train-vqvae \
      --config configs/vqvae/production.json \
      --resume-from checkpoints/vqvae/production_4k/best_model.pt

  # Override parameters
  spinlock train-vqvae \
      --config configs/vqvae/production.json \
      --epochs 500 \
      --batch-size 512

  # Dry run to validate configuration
  spinlock train-vqvae \
      --config configs/vqvae/production.json \
      --dry-run \
      --verbose

Output:
  The output directory will contain:
    - best_model.pt:            Best model checkpoint (composite metric)
    - normalization_stats.npz:  Per-category normalization stats
    - training_history.json:    Training metrics history
    - config.yaml:             Resolved configuration
        """


    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add train-vqvae command arguments."""
        # Required arguments
        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to VQ-VAE training config YAML",
        )

        # Configuration overrides
        override_group = parser.add_argument_group("configuration overrides")

        override_group.add_argument(
            "--input",
            type=Path,
            metavar="PATH",
            help="Override input dataset path from config",
        )

        override_group.add_argument(
            "--output",
            type=Path,
            metavar="PATH",
            help="Override output directory from config",
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
            "--resume-from",
            type=Path,
            metavar="PATH",
            help="Override checkpoint to resume from",
        )

        # Execution options
        exec_group = parser.add_argument_group("execution options")

        exec_group.add_argument(
            "--device",
            type=str,
            choices=["cuda", "cpu"],
            metavar="DEVICE",
            help="Device for computation (default: cuda)",
        )

        exec_group.add_argument(
            "--no-torch-compile",
            action="store_true",
            help="Disable torch.compile() optimization",
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

    def execute(self, args: Namespace) -> int:
        """Execute VQ-VAE training."""
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

        # Validate input dataset exists
        dataset_path = config.get("dataset_path")
        if not self.validate_file_exists(dataset_path, "Dataset"):
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
            return self._run_training(config, args.verbose)
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
        """Load VQ-VAE training config from YAML."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty config file: {config_path}")

        return config

    def _apply_cli_overrides(self, config: Dict[str, Any], args: Namespace) -> Dict[str, Any]:
        """Apply CLI argument overrides to config (assumes new nested format)."""
        if args.input:
            config["dataset_path"] = args.input
        if args.output:
            config["training"]["checkpoint_dir"] = args.output
        if args.epochs:
            config["training"]["num_epochs"] = args.epochs
        if args.batch_size:
            config["training"]["batch_size"] = args.batch_size
        if args.learning_rate:
            config["training"]["learning_rate"] = args.learning_rate
        if args.resume_from:
            config["resume_from"] = args.resume_from
        if args.device:
            config["device"] = args.device
        if args.no_torch_compile:
            config["training"]["use_torch_compile"] = False

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration has required fields and uses new nested format."""

        # Require multi-family nested format
        if "families" not in config:
            raise ValueError(
                "Missing required section: 'families'\n\n"
                "Example:\n"
                "  families:\n"
                "    summary:\n"
                "      encoder: MLPEncoder\n"
                "      encoder_params:\n"
                "        hidden_dims: [256, 128]\n"
                "        output_dim: 64\n\n"
                "See configs/vqvae/default.yaml for full template."
            )

        if "model" not in config:
            raise ValueError(
                "Missing required section: 'model'\n\n"
                "Example:\n"
                "  model:\n"
                "    group_embedding_dim: 64\n"
                "    group_hidden_dim: 128\n"
                "    levels: []\n\n"
                "See configs/vqvae/default.yaml for full template."
            )

        if "training" not in config:
            raise ValueError(
                "Missing required section: 'training'\n\n"
                "Example:\n"
                "  training:\n"
                "    batch_size: 512\n"
                "    learning_rate: 0.001\n"
                "    num_epochs: 500\n"
                "    checkpoint_dir: 'checkpoints/vqvae'\n\n"
                "See configs/vqvae/default.yaml for full template."
            )

        # Validate families structure
        if not isinstance(config["families"], dict) or len(config["families"]) == 0:
            raise ValueError(
                "Invalid 'families' section: must be a non-empty dictionary.\n"
                "Each family must specify 'encoder' and 'encoder_params'."
            )

        for family_name, family_config in config["families"].items():
            if "encoder" not in family_config:
                raise ValueError(f"Family '{family_name}': missing required field 'encoder'")
            if "encoder_params" not in family_config:
                raise ValueError(f"Family '{family_name}': missing required field 'encoder_params'")

        # Validate training section has required fields
        required_training_fields = ["num_epochs", "batch_size", "checkpoint_dir"]
        for field in required_training_fields:
            if field not in config["training"]:
                raise ValueError(
                    f"Missing required field in training section: '{field}'\n"
                    "See configs/vqvae/default.yaml for full template."
                )

        # Check for dataset path
        if "dataset_path" not in config:
            raise ValueError(
                "Missing required field: 'dataset_path'\n"
                "Please specify the path to your HDF5 dataset."
            )

        # Validate paths are Path objects or strings
        for path_field in ["dataset_path", "resume_from"]:
            if path_field in config and config[path_field] is not None:
                config[path_field] = Path(config[path_field])

        # Validate training.checkpoint_dir
        if config["training"]["checkpoint_dir"] is not None:
            config["training"]["checkpoint_dir"] = Path(config["training"]["checkpoint_dir"])

    def _print_config_summary(self, config: Dict[str, Any], args: Namespace) -> None:
        """Print configuration summary (assumes new nested format)."""
        print("\n" + "=" * 70)
        print("VQ-VAE TRAINING CONFIGURATION")
        print("=" * 70)

        # Dataset and output
        dataset_path = config.get("dataset_path")
        checkpoint_dir = config["training"]["checkpoint_dir"]
        print(f"\nDataset: {dataset_path}")
        print(f"Output:  {checkpoint_dir}")

        if config.get("max_samples"):
            print(f"Samples: {config['max_samples']} (limited)")

        if config.get("resume_from"):
            print(f"Resume:  {config['resume_from']}")

        # Feature families
        print(f"\nFeature Families:")
        for family_name, family_config in config["families"].items():
            encoder = family_config.get("encoder", "MLPEncoder")
            params = family_config.get("encoder_params", {})
            hidden = params.get("hidden_dims", [])
            output = params.get("output_dim", 64)
            print(f"  {family_name}:")
            print(f"    Encoder: {encoder}")
            print(f"    Hidden:  {hidden} → {output}")

        # Feature cleaning
        print(f"\nFeature Cleaning:")
        clean = config.get("clean_features", True)
        print(f"  Enabled: {clean}")
        if clean:
            print(f"  Variance threshold:     {config.get('variance_threshold', 1e-8)}")
            print(f"  Deduplicate threshold:  {config.get('deduplicate_threshold', 0.99)}")
            print(f"  MAD outlier threshold:  {config.get('mad_threshold', 5.0)}")

        # Normalization method
        print(f"\nNormalization:")
        normalization_method = config.get("normalization_method", "standard")
        if normalization_method == "mad":
            print(f"  Method: MAD (Median Absolute Deviation)")
            print(f"  Robust to outliers: Yes")
        else:
            print(f"  Method: Standard (mean/std)")
            print(f"  Robust to outliers: No")

        # Category discovery
        print(f"\nCategory Discovery:")
        category_assignment = config["training"].get("category_assignment", "auto")
        print(f"  Assignment: {category_assignment}")
        if category_assignment == "auto":
            num_cat = config["training"].get("num_categories_auto")
            print(f"  Categories: {num_cat if num_cat else 'auto-determine'}")
            print(f"  Orthogonality target: {config['training'].get('orthogonality_target', 0.15)}")
        elif category_assignment == "manual":
            mapping_file = config["training"].get("category_mapping_file", "N/A")
            print(f"  Mapping file: {mapping_file}")

        # Model architecture
        print(f"\nModel Architecture:")
        print(f"  Group embedding dim: {config['model']['group_embedding_dim']}")
        print(f"  Group hidden dim:    {config['model']['group_hidden_dim']}")
        levels = config['model'].get('levels', [])
        if levels:
            print(f"  Hierarchical levels: {len(levels)} levels")
            for i, level in enumerate(levels):
                print(f"    L{i}: latent_dim={level.get('latent_dim')}, num_tokens={level.get('num_tokens')}")
        else:
            print(f"  Hierarchical levels: auto-computed")

        # Training
        print(f"\nTraining:")
        print(f"  Epochs:        {config['training']['num_epochs']}")
        print(f"  Batch size:    {config['training']['batch_size']}")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        print(f"  Optimizer:     {config['training'].get('optimizer', 'adam')}")
        device = config.get("device", "cuda")
        print(f"  Device:        {device}")
        use_compile = config['training'].get('use_torch_compile', False)
        print(f"  torch.compile: {use_compile}")

        # Loss weights
        print(f"\nLoss Weights:")
        print(f"  Reconstruction:  {config['training'].get('reconstruction_weight', 1.0)}")
        print(f"  VQ:              {config['training'].get('vq_weight', 1.0)}")
        print(f"  Commitment:      {config['model'].get('commitment_cost', 0.25)}")
        print(f"  Orthogonality:   {config['training'].get('orthogonality_weight', 0.1)}")
        print(f"  Informativeness: {config['training'].get('informativeness_weight', 0.1)}")
        print(f"  Topographic:     {config['training'].get('topo_weight', 0.02)}")
        print(f"  Topo samples:    {config['training'].get('topo_samples', 64)}")

        # Callbacks
        print(f"\nCallbacks:")
        print(f"  Early stopping patience:  {config['training'].get('early_stopping_patience', 100)}")
        print(f"  Dead code reset interval: {config['training'].get('dead_code_reset_interval', 100)}")
        print(f"  Dead code threshold:      {config['training'].get('dead_code_threshold', 10.0)}th percentile")
        print(f"  Validation every:         {config['training'].get('val_every_n_epochs', 5)} epochs")

        print("=" * 70 + "\n")

    def _run_training(self, config: Dict[str, Any], verbose: bool) -> int:
        """
        Run VQ-VAE training pipeline.

        Pipeline:
        1. Load dataset and extract features
        2. Auto-discover categories (if needed)
        3. Per-category normalization
        4. Build VQ-VAE model
        5. Create data loaders
        6. Train with callbacks
        7. Save final model + metadata
        """
        import torch
        import numpy as np
        import h5py
        import time
        import random
        from pathlib import Path

        start_time = time.time()

        # Set all random seeds for reproducibility
        # This ensures deterministic encoder initialization, clustering, and training
        seed = config.get('random_seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # For full determinism (prevents CUDNN non-determinism)
        # Note: This may slightly reduce performance but ensures reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if verbose:
            print(f"Random seed: {seed} (deterministic mode enabled)")

        # Flatten nested config structure for internal training code
        # NOTE: This is an internal implementation detail. The training code
        # expects a flat structure, so we flatten the validated nested config here.
        flat_config = {}

        # Copy top-level keys
        for k, v in config.items():
            if k not in ["model", "training", "logging", "families"]:
                flat_config[k] = v

        # Flatten model section
        if "model" in config:
            for k, v in config["model"].items():
                flat_config[k] = v

        # Flatten training section
        if "training" in config:
            for k, v in config["training"].items():
                # Map num_epochs → epochs for internal use
                if k == "num_epochs":
                    flat_config["epochs"] = v
                else:
                    flat_config[k] = v

        # Flatten logging section
        if "logging" in config:
            for k, v in config["logging"].items():
                flat_config[k] = v

        # Keep families dict
        if "families" in config:
            flat_config["families"] = config["families"]

        config = flat_config

        # Set default category_assignment if not specified
        if "category_assignment" not in config and "resume_from" not in config:
            config["category_assignment"] = "auto"

        # Create output directory
        output_dir = Path(config.get("checkpoint_dir", "checkpoints/vqvae"))
        output_dir.mkdir(parents=True, exist_ok=True)
        config["output_dir"] = str(output_dir)  # Set for internal use

        if verbose:
            print("Loading dataset and features...")

        # Load features from dataset
        # Returns (features, feature_names, raw_ics, initial_info, encoder_state_dicts)
        # raw_ics and initial_info are non-None when using hybrid INITIAL encoding
        # encoder_state_dicts contains frozen encoder weights for reproducibility
        features, feature_names, raw_ics, initial_info, encoder_state_dicts = self._load_features(config)

        if verbose:
            print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
            if initial_info is not None:
                print(f"  Hybrid INITIAL: {initial_info['manual_dim']}D manual + {initial_info['cnn_dim']}D CNN (end-to-end)")

        # Clean features (remove NaN, zero-variance, duplicates, cap outliers)
        # Check both old flat format and new feature_cleaning section
        feature_cleaning = config.get("feature_cleaning", {})
        clean_enabled = feature_cleaning.get("enabled", config.get("clean_features", True))

        # Initialize preprocessing metadata for checkpoint reproducibility
        feature_mask = None
        feature_cleaning_params = None

        if clean_enabled:
            if verbose:
                print("\nCleaning features...")

            from spinlock.encoding import FeatureProcessor

            # Read parameters from feature_cleaning section (new) or fall back to root level (old)
            # Explicitly cast to proper types to handle YAML parsing
            max_var_thresh = feature_cleaning.get("max_variance_threshold", config.get("max_variance_threshold", None))
            if max_var_thresh is not None:
                max_var_thresh = float(max_var_thresh)

            # Capture cleaning parameters for reproducibility
            variance_threshold = float(feature_cleaning.get("variance_threshold", config.get("variance_threshold", 1e-10)))
            deduplicate_threshold = float(feature_cleaning.get("deduplicate_threshold", config.get("deduplicate_threshold", 0.99)))
            use_intelligent_dedup = bool(feature_cleaning.get("use_intelligent_dedup", config.get("use_intelligent_dedup", True)))
            outlier_method = str(feature_cleaning.get("outlier_method", config.get("outlier_method", "percentile")))
            percentile_range = tuple(feature_cleaning.get("percentile_range", config.get("percentile_range", [0.5, 99.5])))
            iqr_multiplier = float(feature_cleaning.get("iqr_multiplier", config.get("iqr_multiplier", 1.5)))
            mad_threshold = float(feature_cleaning.get("mad_threshold", config.get("mad_threshold", 3.0)))
            max_cv_threshold = float(feature_cleaning.get("max_cv_threshold", config.get("max_cv_threshold", 100.0)))

            feature_cleaning_params = {
                "variance_threshold": variance_threshold,
                "max_variance_threshold": max_var_thresh,
                "max_cv_threshold": max_cv_threshold,
                "deduplicate_threshold": deduplicate_threshold,
                "use_intelligent_dedup": use_intelligent_dedup,
                "outlier_method": outlier_method,
                "percentile_range": list(percentile_range),
                "iqr_multiplier": iqr_multiplier,
                "mad_threshold": mad_threshold,
            }

            processor = FeatureProcessor(
                variance_threshold=variance_threshold,
                max_variance_threshold=max_var_thresh,
                max_cv_threshold=max_cv_threshold,
                deduplicate_threshold=deduplicate_threshold,
                use_intelligent_dedup=use_intelligent_dedup,
                outlier_method=outlier_method,
                percentile_range=percentile_range,
                iqr_multiplier=iqr_multiplier,
                mad_threshold=mad_threshold,
                verbose=verbose,
            )

            # Protect INITIAL features from cleaning if hybrid mode is enabled
            if initial_info is not None:
                initial_offset = initial_info["offset"]
                initial_end = initial_offset + initial_info["manual_dim"]

                # Extract INITIAL features before cleaning
                initial_features = features[:, initial_offset:initial_end].copy()
                initial_names = feature_names[initial_offset:initial_end] if feature_names else None

                # Remove INITIAL from features array for cleaning
                features_for_cleaning = np.concatenate([
                    features[:, :initial_offset],
                    features[:, initial_end:]
                ], axis=1)
                names_for_cleaning = (
                    feature_names[:initial_offset] + feature_names[initial_end:]
                    if feature_names else None
                )

                if verbose:
                    print(f"  Protected {initial_info['manual_dim']} INITIAL features from cleaning")

                # Clean non-INITIAL features
                cleaned_features, feature_mask_non_initial, cleaned_names = processor.clean(
                    features_for_cleaning, names_for_cleaning
                )

                # Reconstruct full feature_mask including INITIAL features (all True for INITIAL)
                # feature_mask maps original 270D → final cleaned dimension
                feature_mask = np.zeros(features.shape[1], dtype=bool)
                # Mark INITIAL features as kept (they were protected)
                feature_mask[initial_offset:initial_end] = True
                # Insert non-INITIAL mask values
                non_initial_indices = list(range(initial_offset)) + list(range(initial_end, features.shape[1]))
                feature_mask[non_initial_indices] = feature_mask_non_initial

                # Count how many features before INITIAL were removed
                features_removed_before = sum(1 for i, kept in enumerate(feature_mask_non_initial) if not kept and i < initial_offset)

                # Reattach INITIAL features at the adjusted offset
                new_offset = initial_offset - features_removed_before
                features = np.concatenate([
                    cleaned_features[:, :new_offset],
                    initial_features,
                    cleaned_features[:, new_offset:]
                ], axis=1)

                # Update feature names
                if cleaned_names and initial_names:
                    feature_names = cleaned_names[:new_offset] + initial_names + cleaned_names[new_offset:]

                # Update initial_info with new offset (manual_dim stays the same)
                initial_info["offset"] = new_offset

                if verbose:
                    print(f"  INITIAL offset: {initial_offset} → {new_offset}")
            else:
                features, feature_mask, feature_names = processor.clean(features, feature_names)

            if verbose:
                print(f"After cleaning: {features.shape[0]} samples × {features.shape[1]} features")
        else:
            if verbose:
                print("Feature cleaning disabled (clean_features=false in config)")

        # Auto-discover categories (if needed)
        if config.get("category_assignment") == "auto" and config.get("resume_from") is None:
            if verbose:
                print("\nAuto-discovering feature categories via clustering...")

            group_indices = self._discover_categories(features, feature_names, config, verbose)

            if verbose:
                print(f"Discovered {len(group_indices)} categories:")
                for cat_name, indices in group_indices.items():
                    print(f"  {cat_name}: {len(indices)} features")
        elif config.get("category_mapping_file"):
            # Load categories from JSON file
            if verbose:
                print(f"\nLoading categories from {config['category_mapping_file']}...")

            group_indices = self._load_category_mapping(config["category_mapping_file"])

            if verbose:
                print(f"Loaded {len(group_indices)} categories:")
                for cat_name, indices in group_indices.items():
                    print(f"  {cat_name}: {len(indices)} features")
        elif config.get("resume_from"):
            # Load categories from checkpoint
            if verbose:
                print("\nLoading categories from checkpoint...")

            checkpoint = torch.load(config["resume_from"], weights_only=False)

            # Load from model_config (authoritative source)
            if "model_config" in checkpoint and "group_indices" in checkpoint["model_config"]:
                group_indices = checkpoint["model_config"]["group_indices"]
                if verbose:
                    print(f"Loaded {len(group_indices)} categories from model_config")
            else:
                return self.error(
                    "Checkpoint missing model_config['group_indices']. "
                    "This checkpoint may be from an old version. Please retrain."
                )
        else:
            return self.error("Must specify category_assignment='auto', category_mapping_file, or resume_from")

        # Pre-compute adaptive compression ratios if "auto" mode
        # IMPORTANT: Do this BEFORE normalization so variance information is preserved!
        if verbose:
            print(f"\nDebug: group_indices before adaptive compression:")
            for cat_name, indices in sorted(group_indices.items()):
                print(f"  {cat_name}: {len(indices)} indices → features shape: {features[:, indices].shape}")

        config = self._precompute_compression_ratios(
            config, features, group_indices, verbose
        )

        # Per-category normalization
        if verbose:
            print("\nNormalizing features per category...")

        normalized_features, normalization_stats = self._normalize_features(
            features, group_indices, config
        )

        # Save normalization stats
        stats_path = output_dir / "normalization_stats.npz"
        self._save_normalization_stats(normalization_stats, stats_path)

        if verbose:
            print(f"Saved normalization stats to {stats_path}")

        if verbose:
            print(f"\nDebug: group_indices before model building:")
            for cat_name, indices in sorted(group_indices.items()):
                print(f"  {cat_name}: {len(indices)} indices → normalized features shape: {normalized_features[:, indices].shape}")

        # Build VQ-VAE model
        if verbose:
            print("\nBuilding VQ-VAE model...")

        model, vqvae_config = self._build_model(
            normalized_features, group_indices, config, verbose, initial_info
        )

        # Create data loaders
        if verbose:
            print("\nCreating data loaders...")

        train_loader, val_loader = self._create_data_loaders(normalized_features, config, raw_ics)

        if verbose:
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Val samples:   {len(val_loader.dataset)}")

        # Create trainer
        if verbose:
            print("\nInitializing trainer...")
            if encoder_state_dicts:
                print(f"  Captured {len(encoder_state_dicts)} frozen encoder state dicts")

        trainer = self._create_trainer(
            model,
            train_loader,
            val_loader,
            config,
            group_indices,
            normalization_stats,
            feature_names,
            encoder_state_dicts,
            feature_mask,
            feature_cleaning_params,
        )

        # Load checkpoint if resuming
        if config.get("resume_from"):
            if verbose:
                print(f"\nLoading checkpoint from {config['resume_from']}...")

            checkpoint = torch.load(config["resume_from"], weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if verbose:
                print(f"Resumed from epoch {checkpoint.get('epoch', 'unknown')}")

        # Train
        if verbose:
            print("\nStarting training...\n")

        history = trainer.train(epochs=config.get("epochs", 500))

        # Save final model
        if verbose:
            print(f"\nSaving final model...")

        final_model_path = output_dir / "final_model.pt"
        self._save_final_model(
            model,
            trainer.optimizer,
            group_indices,
            normalization_stats,
            feature_names,
            config,
            history,
            final_model_path,
            encoder_state_dicts,
            feature_mask,
            feature_cleaning_params,
        )

        # Save training history
        history_path = output_dir / "training_history.json"
        self._save_training_history(history, history_path)

        # Save resolved config
        config_path = output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        elapsed = time.time() - start_time

        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            print(f"\nOutputs:")
            print(f"  Model:       {final_model_path}")
            print(f"  Stats:       {stats_path}")
            print(f"  History:     {history_path}")
            print(f"  Config:      {config_path}")

            # Print final metrics
            if "final_metrics" in history:
                print(f"\nFinal Metrics:")
                for key, val in history["final_metrics"].items():
                    if isinstance(val, float):
                        print(f"  {key}: {val:.4f}")

            print("=" * 70 + "\n")

        return 0

    def _precompute_compression_ratios(
        self,
        config: Dict[str, Any],
        features: np.ndarray,
        group_indices: Dict[str, List[int]],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Pre-compute adaptive compression ratios if config specifies "auto".

        This runs BEFORE normalization to analyze raw feature characteristics
        and compute optimal compression ratios per category.

        IMPORTANT: Features must be UN-normalized (cleaned but not standardized)
        so that variance information is preserved for adaptive computation.

        Args:
            config: Training configuration
            features: Cleaned but UN-normalized features [N, D]
            group_indices: Category → feature indices mapping
            verbose: Whether to print progress

        Returns:
            Updated config with computed compression ratios
        """
        from spinlock.encoding.latent_dim_defaults import compute_adaptive_compression_ratios

        # Check if compression_ratios is set to "auto"
        compression_ratios = config.get('compression_ratios')
        if compression_ratios != "auto":
            return config  # No auto mode, return unchanged

        if verbose:
            print("\n" + "=" * 70)
            print("ADAPTIVE COMPRESSION RATIO COMPUTATION")
            print("=" * 70)
            print(f"Analyzing {len(group_indices)} categories to compute optimal ratios...")

        # Get strategy (default: balanced)
        auto_strategy = config.get('auto_compression_strategy', 'balanced')

        # Compute adaptive ratios per category
        # IMPORTANT: Iterate in group_indices order (NOT sorted) to maintain consistency
        # with category order used in model initialization
        ratios_per_category = {}
        metrics_per_category = {}

        for cat_name in group_indices.keys():  # Preserve insertion order
            cat_indices = group_indices[cat_name]
            cat_features = features[:, cat_indices]  # [N, cat_dim]

            # Compute adaptive ratios
            ratios = compute_adaptive_compression_ratios(
                features=cat_features,
                category_name=cat_name,
                strategy=auto_strategy
            )

            ratios_per_category[cat_name] = ratios

            # Get metrics for logging
            from spinlock.encoding.latent_dim_defaults import analyze_category_characteristics
            metrics = analyze_category_characteristics(cat_features)
            metrics_per_category[cat_name] = metrics

            if verbose:
                print(f"\n  {cat_name}:")
                print(f"    Features: {len(cat_indices)}")
                print(f"    Variance:      {metrics['variance_score']:.3f}")
                print(f"    Dimensionality: {metrics['dimensionality_score']:.3f}")
                print(f"    Information:   {metrics['information_score']:.3f}")
                print(f"    Correlation:   {metrics['correlation_score']:.3f}")
                print(f"    → Ratios: {ratios} (L0={ratios[0]}, L1={ratios[1]}, L2={ratios[2]})")

        # Store computed ratios in config
        config['compression_ratios_computed'] = ratios_per_category
        config['compression_ratios_metrics'] = metrics_per_category
        config['compression_ratios_strategy'] = auto_strategy

        # Replace "auto" with per-category ratios for model initialization
        # The CategoricalVQVAEConfig will use these per-category ratios
        config['compression_ratios_per_category'] = ratios_per_category

        if verbose:
            print("\n" + "=" * 70)
            print(f"✓ Adaptive compression ratios computed using '{auto_strategy}' strategy")
            print("=" * 70 + "\n")

        return config

    def _load_features(self, config: Dict[str, Any]) -> tuple:
        """Load features from HDF5 dataset.

        Supports both single family and multi-family (concatenated) loading.
        Applies per-family encoders from config to transform raw features.

        For hybrid INITIAL encoding (encoder=initial_hybrid), also loads raw ICs
        and defers CNN encoding to training loop for end-to-end learning.

        HDF5 paths:
        - SUMMARY: /features/summary/aggregated/features [N, D]
        - TEMPORAL: /features/temporal/features [N, T, D]
        - INITIAL: /features/initial/aggregated/features [N, D]
        - Raw ICs: /inputs/fields [N, M, H, W] or [N, M, C, H, W]

        Returns:
            Tuple of (features, feature_names, raw_ics, initial_info, encoder_state_dicts)
            raw_ics is None if no hybrid INITIAL encoding
            initial_info contains offset/count for hybrid encoder
            encoder_state_dicts contains state dicts for all frozen encoders used
        """
        import h5py
        import numpy as np
        import torch
        from spinlock.encoding.encoders import get_encoder

        # Extract configuration (assumes new nested format, flattened above)
        dataset_path = config.get("dataset_path")
        feature_type = config.get("feature_type", "aggregated")
        max_samples = config.get("max_samples")
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Extract family configs
        families_config = config["families"]
        feature_families = list(families_config.keys())

        all_features = []
        all_feature_names = []
        raw_ics = None
        initial_info = None
        encoder_state_dicts = {}  # Store state dicts for frozen encoders

        with h5py.File(dataset_path, "r") as f:
            for family_idx, feature_family in enumerate(feature_families):
                family_config = families_config[feature_family]
                encoder_name = family_config.get("encoder")

                # Handle hybrid INITIAL encoding specially
                # For initial_hybrid: load raw ICs and defer CNN encoding to training loop
                if encoder_name in ["initial_hybrid", "InitialHybridEncoder"]:
                    # Load manual features (14D)
                    features_path = f"/features/{feature_family}/{feature_type}"
                    if features_path not in f:
                        raise ValueError(
                            f"Manual INITIAL features not found at {features_path}. "
                            f"Run: poetry run python scripts/dev/extract_initial_features.py --dataset <path>"
                        )

                    group = f[features_path]
                    if max_samples is not None and max_samples > 0:
                        family_features = np.array(group["features"][:max_samples])
                    else:
                        family_features = np.array(group["features"])

                    # Replace NaN with 0
                    nan_count = np.isnan(family_features).sum()
                    if nan_count > 0:
                        family_features = np.nan_to_num(family_features, nan=0.0)
                        print(f"  {feature_family}: Replaced {nan_count} NaN values with 0")

                    # Load raw ICs for CNN encoding
                    if "inputs/fields" not in f:
                        raise ValueError(
                            "Raw ICs not found at /inputs/fields. "
                            "Required for hybrid INITIAL encoding."
                        )

                    if max_samples is not None and max_samples > 0:
                        raw_ics_data = np.array(f["inputs/fields"][:max_samples])
                    else:
                        raw_ics_data = np.array(f["inputs/fields"])

                    # Handle different IC shapes:
                    # [N, M, H, W] -> add channel dim -> [N, M, 1, H, W]
                    # [N, M, C, H, W] -> use as is
                    if len(raw_ics_data.shape) == 4:
                        raw_ics_data = raw_ics_data[:, :, np.newaxis, :, :]

                    # Average across realizations for single IC per operator
                    # [N, M, C, H, W] -> [N, C, H, W]
                    raw_ics = raw_ics_data.mean(axis=1)

                    # Store info for hybrid encoder
                    initial_offset = sum(arr.shape[1] for arr in all_features)
                    initial_info = {
                        "offset": initial_offset,
                        "manual_dim": family_features.shape[1],
                        "cnn_dim": family_config.get("encoder_params", {}).get("cnn_embedding_dim", 28),
                        "in_channels": raw_ics.shape[1],
                    }

                    print(f"  {feature_family}: Hybrid mode - {family_features.shape[1]}D manual + raw ICs {raw_ics.shape}")
                    print(f"    CNN will be trained end-to-end (embedding_dim={initial_info['cnn_dim']})")

                    # Don't apply encoder - just use manual features
                    output_dim = family_features.shape[1]
                    family_names = [f"{feature_family}_{i}" for i in range(output_dim)]

                    if len(feature_families) > 1:
                        family_names = [f"{feature_family}::{name}" for name in family_names]

                    all_features.append(family_features)
                    all_feature_names.extend(family_names)
                    continue

                # Standard feature loading (non-hybrid)
                # Determine correct HDF5 path based on family type
                # TEMPORAL uses /features/temporal directly (no aggregated sublevel)
                # SUMMARY uses /features/summary/aggregated
                if feature_family == "temporal":
                    features_path = f"/features/{feature_family}"
                else:
                    features_path = f"/features/{feature_family}/{feature_type}"

                if features_path not in f:
                    raise ValueError(
                        f"Features not found at {features_path}. "
                        f"Available families: {list(f['/features'].keys())}"
                    )

                group = f[features_path]

                # Load features (with optional sample limit)
                if max_samples is not None and max_samples > 0:
                    family_features = np.array(group["features"][:max_samples])
                else:
                    family_features = np.array(group["features"])

                # Replace NaN with 0 before encoding (encoders can't handle NaN)
                nan_count = np.isnan(family_features).sum()
                if nan_count > 0:
                    family_features = np.nan_to_num(family_features, nan=0.0)
                    print(f"  {feature_family}: Replaced {nan_count} NaN values with 0")

                # Apply per-family encoder if configured
                encoder_params = family_config.get("encoder_params", {})

                if encoder_name and encoder_name not in ["identity", "IdentityEncoder"]:
                    # Get input dimension for encoder
                    if len(family_features.shape) == 3:
                        # Temporal: [N, T, D] -> input_dim is D
                        input_dim = family_features.shape[2]
                    else:
                        # 2D: [N, D] -> input_dim is D
                        input_dim = family_features.shape[1]

                    # Create encoder with input_dim
                    encoder = get_encoder(encoder_name, input_dim=input_dim, **encoder_params)
                    encoder = encoder.to(device)
                    encoder.eval()

                    # Save encoder state dict for checkpoint reproducibility
                    encoder_state_dicts[feature_family] = {
                        'encoder_name': encoder_name,
                        'encoder_params': encoder_params,
                        'state_dict': encoder.state_dict(),
                    }

                    # Apply encoder in batches
                    batch_size = 1024
                    encoded_features = []
                    features_tensor = torch.tensor(family_features, dtype=torch.float32)

                    with torch.no_grad():
                        for i in range(0, len(features_tensor), batch_size):
                            batch = features_tensor[i:i+batch_size].to(device)
                            encoded = encoder(batch)
                            encoded_features.append(encoded.cpu().numpy())

                    family_features = np.concatenate(encoded_features, axis=0)
                    print(f"  {feature_family}: Applied {encoder_name} -> shape {family_features.shape}")

                # Generate feature names
                if len(family_features.shape) == 2:
                    output_dim = family_features.shape[1]
                    family_names = [f"{feature_family}_{i}" for i in range(output_dim)]
                else:
                    # Should not happen after encoder, but handle gracefully
                    raise ValueError(
                        f"Features for {feature_family} are {len(family_features.shape)}D after encoding. "
                        f"Expected 2D. Shape: {family_features.shape}"
                    )

                # Prefix feature names with family name (for multi-family)
                if len(feature_families) > 1:
                    family_names = [f"{feature_family}::{name}" for name in family_names]

                all_features.append(family_features)
                all_feature_names.extend(family_names)

        # Concatenate features if multiple families
        if len(all_features) > 1:
            features = np.concatenate(all_features, axis=1)
        else:
            features = all_features[0]

        return features, all_feature_names, raw_ics, initial_info, encoder_state_dicts

    def _load_category_mapping(self, mapping_file: Path) -> Dict[str, list]:
        """Load category mapping from JSON file."""
        import json

        mapping_path = Path(mapping_file)

        if not mapping_path.exists():
            raise ValueError(f"Category mapping file not found: {mapping_path}")

        with open(mapping_path, "r") as f:
            category_mapping = json.load(f)

        # Validate structure: {category_name: [indices]}
        if not isinstance(category_mapping, dict):
            raise ValueError("Category mapping must be a dictionary")

        for cat_name, indices in category_mapping.items():
            if not isinstance(indices, list):
                raise ValueError(f"Category '{cat_name}' indices must be a list")

            if not all(isinstance(idx, int) for idx in indices):
                raise ValueError(f"Category '{cat_name}' must contain integer indices")

        return category_mapping

    def _discover_categories(
        self, features: np.ndarray, feature_names: list, config: Dict[str, Any], verbose: bool
    ) -> Dict[str, list]:
        """Auto-discover feature categories via clustering and/or gradient refinement.

        Supports three methods:
        - 'clustering': Hierarchical clustering only (fast, approximate)
        - 'gradient': Gumbel-Softmax gradient optimization only
        - 'hybrid': Clustering init + gradient refinement (recommended for best orthogonality)
        """
        from spinlock.encoding import DynamicCategoryAssignment, standard_normalize

        # Get category assignment config
        cat_config = config.get("category_assignment_config", {})
        method = cat_config.get("method", "clustering")

        if verbose:
            print(f"  Method: {method}")

        # Normalize features for clustering (equal importance)
        normalized = standard_normalize(features)

        # Create assignment strategy
        assigner = DynamicCategoryAssignment(
            num_categories=config.get("num_categories_auto"),
            method=method,
            orthogonality_target=config.get("orthogonality_target", 0.15),
            min_features_per_category=config.get("min_features_per_category", 3),
            max_clusters=config.get("max_clusters", 25),
            random_seed=config.get("random_seed", 42),
            # Gradient refinement parameters
            gradient_epochs=cat_config.get("gradient_epochs", 500),
            gradient_lr=cat_config.get("gradient_lr", 0.01),
            subsample_excess_fraction=cat_config.get("subsample_excess_fraction", 0.1),
            device=cat_config.get("device", "cuda"),
            # Family isolation - place specified families in dedicated categories
            isolated_families=cat_config.get("isolated_families"),
            # Orphan reassignment - guarantee 100% feature assignment
            reassign_orphans=cat_config.get("reassign_orphans", False),
        )

        # Compute assignments
        group_indices = assigner.assign_categories(feature_names, normalized)

        return group_indices

    def _normalize_features(
        self, features: np.ndarray, group_indices: Dict[str, list], config: Dict[str, Any]
    ) -> tuple:
        """Normalize features per category."""
        from spinlock.encoding import (
            compute_normalization_stats,
            compute_robust_normalization_stats,
            apply_standard_normalization,
            apply_robust_normalization,
        )
        import numpy as np

        # Get normalization method from config (default: standard)
        normalization_method = config.get("normalization_method", "standard")

        normalized = features.copy()
        stats_dict = {}

        for category, indices in group_indices.items():
            cat_features = features[:, indices]

            if normalization_method == "mad":
                # Use robust MAD-based normalization
                stats = compute_robust_normalization_stats(cat_features)
                normalized[:, indices] = apply_robust_normalization(cat_features, stats)
            else:
                # Use standard mean/std normalization
                stats = compute_normalization_stats(cat_features)
                normalized[:, indices] = apply_standard_normalization(cat_features, stats)

            stats_dict[category] = stats

        return normalized, stats_dict

    def _save_normalization_stats(self, stats_dict: Dict[str, Any], path: Path) -> None:
        """Save normalization stats to .npz file."""
        from spinlock.encoding import NormalizationStats, RobustNormalizationStats
        import numpy as np

        # Convert stats objects to numpy arrays
        save_dict = {}
        for category, stats in stats_dict.items():
            if isinstance(stats, RobustNormalizationStats):
                save_dict[f"{category}_median"] = stats.median
                save_dict[f"{category}_mad"] = stats.mad
            else:  # NormalizationStats
                save_dict[f"{category}_mean"] = stats.mean
                save_dict[f"{category}_std"] = stats.std

        # Add type marker to indicate which normalization method was used
        if stats_dict:
            first_stats = next(iter(stats_dict.values()))
            if isinstance(first_stats, RobustNormalizationStats):
                save_dict["_normalization_method"] = np.array(["mad"])
            else:
                save_dict["_normalization_method"] = np.array(["standard"])

        np.savez(path, **save_dict)

    def _build_model(
        self,
        features: np.ndarray,
        group_indices: Dict[str, list],
        config: Dict[str, Any],
        verbose: bool,
        initial_info: Optional[Dict[str, Any]] = None,
    ):
        """Build VQ-VAE model.

        If initial_info is provided, uses VQVAEWithInitial wrapper for
        end-to-end INITIAL CNN training.
        """
        from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
        from spinlock.encoding.latent_dim_defaults import parse_compression_ratios
        import torch

        # Get factors from config - empty list triggers autoscaling (convert to None)
        factors = config.get("factors")
        if factors is not None and len(factors) == 0:
            factors = None  # Empty list = autoscaling

        # Handle compression_ratios: can be string, list, or per-category dict
        compression_ratios_str = config.get("compression_ratios")
        compression_ratios_per_category = config.get("compression_ratios_per_category")
        compression_ratios = None

        if compression_ratios_per_category is not None:
            # Auto mode was used - we have per-category ratios
            # Convert to the format expected by CategoricalVQVAEConfig
            # We'll pass the per-category dict directly as 'levels' with compression info
            if verbose:
                print("Using adaptive per-category compression ratios")

            # Build per-category levels with computed compression ratios
            levels_dict = {}
            for cat_name, ratios in compression_ratios_per_category.items():
                # Create empty level configs - latent_dims will be auto-computed
                # using these compression ratios
                levels_dict[cat_name] = [
                    {'num_tokens': None, 'latent_dim': None},
                    {'num_tokens': None, 'latent_dim': None},
                    {'num_tokens': None, 'latent_dim': None},
                ]

            # Use the per-category ratios dict
            # We'll need to handle this in the model initialization
            factors = levels_dict
            compression_ratios = compression_ratios_per_category
        elif compression_ratios_str is not None:
            # Check if auto mode
            if compression_ratios_str.lower() == "auto":
                compression_ratios = "auto"  # Pass "auto" string to model
            else:
                # Parse compression_ratios from config (e.g., "0.5:1:1.5" → [0.5, 1.0, 1.5])
                compression_ratios = parse_compression_ratios(compression_ratios_str)

        vqvae_config = CategoricalVQVAEConfig(
            input_dim=features.shape[1],
            group_indices=group_indices,
            group_embedding_dim=config.get("group_embedding_dim", 64),
            group_hidden_dim=config.get("group_hidden_dim", 128),
            levels=factors,  # None = auto-compute, or dict with per-category levels
            commitment_cost=config.get("commitment_cost", 0.45),
            use_ema=config.get("use_ema", True),
            decay=config.get("ema_decay", 0.99),  # Config uses "ema_decay", model uses "decay"
            compression_ratios=compression_ratios,  # Can be list or dict
            uniform_codebook_init=config.get("uniform_codebook_init", False),
        )

        # Build model - use wrapper for hybrid INITIAL encoding
        if initial_info is not None:
            from spinlock.encoding.vqvae_with_initial import VQVAEWithInitial

            model = VQVAEWithInitial(
                vqvae_config=vqvae_config,
                initial_manual_dim=initial_info["manual_dim"],
                initial_cnn_dim=initial_info["cnn_dim"],
                initial_feature_offset=initial_info["offset"],
                initial_feature_count=initial_info["manual_dim"],
                in_channels=initial_info["in_channels"],
            )
            if verbose:
                print("Using VQVAEWithInitial for end-to-end INITIAL CNN training")
        else:
            model = CategoricalHierarchicalVQVAE(vqvae_config)

        if verbose:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model created:")
            print(f"  Total parameters:     {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")

        return model, vqvae_config

    def _create_data_loaders(
        self,
        features: np.ndarray,
        config: Dict[str, Any],
        raw_ics: Optional[np.ndarray] = None,
    ):
        """Create train/val data loaders.

        Args:
            features: Pre-encoded features [N, D]
            config: Training configuration
            raw_ics: Optional raw initial conditions [N, C, H, W] for hybrid INITIAL
        """
        import torch
        from torch.utils.data import Dataset, DataLoader
        import numpy as np

        # Dataset that optionally includes raw ICs
        class FeatureDataset(Dataset):
            def __init__(self, features, raw_ics=None):
                self.features = torch.from_numpy(features).float()
                self.raw_ics = None
                if raw_ics is not None:
                    self.raw_ics = torch.from_numpy(raw_ics).float()

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                item = {"features": self.features[idx]}
                if self.raw_ics is not None:
                    item["raw_ics"] = self.raw_ics[idx]
                return item

        # Split into train/val (90/10)
        n_samples = len(features)
        n_train = int(0.9 * n_samples)

        # Shuffle
        rng = np.random.RandomState(config.get("random_seed", 42))
        indices = rng.permutation(n_samples)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_raw_ics = raw_ics[train_indices] if raw_ics is not None else None
        val_raw_ics = raw_ics[val_indices] if raw_ics is not None else None

        train_dataset = FeatureDataset(features[train_indices], train_raw_ics)
        val_dataset = FeatureDataset(features[val_indices], val_raw_ics)

        batch_size = config.get("batch_size", 512)
        val_batch_size = config.get("val_batch_size", batch_size)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        return train_loader, val_loader

    def _create_trainer(
        self,
        model,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        group_indices: Dict[str, List[int]],
        normalization_stats: Dict[str, Any],
        feature_names: List[str],
        encoder_state_dicts: Optional[Dict[str, Any]] = None,
        feature_mask: Optional[np.ndarray] = None,
        feature_cleaning_params: Optional[Dict[str, Any]] = None,
    ):
        """Create VQVAETrainer with full metadata for checkpoint reproducibility."""
        from spinlock.encoding.training import VQVAETrainer
        from pathlib import Path

        checkpoint_dir = config.get("checkpoint_dir")
        if checkpoint_dir is None and config.get("output_dir"):
            checkpoint_dir = Path(config["output_dir"]) / "checkpoints"

        trainer = VQVAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config.get("learning_rate", 1e-3),
            device=config.get("device", "cuda"),
            orthogonality_weight=config.get("orthogonality_weight", 0.1),
            informativeness_weight=config.get("informativeness_weight", 0.1),
            topo_weight=config.get("topo_weight", 0.02),
            topo_samples=config.get("topo_samples", 64),
            early_stopping_patience=config.get("early_stopping_patience", 100),
            early_stopping_min_delta=config.get("early_stopping_min_delta", 0.01),
            dead_code_reset_interval=config.get("dead_code_reset_interval", 100),
            dead_code_threshold=config.get("dead_code_threshold", 10.0),
            dead_code_max_reset_fraction=config.get("dead_code_max_reset_fraction", 0.25),
            use_smart_reset=config.get("use_smart_reset", False),
            checkpoint_dir=checkpoint_dir,
            use_torch_compile=config.get("use_torch_compile", True),
            val_every_n_epochs=config.get("val_every_n_epochs", 5),
            verbose=config.get("verbose", True),
            # Metadata for checkpoint reproducibility
            config=config,
            group_indices=group_indices,
            normalization_stats=normalization_stats,
            feature_names=feature_names,
            encoder_state_dicts=encoder_state_dicts,
            feature_mask=feature_mask,
            feature_cleaning_params=feature_cleaning_params,
        )

        return trainer

    def _save_final_model(
        self,
        model,
        optimizer,
        group_indices,
        normalization_stats,
        feature_names,
        config,
        history,
        path: Path,
        encoder_state_dicts: Optional[Dict[str, Any]] = None,
        feature_mask: Optional[np.ndarray] = None,
        feature_cleaning_params: Optional[Dict[str, Any]] = None,
    ):
        """Save final model checkpoint with full metadata for reproducibility.

        Args:
            feature_mask: Boolean array indicating which features survived cleaning.
                         This allows diagnostics to reproduce the exact feature space.
            feature_cleaning_params: Exact parameters used for feature cleaning
                                    (variance thresholds, dedup params, etc.)
        """
        import torch
        import random

        # Store complete checkpoint with full config for reproducibility
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # DEPRECATED: Use model_config['group_indices'] instead (this is pre-model-init version)
            "pre_model_group_indices": group_indices,
            "normalization_stats": normalization_stats,
            "feature_names": feature_names,
            "config": config,  # Full raw config (includes families, adaptive ratios, etc.)
            "history": history,
        }

        # Add encoder state dicts if available
        if encoder_state_dicts is not None:
            checkpoint["encoder_state_dicts"] = encoder_state_dicts

        # Add feature preprocessing metadata for reproducibility
        if feature_mask is not None:
            checkpoint["feature_mask"] = feature_mask
        if feature_cleaning_params is not None:
            checkpoint["feature_cleaning_params"] = feature_cleaning_params

        # Capture PRNG states for full reproducibility
        checkpoint["prng_states"] = {
            "random_seed": config.get("random_seed", 42),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        if torch.cuda.is_available():
            checkpoint["prng_states"]["cuda_rng_state"] = torch.cuda.get_rng_state()

        # Also store VQ-VAE model config for easy loading
        if hasattr(model, "config"):
            # Store the actual VQVAEConfig object's key attributes
            # This is redundant with config['model'] but makes loading easier
            config_dict = {
                "input_dim": model.config.input_dim,
                "group_indices": model.config.group_indices,
                "group_embedding_dim": model.config.group_embedding_dim,
                "group_hidden_dim": model.config.group_hidden_dim,
            }

            # Add per-category levels (important for reconstruction)
            if hasattr(model.config, "levels") and model.config.levels:
                config_dict["levels"] = model.config.levels

            checkpoint["model_config"] = config_dict

        torch.save(checkpoint, path)

    def _save_training_history(self, history: dict, path: Path):
        """Save training history to JSON."""
        import json
        import numpy as np

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_history = convert_to_serializable(history)

        with open(path, "w") as f:
            json.dump(serializable_history, f, indent=2)
