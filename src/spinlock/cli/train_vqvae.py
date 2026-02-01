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

  # Train with feature selection (pruning)
  spinlock train-vqvae \
      --config configs/vqvae/production.json \
      --feature-selection datasets/cno_50k_feature_analysis.json

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

        # Data options
        data_group = parser.add_argument_group("data options")

        data_group.add_argument(
            "--feature-selection",
            type=Path,
            metavar="PATH",
            help="Path to feature analysis file for pruning (from analyze-features command)",
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

        # Adaptive tuning options
        tuning_group = parser.add_argument_group("adaptive tuning options")

        tuning_group.add_argument(
            "--enable-adaptive-tuning",
            action="store_true",
            help="Enable upfront adaptive MSE tuning phase before full training",
        )

        tuning_group.add_argument(
            "--tuning-epochs",
            type=int,
            default=50,
            metavar="N",
            help="Epochs per category during tuning phase (default: 50)",
        )

        tuning_group.add_argument(
            "--tuning-samples",
            type=int,
            default=10000,
            metavar="N",
            help="Samples to use for tuning phase (default: 10000)",
        )

        tuning_group.add_argument(
            "--max-sweeps",
            type=int,
            default=3,
            metavar="N",
            help="Maximum categories to run compression sweeps for (default: 3)",
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
            return self._run_training(config, args)
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            import traceback
            print(f"\nError during training: {e}", file=sys.stderr)
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
        if hasattr(args, 'feature_selection') and args.feature_selection:
            config["feature_selection"] = str(args.feature_selection)

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

            # Handle encoder-specific output dimension names
            if encoder == "initial_hybrid":
                output = params.get("cnn_embedding_dim", 64)
            elif encoder == "TemporalCNNEncoder":
                output = params.get("embedding_dim", 128)
            else:
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

    def _run_training(self, config: Dict[str, Any], args: Namespace) -> int:
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

        verbose = args.verbose
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

        # DEBUG: Initial memory baseline
        import resource
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB on Linux
        print(f"[MEMORY DEBUG] Baseline at start: {mem_usage:.1f} MB")

        # Load features from dataset
        # Returns (features, feature_names, raw_ics, initial_info, encoder_state_dicts)
        # raw_ics and initial_info are non-None when using hybrid INITIAL encoding
        # encoder_state_dicts contains frozen encoder weights for reproducibility
        # reference_features and is_interpolated are for reference feature regularization (optional)
        features, feature_names, raw_ics, initial_info, encoder_state_dicts, reference_features, is_interpolated, raw_summary = self._load_features(config)

        # Check for incomplete datasets (allocated but not fully generated)
        # Use multiple strategies to detect ungenerated samples:
        # 1. Check if features are all zeros
        # 2. Check if raw_ics are all zeros (more reliable for hybrid mode)
        # 3. Check if feature norm is extremely small (< 1e-6)

        allocated_samples = len(features)

        # Strategy 1: Simple zero check on features
        feature_nonzero_mask = np.any(features != 0, axis=1)

        # Strategy 2: Check raw_ics if available (most reliable for incomplete data)
        if raw_ics is not None and len(raw_ics.shape) > 1:
            # For raw_ics with shape [N, ...], check if all values are zero
            ics_axes = tuple(range(1, len(raw_ics.shape)))
            ics_nonzero_mask = np.any(raw_ics != 0, axis=ics_axes)
            if verbose:
                print(f"Debug: Incomplete detection strategies:")
                print(f"  Features non-zero: {feature_nonzero_mask.sum()}/{allocated_samples}")
                print(f"  Raw ICs non-zero:  {ics_nonzero_mask.sum()}/{allocated_samples}")
            # Use raw_ics as ground truth if available
            non_zero_mask = ics_nonzero_mask
        else:
            # Strategy 3: Check feature norm threshold (more robust than exact zero)
            feature_norms = np.linalg.norm(features, axis=1)
            norm_threshold = 1e-6
            norm_nonzero_mask = feature_norms > norm_threshold

            if verbose:
                print(f"Debug: Incomplete detection strategies:")
                print(f"  Features non-zero (exact): {feature_nonzero_mask.sum()}/{allocated_samples}")
                print(f"  Features non-zero (norm>{norm_threshold}): {norm_nonzero_mask.sum()}/{allocated_samples}")

            # Use norm-based detection (more robust)
            non_zero_mask = norm_nonzero_mask

        actual_samples = non_zero_mask.sum()

        if actual_samples < allocated_samples:
            if verbose:
                print(f"\n⚠️  Incomplete dataset detected:")
                print(f"   Allocated: {allocated_samples} samples")
                print(f"   Generated: {actual_samples} samples")
                print(f"   Truncating to valid samples only\n")

            # Truncate all data to valid samples
            features = features[non_zero_mask]
            if raw_ics is not None:
                raw_ics = raw_ics[non_zero_mask]
            if reference_features is not None:
                reference_features = reference_features[non_zero_mask]
            if is_interpolated is not None:
                is_interpolated = is_interpolated[non_zero_mask]
            if raw_summary is not None:
                raw_summary = raw_summary[non_zero_mask]
        elif verbose:
            print(f"Debug: No incomplete dataset detected (all {allocated_samples} samples appear valid)")

        # Store reference features, interpolation mask, and raw SUMMARY as instance variables for data loaders
        self._reference_features = reference_features
        self._is_interpolated = is_interpolated
        self._raw_summary = raw_summary

        if verbose:
            print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
            if initial_info is not None:
                print(f"  Hybrid INITIAL: {initial_info['manual_dim']}D manual + {initial_info['cnn_dim']}D CNN (end-to-end)")

        # Compute per-family normalization stats for meta-operator training
        # Dynamically determine family slices based on actual loaded families
        # These stats will be saved to checkpoint for UnifiedFeaturePipeline
        if verbose:
            print("\nComputing per-family normalization statistics...")

        import torch
        per_family_stats = {}

        # Build family slices dynamically from feature_names
        # feature_names have format "family::feature_name" or just "feature_name"
        family_slices = {}
        current_idx = 0

        for i, name in enumerate(feature_names):
            if '::' in name:
                family = name.split('::')[0]
            else:
                # Infer family from position (fallback for non-prefixed names)
                # This shouldn't happen with current code but handle gracefully
                if current_idx == 0:
                    family = 'initial'
                else:
                    family = 'temporal'

            if family not in family_slices:
                family_slices[family] = {'start': i, 'end': i + 1}
            else:
                family_slices[family]['end'] = i + 1

        # Compute stats for each family
        for family, slice_info in family_slices.items():
            start, end = slice_info['start'], slice_info['end']
            family_features = features[:, start:end]

            per_family_stats[family] = (
                torch.from_numpy(family_features.mean(axis=0)).float(),
                torch.from_numpy(family_features.std(axis=0)).float()
            )

            if verbose:
                means, stds = per_family_stats[family]
                print(f"  {family.upper()} ({len(means)}D): mean=[{means.min():.3f}, {means.max():.3f}], std=[{stds.min():.3f}, {stds.max():.3f}]")

        # Clean features (remove NaN, zero-variance, duplicates, cap outliers)
        # Check both old flat format and new feature_cleaning section
        feature_cleaning = config.get("feature_cleaning", {})
        clean_enabled = feature_cleaning.get("enabled", config.get("clean_features", True))

        # CRITICAL: Control whether cleaning happens before or after category discovery
        # Default: false (clean AFTER discovery to preserve semantic diversity)
        pre_categorization = feature_cleaning.get("pre_categorization", False)

        # DEBUG: Print cleaning configuration
        print(f"\n[DEBUG] Feature cleaning config:")
        print(f"  enabled: {clean_enabled}")
        print(f"  pre_categorization: {pre_categorization}")
        print(f"  verbose: {verbose}")
        print(f"  Will clean: {'BEFORE' if (clean_enabled and pre_categorization) else 'AFTER' if clean_enabled else 'NEVER'} category discovery")

        # Initialize preprocessing metadata for checkpoint reproducibility
        feature_mask = None
        feature_cleaning_params = None
        self._feature_mask = None  # Store for later use in _create_data_loaders

        if clean_enabled and pre_categorization:
            print("\nCleaning features (pre-categorization)...")

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
                # Use encoded_dim (hybrid encoded 166D) not manual_dim (raw 38D)
                initial_dim = initial_info["encoded_dim"]
                initial_end = initial_offset + initial_dim

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

                print(f"  Protected {initial_dim}D INITIAL features (hybrid encoded) from cleaning")

                # Clean non-INITIAL features
                cleaned_features, feature_mask_non_initial, cleaned_names = processor.clean(
                    features_for_cleaning, names_for_cleaning
                )

                # Reconstruct full feature_mask including INITIAL features (all True for INITIAL)
                # feature_mask maps original dimension → final cleaned dimension
                feature_mask = np.zeros(features.shape[1], dtype=bool)
                # Mark INITIAL features as kept (they were protected from cleaning)
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

                # Update initial_info with new offset (manual_dim and encoded_dim stay the same)
                initial_info["offset"] = new_offset

                print(f"  INITIAL offset adjusted: {initial_offset} → {new_offset}")

                # Store feature_mask for later use
                self._feature_mask = feature_mask

                # DEBUG: Check for duplicate feature names after cleaning
                if feature_names:
                    from collections import Counter
                    name_counts = Counter(feature_names)
                    duplicates = {name: count for name, count in name_counts.items() if count > 1}
                    if duplicates:
                        print(f"\n[ERROR] Duplicate feature names after pre-categorization cleaning!")
                        print(f"  Total duplicates: {len(duplicates)}")
                        for name, count in list(duplicates.items())[:10]:
                            print(f"    {name}: {count} occurrences")
                        raise ValueError(f"Duplicate feature names found: {len(duplicates)} names have duplicates")
                    else:
                        print(f"  Feature names OK: {len(feature_names)} unique names")
                        print(f"  First 5 feature names: {feature_names[:5]}")
                        print(f"  Last 5 feature names: {feature_names[-5:]}")
                        # Count features per family
                        from collections import Counter
                        families = [name.split('::')[0] if '::' in name else 'unknown' for name in feature_names]
                        family_counts = Counter(families)
                        print(f"  Features per family: {dict(family_counts)}")
            else:
                features, feature_mask, feature_names = processor.clean(features, feature_names)

                # Store feature_mask for later use
                self._feature_mask = feature_mask

            if verbose:
                print(f"After cleaning: {features.shape[0]} samples × {features.shape[1]} features")

                # DEBUG: Check for duplicate feature names after cleaning
                if feature_names:
                    from collections import Counter
                    name_counts = Counter(feature_names)
                    duplicates = {name: count for name, count in name_counts.items() if count > 1}
                    if duplicates:
                        print(f"\n[WARNING] Duplicate feature names after pre-categorization cleaning:")
                        for name, count in list(duplicates.items())[:10]:
                            print(f"  {name}: {count} occurrences")
                        print(f"  Total duplicate names: {len(duplicates)}")
                    else:
                        print(f"[DEBUG] No duplicate feature names after cleaning ({len(feature_names)} unique)")
        elif not clean_enabled:
            if verbose:
                print("Feature cleaning disabled (clean_features=false in config)")
        else:
            # Cleaning deferred until after category discovery
            print("\nFeature cleaning deferred until after categorization")
            print("  This preserves semantic diversity for clustering")

        # NEW: Redundancy-based feature pruning (correlation analysis)
        pruning_enabled = feature_cleaning.get("enable_pruning", False)
        pruning_analysis = None
        pruned_mask = None

        if pruning_enabled and pre_categorization:
            if verbose:
                print("\nAnalyzing feature redundancy for pruning...")

            from datetime import datetime
            from spinlock.features.analyzer import FeatureAnalyzer, AnalysisMetadata

            # Get pruning parameters
            pruning_corr_thresh = float(feature_cleaning.get("pruning_correlation_threshold", 0.99))
            pruning_var_thresh = float(feature_cleaning.get("pruning_variance_threshold", 1e-8))
            pruning_nan_thresh = float(feature_cleaning.get("pruning_nan_threshold", 0.01))

            # Check for cached analysis
            cache_enabled = feature_cleaning.get("cache_pruning_analysis", True)
            cache_path = feature_cleaning.get("pruning_cache_path")

            if cache_path is None:
                dataset_stem = Path(config["dataset_path"]).stem
                cache_path = Path(config["dataset_path"]).parent / f"{dataset_stem}_pruning_analysis.json"
            else:
                cache_path = Path(cache_path)

            # Load or generate analysis
            if cache_enabled and cache_path.exists():
                if verbose:
                    print(f"  Loading cached analysis from: {cache_path}")
                import json
                try:
                    with open(cache_path, 'r') as f:
                        pruning_analysis = json.load(f)

                    # Verify cache is compatible
                    cached_thresholds = pruning_analysis["metadata"]["thresholds"]
                    if (abs(cached_thresholds["correlation"] - pruning_corr_thresh) > 1e-9 or
                        abs(cached_thresholds["variance"] - pruning_var_thresh) > 1e-9 or
                        abs(cached_thresholds["nan"] - pruning_nan_thresh) > 1e-9):
                        if verbose:
                            print("  Cache has different thresholds, re-running analysis...")
                        pruning_analysis = None
                except Exception as e:
                    if verbose:
                        print(f"  Failed to load cache: {e}")
                        print("  Re-running analysis...")
                    pruning_analysis = None

            # Run analysis if not cached
            if pruning_analysis is None:
                analyzer = FeatureAnalyzer(
                    variance_threshold=pruning_var_thresh,
                    correlation_threshold=pruning_corr_thresh,
                    nan_threshold=pruning_nan_thresh,
                )

                # Create metadata
                metadata = AnalysisMetadata(
                    dataset_path=str(config["dataset_path"]),
                    feature_family="temporal",  # Could be detected from config
                    analysis_timestamp=datetime.now().isoformat(),
                    total_samples=features.shape[0],
                    total_timesteps=None,
                    total_features=features.shape[1],
                    thresholds={
                        "variance": pruning_var_thresh,
                        "correlation": pruning_corr_thresh,
                        "nan": pruning_nan_thresh,
                    }
                )

                if verbose:
                    print(f"  Running correlation analysis on {features.shape[1]} features...")
                    print(f"    Correlation threshold: {pruning_corr_thresh}")
                    print(f"    Variance threshold: {pruning_var_thresh}")
                    print(f"    NaN threshold: {pruning_nan_thresh}")

                pruning_analysis = analyzer.analyze(features, feature_names, metadata)

                # Cache results
                if cache_enabled:
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, 'w') as f:
                            json.dump(pruning_analysis, f, indent=2, default=str)
                        if verbose:
                            print(f"  Cached analysis to: {cache_path}")
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Failed to cache analysis: {e}")

            # Apply pruning
            prune_indices = pruning_analysis["recommendations"]["prune_indices"]
            keep_indices = pruning_analysis["recommendations"]["keep_indices"]

            if len(prune_indices) > 0:
                summary = pruning_analysis["summary"]
                if verbose:
                    print(f"\n  Pruning summary:")
                    print(f"    Total features: {summary['features_analyzed']}")
                    print(f"    Zero variance: {summary['zero_variance_features']}")
                    print(f"    Redundant (r>{pruning_corr_thresh}): {summary['redundant_features']}")
                    print(f"    High NaN: {summary['high_nan_features']}")
                    print(f"    → Keeping: {summary['recommended_keep']} ({summary['compression_ratio']:.1%})")
                    print(f"    → Pruning: {summary['recommended_prune']} ({100*(1-summary['compression_ratio']):.1%})")

                # Apply pruning
                features = features[:, keep_indices]

                # Update feature names
                if feature_names:
                    feature_names = [feature_names[i] for i in keep_indices]

                # Create pruned mask for checkpoint
                pruned_mask = np.zeros(summary['features_analyzed'], dtype=bool)
                pruned_mask[prune_indices] = True

                # Update feature_cleaning_params for checkpoint
                if feature_cleaning_params is None:
                    feature_cleaning_params = {}
                feature_cleaning_params["pruning_enabled"] = True
                feature_cleaning_params["pruning_correlation_threshold"] = pruning_corr_thresh
                feature_cleaning_params["pruning_variance_threshold"] = pruning_var_thresh
                feature_cleaning_params["pruning_nan_threshold"] = pruning_nan_thresh
                feature_cleaning_params["pruned_feature_count"] = len(prune_indices)

                if verbose:
                    print(f"  Features after pruning: {features.shape}")
            else:
                if verbose:
                    print("  No features pruned (all pass thresholds)")

        # Auto-discover categories (if needed)
        if config.get("category_assignment") == "auto" and config.get("resume_from") is None:
            if verbose:
                print("\nAuto-discovering feature categories via clustering...")

            # DEBUG: Memory before category discovery
            import resource
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[MEMORY DEBUG] Before category discovery: {mem_usage:.1f} MB")

            group_indices = self._discover_categories(features, feature_names, config, verbose)

            # DEBUG: Validate indices immediately after discovery
            all_indices = []
            for cat_name, indices in group_indices.items():
                all_indices.extend(indices)
            if len(all_indices) != len(set(all_indices)):
                from collections import Counter
                counts = Counter(all_indices)
                duplicates = {idx: count for idx, count in counts.items() if count > 1}
                print(f"\n[ERROR] Duplicate indices detected RIGHT AFTER category discovery!")
                print(f"  Total indices: {len(all_indices)}")
                print(f"  Unique indices: {len(set(all_indices))}")
                print(f"  Duplicates: {len(duplicates)}")
                print(f"  First 10 duplicates: {list(duplicates.items())[:10]}")
                for dup_idx in list(duplicates.keys())[:5]:
                    cats_with_idx = [cat for cat, idxs in group_indices.items() if dup_idx in idxs]
                    print(f"    Index {dup_idx}: appears in {cats_with_idx}")
            else:
                print(f"\n[DEBUG] Category discovery indices OK: {len(all_indices)} total, all unique")

            # DEBUG: Memory after category discovery
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[MEMORY DEBUG] After category discovery: {mem_usage:.1f} MB")

            if verbose:
                print(f"Discovered {len(group_indices)} categories:")
                for cat_name, indices in group_indices.items():
                    print(f"  {cat_name}: {len(indices)} features")
        elif config.get("category_mapping_file"):
            # Load categories from JSON file
            if verbose:
                print(f"\nLoading categories from {config['category_mapping_file']}...")

            group_indices = self._load_category_mapping(config["category_mapping_file"])

            # Log loaded structure (always show)
            print(f"\n{'='*70}")
            print(f"LOADED CATEGORY STRUCTURE")
            print(f"{'='*70}")
            print(f"Number of categories: {len(group_indices)}")
            print(f"Total features:       {len(feature_names)}")

            for cat_name, indices in sorted(group_indices.items()):
                # Get feature names for this category
                cat_feature_names = [feature_names[i] for i in sorted(indices)]

                # Determine dominant family
                families = {}
                for fname in cat_feature_names:
                    family = fname.split('::')[0] if '::' in fname else 'unknown'
                    families[family] = families.get(family, 0) + 1

                family_breakdown = ', '.join(f"{k}:{v}" for k, v in sorted(families.items()))

                print(f"\n{cat_name}:")
                print(f"  Features:  {len(indices)} (indices {min(indices)}-{max(indices)})")
                print(f"  Families:  {family_breakdown}")

            print(f"{'='*70}\n")
        elif config.get("resume_from"):
            # Load categories from checkpoint
            if verbose:
                print(f"\nLoading categories from checkpoint: {config['resume_from']}...")

            checkpoint = torch.load(config["resume_from"], weights_only=False)

            # Load from model_config (authoritative source)
            if "model_config" in checkpoint and "group_indices" in checkpoint["model_config"]:
                group_indices = checkpoint["model_config"]["group_indices"]

                # Log loaded structure from checkpoint (always show)
                print(f"\n{'='*70}")
                print(f"RESUMED CATEGORY STRUCTURE")
                print(f"{'='*70}")
                print(f"Number of categories: {len(group_indices)}")
                print(f"Total features:       {len(feature_names)}")

                for cat_name, indices in sorted(group_indices.items()):
                    # Get feature names for this category
                    cat_feature_names = [feature_names[i] for i in sorted(indices)]

                    # Determine dominant family
                    families = {}
                    for fname in cat_feature_names:
                        family = fname.split('::')[0] if '::' in fname else 'unknown'
                        families[family] = families.get(family, 0) + 1

                    family_breakdown = ', '.join(f"{k}:{v}" for k, v in sorted(families.items()))

                    print(f"\n{cat_name}:")
                    print(f"  Features:  {len(indices)} (indices {min(indices)}-{max(indices)})")
                    print(f"  Families:  {family_breakdown}")

                print(f"{'='*70}\n")
            else:
                return self.error(
                    "Checkpoint missing model_config['group_indices']. "
                    "This checkpoint may be from an old version. Please retrain."
                )
        else:
            return self.error("Must specify category_assignment='auto', category_mapping_file, or resume_from")

        # Apply per-category cleaning if deferred from earlier
        print(f"\n[DEBUG] Checking if post-categorization cleaning should run:")
        print(f"  clean_enabled={clean_enabled}, pre_categorization={pre_categorization}")
        print(f"  Condition: {clean_enabled and not pre_categorization}")

        if clean_enabled and not pre_categorization:
            print(f"\n{'='*70}")
            print("CLEANING FEATURES PER CATEGORY")
            print(f"{'='*70}")
            print("This removes redundancy within each discovered category")
            print("while preserving inter-category semantic diversity.\n")

            from spinlock.encoding import FeatureProcessor

            # Read cleaning parameters (same as would be used in global cleaning)
            max_var_thresh = feature_cleaning.get("max_variance_threshold", config.get("max_variance_threshold", None))
            if max_var_thresh is not None:
                max_var_thresh = float(max_var_thresh)

            variance_threshold = float(feature_cleaning.get("variance_threshold", config.get("variance_threshold", 1e-10)))
            deduplicate_threshold = float(feature_cleaning.get("deduplicate_threshold", config.get("deduplicate_threshold", 0.99)))
            use_intelligent_dedup = bool(feature_cleaning.get("use_intelligent_dedup", config.get("use_intelligent_dedup", True)))
            outlier_method = str(feature_cleaning.get("outlier_method", config.get("outlier_method", "percentile")))
            percentile_range = tuple(feature_cleaning.get("percentile_range", config.get("percentile_range", [0.5, 99.5])))
            iqr_multiplier = float(feature_cleaning.get("iqr_multiplier", config.get("iqr_multiplier", 1.5)))
            mad_threshold = float(feature_cleaning.get("mad_threshold", config.get("mad_threshold", 3.0)))
            max_cv_threshold = float(feature_cleaning.get("max_cv_threshold", config.get("max_cv_threshold", 100.0)))

            # Store for checkpoint
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
                "per_category_cleaning": True,
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

            # Track original-to-cleaned index mapping
            global_feature_mask = np.ones(features.shape[1], dtype=bool)
            cleaned_features_list = []
            cleaned_names_list = []
            updated_group_indices = {}
            current_cleaned_idx = 0

            # Clean each category independently
            for cat_name, cat_indices in sorted(group_indices.items()):
                cat_indices_sorted = sorted(cat_indices)
                cat_features = features[:, cat_indices_sorted]
                cat_names = [feature_names[i] for i in cat_indices_sorted] if feature_names else None

                print(f"\n{cat_name}:")
                print(f"  Original: {len(cat_indices_sorted)} features")

                # Clean this category's features
                cat_cleaned, cat_mask, cat_cleaned_names = processor.clean(cat_features, cat_names)

                # Update global mask
                for local_idx, original_idx in enumerate(cat_indices_sorted):
                    if not cat_mask[local_idx]:
                        global_feature_mask[original_idx] = False

                # Track new indices for this category
                num_kept = cat_mask.sum()
                updated_group_indices[cat_name] = list(range(current_cleaned_idx, current_cleaned_idx + num_kept))

                print(f"  Cleaned:  {num_kept} features (removed {len(cat_indices_sorted) - num_kept})")

                # Accumulate cleaned data
                cleaned_features_list.append(cat_cleaned)
                if cat_cleaned_names:
                    cleaned_names_list.extend(cat_cleaned_names)

                current_cleaned_idx += num_kept

            # Reconstruct full feature matrix
            features = np.hstack(cleaned_features_list)
            feature_names = cleaned_names_list if cleaned_names_list else None
            feature_mask = global_feature_mask
            group_indices = updated_group_indices

            # Store feature_mask for later use in _create_data_loaders
            self._feature_mask = feature_mask

            print(f"\n{'='*70}")
            print(f"CLEANING SUMMARY")
            print(f"{'='*70}")
            print(f"Original features: {global_feature_mask.shape[0]}")
            print(f"Cleaned features:  {features.shape[1]} ({100 * features.shape[1] / global_feature_mask.shape[0]:.1f}%)")
            print(f"Removed:           {global_feature_mask.shape[0] - features.shape[1]} ({100 * (1 - features.shape[1] / global_feature_mask.shape[0]):.1f}%)")
            print(f"{'='*70}\n")

        # Pre-compute adaptive compression ratios if "auto" mode
        # IMPORTANT: Do this BEFORE normalization so variance information is preserved!
        if verbose:
            print(f"\nDebug: group_indices before adaptive compression:")
            for cat_name, indices in sorted(group_indices.items()):
                print(f"  {cat_name}: {len(indices)} indices → features shape: {features[:, indices].shape}")

        config = self._precompute_compression_ratios(
            config, features, group_indices, verbose
        )

        # Run adaptive tuning if enabled
        if args.enable_adaptive_tuning and not config.get("resume_from"):
            from spinlock.encoding.training.tuning_trainer import run_adaptive_tuning_workflow
            import json

            if verbose:
                print("\n" + "=" * 70)
                print("RUNNING ADAPTIVE MSE TUNING")
                print("=" * 70)

            # Get initial compression ratios (either from auto-compute or config)
            initial_compression_ratios = config.get("compression_ratios_per_category")

            # If compression ratios weren't computed automatically, create uniform dict
            if initial_compression_ratios is None:
                compression_ratios_str = config.get("compression_ratios")
                if compression_ratios_str:
                    from spinlock.encoding.latent_dim_defaults import parse_compression_ratios
                    if compression_ratios_str.lower() == "auto":
                        # This shouldn't happen since _precompute_compression_ratios should have handled it
                        initial_compression_ratios = {
                            cat: [0.5, 1.0, 1.5] for cat in group_indices.keys()
                        }
                    else:
                        ratios = parse_compression_ratios(compression_ratios_str)
                        initial_compression_ratios = {
                            cat: ratios for cat in group_indices.keys()
                        }
                else:
                    # Default to [0.5, 1.0, 1.5] for all categories
                    initial_compression_ratios = {
                        cat: [0.5, 1.0, 1.5] for cat in group_indices.keys()
                    }

            # Create temporary dataset file for tuning
            import tempfile
            import h5py

            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                tmp_dataset_path = tmp_file.name

            # Write features to temporary HDF5 file
            with h5py.File(tmp_dataset_path, 'w') as f:
                f.create_dataset('features', data=features)

            try:
                # Run adaptive tuning workflow
                optimized_ratios = run_adaptive_tuning_workflow(
                    dataset_path=tmp_dataset_path,
                    group_indices=group_indices,
                    initial_compression_ratios=initial_compression_ratios,
                    tuning_epochs=args.tuning_epochs,
                    tuning_samples=args.tuning_samples,
                    max_sweeps=args.max_sweeps,
                    device=config.get("device", "cuda")
                )

                # Replace initial ratios with optimized ratios
                config["compression_ratios_per_category"] = optimized_ratios
                config["compression_ratios_optimized"] = True

                # Save tuning report
                tuning_report_path = output_dir / "adaptive_tuning_report.json"
                tuning_report = {
                    "initial_ratios": initial_compression_ratios,
                    "optimized_ratios": optimized_ratios,
                    "tuning_config": {
                        "tuning_epochs": args.tuning_epochs,
                        "tuning_samples": args.tuning_samples,
                        "max_sweeps": args.max_sweeps
                    }
                }
                with open(tuning_report_path, "w") as f:
                    json.dump(tuning_report, f, indent=2)

                if verbose:
                    print(f"\nTuning report saved: {tuning_report_path}")

            finally:
                # Clean up temporary dataset
                import os
                if os.path.exists(tmp_dataset_path):
                    os.unlink(tmp_dataset_path)

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

        # DEBUG: Memory before model building
        import resource
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[MEMORY DEBUG] Before model building: {mem_usage:.1f} MB")

        # DEBUG: Validate indices RIGHT before model building
        all_indices = []
        for cat_name, indices in group_indices.items():
            all_indices.extend(indices)
        if len(all_indices) != len(set(all_indices)):
            from collections import Counter
            counts = Counter(all_indices)
            duplicates = {idx: count for idx, count in counts.items() if count > 1}
            print(f"\n[ERROR] Duplicate indices detected RIGHT BEFORE model building!")
            print(f"  Total indices: {len(all_indices)}")
            print(f"  Unique indices: {len(set(all_indices))}")
            print(f"  Duplicates: {len(duplicates)}")
            for dup_idx in list(duplicates.keys())[:10]:
                cats_with_idx = [cat for cat, idxs in group_indices.items() if dup_idx in idxs]
                print(f"    Index {dup_idx} (count={duplicates[dup_idx]}): appears in {cats_with_idx}")
        else:
            print(f"\n[DEBUG] Indices still OK before model building: {len(all_indices)} total, all unique")

        model, vqvae_config = self._build_model(
            normalized_features, group_indices, config, verbose, initial_info
        )

        # DEBUG: Memory after model building
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[MEMORY DEBUG] After model building: {mem_usage:.1f} MB")

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
            per_family_stats,
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

    # Feature loading helper methods

    def _load_raw_ics(
        self,
        f: "h5py.File",
        max_samples: Optional[int],
    ) -> np.ndarray:
        """Load and prepare raw initial conditions for hybrid encoding.

        Args:
            f: Open HDF5 file handle
            max_samples: Optional limit on number of samples to load

        Returns:
            Raw ICs with shape [N, C, H, W]
        """
        import numpy as np

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
        return raw_ics_data.mean(axis=1)

    def _trim_preallocated_samples(
        self,
        features: np.ndarray,
        raw_ics: Optional[np.ndarray],
        feature_family: str,
    ) -> tuple:
        """Trim pre-allocated but unpopulated samples (all zeros).

        Args:
            features: Features array to trim
            raw_ics: Optional raw ICs to trim in sync
            feature_family: Family name for logging

        Returns:
            Tuple of (trimmed_features, trimmed_raw_ics)
        """
        import numpy as np

        sample_norms = np.linalg.norm(
            features.reshape(features.shape[0], -1), axis=1
        )
        populated_mask = sample_norms > 1e-10
        num_populated = populated_mask.sum()
        num_allocated = features.shape[0]

        if num_populated < num_allocated:
            features = features[populated_mask]
            if raw_ics is not None:
                raw_ics = raw_ics[populated_mask]
            print(
                f"  {feature_family}: Dataset pre-allocated to {num_allocated} "
                f"but only {num_populated} samples populated"
            )
            print(f"    Trimmed to {num_populated} non-zero samples")

        return features, raw_ics

    def _apply_hybrid_encoder(
        self,
        manual_features: np.ndarray,
        raw_ics: np.ndarray,
        encoder_params: dict,
        initial_info: dict,
        device: Any,
        chunk_size: int,
    ) -> np.ndarray:
        """Apply InitialHybridEncoder with chunking to prevent OOM.

        Args:
            manual_features: Manual INITIAL features [N, manual_dim]
            raw_ics: Raw initial conditions [N, C, H, W]
            encoder_params: Encoder parameters
            initial_info: Dict with manual_dim, cnn_dim, in_channels
            device: Device to use for encoding
            chunk_size: Samples per chunk

        Returns:
            Hybrid encoded features [N, manual_dim + cnn_dim]
        """
        import torch
        import numpy as np
        from spinlock.encoding.encoders import get_encoder

        encoder = get_encoder(
            "initial_hybrid",
            manual_dim=initial_info["manual_dim"],
            cnn_embedding_dim=initial_info["cnn_dim"],
            encode_manual=encoder_params.get("encode_manual", False),
            in_channels=initial_info["in_channels"],
            use_final_batchnorm=encoder_params.get("use_final_batchnorm", False),
        )
        encoder = encoder.to(device)
        encoder.eval()

        # Apply encoder in chunks
        encoded_chunks = []
        with torch.no_grad():
            for i in range(0, len(manual_features), chunk_size):
                chunk_end = min(i + chunk_size, len(manual_features))

                manual_chunk = torch.from_numpy(
                    manual_features[i:chunk_end]
                ).float().to(device)
                ics_chunk = torch.from_numpy(
                    raw_ics[i:chunk_end]
                ).float().to(device)

                hybrid_chunk = encoder(manual_chunk, ics_chunk)
                encoded_chunks.append(hybrid_chunk.cpu().numpy())

                # Progress logging
                if (i // chunk_size) % 5 == 0:
                    total_chunks = (len(manual_features) + chunk_size - 1) // chunk_size
                    print(f"      Encoded chunk {i//chunk_size + 1}/{total_chunks}")

        return np.concatenate(encoded_chunks, axis=0)

    def _load_initial_hybrid_features(
        self,
        f: "h5py.File",
        feature_family: str,
        family_config: dict,
        feature_type: str,
        max_samples: Optional[int],
        device: Any,
        chunk_size: int,
    ) -> tuple:
        """Load and encode initial hybrid features (manual + CNN).

        Args:
            f: Open HDF5 file handle
            feature_family: Name of feature family
            family_config: Configuration for this family
            feature_type: Type of features (e.g., 'aggregated')
            max_samples: Optional limit on samples
            device: Device for encoding
            chunk_size: Samples per encoding chunk

        Returns:
            Tuple of (encoded_features, raw_ics, initial_info, feature_names)
        """
        import numpy as np

        # Load manual features
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

        # Load raw ICs
        raw_ics = self._load_raw_ics(f, max_samples)

        # Trim pre-allocated samples
        family_features, raw_ics = self._trim_preallocated_samples(
            family_features, raw_ics, feature_family
        )

        # Prepare initial_info
        manual_dim = family_features.shape[1]
        encoder_params = family_config.get("encoder_params", {})
        cnn_dim = encoder_params.get("cnn_embedding_dim", 28)

        initial_info = {
            "offset": 0,  # Will be updated by caller
            "manual_dim": manual_dim,
            "cnn_dim": cnn_dim,
            "in_channels": raw_ics.shape[1],
            "encoded_dim": manual_dim,  # Will be updated after encoding
        }

        print(f"  {feature_family}: Hybrid mode - {manual_dim}D manual + raw ICs {raw_ics.shape}")
        print(f"    CNN embedding_dim={cnn_dim}")

        # Backward compatibility: no CNN encoding
        if cnn_dim == 0:
            print(f"    CNN disabled, using manual features only")
            feature_names = [f"{feature_family}_{i}" for i in range(manual_dim)]
            return family_features, raw_ics, initial_info, feature_names

        # Apply hybrid encoder
        print(f"    Applying InitialHybridEncoder for category discovery...")
        encoded_features = self._apply_hybrid_encoder(
            manual_features=family_features,
            raw_ics=raw_ics,
            encoder_params=encoder_params,
            initial_info=initial_info,
            device=device,
            chunk_size=chunk_size,
        )

        # Update encoded_dim
        output_dim = encoded_features.shape[1]
        initial_info["encoded_dim"] = output_dim

        print(f"    Hybrid encoding complete: {manual_dim}D manual + {cnn_dim}D CNN = {output_dim}D")

        # Generate feature names
        feature_names = [f"{feature_family}_{i}" for i in range(output_dim)]

        return encoded_features, raw_ics, initial_info, feature_names

    def _handle_nan_and_zero_variance(
        self,
        features: np.ndarray,
        feature_family: str,
    ) -> np.ndarray:
        """Replace NaN values and remove zero-variance features.

        Args:
            features: Features array (2D or 3D)
            feature_family: Family name for logging

        Returns:
            Cleaned features array
        """
        import numpy as np

        # Replace NaN with 0
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            features = np.nan_to_num(features, nan=0.0)
            print(f"  {feature_family}: Replaced {nan_count} NaN values with 0")

            # Remove zero-variance features (from NaN→0 replacement)
            if len(features.shape) == 3:  # [N, T, D]
                stds = features.std(axis=(0, 1))
                zero_var_mask = stds > 1e-10
                zero_var_count = (~zero_var_mask).sum()
                if zero_var_count > 0:
                    features = features[:, :, zero_var_mask]
                    print(f"  {feature_family}: Removed {zero_var_count} zero-variance features")
            elif len(features.shape) == 2:  # [N, D]
                stds = features.std(axis=0)
                zero_var_mask = stds > 1e-10
                zero_var_count = (~zero_var_mask).sum()
                if zero_var_count > 0:
                    features = features[:, zero_var_mask]
                    print(f"  {feature_family}: Removed {zero_var_count} zero-variance features")

        return features

    def _compute_sample_norms_chunked(
        self,
        features: np.ndarray,
        use_chunking: bool,
    ) -> Optional[np.ndarray]:
        """Compute per-sample norms to detect populated samples.

        Args:
            features: Features array (2D or 3D)
            use_chunking: Whether to use chunked computation

        Returns:
            Array of sample norms, or None if features are not 2D/3D
        """
        import numpy as np

        if len(features.shape) == 3:  # [N, M, D]
            if use_chunking and len(features) > 5000:
                print(f"    Computing norms in chunks to find populated samples...")
                norm_chunk_size = 5000
                sample_norms = np.empty(len(features), dtype=np.float32)
                for norm_start in range(0, len(features), norm_chunk_size):
                    norm_end = min(norm_start + norm_chunk_size, len(features))
                    chunk_reshaped = features[norm_start:norm_end].reshape(norm_end - norm_start, -1)
                    sample_norms[norm_start:norm_end] = np.linalg.norm(chunk_reshaped, axis=1)
                    del chunk_reshaped
            else:
                sample_norms = np.linalg.norm(features.reshape(features.shape[0], -1), axis=1)
        elif len(features.shape) == 2:  # [N, D]
            if use_chunking and len(features) > 5000:
                print(f"    Computing norms in chunks to find populated samples...")
                norm_chunk_size = 5000
                sample_norms = np.empty(len(features), dtype=np.float32)
                for norm_start in range(0, len(features), norm_chunk_size):
                    norm_end = min(norm_start + norm_chunk_size, len(features))
                    sample_norms[norm_start:norm_end] = np.linalg.norm(
                        features[norm_start:norm_end], axis=1
                    )
            else:
                sample_norms = np.linalg.norm(features, axis=1)
        else:
            sample_norms = None

        return sample_norms

    def _load_reference_features(
        self,
        f: "h5py.File",
        max_samples: Optional[int],
        feature_families: list,
    ) -> tuple:
        """Load reference features and interpolation mask for regularization.

        Args:
            f: Open HDF5 file handle
            max_samples: Optional limit on samples
            feature_families: List of feature families

        Returns:
            Tuple of (reference_features, is_interpolated, raw_summary)
        """
        import numpy as np

        reference_features = None
        is_interpolated = None
        raw_summary = None

        if 'features/reference_features' in f:
            reference_features = f['features/reference_features'][:]
            if max_samples is not None:
                reference_features = reference_features[:max_samples]

            nan_count = np.isnan(reference_features).sum()
            if nan_count > 0:
                reference_features = np.nan_to_num(reference_features, nan=0.0)
                print(f"  summary: Replaced {nan_count} NaN values in reference features")
            print(f"  Loaded reference features for regularization: {reference_features.shape}")

        if 'metadata/is_interpolated' in f:
            is_interpolated = f['metadata/is_interpolated'][:]
            if max_samples is not None:
                is_interpolated = is_interpolated[:max_samples]
            interpolated_count = is_interpolated.sum() if is_interpolated is not None else 0
            print(f"  Loaded interpolation mask: {interpolated_count} / {len(is_interpolated)}")

        # Load raw SUMMARY features for reference regularization
        if any('summary' in fam for fam in feature_families):
            if 'features/summary/per_trajectory/features' in f:
                raw_summary = f['features/summary/per_trajectory/features'][:]
                if max_samples is not None:
                    raw_summary = raw_summary[:max_samples]

                # Reshape from [N, M, D] to [N, M*D]
                if len(raw_summary.shape) == 3:
                    N, M, D = raw_summary.shape
                    raw_summary = raw_summary.reshape(N, M * D)
                    print(f"  Reshaped raw SUMMARY from [{N}, {M}, {D}] to [{N}, {M*D}]")

                nan_count = np.isnan(raw_summary).sum()
                if nan_count > 0:
                    raw_summary = np.nan_to_num(raw_summary, nan=0.0)
                    print(f"  summary: Replaced {nan_count} NaN values in raw SUMMARY")
                print(f"  Loaded raw SUMMARY for reference regularization: {raw_summary.shape}")

        return reference_features, is_interpolated, raw_summary

    def _load_features(self, config: Dict[str, Any]) -> tuple:
        """Load features from HDF5 dataset.

        Supports both single family and multi-family (concatenated) loading.
        Applies per-family encoders from config to transform raw features.

        For hybrid INITIAL encoding (encoder=initial_hybrid), also loads raw ICs
        and defers CNN encoding to training loop for end-to-end learning.

        HDF5 paths:
        - SUMMARY: /features/summary/per_trajectory/features [N, M, D] -> reshaped to [N, M*D]
        - TEMPORAL: /features/temporal/features [N, T, D]
        - INITIAL: /features/initial/aggregated/features [N, D]
        - Raw ICs: /inputs/fields [N, M, H, W] or [N, M, C, H, W]

        Note: SUMMARY uses per_trajectory instead of aggregated to bypass corrupted
        aggregated features (std/cv blocks are near-zero due to low variance across realizations).

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
        raw_temporal_features = None  # For variable-length mode
        raw_temporal_family = None  # Track which family is variable-length
        initial_features_only = None  # Track initial features before temporal concatenation

        with h5py.File(dataset_path, "r") as f:
            for family_idx, feature_family in enumerate(feature_families):
                family_config = families_config[feature_family]
                encoder_name = family_config.get("encoder")

                # Check if variable-length mode is enabled for this family (needed for chunked loading)
                encoder_params = family_config.get("encoder_params", {})
                vl_config = encoder_params.get("variable_length", {})
                vl_enabled = vl_config.get("enabled", False)

                # Handle hybrid INITIAL encoding specially
                if encoder_name in ["initial_hybrid", "InitialHybridEncoder"]:
                    family_features, raw_ics, initial_info, family_names = self._load_initial_hybrid_features(
                        f=f,
                        feature_family=feature_family,
                        family_config=family_config,
                        feature_type=feature_type,
                        max_samples=max_samples,
                        device=device,
                        chunk_size=config.get("encode_chunk_size", 5000),
                    )

                    # Update offset in initial_info
                    initial_info["offset"] = sum(arr.shape[1] for arr in all_features)

                    # Prefix feature names if multi-family
                    if len(feature_families) > 1:
                        family_names = [f"{feature_family}::{name}" for name in family_names]

                    all_features.append(family_features)
                    all_feature_names.extend(family_names)
                    continue

                # Standard feature loading (non-hybrid)
                # Determine correct HDF5 path based on family type
                # TEMPORAL uses /features/temporal directly (no aggregated sublevel)
                # SUMMARY uses /features/summary/per_trajectory (FIX: bypasses corrupted aggregated features)
                if feature_family == "temporal":
                    print(f"  {feature_family}: Starting to load temporal features...")
                    features_path = f"/features/{feature_family}"
                elif feature_family == "summary":
                    # Use per_trajectory features instead of aggregated
                    # Rationale: aggregated features have near-zero std/cv blocks due to low variance
                    # across realizations. per_trajectory [N, M, D] contains all information.
                    features_path = f"/features/{feature_family}/per_trajectory"
                else:
                    features_path = f"/features/{feature_family}/{feature_type}"

                if features_path not in f:
                    raise ValueError(
                        f"Features not found at {features_path}. "
                        f"Available families: {list(f['/features'].keys())}"
                    )

                group = f[features_path]

                # Load features (with optional sample limit)
                print(f"  {feature_family}: Loading features from HDF5...")

                # Get dataset and shape info
                h5_dataset = group["features"]
                dataset_shape = h5_dataset.shape
                total_samples = min(max_samples, dataset_shape[0]) if max_samples and max_samples > 0 else dataset_shape[0]

                # Get chunk size configuration
                chunk_size_config = config.get("hdf5_chunk_size", "auto")

                # Auto-calculate chunk size if needed
                if chunk_size_config == "auto" or chunk_size_config is None:
                    if len(dataset_shape) == 3:
                        # Target 400 MB per chunk
                        sample_bytes = dataset_shape[1] * dataset_shape[2] * 4  # float32 = 4 bytes
                        chunk_size = max(100, min(5000, (400 * 1024 * 1024) // sample_bytes))
                    else:
                        chunk_size = 1000
                elif chunk_size_config == 0:
                    chunk_size = total_samples  # Disable chunking
                else:
                    chunk_size = int(chunk_size_config)

                # Decide whether to use chunking
                use_chunking = (
                    chunk_size < total_samples and
                    vl_enabled and
                    len(dataset_shape) == 3
                )

                if use_chunking:
                    print(f"    Using chunked loading: {total_samples} samples in chunks of {chunk_size}")

                    # Pre-allocate array
                    family_features = np.empty((total_samples, dataset_shape[1], dataset_shape[2]),
                                               dtype=np.float32)

                    # Load in chunks
                    for chunk_start in range(0, total_samples, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, total_samples)

                        # Progress logging every 5 chunks
                        if chunk_start % (chunk_size * 5) == 0:
                            print(f"    Loading chunk {chunk_start//chunk_size + 1} (samples {chunk_start}-{chunk_end})")

                        # Load chunk from HDF5
                        chunk = h5_dataset[chunk_start:chunk_end]
                        family_features[chunk_start:chunk_end] = chunk
                        del chunk
                else:
                    # Original path for small datasets or non-variable-length mode
                    if max_samples is not None and max_samples > 0:
                        family_features = np.array(h5_dataset[:max_samples])
                    else:
                        family_features = np.array(h5_dataset[:])

                print(f"  {feature_family}: Loaded {family_features.shape}")

                # Check for pre-allocated but unpopulated samples
                sample_norms = self._compute_sample_norms_chunked(family_features, use_chunking)

                if sample_norms is not None:
                    populated_mask = sample_norms > 1e-10
                    num_populated = populated_mask.sum()
                    num_allocated = family_features.shape[0]

                    if num_populated < num_allocated:
                        family_features = family_features[populated_mask]
                        print(f"  {feature_family}: Dataset pre-allocated to {num_allocated} but only {num_populated} samples populated")
                        print(f"    Trimmed to {num_populated} non-zero samples")

                # Reshape per_trajectory features from [N, M, D] to [N, M*D]
                if feature_family == "summary" and len(family_features.shape) == 3:
                    N, M, D = family_features.shape
                    family_features = family_features.reshape(N, M * D)
                    print(f"  {feature_family}: Reshaped per_trajectory from [{N}, {M}, {D}] to [{N}, {M*D}]")

                # Replace NaN with 0 before encoding and remove zero-variance features
                family_features = self._handle_nan_and_zero_variance(family_features, feature_family)

                # Apply per-family encoder if configured
                # (encoder_params and vl_enabled already defined at top of loop)
                encoder = None  # Track encoder for pyramid detection below

                # CRITICAL: Variable-length mode requires special handling
                # 1. Encode ONCE at full length for category discovery (clustering needs encoded features)
                # 2. Store raw features for variable-length training
                # 3. Store encoder for runtime variable-length encoding
                if vl_enabled and len(family_features.shape) == 3:
                    print(f"  {feature_family}: Variable-length mode detected - dual encoding strategy")
                    print(f"    Raw temporal features: {family_features.shape}")

                    # Create the temporal encoder
                    from spinlock.encoding.encoders import get_encoder

                    N, T, D = family_features.shape
                    input_dim = D  # Per-timestep feature dimension

                    # Prepare encoder kwargs
                    encoder_kwargs = dict(encoder_params)
                    if "variable_length" in encoder_kwargs:
                        encoder_kwargs["variable_length_config"] = encoder_kwargs.pop("variable_length")

                    # Create encoder
                    temporal_encoder = get_encoder(encoder_name, input_dim=input_dim, **encoder_kwargs)
                    temporal_encoder = temporal_encoder.to(device)
                    temporal_encoder.eval()

                    # STEP 1: Encode at full length for category discovery
                    print(f"    Encoding at full length (T={T}) for category discovery...")
                    batch_size = 1024
                    encode_chunk_size = config.get("encode_chunk_size", 5000)  # Samples to convert to tensor at once

                    # Use chunked encoding to avoid loading full array into GPU memory
                    if use_chunking and encode_chunk_size < len(family_features):
                        print(f"    Using chunked encoding: {len(family_features)} samples in chunks of {encode_chunk_size}")
                        encoded_features_for_clustering = []

                        with torch.no_grad():
                            for chunk_start in range(0, len(family_features), encode_chunk_size):
                                chunk_end = min(chunk_start + encode_chunk_size, len(family_features))

                                # Progress logging
                                if chunk_start % (encode_chunk_size * 3) == 0:
                                    print(f"    Encoding chunk (samples {chunk_start}-{chunk_end})")

                                # Convert chunk to tensor (avoids full array conversion)
                                chunk_tensor = torch.tensor(
                                    family_features[chunk_start:chunk_end],
                                    dtype=torch.float32
                                ).to(device)

                                # Encode chunk in batches
                                for i in range(0, len(chunk_tensor), batch_size):
                                    batch = chunk_tensor[i:i+batch_size]
                                    encoded = temporal_encoder(batch)  # [B, 320] at full length
                                    encoded_features_for_clustering.append(encoded.cpu().numpy())

                                # Free GPU memory
                                del chunk_tensor
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()

                        encoded_for_clustering = np.concatenate(encoded_features_for_clustering, axis=0)
                    else:
                        # Original batched encoding path for small datasets
                        encoded_features_for_clustering = []
                        features_tensor = torch.tensor(family_features, dtype=torch.float32)

                        with torch.no_grad():
                            for i in range(0, len(features_tensor), batch_size):
                                batch = features_tensor[i:i+batch_size].to(device)
                                encoded = temporal_encoder(batch)  # [B, 320] at full length
                                encoded_features_for_clustering.append(encoded.cpu().numpy())

                        encoded_for_clustering = np.concatenate(encoded_features_for_clustering, axis=0)

                    print(f"    Encoded for clustering: {encoded_for_clustering.shape}")

                    # DEBUG: Memory usage after encoding
                    import resource
                    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    print(f"    [MEMORY DEBUG] After encoding: {mem_usage:.1f} MB")

                    # STEP 2: Store raw features for variable-length training
                    raw_temporal_features = family_features
                    raw_temporal_family = feature_family
                    print(f"    Keeping raw features for variable-length training: {raw_temporal_features.shape}")

                    # STEP 3: Store encoder for training loop
                    self._temporal_encoder = temporal_encoder
                    self._temporal_encoder_output_dim = temporal_encoder.total_output_dim

                    # Store encoder config for checkpoint
                    encoder_state_dicts[feature_family] = {
                        'encoder_name': encoder_name,
                        'encoder_params': encoder_params,
                        'state_dict': temporal_encoder.state_dict(),
                        'variable_length_enabled': True,
                    }

                    # IMPORTANT: Capture initial features BEFORE adding temporal for category discovery
                    # Store initial-only features for variable-length training
                    if len(all_features) > 0:
                        initial_features_only = np.concatenate(all_features, axis=1)
                        print(f"    Captured initial-only features: {initial_features_only.shape}")
                    else:
                        initial_features_only = None
                        print(f"    No initial features (temporal-only mode)")

                    # Use encoded features for clustering, but will use raw for training
                    family_features = encoded_for_clustering  # Switch to encoded for clustering
                    print(f"    Proceeding with encoded features for category discovery")

                    # Continue with normal processing to split into pyramid levels
                    encoder = temporal_encoder  # Set encoder so pyramid splitting works below

                elif encoder_name and encoder_name not in ["identity", "IdentityEncoder"]:
                    # Get input dimension for encoder
                    if len(family_features.shape) == 3:
                        # Temporal: [N, T, D] -> input_dim is D
                        input_dim = family_features.shape[2]
                    else:
                        # 2D: [N, D] -> input_dim is D
                        input_dim = family_features.shape[1]

                    # Prepare encoder kwargs - transform variable_length to variable_length_config
                    encoder_kwargs = dict(encoder_params)
                    if "variable_length" in encoder_kwargs:
                        # Rename variable_length -> variable_length_config for PyramidTemporalEncoder
                        encoder_kwargs["variable_length_config"] = encoder_kwargs.pop("variable_length")

                    # Create encoder with input_dim
                    encoder = get_encoder(encoder_name, input_dim=input_dim, **encoder_kwargs)
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
                    # Pyramid encoder: split concatenated output into per-level families
                    if encoder is not None and hasattr(encoder, 'output_dims_per_level'):
                        dims = encoder.output_dims_per_level
                        offset = 0
                        for level_idx, level_dim in enumerate(dims):
                            level_features = family_features[:, offset:offset + level_dim]
                            level_family = f"{feature_family}_p{level_idx}"
                            level_names = [
                                f"{level_family}::{level_family}_{i}"
                                for i in range(level_dim)
                            ]
                            all_features.append(level_features)
                            all_feature_names.extend(level_names)
                            offset += level_dim
                        print(f"  {feature_family}: Split into {len(dims)} pyramid levels: "
                              f"{[f'{feature_family}_p{i} ({d}D)' for i, d in enumerate(dims)]}")
                        continue  # Skip normal feature name generation
                    else:
                        output_dim = family_features.shape[1]
                        family_names = [f"{feature_family}_{i}" for i in range(output_dim)]
                elif len(family_features.shape) == 3:
                    # Variable-length temporal features: keep as [N, T, D] (not encoded yet)
                    # These will be encoded during training loop with sampled lengths
                    # Don't append to all_features - store separately
                    print(f"  {feature_family}: Stored raw temporal features {family_features.shape} (will encode at runtime)")

                    # Store raw temporal features for variable-length training
                    # We'll handle these separately - don't concatenate with other families
                    raw_temporal_features = family_features
                    raw_temporal_family = feature_family

                    # Continue to next family without adding to all_features
                    continue
                else:
                    # Should not happen after encoder, but handle gracefully
                    raise ValueError(
                        f"Features for {feature_family} are {len(family_features.shape)}D after encoding. "
                        f"Expected 2D or 3D (for variable-length). Shape: {family_features.shape}"
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

        # Apply feature exclusion if specified
        # This is useful for excluding features that are not computable on all data types
        # (e.g., cross-channel features that require multiple channels)
        exclude_features = config.get("exclude_features")
        if exclude_features:
            verbose = config.get("verbose", False)
            features, all_feature_names = self._apply_feature_exclusion(
                features, all_feature_names, exclude_features, verbose=verbose
            )

        # Load reference features and interpolation mask for regularization (optional)
        with h5py.File(dataset_path, "r") as f:
            reference_features, is_interpolated, raw_summary = self._load_reference_features(
                f, max_samples, feature_families
            )

        # Store raw temporal features and initial-only features for dataloader creation
        self._raw_temporal_features = raw_temporal_features
        self._raw_temporal_family = raw_temporal_family
        self._initial_features_only = initial_features_only

        # DEBUG: Memory usage after feature loading
        import resource
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"\n[MEMORY DEBUG] After feature loading: {mem_usage:.1f} MB")

        # DEBUG: Verify dimensional consistency
        if initial_info is not None:
            print(f"\n[DIMENSION DEBUG] Category discovery complete:")
            print(f"  Initial features (hybrid encoded): {initial_info['encoded_dim']}D")
            print(f"    - Manual: {initial_info['manual_dim']}D")
            print(f"    - CNN embedding: {initial_info['cnn_dim']}D")
            print(f"  Total features (initial + temporal): {features.shape}")
            print(f"  Model will be built expecting: {features.shape[1]}D input")

        return features, all_feature_names, raw_ics, initial_info, encoder_state_dicts, reference_features, is_interpolated, raw_summary

    def _apply_feature_exclusion(
        self,
        features: np.ndarray,
        feature_names: List[str],
        patterns: List[str],
        verbose: bool = False,
    ) -> tuple:
        """
        Exclude features matching the given patterns.

        Supports wildcard patterns (e.g., "cross_channel_*") and exact matches.
        This is useful for excluding features that are not computable on all data types
        (e.g., cross-channel features that require C > 1).

        Args:
            features: Feature array [N, D]
            feature_names: List of feature names
            patterns: List of patterns to exclude (supports * wildcards)
            verbose: Whether to print excluded feature names

        Returns:
            Tuple of (filtered_features, filtered_feature_names)
        """
        import re

        original_dim = features.shape[1]
        mask = np.ones(len(feature_names), dtype=bool)
        excluded_names = []

        for pattern in patterns:
            if '*' in pattern:
                # Wildcard matching: convert glob pattern to regex
                # Escape special regex chars except *, then convert * to .*
                regex_pattern = pattern.replace('*', '.*')
                regex = re.compile(f'^{regex_pattern}$')

                for i, name in enumerate(feature_names):
                    if regex.match(name) and mask[i]:
                        mask[i] = False
                        excluded_names.append(name)
            else:
                # Exact matching
                for i, name in enumerate(feature_names):
                    if name == pattern and mask[i]:
                        mask[i] = False
                        excluded_names.append(name)

        # Apply mask to features and names
        filtered_features = features[:, mask]
        filtered_names = [n for n, keep in zip(feature_names, mask) if keep]

        # Report exclusions
        if excluded_names:
            print(f"\n{'='*70}")
            print(f"FEATURE EXCLUSION")
            print(f"{'='*70}")
            print(f"Excluded {len(excluded_names)} features ({original_dim}D → {filtered_features.shape[1]}D)")
            print(f"Exclusion patterns: {patterns}")

            if verbose:
                print(f"\nExcluded features:")
                for name in sorted(excluded_names):
                    print(f"  - {name}")
            print(f"{'='*70}\n")

        return filtered_features, filtered_names

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
            min_clusters=config.get("min_clusters", 2),
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
            # Per-family clustering - cluster each family independently
            per_family_clustering=cat_config.get("per_family_clustering", False),
            per_family_params=cat_config.get("per_family_params", {}),
        )

        # Compute assignments
        group_indices = assigner.assign_categories(feature_names, normalized)

        # Log discovered structure (always show, not just verbose)
        print(f"\n{'='*70}")
        print(f"DISCOVERED CATEGORY STRUCTURE")
        print(f"{'='*70}")
        print(f"Number of categories: {len(group_indices)}")
        print(f"Total features:       {len(feature_names)}")

        for cat_name, indices in sorted(group_indices.items()):
            # Get feature names for this category
            cat_feature_names = [feature_names[i] for i in sorted(indices)]

            # Determine dominant family
            families = {}
            for fname in cat_feature_names:
                family = fname.split('::')[0] if '::' in fname else 'unknown'
                families[family] = families.get(family, 0) + 1

            dominant_family = max(families, key=families.get) if families else 'unknown'
            family_breakdown = ', '.join(f"{k}:{v}" for k, v in sorted(families.items()))

            print(f"\n{cat_name}:")
            print(f"  Features:  {len(indices)} (indices {min(indices)}-{max(indices)})")
            print(f"  Families:  {family_breakdown}")
            print(f"  Dominant:  {dominant_family}")

        print(f"{'='*70}\n")

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

        # DEBUG: Final check before creating config
        print(f"\n[DEBUG] Creating CategoricalVQVAEConfig with:")
        print(f"  input_dim: {features.shape[1]}")
        print(f"  num_categories: {len(group_indices)}")
        all_indices_check = []
        for cat_name, indices in group_indices.items():
            all_indices_check.extend(indices)
            print(f"  {cat_name}: {len(indices)} indices")
        print(f"  Total indices: {len(all_indices_check)}, unique: {len(set(all_indices_check))}")

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

        print(f"\n[DEBUG] CategoricalVQVAEConfig created successfully")

        # Build model - use wrapper for hybrid INITIAL encoding
        if initial_info is not None:
            from spinlock.encoding.vqvae_with_initial import VQVAEWithInitial

            model = VQVAEWithInitial(
                vqvae_config=vqvae_config,
                initial_manual_dim=initial_info["manual_dim"],
                initial_cnn_dim=initial_info["cnn_dim"],
                initial_feature_offset=initial_info["offset"],
                initial_feature_count=initial_info["encoded_dim"],  # Use encoded_dim (270D) instead of manual_dim (14D)
                in_channels=initial_info["in_channels"],
            )
            if verbose:
                print("Using VQVAEWithInitial for end-to-end INITIAL CNN training")
        else:
            model = CategoricalHierarchicalVQVAE(vqvae_config)

        # Always show model structure (not just verbose)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created:")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # DEBUG: Verify model input dimension matches features
        if initial_info is not None:
            print(f"\n[DIMENSION DEBUG] Model architecture:")
            print(f"  Model expects input_dim: {vqvae_config.input_dim}D")
            print(f"  Initial feature offset: {initial_info['offset']}")
            print(f"  Initial feature count: {initial_info['encoded_dim']}D (hybrid encoded)")
            print(f"  Features shape: {features.shape}")
            if vqvae_config.input_dim != features.shape[1]:
                print(f"  WARNING: Dimension mismatch! Model expects {vqvae_config.input_dim}D but got {features.shape[1]}D")

        # Log category structure and compression details
        print(f"\n{'='*70}")
        print(f"MODEL ARCHITECTURE")
        print(f"{'='*70}")
        print(f"Number of categories: {len(vqvae_config.categories)}")
        print(f"Total features:       {features.shape[1]}")
        print(f"Total latent dims:    {vqvae_config.total_latent_dim}")
        print(f"Total codebook size:  {vqvae_config.total_tokens:,} codes")

        for cat_idx, cat_name in enumerate(vqvae_config.categories):
            cat_indices = vqvae_config.group_indices[cat_name]
            cat_levels = vqvae_config.levels[cat_name]

            print(f"\n{cat_name.upper()}:")
            print(f"  Features: {len(cat_indices)} (indices {min(cat_indices)}-{max(cat_indices)})")

            # Show hierarchical structure
            print(f"  Hierarchical VQ Levels:")
            for level_idx, level_config in enumerate(cat_levels):
                latent_dim = level_config['latent_dim']
                num_tokens = level_config['num_tokens']

                # Compute compression ratio
                input_dim = len(cat_indices)
                compression_ratio = latent_dim / input_dim

                print(f"    L{level_idx}: {input_dim}D → {latent_dim}D ({num_tokens:,} codes, ratio={compression_ratio:.2f})")

        print(f"{'='*70}\n")

        return model, vqvae_config

    def _create_data_loaders(
        self,
        features: np.ndarray,
        config: Dict[str, Any],
        raw_ics: Optional[np.ndarray] = None,
    ):
        """Create train/val data loaders with optional variable-length support.

        Args:
            features: Pre-encoded features [N, D] or [N, T, D]
            config: Training configuration
            raw_ics: Optional raw initial conditions [N, C, H, W] for hybrid INITIAL

        Returns:
            Tuple of (train_loader, val_loader)
        """
        from spinlock.encoding.training.data_utils import create_train_val_dataloaders
        from spinlock.encoding.variable_length_utils import (
            parse_variable_length_config,
            create_length_sampler,
        )

        # Load reference features and interpolation mask for regularization (if available)
        reference_features = None
        is_interpolated = None
        raw_summary = None
        if hasattr(self, '_reference_features') and self._reference_features is not None:
            reference_features = self._reference_features
        if hasattr(self, '_is_interpolated') and self._is_interpolated is not None:
            is_interpolated = self._is_interpolated
        if hasattr(self, '_raw_summary') and self._raw_summary is not None:
            raw_summary = self._raw_summary

        # Parse variable-length config and create sampler if enabled
        vl_config = parse_variable_length_config(config)
        length_sampler = create_length_sampler(vl_config) if vl_config else None

        if length_sampler is not None and config.get("verbose", False):
            print(f"\nVariable-length mode enabled:")
            print(f"  {length_sampler}")

        # CRITICAL: Handle variable-length mode with separate temporal and initial features
        # When variable-length is enabled:
        # - Raw temporal features [N, T, D] for length sampling (NOT pre-encoded)
        # - Pre-encoded initial features [N, D_initial] to concatenate after temporal encoding
        # - During training, temporal will be encoded with masks, then concatenated with initial
        dataset_features = features
        encoded_initial_features = None

        if length_sampler is not None:
            if hasattr(self, '_raw_temporal_features') and self._raw_temporal_features is not None:
                print(f"\nVariable-length mode: Using raw temporal features for runtime encoding:")
                print(f"  Raw temporal shape: {self._raw_temporal_features.shape}")

                # Use raw temporal features as main features for dataset (with length sampling)
                dataset_features = self._raw_temporal_features

                # CRITICAL FIX: Use initial-only features (NOT full concatenation)
                # The full 'features' includes both initial + temporal_encoded (for clustering)
                # We only want initial features to concatenate with runtime-encoded temporal
                if hasattr(self, '_initial_features_only') and self._initial_features_only is not None:
                    initial_only = self._initial_features_only  # Now hybrid encoded (e.g., 270D)

                    # Apply feature cleaning mask to initial features if cleaning was applied
                    if self._feature_mask is not None:
                        # Extract the portion of feature_mask that corresponds to initial features
                        # NOTE: initial_only is now hybrid encoded (e.g., 270D = 14D manual + 256D CNN)
                        num_initial_features = initial_only.shape[1]
                        initial_feature_mask = self._feature_mask[:num_initial_features]

                        # Apply mask to initial features
                        initial_only_cleaned = initial_only[:, initial_feature_mask]
                        print(f"  Initial features (hybrid encoded) before cleaning: {initial_only.shape}")
                        print(f"  Initial features (hybrid encoded) after cleaning: {initial_only_cleaned.shape}")
                        print(f"  These will be concatenated with encoded temporal features during training")

                        self._encoded_initial_features = initial_only_cleaned
                        encoded_initial_features = initial_only_cleaned
                    else:
                        print(f"  Pre-encoded initial-only features (hybrid) shape: {self._initial_features_only.shape}")
                        print(f"  These will be concatenated with encoded temporal features during training")
                        self._encoded_initial_features = initial_only
                        encoded_initial_features = initial_only
                else:
                    print(f"  No initial features - using temporal only")
                    self._encoded_initial_features = None
            else:
                print(f"\nWARNING: Variable-length sampler enabled but no raw temporal features found!")
                print(f"  This will fail if features are not [N, T, D]. Shape: {features.shape}")

        # Create data loaders using refactored utility
        return create_train_val_dataloaders(
            features=dataset_features,
            config=config,
            raw_ics=raw_ics,
            reference_features=reference_features,
            is_interpolated=is_interpolated,
            raw_summary=raw_summary,
            length_sampler=length_sampler,
            encoded_initial_features=encoded_initial_features,
            verbose=config.get("verbose", False),
        )

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
        per_family_stats: Optional[Dict[str, tuple]] = None,
    ):
        """Create VQVAETrainer with full metadata for checkpoint reproducibility."""
        from spinlock.encoding.training import VQVAETrainer
        from pathlib import Path

        checkpoint_dir = config.get("checkpoint_dir")
        if checkpoint_dir is None and config.get("output_dir"):
            checkpoint_dir = Path(config["output_dir"]) / "checkpoints"

        # Extract reference regularization config
        ref_reg_config = config.get("reference_regularization", {})
        ref_reg_enabled = ref_reg_config.get("enabled", False)
        ref_reg_weight = ref_reg_config.get("weight", 0.0) if ref_reg_enabled else 0.0

        # Entropy regularization for uniform codebook usage
        entropy_weight = config.get("entropy_weight", 0.0)

        # Check if we have temporal encoder for variable-length mode
        temporal_encoder = getattr(self, '_temporal_encoder', None)
        temporal_encoder_output_dim = getattr(self, '_temporal_encoder_output_dim', None)
        encoded_initial_features = getattr(self, '_encoded_initial_features', None)

        # CRITICAL: Disable torch.compile for variable-length mode
        # torch.compile warmup runs model.forward() with a sample batch before training
        # But with variable-length, raw temporal features [B, T, D] need runtime encoding
        # This causes dimension mismatches during compilation
        # TODO: Support torch.compile by creating a wrapper that encodes before model forward
        use_torch_compile = config.get("use_torch_compile", True)
        if temporal_encoder is not None:
            if use_torch_compile and config.get("verbose", True):
                print("\nWARNING: Disabling torch.compile for variable-length mode")
                print("  torch.compile warmup incompatible with runtime temporal encoding")
            use_torch_compile = False

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
            reference_reg_weight=ref_reg_weight,
            entropy_weight=entropy_weight,
            normalize_mse=config.get("normalize_mse", True),  # Normalize MSE losses by default
            early_stopping_patience=config.get("early_stopping_patience", 100),
            early_stopping_min_delta=config.get("early_stopping_min_delta", 0.01),
            dead_code_reset_interval=config.get("dead_code_reset_interval", 100),
            dead_code_threshold=config.get("dead_code_threshold", 10.0),
            dead_code_max_reset_fraction=config.get("dead_code_max_reset_fraction", 0.25),
            use_smart_reset=config.get("use_smart_reset", False),
            checkpoint_dir=checkpoint_dir,
            use_torch_compile=use_torch_compile,  # Modified above for variable-length mode
            val_every_n_epochs=config.get("val_every_n_epochs", 5),
            gradient_clip_norm=config.get("gradient_clip_norm"),
            warmup_epochs=config.get("warmup_epochs", 0),
            verbose=config.get("verbose", True),
            # Metadata for checkpoint reproducibility
            config=config,
            group_indices=group_indices,
            normalization_stats=normalization_stats,
            per_family_normalization_stats=per_family_stats,
            feature_names=feature_names,
            encoder_state_dicts=encoder_state_dicts,
            feature_mask=feature_mask,
            feature_cleaning_params=feature_cleaning_params,
            # Variable-length mode support
            temporal_encoder=temporal_encoder,
            temporal_encoder_output_dim=temporal_encoder_output_dim,
            encoded_initial_features=encoded_initial_features,
        )

        # A3: Feature weighting for reconstruction loss
        feature_weighting_config = config.get("feature_weighting", {})
        if feature_weighting_config.get("enabled", False):
            from spinlock.encoding.feature_weighting import compute_feature_weights
            import torch

            strategy = feature_weighting_config.get("strategy", "variance_information")
            verbose = config.get("verbose", True)

            if verbose:
                print(f"\nComputing feature weights (strategy: {strategy})...")

            # Get training features from train_loader
            # Need to get the actual reconstruction target dimensions
            # (which may be expanded by hybrid encoders)
            sample_batch = next(iter(train_loader))
            sample_features = sample_batch["features"].to(config.get("device", "cuda"))
            sample_raw_ics = sample_batch.get("raw_ics")
            if sample_raw_ics is not None:
                sample_raw_ics = sample_raw_ics.to(config.get("device", "cuda"))

            # Get model output to see actual reconstruction dimension
            with torch.no_grad():
                if sample_raw_ics is not None:
                    sample_output = model(sample_features, raw_ics=sample_raw_ics)
                else:
                    sample_output = model(sample_features)

            # Use input_features if available (hybrid mode), otherwise use features
            if "input_features" in sample_output:
                recon_dim = sample_output["input_features"].shape[1]
                # Get expanded features for weight computation
                all_train_features = []
                for batch in train_loader:
                    features = batch["features"].to(config.get("device", "cuda"))
                    raw_ics = batch.get("raw_ics")
                    if raw_ics is not None:
                        raw_ics = raw_ics.to(config.get("device", "cuda"))
                    with torch.no_grad():
                        if raw_ics is not None:
                            outputs = model(features, raw_ics=raw_ics)
                        else:
                            outputs = model(features)
                        expanded_features = outputs["input_features"]
                        all_train_features.append(expanded_features.cpu().numpy())
                    if len(all_train_features) >= 10:  # Use first ~5000 samples
                        break
            else:
                recon_dim = sample_features.shape[1]
                # Use regular features
                all_train_features = []
                for batch in train_loader:
                    features = batch["features"]
                    all_train_features.append(features.cpu().numpy())
                    if len(all_train_features) >= 10:
                        break

            train_features_np = np.concatenate(all_train_features, axis=0)

            if verbose:
                print(f"Computing weights for {recon_dim}D reconstruction target...")

            # Compute weights
            weights_np = compute_feature_weights(
                train_features_np,
                strategy=strategy,
                normalize=True,
            )

            # Convert to torch and move to device
            feature_weights = torch.from_numpy(weights_np).float().to(config.get("device", "cuda"))

            # Set in trainer
            trainer.feature_weights = feature_weights

            if verbose:
                print(f"Feature weights computed: min={weights_np.min():.3f}, "
                      f"max={weights_np.max():.3f}, mean={weights_np.mean():.3f}")
                print(f"High-importance features: {(weights_np > 1.5).sum()}/{len(weights_np)}")
                print(f"Low-importance features: {(weights_np < 0.5).sum()}/{len(weights_np)}")

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
