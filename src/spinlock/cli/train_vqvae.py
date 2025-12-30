"""
Train VQ-VAE command for Spinlock CLI.

Trains categorical hierarchical VQ-VAE for operator feature tokenization.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import yaml
import numpy as np

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
        if not self.validate_file_exists(config["input_path"], "Dataset"):
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
        """Apply CLI argument overrides to config."""
        if args.input:
            config["input_path"] = args.input
        if args.output:
            config["output_dir"] = args.output
        if args.epochs:
            config["epochs"] = args.epochs
        if args.batch_size:
            config["batch_size"] = args.batch_size
        if args.learning_rate:
            config["learning_rate"] = args.learning_rate
        if args.resume_from:
            config["resume_from"] = args.resume_from
        if args.device:
            config["device"] = args.device
        if args.no_torch_compile:
            config["use_torch_compile"] = False

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration has required fields."""
        required_fields = ["input_path", "output_dir"]
        for field in required_fields:
            if field not in config or config[field] is None:
                raise ValueError(f"Missing required field: {field}")

        # Validate paths are Path objects or strings
        for path_field in ["input_path", "output_dir", "resume_from", "checkpoint_dir"]:
            if path_field in config and config[path_field] is not None:
                config[path_field] = Path(config[path_field])

    def _print_config_summary(self, config: Dict[str, Any], args: Namespace) -> None:
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("VQ-VAE TRAINING CONFIGURATION")
        print("=" * 70)

        print(f"\nDataset: {config['input_path']}")
        print(f"Output:  {config['output_dir']}")

        if config.get("resume_from"):
            print(f"Resume:  {config['resume_from']}")

        print(f"\nFeature Selection:")
        print(f"  Type:   {config.get('feature_type', 'aggregated')}")
        print(f"  Family: {config.get('feature_family', 'sdf')}")

        print(f"\nCategory Discovery:")
        print(f"  Assignment: {config.get('category_assignment', 'auto')}")
        if config.get("category_assignment") == "auto":
            num_cat = config.get("num_categories_auto")
            print(f"  Categories: {num_cat if num_cat else 'auto-determine'}")
            print(f"  Orthogonality target: {config.get('orthogonality_target', 0.15)}")

        print(f"\nModel Architecture:")
        print(f"  Group embedding dim: {config.get('group_embedding_dim', 64)}")
        print(f"  Group hidden dim:    {config.get('group_hidden_dim', 128)}")
        print(f"  Hierarchical levels: {config.get('factors', 'auto-computed')}")

        print(f"\nTraining:")
        print(f"  Epochs:        {config.get('epochs', 500)}")
        print(f"  Batch size:    {config.get('batch_size', 512)}")
        print(f"  Learning rate: {config.get('learning_rate', 0.0005)}")
        print(f"  Device:        {config.get('device', 'cuda')}")
        print(f"  torch.compile: {config.get('use_torch_compile', True)}")

        print(f"\nLoss Weights:")
        print(f"  Commitment:      {config.get('commitment_cost', 0.45)}")
        print(f"  Orthogonality:   {config.get('orthogonality_weight', 0.1)}")
        print(f"  Informativeness: {config.get('informativeness_weight', 0.1)}")
        print(f"  Topographic:     {config.get('topo_weight', 0.3)}")

        print(f"\nCallbacks:")
        print(f"  Early stopping patience: {config.get('early_stopping_patience', 100)}")
        print(f"  Dead code reset interval: {config.get('dead_code_reset_interval', 100)}")
        print(f"  Dead code threshold: {config.get('dead_code_threshold', 10.0)}th percentile")

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
        from pathlib import Path

        start_time = time.time()

        # Create output directory
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print("Loading dataset and features...")

        # Load features from dataset
        features, feature_names = self._load_features(config)

        if verbose:
            print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")

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
            group_indices = checkpoint["group_indices"]

            if verbose:
                print(f"Loaded {len(group_indices)} categories from checkpoint")
        else:
            return self.error("Must specify category_assignment='auto', category_mapping_file, or resume_from")

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

        # Build VQ-VAE model
        if verbose:
            print("\nBuilding VQ-VAE model...")

        model, vqvae_config = self._build_model(normalized_features, group_indices, config, verbose)

        # Create data loaders
        if verbose:
            print("\nCreating data loaders...")

        train_loader, val_loader = self._create_data_loaders(normalized_features, config)

        if verbose:
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Val samples:   {len(val_loader.dataset)}")

        # Create trainer
        if verbose:
            print("\nInitializing trainer...")

        trainer = self._create_trainer(model, train_loader, val_loader, config)

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

    def _load_features(self, config: Dict[str, Any]) -> tuple:
        """Load features from HDF5 dataset."""
        import h5py
        import numpy as np

        dataset_path = config["input_path"]
        feature_type = config.get("feature_type", "aggregated")
        feature_family = config.get("feature_family", "sdf")

        with h5py.File(dataset_path, "r") as f:
            # Navigate to features group
            features_path = f"/features/{feature_family}/{feature_type}"

            if features_path not in f:
                raise ValueError(
                    f"Features not found at {features_path}. "
                    f"Run 'spinlock extract-features' first."
                )

            group = f[features_path]

            # Load features
            features = np.array(group["features"])

            # Load feature names
            if "feature_names" in group.attrs:
                feature_names = [name.decode() if isinstance(name, bytes) else name
                               for name in group.attrs["feature_names"]]
            else:
                feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        return features, feature_names

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
        """Auto-discover feature categories via hierarchical clustering."""
        from spinlock.encoding import (
            hierarchical_clustering_assignment,
            standard_normalize,
        )

        # Normalize features for clustering (equal importance)
        normalized = standard_normalize(features)

        # Cluster
        group_indices = hierarchical_clustering_assignment(
            features=normalized,
            feature_names=feature_names,
            num_clusters=config.get("num_categories_auto"),
            orthogonality_target=config.get("orthogonality_target", 0.15),
            min_features_per_cluster=config.get("min_features_per_category", 3),
            max_clusters=config.get("max_clusters", 25),
            random_seed=config.get("random_seed", 42),
        )

        return group_indices

    def _normalize_features(
        self, features: np.ndarray, group_indices: Dict[str, list], config: Dict[str, Any]
    ) -> tuple:
        """Normalize features per category."""
        from spinlock.encoding import (
            compute_normalization_stats,
            apply_standard_normalization,
        )
        import numpy as np

        normalized = features.copy()
        stats_dict = {}

        for category, indices in group_indices.items():
            cat_features = features[:, indices]
            stats = compute_normalization_stats(cat_features)
            normalized[:, indices] = apply_standard_normalization(cat_features, stats)
            stats_dict[category] = stats

        return normalized, stats_dict

    def _save_normalization_stats(self, stats_dict: Dict[str, Any], path: Path) -> None:
        """Save normalization stats to .npz file."""
        import numpy as np

        # Convert stats objects to numpy arrays
        save_dict = {}
        for category, stats in stats_dict.items():
            save_dict[f"{category}_mean"] = stats.mean
            save_dict[f"{category}_std"] = stats.std

        np.savez(path, **save_dict)

    def _build_model(
        self, features: np.ndarray, group_indices: Dict[str, list], config: Dict[str, Any], verbose: bool
    ):
        """Build VQ-VAE model."""
        from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
        import torch

        vqvae_config = CategoricalVQVAEConfig(
            input_dim=features.shape[1],
            group_indices=group_indices,
            group_embedding_dim=config.get("group_embedding_dim", 64),
            group_hidden_dim=config.get("group_hidden_dim", 128),
            levels=config.get("factors"),  # None = auto-compute
            commitment_cost=config.get("commitment_cost", 0.45),
            use_ema=config.get("use_ema", True),
            ema_decay=config.get("ema_decay", 0.99),
        )

        model = CategoricalHierarchicalVQVAE(vqvae_config)

        if verbose:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model created:")
            print(f"  Total parameters:     {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")

        return model, vqvae_config

    def _create_data_loaders(self, features: np.ndarray, config: Dict[str, Any]):
        """Create train/val data loaders."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        import numpy as np

        # Split into train/val (90/10)
        n_samples = len(features)
        n_train = int(0.9 * n_samples)

        # Shuffle
        rng = np.random.RandomState(config.get("random_seed", 42))
        indices = rng.permutation(n_samples)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_features = torch.from_numpy(features[train_indices]).float()
        val_features = torch.from_numpy(features[val_indices]).float()

        train_dataset = TensorDataset(train_features)
        val_dataset = TensorDataset(val_features)

        batch_size = config.get("batch_size", 512)
        val_batch_size = config.get("val_batch_size", batch_size)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        return train_loader, val_loader

    def _create_trainer(self, model, train_loader, val_loader, config: Dict[str, Any]):
        """Create VQVAETrainer."""
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
            checkpoint_dir=checkpoint_dir,
            use_torch_compile=config.get("use_torch_compile", True),
            val_every_n_epochs=config.get("val_every_n_epochs", 5),
            verbose=config.get("verbose", True),
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
    ):
        """Save final model checkpoint."""
        import torch

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "group_indices": group_indices,
            "normalization_stats": normalization_stats,
            "feature_names": feature_names,
            "config": config,
            "history": history,
        }

        # Add model config
        if hasattr(model, "config"):
            config_dict = {
                "input_dim": model.config.input_dim,
                "group_indices": model.config.group_indices,
                "group_embedding_dim": model.config.group_embedding_dim,
                "group_hidden_dim": model.config.group_hidden_dim,
            }

            # Add levels
            if hasattr(model.config, "category_levels") and model.config.category_levels:
                config_dict["category_levels"] = model.config.category_levels
            elif hasattr(model.config, "levels") and model.config.levels:
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
