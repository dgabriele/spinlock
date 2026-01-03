"""
Extract features command for Spinlock CLI.

Post-processing command that extracts features from existing datasets
for downstream VQ-VAE training and analysis.

Two sibling feature families:
- TEMPORAL: Per-timestep time series [N, T, D] (spatial, spectral, cross_channel)
- SUMMARY: Aggregated scalars [N, D] (temporal dynamics, causality, invariant_drift)
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional
import sys

from .base import CLICommand


class ExtractFeaturesCommand(CLICommand):
    """
    Command to extract features from generated datasets.

    Two sibling feature families:
    - TEMPORAL: Per-timestep time series [N, T, D]
    - SUMMARY: Aggregated scalars [N, D]
    """

    @property
    def name(self) -> str:
        return "extract-features"

    @property
    def help(self) -> str:
        return "Extract features from dataset for downstream analysis"

    @property
    def description(self) -> str:
        return """
Extract TEMPORAL and SUMMARY features from a Spinlock dataset.

Two sibling feature families:
- TEMPORAL: Per-timestep time series [N, T, D] (spatial, spectral, cross_channel)
- SUMMARY: Aggregated scalars [N, D] (temporal dynamics, causality, invariant_drift)

Features are extracted at multiple temporal granularities:
  - Per-timestep: Features for each evolution step (TEMPORAL)
  - Per-trajectory: Aggregated over time for each realization (SUMMARY)
  - Aggregated: Final summary across all realizations (SUMMARY)

Extracted features are stored in /features/temporal/ and /features/summary/.

Examples:
  # Extract SUMMARY features with default settings
  spinlock extract-features --dataset datasets/benchmark_10k.h5

  # Extract with custom config
  spinlock extract-features \\
      --dataset datasets/benchmark_10k.h5 \\
      --config configs/feature_extraction/sdf_full.yaml

  # Extract to separate output file
  spinlock extract-features \\
      --dataset datasets/benchmark_10k.h5 \\
      --output datasets/benchmark_10k_features.h5

  # Dry run to preview extraction
  spinlock extract-features \\
      --dataset datasets/benchmark_10k.h5 \\
      --dry-run --verbose
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add extract-features command arguments."""
        # Required arguments
        parser.add_argument(
            "--dataset",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to HDF5 dataset to extract features from",
        )

        # Configuration
        config_group = parser.add_argument_group("configuration")

        config_group.add_argument(
            "--config",
            type=Path,
            metavar="PATH",
            help="Path to feature extraction config YAML (optional, uses defaults if not provided)",
        )

        config_group.add_argument(
            "--output",
            type=Path,
            metavar="PATH",
            help="Output dataset path (default: writes to input dataset)",
        )

        # Feature selection
        feature_group = parser.add_argument_group("feature selection")

        feature_group.add_argument(
            "--enable-spatial",
            action="store_true",
            help="Enable SUMMARY spatial statistics features",
        )

        feature_group.add_argument(
            "--enable-spectral",
            action="store_true",
            help="Enable SUMMARY spectral features",
        )

        feature_group.add_argument(
            "--enable-temporal",
            action="store_true",
            help="Enable SUMMARY temporal features",
        )

        feature_group.add_argument(
            "--num-fft-scales",
            type=int,
            metavar="N",
            help="Number of FFT frequency scales (default: 5)",
        )

        # Execution options
        exec_group = parser.add_argument_group("execution options")

        exec_group.add_argument(
            "--batch-size",
            type=int,
            default=32,
            metavar="N",
            help="Batch size for feature extraction (default: 32)",
        )

        exec_group.add_argument(
            "--max-samples",
            type=int,
            metavar="N",
            help="Maximum number of samples to extract features from (default: all)",
        )

        exec_group.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "cpu"],
            metavar="DEVICE",
            help="Device for computation (default: cuda)",
        )

        exec_group.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing features (default: skip if exists)",
        )

        exec_group.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate configuration without extracting features",
        )

        exec_group.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed progress information",
        )

    def execute(self, args: Namespace) -> int:
        """Execute feature extraction pipeline."""
        # Validate dataset exists
        if not self.validate_file_exists(args.dataset, "Dataset"):
            return 1

        # Load or create config
        if args.config:
            if not self.validate_file_exists(args.config, "Config"):
                return 1
            try:
                config = self._load_feature_config(args.config)
            except Exception as e:
                return self.error(f"Failed to load config: {e}")
        else:
            # Use defaults with CLI overrides
            config = self._create_default_config(args)

        # Apply CLI overrides
        config = self._apply_cli_overrides(config, args)

        # Validate output path
        if args.output:
            if args.output.resolve() == args.dataset.resolve():
                return self.error("Output path must differ from input dataset")
            config.output_dataset = args.output

        # Print configuration summary
        if args.verbose or args.dry_run:
            self._print_config_summary(config, args)

        # Dry run: validate and exit
        if args.dry_run:
            print("\n✓ Configuration valid (dry-run mode, no features extracted)")
            return 0

        # Execute extraction
        try:
            return self._run_extraction(config, args.verbose)
        except KeyboardInterrupt:
            print("\n\nExtraction interrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            import traceback
            print(f"\nError during extraction: {e}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 1

    def _load_feature_config(self, config_path: Path):
        """Load feature extraction config from YAML."""
        from spinlock.features.config import FeatureExtractionConfig
        import yaml

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return FeatureExtractionConfig(**config_dict)

    def _create_default_config(self, args: Namespace):
        """Create default feature extraction config."""
        from spinlock.features.config import FeatureExtractionConfig
        from spinlock.features.summary.config import SummaryConfig

        return FeatureExtractionConfig(
            input_dataset=args.dataset,
            summary=SummaryConfig(),  # All defaults enabled
            batch_size=args.batch_size,
            device=args.device,
            overwrite=args.overwrite,
        )

    def _apply_cli_overrides(self, config, args: Namespace):
        """Apply CLI argument overrides to config."""
        # SDF feature category toggles
        if args.enable_spatial:
            config.summary.spatial.enabled = True
        if args.enable_spectral:
            config.summary.spectral.enabled = True
        if args.enable_temporal:
            config.summary.temporal.enabled = True

        # Multiscale parameters
        if args.num_fft_scales:
            config.summary.spectral.num_fft_scales = args.num_fft_scales

        # Execution settings
        config.batch_size = args.batch_size
        config.device = args.device
        config.overwrite = args.overwrite

        # Subsetting
        if args.max_samples:
            config.max_samples = args.max_samples

        return config

    def _print_config_summary(self, config, args: Namespace) -> None:
        """Print configuration summary."""
        print("\n" + "="*60)
        print("FEATURE EXTRACTION CONFIGURATION")
        print("="*60)

        print(f"\nDataset: {config.input_dataset}")
        if config.output_dataset:
            print(f"Output:  {config.output_dataset}")

        print(f"\nSDF Features:")
        if config.summary:
            est_count = config.summary.estimate_feature_count()
            print(f"  Estimated features: ~{est_count}")
            print(f"  Spatial:            {'enabled' if config.summary.spatial.enabled else 'disabled'}")
            print(f"  Spectral:           {'enabled' if config.summary.spectral.enabled else 'disabled'} (scales: {config.summary.spectral.num_fft_scales})")
            print(f"  Temporal:           {'enabled' if config.summary.temporal.enabled else 'disabled'}")
            print(f"  Structural:         {'enabled' if config.summary.structural.enabled else 'disabled'}")
            print(f"  Physics:            {'enabled' if config.summary.physics.enabled else 'disabled'}")
            print(f"  Morphological:      {'enabled' if config.summary.morphological.enabled else 'disabled'}")
            print(f"  Multiscale:         {'enabled' if config.summary.multiscale.enabled else 'disabled'}")

        print(f"\nExecution:")
        print(f"  Batch size:         {config.batch_size}")
        print(f"  Device:             {config.device}")
        print(f"  Overwrite:          {config.overwrite}")
        if config.max_samples is not None:
            print(f"  Max samples:        {config.max_samples}")

        print("="*60 + "\n")

    def _run_extraction(self, config, verbose: bool) -> int:
        """
        Run feature extraction pipeline.

        Pipeline:
        1. Load dataset and read outputs
        2. Initialize feature extractors
        3. Extract features in batches
        4. Write features to HDF5
        """
        from spinlock.features.extractor import FeatureExtractor
        import time

        start_time = time.time()

        if verbose:
            print("Initializing feature extractor...")

        # Create extractor
        extractor = FeatureExtractor(config)

        # Run extraction
        if verbose:
            print("\nExtracting features...")

        extractor.extract(verbose=verbose)

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n✓ Feature extraction complete ({elapsed:.1f}s)")

        return 0
