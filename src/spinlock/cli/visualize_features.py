"""
Feature visualization CLI command.

Samples diverse operators and creates SVG visualizations of time series
features for documentation and analysis.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import numpy as np
import sys

from .base import CLICommand
from spinlock.visualization.features import (
    select_diverse_operators,
    FeatureLinePlotter,
    FeatureLayoutManager,
    load_parameters,
    load_per_timestep_features,
)
from spinlock.visualization.features.data_loader import check_dataset_compatibility


class VisualizeFeaturesCommand(CLICommand):
    """
    Visualize time series features from neural operators.

    Samples diverse operators (maximizing diversity in parameter + feature space)
    and creates tall SVG visualizations showing all per-timestep features
    grouped by category (spatial, spectral, temporal, etc.).

    Each SVG shows:
    - One line chart per feature (mean ± std across realizations)
    - Features grouped by category with headers
    - Clean, publication-ready styling
    - 256px wide (default), unlimited height
    """

    @property
    def name(self) -> str:
        return "visualize-features"

    @property
    def help(self) -> str:
        return "Visualize time series features from operators"

    @property
    def description(self) -> str:
        return """
Visualize per-timestep features from SDF-extracted operators.

Samples diverse operators (maximizing diversity in parameter + feature space)
and creates tall SVG visualizations showing all time series features
grouped by category.

Each visualization includes:
  - Mean line + shaded ±1 std envelope across realizations
  - Features grouped by category (spatial, spectral, temporal, ...)
  - Clean SVG format for markdown embedding
  - 256px wide (default), unlimited height

Examples:
  # Basic usage (sample 2 diverse operators)
  spinlock visualize-features \\
      --dataset datasets/test_1k_inline_features.h5 \\
      --output visualizations/features/

  # Sample 5 diverse operators
  spinlock visualize-features \\
      --dataset datasets/vqvae_baseline_10k_temporal.h5 \\
      --output visualizations/features/ \\
      --n-operators 5

  # Visualize specific operators
  spinlock visualize-features \\
      --dataset datasets/test_1k_inline_features.h5 \\
      --output visualizations/features/ \\
      --operator-indices 42 123 456

  # Adjust diversity weighting (more feature-based)
  spinlock visualize-features \\
      --dataset datasets/test_1k_inline_features.h5 \\
      --output visualizations/features/ \\
      --diversity-alpha 0.3  # Lower = more feature diversity
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add command-line arguments."""

        # Required arguments
        parser.add_argument(
            "--dataset",
            type=Path,
            required=True,
            help="Path to HDF5 dataset with SDF features"
        )
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Output directory for SVG files"
        )

        # Sampling arguments
        sampling_group = parser.add_argument_group("sampling options")
        sampling_group.add_argument(
            "--n-operators",
            type=int,
            default=2,
            help="Number of operators to sample (default: 2)"
        )
        sampling_group.add_argument(
            "--operator-indices",
            type=int,
            nargs="+",
            help="Specific operator indices (overrides diversity sampling)"
        )
        sampling_group.add_argument(
            "--diversity-alpha",
            type=float,
            default=0.5,
            help="Weight for parameter vs feature diversity [0-1] (default: 0.5)"
        )
        sampling_group.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for sampling (default: 42)"
        )

        # Visualization arguments
        viz_group = parser.add_argument_group("visualization options")
        viz_group.add_argument(
            "--width-px",
            type=int,
            default=256,
            help="Chart width in pixels (default: 256)"
        )
        viz_group.add_argument(
            "--dpi",
            type=int,
            default=100,
            help="Resolution in DPI (default: 100)"
        )

        # Execution arguments
        exec_group = parser.add_argument_group("execution options")
        exec_group.add_argument(
            "--verbose",
            action="store_true",
            help="Show progress and details"
        )
        exec_group.add_argument(
            "--dry-run",
            action="store_true",
            help="Check dataset compatibility without generating visualizations"
        )

    def execute(self, args: Namespace) -> int:
        """Execute the feature visualization command."""

        # Validate dataset exists
        if not self.validate_file_exists(args.dataset, "Dataset"):
            return 1

        # Check dataset compatibility
        if args.verbose or args.dry_run:
            print(f"Checking dataset: {args.dataset}")

        metadata = check_dataset_compatibility(args.dataset)

        if not metadata['has_sdf']:
            return self.error(
                f"Dataset {args.dataset} does not contain SDF features.\n"
                "Run feature extraction first: spinlock extract-features --dataset ..."
            )

        if metadata['num_timesteps'] is None:
            return self.error(
                f"Dataset {args.dataset} does not have per-timestep features.\n"
                "Only aggregated features found. This command requires per-timestep features."
            )

        if args.verbose or args.dry_run:
            print(f"\nDataset summary:")
            print(f"  Operators: {metadata['num_operators']}")
            print(f"  Timesteps: {metadata['num_timesteps']}")
            print(f"  Features: {metadata['num_features']}")
            print(f"  Categories: {', '.join(metadata['feature_categories'])}")

        if args.dry_run:
            print("\nDry run complete. Dataset is compatible.")
            return 0

        # Load data
        try:
            if args.verbose:
                print(f"\nLoading features...")

            features, registry = load_per_timestep_features(args.dataset)
            N, T, D = features.shape

            if args.verbose:
                print(f"  Loaded features: {features.shape} [N, T, D]")

        except Exception as e:
            return self.error(f"Failed to load features: {e}")

        # Determine which operators to visualize
        if args.operator_indices:
            # Use explicitly specified indices
            selected_indices = args.operator_indices

            # Validate indices
            invalid_indices = [idx for idx in selected_indices if idx < 0 or idx >= N]
            if invalid_indices:
                return self.error(
                    f"Invalid operator indices: {invalid_indices}\n"
                    f"Valid range: [0, {N-1}]"
                )

            if args.verbose:
                print(f"\nUsing specified operator indices: {selected_indices}")

        else:
            # Sample diverse operators
            if args.verbose:
                print(f"\nSampling {args.n_operators} diverse operators...")
                print(f"  Diversity weighting: {args.diversity_alpha:.2f} (param) + "
                      f"{1-args.diversity_alpha:.2f} (feature)")

            try:
                # Load parameters for diversity sampling
                parameters = load_parameters(args.dataset)

                # Sample diverse operators
                selected_indices = select_diverse_operators(
                    parameters=parameters,
                    features=features,
                    n_select=args.n_operators,
                    alpha=args.diversity_alpha,
                    seed=args.seed
                )

                if args.verbose:
                    print(f"  Selected operators: {selected_indices}")

            except Exception as e:
                return self.error(f"Diversity sampling failed: {e}")

        # Initialize visualization components
        plotter = FeatureLinePlotter(
            figsize_per_feature=(args.width_px / args.dpi, 0.6),
            dpi=args.dpi
        )
        layout_manager = FeatureLayoutManager()

        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        if args.verbose:
            print(f"\nGenerating visualizations...")

        for op_idx in selected_indices:
            try:
                # Extract features for this operator [T, D]
                op_features = features[op_idx]

                # Organize by category
                organized = layout_manager.organize_features(op_features, registry)

                if not organized:
                    print(f"Warning: No features found for operator {op_idx}, skipping")
                    continue

                # Create SVG
                output_path = args.output / f"operator_{op_idx:05d}_features.svg"

                timesteps = np.arange(T)
                plotter.create_tall_svg(
                    features_by_category=organized,
                    timesteps=timesteps,
                    output_path=output_path,
                    operator_idx=op_idx
                )

                if args.verbose:
                    print(f"  Created: {output_path}")
                    # Print summary
                    summary = layout_manager.get_category_summary(organized)
                    total_features = sum(s['num_features'] for s in summary.values())
                    print(f"    Categories: {len(summary)}, Total features: {total_features}")

            except Exception as e:
                print(f"Error generating visualization for operator {op_idx}: {e}", file=sys.stderr)
                continue

        if args.verbose:
            print(f"\n✅ Successfully generated {len(selected_indices)} visualization(s)")
            print(f"Output directory: {args.output}")
            print(f"\nTo embed in markdown:")
            for op_idx in selected_indices:
                print(f"  ![Operator {op_idx}](./operator_{op_idx:05d}_features.svg)")

        return 0
