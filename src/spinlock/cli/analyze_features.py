"""Analyze features command for feature pruning and quality assessment.

Post-processing command that analyzes extracted features to identify:
- Near-constant features (low variance)
- Redundant features (high correlation)
- High-NaN features (poor quality)

Generates actionable recommendations for feature selection before VQ-VAE training.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import sys
import json

import h5py
import numpy as np

from .base import CLICommand
from spinlock.features.analyzer import FeatureAnalyzer, AnalysisMetadata
from spinlock.features.registry import FeatureRegistry


class AnalyzeFeaturesCommand(CLICommand):
    """Command to analyze dataset features for pruning recommendations.

    Analyzes extracted features across three dimensions:
    1. Variance: Near-constant features
    2. Correlation: Redundant features
    3. NaN Frequency: Unreliable features
    """

    @property
    def name(self) -> str:
        return "analyze-features"

    @property
    def help(self) -> str:
        return "Analyze dataset features for quality and redundancy"

    @property
    def description(self) -> str:
        return """
Analyze extracted features to identify low-value features for pruning.

Identifies features that are:
- Near-constant (low variance)
- Redundant (highly correlated)
- Frequently NaN (poor quality)

Generates pruning recommendations to reduce computational overhead
and improve VQ-VAE training efficiency.

Examples:
  # Basic analysis with default thresholds
  spinlock analyze-features --dataset datasets/benchmark_10k.h5

  # Custom thresholds
  spinlock analyze-features \\
      --dataset datasets/cno_50k_v3_1.h5 \\
      --variance-threshold 1e-6 \\
      --correlation-threshold 0.95 \\
      --nan-threshold 0.05

  # Generate visualizations
  spinlock analyze-features \\
      --dataset datasets/cno_50k_v3_1.h5 \\
      --visualize \\
      --output analysis_results.json

  # Dry run (preview without saving)
  spinlock analyze-features \\
      --dataset datasets/cno_50k_v3_1.h5 \\
      --dry-run --verbose
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add analyze-features command arguments."""
        # Required arguments
        parser.add_argument(
            "--dataset",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to HDF5 dataset with extracted features",
        )

        # Analysis options
        analysis_group = parser.add_argument_group("analysis options")
        analysis_group.add_argument(
            "--feature-family",
            type=str,
            default="temporal",
            choices=["temporal", "summary_per_trajectory", "summary_aggregated"],
            help="Feature family to analyze (default: temporal)"
        )
        analysis_group.add_argument(
            "--variance-threshold",
            type=float,
            default=1e-8,
            metavar="FLOAT",
            help="Variance threshold for near-zero features (default: 1e-8)"
        )
        analysis_group.add_argument(
            "--correlation-threshold",
            type=float,
            default=0.99,
            metavar="FLOAT",
            help="Correlation threshold for redundancy (default: 0.99)"
        )
        analysis_group.add_argument(
            "--nan-threshold",
            type=float,
            default=0.01,
            metavar="FLOAT",
            help="Max NaN fraction (default: 0.01 = 1%%)"
        )

        # Output options
        output_group = parser.add_argument_group("output options")
        output_group.add_argument(
            "--output",
            type=Path,
            metavar="PATH",
            help="Output path for analysis (default: {dataset}_analysis.json)"
        )
        output_group.add_argument(
            "--output-format",
            type=str,
            default="json",
            choices=["json", "yaml"],
            help="Output format (default: json)"
        )
        output_group.add_argument(
            "--save-pruned-registry",
            action="store_true",
            help="Save pruned feature registry alongside analysis"
        )
        output_group.add_argument(
            "--visualize",
            action="store_true",
            help="Generate visualization plots"
        )

        # Execution options
        exec_group = parser.add_argument_group("execution options")
        exec_group.add_argument(
            "--verbose",
            action="store_true",
            help="Detailed progress output"
        )
        exec_group.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview analysis without saving results"
        )

    def execute(self, args: Namespace) -> int:
        """Execute feature analysis."""
        # Validate dataset exists
        if not self.validate_file_exists(args.dataset, "Dataset"):
            return 1

        try:
            # Load features from HDF5
            if args.verbose:
                print(f"Loading features from: {args.dataset}")

            features, feature_names, metadata = self._load_features(
                args.dataset,
                args.feature_family,
                args.verbose
            )

            # Update metadata with thresholds
            metadata.thresholds = {
                "variance": args.variance_threshold,
                "correlation": args.correlation_threshold,
                "nan": args.nan_threshold,
            }

            # Run analysis
            if args.verbose:
                print(f"\nAnalyzing {features.shape[0]} samples with {features.shape[1]} features...")

            analyzer = FeatureAnalyzer(
                variance_threshold=args.variance_threshold,
                correlation_threshold=args.correlation_threshold,
                nan_threshold=args.nan_threshold,
            )

            results = analyzer.analyze(features, feature_names, metadata)

            # Print summary
            self._print_summary(results, args.verbose)

            # Save results (unless dry-run)
            if not args.dry_run:
                output_path = args.output or self._default_output_path(args.dataset)
                self._save_results(results, output_path, args.output_format)

                if args.save_pruned_registry:
                    self._save_pruned_registry(
                        results,
                        feature_names,
                        args.feature_family,
                        output_path,
                        args.verbose
                    )

                if args.visualize:
                    self._generate_visualizations(
                        results,
                        output_path,
                        args.verbose
                    )
            else:
                if args.verbose:
                    print("\nDry run - no files saved")

            return 0

        except Exception as e:
            return self.error(f"Analysis failed: {e}")

    def _load_features(
        self,
        dataset_path: Path,
        family: str,
        verbose: bool
    ) -> Tuple[np.ndarray, List[str], AnalysisMetadata]:
        """Load features from HDF5 dataset.

        Args:
            dataset_path: Path to HDF5 dataset
            family: Feature family name
            verbose: Print loading progress

        Returns:
            Tuple of (features, feature_names, metadata)
        """
        with h5py.File(dataset_path, "r") as f:
            # Determine feature path and shape based on family
            if family == "temporal":
                if "/features/temporal/features" not in f:
                    raise ValueError(f"Dataset does not contain temporal features")

                features = f["/features/temporal/features"][:]  # [N, T, D]
                N, T, D = features.shape

                if verbose:
                    print(f"  Loaded temporal features: {N} samples x {T} timesteps x {D} features")
                    print(f"  Flattening to {N*T} samples x {D} features for analysis")

                # Flatten temporal dimension for comprehensive analysis
                features = features.reshape(N * T, D)
                total_timesteps = T

            elif family == "summary_aggregated":
                if "/features/summary/aggregated/features" not in f:
                    raise ValueError(f"Dataset does not contain summary/aggregated features")

                features = f["/features/summary/aggregated/features"][:]  # [N, D]
                N, D = features.shape

                if verbose:
                    print(f"  Loaded summary/aggregated features: {N} samples x {D} features")

                total_timesteps = None

            elif family == "summary_per_trajectory":
                if "/features/summary/per_trajectory/features" not in f:
                    raise ValueError(f"Dataset does not contain summary/per_trajectory features")

                features = f["/features/summary/per_trajectory/features"][:]  # [N, M, D]
                N, M, D = features.shape

                if verbose:
                    print(f"  Loaded summary/per_trajectory features: {N} samples x {M} realizations x {D} features")
                    print(f"  Flattening to {N*M} samples x {D} features for analysis")

                # Flatten realization dimension
                features = features.reshape(N * M, D)
                total_timesteps = None

            else:
                raise ValueError(f"Unknown feature family: {family}")

            # Load feature registry for names
            registry_key = family.split('_')[0]  # 'temporal' or 'summary'
            registry_attr_key = f"/features/{registry_key}"

            if registry_attr_key in f and "feature_registry" in f[registry_attr_key].attrs:
                registry_json = f[registry_attr_key].attrs["feature_registry"]
                registry = FeatureRegistry.from_json(registry_json, family)
                feature_names = [feat.name for feat in registry.get_all_features()]

                if verbose:
                    print(f"  Loaded feature names from registry")
            else:
                feature_names = [f"feature_{i}" for i in range(D)]

                if verbose:
                    print(f"  No registry found, using default feature names")

            # Create metadata
            metadata = AnalysisMetadata(
                dataset_path=str(dataset_path),
                feature_family=family,
                analysis_timestamp="",  # Will be set by analyzer
                total_samples=N if family == "temporal" or family == "summary_aggregated" else N,
                total_timesteps=total_timesteps,
                total_features=D,
                thresholds={}  # Will be set by caller
            )

        return features, feature_names, metadata

    def _print_summary(self, results: Dict, verbose: bool) -> None:
        """Print analysis summary.

        Args:
            results: Analysis results dictionary
            verbose: Print additional details
        """
        summary = results["summary"]
        metadata = results["metadata"]

        print("\n" + "="*70)
        print("FEATURE ANALYSIS SUMMARY")
        print("="*70)
        print(f"Dataset:                    {Path(metadata['dataset_path']).name}")
        print(f"Feature family:             {metadata['feature_family']}")
        print(f"Total features analyzed:    {summary['features_analyzed']}")
        print()
        print(f"Zero variance features:     {summary['zero_variance_features']}")
        print(f"High NaN features:          {summary['high_nan_features']}")
        print(f"Redundant features:         {summary['redundant_features']}")
        print()
        print(f"Recommended to KEEP:        {summary['recommended_keep']}")
        print(f"Recommended to PRUNE:       {summary['recommended_prune']}")
        print(f"Compression ratio:          {summary['compression_ratio']:.2%}")
        print("="*70)

        if verbose and summary['recommended_prune'] > 0:
            print("\nPrune reasons breakdown:")
            prune_reasons = results["recommendations"]["prune_reasons"]
            reason_counts = {}
            for reason in prune_reasons.values():
                reason_type = reason.split('_with_')[0] if 'redundant' in reason else reason
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1

            for reason, count in sorted(reason_counts.items()):
                print(f"  {reason}: {count}")

    def _save_results(
        self,
        results: Dict,
        output_path: Path,
        format: str
    ) -> None:
        """Save analysis results to file.

        Args:
            results: Analysis results dictionary
            output_path: Output file path
            format: Output format ('json' or 'yaml')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
        elif format == "yaml":
            import yaml
            with open(output_path, "w") as f:
                yaml.dump(results, f, default_flow_style=False)

        print(f"\nAnalysis saved to: {output_path}")

    def _save_pruned_registry(
        self,
        results: Dict,
        feature_names: List[str],
        family: str,
        output_path: Path,
        verbose: bool
    ) -> None:
        """Save pruned feature registry.

        Args:
            results: Analysis results dictionary
            feature_names: Original feature names
            family: Feature family name
            output_path: Base output path for naming
            verbose: Print progress
        """
        keep_indices = results["recommendations"]["keep_indices"]

        # Create new registry with only kept features
        pruned_registry = FeatureRegistry(family)

        for new_idx, old_idx in enumerate(keep_indices):
            feature_name = feature_names[old_idx]
            # Register with generic category (we don't have original category info)
            pruned_registry.register(
                name=feature_name,
                category="selected",
                description=f"Kept feature (original index {old_idx})"
            )

        # Save to JSON
        registry_path = output_path.parent / f"{output_path.stem}_pruned_registry.json"
        registry_json = pruned_registry.to_json()

        with open(registry_path, "w") as f:
            f.write(registry_json)

        if verbose:
            print(f"Pruned registry saved to: {registry_path}")

    def _generate_visualizations(
        self,
        results: Dict,
        output_path: Path,
        verbose: bool
    ) -> None:
        """Generate visualization plots.

        Args:
            results: Analysis results dictionary
            output_path: Base output path for naming
            verbose: Print progress
        """
        try:
            from spinlock.visualization.features.analyzer_plots import (
                plot_variance_distribution,
                plot_correlation_heatmap,
                plot_nan_frequency
            )

            if verbose:
                print("\nGenerating visualizations...")

            output_dir = output_path.parent / f"{output_path.stem}_plots"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Variance distribution
            variances = np.array(results["analysis"]["variance"]["per_feature_variance"])
            var_threshold = results["metadata"]["thresholds"]["variance"]
            plot_variance_distribution(
                variances,
                var_threshold,
                output_dir / "variance_distribution.png"
            )
            if verbose:
                print(f"  Saved: variance_distribution.png")

            # Correlation heatmap (sample if too many features)
            corr_matrix = np.array(results["analysis"]["correlation"]["correlation_matrix"])
            feature_names = [f"f{i}" for i in range(corr_matrix.shape[0])]
            plot_correlation_heatmap(
                corr_matrix,
                feature_names,
                output_dir / "correlation_heatmap.png"
            )
            if verbose:
                print(f"  Saved: correlation_heatmap.png")

            # NaN frequency
            nan_fractions = np.array(results["analysis"]["nan_analysis"]["per_feature_nan_fraction"])
            nan_threshold = results["metadata"]["thresholds"]["nan"]
            plot_nan_frequency(
                nan_fractions,
                nan_threshold,
                output_dir / "nan_frequency.png"
            )
            if verbose:
                print(f"  Saved: nan_frequency.png")

            print(f"\nVisualizations saved to: {output_dir}/")

        except ImportError as e:
            print(f"\nWarning: Could not generate visualizations: {e}", file=sys.stderr)
            if "sklearn" in str(e):
                print("Note: Visualization currently requires sklearn due to package imports.", file=sys.stderr)
                print("This is a known limitation and will be fixed in a future update.", file=sys.stderr)
            else:
                print("Install visualization dependencies: pip install matplotlib seaborn", file=sys.stderr)

    def _default_output_path(self, dataset_path: Path) -> Path:
        """Generate default output path.

        Args:
            dataset_path: Input dataset path

        Returns:
            Default output path
        """
        return dataset_path.parent / f"{dataset_path.stem}_feature_analysis.json"
