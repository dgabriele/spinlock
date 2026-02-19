"""CLI command for VQ-VAE visualization.

Creates engineering and semantic dashboards for trained VQ-VAE models.

Usage:
    spinlock visualize-vqvae --checkpoint checkpoints/production/100k_full_features/ \
        --output visualizations/ --type both
"""

import argparse
from pathlib import Path
from typing import Optional

from .base import CLICommand


class VisualizeVQVAECommand(CLICommand):
    """Generate visualizations for trained VQ-VAE models."""

    @property
    def name(self) -> str:
        return "visualize-vqvae"

    @property
    def help(self) -> str:
        return "Create engineering and semantic dashboards for VQ-VAE models"

    @property
    def description(self) -> str:
        return """Create comprehensive visualizations for trained VQ-VAE models.

Four dashboard types are available:
  - engineering: Model architecture, training curves, utilization heatmap
  - topological: t-SNE codebook embeddings, usage heatmap, similarity matrix
  - semantic: Feature-category mapping, category profiles, correlation matrix
  - roundtrip: Roundtrip consistency, semantic structure, compositional analysis
  - hierarchical: Hierarchical token pattern analysis (requires --tokenized-dataset)

Examples:
  spinlock visualize-vqvae --checkpoint checkpoints/production/100k_full_features/
  spinlock visualize-vqvae --checkpoint checkpoints/my_model/ --type roundtrip
  spinlock visualize-vqvae --checkpoint checkpoints/my_model/ --type all --tokenized-dataset datasets/50k_tokenized.h5
  spinlock visualize-vqvae --checkpoint checkpoints/my_model/ --type hierarchical --tokenized-dataset datasets/50k_tokenized.h5
"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="Path to checkpoint directory (containing final_model.pt)",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="visualizations",
            help="Output directory for visualizations (default: visualizations/)",
        )
        parser.add_argument(
            "--type",
            type=str,
            choices=[
                "engineering", "topological", "semantic", "roundtrip", "hierarchical",
                "binned-similarity", "physics-alignment", "all",
            ],
            default="topological",
            help="Type of visualization to create (default: topological)",
        )
        parser.add_argument(
            "--dpi",
            type=int,
            default=150,
            help="Resolution for saved figures (default: 150)",
        )
        parser.add_argument(
            "--no-display",
            action="store_true",
            help="Don't display figures, only save",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default=None,
            help="Path to dataset HDF5 for topographic metrics (optional, auto-detected from config)",
        )
        parser.add_argument(
            "--no-topo",
            action="store_true",
            help="Skip topographic metrics computation (faster but less complete)",
        )
        parser.add_argument(
            "--topo-samples",
            type=int,
            default=1000,
            help="Number of samples for topographic metrics (default: 1000)",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Device for topographic computation (default: cuda)",
        )
        parser.add_argument(
            "--tokenized-dataset",
            type=str,
            default=None,
            help="Path to pretokenized dataset HDF5 for token frequency analysis (optional, for roundtrip dashboard)",
        )
        parser.add_argument(
            "--family",
            type=str,
            default="all",
            choices=["temporal", "initial", "theta", "all"],
            help="Feature family to analyze for hierarchical visualization. Use 'all' to analyze all families (default: all)",
        )
        parser.add_argument(
            "--max-rollouts",
            type=int,
            default=5000,
            help="Maximum number of rollouts to analyze (default: 5000)",
        )
        parser.add_argument(
            "--similarity-metric",
            type=str,
            default="jaccard",
            choices=["jaccard", "cosine"],
            help="Similarity metric for hierarchical analysis (default: jaccard)",
        )
        parser.add_argument(
            "--metric",
            type=str,
            default="jaccard",
            choices=["jaccard", "js"],
            help="Similarity metric for binned-similarity dashboard (default: jaccard)",
        )
        parser.add_argument(
            "--n-bins",
            type=int,
            default=5000,
            help="Number of bins for binned-similarity dashboard (default: 5000)",
        )
        parser.add_argument(
            "--families",
            type=str,
            nargs="+",
            default=None,
            metavar="FAMILY",
            help="Token families to include in binned-similarity (e.g. temporal initial theta). Default: all",
        )
        parser.add_argument(
            "--params",
            type=str,
            default=None,
            help="Path to HDF5 with physical parameters for physics-alignment dashboard",
        )
        parser.add_argument(
            "--subsample",
            type=int,
            default=5000,
            help="Max samples for discrimination ratio computation (default: 5000)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the visualization command."""
        from spinlock.visualization.vqvae import (
            create_engineering_dashboard,
            create_semantic_dashboard,
            create_topological_dashboard,
            create_roundtrip_dashboard,
        )
        from spinlock.visualization.vqvae.roundtrip_dashboard import (
            generate_hierarchical_pattern_analysis,
        )
        import matplotlib
        import matplotlib.pyplot as plt

        checkpoint_path = Path(args.checkpoint)
        output_dir = Path(args.output)

        # Validate checkpoint exists
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint directory not found: {checkpoint_path}")
            return 1

        # Check for model files (support both naming conventions)
        has_model = (
            (checkpoint_path / "final_model.pt").exists()
            or (checkpoint_path / "best_model.pt").exists()
            or (checkpoint_path / "vq_tokenizer_final.pt").exists()
            or (checkpoint_path / "vq_tokenizer_best.pt").exists()
        )
        if not has_model:
            print(f"Error: No model checkpoint found in {checkpoint_path}")
            print(
                "Expected: final_model.pt, best_model.pt, vq_tokenizer_final.pt, or vq_tokenizer_best.pt"
            )
            return 1

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Disable display if requested
        if args.no_display:
            matplotlib.use("Agg")

        checkpoint_name = checkpoint_path.name
        print(f"Creating VQ-VAE visualizations for: {checkpoint_name}")
        print(f"Output directory: {output_dir}")
        print()

        # Create requested visualizations
        try:
            if args.type in ["binned-similarity", "all"]:
                if not args.tokenized_dataset:
                    print("Error: --tokenized-dataset is required for binned-similarity")
                    return 1
                from spinlock.visualization.vqvae import create_binned_similarity_dashboard
                output_path = output_dir / f"binned_similarity_{args.metric}.png"
                print(f"Creating binned-similarity dashboard ({args.metric}, {args.n_bins} bins)...")
                fig = create_binned_similarity_dashboard(
                    tokenized_h5_path=args.tokenized_dataset,
                    output_path=output_path,
                    n_bins=args.n_bins,
                    metric=args.metric,
                    families=args.families,
                    dpi=args.dpi,
                )
                if not args.no_display:
                    plt.show(block=False)
                plt.close(fig)

            if args.type in ["physics-alignment", "all"]:
                if not args.tokenized_dataset:
                    print("Error: --tokenized-dataset is required for physics-alignment")
                    return 1
                if not args.params:
                    print("Error: --params is required for physics-alignment")
                    return 1
                from spinlock.visualization.vqvae import create_physics_alignment_dashboard
                output_path = output_dir / "physics_alignment.png"
                print(f"Creating physics-alignment dashboard...")
                fig = create_physics_alignment_dashboard(
                    tokens_h5_path=args.tokenized_dataset,
                    params_h5_path=args.params,
                    output_path=output_path,
                    subsample=args.subsample,
                    dpi=args.dpi,
                )
                if not args.no_display:
                    plt.show(block=False)
                plt.close(fig)

            if args.type in ["engineering", "all"]:
                output_path = output_dir / f"{checkpoint_name}_engineering.png"
                print(f"Creating engineering dashboard...")
                fig = create_engineering_dashboard(
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    dpi=args.dpi,
                )
                if not args.no_display:
                    plt.show(block=False)
                plt.close(fig)

            if args.type in ["topological", "all"]:
                output_path = output_dir / f"{checkpoint_name}_topological.png"
                print(f"Creating topological dashboard...")
                fig = create_topological_dashboard(
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    dpi=args.dpi,
                )
                if not args.no_display:
                    plt.show(block=False)
                plt.close(fig)

            if args.type in ["semantic", "all"]:
                output_path = output_dir / f"{checkpoint_name}_semantic.png"
                print(f"Creating semantic dashboard...")
                fig = create_semantic_dashboard(
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    dataset_path=args.dataset,
                    compute_topo=not args.no_topo,
                    topo_samples=args.topo_samples,
                    device=args.device,
                    dpi=args.dpi,
                )
                if not args.no_display:
                    plt.show(block=False)
                plt.close(fig)

            if args.type in ["roundtrip", "all"]:
                output_path = output_dir / f"{checkpoint_name}_roundtrip.png"
                print(f"Creating roundtrip dashboard...")
                fig = create_roundtrip_dashboard(
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    tokenized_dataset_path=Path(args.tokenized_dataset) if args.tokenized_dataset else None,
                    dpi=args.dpi,
                )
                if not args.no_display:
                    plt.show(block=False)
                plt.close(fig)

            if args.type in ["hierarchical", "all"]:
                if not args.tokenized_dataset:
                    print("Error: --tokenized-dataset is required for hierarchical pattern analysis")
                    return 1

                hierarchical_output_dir = output_dir / f"{checkpoint_name}_hierarchical"
                print(f"Creating hierarchical pattern analysis...")
                generate_hierarchical_pattern_analysis(
                    checkpoint_path=checkpoint_path,
                    tokenized_dataset_path=Path(args.tokenized_dataset),
                    output_dir=hierarchical_output_dir,
                    family=args.family,
                    max_rollouts=args.max_rollouts,
                    similarity_metric=args.similarity_metric,
                )

            print()
            print("Visualization complete!")
            return 0

        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
            return 1
