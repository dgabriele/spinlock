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

Three dashboard types are available:
  - engineering: Model architecture, training curves, utilization heatmap
  - topological: t-SNE codebook embeddings, usage heatmap, similarity matrix
  - semantic: Feature-category mapping, category profiles, correlation matrix

Examples:
  spinlock visualize-vqvae --checkpoint checkpoints/production/100k_full_features/
  spinlock visualize-vqvae --checkpoint checkpoints/my_model/ --type topological
  spinlock visualize-vqvae --checkpoint checkpoints/my_model/ --type all
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
            choices=["engineering", "topological", "semantic", "all"],
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

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the visualization command."""
        from spinlock.visualization.vqvae import (
            create_engineering_dashboard,
            create_semantic_dashboard,
            create_topological_dashboard,
        )
        import matplotlib
        import matplotlib.pyplot as plt

        checkpoint_path = Path(args.checkpoint)
        output_dir = Path(args.output)

        # Validate checkpoint exists
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint directory not found: {checkpoint_path}")
            return 1

        # Check for model files
        has_model = (checkpoint_path / "final_model.pt").exists() or (
            checkpoint_path / "best_model.pt"
        ).exists()
        if not has_model:
            print(f"Error: No model checkpoint found in {checkpoint_path}")
            print("Expected: final_model.pt or best_model.pt")
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
                    dpi=args.dpi,
                )
                if not args.no_display:
                    plt.show(block=False)
                plt.close(fig)

            print()
            print("Visualization complete!")
            return 0

        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
            return 1
