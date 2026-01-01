"""
Visualize IC types command for Spinlock CLI.

Generates and visualizes all available IC types with multiple variations
to show the diversity of initial conditions available in the system.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np

from .base import CLICommand


# All available IC types in the system
ALL_IC_TYPES = [
    # Baseline ICs
    "gaussian_random_field",
    "structured",
    "mixed",
    "multiscale_grf",
    "localized",
    "composite",
    "heavy_tailed",
    # Tier 1: Foundational physics/biology patterns
    "quantum_wave_packet",
    "turing_pattern",
    "thermal_gradient",
    "morphogen_gradient",
    "reaction_front",
    # Tier 2: Specialized scientific domains
    "light_cone",
    "critical_fluctuation",
    "phase_boundary",
    "bz_reaction",
    "shannon_entropy",
    # Tier 3: Complex systems and biology
    "interference_pattern",
    "cell_population",
    "chromatin_domain",
    "shock_front",
    "gene_expression",
    # Tier 4: Research frontiers
    "coherent_state",
    "relativistic_wave_packet",
    "mutual_information",
    "regulatory_network",
    "dla_cluster",
    "error_correcting_code",
]

# Organize by tier for better visualization
IC_TIERS = {
    "Baseline": [
        "gaussian_random_field",
        "structured",
        "mixed",
        "multiscale_grf",
        "localized",
        "composite",
        "heavy_tailed",
    ],
    "Tier 1: Foundational": [
        "quantum_wave_packet",
        "turing_pattern",
        "thermal_gradient",
        "morphogen_gradient",
        "reaction_front",
    ],
    "Tier 2: Specialized": [
        "light_cone",
        "critical_fluctuation",
        "phase_boundary",
        "bz_reaction",
        "shannon_entropy",
    ],
    "Tier 3: Complex Systems": [
        "interference_pattern",
        "cell_population",
        "chromatin_domain",
        "shock_front",
        "gene_expression",
    ],
    "Tier 4: Research Frontiers": [
        "coherent_state",
        "relativistic_wave_packet",
        "mutual_information",
        "regulatory_network",
        "dla_cluster",
        "error_correcting_code",
    ],
}


class VisualizeICTypesCommand(CLICommand):
    """
    Command to visualize all available IC types.

    Generates multiple variations of each IC type and displays them
    in a grid layout for easy comparison and understanding of the
    diversity of initial conditions available.
    """

    @property
    def name(self) -> str:
        return "visualize-ic-types"

    @property
    def help(self) -> str:
        return "Visualize all available IC types with multiple variations"

    @property
    def description(self) -> str:
        return """
Visualize all available initial condition (IC) types in the system.

Shows multiple stratified variations of each IC type to demonstrate
the diversity and characteristics of different initial conditions.

Layout:
  - One column per tier
  - Each IC type shown with N variations (rows)
  - RGB visualization with configurable colormap

Examples:
  # Basic visualization (3 variations per type)
  spinlock visualize-ic-types --output ic_types.png

  # More variations per type
  spinlock visualize-ic-types --variations 5 --output ic_types.png

  # Specific tiers only
  spinlock visualize-ic-types --tiers baseline tier1 --output baseline_tier1.png

  # Custom grid size
  spinlock visualize-ic-types --size 64x64 --output ic_types_small.png
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add command arguments."""
        # Required arguments
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            metavar="PATH",
            help="Output image path (.png, .jpg, etc.)",
        )

        # IC selection
        ic_group = parser.add_argument_group("IC selection")

        ic_group.add_argument(
            "--tiers",
            type=str,
            nargs="+",
            choices=["baseline", "tier1", "tier2", "tier3", "tier4", "all"],
            default=["all"],
            help="Which tiers to visualize (default: all)",
        )

        ic_group.add_argument(
            "--ic-types",
            type=str,
            nargs="+",
            metavar="TYPE",
            help="Specific IC types to visualize (overrides --tiers)",
        )

        # Generation parameters
        gen_group = parser.add_argument_group("generation parameters")

        gen_group.add_argument(
            "--variations",
            type=int,
            default=3,
            metavar="N",
            help="Number of variations per IC type (default: 3)",
        )

        gen_group.add_argument(
            "--grid-size",
            type=int,
            default=128,
            metavar="SIZE",
            help="IC grid resolution (default: 128)",
        )

        gen_group.add_argument(
            "--num-channels",
            type=int,
            default=3,
            metavar="C",
            help="Number of channels (default: 3)",
        )

        gen_group.add_argument(
            "--seed",
            type=int,
            default=42,
            metavar="SEED",
            help="Random seed (default: 42)",
        )

        # Rendering configuration
        render_group = parser.add_argument_group("rendering configuration")

        render_group.add_argument(
            "--cell-size",
            type=str,
            default="128x128",
            metavar="HxW",
            help="Size of each cell in output image (default: 128x128)",
        )

        render_group.add_argument(
            "--colormap",
            type=str,
            default="viridis",
            help="Colormap for rendering (default: viridis)",
        )

        render_group.add_argument(
            "--add-labels",
            action="store_true",
            help="Add IC type labels to visualization",
        )

        render_group.add_argument(
            "--add-spacing",
            action="store_true",
            help="Add white spacing between cells",
        )

        # Execution options
        exec_group = parser.add_argument_group("execution options")

        exec_group.add_argument(
            "--device",
            type=str,
            default="cuda",
            metavar="DEVICE",
            help="Torch device (default: cuda)",
        )

        exec_group.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed progress information",
        )

    def execute(self, args: Namespace) -> int:
        """Execute IC type visualization."""
        # Parse cell size
        try:
            cell_h, cell_w = map(int, args.cell_size.split("x"))
        except ValueError:
            return self.error(f"Invalid cell size format: {args.cell_size}. Use HxW (e.g., 64x64)")

        # Determine which IC types to visualize
        ic_types_to_viz = self._select_ic_types(args)

        if not ic_types_to_viz:
            return self.error("No IC types selected for visualization")

        if args.verbose:
            print("="*60)
            print("SPINLOCK IC TYPE VISUALIZATION")
            print("="*60)
            print(f"IC types: {len(ic_types_to_viz)}")
            print(f"Variations per type: {args.variations}")
            print(f"Grid size: {args.grid_size}×{args.grid_size}")
            print(f"Cell size: {cell_h}×{cell_w}")
            print(f"Output: {args.output}")
            print("="*60 + "\n")

        # Generate and visualize
        try:
            return self._visualize_ic_types(
                ic_types=ic_types_to_viz,
                num_variations=args.variations,
                grid_size=args.grid_size,
                num_channels=args.num_channels,
                cell_size=(cell_h, cell_w),
                colormap=args.colormap,
                add_labels=args.add_labels,
                add_spacing=args.add_spacing,
                seed=args.seed,
                device=args.device,
                output_path=args.output,
                verbose=args.verbose,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.error(f"Visualization failed: {e}")

    def _select_ic_types(self, args: Namespace) -> List[str]:
        """Determine which IC types to visualize based on args."""
        if args.ic_types:
            # Explicit IC types specified
            return args.ic_types

        # Select by tier
        if "all" in args.tiers:
            return ALL_IC_TYPES

        selected = []
        tier_map = {
            "baseline": "Baseline",
            "tier1": "Tier 1: Foundational",
            "tier2": "Tier 2: Specialized",
            "tier3": "Tier 3: Complex Systems",
            "tier4": "Tier 4: Research Frontiers",
        }

        for tier_arg in args.tiers:
            tier_name = tier_map.get(tier_arg)
            if tier_name and tier_name in IC_TIERS:
                selected.extend(IC_TIERS[tier_name])

        return selected

    def _visualize_ic_types(
        self,
        ic_types: List[str],
        num_variations: int,
        grid_size: int,
        num_channels: int,
        cell_size: Tuple[int, int],
        colormap: str,
        add_labels: bool,
        add_spacing: bool,
        seed: int,
        device: str,
        output_path: Path,
        verbose: bool,
    ) -> int:
        """Generate ICs and create visualization grid."""
        from spinlock.dataset.generators import InputFieldGenerator
        from spinlock.visualization import create_render_strategy
        from PIL import Image
        import torchvision.transforms.functional as TF

        torch_device = torch.device(device)

        if verbose:
            print("Generating initial conditions...")

        # Initialize generator
        generator = InputFieldGenerator(
            grid_size=grid_size,
            num_channels=num_channels,
            device=torch_device
        )

        # Generate all ICs
        all_ics = {}  # ic_type -> [variations, C, H, W]

        for ic_idx, ic_type in enumerate(ic_types):
            if verbose:
                print(f"  [{ic_idx+1}/{len(ic_types)}] {ic_type}...", end=" ", flush=True)

            variations = []
            for v in range(num_variations):
                ic_seed = seed + ic_idx * 1000 + v
                ic = generator.generate_batch(
                    batch_size=1,
                    field_type=ic_type,  # type: ignore[arg-type]
                    seed=ic_seed
                )
                variations.append(ic[0])  # Remove batch dimension

            all_ics[ic_type] = torch.stack(variations, dim=0)  # [variations, C, H, W]

            if verbose:
                print(f"✓ {num_variations} variations")

        if verbose:
            print(f"\nRendering {len(ic_types)} × {num_variations} = {len(ic_types) * num_variations} cells...")

        # Create renderer
        renderer = create_render_strategy(
            num_channels=num_channels,
            strategy="auto",
            colormap=colormap,
            device=torch_device
        )

        # Calculate grid dimensions
        cell_h, cell_w = cell_size
        spacing = 2 if add_spacing else 0
        label_height = 30 if add_labels else 0

        num_rows = len(ic_types)
        num_cols = num_variations

        grid_h = num_rows * (cell_h + label_height) + (num_rows - 1) * spacing
        grid_w = num_cols * cell_w + (num_cols - 1) * spacing

        # Create canvas
        grid = torch.ones(3, grid_h, grid_w, dtype=torch.float32, device=torch_device)

        # Render each IC type (row)
        for row, ic_type in enumerate(ic_types):
            ics = all_ics[ic_type]  # [variations, C, H, W]

            row_start = row * (cell_h + label_height + spacing)

            # Render variations (columns)
            for col in range(num_variations):
                ic = ics[col:col+1]  # [1, C, H, W]
                rgb = renderer.render(ic)  # [1, 3, H_orig, W_orig]

                # Resize to cell size
                if rgb.shape[-2:] != (cell_h, cell_w):
                    rgb = torch.nn.functional.interpolate(
                        rgb, size=(cell_h, cell_w), mode='bilinear', align_corners=False
                    )

                col_start = col * (cell_w + spacing)
                col_end = col_start + cell_w
                row_end = row_start + cell_h

                grid[:, row_start:row_end, col_start:col_end] = rgb[0]

        # Convert to PIL and save
        grid_np = (grid.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(grid_np)

        # Add labels if requested
        if add_labels:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except:
                font = ImageFont.load_default()

            for row, ic_type in enumerate(ic_types):
                y_pos = row * (cell_h + label_height + spacing) + cell_h + 5
                # Draw label centered below the row
                label = ic_type.replace("_", " ").title()
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                x_pos = (grid_w - text_width) // 2
                draw.text((x_pos, y_pos), label, fill=(0, 0, 0), font=font)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

        if verbose:
            print(f"\n✓ Visualization saved: {output_path}")
            print(f"  Grid: {num_rows} IC types × {num_variations} variations")
            print(f"  Image size: {grid_w}×{grid_h}")

        return 0
