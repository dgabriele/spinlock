"""
Visualize command for Spinlock CLI.

Handles temporal evolution visualization of dataset operators with grid rendering.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import time
import torch
import numpy as np

from .base import CLICommand


class VisualizeCommand(CLICommand):
    """
    Command to visualize temporal evolution of dataset operators.

    Loads operators from dataset, evolves them through time, and renders
    multi-operator multi-realization grids as video or image sequence.
    """

    @property
    def name(self) -> str:
        return "visualize-dataset"

    @property
    def help(self) -> str:
        return "Visualize temporal evolution of dataset operators"

    @property
    def description(self) -> str:
        return """
Visualize temporal evolution of operators from a dataset.

Creates grid visualizations showing N operators (rows) × M realizations + aggregates (columns).
Renders temporal dynamics as video (MP4) or image sequence (PNG frames).

Examples:
  # Basic visualization (10 operators, 10 realizations, 500 steps)
  spinlock visualize-dataset --dataset datasets/benchmark_10k.h5 \\
      --output visualizations/evolution.mp4

  # Custom grid size and timesteps
  spinlock visualize-dataset --dataset datasets/benchmark_10k.h5 \\
      --output visualizations/evolution.mp4 \\
      --n-operators 5 \\
      --n-realizations 20 \\
      --steps 1000 \\
      --size 128x128

  # Export as image sequence
  spinlock visualize-dataset --dataset datasets/benchmark_10k.h5 \\
      --output visualizations/frames/ \\
      --format frames

  # Dry run to validate parameters
  spinlock visualize-dataset --dataset datasets/benchmark_10k.h5 \\
      --dry-run --verbose
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add visualize command arguments."""
        # Required arguments
        parser.add_argument(
            "--dataset",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to HDF5 dataset",
        )

        # Output configuration
        output_group = parser.add_argument_group("output configuration")

        output_group.add_argument(
            "--output",
            type=Path,
            metavar="PATH",
            help="Output path (.mp4 for video, directory for frames)",
        )

        output_group.add_argument(
            "--format",
            type=str,
            choices=["video", "frames", "both"],
            default="video",
            help="Export format (default: video)",
        )

        output_group.add_argument(
            "--fps",
            type=int,
            default=30,
            metavar="N",
            help="Frames per second for video export (default: 30)",
        )

        # Visualization parameters
        vis_group = parser.add_argument_group("visualization parameters")

        vis_group.add_argument(
            "--n-operators",
            type=int,
            default=10,
            metavar="N",
            help="Number of operators to visualize (default: 10)",
        )

        vis_group.add_argument(
            "--n-realizations",
            type=int,
            default=None,
            metavar="M",
            help="Number of stochastic realizations per operator (default: use dataset value)",
        )

        vis_group.add_argument(
            "--steps",
            type=int,
            default=500,
            metavar="T",
            help="Number of evolution timesteps (default: 500)",
        )

        vis_group.add_argument(
            "--size",
            type=str,
            default="64x64",
            metavar="HxW",
            help="Grid cell size (default: 64x64)",
        )

        vis_group.add_argument(
            "--stride",
            type=int,
            default=1,
            metavar="N",
            help="Render every Nth timestep (default: 1)",
        )

        # Rendering configuration
        render_group = parser.add_argument_group("rendering configuration")

        render_group.add_argument(
            "--colormap",
            type=str,
            default="viridis",
            help="Colormap for single-channel visualization (default: viridis)",
        )

        render_group.add_argument(
            "--aggregates",
            type=str,
            nargs="+",
            choices=["mean", "variance", "stddev"],
            default=["mean", "variance"],
            help="Aggregate renderers to include (default: mean variance)",
        )

        render_group.add_argument(
            "--add-spacing",
            action="store_true",
            help="Add white spacing between grid cells",
        )

        # Sampling configuration
        sample_group = parser.add_argument_group("operator sampling")

        sample_group.add_argument(
            "--sampling-method",
            type=str,
            choices=["sobol", "random", "sequential"],
            default="sobol",
            help="Operator sampling method (default: sobol)",
        )

        sample_group.add_argument(
            "--operator-indices",
            type=int,
            nargs="+",
            metavar="IDX",
            help="Explicit operator indices to visualize (overrides --n-operators)",
        )

        sample_group.add_argument(
            "--seed",
            type=int,
            default=42,
            metavar="SEED",
            help="Random seed for operator sampling (default: 42)",
        )

        # Evolution configuration
        evolution_group = parser.add_argument_group("evolution configuration")

        evolution_group.add_argument(
            "--normalization",
            type=str,
            choices=["none", "minmax", "zscore"],
            default="none",
            help="Normalization method for evolution (default: none - preserves raw dynamics for downstream tasks)",
        )

        evolution_group.add_argument(
            "--clamp-min",
            type=float,
            metavar="VALUE",
            help="Minimum clamp value for evolution",
        )

        evolution_group.add_argument(
            "--clamp-max",
            type=float,
            metavar="VALUE",
            help="Maximum clamp value for evolution",
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
            "--dry-run",
            action="store_true",
            help="Validate configuration without running visualization",
        )

        exec_group.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed progress information",
        )

    def execute(self, args: Namespace) -> int:
        """Execute visualization pipeline."""
        # Validate dataset exists
        if not self.validate_file_exists(args.dataset, "Dataset"):
            return 1

        # Parse grid size
        try:
            grid_size = self._parse_grid_size(args.size)
        except ValueError as e:
            return self.error(str(e))

        # Determine output paths
        output_video = None
        output_frames = None

        if args.output:
            if args.format == "video" or (args.format == "both"):
                output_video = args.output if args.output.suffix == ".mp4" else args.output / "evolution.mp4"
            if args.format == "frames" or (args.format == "both"):
                output_frames = args.output if args.output.is_dir() or not args.output.suffix else args.output.parent / "frames"
        else:
            # Default output
            output_video = Path("visualizations/evolution.mp4")

        # Print configuration summary
        if args.verbose:
            self._print_config_summary(args, grid_size, output_video, output_frames)

        # Dry run: validate and exit
        if args.dry_run:
            print("\n✓ Configuration valid (dry-run mode, no visualization generated)")
            return 0

        # Execute visualization pipeline
        try:
            return self._run_visualization(
                dataset_path=args.dataset,
                output_video=output_video,
                output_frames=output_frames,
                n_operators=args.n_operators,
                n_realizations=args.n_realizations,
                num_timesteps=args.steps,
                grid_size=grid_size,
                stride=args.stride,
                colormap=args.colormap,
                aggregates=args.aggregates,
                add_spacing=args.add_spacing,
                sampling_method=args.sampling_method,
                operator_indices=args.operator_indices,
                seed=args.seed,
                normalization=args.normalization if args.normalization != "none" else None,
                clamp_min=args.clamp_min,
                clamp_max=args.clamp_max,
                device=args.device,
                fps=args.fps,
                verbose=args.verbose,
            )
        except KeyboardInterrupt:
            print("\n\nVisualization interrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            import traceback
            print(f"\nError during visualization: {e}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 1

    def _parse_grid_size(self, size_str: str) -> int:
        """
        Parse grid size string (e.g., '64x64' -> 64).

        Args:
            size_str: Size string (HxW)

        Returns:
            Grid size (H)

        Raises:
            ValueError: If size string invalid or H != W
        """
        try:
            parts = size_str.split('x')
            if len(parts) != 2:
                raise ValueError(f"Invalid size format: {size_str}. Expected HxW (e.g., '64x64')")

            h, w = int(parts[0]), int(parts[1])

            if h != w:
                raise ValueError(f"Grid must be square (H=W), got {h}x{w}")

            if h <= 0:
                raise ValueError(f"Grid size must be positive, got {h}")

            return h

        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid size format: {size_str}. Expected HxW (e.g., '64x64')")
            raise

    def _print_config_summary(
        self,
        args: Namespace,
        grid_size: int,
        output_video: Optional[Path],
        output_frames: Optional[Path]
    ) -> None:
        """Print configuration summary."""
        print("\n" + "="*60)
        print("VISUALIZATION CONFIGURATION")
        print("="*60)

        print(f"\nDataset: {args.dataset}")
        print(f"\nGrid Layout:")
        print(f"  Operators:     {args.n_operators}")
        print(f"  Realizations:  {args.n_realizations}")
        print(f"  Aggregates:    {', '.join(args.aggregates)}")
        print(f"  Cell size:     {grid_size}×{grid_size}")
        print(f"  Spacing:       {'enabled' if args.add_spacing else 'disabled'}")

        print(f"\nEvolution:")
        print(f"  Policy:        per-operator (from dataset)")
        print(f"  Timesteps:     {args.steps}")
        print(f"  Stride:        {args.stride}")
        print(f"  Normalization: {args.normalization}")
        if args.clamp_min is not None or args.clamp_max is not None:
            print(f"  Clamp range:   [{args.clamp_min}, {args.clamp_max}]")

        print(f"\nRendering:")
        print(f"  Colormap:      {args.colormap}")
        print(f"  Device:        {args.device}")

        print(f"\nOutput:")
        if output_video:
            print(f"  Video:         {output_video} ({args.fps} fps)")
        if output_frames:
            print(f"  Frames:        {output_frames}/")

        print("="*60 + "\n")

    def _run_visualization(
        self,
        dataset_path: Path,
        output_video: Optional[Path],
        output_frames: Optional[Path],
        n_operators: int,
        n_realizations: Optional[int],
        num_timesteps: int,
        grid_size: int,
        stride: int,
        colormap: str,
        aggregates: List[str],
        add_spacing: bool,
        sampling_method: str,
        operator_indices: Optional[List[int]],
        seed: int,
        normalization: Optional[str],
        clamp_min: Optional[float],
        clamp_max: Optional[float],
        device: str,
        fps: int,
        verbose: bool,
    ) -> int:
        """
        Run visualization pipeline.

        Pipeline:
        1. Load dataset and sample operators
        2. Build operators from parameters
        3. Run temporal evolution (OperatorRollout)
        4. Render frames (VisualizationGrid)
        5. Export (VideoExporter / ImageSequenceExporter)
        """
        from spinlock.dataset.storage import HDF5DatasetReader
        from spinlock.operators.builder import OperatorBuilder
        from spinlock.operators.parameters import OperatorParameters
        from spinlock.evolution import OperatorRollout, create_update_policy
        from spinlock.visualization import (
            create_render_strategy,
            create_aggregate_renderer,
            VisualizationGrid,
            VideoExporter,
            ImageSequenceExporter,
        )

        start_time = time.time()
        torch_device = torch.device(device)

        if verbose:
            print("Step 1/5: Loading dataset and sampling operators...")

        # Load dataset
        with HDF5DatasetReader(dataset_path) as reader:
            metadata = reader.get_metadata()
            num_total_operators = int(metadata.get("num_samples", metadata.get("num_parameter_sets", 0)))

            # Get dataset defaults for realizations
            dataset_num_realizations = int(metadata.get("num_realizations", 10))

            # Sample operator indices
            if operator_indices is not None:
                selected_indices = operator_indices
                if verbose:
                    print(f"  Using explicit indices: {selected_indices}")
            else:
                selected_indices = self._sample_operator_indices(
                    n_total=num_total_operators,
                    n_sample=n_operators,
                    method=sampling_method,
                    seed=seed
                )
                if verbose:
                    print(f"  Sampled {len(selected_indices)} operators via {sampling_method}")

            # Read operator parameters
            all_parameters = reader.get_parameters()  # [N, P]
            selected_params = all_parameters[selected_indices]  # [n_operators, P]

            # Read stored input fields from dataset
            all_inputs = reader.get_inputs()  # [N, C, H, W]
            selected_inputs = all_inputs[selected_indices]  # [n_operators, C, H, W]

            # Convert to torch tensors
            selected_inputs = torch.from_numpy(selected_inputs).to(torch_device)

            # Extract parameter space from metadata
            param_space = metadata["config"]["parameter_space"]

        # Use dataset num_realizations if not overridden by CLI
        if n_realizations is None:
            n_realizations = dataset_num_realizations
            if verbose:
                print(f"  Using dataset num_realizations: {n_realizations}")

        if verbose:
            print(f"  Dataset: {num_total_operators} operators, {selected_params.shape[1]} parameters")
            print(f"  Loaded {len(selected_indices)} input fields from dataset")

        # Build operators
        if verbose:
            print("\nStep 2/5: Building operators...")

        builder = OperatorBuilder()
        operators_dict = {}
        operator_params_dict = {}  # Store parameters for each operator

        for idx, param_vec in enumerate(selected_params):
            # Convert parameter vector to OperatorParameters
            op_params = self._params_from_vector(param_vec, param_space, grid_size)

            # Build operator using build_simple_cnn
            operator = builder.build_simple_cnn(op_params)
            operator = operator.to(torch_device)
            operator.eval()  # Set to eval mode

            operators_dict[idx] = operator
            operator_params_dict[idx] = op_params  # Store parameters

            if verbose:
                print(f"  Operator {idx}: {op_params.num_layers} layers, {op_params.base_channels} channels, {op_params.activation} activation")

        if verbose:
            print(f"  Built {len(selected_indices)} operators")

        # Initialize evolution components
        if verbose:
            print("\nStep 3/5: Running temporal evolution...")

        # Setup clamp range (common for all operators)
        clamp_range_val = None
        if clamp_min is not None and clamp_max is not None:
            clamp_range_val = (clamp_min, clamp_max)

        # Evolve operators using stored inputs and parameters from dataset
        if verbose:
            print(f"\nStep 3/5: Evolving {len(operators_dict)} operators ({num_timesteps} timesteps × {n_realizations} realizations)...")

        trajectories = {}
        for i, (op_idx, operator) in enumerate(operators_dict.items()):
            if verbose:
                print(f"  [{i+1}/{len(operators_dict)}] Operator {op_idx}...", end=" ", flush=True)

            # Get stored evolution parameters for this operator
            op_params = operator_params_dict[op_idx]

            # Create update policy based on stored parameters
            if op_params.update_policy == "autoregressive":
                policy = create_update_policy("autoregressive")
            elif op_params.update_policy == "residual":
                policy = create_update_policy("residual", dt=op_params.dt)
            elif op_params.update_policy == "convex":
                policy = create_update_policy("convex", alpha=op_params.alpha)
            else:
                raise ValueError(f"Unknown update policy: {op_params.update_policy}")

            # Create evolution engine for this operator
            rollout = OperatorRollout(
                policy=policy,
                num_timesteps=num_timesteps,
                device=torch_device,
                normalization=normalization,  # type: ignore[arg-type]
                clamp_range=clamp_range_val,
                compute_metrics=False,  # Disable for performance
            )

            # Use stored input from dataset as initial condition
            initial_condition = selected_inputs[op_idx]  # [C, H, W]

            traj, _ = rollout.evolve_operator(
                operator,
                initial_condition,
                num_realizations=n_realizations,
                base_seed=seed + op_idx * 1000,
                show_progress=False
            )
            # Reorder to [T, M, C, H, W] for VisualizationGrid
            # Move to CPU to free GPU memory
            trajectories[op_idx] = traj.permute(1, 0, 2, 3, 4).cpu()  # [M, T, C, H, W] -> [T, M, C, H, W]

            # Free operator and rollout engine immediately
            del operator, rollout, policy, traj

            # Clear GPU cache after each operator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if verbose:
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"✓ {op_params.update_policy} | GPU: {mem_alloc:.2f}/{mem_reserved:.2f} GiB")
                else:
                    print(f"✓ {op_params.update_policy}")

        # Free operators dict and inputs to maximize available memory
        if verbose:
            print(f"\n  Freeing operators and inputs...")
        del operators_dict, selected_inputs, operator_params_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if verbose:
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU memory after cleanup: {mem_alloc:.2f} GiB")

        if verbose:
            print(f"  Evolution complete!")

        # Render frames
        if verbose:
            print("\nStep 4/5: Rendering visualization grid...")

        # Determine number of channels
        num_channels = 3  # TODO: Read from dataset/operators

        # Create render strategy
        base_renderer = create_render_strategy(
            num_channels=num_channels,
            strategy="auto",
            colormap=colormap,
            device=torch_device
        )

        # Create aggregate renderers
        aggregate_renderers = [
            create_aggregate_renderer(
                agg_type,
                base_renderer=base_renderer if agg_type == "mean" else None,
                colormap=colormap,
                device=torch_device
            )
            for agg_type in aggregates
        ]

        # Create visualization grid
        grid = VisualizationGrid(
            render_strategy=base_renderer,
            aggregate_renderers=aggregate_renderers,
            grid_size=grid_size,
            device=torch_device,
            add_spacing=add_spacing
        )

        # Now that operators are freed, move trajectories to GPU for fast rendering
        if verbose:
            # Calculate trajectory size
            first_traj = next(iter(trajectories.values()))
            traj_size_gb = (first_traj.numel() * first_traj.element_size() * len(trajectories)) / 1024**3
            print(f"  Moving {len(trajectories)} trajectories to GPU ({traj_size_gb:.2f} GiB)...", end=" ", flush=True)

        try:
            trajectories_gpu = {k: v.to(torch_device) for k, v in trajectories.items()}
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    f"OOM while moving trajectories to GPU. "
                    f"Tried to allocate {traj_size_gb:.2f} GiB. "
                    f"Try reducing --n-operators or --steps."
                ) from e
            raise

        if verbose:
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                print(f"✓ GPU: {mem_alloc:.2f} GiB")
            else:
                print("✓")

        # Clear CPU tensors to free RAM
        del trajectories

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print(f"  Rendering {num_timesteps//stride} frames...", end=" ", flush=True)

        frames = grid.create_animation_frames(
            trajectories_gpu,
            num_timesteps=num_timesteps,
            stride=stride
        )

        if verbose:
            print(f"✓")

        grid_info = grid.get_grid_info(trajectories_gpu)

        if verbose:
            print(f"  Rendered {frames.shape[0]} frames ({grid_info['grid_height']}×{grid_info['grid_width']})")
            print(f"  Grid: {grid_info['num_operators']} operators × {grid_info['num_realizations'] + grid_info['num_aggregates']} columns")

        # Move frames to CPU and clear GPU memory
        if verbose:
            print(f"  Moving frames to CPU...", end=" ", flush=True)

        frames = frames.cpu()
        del trajectories_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print(f"✓")

        # Export
        if verbose:
            print("\nStep 5/5: Exporting visualization...")

        if output_video:
            output_video.parent.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"  Encoding video...", end=" ", flush=True)
            exporter = VideoExporter(fps=fps, codec="libx264")
            exporter.export(frames, output_path=output_video)
            if verbose:
                print(f"✓")
                print(f"  Output: {output_video} ({frames.shape[0]} frames @ {fps} fps)")

        if output_frames:
            output_frames.mkdir(parents=True, exist_ok=True)
            exporter = ImageSequenceExporter(format="png", prefix="frame", padding=4)
            exporter.export_with_metadata(
                frames,
                output_dir=output_frames,
                metadata={
                    "fps": fps,
                    "num_frames": frames.shape[0],
                    "grid_size": grid_size,
                    "operators": selected_indices,
                    "realizations": n_realizations,
                    "timesteps": num_timesteps,
                    "stride": stride,
                }
            )
            if verbose:
                print(f"  Frames: {output_frames}/ ({frames.shape[0]} images)")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n✓ Visualization complete ({elapsed:.1f}s)")

        return 0

    def _sample_operator_indices(
        self,
        n_total: int,
        n_sample: int,
        method: str,
        seed: int
    ) -> List[int]:
        """
        Sample operator indices using specified method.

        Args:
            n_total: Total number of operators in dataset
            n_sample: Number of operators to sample
            method: Sampling method ("sobol", "random", "sequential")
            seed: Random seed

        Returns:
            List of operator indices
        """
        if n_sample > n_total:
            print(
                f"Warning: Requested {n_sample} operators but dataset has {n_total}. "
                f"Using all {n_total} operators.",
                file=sys.stderr
            )
            n_sample = n_total

        if method == "sobol":
            # Use Sobol sequence for low-discrepancy sampling
            from scipy.stats.qmc import Sobol

            # Round up to nearest power of 2 for optimal Sobol balance
            n_pow2 = 1 << (n_sample - 1).bit_length()

            sampler = Sobol(d=1, scramble=True)
            uniform_samples_pow2 = sampler.random(n_pow2)

            # Use only the first n_sample points
            uniform_samples = uniform_samples_pow2[:n_sample]
            indices = (uniform_samples[:, 0] * n_total).astype(int)
            indices = np.clip(indices, 0, n_total - 1)
            return indices.tolist()

        elif method == "random":
            # Uniform random sampling
            rng = np.random.default_rng(seed)
            return rng.choice(n_total, size=n_sample, replace=False).tolist()

        elif method == "sequential":
            # Sequential indices
            return list(range(n_sample))

        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _params_from_vector(
        self,
        param_vec: np.ndarray,
        param_space: dict,
        grid_size: int
    ) -> Any:  # type: ignore[misc]
        """
        Convert parameter vector to OperatorParameters object.

        Args:
            param_vec: Parameter vector from dataset [0,1]^d
            param_space: Parameter space specification from config
            grid_size: Spatial grid size

        Returns:
            OperatorParameters object
        """
        from spinlock.operators.builder import OperatorBuilder
        from spinlock.operators.parameters import OperatorParameters

        # Flatten parameter space into ordered dict (same order as generation)
        flat_spec = {}
        for category in ["architecture", "stochastic", "operator"]:
            if category in param_space:
                for name, spec in param_space[category].items():
                    flat_spec[name] = spec

        # Use OperatorBuilder to map parameters
        builder = OperatorBuilder()
        param_dict = builder.map_parameters(param_vec, flat_spec)

        # Fill in fixed parameters
        param_dict["input_channels"] = 3
        param_dict["output_channels"] = 3
        param_dict["grid_size"] = grid_size

        # Add default evolution parameters (dataset doesn't have them yet)
        param_dict["update_policy"] = "convex"
        param_dict["alpha"] = 0.5
        param_dict["dt"] = 0.01

        # Create OperatorParameters object
        # Note: use_batch_norm is in the dataclass but might not be in param_dict
        if "use_batch_norm" not in param_dict:
            param_dict["use_batch_norm"] = False

        return OperatorParameters(**param_dict)
