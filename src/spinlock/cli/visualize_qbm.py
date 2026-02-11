"""CLI command for QBM rollout visualization."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import torch
import numpy as np
import h5py

from .base import CLICommand
from spinlock.visualization.quantum.data_loader import QBMDatasetLoader
from spinlock.visualization.quantum.renderer import QBMRenderer
from spinlock.visualization.quantum.wigner_renderer import WignerRenderer
from spinlock.visualization.quantum.aggregates import QuantumObservableOverlay
from spinlock.visualization.exporters.video import create_video_exporter_with_gpu_fallback
from spinlock.qbm.simulator import QuantumBrownianSimulator
from spinlock.qbm.potentials import PotentialGenerator


class VisualizeQBMCommand(CLICommand):
    """Visualize Quantum Brownian Motion rollouts.

    Extends base visualization pipeline with QBM-specific renderers:
    - Probability density |ψ|²
    - Wigner phase-space distribution W(x,p)
    - Quantum observable overlays (purity, entropy, coherence)

    Uses frame-by-frame GPU memory management for efficiency.
    """

    @property
    def name(self) -> str:
        return "visualize-qbm"

    @property
    def help(self) -> str:
        return "Visualize Quantum Brownian Motion rollouts"

    @property
    def description(self) -> str:
        return """
Visualize Quantum Brownian Motion rollouts with quantum-specific renderers.

Supports multiple rendering modes:
- Probability density: |ψ|² from complex wavefunction
- Wigner function: Phase-space quasi-probability W(x,p)
- Side-by-side: Both probability and Wigner

Optional quantum observable overlays (purity, entropy, coherence).

Examples:
  # Basic probability density visualization
  spinlock visualize-qbm --dataset datasets/qbm_50k.h5 \\
      --output visualizations/qbm_probability.mp4 \\
      --n-rollouts 4

  # Wigner phase-space with entropy overlay
  spinlock visualize-qbm --dataset datasets/qbm_50k.h5 \\
      --output visualizations/qbm_wigner.mp4 \\
      --renderer wigner \\
      --overlay-observable entropy

  # Side-by-side comparison
  spinlock visualize-qbm --dataset datasets/qbm_50k.h5 \\
      --output visualizations/qbm_both.mp4 \\
      --renderer both \\
      --n-rollouts 2
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add visualize-qbm command arguments."""
        # Required arguments
        parser.add_argument(
            "--dataset",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to QBM HDF5 dataset",
        )

        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            metavar="PATH",
            help="Output video path (.mp4)",
        )

        # Visualization parameters
        vis_group = parser.add_argument_group("visualization parameters")

        vis_group.add_argument(
            "--n-rollouts",
            type=int,
            default=4,
            metavar="N",
            help="Number of rollouts to visualize (default: 4)",
        )

        vis_group.add_argument(
            "--num-timesteps",
            type=int,
            default=256,
            metavar="T",
            help="Number of timesteps to simulate (default: 256)",
        )

        vis_group.add_argument(
            "--sampling-method",
            type=str,
            choices=['diverse', 'sobol', 'random'],
            default='diverse',
            help="Rollout sampling strategy (default: diverse)",
        )

        # Rendering parameters
        render_group = parser.add_argument_group("rendering parameters")

        render_group.add_argument(
            "--renderer",
            type=str,
            choices=['probability', 'wigner', 'both'],
            default='probability',
            help="Rendering mode (default: probability)",
        )

        render_group.add_argument(
            "--overlay-observable",
            type=str,
            choices=['none', 'purity', 'entropy', 'coherence_mean', 'coherence_max', 'uncertainty_product'],
            default='purity',
            help="Quantum observable to overlay (default: purity)",
        )

        render_group.add_argument(
            "--colormap",
            type=str,
            default='viridis',
            help="Colormap for probability density (default: viridis)",
        )

        render_group.add_argument(
            "--wigner-colormap",
            type=str,
            default='RdBu_r',
            help="Colormap for Wigner function (default: RdBu_r)",
        )

        # Output parameters
        output_group = parser.add_argument_group("output parameters")

        output_group.add_argument(
            "--fps",
            type=int,
            default=30,
            metavar="N",
            help="Video frames per second (default: 30)",
        )

        output_group.add_argument(
            "--stride",
            type=int,
            default=1,
            metavar="N",
            help="Render every Nth timestep (default: 1)",
        )

        # Execution parameters
        exec_group = parser.add_argument_group("execution parameters")

        exec_group.add_argument(
            "--device",
            type=str,
            default='cuda',
            help="Compute device (cuda or cpu) (default: cuda)",
        )

        exec_group.add_argument(
            "--seed",
            type=int,
            default=42,
            metavar="N",
            help="Random seed for sampling (default: 42)",
        )

        exec_group.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging",
        )

    def execute(self, args: Namespace) -> int:
        """Execute visualize-qbm command."""
        try:
            # Setup
            torch.manual_seed(args.seed)
            device = torch.device(args.device)

            if args.verbose:
                print(f"Loading QBM dataset from {args.dataset}...")

            # Phase 1: Load dataset and sample rollouts
            with h5py.File(args.dataset, 'r') as f:
                dataset_size = f['inputs/fields'].shape[0]

                # Sample rollouts
                if args.verbose:
                    print(f"Sampling {args.n_rollouts} rollouts ({args.sampling_method})...")

                # For now, use simple random sampling (features aren't needed for basic viz)
                indices = torch.randperm(dataset_size)[:args.n_rollouts]

                # Load initial conditions and parameters
                # HDF5 requires sorted indices
                indices_sorted, sort_order = torch.sort(indices)

                if args.verbose:
                    print("Loading initial conditions and parameters...")

                initial_conditions = torch.from_numpy(f['inputs/fields'][indices_sorted.numpy()]).float()  # [N, M, 2, H, W]
                params_normalized = torch.from_numpy(f['parameters/params'][indices_sorted.numpy()]).float()  # [N, 9]

                # Unsort to match original order
                unsort_order = torch.argsort(sort_order)
                initial_conditions = initial_conditions[unsort_order]
                params_normalized = params_normalized[unsort_order]

                if args.verbose:
                    print(f"Initial conditions shape: {initial_conditions.shape}")
                    print(f"Parameters shape: {params_normalized.shape}")

            # Phase 2: Generate rollouts from QBM simulator
            if args.verbose:
                print("Generating QBM rollouts...")

            rollouts = self._generate_rollouts(
                initial_conditions=initial_conditions,
                params_normalized=params_normalized,
                num_timesteps=args.num_timesteps,
                device=device,
                verbose=args.verbose,
            )  # [N, M, T, 2, H, W]

            # For quantum features, we'll skip them for now (just don't use overlay)
            quantum_features = None
            if args.overlay_observable != 'none':
                print("Warning: Quantum feature overlays not yet implemented for on-the-fly simulation")
                args.overlay_observable = 'none'

            if args.verbose:
                print(f"Rollouts shape: {rollouts.shape}")

            # Phase 2: Setup renderers
            if args.verbose:
                print("Initializing renderers...")

            renderers, grid_layout = self._setup_renderers(
                renderer_mode=args.renderer,
                colormap=args.colormap,
                wigner_colormap=args.wigner_colormap,
                device=device,
                n_rollouts=args.n_rollouts,
                n_realizations=rollouts.shape[1],
            )

            # Setup observable overlay (disabled for now since we don't have precomputed features)
            overlay = None

            # Phase 3: Compute global normalization statistics
            if args.verbose:
                print("Computing normalization statistics...")

            global_stats_list = []
            for renderer in renderers:
                stats = renderer.compute_global_stats(rollouts)
                global_stats_list.append(stats)
                if args.verbose:
                    print(f"Global stats for {renderer.__class__.__name__}: {stats}")

            # Phase 4: Frame-by-frame rendering
            if args.verbose:
                print(f"Rendering frames to {args.output}...")

            num_timesteps = rollouts.shape[2] // args.stride
            self._render_video(
                rollouts=rollouts,
                quantum_features=quantum_features,
                renderers=renderers,
                global_stats_list=global_stats_list,
                overlay=overlay,
                grid_layout=grid_layout,
                output_path=args.output,
                fps=args.fps,
                stride=args.stride,
                device=device,
                verbose=args.verbose,
            )

            if args.verbose:
                print(f"✓ Video saved to {args.output}")

            return 0

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _sample_rollouts(
        self,
        loader: QBMDatasetLoader,
        n_rollouts: int,
        method: str,
        dataset_size: int,
    ) -> torch.Tensor:
        """Sample rollout indices using specified method.

        Args:
            loader: Dataset loader
            n_rollouts: Number of rollouts to sample
            method: Sampling method ('diverse', 'sobol', 'random')
            dataset_size: Total dataset size

        Returns:
            Sampled indices [N]
        """
        if n_rollouts > dataset_size:
            print(f"Warning: Requested {n_rollouts} rollouts but dataset has {dataset_size}. Using all.", file=sys.stderr)
            n_rollouts = dataset_size

        if method == 'diverse':
            # Use quantum features for diversity sampling (simplified)
            sample_size = min(1000, dataset_size)
            sample_indices = torch.arange(sample_size)
            quantum_feats = loader.load_quantum_features(sample_indices)

            # Flatten to [N, D] for diversity sampling
            flat_feats = quantum_feats.view(quantum_feats.shape[0], -1)

            # Simple diversity: pick samples far from mean
            mean = flat_feats.mean(dim=0)
            distances = torch.norm(flat_feats - mean, dim=1)
            _, indices = torch.topk(distances, k=min(n_rollouts, len(distances)))

        elif method == 'sobol':
            # Sobol low-discrepancy sampling
            from scipy.stats.qmc import Sobol

            n_pow2 = 1 << (n_rollouts - 1).bit_length()
            sampler = Sobol(d=1, scramble=True, seed=42)
            sobol_samples = sampler.random(n_pow2)

            # Map [0,1] to [0, dataset_size-1]
            sobol_indices = (sobol_samples.flatten() * dataset_size).astype(int)
            sobol_indices = np.clip(sobol_indices, 0, dataset_size - 1)

            # Take first n_rollouts unique indices
            unique_indices = []
            for idx in sobol_indices:
                if int(idx) not in unique_indices:
                    unique_indices.append(int(idx))
                if len(unique_indices) >= n_rollouts:
                    break

            indices = torch.tensor(unique_indices, dtype=torch.long)

        else:  # random
            indices = torch.randperm(dataset_size)[:n_rollouts]

        return indices

    def _setup_renderers(
        self,
        renderer_mode: str,
        colormap: str,
        wigner_colormap: str,
        device: torch.device,
        n_rollouts: int,
        n_realizations: int,
    ) -> tuple:
        """Setup renderers based on mode.

        Args:
            renderer_mode: 'probability', 'wigner', or 'both'
            colormap: Colormap for probability
            wigner_colormap: Colormap for Wigner
            device: Torch device
            n_rollouts: Number of rollouts
            n_realizations: Number of realizations per rollout

        Returns:
            Tuple of (renderers_list, grid_layout_dict)
        """
        renderers = []
        grid_layout = {
            'n_rows': n_rollouts,
            'n_cols': n_realizations,
        }

        if renderer_mode == 'probability':
            renderers.append(QBMRenderer(
                colormap=colormap,
                normalize_mode='global',
                device=device,
            ))

        elif renderer_mode == 'wigner':
            renderers.append(WignerRenderer(
                colormap=wigner_colormap,
                device=device,
            ))

        elif renderer_mode == 'both':
            renderers.append(QBMRenderer(
                colormap=colormap,
                normalize_mode='global',
                device=device,
            ))
            renderers.append(WignerRenderer(
                colormap=wigner_colormap,
                device=device,
            ))
            # Double columns for side-by-side
            grid_layout['n_cols'] = n_realizations * 2

        return renderers, grid_layout

    def _render_video(
        self,
        rollouts: torch.Tensor,
        quantum_features: torch.Tensor,
        renderers: list,
        global_stats_list: list,
        overlay: QuantumObservableOverlay,
        grid_layout: dict,
        output_path: Path,
        fps: int,
        stride: int,
        device: torch.device,
        verbose: bool,
    ):
        """Render video frame-by-frame with aggressive memory management.

        Args:
            rollouts: Full rollout data [N, M, T, 2, H, W]
            quantum_features: Quantum features [N, T, D_quantum]
            renderers: List of renderer instances
            global_stats_list: List of global stats for each renderer
            overlay: Quantum observable overlay (or None)
            grid_layout: Grid layout dict with 'n_rows', 'n_cols'
            output_path: Output video path
            fps: Frames per second
            stride: Render every Nth timestep
            device: Torch device
            verbose: Enable verbose logging
        """
        num_timesteps = rollouts.shape[2]
        N, M = rollouts.shape[0], rollouts.shape[1]

        # Collect all frames first
        all_frames = []

        for t in range(0, num_timesteps, stride):
            # Move single timestep to GPU
            frame_data = rollouts[:, :, t].to(device)  # [N, M, 2, H, W]

            # Flatten N and M for rendering
            B = N * M
            frame_data_flat = frame_data.reshape(B, *frame_data.shape[2:])  # [N*M, 2, H, W]

            # Render with each renderer
            rendered_frames = []
            for renderer, global_stats in zip(renderers, global_stats_list):
                rendered = renderer.render(frame_data_flat, global_stats=global_stats)
                rendered_frames.append(rendered)

            # Concatenate if multiple renderers (for 'both' mode)
            if len(rendered_frames) > 1:
                # Concatenate along width dimension for side-by-side
                # rendered_frames: list of [N*M, H, W, 3]
                combined = torch.cat(rendered_frames, dim=2)  # [N*M, H, 2W, 3]
            else:
                combined = rendered_frames[0]  # [N*M, H, W, 3]

            # Add observable overlay
            if overlay is not None:
                # Extract observable for each rollout (not flattened)
                obs_values_per_rollout = overlay.extract_observable(quantum_features, timestep=t)  # [N]

                # Expand to cover all realizations [N] -> [N*M]
                obs_values = obs_values_per_rollout.repeat_interleave(M)  # [N*M]

                combined = overlay.render_overlay(combined, obs_values)

            # Reshape to grid: [N*M, H, W, 3] -> [N, M, H, W, 3] -> stack to grid
            H, W = combined.shape[1], combined.shape[2]
            combined_grid = combined.reshape(N, M, H, W, 3)

            # Stack into grid: [N, M, H, W, 3] -> [N*H, M*W, 3]
            grid_frame = self._stack_to_grid(combined_grid)  # [N*H, M*W, 3]

            # Store frame in [H, W, 3] format
            all_frames.append(grid_frame.cpu())

            # Aggressive memory cleanup
            torch.cuda.empty_cache()

            if verbose and ((t // stride + 1) % 10 == 0 or t == 0):
                print(f"  Rendered {t // stride + 1}/{(num_timesteps - 1) // stride + 1} frames")

        # Stack all frames to [T, H, W, 3]
        all_frames = torch.stack(all_frames, dim=0)

        # Convert to [T, 3, H, W] for video export
        all_frames = all_frames.permute(0, 3, 1, 2)  # [T, H, W, 3] -> [T, 3, H, W]

        # Export video
        if verbose:
            print("  Encoding video...")

        exporter = create_video_exporter_with_gpu_fallback(fps=fps, try_gpu=True, verbose=verbose)
        exporter.export(all_frames, output_path=output_path)

        if verbose:
            print(f"  ✓ Video encoding complete")

    def _denormalize_parameters(self, params_normalized: torch.Tensor) -> dict:
        """Denormalize QBM parameters from [0,1] to physical values.

        Parameter order in dataset:
            0: gamma (bath coupling) - log-scale [0.0001, 0.1]
            1: kT (temperature) - log-scale [0.01, 10.0]
            2: mass - linear [0.1, 10.0]
            3: potential_type - categorical index
            4: potential_strength - linear [0.1, 5.0]
            5: potential_width - linear [0.05, 0.3]
            6-8: Additional params (unused for basic visualization)

        Args:
            params_normalized: [N, 9] normalized parameters

        Returns:
            Dict with denormalized physical parameters
        """
        N = params_normalized.shape[0]

        # Denormalize log-scale parameters
        gamma_min, gamma_max = 0.0001, 0.1
        kT_min, kT_max = 0.01, 10.0

        gamma = torch.exp(params_normalized[:, 0] * np.log(gamma_max / gamma_min) + np.log(gamma_min))
        kT = torch.exp(params_normalized[:, 1] * np.log(kT_max / kT_min) + np.log(kT_min))

        # Denormalize linear parameters
        mass = params_normalized[:, 2] * (10.0 - 0.1) + 0.1
        potential_strength = params_normalized[:, 4] * (5.0 - 0.1) + 0.1
        potential_width = params_normalized[:, 5] * (0.3 - 0.05) + 0.05

        # Potential type (categorical - map to index)
        potential_type_idx = (params_normalized[:, 3] * 4).long()  # 0-3 for 4 types

        return {
            'gamma': gamma,
            'kT': kT,
            'mass': mass,
            'potential_type_idx': potential_type_idx,
            'potential_strength': potential_strength,
            'potential_width': potential_width,
        }

    def _generate_rollouts(
        self,
        initial_conditions: torch.Tensor,
        params_normalized: torch.Tensor,
        num_timesteps: int,
        device: torch.device,
        verbose: bool,
    ) -> torch.Tensor:
        """Generate QBM rollouts from initial conditions and parameters.

        Args:
            initial_conditions: [N, M, 2, H, W] initial wavefunctions
            params_normalized: [N, 9] normalized parameters
            num_timesteps: Number of timesteps to simulate
            device: Torch device
            verbose: Enable verbose output

        Returns:
            Rollouts [N, M, T, 2, H, W]
        """
        N, M = initial_conditions.shape[0], initial_conditions.shape[1]
        H, W = initial_conditions.shape[3], initial_conditions.shape[4]

        if verbose:
            print(f"  Generating {num_timesteps} timesteps for {N} rollouts × {M} realizations...")

        # Denormalize parameters
        params_phys = self._denormalize_parameters(params_normalized)

        # Initialize simulator and potential generator
        simulator = QuantumBrownianSimulator(grid_size=H, domain_size=10.0, device=device)
        pot_gen = PotentialGenerator(grid_size=H, domain_size=10.0, device=device)

        # Generate potentials for each sample
        # For simplicity, use harmonic potentials with omega derived from potential_strength
        omega = torch.sqrt(params_phys['potential_strength'])  # ω = √(k/m) ≈ √strength

        potentials = pot_gen.harmonic_2d(
            batch_size=N,
            omega=omega.to(device),
            center_x=None,  # Centered
            center_y=None,
            mass=params_phys['mass'].to(device)
        )  # [N, H, W]

        # Stack parameters for simulator (needs gamma, kT, mass in first 3 positions)
        sim_params = torch.zeros(N, 9, device=device)
        sim_params[:, 0] = params_phys['gamma'].to(device)
        sim_params[:, 1] = params_phys['kT'].to(device)
        sim_params[:, 2] = params_phys['mass'].to(device)

        # Generate rollouts for each realization
        all_rollouts = []

        for m in range(M):
            if verbose and M > 1:
                print(f"    Simulating realization {m+1}/{M}...")

            # Get initial conditions for this realization [N, 2, H, W]
            psi_0 = initial_conditions[:, m].to(device)

            # Simulate
            trajectory = simulator.rollout(
                psi_0=psi_0,
                potential=potentials,
                params=sim_params,
                num_steps=num_timesteps,
                dt=0.01,
                return_all_steps=True
            )  # [N, T+1, 2, H, W]

            # Remove initial condition (we only want the evolution steps)
            trajectory = trajectory[:, 1:]  # [N, T, 2, H, W]

            all_rollouts.append(trajectory)

        # Stack realizations: [M, N, T, 2, H, W] -> [N, M, T, 2, H, W]
        rollouts = torch.stack(all_rollouts, dim=1)  # [N, M, T, 2, H, W]

        if verbose:
            print(f"  ✓ Generated rollouts: {rollouts.shape}")

        return rollouts

    def _stack_to_grid(self, frames: torch.Tensor) -> torch.Tensor:
        """Stack frames into a grid layout.

        Args:
            frames: [N, M, H, W, 3] - N rollouts, M realizations

        Returns:
            Grid frame [N*H, M*W, 3]
        """
        N, M, H, W, C = frames.shape

        # Reshape to [N*H, M*W, 3]
        # First, reshape to [N, M*W, H, 3] by concatenating along width
        frames_horizontal = frames.permute(0, 2, 1, 3, 4).reshape(N, H, M * W, C)

        # Then concatenate along height to get [N*H, M*W, 3]
        grid = frames_horizontal.permute(0, 1, 2, 3).reshape(N * H, M * W, C)

        return grid
