"""CLI command for visualizing Lenia cellular automata from a SpinlockDataset.

Randomly samples N CAs, simulates them on GPU via LeniaReplayAdapter,
and renders temporal evolution as a grid video. Channel-dependent rendering:
C=1 thermal colormap, C=2 thermal+alpha, C=3 RGB, C>=4 PCA projection.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class VisualizeLeniaCaCommand(CLICommand):
    @property
    def name(self) -> str:
        return "visualize-lenia-ca"

    @property
    def help(self) -> str:
        return "Visualize Lenia CA temporal evolution as a grid video"

    @property
    def description(self) -> str:
        return """
Randomly sample N Lenia CAs from a SpinlockDataset, simulate them on GPU,
and render temporal evolution as a grid video (MP4 or GIF).

Channel-dependent rendering:
  C=1  Thermal colormap (blue-to-red)
  C=2  Thermal + alpha compositing (ch0 → color, ch1 → opacity)
  C=3  Direct RGB mapping
  C>=4 PCA projection to RGB

Examples:
  # 4x4 grid of 16 random CAs, 256 timesteps
  spinlock visualize-lenia-ca \\
      --dataset datasets/lenia_500k.h5 \\
      --output visualizations/lenia_grid.mp4

  # Single CA full-screen
  spinlock visualize-lenia-ca \\
      --dataset datasets/lenia_500k.h5 \\
      --output visualizations/lenia_single.mp4 \\
      --sample-index 42

  # Quick smoke test
  spinlock visualize-lenia-ca \\
      --dataset datasets/lenia_500k.h5 \\
      --output /tmp/test.mp4 \\
      --n-samples 4 --timesteps 32
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--dataset", type=Path, required=True, metavar="PATH",
            help="SpinlockDataset HDF5 path",
        )
        parser.add_argument(
            "--output", type=Path, required=True, metavar="PATH",
            help="Output video path (.mp4 or .gif)",
        )
        parser.add_argument(
            "--sample-index", type=int, default=None,
            help="Specific sample index (overrides --n-samples)",
        )
        parser.add_argument(
            "--n-samples", type=int, default=16,
            help="Number of random CAs to visualize (default: 16)",
        )
        parser.add_argument(
            "--timesteps", type=int, default=None,
            help="Simulation timesteps (auto-detect from HDF5, else 256)",
        )
        parser.add_argument(
            "--batch-size", type=int, default=16,
            help="GPU simulation batch size (default: 16)",
        )
        parser.add_argument(
            "--grid-cols", type=int, default=None,
            help="Grid columns (default: ceil(sqrt(n)))",
        )
        parser.add_argument(
            "--fps", type=int, default=30,
            help="Output video FPS (default: 30)",
        )
        parser.add_argument(
            "--device", type=str, default="cuda",
            help="Computation device (default: cuda)",
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Random seed for sample selection (default: 42)",
        )
        parser.add_argument(
            "--colormap", type=str, default="coolwarm",
            help="Colormap for C=1/C=2 thermal rendering (default: coolwarm)",
        )
        parser.add_argument(
            "--substeps", type=int, default=1,
            help="Euler sub-steps per visible frame for stability (default: 1)",
        )

    def execute(self, args: Namespace) -> int:
        import logging
        import math

        import numpy as np
        import torch
        from torchvision.utils import make_grid

        from spinlock.data import SpinlockDataset
        from spinlock.lenia.replay_adapter import LeniaReplayAdapter
        from spinlock.visualization.core.renderer import (
            HeatmapRenderer,
            create_render_strategy,
        )
        from spinlock.visualization.exporters.video import (
            GIFExporter,
            create_video_exporter_with_gpu_fallback,
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(__name__)

        # ── Validate inputs ──────────────────────────────────────────
        if not self.validate_file_exists(args.dataset, "Dataset"):
            return 1

        output_path = Path(args.output)
        suffix = output_path.suffix.lower()
        if suffix not in (".mp4", ".gif"):
            return self.error(f"Unsupported output format '{suffix}'. Use .mp4 or .gif")

        device = torch.device(args.device)

        # ── Load dataset (introspection only) ─────────────────────────
        logger.info(f"Loading dataset: {args.dataset}")
        dataset = SpinlockDataset(str(args.dataset))  # metadata only

        n_channels = dataset.num_channels
        raw_shape = dataset.raw_input_shape  # (N, M, C, H, W)
        total_samples = raw_shape[0]
        grid_size = raw_shape[-1]  # W (assumes square)
        logger.info(
            f"Dataset: {total_samples} samples, "
            f"{n_channels}ch, {grid_size}x{grid_size} grid"
        )

        # ── Select samples ───────────────────────────────────────────
        if args.sample_index is not None:
            if args.sample_index >= total_samples:
                return self.error(
                    f"--sample-index {args.sample_index} out of range "
                    f"(dataset has {total_samples} samples)"
                )
            indices = np.array([args.sample_index])
        else:
            if total_samples < args.n_samples:
                return self.error(
                    f"Requested {args.n_samples} samples but dataset has "
                    f"only {total_samples}"
                )
            rng = np.random.RandomState(args.seed)
            indices = rng.choice(total_samples, size=args.n_samples, replace=False)
            indices.sort()

        N = len(indices)

        # Load ICs and params directly from HDF5 for selected indices
        import h5py

        with h5py.File(args.dataset, "r") as f:
            ics = torch.from_numpy(f["inputs/fields"][indices, 0, :, :, :]).float()
            params = torch.from_numpy(f["parameters/params"][indices]).float()

        logger.info(f"Selected {N} samples: indices {list(indices[:8])}{'...' if N > 8 else ''}")

        # ── Auto-detect timesteps ────────────────────────────────────
        if args.timesteps is not None:
            timesteps = args.timesteps
        else:
            timesteps = _detect_timesteps(args.dataset)
            logger.info(f"Auto-detected timesteps: {timesteps}")

        # ── Simulate ─────────────────────────────────────────────────
        logger.info(
            f"Simulating {N} CAs for {timesteps} timesteps on {device} "
            f"(batch_size={args.batch_size})"
        )
        adapter = LeniaReplayAdapter(
            n_channels=n_channels,
            grid_size=grid_size,
            device=str(device),
            max_gpu_batch=args.batch_size,
            substeps=args.substeps,
        )
        trajectories = adapter.rollout_batch(
            params_batch=params,
            ics=ics,
            timesteps=timesteps,
            return_all_steps=True,
        )  # [N, T+1, C, H, W] on CPU
        T = trajectories.shape[1]
        logger.info(f"Trajectories: {list(trajectories.shape)}")

        # ── Build renderer ───────────────────────────────────────────
        if n_channels == 2:
            renderer = _AlphaCompositeRenderer(
                colormap=args.colormap, device=device,
            )
        elif n_channels == 1:
            renderer = HeatmapRenderer(
                colormap=args.colormap, device=device,
            )
        else:
            renderer = create_render_strategy(
                num_channels=n_channels,
                colormap=args.colormap,
                device=device,
            )

        # For PCA renderer, fit on first timestep across all samples
        if n_channels >= 4:
            renderer.fit(trajectories[:, 0].to(device))

        # ── Grid layout ──────────────────────────────────────────────
        grid_cols = args.grid_cols or math.ceil(math.sqrt(N))

        # ── Compute global normalization bounds ──────────────────────
        # Without this, each frame gets its own vmin/vmax from percentile
        # clipping, causing brightness to strobe as the CA evolves.
        # Subsample to avoid torch.quantile's element limit on large tensors.
        flat = trajectories.flatten()
        MAX_QUANTILE_SAMPLES = 10_000_000
        if flat.numel() > MAX_QUANTILE_SAMPLES:
            idx = torch.randperm(flat.numel())[:MAX_QUANTILE_SAMPLES]
            flat = flat[idx]
        global_vmin = torch.quantile(flat, 0.01)
        global_vmax = torch.quantile(flat, 0.99)
        logger.info(f"Global intensity range: [{global_vmin:.4f}, {global_vmax:.4f}]")

        # ── Render frames ────────────────────────────────────────────
        logger.info(f"Rendering {T} frames ({N} CAs in {grid_cols}-col grid)...")
        frames = []
        for t in range(T):
            # Move single timestep to GPU for rendering
            step_data = trajectories[:, t].to(device)  # [N, C, H, W]
            rgb = renderer.render(step_data, vmin=global_vmin, vmax=global_vmax)

            grid_frame = make_grid(
                rgb.cpu(), nrow=grid_cols, padding=2, normalize=False,
            )  # [3, grid_H, grid_W]
            frames.append(grid_frame)

            if (t + 1) % 50 == 0:
                logger.info(f"  Rendered {t + 1}/{T} frames")

        video_tensor = torch.stack(frames)  # [T, 3, grid_H, grid_W]

        # ── Upscale for watchability + NVENC compat ───────────────────
        # NVENC requires ≥256 on each dimension; small CA grids also
        # look better upscaled with nearest-neighbor (preserves sharp pixels).
        MIN_DIM = 512
        h, w = video_tensor.shape[2], video_tensor.shape[3]
        if h < MIN_DIM or w < MIN_DIM:
            scale = math.ceil(MIN_DIM / min(h, w))
            video_tensor = torch.nn.functional.interpolate(
                video_tensor, scale_factor=scale, mode="nearest",
            )
            logger.info(
                f"Upscaled {h}x{w} → {video_tensor.shape[2]}x{video_tensor.shape[3]} "
                f"({scale}x nearest)"
            )

        logger.info(f"Video tensor: {list(video_tensor.shape)}")

        # ── Export ───────────────────────────────────────────────────
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if suffix == ".gif":
            exporter = GIFExporter(fps=args.fps)
        else:
            exporter = create_video_exporter_with_gpu_fallback(
                fps=args.fps, verbose=True,
            )

        logger.info(f"Exporting to {output_path}...")
        exporter.export(video_tensor, output_path)

        print(f"\nSaved {suffix.upper().strip('.')} to: {output_path}")
        print(f"  Samples: {N}, Timesteps: {T}, Grid: {grid_cols} cols")
        print(f"  Resolution: {video_tensor.shape[2]}x{video_tensor.shape[3]}")
        return 0


# ── Helpers ──────────────────────────────────────────────────────────


def _detect_timesteps(dataset_path: Path) -> int:
    """Try to read simulation timesteps from HDF5 metadata, else default 256."""
    import h5py

    with h5py.File(dataset_path, "r") as f:
        # Check root attrs
        ts = f.attrs.get("num_timesteps", None)
        if ts is not None:
            return int(ts)

        # Check simulation group
        if "simulation" in f:
            ts = f["simulation"].attrs.get("num_timesteps", None)
            if ts is not None:
                return int(ts)

        # Check generation_config
        if "generation_config" in f.attrs:
            import json

            try:
                cfg = json.loads(f.attrs["generation_config"])
                ts = cfg.get("simulation", {}).get("timesteps", None)
                if ts is not None:
                    return int(ts)
            except (json.JSONDecodeError, TypeError):
                pass

    return 256  # sensible default for Lenia


class _AlphaCompositeRenderer:
    """C=2 renderer: ch0 → thermal RGB, ch1 → alpha opacity over black."""

    def __init__(self, colormap: str = "coolwarm", device=None):
        import torch
        from spinlock.visualization.core.renderer import HeatmapRenderer

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._heatmap = HeatmapRenderer(colormap=colormap, device=device)
        self.device = device

    def render(self, data, vmin=None, vmax=None):
        """Render [B, 2, H, W] → [B, 3, H, W] via alpha compositing over black."""
        rgb = self._heatmap.render(data[:, 0:1], vmin=vmin, vmax=vmax)
        alpha = data[:, 1:2].clamp(0, 1)           # [B, 1, H, W]
        return rgb * alpha  # composited over black
