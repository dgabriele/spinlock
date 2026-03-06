"""
Pretokenize Dataset command for Spinlock CLI.

Pre-tokenizes CNO dataset with VQTokenizer v2 for fast diffusion training by:
1. Loading dataset features (temporal + initial)
2. Batch tokenizing all samples with VQTokenizer
3. Saving tokens to HDF5 for instant loading during training

This eliminates the on-the-fly tokenization bottleneck during diffusion training,
providing ~100x speedup per batch.

Prerequisites:
    1. CNO dataset with features (e.g., datasets/50k_baseline.h5)
    2. Trained VQTokenizer v2 checkpoint

Documentation:
    - Diffusion training: experiments/diffusion/README.md
    - VQTokenizer v2: src/spinlock/tokens/
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Tuple, Optional
import h5py
import numpy as np
import queue
import threading
import torch
from tqdm import tqdm

from .base import CLICommand


class _AsyncHDF5Writer:
    """Background thread for writing HDF5 token batches concurrently with GPU compute.

    Accepts lists of (save_key, tokens_np, start, end, indices) tuples via a bounded
    queue.  A single writer thread drains the queue, keeping HDF5 writes serialized
    while allowing the main thread to proceed with GPU work.
    """

    def __init__(self, tokens_group: "h5py.Group", maxsize: int = 2):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._tokens_group = tokens_group
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            item = self._queue.get()
            if item is None:  # sentinel → shut down
                break
            try:
                for save_key, tokens_np, start, end, indices in item:
                    if indices is not None:
                        # HDF5 requires indices in increasing order
                        order = np.argsort(indices)
                        self._tokens_group[save_key][indices[order]] = tokens_np[order]
                    else:
                        self._tokens_group[save_key][start:end] = tokens_np
            except Exception as e:
                self._error = e
            self._queue.task_done()

    def submit(self, writes: list):
        """Enqueue a batch of writes. Blocks if the queue is full (backpressure)."""
        if self._error is not None:
            raise self._error
        self._queue.put(writes)

    def close(self):
        """Drain the queue, stop the writer thread, and re-raise any error."""
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error


class _AccumulationBuffer:
    """Accumulates simulation batch results on CPU for mega-batch tokenization.

    Simulation is memory-constrained to small batches (B=12), but VQ tokenization
    at B=12 vastly underutilizes GPU compute. This buffer collects N simulation
    batches' outputs on CPU, then flushes them as a single mega-batch for
    tokenization at higher throughput.
    """

    def __init__(self, accumulate_n: int):
        self.accumulate_n = accumulate_n
        self.trajectories: list = []      # [B_i, T+1, C, H, W] CPU tensors
        self.theta_batches: list = []     # [B_i, P] CPU tensors
        self.ic_batches: list = []        # [B_i, C, H, W] CPU tensors
        self.write_indices: list = []     # per-sample virtual indices (numpy)
        self.total_samples = 0

    @property
    def is_full(self) -> bool:
        return len(self.trajectories) >= self.accumulate_n

    def append(
        self,
        full_traj: torch.Tensor,
        theta: torch.Tensor,
        ic: torch.Tensor,
        virtual_indices: torch.Tensor,
    ):
        """Append one simulation batch's results (moved to CPU)."""
        self.trajectories.append(full_traj.cpu() if full_traj.is_cuda else full_traj)
        self.theta_batches.append(theta.cpu() if theta.is_cuda else theta)
        self.ic_batches.append(ic.cpu() if ic.is_cuda else ic)
        self.write_indices.append(virtual_indices.numpy())
        self.total_samples += full_traj.shape[0]

    def flush(self):
        """Return concatenated data and clear the buffer."""
        mega_traj = torch.cat(self.trajectories)
        mega_theta = torch.cat(self.theta_batches)
        mega_ic = torch.cat(self.ic_batches)
        all_vi = np.concatenate(self.write_indices)
        self.trajectories.clear()
        self.theta_batches.clear()
        self.ic_batches.clear()
        self.write_indices.clear()
        self.total_samples = 0
        return mega_traj, mega_theta, mega_ic, all_vi


def _compute_accumulation_count(
    sim_batch_size: int,
    max_T: int,
    n_channels: int,
    grid_size: int,
    max_cpu_bytes: int = 4 * 1024**3,
) -> int:
    """Determine how many simulation batches to accumulate before flushing.

    Balances two constraints:
    - CPU memory budget (default 4 GB for trajectory storage)
    - Minimum accumulated samples for GPU saturation (target >= 48)

    Returns:
        Number of simulation batches to accumulate (clamped to [2, 8]).
    """
    # bytes per batch: B * (T+1) * C * H * W * sizeof(float32)
    per_batch_bytes = sim_batch_size * (max_T + 1) * n_channels * grid_size * grid_size * 4
    max_batches = max(2, max_cpu_bytes // max(1, per_batch_bytes))
    # Target at least 48 accumulated samples for meaningful GPU saturation
    min_batches = max(2, 48 // max(1, sim_batch_size))
    return min(max_batches, max(min_batches, 4), 8)


def _compute_tokenization_batch_size(
    trunc_len: int,
    n_channels: int,
    grid_size: int,
    device: torch.device,
) -> int:
    """Compute adaptive tokenization batch size based on available GPU memory.

    Memory model accounts for trajectory frames + ~1.875x pyramid expansion
    on GPU (sum of 1/2^i for i=0..3).

    Returns:
        Batch size rounded down to a multiple of 12 (at least 12).
    """
    frames = trunc_len + 1
    pyramid_factor = 1.875  # sum(1/2^i for i=0..3) approximate pyramid expansion
    bytes_per_sample = int(frames * pyramid_factor * n_channels * grid_size * grid_size * 4)

    if device.type == "cuda":
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        cuda_free = torch.cuda.mem_get_info(device)[0]
        available = (reserved - allocated) + cuda_free
    else:
        available = 4 * 1024**3

    # Reserve 1.5 GB for CNN chunks + VQ codebooks, use 60% of remainder
    budget = int((available - 1.5 * 1024**3) * 0.6)
    optimal = max(12, budget // max(1, bytes_per_sample))
    # Round down to multiple of 12 (sim batch size alignment)
    return max(12, (optimal // 12) * 12)


class PretokenizeDatasetCommand(CLICommand):
    """
    Command to pre-tokenize CNO dataset for fast diffusion training.

    Tokenizes all samples once using batch processing and saves tokens to HDF5,
    eliminating the need for on-the-fly tokenization during training.

    Supports both manual-mode (pre-extracted features) and learned-mode
    (CNN temporal features with on-the-fly trajectory generation via replayer).
    """

    @property
    def name(self) -> str:
        return "pretokenize-dataset"

    @property
    def help(self) -> str:
        return "Pre-tokenize CNO dataset with VQTokenizer for fast training"

    @property
    def description(self) -> str:
        return """
Pre-tokenize CNO dataset with VQTokenizer v2 for fast diffusion training.

This command batch-tokenizes all samples in a CNO dataset and saves the tokens
to a new HDF5 file. This eliminates the on-the-fly tokenization bottleneck during
training, providing ~100x speedup per batch.

Process:
1. Load CNO dataset features (temporal + initial)
2. Load VQTokenizer v2 checkpoint
3. Batch tokenize all samples (parallelized on GPU)
4. Save tokens to HDF5 with compression

Output HDF5 structure:
  - tokens/{category_level}: [N] int32 token indices per category-level
  - features/temporal: [N, T, D] original temporal features (optional)
  - features/initial: [N, D] original initial features (optional)

Examples:
  # Pre-tokenize 50K baseline dataset
  spinlock pretokenize-dataset \\
      --dataset datasets/50k_baseline.h5 \\
      --tokenizer checkpoints/vqvae/vq_tokenizer_best.pt \\
      --output datasets/50k_baseline_tokenized.h5 \\
      --batch-size 128

  # Include features in output (standalone file)
  spinlock pretokenize-dataset \\
      --dataset datasets/50k_baseline.h5 \\
      --tokenizer checkpoints/vqvae/vq_tokenizer_best.pt \\
      --output datasets/50k_baseline_tokenized.h5 \\
      --copy-features

  # Temporal resolution mode (requires pyramid encoder)
  spinlock pretokenize-dataset \\
      --dataset datasets/50k_baseline.h5 \\
      --tokenizer checkpoints/vqvae/vq_tokenizer_best.pt \\
      --output datasets/50k_temporal_resolution.h5 \\
      --temporal-resolution \\
      --batch-size 128
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add pretokenize-dataset command arguments."""
        parser.add_argument(
            "--dataset",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to input dataset HDF5 file",
        )

        parser.add_argument(
            "--tokenizer",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to VQTokenizer v2 checkpoint",
        )

        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            metavar="PATH",
            help="Output path for pre-tokenized dataset HDF5 file",
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=128,
            metavar="N",
            help="Batch size for tokenization (default: 128)",
        )

        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "cpu"],
            help="Device for tokenization (default: cuda)",
        )

        parser.add_argument(
            "--copy-features",
            action="store_true",
            help="Copy original features to output file (makes it standalone)",
        )

        parser.add_argument(
            "--temporal-resolution",
            action="store_true",
            help="Enable temporal resolution mode: tokenize at multiple truncation lengths for temporal resolution D3PM",
        )

        parser.add_argument(
            "--no-compile",
            action="store_true",
            help="Disable torch.compile for Lenia simulator (frees ~2 GB GPU for larger batches)",
        )

    def execute(self, args: Namespace) -> int:
        """Execute dataset pre-tokenization."""
        # Validate inputs
        device = self._validate_and_setup(args)
        if device is None:
            return 1

        # Load tokenizer
        tokenizer = self._load_tokenizer(args.tokenizer, device)
        if tokenizer is None:
            return 1

        # Detect learned mode from tokenizer config
        self.is_learned_mode = tokenizer.config.feature_source == "learned"
        self.is_temporal_only = tokenizer.config.temporal_only
        self._realization_mode = getattr(tokenizer.config, "realization_mode", "single")
        self.replayer = None

        # Setup replayer for learned mode (trajectory generation)
        self._use_compile = not getattr(args, 'no_compile', False)
        if self.is_learned_mode:
            self.replayer = self._setup_replayer(tokenizer.config, device)
            if self.replayer is None:
                return 1

        # Load dataset features
        features = self._load_dataset_features(args.dataset, tokenizer)
        if features is None:
            return 1

        # Apply feature cleaning to match tokenizer's expected dimensions
        # (only relevant for manual mode — learned mode has no pre-extracted temporal)
        if not self.is_learned_mode:
            features = self._apply_feature_cleaning(tokenizer, features)

        # Extract truncation lengths if temporal resolution mode
        truncation_lengths = None
        if args.temporal_resolution:
            truncation_lengths = self._extract_truncation_lengths(tokenizer, features)
            if truncation_lengths is None:
                return 1
            print(f"\n⚡ Temporal resolution mode enabled")
            print(f"  Truncation lengths: {truncation_lengths}")

        # Analyze token structure
        category_levels = self._analyze_token_structure(
            tokenizer, features, device, truncation_lengths
        )
        if category_levels is None:
            return 1

        # Create output file
        output_file = self._create_output_file(
            args.output,
            features,
            category_levels,
            args.copy_features,
            truncation_lengths,
        )
        if output_file is None:
            return 1

        # Batch tokenize and save
        try:
            if features.get('dataset') is not None:
                # Streaming mode: DataLoader-based batch loop (learned mode)
                self._batch_tokenize_streaming(
                    tokenizer,
                    features['dataset'],
                    category_levels,
                    output_file,
                    args.batch_size,
                    device,
                    truncation_lengths,
                )
            else:
                # Eager mode: slice pre-loaded numpy arrays (manual mode)
                self._batch_tokenize_and_save(
                    tokenizer,
                    features,
                    category_levels,
                    output_file,
                    args.batch_size,
                    device,
                    truncation_lengths,
                )
        finally:
            output_file.close()
            if features.get('dataset') is not None:
                features['dataset'].close_lazy()

        # Report results
        self._report_results(args.output, features, category_levels, truncation_lengths)
        return 0

    def _validate_and_setup(self, args: Namespace) -> Optional[torch.device]:
        """Validate inputs and determine device."""
        if not args.dataset.exists():
            self.error(f"Dataset not found: {args.dataset}")
            return None

        if not args.tokenizer.exists():
            self.error(f"Tokenizer checkpoint not found: {args.tokenizer}")
            return None

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        return device

    def _load_tokenizer(self, checkpoint_path: Path, device: torch.device):
        """Load and prepare VQTokenizer."""
        from spinlock.tokens.tokenizer import VQTokenizer

        print(f"\nLoading VQTokenizer from {checkpoint_path}")
        try:
            tokenizer = VQTokenizer.from_checkpoint(checkpoint_path)
            tokenizer.model.to(device)
            tokenizer.model.eval()
            print("✓ Tokenizer loaded")
            return tokenizer
        except Exception as e:
            self.error(f"Failed to load tokenizer: {e}")
            return None

    def _load_dataset_features(
        self, dataset_path: Path, tokenizer=None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load features from SpinlockDataset.

        In learned mode, temporal features are NOT loaded from the dataset
        (they come from the CNN operating on generated trajectories). Only
        theta parameters and raw ICs are loaded.
        """
        from spinlock.data import SpinlockDataset

        print(f"\nLoading dataset from {dataset_path}")
        try:
            if self.is_learned_mode:
                # ── Streaming mode (learned) ─────────────────────────
                # Only params (~32 MB) loaded eagerly; ICs read per-sample
                # from HDF5 in __getitem__ (lazy_ics=True). This avoids
                # the 73+ GB eager load of inputs/fields.
                streaming_ds = SpinlockDataset(
                    str(dataset_path),
                    max_samples=int(1e9),  # Clamped to actual total
                    lazy_ics=True,
                    realization_mode=self._realization_mode,
                )
                print(f"✓ Streaming dataset: {streaming_ds.n_samples:,} samples (lazy ICs, realization_mode='{self._realization_mode}')")
                print(f"  Mode: learned (temporal features from CNN, not dataset)")
                return {
                    'dataset': streaming_ds,
                    'temporal': None,
                    'initial_manual': None,
                    'initial_raw': None,
                    'theta': None,
                    'num_samples': streaming_ds.n_samples,
                }

            # ── Eager mode (manual) ──────────────────────────────────
            dataset = SpinlockDataset.from_file(str(dataset_path))
            with dataset.open():
                # Load temporal features (skip in learned mode — CNN generates them)
                temporal = None
                if not self.is_learned_mode:
                    temporal = dataset.features.temporal.load_all() if dataset.features.temporal else None

                # Load initial manual features (aggregated)
                initial_manual = dataset.features.initial.load_all() if dataset.features.initial else None

                # Load initial raw features (raw ICs from inputs)
                initial_raw = None
                if dataset.inputs is not None:
                    ics = dataset.inputs.load_all()
                    # Aggregate over realizations if needed
                    if ics.ndim == 5:  # [N, M, C, H, W]
                        ics = ics.mean(axis=1)  # [N, C, H, W]
                    elif ics.ndim == 3:  # [N, H, W]
                        ics = ics[:, None, :, :]  # [N, 1, H, W]
                    initial_raw = ics

                # Load theta (parameters) features
                theta = None
                if dataset.parameters is not None and dataset.parameters.params is not None:
                    theta = dataset.parameters.params.load_all()
                    print(f"  Theta (parameters): {theta.shape}")

                # Determine num_samples from whatever is available
                if temporal is not None:
                    num_samples = temporal.shape[0]
                elif theta is not None:
                    num_samples = theta.shape[0]
                elif initial_raw is not None:
                    num_samples = initial_raw.shape[0]
                elif initial_manual is not None:
                    num_samples = initial_manual.shape[0]
                else:
                    self.error("No features found in dataset")
                    return None

                print(f"✓ Loaded {num_samples:,} samples")

                if self.is_learned_mode:
                    print(f"  Mode: learned (temporal features from CNN, not dataset)")
                if temporal is not None:
                    print(f"  Temporal features: {temporal.shape}")
                if initial_manual is not None:
                    print(f"  Initial manual features: {initial_manual.shape}")
                if initial_raw is not None:
                    print(f"  Initial raw features: {initial_raw.shape}")

                return {
                    'temporal': temporal,
                    'initial_manual': initial_manual,
                    'initial_raw': initial_raw,
                    'theta': theta,
                    'num_samples': num_samples,
                }
        except Exception as e:
            self.error(f"Failed to load dataset: {e}")
            return None

    def _setup_replayer(self, config, device: torch.device):
        """Set up trajectory replayer for learned-mode trajectory generation.

        Auto-detects operator type from generation config and creates the
        appropriate replayer (LeniaReplayAdapter, CNOReplayer, etc.).

        Args:
            config: TokenizerConfig from loaded checkpoint
            device: Computation device

        Returns:
            Replayer instance, or None on failure
        """
        import yaml as _yaml

        config_path = config.generation_config_path or config.cno_config_path
        if config_path is None:
            self.error(
                "Learned-mode tokenizer requires generation_config_path (or cno_config_path) in config.\n"
                "Set it in the training config YAML."
            )
            return None

        try:
            with open(config_path) as f:
                gen_config = _yaml.safe_load(f)
            operator_type = gen_config.get("simulation", {}).get("operator_type")

            match operator_type:
                case "lenia":
                    from spinlock.lenia.replay_adapter import LeniaReplayAdapter
                    replayer = LeniaReplayAdapter.from_config(
                        config_path, device=str(device), compile=self._use_compile)
                case "cnn" | "u_afno":
                    from spinlock.mno.cno_replay import CNOReplayer
                    replayer = CNOReplayer.from_config(
                        config_path, device=str(device),
                        cache_size=config.replayer_cache_size,
                    )
                case "qbm":
                    raise NotImplementedError("QBM replay adapter not yet implemented.")
                case _:
                    raise NotImplementedError(
                        f"No replay adapter for operator_type='{operator_type}'."
                    )

            print(f"✓ {type(replayer).__name__} created from {config_path}")
            print(f"  Generation timesteps: {config.generation_timesteps}")
            return replayer
        except Exception as e:
            self.error(f"Failed to create replayer: {e}")
            return None

    def _build_cfl_sorted_sampler(self, dataset, batch_size: int):
        """Build a sampler that groups samples by CFL substep count.

        Lenia's CFL-adaptive substeps force the BATCH maximum K for all samples.
        With uniform Sobol sampling, 52% of samples need K=1 but random batching
        gives average batch-K=22.  Sorting by CFL groups low-K samples together,
        dropping average batch-K to ~3 and speeding simulation by ~7×.

        Returns:
            A sequential sampler iterating in CFL-sorted order, or None if
            CFL computation fails (falls back to sequential order).
        """
        import math as _math

        try:
            all_params = dataset.params  # [N, 34] tensor
            if all_params is None or all_params.shape[0] == 0:
                return None

            n_channels = getattr(self.replayer, 'n_channels', 3)
            param_ranges = getattr(self.replayer, 'param_ranges', None)

            if param_ranges is None:
                return None

            from spinlock.lenia.params import sobol_batch_to_tensors

            # Vectorized CFL computation on CPU (fast for 500K samples)
            tensors = sobol_batch_to_tensors(
                all_params.numpy() if isinstance(all_params, torch.Tensor) else all_params,
                n_channels, 'cpu', ranges=param_ranges,
            )
            sigma_min = tensors.growth_sigma.min(dim=1).values.clamp(min=1e-8)
            g_prime = torch.full_like(tensors.dt, 1.716)
            if tensors.growth_type is not None:
                g_prime[tensors.growth_type == 1] = 3.079
                g_prime[tensors.growth_type == 2] = 5.0
            cfl = tensors.dt * g_prime / sigma_min

            sort_indices = torch.argsort(cfl).tolist()

            # Estimate K distribution for logging
            K_sorted = []
            for i in range(0, len(sort_indices), batch_size):
                batch_cfl = cfl[sort_indices[i:i + batch_size]]
                K = min(32, max(1, _math.ceil(batch_cfl.max().item())))
                K_sorted.append(K)
            avg_K = sum(K_sorted) / len(K_sorted)
            pct_32 = sum(1 for k in K_sorted if k == 32) / len(K_sorted) * 100

            print(f"  CFL-sorted batching: avg batch K={avg_K:.1f} "
                  f"(K=32: {pct_32:.0f}% of batches)")

            return sort_indices

        except Exception as e:
            print(f"  Warning: CFL sorting failed ({e}), using sequential order")
            return None

    def _generate_trajectories(
        self,
        theta_batch: torch.Tensor,
        initial_raw_batch: torch.Tensor,
        generation_timesteps: int,
    ) -> torch.Tensor:
        """Generate trajectories via replayer for a batch.

        Prefers batched rollout_batch() when available (e.g. LeniaReplayAdapter),
        falls back to per-sample rollout() loop (e.g. CNOReplayer).

        Args:
            theta_batch: [B, param_dim] on any device
            initial_raw_batch: [B, C, H, W] on any device
            generation_timesteps: Number of timesteps to generate

        Returns:
            trajectories: [B, T+1, C, H, W] on CPU
        """
        # Batched path (e.g. LeniaReplayAdapter)
        if hasattr(self.replayer, 'rollout_batch'):
            with torch.no_grad():
                return self.replayer.rollout_batch(
                    params_batch=theta_batch.cpu(),
                    ics=initial_raw_batch,
                    timesteps=generation_timesteps,
                    return_all_steps=True,
                )  # [B, T+1, C, H, W] on CPU

        # Per-sample path (e.g. CNOReplayer)
        B = theta_batch.shape[0]
        trajectories = []
        with torch.no_grad():
            for i in range(B):
                traj = self.replayer.rollout(
                    params_vector=theta_batch[i].cpu(),
                    ic=initial_raw_batch[i],
                    timesteps=generation_timesteps,
                    return_all_steps=True,
                )  # [1, T+1, C, H, W] on replayer device
                trajectories.append(traj.squeeze(0).cpu())  # [T+1, C, H, W]
        return torch.stack(trajectories, dim=0)  # [B, T+1, C, H, W]

    def _apply_feature_cleaning(
        self,
        tokenizer,
        features: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply feature cleaning to match tokenizer's expected input dimensions.

        NEW BEHAVIOR (v2.1+): If checkpoint contains feature_metadata, use the stored
        feature_mask directly instead of re-running FeatureProcessor. This eliminates
        code duplication and ensures perfect consistency.

        FALLBACK: For old checkpoints without feature_metadata, fall back to config-based
        cleaning with FeatureProcessor (legacy behavior).
        """
        # Check if tokenizer was trained with feature cleaning
        config = tokenizer.config
        if hasattr(config, 'feature_cleaning') and config.feature_cleaning and config.feature_cleaning.enabled:
            temporal = features['temporal']
            actual_dim = temporal.shape[-1]

            print(f"\n⚠ Feature cleaning enabled in tokenizer, applying to dataset")
            print(f"  Input features: {actual_dim}")

            # NEW PATH: Use feature_metadata if available (v2.1+)
            if hasattr(tokenizer, 'feature_metadata') and tokenizer.feature_metadata is not None:
                print("✓ Using feature metadata from checkpoint (v2.1+)")

                # Validate dataset compatibility
                feature_metadata = tokenizer.feature_metadata
                if 'temporal' in feature_metadata.families:
                    temporal_family = feature_metadata.families['temporal']

                    # Check that dataset has expected number of features
                    if actual_dim != temporal_family.original_feature_count:
                        raise ValueError(
                            f"Dataset feature count mismatch!\n"
                            f"  Dataset: {actual_dim} features\n"
                            f"  Checkpoint expects: {temporal_family.original_feature_count} features\n"
                            f"  This means the dataset and checkpoint are incompatible."
                        )

                    # Use stored feature mask (no FeatureProcessor duplication!)
                    feature_mask = np.array(temporal_family.kept_feature_indices)
                    temporal_cleaned = temporal[:, :, feature_mask]
                    features['temporal'] = temporal_cleaned

                    print(f"✓ Feature mask loaded from checkpoint: {actual_dim} → {temporal_cleaned.shape[-1]} features")
                    print(f"  (Removed {len(temporal_family.removed_feature_indices)} features)")
                else:
                    raise ValueError("Checkpoint feature_metadata missing 'temporal' family")

            # FALLBACK PATH: Re-run FeatureProcessor for old checkpoints
            else:
                print("⚠ Checkpoint missing feature_metadata (v2.0 format), using fallback cleaning")
                from spinlock.encoding.feature_processor import FeatureProcessor

                # Aggregate temporal for cleaning analysis (use mean across time)
                temporal_agg = temporal.mean(axis=1)  # [N, D]

                # Initialize processor with tokenizer's config
                processor = FeatureProcessor(
                    variance_threshold=config.feature_cleaning.variance_threshold,
                    deduplicate_threshold=config.feature_cleaning.deduplicate_threshold,
                    use_intelligent_dedup=config.feature_cleaning.use_intelligent_dedup,
                    outlier_method=config.feature_cleaning.outlier_method,
                    percentile_range=config.feature_cleaning.percentile_range,
                    verbose=False,
                )

                # Clean features
                temporal_cleaned_np, feature_mask, _, _ = processor.clean(
                    temporal_agg,
                    feature_names=None,
                )

                # Apply mask to full temporal tensor
                temporal_cleaned = temporal[:, :, feature_mask]
                features['temporal'] = temporal_cleaned

                print(f"✓ Feature cleaning applied (fallback): {actual_dim} → {temporal_cleaned.shape[-1]} features")

        return features

    def _extract_truncation_lengths(
        self,
        tokenizer,
        features: Dict[str, np.ndarray],
    ) -> Optional[list]:
        """Extract truncation lengths for temporal resolution mode.

        Only works if tokenizer uses pyramid temporal encoder.

        Returns:
            Sorted list of truncation lengths [32, 64, 128, 256] or None if not applicable
        """
        config = tokenizer.config

        # Check if pyramid encoder is used
        if config.encoder.temporal.variant != "pyramid":
            self.error(
                f"Temporal resolution mode requires pyramid temporal encoder, "
                f"but tokenizer uses '{config.encoder.temporal.variant}'"
            )
            return None

        # Extract truncation lengths from variable_length.length_bins
        vl = config.encoder.temporal.variable_length
        if vl is None or not vl.length_bins:
            self.error(
                "Temporal resolution mode requires variable_length.length_bins "
                "in tokenizer config, but none found."
            )
            return None

        truncation_lengths = sorted(vl.length_bins)
        print(f"✓ Truncation lengths: {truncation_lengths}")

        return truncation_lengths

    def _analyze_token_structure(
        self,
        tokenizer,
        features: Dict[str, np.ndarray],
        device: torch.device,
        truncation_lengths: Optional[list] = None,
    ) -> Optional[list]:
        """Analyze token structure from first sample.

        If truncation_lengths is provided, returns all keys across all truncations.
        """
        print("\nAnalyzing token structure...")
        try:
            all_category_levels = set()

            # Determine truncation lengths to process
            if truncation_lengths is not None:
                lengths_to_process = truncation_lengths
            else:
                # Standard mode: just use full length
                lengths_to_process = [None]

            # Process each truncation length
            for trunc_len in lengths_to_process:
                with torch.no_grad():
                    # Prepare probe batch (first sample)
                    if features.get('dataset') is not None:
                        # Streaming mode: get first sample from dataset
                        sample = features['dataset'][0]
                        init_raw_batch = sample['ic'].unsqueeze(0).to(device)
                        theta_batch = sample['params'].unsqueeze(0).to(device)
                        temp_batch = None
                        init_manual_batch = None
                    else:
                        # Eager mode: slice from numpy arrays
                        if trunc_len is not None and features['temporal'] is not None:
                            temp_batch = torch.from_numpy(features['temporal'][:1, :trunc_len, :]).to(device)
                        else:
                            temp_batch = (
                                torch.from_numpy(features['temporal'][:1]).to(device)
                                if features['temporal'] is not None
                                else None
                            )

                        init_manual_batch = (
                            torch.from_numpy(features['initial_manual'][:1]).to(device)
                            if features['initial_manual'] is not None
                            else None
                        )
                        init_raw_batch = (
                            torch.from_numpy(features['initial_raw'][:1]).to(device)
                            if features['initial_raw'] is not None
                            else None
                        )
                        theta_batch = (
                            torch.from_numpy(features['theta'][:1]).to(device)
                            if features['theta'] is not None
                            else None
                        )

                    # Generate trajectories for learned mode
                    # Use trunc_len when in temporal resolution mode, else full timesteps
                    temporal_raw_batch = None
                    if self.is_learned_mode and theta_batch is not None and init_raw_batch is not None:
                        gen_timesteps = trunc_len if trunc_len is not None else (tokenizer.config.generation_timesteps or 64)
                        temporal_raw_batch = self._generate_trajectories(
                            theta_batch, init_raw_batch, gen_timesteps,
                        )  # [1, gen_timesteps+1, C, H, W] on CPU

                    # In temporal_only mode, don't pass theta/initial to tokenize
                    # (they're decoded from temporal tokens via aux heads, not separate families)
                    tok_theta = None if self.is_temporal_only else theta_batch
                    tok_init_manual = None if self.is_temporal_only else init_manual_batch
                    tok_init_raw = None if self.is_temporal_only else init_raw_batch

                    sample_tokens = tokenizer.tokenize(
                        temporal_features=temp_batch,
                        initial_manual=tok_init_manual,
                        initial_raw=tok_init_raw,
                        theta_features=tok_theta,
                        temporal_raw=temporal_raw_batch,
                    )

                # Add truncation suffix if temporal resolution mode
                if trunc_len is not None:
                    for key in sample_tokens.keys():
                        if "temporal" in key:
                            # Add truncation suffix
                            base_key, level_suffix = key.rsplit("_L", 1)
                            new_key = f"{base_key}_trunc_T{trunc_len:03d}_L{level_suffix}"
                            all_category_levels.add(new_key)
                        else:
                            # Initial/theta: store only once (will be saved from final truncation)
                            if trunc_len == lengths_to_process[-1]:
                                all_category_levels.add(key)
                else:
                    # Standard mode: use keys as-is
                    all_category_levels.update(sample_tokens.keys())

            category_levels = sorted(all_category_levels)
            print(f"✓ Found {len(category_levels)} category-levels")
            if truncation_lengths:
                temporal_keys = [k for k in category_levels if "temporal" in k and "trunc" in k]
                other_keys = [k for k in category_levels if k not in temporal_keys]
                print(f"  Temporal (with truncation): {len(temporal_keys)}")
                print(f"  Other (initial/theta): {len(other_keys)}")
            print(f"  Example keys: {', '.join(category_levels[:5])}...")
            return category_levels
        except Exception as e:
            self.error(f"Failed to analyze token structure: {e}")
            return None

    def _create_output_file(
        self,
        output_path: Path,
        features: Dict[str, np.ndarray],
        category_levels: list,
        copy_features: bool,
        truncation_lengths: Optional[list] = None,
    ) -> Optional[h5py.File]:
        """Create output HDF5 file with appropriate structure."""
        print(f"\nCreating output file: {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            f = h5py.File(output_path, 'w')

            # Create tokens group
            tokens_group = f.create_group('tokens')
            n = features['num_samples']
            chunk_size = min(n, 4096)
            for key in category_levels:
                tokens_group.create_dataset(
                    key,
                    shape=(n,),
                    dtype='int32',
                    chunks=(chunk_size,),
                    compression='gzip',
                    compression_opts=4,
                )

            # Store metadata about temporal resolution
            if truncation_lengths is not None:
                f.attrs["temporal_resolution_mode"] = True
                f.attrs["truncation_lengths"] = truncation_lengths
                f.attrs["num_truncations"] = len(truncation_lengths)

                # Count categories by type
                temporal_keys = [k for k in category_levels if "temporal" in k and "trunc" in k]
                initial_keys = [k for k in category_levels if "initial" in k]
                theta_keys = [k for k in category_levels if "theta" in k]

                f.attrs["num_temporal_categories"] = len(temporal_keys)
                f.attrs["num_initial_categories"] = len(initial_keys)
                f.attrs["num_theta_categories"] = len(theta_keys)
            else:
                f.attrs["temporal_resolution_mode"] = False

            # Optionally copy features
            if copy_features:
                has_copyable = any(
                    features[k] is not None
                    for k in ('temporal', 'initial_manual', 'initial_raw')
                )
                if not has_copyable:
                    print("  --copy-features: skipped (no eager features in streaming mode)")
                else:
                    print("  Copying features to output...")
                    features_group = f.create_group('features')
                    if features['temporal'] is not None:
                        features_group.create_dataset(
                            'temporal',
                            data=features['temporal'],
                            compression='gzip',
                        )
                    if features['initial_manual'] is not None:
                        features_group.create_dataset(
                            'initial_manual',
                            data=features['initial_manual'],
                            compression='gzip',
                        )
                    if features['initial_raw'] is not None:
                        features_group.create_dataset(
                            'initial_raw',
                            data=features['initial_raw'],
                            compression='gzip',
                        )

            return f
        except Exception as e:
            self.error(f"Failed to create output file: {e}")
            return None

    def _tokenize_mega_batch(
        self,
        tokenizer,
        mega_traj: torch.Tensor,
        mega_theta: torch.Tensor,
        mega_ic: torch.Tensor,
        all_vi: np.ndarray,
        lengths_to_process: list,
        device: torch.device,
        use_gpu_traj: bool,
    ) -> list:
        """Tokenize accumulated mega-batch at all truncation lengths.

        Processes the mega-batch in adaptive sub-batches sized per truncation
        length. Short truncations (T=32) use larger sub-batches (B=48-96) to
        saturate GPU compute; long truncations (T=256+) use smaller sub-batches
        (B=12-24) to fit in GPU memory.

        Args:
            tokenizer: Loaded VQTokenizer instance.
            mega_traj: [N, max_T+1, C, H, W] on CPU — concatenated trajectories.
            mega_theta: [N, P] on CPU — concatenated parameters.
            mega_ic: [N, C, H, W] on CPU — concatenated initial conditions.
            all_vi: [N] numpy array of virtual HDF5 write indices.
            lengths_to_process: Truncation lengths (or [None] for full-length).
            device: Computation device.
            use_gpu_traj: Whether to attempt GPU trajectory transfer.

        Returns:
            List of (save_key, tokens_np, start, end, indices) tuples for the writer.
        """
        N = mega_traj.shape[0]
        all_writes = []

        for trunc_len in lengths_to_process:
            if trunc_len is not None:
                traj = mega_traj[:, :trunc_len + 1]
            else:
                traj = mega_traj

            effective_T = trunc_len if trunc_len is not None else (mega_traj.shape[1] - 1)
            tok_batch = _compute_tokenization_batch_size(
                effective_T, mega_traj.shape[2], mega_traj.shape[3], device)
            tok_batch = min(tok_batch, N)

            sub_start = 0
            while sub_start < N:
                sub_end = min(sub_start + tok_batch, N)
                sub_traj = traj[sub_start:sub_end]

                if use_gpu_traj:
                    try:
                        sub_traj = sub_traj.to(device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        tok_batch = max(12, tok_batch // 2)
                        continue  # retry with smaller batch

                tok_theta = None if self.is_temporal_only else mega_theta[sub_start:sub_end].to(device)
                tok_ic = None if self.is_temporal_only else mega_ic[sub_start:sub_end].to(device)

                batch_tokens = tokenizer.tokenize(
                    temporal_features=None,
                    initial_manual=None,
                    initial_raw=tok_ic,
                    theta_features=tok_theta,
                    temporal_raw=sub_traj,
                )

                sub_vi = all_vi[sub_start:sub_end]
                for key, tokens in batch_tokens.items():
                    if trunc_len is not None and "temporal" in key:
                        base_key, level_suffix = key.rsplit("_L", 1)
                        save_key = f"{base_key}_trunc_T{trunc_len:03d}_L{level_suffix}"
                    elif trunc_len is not None:
                        # Initial/theta: save only from final truncation
                        if trunc_len != lengths_to_process[-1]:
                            continue
                        save_key = key
                    else:
                        save_key = key

                    tokens_np = tokens.cpu().numpy()
                    # Detect contiguous slice for efficient HDF5 writes
                    vi_sorted = np.sort(sub_vi)
                    contiguous = (
                        int(vi_sorted[-1]) - int(vi_sorted[0]) + 1 == len(vi_sorted)
                        and np.array_equal(sub_vi, vi_sorted)
                    )
                    all_writes.append((
                        save_key, tokens_np,
                        int(sub_vi[0]) if contiguous else 0,
                        int(sub_vi[-1]) + 1 if contiguous else 0,
                        None if contiguous else sub_vi,
                    ))

                sub_start = sub_end

        return all_writes

    def _batch_tokenize_streaming(
        self,
        tokenizer,
        dataset,
        category_levels: list,
        output_file: h5py.File,
        batch_size: int,
        device: torch.device,
        truncation_lengths: Optional[list] = None,
    ):
        """Tokenize dataset via DataLoader streaming and save to HDF5.

        Used for learned mode: ICs are read lazily per-batch from HDF5,
        trajectories generated via replayer, then tokenized.

        Optimizations:
        - Generate trajectory ONCE at max truncation length, then truncate
        - Accumulate N simulation batches on CPU → mega-batch tokenize at
          adaptive batch sizes (B=48-96 for short T, B=12-24 for long T)
        - encode_only path in model.forward() skips decoder + inverse heads
        - Async HDF5 writer overlaps disk I/O with GPU compute

        Args:
            tokenizer: Loaded VQTokenizer instance.
            dataset: SpinlockDataset in lazy_ics mode.
            category_levels: List of token key names for HDF5 output.
            output_file: Open HDF5 file with 'tokens' group pre-created.
            batch_size: Samples per batch (simulation batch size).
            device: Computation device.
            truncation_lengths: If set, tokenize at each truncation point.
        """
        from torch.utils.data import DataLoader

        tokens_group = output_file['tokens']
        generation_timesteps = tokenizer.config.generation_timesteps or 64

        if truncation_lengths is not None:
            lengths_to_process = sorted(truncation_lengths)
            max_gen_timesteps = max(lengths_to_process)
            print(
                f"\nTokenizing {dataset.n_samples:,} samples at "
                f"{len(truncation_lengths)} truncation lengths "
                f"[learned mode, streaming, generate-once, accumulated-batch]..."
            )
        else:
            lengths_to_process = [None]
            max_gen_timesteps = generation_timesteps
            print(
                f"\nTokenizing {dataset.n_samples:,} samples "
                f"(batch_size={batch_size}) [learned mode, streaming, accumulated-batch]..."
            )

        # Close any lazy HDF5 handle opened during token structure analysis,
        # so forked worker processes don't inherit a shared file descriptor.
        # Each worker will open its own handle via _ensure_h5_open().
        dataset.close_lazy()

        # CFL-sorted batching: group samples with similar CFL substep counts
        # to avoid the "one extreme sample forces K=32 for all 64" problem.
        # With Sobol params, 52% of samples need K=1 but batch-max K averages
        # 22 without sorting.  Sorting drops average batch K from 22 to ~3.
        sampler = None
        if self.is_learned_mode and hasattr(dataset, 'params') and self.replayer is not None:
            sampler = self._build_cfl_sorted_sampler(dataset, batch_size)

        use_pin_memory = device.type == "cuda"
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=3,
            prefetch_factor=2, shuffle=False, persistent_workers=False,
            pin_memory=use_pin_memory,
            sampler=sampler,
        )

        # GPU trajectory optimization: when the truncated trajectory fits on
        # GPU, the entire pipeline (pyramid pool1d, CNN, aggregation, VQ) runs
        # ~40× faster because pool1d on GPU avoids the 3 GB strided CPU access
        # and per-chunk CPU→GPU CNN transfers are eliminated.  We try-except
        # OOM to adaptively use GPU when possible, falling back to CPU.
        use_gpu_traj = device.type == "cuda"
        if use_gpu_traj:
            total_mem = torch.cuda.get_device_properties(device).total_memory
            print(f"  GPU trajectory mode: enabled ({total_mem / 1e9:.1f} GB GPU, try-except OOM)")

        # Compute accumulation count: how many sim batches to collect on CPU
        # before flushing as a single mega-batch for tokenization.
        # Detect grid size from first sample for memory estimation.
        sample0 = dataset[0]
        grid_size = sample0['ic'].shape[-1]
        n_channels = sample0['ic'].shape[0] if sample0['ic'].ndim >= 3 else 1
        accumulate_n = _compute_accumulation_count(
            batch_size, max_gen_timesteps, n_channels, grid_size)
        print(f"  Accumulation: {accumulate_n} sim batches → "
              f"~{accumulate_n * batch_size} samples per mega-batch tokenization")
        dataset.close_lazy()  # Close again after sample0 access

        buffer = _AccumulationBuffer(accumulate_n)

        # Async writer: GPU compute on batch N+1 overlaps with HDF5 I/O for batch N
        writer = _AsyncHDF5Writer(tokens_group)

        try:
            with torch.no_grad():
                for batch in tqdm(loader, desc="Batches", unit="batch"):
                    theta_batch = batch['params'].to(device)
                    ic_batch = batch['ic'].to(device)
                    virtual_indices = batch['virtual_idx']  # [B] unique DataLoader indices

                    # Phase 1: Simulate (GPU memory-constrained at sim batch_size)
                    full_traj = self._generate_trajectories(
                        theta_batch, ic_batch, max_gen_timesteps,
                    )  # [B, max_T+1, C, H, W] on CPU

                    # Phase 2: Accumulate on CPU
                    buffer.append(full_traj, theta_batch, ic_batch, virtual_indices)

                    # Phase 3: Flush when buffer full → mega-batch tokenize → write
                    if buffer.is_full:
                        mega_traj, mega_theta, mega_ic, all_vi = buffer.flush()
                        writes = self._tokenize_mega_batch(
                            tokenizer, mega_traj, mega_theta, mega_ic,
                            all_vi, lengths_to_process, device, use_gpu_traj)
                        writer.submit(writes)

                # Final flush for remaining samples
                if buffer.total_samples > 0:
                    mega_traj, mega_theta, mega_ic, all_vi = buffer.flush()
                    writes = self._tokenize_mega_batch(
                        tokenizer, mega_traj, mega_theta, mega_ic,
                        all_vi, lengths_to_process, device, use_gpu_traj)
                    writer.submit(writes)
        finally:
            writer.close()

    def _batch_tokenize_and_save(
        self,
        tokenizer,
        features: Dict[str, np.ndarray],
        category_levels: list,
        output_file: h5py.File,
        batch_size: int,
        device: torch.device,
        truncation_lengths: Optional[list] = None,
    ):
        """Tokenize dataset in batches and save to HDF5.

        If truncation_lengths is provided, tokenizes at each truncation point.
        """
        num_samples = features['num_samples']
        tokens_group = output_file['tokens']

        if truncation_lengths is not None:
            # Temporal resolution mode: tokenize at each truncation length
            # For learned mode: generate trajectory once at max length, then truncate
            sorted_lengths = sorted(truncation_lengths)
            max_trunc = max(sorted_lengths)
            mode_str = " [learned mode]" if self.is_learned_mode else ""
            print(f"\nTokenizing {num_samples:,} samples at {len(truncation_lengths)} truncation lengths{mode_str}...")
            num_batches = (num_samples + batch_size - 1) // batch_size

            writer = _AsyncHDF5Writer(tokens_group)
            try:
                with torch.no_grad():
                    for batch_idx in tqdm(range(num_batches), desc="Batches", unit="batch"):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, num_samples)

                        init_manual_batch = (
                            torch.from_numpy(features['initial_manual'][start_idx:end_idx]).to(device)
                            if features['initial_manual'] is not None
                            else None
                        )
                        init_raw_batch = (
                            torch.from_numpy(features['initial_raw'][start_idx:end_idx]).to(device)
                            if features['initial_raw'] is not None
                            else None
                        )
                        theta_batch = (
                            torch.from_numpy(features['theta'][start_idx:end_idx]).to(device)
                            if features['theta'] is not None
                            else None
                        )

                        # Generate trajectory once at max length (learned mode)
                        full_traj = None
                        if self.is_learned_mode and theta_batch is not None and init_raw_batch is not None:
                            full_traj = self._generate_trajectories(
                                theta_batch, init_raw_batch, max_trunc,
                            )  # [B, max_trunc+1, C, H, W] on CPU

                        writes = []
                        for trunc_len in sorted_lengths:
                            # Temporal features: truncate from pre-extracted or trajectory
                            temp_batch = (
                                torch.from_numpy(features['temporal'][start_idx:end_idx, :trunc_len, :]).to(device)
                                if features['temporal'] is not None
                                else None
                            )

                            temporal_raw_batch = None
                            if full_traj is not None:
                                temporal_raw_batch = full_traj[:, : trunc_len + 1]
                                # Move to GPU if it fits — avoids CPU pyramid bottleneck
                                if device.type == "cuda":
                                    try:
                                        temporal_raw_batch = temporal_raw_batch.to(device)
                                    except torch.cuda.OutOfMemoryError:
                                        torch.cuda.empty_cache()

                            # In temporal_only mode, don't pass theta/initial to tokenize
                            tok_theta = None if self.is_temporal_only else theta_batch
                            tok_init_manual = None if self.is_temporal_only else init_manual_batch
                            tok_init_raw = None if self.is_temporal_only else init_raw_batch

                            batch_tokens = tokenizer.tokenize(
                                temporal_features=temp_batch,
                                initial_manual=tok_init_manual,
                                initial_raw=tok_init_raw,
                                theta_features=tok_theta,
                                temporal_raw=temporal_raw_batch,
                            )

                            for key, tokens in batch_tokens.items():
                                tokens_np = tokens.cpu().numpy()
                                if "temporal" in key:
                                    base_key, level_suffix = key.rsplit("_L", 1)
                                    save_key = f"{base_key}_trunc_T{trunc_len:03d}_L{level_suffix}"
                                else:
                                    if trunc_len != sorted_lengths[-1]:
                                        continue
                                    save_key = key
                                writes.append((save_key, tokens_np, start_idx, end_idx, None))

                        writer.submit(writes)
            finally:
                writer.close()

        else:
            # Standard mode: tokenize once at full length
            mode_str = " [learned mode]" if self.is_learned_mode else ""
            print(f"\nTokenizing {num_samples:,} samples (batch_size={batch_size}){mode_str}...")
            num_batches = (num_samples + batch_size - 1) // batch_size
            generation_timesteps = tokenizer.config.generation_timesteps or 64

            writer = _AsyncHDF5Writer(tokens_group)
            try:
                with torch.no_grad():
                    for batch_idx in tqdm(range(num_batches), desc="Batches", unit="batch"):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, num_samples)

                        # Extract batch
                        temp_batch = (
                            torch.from_numpy(features['temporal'][start_idx:end_idx]).to(device)
                            if features['temporal'] is not None
                            else None
                        )
                        init_manual_batch = (
                            torch.from_numpy(features['initial_manual'][start_idx:end_idx]).to(device)
                            if features['initial_manual'] is not None
                            else None
                        )
                        init_raw_batch = (
                            torch.from_numpy(features['initial_raw'][start_idx:end_idx]).to(device)
                            if features['initial_raw'] is not None
                            else None
                        )
                        theta_batch = (
                            torch.from_numpy(features['theta'][start_idx:end_idx]).to(device)
                            if features['theta'] is not None
                            else None
                        )

                        # Generate trajectories for learned mode
                        temporal_raw_batch = None
                        if self.is_learned_mode and theta_batch is not None and init_raw_batch is not None:
                            temporal_raw_batch = self._generate_trajectories(
                                theta_batch, init_raw_batch, generation_timesteps,
                            )  # [B, T+1, C, H, W] on CPU

                        # In temporal_only mode, don't pass theta/initial to tokenize
                        tok_theta = None if self.is_temporal_only else theta_batch
                        tok_init_manual = None if self.is_temporal_only else init_manual_batch
                        tok_init_raw = None if self.is_temporal_only else init_raw_batch

                        # Tokenize
                        batch_tokens = tokenizer.tokenize(
                            temporal_features=temp_batch,
                            initial_manual=tok_init_manual,
                            initial_raw=tok_init_raw,
                            theta_features=tok_theta,
                            temporal_raw=temporal_raw_batch,
                        )

                        # Async write
                        writes = []
                        for key in category_levels:
                            tokens_np = batch_tokens[key].cpu().numpy()
                            writes.append((key, tokens_np, start_idx, end_idx, None))
                        writer.submit(writes)
            finally:
                writer.close()

    def _report_results(
        self,
        output_path: Path,
        features: Dict[str, np.ndarray],
        category_levels: list,
        truncation_lengths: Optional[list] = None,
    ):
        """Report tokenization results."""
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"\n✓ Pre-tokenization complete!")
        print(f"  Output: {output_path}")
        print(f"  Samples: {features['num_samples']:,}")
        print(f"  Category-levels: {len(category_levels)}")

        if truncation_lengths is not None:
            temporal_keys = [k for k in category_levels if "temporal" in k and "trunc" in k]
            other_keys = [k for k in category_levels if k not in temporal_keys]
            print(f"  Temporal resolution mode:")
            print(f"    Truncation lengths: {truncation_lengths}")
            print(f"    Temporal tokens: {len(temporal_keys)}")
            print(f"    Other tokens: {len(other_keys)}")

        print(f"  File size: {file_size_mb:.1f} MB")
