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
import torch
from tqdm import tqdm

from .base import CLICommand


class PretokenizeDatasetCommand(CLICommand):
    """
    Command to pre-tokenize CNO dataset for fast diffusion training.

    Tokenizes all samples once using batch processing and saves tokens to HDF5,
    eliminating the need for on-the-fly tokenization during training.
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
            help="Path to input CNO dataset HDF5 file",
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

        # Load dataset features
        features = self._load_dataset_features(args.dataset)
        if features is None:
            return 1

        # Apply feature cleaning to match tokenizer's expected dimensions
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

    def _load_dataset_features(self, dataset_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Load features from SpinlockDataset."""
        from spinlock.data import SpinlockDataset

        print(f"\nLoading dataset from {dataset_path}")
        try:
            dataset = SpinlockDataset.from_file(str(dataset_path))
            with dataset.open():
                # Load temporal features
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

                if temporal is None and initial_manual is None:
                    self.error("No features found in dataset")
                    return None

                num_samples = temporal.shape[0] if temporal is not None else initial_manual.shape[0]
                print(f"✓ Loaded {num_samples:,} samples")

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

        # Extract truncation lengths from downsample_factors
        downsample_factors = config.encoder.temporal.downsample_factors
        max_timesteps = config.encoder.temporal.max_timesteps

        # Compute truncation points: max_timesteps / factor
        truncation_lengths = [max_timesteps // factor for factor in downsample_factors]
        truncation_lengths = sorted(truncation_lengths)

        # Validate against dataset
        if features['temporal'] is not None:
            dataset_timesteps = features['temporal'].shape[1]
            max_trunc = max(truncation_lengths)
            if max_trunc > dataset_timesteps:
                self.error(
                    f"Max truncation length {max_trunc} exceeds dataset timesteps {dataset_timesteps}"
                )
                return None

        print(f"✓ Extracted truncation lengths from pyramid config:")
        print(f"  max_timesteps={max_timesteps}, downsample_factors={downsample_factors}")
        print(f"  Truncation points: {truncation_lengths}")

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
                    # Prepare batch (truncate temporal if needed)
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

                    sample_tokens = tokenizer.tokenize(
                        temporal_features=temp_batch,
                        initial_manual=init_manual_batch,
                        initial_raw=init_raw_batch,
                        theta_features=theta_batch,
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
            print(f"\nTokenizing {num_samples:,} samples at {len(truncation_lengths)} truncation lengths...")

            for trunc_idx, trunc_len in enumerate(truncation_lengths):
                print(f"\n  Truncation {trunc_idx+1}/{len(truncation_lengths)}: t={trunc_len}")
                num_batches = (num_samples + batch_size - 1) // batch_size

                with torch.no_grad():
                    for batch_idx in tqdm(range(num_batches), desc=f"  t={trunc_len}", unit="batch"):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, num_samples)

                        # Extract batch (truncate temporal)
                        temp_batch = (
                            torch.from_numpy(features['temporal'][start_idx:end_idx, :trunc_len, :]).to(device)
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

                        # Tokenize
                        batch_tokens = tokenizer.tokenize(
                            temporal_features=temp_batch,
                            initial_manual=init_manual_batch,
                            initial_raw=init_raw_batch,
                            theta_features=theta_batch,
                        )

                        # Save tokens with truncation suffix
                        for key, tokens in batch_tokens.items():
                            if "temporal" in key:
                                # Add truncation suffix
                                base_key, level_suffix = key.rsplit("_L", 1)
                                save_key = f"{base_key}_trunc_T{trunc_len:03d}_L{level_suffix}"
                            else:
                                # Initial/theta: save only from final truncation
                                if trunc_len != truncation_lengths[-1]:
                                    continue
                                save_key = key

                            tokens_cpu = tokens.cpu().numpy()
                            tokens_group[save_key][start_idx:end_idx] = tokens_cpu

        else:
            # Standard mode: tokenize once at full length
            print(f"\nTokenizing {num_samples:,} samples (batch_size={batch_size})...")
            num_batches = (num_samples + batch_size - 1) // batch_size

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

                    # Tokenize
                    batch_tokens = tokenizer.tokenize(
                        temporal_features=temp_batch,
                        initial_manual=init_manual_batch,
                        initial_raw=init_raw_batch,
                        theta_features=theta_batch,
                    )

                    # Save tokens
                    for key in category_levels:
                        tokens_cpu = batch_tokens[key].cpu().numpy()
                        tokens_group[key][start_idx:end_idx] = tokens_cpu

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
