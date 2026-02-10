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

        # Analyze token structure
        category_levels = self._analyze_token_structure(tokenizer, features, device)
        if category_levels is None:
            return 1

        # Create output file
        output_file = self._create_output_file(
            args.output,
            features,
            category_levels,
            args.copy_features,
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
            )
        finally:
            output_file.close()

        # Report results
        self._report_results(args.output, features, category_levels)
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

    def _analyze_token_structure(
        self,
        tokenizer,
        features: Dict[str, np.ndarray],
        device: torch.device,
    ) -> Optional[list]:
        """Analyze token structure from first sample."""
        print("\nAnalyzing token structure...")
        try:
            with torch.no_grad():
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

            category_levels = sorted(sample_tokens.keys())
            print(f"✓ Found {len(category_levels)} category-levels")
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
    ) -> Optional[h5py.File]:
        """Create output HDF5 file with appropriate structure."""
        print(f"\nCreating output file: {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            f = h5py.File(output_path, 'w')

            # Create tokens group
            tokens_group = f.create_group('tokens')
            for key in category_levels:
                tokens_group.create_dataset(
                    key,
                    shape=(features['num_samples'],),
                    dtype='int32',
                    compression='gzip',
                    compression_opts=4,
                )

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
    ):
        """Tokenize dataset in batches and save to HDF5."""
        num_samples = features['num_samples']
        print(f"\nTokenizing {num_samples:,} samples (batch_size={batch_size})...")

        num_batches = (num_samples + batch_size - 1) // batch_size
        tokens_group = output_file['tokens']

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
    ):
        """Report tokenization results."""
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"\n✓ Pre-tokenization complete!")
        print(f"  Output: {output_path}")
        print(f"  Samples: {features['num_samples']:,}")
        print(f"  Category-levels: {len(category_levels)}")
        print(f"  File size: {file_size_mb:.1f} MB")
