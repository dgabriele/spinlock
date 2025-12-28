"""
Info command for Spinlock CLI.

Displays dataset information and metadata.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class InfoCommand(CLICommand):
    """
    Command to display dataset information.

    Shows metadata, dimensions, and optional detailed statistics.
    """

    @property
    def name(self) -> str:
        return "info"

    @property
    def help(self) -> str:
        return "Display dataset information"

    @property
    def description(self) -> str:
        return """
Display information about a Spinlock dataset.

Shows dataset dimensions, metadata, and optional detailed statistics.

Examples:
  # Basic information
  spinlock info --dataset datasets/default_10k.h5

  # Detailed information with metadata
  spinlock info --dataset datasets/default_10k.h5 --verbose
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add info command arguments."""
        parser.add_argument(
            "--dataset", type=Path, required=True, metavar="PATH", help="Path to HDF5 dataset file"
        )

        parser.add_argument(
            "--verbose", action="store_true", help="Show detailed information including metadata"
        )

    def execute(self, args: Namespace) -> int:
        """Execute dataset info command."""
        if not self.validate_file_exists(args.dataset, "Dataset"):
            return 1

        try:
            return self._print_dataset_info(args.dataset, args.verbose)
        except Exception as e:
            return self.error(f"Failed to read dataset: {e}")

    def _print_dataset_info(self, dataset_path: Path, verbose: bool) -> int:
        """
        Print dataset information.

        Args:
            dataset_path: Path to HDF5 dataset
            verbose: Show detailed metadata

        Returns:
            Exit code (0 for success)
        """
        import h5py
        import json
        from typing import cast

        with h5py.File(dataset_path, "r") as f:
            print("=" * 60)
            print(f"SPINLOCK DATASET: {dataset_path.name}")
            print("=" * 60)

            # Dataset dimensions
            print("\nDimensions:")
            for key in f.keys():
                dataset = f[key]
                if isinstance(dataset, h5py.Group):
                    # Group - show subdatasets
                    print(f"  {key}/ (group):")
                    for subkey in dataset.keys():
                        subdataset = dataset[subkey]
                        if isinstance(subdataset, h5py.Dataset):
                            print(f"    {subkey}: {subdataset.shape} {subdataset.dtype}")
                elif isinstance(dataset, h5py.Dataset):
                    # Dataset - show shape and dtype
                    print(f"  {key}: {dataset.shape} {dataset.dtype}")

            # Metadata
            if "metadata" in f.attrs:
                metadata = json.loads(cast(str, f.attrs["metadata"]))
                print("\nMetadata:")
                print(f"  Version: {metadata.get('version', 'unknown')}")
                print(f"  Created: {metadata.get('creation_date', 'unknown')}")
                print(f"  Grid size: {metadata.get('grid_size', 'unknown')}")
                print(f"  Parameter sets: {metadata.get('num_parameter_sets', 'unknown')}")
                print(f"  Realizations: {metadata.get('num_realizations', 'unknown')}")

                # Sampling metrics
                if "sampling_metrics" in metadata and verbose:
                    metrics = metadata["sampling_metrics"]
                    print("\nSampling Metrics:")
                    if "discrepancy" in metrics:
                        status = "✓" if metrics.get("discrepancy_pass", False) else "✗"
                        print(f"  {status} Discrepancy: {metrics['discrepancy']:.6f}")
                    if "max_correlation" in metrics:
                        status = "✓" if metrics.get("correlation_pass", False) else "✗"
                        print(f"  {status} Max correlation: {metrics['max_correlation']:.6f}")

                # Full metadata dump
                if verbose:
                    print("\nFull Metadata:")
                    print(json.dumps(metadata, indent=2))

            # Storage info
            file_size_mb = dataset_path.stat().st_size / (1024**2)
            print(f"\nStorage:")
            print(f"  File size: {file_size_mb:.2f} MB")

            print("=" * 60)

        return 0
