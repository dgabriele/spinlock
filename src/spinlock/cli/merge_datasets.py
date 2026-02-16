"""
Merge multiple HDF5 datasets command.

Handles merging of distributed dataset generation results.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from glob import glob
import sys

from .base import CLICommand


class MergeDatasetsCommand(CLICommand):
    """Command to merge multiple HDF5 datasets into a single file."""

    @property
    def name(self) -> str:
        return "merge-datasets"

    @property
    def help(self) -> str:
        return "Merge multiple HDF5 datasets into a single file"

    @property
    def description(self) -> str:
        return """
Merge multiple HDF5 datasets into a single file.

This command is used after distributed generation to combine results:
1. Reads all matching HDF5 files
2. Concatenates datasets along the sample dimension
3. Validates no duplicate parameter sets (Sobol determinism check)
4. Writes merged HDF5 file

Examples:
  # Merge all parts
  spinlock merge-datasets \\
      --input "datasets/parts/qbm_1m_wide_part*.h5" \\
      --output datasets/qbm_1m_wide.h5

  # Merge with validation disabled (faster)
  spinlock merge-datasets \\
      --input "datasets/parts/*.h5" \\
      --output datasets/merged.h5 \\
      --no-validate

  # Verbose output
  spinlock merge-datasets \\
      --input "datasets/parts/*.h5" \\
      --output datasets/merged.h5 \\
      --verbose
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add merge-datasets command arguments."""
        parser.add_argument(
            "--input",
            type=str,
            required=True,
            metavar="PATTERN",
            help="Input file pattern (e.g., 'datasets/parts/qbm_*_part*.h5')",
        )

        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            metavar="PATH",
            help="Output merged HDF5 file",
        )

        parser.add_argument(
            "--validate/--no-validate",
            dest="validate",
            action="store_true",
            default=True,
            help="Validate no duplicate parameter sets (default: enabled)",
        )

        parser.add_argument(
            "--verbose", action="store_true", help="Print detailed progress information"
        )

    def execute(self, args: Namespace) -> int:
        """Execute dataset merge."""
        import h5py
        import numpy as np

        # Find input files
        input_paths = sorted(glob(args.input))

        if not input_paths:
            print(f"Error: No files matching pattern: {args.input}", file=sys.stderr)
            return 1

        print(f"[MergeDatasets] Found {len(input_paths)} files to merge")

        if args.verbose:
            for p in input_paths:
                print(f"  - {p}")

        # Create output directory if needed
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Merge datasets
        try:
            print(f"[MergeDatasets] Merging into {args.output}...")
            self._merge_hdf5_files(
                input_paths, args.output, validate=args.validate, verbose=args.verbose
            )
        except KeyboardInterrupt:
            print("\n\nMerge interrupted by user", file=sys.stderr)
            # Clean up partial output
            if args.output.exists():
                args.output.unlink()
            return 130
        except Exception as e:
            print(f"\nError during merge: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            # Clean up partial output
            if args.output.exists():
                args.output.unlink()
            return 1

        # Success summary
        print("\n" + "=" * 60)
        print("✓ MERGE COMPLETE")
        print("=" * 60)
        print(f"Output: {args.output}")

        # Print dataset info
        try:
            with h5py.File(args.output, "r") as f:
                if "/parameters/params" in f:
                    n_samples = f["/parameters/params"].shape[0]
                    print(f"Total samples: {n_samples:,}")

                if "/features/temporal/per_timestep" in f:
                    features_shape = f["/features/temporal/per_timestep"].shape
                    print(f"Features shape: {features_shape}")

                # File size
                size_mb = args.output.stat().st_size / (1024**2)
                size_gb = size_mb / 1024
                if size_gb >= 1:
                    print(f"File size: {size_gb:.2f} GB")
                else:
                    print(f"File size: {size_mb:.2f} MB")
        except Exception as e:
            print(f"Warning: Could not read output file info: {e}")

        print("=" * 60)

        return 0

    def _merge_hdf5_files(
        self, input_paths: list, output_path: Path, validate: bool, verbose: bool
    ) -> None:
        """
        Merge multiple HDF5 files into one.

        Args:
            input_paths: List of input HDF5 file paths
            output_path: Output HDF5 file path
            validate: Whether to validate no duplicates
            verbose: Print detailed progress
        """
        import h5py
        import numpy as np

        # Datasets to concatenate (common structure)
        datasets_to_merge = [
            "/parameters/params",
            "/features/initial/aggregated",
            "/features/temporal/per_timestep",
        ]

        with h5py.File(output_path, "w") as out_f:
            # Read first file to get structure and metadata
            with h5py.File(input_paths[0], "r") as f0:
                # Copy root attributes
                for attr_name, attr_val in f0.attrs.items():
                    out_f.attrs[attr_name] = attr_val

                # Determine which datasets exist
                existing_datasets = [ds for ds in datasets_to_merge if ds in f0]

                if not existing_datasets:
                    raise ValueError(
                        f"No standard datasets found in {input_paths[0]}. "
                        f"Expected one of: {datasets_to_merge}"
                    )

                # For each dataset, determine total size
                for ds_path in existing_datasets:
                    if verbose:
                        print(f"[MergeDatasets] Processing {ds_path}...")

                    # Collect shapes from all files
                    shapes = []
                    for p in input_paths:
                        with h5py.File(p, "r") as f:
                            if ds_path not in f:
                                print(
                                    f"  Warning: {ds_path} not in {p}, skipping",
                                    file=sys.stderr,
                                )
                                continue
                            shapes.append(f[ds_path].shape)

                    if not shapes:
                        print(f"  Warning: {ds_path} not found in any input files", file=sys.stderr)
                        continue

                    # Calculate total samples
                    total_samples = sum(s[0] for s in shapes)

                    # Verify all have same dimensions (except first)
                    if len(set(s[1:] for s in shapes)) > 1:
                        raise ValueError(
                            f"Inconsistent shapes for {ds_path}: {shapes}"
                        )

                    # Create concatenated dataset
                    if verbose:
                        print(f"  Creating dataset: {total_samples:,} samples, shape {(total_samples, *shapes[0][1:])}")

                    dset = out_f.create_dataset(
                        ds_path,
                        shape=(total_samples, *shapes[0][1:]),
                        dtype=f0[ds_path].dtype,
                        compression="gzip",
                        compression_opts=4,
                    )

                    # Copy attributes
                    for attr_name, attr_val in f0[ds_path].attrs.items():
                        dset.attrs[attr_name] = attr_val

                    # Write each chunk
                    offset = 0
                    for idx, p in enumerate(input_paths):
                        with h5py.File(p, "r") as f:
                            if ds_path not in f:
                                continue

                            data = f[ds_path][:]
                            dset[offset : offset + len(data)] = data

                            if verbose:
                                print(
                                    f"    [{idx+1}/{len(input_paths)}] "
                                    f"{Path(p).name}: {len(data):,} samples → "
                                    f"offset {offset:,}"
                                )

                            offset += len(data)

                    if verbose:
                        print(f"  ✓ Merged {ds_path}: {total_samples:,} total samples")

        # Validation
        if validate:
            self._validate_merged_dataset(output_path, verbose)

    def _validate_merged_dataset(self, output_path: Path, verbose: bool) -> None:
        """
        Validate merged dataset for duplicates.

        Args:
            output_path: Path to merged HDF5 file
            verbose: Print detailed progress
        """
        import h5py
        import numpy as np

        print("[MergeDatasets] Validating merged dataset...")

        with h5py.File(output_path, "r") as f:
            if "/parameters/params" not in f:
                print("  Warning: No parameters found, skipping validation")
                return

            params = f["/parameters/params"][:]

            # Check for duplicate parameter sets
            # Convert to tuples for set comparison (rows are parameter vectors)
            if verbose:
                print(f"  Checking {len(params):,} parameter sets for duplicates...")

            # For large datasets, use a hash-based approach
            param_hashes = set()
            duplicates = 0

            for idx, p in enumerate(params):
                # Create hash of parameter vector
                p_hash = hash(tuple(p))

                if p_hash in param_hashes:
                    duplicates += 1
                else:
                    param_hashes.add(p_hash)

            if duplicates > 0:
                print(
                    f"  ⚠ WARNING: Found {duplicates} duplicate parameter sets "
                    f"({100*duplicates/len(params):.2f}%)",
                    file=sys.stderr,
                )
                print(
                    "    This may indicate Sobol offset issue or seed collision",
                    file=sys.stderr,
                )
            else:
                print(f"  ✓ No duplicate parameter sets found")

            # Check parameter distribution
            if verbose:
                print(f"  Parameter ranges:")
                for dim in range(params.shape[1]):
                    p_min = params[:, dim].min()
                    p_max = params[:, dim].max()
                    p_mean = params[:, dim].mean()
                    print(f"    Dim {dim}: [{p_min:.6f}, {p_max:.6f}] (mean={p_mean:.6f})")

        print("[MergeDatasets] ✓ Validation complete")
