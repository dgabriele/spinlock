"""
Validate command for Spinlock CLI.

Validates dataset integrity and consistency.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class ValidateCommand(CLICommand):
    """
    Command to validate dataset integrity.

    Performs comprehensive validation checks on Spinlock datasets.
    """

    @property
    def name(self) -> str:
        return "validate"

    @property
    def help(self) -> str:
        return "Validate dataset integrity"

    @property
    def description(self) -> str:
        return """
Validate integrity and quality of a Spinlock dataset.

Performs various checks including:
- File structure validation
- Dimension consistency
- Metadata completeness
- Sample data ranges (optional)

Examples:
  # Basic validation
  spinlock validate --dataset datasets/default_10k.h5

  # Full validation with all checks
  spinlock validate --dataset datasets/default_10k.h5 --check-samples --check-metadata
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add validate command arguments."""
        parser.add_argument(
            "--dataset", type=Path, required=True, metavar="PATH", help="Path to HDF5 dataset file"
        )

        parser.add_argument(
            "--check-samples", action="store_true", help="Verify sample data ranges and statistics"
        )

        parser.add_argument(
            "--check-metadata", action="store_true", help="Verify metadata consistency"
        )

    def execute(self, args: Namespace) -> int:
        """Execute dataset validation."""
        if not self.validate_file_exists(args.dataset, "Dataset"):
            return 1

        try:
            return self._validate_dataset(
                args.dataset, check_samples=args.check_samples, check_metadata=args.check_metadata
            )
        except Exception as e:
            return self.error(f"Validation failed: {e}")

    def _validate_dataset(
        self, dataset_path: Path, check_samples: bool, check_metadata: bool
    ) -> int:
        """
        Validate dataset.

        Args:
            dataset_path: Path to HDF5 dataset
            check_samples: Verify sample data
            check_metadata: Verify metadata consistency

        Returns:
            Exit code (0 for success, 1 for validation failure)
        """
        import h5py
        import json
        import numpy as np
        from typing import cast, Optional, Tuple

        print("=" * 60)
        print(f"VALIDATING DATASET: {dataset_path.name}")
        print("=" * 60)

        passed = []
        failed = []
        warnings = []

        with h5py.File(dataset_path, "r") as f:
            # Check 1: Required groups exist
            print("\nChecking file structure...")
            required_groups = ["parameters", "inputs", "outputs"]
            for group in required_groups:
                if group in f:
                    passed.append(f"Group '{group}' exists")
                else:
                    failed.append(f"Missing required group: '{group}'")

            # Check 2: Dataset shapes consistency
            params_shape: Optional[Tuple[int, ...]] = None
            inputs_shape: Optional[Tuple[int, ...]] = None
            outputs_shape: Optional[Tuple[int, ...]] = None

            if all(g in f for g in required_groups):
                print("Checking dimension consistency...")
                if "parameters/params" in f:
                    params_shape = cast(h5py.Dataset, f["parameters/params"]).shape
                if "inputs/input_fields" in f:
                    inputs_shape = cast(h5py.Dataset, f["inputs/input_fields"]).shape
                if "outputs/operator_outputs" in f:
                    outputs_shape = cast(h5py.Dataset, f["outputs/operator_outputs"]).shape

                if params_shape and inputs_shape and outputs_shape:
                    n_params = params_shape[0]
                    n_inputs = inputs_shape[0]
                    n_outputs = outputs_shape[0]

                    if n_params == n_inputs == n_outputs:
                        passed.append(f"Dimension consistency: {n_params} samples")
                    else:
                        failed.append(
                            f"Dimension mismatch: params={n_params}, "
                            f"inputs={n_inputs}, outputs={n_outputs}"
                        )
                else:
                    warnings.append("Could not verify dimensions (missing datasets)")

            # Check 3: Metadata
            if "metadata" in f.attrs:
                metadata = json.loads(cast(str, f.attrs["metadata"]))
                passed.append("Metadata present")

                # Check metadata completeness
                if check_metadata:
                    print("Checking metadata completeness...")
                    required_fields = [
                        "version",
                        "creation_date",
                        "grid_size",
                        "num_parameter_sets",
                        "num_realizations",
                    ]
                    for field in required_fields:
                        if field in metadata:
                            passed.append(f"Metadata field '{field}' present")
                        else:
                            failed.append(f"Missing metadata field: '{field}'")

                    # Check metadata consistency with data
                    if "num_parameter_sets" in metadata and params_shape:
                        if metadata["num_parameter_sets"] == params_shape[0]:
                            passed.append("Metadata num_parameter_sets matches data")
                        else:
                            failed.append(
                                f"Metadata mismatch: num_parameter_sets "
                                f"({metadata['num_parameter_sets']}) != actual ({params_shape[0]})"
                            )
            else:
                warnings.append("No metadata found")

            # Check 4: Sample data ranges
            if check_samples:
                print("Checking sample data ranges...")

                # Check parameters in [0,1]
                if "parameters/params" in f:
                    params = cast(h5py.Dataset, f["parameters/params"])[:]
                    if np.all((params >= 0) & (params <= 1)):
                        passed.append("Parameters in valid range [0,1]")
                    else:
                        failed.append(
                            f"Parameters out of range: [{params.min():.3f}, {params.max():.3f}]"
                        )

                    # Check for NaN/Inf
                    if not np.any(np.isnan(params)) and not np.any(np.isinf(params)):
                        passed.append("Parameters contain no NaN/Inf")
                    else:
                        failed.append("Parameters contain NaN or Inf values")

                # Check outputs for NaN/Inf
                if "outputs/operator_outputs" in f:
                    # Sample first batch to avoid loading huge dataset
                    outputs = cast(h5py.Dataset, f["outputs/operator_outputs"])[:100]
                    if not np.any(np.isnan(outputs)) and not np.any(np.isinf(outputs)):
                        passed.append("Outputs contain no NaN/Inf (sampled)")
                    else:
                        failed.append("Outputs contain NaN or Inf values")

        # Print results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        if passed:
            print(f"\n✓ Passed ({len(passed)}):")
            for msg in passed:
                print(f"  • {msg}")

        if warnings:
            print(f"\n⚠ Warnings ({len(warnings)}):")
            for msg in warnings:
                print(f"  • {msg}")

        if failed:
            print(f"\n✗ Failed ({len(failed)}):")
            for msg in failed:
                print(f"  • {msg}")

        print("\n" + "=" * 60)
        if failed:
            print("✗ VALIDATION FAILED")
            print("=" * 60)
            return 1
        elif warnings:
            print("⚠ VALIDATION PASSED WITH WARNINGS")
            print("=" * 60)
            return 0
        else:
            print("✓ VALIDATION PASSED")
            print("=" * 60)
            return 0
