"""
Example usage of the V2 Dataset API.

Demonstrates loading, validation, and sanitization of datasets.
"""

from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import h5py

from spinlock.v2.data import (
    SpinlockDataset,
    DatasetMetadata,
    ShapeDescriptor,
    SanitizationParams,
)


def create_example_dataset():
    """Create a simple example dataset."""
    temp_path = Path(tempfile.gettempdir()) / "example_dataset.h5"

    n_samples = 100
    n_params = 13
    n_realizations = 10
    grid_size = 64
    n_channels = 3

    with h5py.File(temp_path, "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["version"] = "1.0"
        meta.attrs["creation_date"] = datetime.now().isoformat()
        meta.attrs["grid_size"] = grid_size
        meta.attrs["num_channels"] = n_channels
        meta.attrs["num_realizations"] = n_realizations
        meta.attrs["num_parameter_sets"] = n_samples
        meta.attrs["track_ic_metadata"] = False
        meta.attrs["compression"] = "gzip"

        # Parameters
        f.create_dataset(
            "/parameters/params",
            data=np.random.randn(n_samples, n_params).astype(np.float32),
            compression="gzip",
        )

        # Temporal features (with some NaNs)
        features = np.random.randn(n_samples, 50, 64).astype(np.float32)
        features[0:5, :, 0:3] = np.nan  # Add some NaNs

        f.create_dataset(
            "/features/temporal/features",
            data=features,
            compression="gzip",
        )

        # Summary features
        f.create_dataset(
            "/features/summary/aggregated/features",
            data=np.random.randn(n_samples, 360).astype(np.float32),
            compression="gzip",
        )

    return temp_path


def main():
    """Run example demonstrating V2 API."""
    print("=" * 60)
    print("Spinlock V2 Dataset API Example")
    print("=" * 60)

    # Create example dataset
    print("\n1. Creating example dataset...")
    dataset_path = create_example_dataset()
    print(f"   Created: {dataset_path}")

    # Load dataset
    print("\n2. Loading dataset with validation...")
    dataset = SpinlockDataset.from_file(dataset_path, validate=True)
    print(f"   Loaded successfully")

    # Display metadata
    print("\n3. Dataset metadata:")
    print(f"   Samples: {dataset.metadata.num_parameter_sets}")
    print(f"   Grid: {dataset.metadata.grid_size}")
    print(f"   Channels: {dataset.metadata.num_channels}")
    print(f"   Created: {dataset.metadata.creation_date}")

    # Validate dataset
    print("\n4. Running validation...")
    report = dataset.validate(strict=False)
    print(f"   Valid: {report['valid']}")
    print(f"   Errors: {len(report['errors'])}")
    print(f"   Warnings: {len(report['warnings'])}")

    if report["warnings"]:
        print("   Warning messages:")
        for warning in report["warnings"][:3]:  # Show first 3
            print(f"     - {warning}")

    # Access data
    print("\n5. Accessing data (lazy loading)...")
    with dataset.open():
        # Load specific slices
        params = dataset.parameters[0:10]
        print(f"   Parameters shape: {params.shape}")

        temporal = dataset.features.temporal[0:5]
        print(f"   Temporal features shape: {temporal.shape}")
        print(f"   Contains NaNs: {np.any(np.isnan(temporal))}")

    # Sanitize dataset
    print("\n6. Sanitizing dataset...")
    san_params = SanitizationParams(
        remove_nans=True,
        replace_nan_value=0.0,
        remove_zero_variance=False,
    )

    clean_dataset = dataset.sanitize(params=san_params, inplace=False)
    print(f"   Created sanitized dataset: {clean_dataset.path}")

    # Verify sanitization
    print("\n7. Verifying sanitization...")
    with clean_dataset.open():
        clean_temporal = clean_dataset.features.temporal[:]
        print(f"   Contains NaNs: {np.any(np.isnan(clean_temporal))}")
        print(f"   All values finite: {np.all(np.isfinite(clean_temporal))}")

    # Display summary
    print("\n8. Dataset summary:")
    print(dataset.summary())

    # Cleanup
    print("\n9. Cleaning up...")
    dataset_path.unlink()
    if clean_dataset.path.exists():
        clean_dataset.path.unlink()
    print("   Done")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
