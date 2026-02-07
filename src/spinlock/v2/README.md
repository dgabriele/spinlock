# Spinlock V2 Data API

Type-safe, validated HDF5 dataset interface built from the ground up with modern Python best practices.

## Overview

The V2 data API provides a clean, type-safe interface for working with Spinlock HDF5 datasets. It replaces the error-prone dict-based h5py access with Pydantic-validated models and lazy loading for memory efficiency.

## Key Features

- **Type Safety**: Pydantic models ensure runtime validation and catch schema deviations
- **Lazy Loading**: Data isn't loaded into memory until accessed
- **Clean API**: Context manager support and intuitive property access
- **Validation**: Comprehensive validation with detailed error reporting
- **Sanitization**: Configurable data cleaning with builder pattern
- **Zero Migration**: Built from scratch, no backward compatibility concerns

## Quick Start

### Loading a Dataset

```python
from spinlock.v2.data import SpinlockDataset

# Load with automatic validation
dataset = SpinlockDataset.from_file("data/50k_baseline.h5")

# Type-safe metadata access
print(f"Samples: {dataset.metadata.num_parameter_sets}")
print(f"Grid: {dataset.metadata.grid_size}")
print(f"Created: {dataset.metadata.creation_date}")
```

### Accessing Data (Lazy Loading)

```python
# Use context manager for safe file access
with dataset.open():
    # Load specific slices (doesn't load entire dataset)
    params_batch = dataset.parameters[0:100]  # [100, P]
    first_input = dataset.inputs[0]           # [M, C, H, W]

    # Nested feature access
    temporal = dataset.features.temporal[0:10]              # [10, T, D]
    summary_agg = dataset.features.summary.aggregated[:]    # [N, D]
    summary_per = dataset.features.summary.per_trajectory[0] # [M, D]
```

### Validation

```python
# Run validation with detailed report
report = dataset.validate(strict=False)

if not report["valid"]:
    print(f"Errors: {report['errors']}")
    print(f"Warnings: {report['warnings']}")

# Checks performed
print(report["checks"])
# {
#   "schema": True,
#   "shapes": True,
#   "dtypes": True,
#   "nans": True,
#   "ranges": True,
# }
```

### Sanitization

#### Default Parameters (Functional Style)

```python
# Use default sanitization params
clean_dataset = dataset.sanitize(inplace=False)
clean_dataset.save("clean.h5")
```

#### Custom Parameters (Functional Style)

```python
from spinlock.v2.data.models import SanitizationParams

# Create custom params
params = SanitizationParams(
    remove_nans=True,
    remove_zero_variance=True,
    variance_threshold=1e-10,
)

# Immutable transformation (returns new instance)
clean_dataset = dataset.sanitize(params=params, inplace=False)
clean_dataset.save("clean.h5")
```

#### Custom Parameters (Numpy-style Inplace)

```python
params = SanitizationParams(
    remove_nans=True,
    remove_outliers=True,
    outlier_std_threshold=5.0,
)

# Mutable transformation (modifies in-place, returns self)
dataset.sanitize(params=params, inplace=True).save("sanitized.h5")
```

### VQ-VAE Training Integration

```python
from spinlock.v2.data import SpinlockDataset
from spinlock.v2.data.models import SanitizationParams
from spinlock.config import load_config

# Load config with sanitization params
config = load_config("configs/vqvae/50k_baseline.yaml")

# Config YAML:
# dataset:
#   path: "data/50k_baseline.h5"
#   sanitization:
#     remove_nans: true
#     replace_nan_value: 0.0
#     remove_zero_variance: true
#     variance_threshold: 1e-10

# Load dataset
dataset = SpinlockDataset.from_file(config.dataset.path)

# Parse sanitization params from config
san_params = SanitizationParams(**config.dataset.sanitization)

# Apply sanitization
clean_dataset = dataset.sanitize(params=san_params, inplace=False)

# Proceed with training
train_vqvae(clean_dataset, config)
```

### Creating a New Dataset

```python
from spinlock.v2.data import DatasetMetadata, ShapeDescriptor
from datetime import datetime

metadata = DatasetMetadata(
    creation_date=datetime.now(),
    grid_size=64,
    num_channels=3,
    num_realizations=10,
    num_parameter_sets=1000,
)

params_shape = ShapeDescriptor(n_samples=1000, feature_dim=13)
inputs_shape = ShapeDescriptor(n_samples=1000, n_realizations=10, grid_size=64)

new_dataset = SpinlockDataset.create(
    path="new_dataset.h5",
    metadata=metadata,
    parameters_shape=params_shape,
    inputs_shape=inputs_shape,
)
```

## Architecture

### Pydantic Models (`models.py`)

Type-safe models for schema validation:

- `DatasetMetadata`: Metadata with automatic type conversion
- `ShapeDescriptor`: Semantic shape validation
- `ParametersData`, `FieldsData`, `FeaturesData`: Data component schemas
- `SanitizationParams`: Configurable sanitization parameters
- `DatasetSchema`: Top-level schema with cross-field validation

### SpinlockDataset Class (`dataset.py`)

Main interface for dataset operations:

- `from_file()`: Load existing dataset
- `create()`: Create new dataset
- `open()`: Context manager for safe access
- Properties: `metadata`, `parameters`, `inputs`, `outputs`, `features`
- `validate()`: Comprehensive validation
- `sanitize()`: Configurable data cleaning
- `save()`: Save to new location
- `summary()`: Human-readable summary

### Lazy Loading (`lazy.py`)

Memory-efficient data access:

- `LazyDataArray`: Wrapper that loads data on-demand
- `FeatureAccessor`: Nested accessor for feature families
- `SummaryFeatureAccessor`: Handles per_trajectory and aggregated

### Validation (`validation.py`)

Comprehensive validation with detailed reports:

- Schema validation (via Pydantic)
- Shape consistency checks
- Data type verification
- NaN detection (warnings)
- Value range checks (warnings)

### Sanitization (`sanitize.py`)

Configurable data cleaning operations:

- NaN replacement
- Zero-variance feature removal
- Outlier capping (MAD or std-based)
- Pre-allocated sample removal
- Builder pattern with `inplace` parameter

## HDF5 File Structure

```
/metadata/          - Attributes (creation_date, version, grid_size, etc.)
/parameters/params  - [N, P] parameter vectors
/inputs/fields      - [N, M, C, H, W] input fields
/outputs/fields     - [N, M, C, H, W] or [N, M, T, C, H, W] outputs
/features/
  /temporal/features         - [N, T, D]
  /summary/per_trajectory/features - [N, M, D]
  /summary/aggregated/features     - [N, D]
  /initial/aggregated/features     - [N, D]
```

## Design Principles

### DRY (Don't Repeat Yourself)

- Single `SpinlockDataset` class handles all operations
- Pydantic handles validation automatically
- `LazyDataArray` reused for all data types

### OOP Best Practices

- **Single Responsibility**: Validator, Sanitizer, Dataset are separate
- **Open/Closed**: Extensible via new Pydantic models
- **Factory Pattern**: `from_file()`, `create()` class methods
- **Builder Pattern**: `sanitize()` with `inplace` parameter
- **Context Manager**: Safe resource management

### Type Safety

- Pydantic ensures runtime validation
- Type hints everywhere for IDE support
- Semantic shape descriptors (not just tuples)

### Performance

- Lazy loading (don't load until needed)
- Efficient slicing via h5py
- Optional in-place operations

## Testing

Run tests with pytest:

```bash
# All v2 tests
pytest tests/v2/

# Specific test modules
pytest tests/v2/data/test_models.py
pytest tests/v2/data/test_dataset.py

# With coverage
pytest tests/v2/ --cov=src/spinlock/v2
```

## Examples

See `tests/v2/data/test_dataset.py` for comprehensive examples of:

- Creating datasets
- Loading and validation
- Lazy loading and data access
- Sanitization with various parameters
- Context manager usage

## Future Enhancements

Planned for future versions:

- Feature name registry for semantic access
- Automatic schema version migration
- Parallel sanitization operations
- Integration with feature category discovery
- Performance profiling and optimization
