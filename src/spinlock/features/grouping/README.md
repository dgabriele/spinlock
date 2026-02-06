# V2 Feature Grouping System

A clean, inheritance-based architecture for feature grouping (formerly "category discovery") that abstracts clustering and gradient-based refinement from VQ-VAE training.

## Key Features

- **Family-specific groupers**: `TemporalFeatureGrouper`, `InitialFeatureGrouper`
- **Sequential pipeline**: Clustering → Gradient refinement
- **GPU-first**: CuPy required for clustering, PyTorch for gradient refinement
- **Type-safe**: Pydantic configuration models
- **Separation of concerns**: Standalone module independent of VQ-VAE training
- **Reusable**: Works with any feature extraction pipeline

## Architecture

```
FeatureGrouper (Abstract Base Class)
    ├── TemporalFeatureGrouper
    └── InitialFeatureGrouper

Shared Components (Composition):
    ├── ClusteringEngine - GPU hierarchical clustering
    ├── GradientRefiner - Gumbel-Softmax optimization
    └── RecursiveSplitter - Mega-group decomposition
```

## Quick Start

### Basic Usage

```python
from spinlock.v2.data import SpinlockDataset, SanitizationParams
from spinlock.features.grouping import create_grouper

# Load and sanitize dataset FIRST
dataset = SpinlockDataset.from_file("data/50k_baseline.h5")
san_params = SanitizationParams(
    remove_nans=True,
    remove_zero_variance=True,
)
clean_dataset = dataset.sanitize(params=san_params, inplace=False)

# Extract temporal features
with clean_dataset.open():
    temporal_features = clean_dataset.features.temporal.load_all()

# Aggregate over time
temporal_agg = temporal_features.mean(axis=1)  # [N, D]

# Get feature names
feature_names = [f"temporal_{i}" for i in range(temporal_agg.shape[1])]

# Create grouper with default config
grouper = create_grouper("temporal")

# Group features (clustering → gradient pipeline)
result = grouper.group_features(temporal_agg, feature_names)

# Access results
print(f"Discovered {result.num_groups} groups")
for name, group in result.groups.items():
    print(f"{name}: {group.size} features")

# Convert to dict for VQ-VAE
group_dict = result.to_dict()
```

### Custom Configuration

```python
from spinlock.features.grouping import (
    create_grouper,
    TemporalGroupingConfig,
    ClusteringParams,
    GradientParams,
)

# Custom config for temporal features
config = TemporalGroupingConfig()
config.clustering = ClusteringParams(
    min_groups=10,
    max_groups=25,
    use_gpu=True,
)
config.gradient = GradientParams(
    num_epochs=1000,
    orthogonality_target=0.10,
)

# Create grouper
grouper = create_grouper("temporal", config=config)

# Group features
result = grouper.group_features(features, feature_names)
```

## Pipeline Details

### Stage 1: Clustering Initialization

The clustering engine performs hierarchical clustering with GPU acceleration:

1. **Distance Computation** (GPU via CuPy)
   - Correlation distance: `1 - |Pearson correlation|`
   - Alternative: Euclidean, Cosine

2. **Hierarchical Clustering**
   - Linkage: Ward, Average, Complete, Single
   - K-selection: Silhouette score, Gap statistic, Elbow method

3. **Optional Recursive Splitting**
   - Breaks oversized groups (> `max_group_size`)
   - Max recursion depth: 3

### Stage 2: Gradient Refinement (Optional)

The gradient refiner uses Gumbel-Softmax for differentiable assignments:

1. **Initialization**: From clustering results
2. **Optimization**: Adam optimizer with temperature annealing
3. **Objectives**:
   - **Orthogonality**: Minimize inter-group correlation
   - **Informativeness**: Maximize per-group variance
   - **Custom loss**: Optional (e.g., VQ-VAE reconstruction)

## Configuration

### Clustering Parameters

```python
ClusteringParams(
    linkage_method=LinkageMethod.WARD,
    distance_metric=DistanceMetric.CORRELATION,
    k_selection_method=KSelectionMethod.SILHOUETTE,
    num_groups=None,  # Auto-select if None
    min_groups=2,
    max_groups=20,
    subsample_size=None,  # Subsample for large datasets
    use_gpu=True,  # Required
)
```

### Gradient Parameters

```python
GradientParams(
    num_epochs=500,
    learning_rate=0.01,
    temperature_start=1.0,
    temperature_end=0.5,
    orthogonality_weight=1.0,
    informativeness_weight=1.0,
    custom_loss_weight=0.0,  # Enable custom loss injection
    orthogonality_target=0.15,  # Early stopping
    device="auto",  # "cuda", "cpu", or "auto"
)
```

### Preprocessing Parameters

```python
PreprocessingParams(
    method="mad",  # "mad", "zscore", "minmax", "none"
    mad_constant=1.4826,
    clip_outliers=False,
    clip_std_threshold=5.0,
)
```

### Splitting Parameters

```python
SplittingParams(
    enabled=False,
    max_group_size=40,
    max_recursion_depth=3,
    min_features_per_group=3,
)
```

## Family-Specific Defaults

### Temporal Features

```python
TemporalGroupingConfig(
    clustering=ClusteringParams(
        min_groups=8,
        max_groups=20,
    ),
    min_samples_required=100,
)
```

### Initial Features

```python
InitialGroupingConfig(
    clustering=ClusteringParams(
        min_groups=2,
        max_groups=5,
    ),
    min_samples_required=50,
)
```

## VQ-VAE Integration

### Custom Loss Injection

```python
import torch
from spinlock.features.grouping import create_grouper

# Define VQ-VAE reconstruction loss
def vqvae_reconstruction_loss(features, assignment_probs):
    """Compute VQ-VAE reconstruction loss given soft assignments."""
    group_features = torch.matmul(features, assignment_probs)
    reconstructed = model.forward_with_soft_assignments(features, assignment_probs)
    return torch.nn.functional.mse_loss(reconstructed, features)

# Configure grouping with custom loss
config = TemporalGroupingConfig()
config.gradient.custom_loss_weight = 1.0

# Create grouper and inject loss
grouper = create_grouper("temporal", config=config)
grouper.gradient_refiner.set_custom_loss(vqvae_reconstruction_loss)

# Group features (optimizes: orthogonality + informativeness + VQ-VAE recon)
result = grouper.group_features(features, feature_names)
```

## Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **CuPy**: `pip install cupy-cuda12x` (for GPU clustering)
- **PyTorch**: `pip install torch` (for gradient refinement)
- **NumPy**: `pip install numpy`
- **SciPy**: `pip install scipy`
- **scikit-learn**: `pip install scikit-learn`

## Performance

- **GPU Clustering**: ~100x faster than CPU for large feature sets
- **Gradient Refinement**: Benefits from GPU (PyTorch CUDA)
- **Memory**: Efficient for up to 1000 features, 100K samples

## Differences from V1

| Feature | V1 | V2 |
|---------|----|----|
| **Architecture** | Monolithic | Inheritance-based |
| **Coupling** | Embedded in VQ-VAE | Standalone module |
| **Configuration** | Dict-based | Pydantic models |
| **Family Logic** | Mixed | Separate subclasses |
| **Reusability** | VQ-VAE only | Any pipeline |
| **Testing** | Integration only | Unit + integration |
| **GPU Support** | Optional | Required (CuPy) |

## Examples

See `examples/v2_grouping_example.py` for complete examples:

1. Basic usage with default config
2. Custom configuration
3. Per-family grouping pipeline
4. Clustering-only mode (skip gradient)
5. VQ-VAE integration with custom loss

## API Reference

### Factory

```python
create_grouper(family: Literal["temporal", "initial"], config: Optional[GroupingConfig] = None) -> FeatureGrouper
```

### Base Class

```python
class FeatureGrouper(ABC):
    def group_features(self, features: np.ndarray, feature_names: List[str]) -> GroupingResult
    def validate_features(self, features: np.ndarray, feature_names: List[str]) -> None
    def preprocess_features(self, features: np.ndarray) -> np.ndarray
```

### Result

```python
class GroupingResult(BaseModel):
    groups: Dict[str, FeatureGroup]
    num_groups: int
    total_features: int
    config: GroupingConfig

    def to_dict(self) -> Dict[str, List[int]]
```

## Testing

Run tests with pytest:

```bash
# All tests
pytest tests/v2/grouping/

# Specific test files
pytest tests/v2/grouping/test_models.py
pytest tests/v2/grouping/test_clustering.py
pytest tests/v2/grouping/test_gradient.py
pytest tests/v2/grouping/test_integration.py
```

Note: GPU tests require CuPy and will be skipped if not available.

## Troubleshooting

### CuPy Not Found

```bash
pip install cupy-cuda12x
# Or for CUDA 11.x:
pip install cupy-cuda11x
```

### GPU Out of Memory

Reduce `subsample_size` in `ClusteringParams`:

```python
config.clustering.subsample_size = 10000
```

### Too Few/Many Groups

Adjust `min_groups` and `max_groups`:

```python
config.clustering.min_groups = 5
config.clustering.max_groups = 15
```

## Future Enhancements

- [ ] Implement Gap statistic K-selection
- [ ] Implement Elbow method K-selection
- [ ] Add visualization tools for grouping results
- [ ] Add group quality metrics (orthogonality, informativeness)
- [ ] Support for custom distance metrics
- [ ] CLI tool for standalone grouping
- [ ] Summary feature grouper (4-10 groups)

## License

See main project LICENSE.
