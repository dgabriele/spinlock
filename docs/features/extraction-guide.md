# Feature Extraction Guide

## Overview

Spinlock's feature extraction system computes **Summary Descriptor Features (SDF)** from neural operator rollouts. These features characterize spatial patterns, spectral properties, and temporal dynamics, making them ideal for:

- **VQ-VAE training**: Compact, informative features for discrete latent space learning
- **Dataset analysis**: Understanding pattern diversity and dynamics
- **Downstream ML tasks**: Classification, clustering, anomaly detection
- **Scientific discovery**: Identifying invariants and emergent behaviors

## Quick Start

### Basic Usage

Extract features from an existing dataset:

```bash
# Extract SDF features with defaults
spinlock extract-features --dataset datasets/benchmark_10k.h5

# View extraction configuration without running
spinlock extract-features --dataset datasets/benchmark_10k.h5 --dry-run

# Verbose output with progress
spinlock extract-features --dataset datasets/benchmark_10k.h5 --verbose
```

### Command Options

```bash
spinlock extract-features [OPTIONS]

Required:
  --dataset PATH           Path to HDF5 dataset

Optional:
  --config PATH           Custom config file (YAML)
  --output PATH           Output dataset path (default: write to input dataset)
  --batch-size INT        Batch size for extraction (default: 32)
  --device DEVICE         Device: cuda or cpu (default: cuda)
  --overwrite             Overwrite existing features
  --dry-run              Show config without extracting
  --verbose              Verbose progress output
```

## Feature Categories

SDF features are organized into three temporal levels:

### 1. Per-Timestep Features (46 features)

Computed **at each timestep** for spatial and spectral analysis:

**Spatial Statistics (19 features)**:
- Moments: mean, variance, std, skewness, kurtosis
- Extrema: min, max, range, IQR, MAD
- Gradients: magnitude (mean/std/max), directional means, anisotropy
- Curvature: Laplacian (mean/std/energy)

**Spectral Features (27 features)**:
- FFT Power Spectrum (5 scales × 3 stats): mean, max, std per frequency band
- Dominant Frequencies: peak location (x/y), magnitude
- Spectral Centroids: centroid location (x/y), bandwidth
- Frequency Ratios: low/mid/high frequency energy distribution
- Spectral Shape: flatness, rolloff, anisotropy

### 2. Per-Trajectory Features (13 features)

Computed **per realization** from temporal evolution (requires T > 1):

**Temporal Dynamics**:
- Growth/Decay: energy growth rate, acceleration, variance growth
- Oscillations: dominant frequency, amplitude, period
- Stability: autocorrelation decay, Lyapunov approximation, smoothness
- Stationarity: regime switches, trend strength, final/initial ratio

**Note**: These features return NaN for single-timestep datasets (T=1).

### 3. Aggregated Features (39 features)

Final features aggregated across realizations:
- Temporal features aggregated with: mean, std, coefficient of variation (CV)
- 13 temporal features × 3 aggregations = 39 features

## Configuration

### Default Configuration

The default configuration extracts all features:

```yaml
# Built-in defaults (no config file needed)
sdf:
  spatial:
    enabled: true
    include_moments: true
    include_gradients: true
    include_laplacian: true

  spectral:
    enabled: true
    num_fft_scales: 5
    include_fft_power: true
    include_dominant_freq: true
    include_spectral_centroid: true
    include_low_freq_ratio: true
    include_spectral_flatness: true
    include_spectral_rolloff: true
    include_spectral_anisotropy: true

  temporal:
    enabled: true
    include_energy_growth: true
    include_oscillations: true
    include_stability: true

  # Aggregation methods
  per_channel: false  # Average across channels
  temporal_aggregation: [mean, std]
  realization_aggregation: [mean, std, cv]
```

### Custom Configuration

Create a custom config to select specific features:

```yaml
# configs/features/sdf_minimal.yaml
sdf:
  spatial:
    enabled: true
    include_moments: true
    include_gradients: false
    include_laplacian: false

  spectral:
    enabled: true
    num_fft_scales: 3  # Reduce from 5 to 3 scales
    include_fft_power: true
    include_dominant_freq: true
    # Disable other spectral features
    include_spectral_centroid: false
    include_low_freq_ratio: false
    include_spectral_flatness: false
    include_spectral_rolloff: false
    include_spectral_anisotropy: false

  temporal:
    enabled: false  # Skip temporal features
```

Use custom config:

```bash
spinlock extract-features \
  --dataset datasets/benchmark_10k.h5 \
  --config configs/features/sdf_minimal.yaml
```

## HDF5 Schema

Extracted features are stored in the dataset under `/features/`:

```
/features/
├── @family_versions        {"sdf": "1.0.0"}
├── @extraction_timestamp   "2025-12-28T21:30:00"
├── @extraction_config      JSON config
│
└── sdf/
    ├── @version                "1.0.0"
    ├── @feature_registry       JSON name-to-index mapping
    ├── @num_features           59
    │
    ├── per_timestep/
    │   └── features [N, T, 46]
    │
    ├── per_trajectory/
    │   └── features [N, M, 13]
    │
    └── aggregated/
        ├── features [N, 39]
        └── metadata/
            └── extraction_time [N]
```

### Dimensions

- `N`: Number of samples
- `T`: Number of timesteps per trajectory
- `M`: Number of realizations per sample
- Features averaged across channels (C)

## Reading Extracted Features

### Python API

```python
from pathlib import Path
from spinlock.features.storage import HDF5FeatureReader

# Read features
with HDF5FeatureReader(Path("datasets/benchmark_10k.h5")) as reader:
    # Check what's available
    if reader.has_sdf():
        # Get feature registry
        registry = reader.get_sdf_registry()
        print(f"Total features: {registry.num_features}")

        # Get spatial feature names
        spatial_features = registry.get_features_by_category('spatial')
        for feat in spatial_features[:5]:
            print(f"  {feat.name} (index {feat.index})")

        # Read aggregated features (most compact)
        features = reader.get_sdf_aggregated()  # [N, 39]
        print(f"Aggregated features shape: {features.shape}")

        # Read per-timestep features
        per_timestep = reader.get_sdf_per_timestep()  # [N, T, 46]
        print(f"Per-timestep shape: {per_timestep.shape}")

        # Read specific samples
        sample_features = reader.get_sdf_aggregated(idx=slice(0, 10))
```

### Command-Line Inspection

```bash
# View HDF5 structure
h5ls -r datasets/benchmark_10k.h5/features

# Dump feature registry
h5dump -d /features/sdf -A datasets/benchmark_10k.h5 | grep feature_registry
```

## Feature Normalization

All features are normalized for grid-size independence:

### Spatial Features
- **Energy features**: Normalized by `H × W` (energy per pixel)
- **Statistics**: Computed from normalized fields

### Spectral Features
- **FFT**: Uses orthonormal FFT (`norm='ortho'`)
- **Power spectrum**: Grid-size independent
- **Frequency coordinates**: Normalized by grid dimensions

### Expected Value Ranges

| Feature Type           | Typical Range    | Notes                          |
|------------------------|------------------|--------------------------------|
| Spatial moments        | 0 - 10           | Depends on input data scale    |
| Spatial gradients      | 0 - 100          | Higher for sharp features      |
| Laplacian energy       | 0 - 20           | Per-pixel normalized           |
| FFT power (low freq)   | 0 - 500          | DC and low frequencies         |
| FFT power (high freq)  | 0 - 1            | High frequencies (noise)       |
| Dominant freq mag      | 10 - 10,000      | Can be large for periodic data |
| Temporal growth        | -10 - 10         | Normalized per timestep        |

**Note**: Large spectral peaks (>1000) are physically valid for data with strong DC components or periodic patterns.

## Performance

### GPU Optimization

Feature extraction is fully GPU-optimized:

```bash
# Check GPU utilization during extraction
watch -n 1 nvidia-smi

# Expected: 50-80% GPU utilization during extraction
```

### Batching

Adjust batch size based on GPU memory:

```bash
# Small GPU (8GB)
spinlock extract-features --dataset data.h5 --batch-size 16

# Medium GPU (16GB) - default
spinlock extract-features --dataset data.h5 --batch-size 32

# Large GPU (24GB+)
spinlock extract-features --dataset data.h5 --batch-size 64
```

### Benchmarks

Extraction performance (single GPU, 128×128 grids):

| Dataset Size | Batch Size | Time     | Throughput   |
|--------------|------------|----------|--------------|
| 100 samples  | 16         | 12s      | 8 samples/s  |
| 1,000 samples| 32         | 2 min    | 8.3 samples/s|
| 10,000 samples| 32        | 20 min   | 8.3 samples/s|

**Note**: Extraction is I/O-bound for small batches. Use larger batches for better GPU utilization.

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size
```bash
spinlock extract-features --dataset data.h5 --batch-size 8
```

**2. NaN Values in Temporal Features**

```
Warning: Found NaN values in per_trajectory features
```

**Cause**: Dataset has T=1 (single timestep)
**Solution**: This is expected. Temporal features require T>1. Either:
- Disable temporal features in config
- Generate multi-timestep datasets

**3. Features Already Exist**

```
ValueError: SDF features already exist. Set overwrite=True to replace.
```

**Solution**: Use `--overwrite` flag
```bash
spinlock extract-features --dataset data.h5 --overwrite
```

**4. Extremely Large Spectral Values**

**Cause**: Data with strong DC component or periodic patterns
**Solution**: This is physically valid. For downstream ML, consider:
```python
# Log-scale large dynamic range features
features_scaled = np.log1p(features)

# Or clip outliers
features_clipped = np.clip(features, 0, percentile_99)
```

### Debugging

Enable verbose output for detailed progress:

```bash
spinlock extract-features --dataset data.h5 --verbose
```

Dry-run to verify configuration:

```bash
spinlock extract-features --dataset data.h5 --dry-run
```

## Best Practices

### 1. Multi-Timestep Datasets

For meaningful temporal features:
- Generate datasets with **T ≥ 10** timesteps
- Ensure trajectories capture dynamics (not just initial conditions)

### 2. Feature Selection

For VQ-VAE training:
- **Start with defaults**: All features provide complementary information
- **Profile importance**: Use feature importance analysis to identify key features
- **Remove redundant**: Drop features with low variance or high correlation

### 3. Normalization

For downstream ML:
```python
from sklearn.preprocessing import StandardScaler

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
```

### 4. Batch Processing

For large datasets:
```python
# Process in chunks to avoid memory issues
from spinlock.features.extractor import FeatureExtractor

extractor = FeatureExtractor(
    input_dataset=Path("large_dataset.h5"),
    output_dataset=Path("large_dataset.h5"),
    batch_size=16,
    device='cuda'
)

extractor.extract(verbose=True)
```

## Next Steps

- **Feature Analysis**: See `docs/features/feature-reference.md` for detailed feature descriptions
- **VQ-VAE Training**: Use extracted features with `spinlock train-vqvae`
- **Custom Features**: Extend the system with custom feature families

## References

- **FFT Normalization**: Orthonormal FFT ensures energy conservation (Parseval's theorem)
- **Spectral Features**: Based on audio signal processing (MFCCs, spectral descriptors)
- **Temporal Dynamics**: Growth rates, stability metrics from dynamical systems theory
