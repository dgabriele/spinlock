# Spinlock

**High-Performance Stochastic Neural Operator Dataset Generator**

Spinlock is a production-grade system for systematic sampling, simulation, and dataset generation of CNN-based stochastic neural operators. Designed for exploring emergent dynamical behaviors through VQ-VAE tokenization and representation learning.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Stratified Low-Discrepancy Sampling** - Sobol sequences with Owen scrambling for efficient parameter space exploration
- **Flexible CNN Operators** - YAML-configurable architectures with stochastic elements
- **28 IC Types** - Baseline + 4 domain-specific tiers covering quantum physics, biology, chemistry, information theory, and more
- **Variable Grid Sizes** - Multi-scale exploration with 64×64, 128×128, 256×256 resolutions
- **Rollout Policies** - 3 temporal update strategies (autoregressive, residual, convex) for diverse dynamical behaviors
- **Rich Metadata Tracking** - IC types, rollout policies, grid sizes, noise regimes for hypothesis generation and analysis
- **Advanced Visualization** - 9 aggregate renderers (entropy, PCA, SSIM, spectral analysis, etc.), per-operator color normalization, IC type gallery generation
- **Feature Extraction System (SDF v2.0)** - GPU-optimized extraction with 153 features across 7 categories: spatial (19), spectral (27), temporal (13), operator sensitivity (10), cross-channel (10), causality (14), invariant drift (60) - NEW v2.0 operator-aware features!
- **GPU-Optimized Performance** - Phase 1 optimizations: coordinate grid caching, vectorized input generation, 70% GPU memory utilization (1.8-2.2x speedup)
- **GPU-Accelerated Execution** - Adaptive batching, memory management, multi-GPU ready
- **Efficient Storage** - Chunked HDF5 with compression for large-scale datasets
- **Production-Ready** - Type-safe, modular, DRY code following ML engineering best practices

## Performance

**Generation Speed** (Phase 1 Optimizations - Implemented):
- ⚡ **1.8-2.2x faster** than baseline through GPU optimizations
- 🎯 **23.35 samples/sec** for 100-sample benchmark (4.28 seconds total)
- 📊 **70% GPU memory utilization** (increased from 35% conservative limit)

**Key Optimizations:**
- Coordinate grid caching (15-20% speedup)
- Vectorized input generation (30-40% speedup)
- Increased batch size memory limits (5-10% speedup)

**Visualization Rendering:**
- ⚡ **20.9x faster** frame rendering (359s → 17.2s for 8 operators × 120 timesteps)
- 🎬 **GPU-accelerated encoding** via NVENC (1.2x faster, better compression: 24MB → 5MB)
- 📊 **Vectorized entropy computation** (86x speedup: 344s → 4s)

**10k Sample Benchmark** (Original):
- 7.26 minutes for 10k samples × 10 realizations (100k total outputs)
- 8.3× faster than 1-hour target specification
- Sample quality 1000× better than specification (discrepancy 0.000010 vs target 0.01)
- 5.04 GB compressed dataset size

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/spinlock.git
cd spinlock

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Basic Usage

**All Spinlock operations use the official CLI:**

```bash
# Generate dataset from configuration
python scripts/spinlock.py generate --config configs/experiments/benchmark_10k.yaml

# Override parameters for quick tests
python scripts/spinlock.py generate \
    --config configs/experiments/test_100.yaml \
    --total-samples 500 \
    --batch-size 50 \
    --output datasets/my_dataset.h5

# Get dataset information
python scripts/spinlock.py info --dataset datasets/benchmark_10k.h5

# Validate dataset integrity
python scripts/spinlock.py validate --dataset datasets/benchmark_10k.h5

# Visualize operator trajectories
python scripts/spinlock.py visualize-dataset
    --dataset datasets/benchmark_10k.h5 \
    --output visualizations/ \
    --format video \
    --n-realizations 5 \
    --n-operators 5 \
    --size 256x256 \
    --steps 100 \
    --fps 20 \
    --seed 7 \
    --display-realizations 3 \
    --color-norm-mode per-operator \
    --aggregates mean entropy spectral \
    --add-spacing \
    --normalization none \
    --verbose

# Visualize all IC types (gallery view)
python scripts/spinlock.py visualize-ic-types \
    --output visualizations/all_ics.png \
    --variations 3 \
    --add-labels \
    --add-spacing \
    --verbose

# Visualize specific tiers
python scripts/spinlock.py visualize-ic-types \
    --tiers baseline tier1 tier4 \
    --output visualizations/selected_tiers.png \
    --variations 5

# Visualize specific IC types
python scripts/spinlock.py visualize-ic-types \
    --ic-types quantum_wave_packet turing_pattern coherent_state \
    --output visualizations/quantum_patterns.png
```

**Available Commands:**
- `generate` - Generate datasets from configuration
- `info` - Display dataset information and metadata
- `validate` - Verify dataset integrity and quality
- `visualize-dataset` - Create temporal evolution visualizations from stored datasets
- `visualize-ic-types` - Generate gallery visualization of all 28 IC types
- `extract-features` - Extract SDF features for VQ-VAE training and analysis (NEW!)

Use `python scripts/spinlock.py --help` for full documentation.

### Python API

```python
from spinlock.config import load_config
from spinlock.dataset import DatasetGenerationPipeline

# Load configuration
config = load_config("configs/experiments/benchmark_10k.yaml")

# Create and run pipeline
pipeline = DatasetGenerationPipeline(config)
pipeline.generate()
```

## Architecture

### Pipeline Overview

```
Parameter Space (YAML Config)
    ↓
Sobol Sampler (Stratified, Owen scrambling)
    ↓
Parameter Sets [N, P]
    ↓
Operator Builder (CNN factory from parameters)
    ↓
Neural Operators [N models with stochastic elements]
    ↓
Input Generator (Diverse ICs: multi-scale GRF, localized, composite, heavy-tailed)
    ↓
GPU Execution (M stochastic realizations per operator, 3 rollout policies)
    ↓
HDF5 Storage (Chunked, compressed, with discovery metadata)
    ↓
Visualization Pipeline (Temporal evolution, grid layouts, video export) [Optional]
```

### Key Components

- **Configuration System** (`src/spinlock/config/`) - Type-safe Pydantic schemas, YAML loading
- **Sampling System** (`src/spinlock/sampling/`) - Sobol sequences, quality validation metrics
- **Operator System** (`src/spinlock/operators/`) - Modular CNN blocks, builder pattern
- **Rollout Policies** (`src/spinlock/rollout/`) - Autoregressive, residual, convex temporal update strategies
- **Execution System** (`src/spinlock/execution/`) - Parallelization, adaptive batching, memory management
- **Dataset Generation** (`src/spinlock/dataset/`) - Input generators, HDF5 storage, pipeline orchestrator
- **Visualization System** (`src/spinlock/visualization/`) - Rendering engine, temporal evolution, grid layouts, video export

## Configuration

Create YAML configuration files defining your parameter space and execution settings:

```yaml
# configs/experiments/my_experiment.yaml
version: "1.0"

parameter_space:
  architecture:
    num_layers:
      type: integer
      bounds: [2, 6]
    base_channels:
      type: integer
      bounds: [16, 64]
    activation:
      type: choice
      choices: ["relu", "gelu", "silu"]

  stochastic:
    noise_scale:
      type: continuous
      bounds: [0.00001, 1.0]  # Expanded range for diversity
      log_scale: true
    noise_schedule:
      type: choice
      choices: ["constant", "annealing", "periodic"]
    spatial_correlation:
      type: continuous
      bounds: [0.0, 0.3]

  operator:
    grid_size:  # Variable resolutions for multi-scale exploration
      type: choice
      choices: [64, 128, 256]

  rollout:  # Temporal dynamics configuration
    update_policy:
      type: choice
      choices: ["autoregressive", "residual", "convex"]
    alpha:
      type: continuous
      bounds: [0.1, 0.9]
    dt:
      type: continuous
      bounds: [0.001, 0.1]
      log_scale: true

sampling:
  total_samples: 10000
  batch_size: 1000
  sobol:
    scramble: true
    seed: 42

simulation:
  device: "cuda"
  num_realizations: 10
  input_generation:
    method: "sampled"  # Diverse IC sampling
    ic_type_weights:
      multiscale_grf: 0.30
      localized: 0.25
      composite: 0.25
      gaussian_random_field: 0.15
      heavy_tailed: 0.05

dataset:
  output_path: "./datasets/my_experiment.h5"
  storage:
    compression: "gzip"
    compression_level: 4
```

See `configs/experiments/` for complete examples.

## Initial Conditions

Spinlock provides **28 diverse initial condition types** organized across 5 tiers, covering 7 scientific domains.

### IC Type Organization

**Baseline ICs (7 types)** - Core stochastic patterns:
- Gaussian Random Field (GRF), Multi-scale GRF, Localized, Composite, Heavy-tailed, Structured, Mixed

**Tier 1: Foundational (5 types)** - Core physics/biology:
- `quantum_wave_packet` - Quantum mechanics baseline
- `turing_pattern` - Reaction-diffusion morphogenesis
- `thermal_gradient` - Heat transport
- `morphogen_gradient` - Biological patterning
- `reaction_front` - Chemical reaction waves

**Tier 2: Specialized (5 types)** - Advanced scientific domains:
- `light_cone` - Relativistic causality
- `critical_fluctuation` - Phase transition critical points
- `phase_boundary` - Multi-phase coexistence
- `bz_reaction` - Belousov-Zhabotinsky oscillations
- `shannon_entropy` - Information-theoretic patterns

**Tier 3: Complex Systems (5 types)** - Emergent phenomena:
- `interference_pattern` - Wave interference
- `cell_population` - Spatial population dynamics
- `chromatin_domain` - Genomic spatial organization
- `shock_front` - Nonlinear shock propagation
- `gene_expression` - Spatiotemporal gene regulation

**Tier 4: Research Frontiers (6 types)** - Cutting-edge patterns:
- `coherent_state` - Quantum optics (minimal spreading)
- `relativistic_wave_packet` - Lorentz-contracted structures
- `mutual_information` - Information flow between regions
- `regulatory_network` - Gene networks with spatial embedding
- `dla_cluster` - Diffusion-limited aggregation (fractal growth)
- `error_correcting_code` - Redundancy and error correction

### Domain Coverage

The 21 domain-specific ICs span 7 scientific fields:
- **Quantum Physics**: quantum_wave_packet, interference_pattern, coherent_state
- **Relativistic Physics**: light_cone, shock_front, relativistic_wave_packet
- **Thermodynamics**: thermal_gradient, phase_boundary, critical_fluctuation
- **Biology**: turing_pattern, morphogen_gradient, cell_population
- **Chemistry**: reaction_front, bz_reaction, dla_cluster
- **Genomics**: chromatin_domain, gene_expression, regulatory_network
- **Information Theory**: shannon_entropy, mutual_information, error_correcting_code

### Usage

```yaml
# Sample from all 21 domain-specific ICs (stratified)
simulation:
  input_generation:
    method: "sampled"
    ic_type_weights:
      # Tier 1
      quantum_wave_packet: 0.0476
      turing_pattern: 0.0476
      # ... (equal weights for all 21)
```

Or use specific tiers:
- `configs/experiments/domain_ics_tier1.yaml` - Foundational (5 types)
- `configs/experiments/domain_ics_tier4.yaml` - Research frontiers (6 types)
- `configs/experiments/domain_ics_all_21.yaml` - All domain-specific ICs

### Visualizing IC Types

Generate a visual gallery of all IC types:

```bash
python scripts/spinlock.py visualize-ic-types \
    --output ic_gallery.png \
    --variations 3 \
    --tiers all \
    --add-labels
```

## Visualization

Spinlock includes a comprehensive visualization system for temporal evolution analysis.

### Aggregate Renderers

**9 aggregate rendering modes** to analyze ensemble behavior:

1. **Mean** - Expected value across realizations
2. **Variance** - Magnitude of variation (L2 norm)
3. **StdDev** - Standard deviation magnitude
4. **Envelope** - Min/max range visualization (coolwarm colormap)
5. **Overlay** - Alpha-blended ensemble composite ("spaghetti plot")
6. **Entropy** ⭐ - Shannon entropy per pixel (reveals multimodality)
7. **PCA** ⭐ - Principal components as RGB (shows structural variation)
8. **SSIM** - Structural similarity index (structure-aware consistency)
9. **Spectral** - 2D FFT power spectrum (characteristic length scales)

**Default aggregates**: `mean`, `envelope`, `entropy` (scientifically superior to variance-based metrics)

### Color Normalization Modes

Control how color scales are applied across the visualization grid:

- `--color-norm-mode global` - All operators share same color scale (compare absolute magnitudes)
- `--color-norm-mode per-operator` - Each operator row has own scale (compare dynamics within operator)
- `--color-norm-mode per-cell` - Each cell normalized independently (default, maximum contrast)

### Display Control

**Smart realization display** via `--display-realizations`:
- Shows K individual realizations + aggregates computed from ALL M realizations
- Example: `--display-realizations 3` shows 3 realizations + 3 aggregates (6 columns total)
- Default: 2 realizations + aggregates (vs showing all 10 realizations = 10 columns)

### Performance

- **20.9x faster rendering** through aggregate pre-computation and vectorization
- **GPU-accelerated encoding** via NVENC H.264 (1.2x faster, better compression)
- **Vectorized entropy computation** (86x speedup for complex aggregates)

### Examples

```bash
# Basic temporal evolution
python scripts/spinlock.py visualize-dataset \
    --dataset datasets/my_data.h5 \
    --output evolution.mp4

# Advanced: entropy + spectral analysis with per-operator normalization
python scripts/spinlock.py visualize-dataset \
    --dataset datasets/my_data.h5 \
    --output analysis.mp4 \
    --aggregates entropy spectral pca \
    --color-norm-mode per-operator \
    --display-realizations 2
```

## Feature Extraction

Spinlock includes a GPU-optimized feature extraction system for computing **Summary Descriptor Features (SDF)** from neural operator rollouts.

### Overview

Extract comprehensive spatial, spectral, and temporal features for:
- **VQ-VAE Training**: Compact, informative features for discrete latent spaces
- **Dataset Analysis**: Understanding pattern diversity and dynamics
- **Scientific Discovery**: Identifying invariants and emergent behaviors
- **Downstream ML**: Classification, clustering, anomaly detection

### Quick Start

```bash
# Extract features with defaults (all 59 features)
python scripts/spinlock.py extract-features --dataset datasets/benchmark_10k.h5

# Verbose output
python scripts/spinlock.py extract-features --dataset datasets/benchmark_10k.h5 --verbose

# Custom batch size for GPU memory
python scripts/spinlock.py extract-features --dataset datasets/benchmark_10k.h5 --batch-size 16
```

### Feature Categories

**All features are scalars** - each operator produces a single scalar value per feature.

**Per-Timestep Features (46 scalars)**:
- **Spatial Statistics** (19): moments, gradients, curvature (Laplacian)
- **Spectral Features** (27): FFT power spectrum (5 scales), dominant frequencies, spectral shape
- **Output**: `[N, T, 46]` - 46 scalars per operator per timestep

**Per-Trajectory Features (13 scalars, requires T>1)**:
- **Temporal Dynamics**: growth rates, oscillations, stability metrics, stationarity measures
- **Output**: `[N, M, 13]` - 13 scalars per realization

**Aggregated Features (39 scalars)**:
- Temporal features aggregated across realizations with mean, std, coefficient of variation
- **Output**: `[N, 39]` - 39 scalars per operator (ready for VQ-VAE/ML)

### Performance

- **GPU-Optimized**: 50-80% GPU utilization during extraction
- **Throughput**: ~8 samples/sec on 128×128 grids
- **Benchmarks**:
  - 100 samples: 12s
  - 1,000 samples: 2 min
  - 10,000 samples: 20 min

### Reading Features

```python
from pathlib import Path
from spinlock.features.storage import HDF5FeatureReader

with HDF5FeatureReader(Path("datasets/benchmark_10k.h5")) as reader:
    # Get feature registry
    registry = reader.get_sdf_registry()
    print(f"Total features: {registry.num_features}")

    # Read aggregated features (most compact)
    features = reader.get_sdf_aggregated()  # [N, 39]

    # Read per-timestep features
    per_timestep = reader.get_sdf_per_timestep()  # [N, T, 46]
```

### Documentation

- **Extraction Guide**: [`docs/features/extraction-guide.md`](docs/features/extraction-guide.md) - Complete usage guide
- **Feature Reference**: [`docs/features/feature-reference.md`](docs/features/feature-reference.md) - Detailed feature descriptions
- **Tutorial**: [`examples/demos/feature_extraction_tutorial.py`](examples/demos/feature_extraction_tutorial.py) - Python examples

## Dataset Schema

Generated HDF5 datasets follow this structure:

```
/metadata/
    - config (JSON)                  # Complete configuration
    - sampling_metrics (JSON)        # Discrepancy, correlations
    - creation_date, version
    - ic_types [N]                   # Initial condition type per operator (29 types available)
    - rollout_policies [N]           # Rollout policy per operator (autoregressive/residual/convex)
    - grid_sizes [N]                 # Grid resolution per operator (64/128/256)
    - noise_regimes [N]              # Noise classification (low/medium/high)

/parameters/
    - params [N, P]                  # Parameter sets in [0,1]^P

/inputs/
    - fields [N, C_in, 256, 256]     # Input fields (padded to max size)

/outputs/
    - fields [N, M, C_out, 256, 256] # Output fields (M stochastic realizations, padded)

/features/                           # [Optional] Extracted SDF features
    @family_versions                 # {"sdf": "1.0.0"}
    @extraction_timestamp
    /sdf/
        @version                     # "1.0.0"
        @feature_registry            # JSON name-to-index mapping
        @num_features                # 59
        /per_timestep/
            features [N, T, 46]      # Per-timestep spatial & spectral features
        /per_trajectory/
            features [N, M, 13]      # Per-trajectory temporal dynamics
        /aggregated/
            features [N, 39]         # Aggregated across realizations
            /metadata/
                extraction_time [N]  # Extraction time per sample
```

## Development

### Project Structure

```
spinlock/
├── src/spinlock/           # Main package
│   ├── config/            # Configuration system
│   ├── sampling/          # Parameter space sampling
│   ├── operators/         # CNN operator building
│   ├── execution/         # GPU execution strategies
│   └── dataset/           # Dataset generation & storage
├── scripts/
│   └── spinlock.py        # Official CLI (use this!)
├── configs/
│   └── experiments/       # Experiment configurations
├── docs/
│   └── sessions/          # Development session logs
└── tests/                 # Unit tests (coming soon)
```

### Design Principles

- **Modularity** - Clean separation of concerns with well-defined interfaces
- **DRY** - Shared abstractions, composition over inheritance
- **Extensibility** - Strategy/Factory/Registry patterns throughout
- **Performance** - GPU-optimized, memory-efficient, adaptive batching
- **Testability** - Dependency injection, abstract base classes
- **Configuration-driven** - YAML-based reproducible experiments

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=spinlock --cov-report=html

# Run specific test module
poetry run pytest tests/test_sampling/
```

## Roadmap

### ✅ MVP (Complete)
- [x] Configuration system with Pydantic schemas
- [x] Sobol sampler with stratification
- [x] CNN operator builder from parameters
- [x] GPU execution with adaptive batching
- [x] HDF5 storage with compression
- [x] End-to-end pipeline
- [x] CLI orchestrator
- [x] **29 Initial Conditions** (8 baseline + 21 domain-specific across 4 tiers)
- [x] Variable grid sizes (64/128/256)
- [x] Metadata tracking for discovery
- [x] **Advanced Visualization System**:
  - [x] Temporal evolution rendering with 9 aggregate modes
  - [x] Per-operator/global/per-cell color normalization
  - [x] IC type gallery visualization command
  - [x] GPU-accelerated video encoding (NVENC)
  - [x] 20.9x rendering speedup via vectorization
- [x] Rollout policies (autoregressive, residual, convex)
- [x] **Phase 1 Performance Optimizations** (1.8-2.2x speedup):
  - [x] Coordinate grid caching
  - [x] Vectorized input generation
  - [x] 70% GPU memory utilization

### 🚧 Near-term
- [ ] Unit tests for all components
- [ ] Documentation website (Sphinx/mkdocs)
- [ ] Example notebooks and tutorials
- [ ] Heterogeneous operators (variable dimensions)
- [ ] Analysis tools for metadata exploration

### 🔮 Future
- [ ] DDP multi-GPU support
- [ ] Zarr storage backend
- [ ] VQ-VAE tokenization integration
- [ ] Motif discovery tools
- [ ] Custom CUDA kernels (profile-guided)
- [ ] Web dashboard for monitoring

## Performance Tuning

### GPU Memory Optimization

```bash
# Reduce batch size for smaller GPUs
python scripts/spinlock.py generate \
    --config configs/experiments/benchmark_10k.yaml \
    --batch-size 50

# Use CPU if CUDA unavailable
python scripts/spinlock.py generate \
    --config configs/experiments/test_100.yaml \
    --device cpu
```

### Multi-GPU Support (Coming Soon)

```yaml
# In config file
simulation:
  parallelism:
    strategy: "ddp"  # Distributed Data Parallel
    devices: [0, 1, 2, 3]  # Use 4 GPUs
```

## Documentation

- **Session Logs** - Detailed development notes in [`docs/sessions/`](docs/sessions/)
- **IC Type Reference** - All 28 IC types documented in [`src/spinlock/dataset/generators.py`](src/spinlock/dataset/generators.py)
- **Visualization Guide** - 9 aggregate renderers in [`src/spinlock/visualization/core/aggregator.py`](src/spinlock/visualization/core/aggregator.py)
- **Configuration Examples** - Tier-specific configs in [`configs/experiments/`](configs/experiments/)
- **API Reference** - Coming soon
- **Tutorials** - Coming soon

## Citation

If you use Spinlock in your research, please cite:

```bibtex
@software{spinlock2025,
  title = {Spinlock: High-Performance Stochastic Neural Operator Dataset Generator},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/spinlock}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please open an issue or pull request.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/spinlock/issues)

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Neural network execution
- [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) - Numerical computing
- [H5py](https://www.h5py.org/) - HDF5 storage
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Configuration validation
- [Poetry](https://python-poetry.org/) - Dependency management

---

**Status**: Production Ready | **Version**: 1.0.0 | **Python**: 3.11+ | **Features**: 28 IC Types, 9 Aggregate Renderers, 1.8-2.2x Performance Boost
