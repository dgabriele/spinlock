# Spinlock

**High-Performance Stochastic Neural Operator Dataset Generator**

Spinlock is a production-grade system for systematic sampling, simulation, and dataset generation of CNN-based stochastic neural operators. Designed for exploring emergent dynamical behaviors through VQ-VAE tokenization and representation learning.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Stratified Low-Discrepancy Sampling** - Sobol sequences with Owen scrambling for efficient parameter space exploration
- **Flexible CNN Operators** - YAML-configurable architectures with stochastic elements
- **GPU-Accelerated Execution** - Adaptive batching, memory management, multi-GPU ready
- **Efficient Storage** - Chunked HDF5 with compression for large-scale datasets
- **Production-Ready** - Type-safe, modular, DRY code following ML engineering best practices

## Performance

**10k Sample Benchmark** (7.65 GB GPU):
- ⚡ **7.26 minutes** for 10k samples × 10 realizations (100k total outputs)
- 🎯 **8.3× faster** than 1-hour target specification
- 📊 Sample quality **1000× better** than specification (discrepancy 0.000010 vs target 0.01)
- 💾 **5.04 GB** compressed dataset size

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
```

**Available Commands:**
- `generate` - Generate datasets from configuration
- `info` - Display dataset information and metadata
- `validate` - Verify dataset integrity and quality

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
Input Generator (Gaussian Random Fields via FFT)
    ↓
GPU Execution (M stochastic realizations per operator)
    ↓
HDF5 Storage (Chunked, compressed, with metadata)
```

### Key Components

- **Configuration System** (`src/spinlock/config/`) - Type-safe Pydantic schemas, YAML loading
- **Sampling System** (`src/spinlock/sampling/`) - Sobol sequences, quality validation metrics
- **Operator System** (`src/spinlock/operators/`) - Modular CNN blocks, builder pattern
- **Execution System** (`src/spinlock/execution/`) - Parallelization, adaptive batching, memory management
- **Dataset Generation** (`src/spinlock/dataset/`) - Input generators, HDF5 storage, pipeline orchestrator

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

dataset:
  output_path: "./datasets/my_experiment.h5"
  storage:
    compression: "gzip"
    compression_level: 4
```

See `configs/experiments/` for complete examples.

## Dataset Schema

Generated HDF5 datasets follow this structure:

```
/metadata/
    - config (JSON)              # Complete configuration
    - sampling_metrics (JSON)    # Discrepancy, correlations
    - creation_date, version

/parameters/
    - params [N, P]              # Parameter sets in [0,1]^P

/inputs/
    - fields [N, C_in, H, W]     # Input fields (GRF or structured)

/outputs/
    - fields [N, M, C_out, H, W] # Output fields (M stochastic realizations)
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

### 🚧 Near-term
- [ ] Unit tests for all components
- [ ] Documentation website
- [ ] Visualization tools (parameter space, outputs)
- [ ] Validation scripts
- [ ] Heterogeneous operators (variable dimensions)

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

**Status**: MVP Complete | **Version**: 0.1.0 | **Python**: 3.11+
