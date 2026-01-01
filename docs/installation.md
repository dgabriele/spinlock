# Installation

## Requirements

- **Python:** 3.11 or higher
- **CUDA:** 11.8+ (for GPU acceleration)
- **GPU:** NVIDIA GPU with 8GB+ VRAM (recommended for dataset generation)
- **Storage:** 100GB+ free space (for datasets and checkpoints)

## Quick Install

### Using Poetry (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/spinlock.git
cd spinlock

# Install dependencies
poetry install

# Verify installation
poetry run spinlock --help
```

### Using pip

```bash
# Clone repository
git clone https://github.com/yourusername/spinlock.git
cd spinlock

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Verify installation
spinlock --help
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-dev python3-pip

# Install CUDA (if not already installed)
# Follow: https://developer.nvidia.com/cuda-downloads

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install Spinlock
poetry install
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Install Poetry
brew install poetry

# Install Spinlock
poetry install
```

**Note:** GPU acceleration requires macOS with Apple Silicon (M1/M2/M3) using MPS backend. Performance may be slower than CUDA.

### Windows

```powershell
# Install Python 3.11 from python.org

# Install Poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Install Spinlock
poetry install
```

**Note:** Windows GPU support requires WSL2 + CUDA. Native Windows CUDA support is experimental.

## Docker Installation

### Using Pre-built Image

```bash
# Pull image
docker pull yourusername/spinlock:latest

# Run container
docker run --gpus all -it yourusername/spinlock:latest bash
```

### Building from Source

```bash
# Build image
docker build -t spinlock:local .

# Run container with GPU support
docker run --gpus all -v $(pwd)/datasets:/workspace/datasets -it spinlock:local bash
```

**Docker Compose:**

```yaml
version: '3.8'
services:
  spinlock:
    build: .
    volumes:
      - ./datasets:/workspace/datasets
      - ./checkpoints:/workspace/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## From Source

### Development Installation

```bash
# Clone repository
git clone https://github.com/yourusername/spinlock.git
cd spinlock

# Install in editable mode with dev dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest
```

### Building CUDA Extensions

If you need to rebuild CUDA kernels:

```bash
# Navigate to CUDA source
cd src/spinlock/cuda

# Clean build artifacts
rm -rf build/

# Rebuild
poetry run python setup.py build_ext --inplace
```

## GPU Configuration

### Verify CUDA Installation

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Test PyTorch CUDA
poetry run python -c "import torch; print(torch.cuda.is_available())"
```

### Multiple GPUs

To use specific GPUs:

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 poetry run spinlock generate ...

# Use GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 poetry run spinlock generate ...
```

### CPU-Only Mode

For testing without GPU:

```bash
# Force CPU execution
CUDA_VISIBLE_DEVICES="" poetry run spinlock generate ...
```

**Note:** CPU-only execution is significantly slower (10-50× depending on operation).

## Troubleshooting

### CUDA Out of Memory

**Solution 1:** Reduce batch size

```yaml
# In config YAML
execution:
  batch_size: 4  # Reduce from 16
```

**Solution 2:** Use gradient checkpointing

```yaml
vqvae:
  gradient_checkpointing: true
```

**Solution 3:** Use smaller grid size

```yaml
dataset:
  grid_size: 64  # Instead of 128
```

### Import Errors

```bash
# Reinstall dependencies
poetry install --no-cache

# If using pip, upgrade
pip install --upgrade --force-reinstall -e .
```

### Permission Errors

```bash
# Fix Poetry permissions
chmod -R 755 ~/.cache/pypoetry

# Fix dataset directory
sudo chown -R $USER:$USER datasets/
```

### CUDA Version Mismatch

```bash
# Check PyTorch CUDA version
poetry run python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
poetry add torch --source pytorch-cu118  # For CUDA 11.8
```

## Optional Dependencies

### Visualization (recommended)

```bash
poetry install --extras "viz"
```

Includes: matplotlib, seaborn, ffmpeg-python

### Development Tools

```bash
poetry install --with dev
```

Includes: pytest, black, ruff, mypy, pre-commit

### Documentation

```bash
poetry install --with docs
```

Includes: sphinx, sphinx-rtd-theme, myst-parser

## Verifying Installation

### Quick Test

```bash
# Generate small test dataset
poetry run spinlock generate \
    --config configs/experiments/test_100.yaml \
    --output datasets/test_verify.h5

# Check output
poetry run spinlock inspect datasets/test_verify.h5
```

### Full Test Suite

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=spinlock --cov-report=html
```

## Next Steps

- [Getting Started](getting-started.md) - Tutorials and examples
- [Architecture](architecture.md) - System overview
- [NOA Roadmap](noa-roadmap.md) - Development plan

## Support

- **Issues:** https://github.com/yourusername/spinlock/issues
- **Discussions:** https://github.com/yourusername/spinlock/discussions
- **Documentation:** https://spinlock.readthedocs.io
