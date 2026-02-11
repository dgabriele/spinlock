# Dataset Generation Guide

## Overview

Spinlock provides CLI tools for generating synthetic PDE datasets with multiple operator families. This guide covers all supported operators and configuration options.

## Supported Operators

| Operator Family | Command | Key Features |
|----------------|---------|--------------|
| **Convex PDEs** | `generate-cno-dataset` | Heat, wave, advection, reaction-diffusion, Burgers |
| **Quantum Brownian Motion** | `generate-qbm-dataset` | Lindblad dynamics, decoherence, quantum features |
| **Meta Neural Operator** | `generate-mno-dataset` | Rollouts from pre-trained MNO model |

---

## Convex PDE Dataset Generation

### Basic Usage

```bash
poetry run spinlock generate-cno-dataset \
  --num-samples 50000 \
  --num-realizations 3 \
  --output datasets/cno_50k.h5 \
  --device cuda
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num-samples` | int | Required | Number of PDE parameter sets to sample |
| `--num-realizations` | int | 3 | Realizations per parameter set (different ICs) |
| `--output` | path | Required | Output HDF5 file path |
| `--device` | str | `cuda` | Compute device (`cuda` or `cpu`) |
| `--timesteps` | int | 128 | Number of time steps |
| `--grid-size` | int | 64 | Spatial resolution (64×64 default) |
| `--seed` | int | 42 | Random seed for reproducibility |

### Operators Sampled

**Heat Equation (Diffusion):**
```
∂u/∂t = α∇²u
```
Parameters: Diffusion coefficient α ∈ [0.01, 0.5]

**Wave Equation:**
```
∂²u/∂t² = c²∇²u
```
Parameters: Wave speed c ∈ [0.5, 2.0]

**Advection:**
```
∂u/∂t + v·∇u = 0
```
Parameters: Velocity field v ∈ [-1, 1]²

**Reaction-Diffusion:**
```
∂u/∂t = D∇²u + f(u)
```
Parameters: Diffusion D, reaction rates (Fisher, FitzHugh-Nagumo, etc.)

**Burgers' Equation:**
```
∂u/∂t + u·∇u = ν∇²u
```
Parameters: Viscosity ν ∈ [0.001, 0.1]

### Dataset Structure

```
cno_50k.h5
├── inputs/              # [N, M, C, H, W] initial conditions
│   shape: [50000, 3, 1, 64, 64]
├── outputs/             # [N, M, T, C, H, W] evolved states
│   shape: [50000, 3, 128, 1, 64, 64]
├── parameters/
│   ├── params           # [N, 14] operator parameters
│   └── operator_types   # [N] categorical (heat=0, wave=1, etc.)
├── features/
│   ├── temporal/        # [N, T, 178] or [N, T, 188] with quantum
│   └── initial/         # [N, 426] (384 CNN + 42 manual)
└── metadata/
    ├── grid_size        # 64
    ├── timesteps        # 128
    └── dt               # Time step size
```

---

## Quantum Brownian Motion (QBM) Dataset

### Basic Usage

```bash
poetry run spinlock generate-qbm-dataset \
  --num-samples 10000 \
  --num-realizations 3 \
  --output datasets/qbm_10k.h5 \
  --device cuda
```

### Quantum-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--extract-quantum` | flag | True | Extract quantum-specific features (purity, entropy, etc.) |
| `--hilbert-dim` | int | 20 | Truncated Hilbert space dimension |
| `--gamma-range` | tuple | (0.01, 0.5) | Dissipation rate sampling range |
| `--nbar-range` | tuple | (0, 2) | Thermal photon number range |
| `--alpha-range` | tuple | (0, 3) | Initial coherent state amplitude |

### Example with Custom Ranges

```bash
poetry run spinlock generate-qbm-dataset \
  --num-samples 10000 \
  --gamma-range 0.05 0.3 \
  --nbar-range 0.1 1.5 \
  --hilbert-dim 30 \
  --output datasets/qbm_high_dim.h5
```

### QBM Physics

**Lindblad Master Equation:**
```
dρ/dt = -i[H, ρ] + γ(n̄ + 1)D[a]ρ + γn̄D[a†]ρ
```

**Parameters Sampled:**
- **γ**: Dissipation rate (coupling to environment)
- **n̄**: Thermal occupation number (environment temperature)
- **α₀**: Initial coherent state amplitude
- **ω**: Oscillator frequency (fixed at 1.0 in natural units)

### Quantum Features Extracted

**Standard Temporal (178D):**
- Energy, gradients, curl, divergence
- Statistical moments
- Spatial Fourier modes

**Quantum Extension (+10-11D):**
```python
quantum_features = {
    'purity': Tr(ρ²),                      # State purity
    'entropy': -Tr(ρ log ρ),               # Von Neumann entropy
    'coherence_mean': mean(|ρᵢⱼ|) i≠j,   # Average coherence
    'coherence_max': max(|ρᵢⱼ|) i≠j,     # Peak coherence
    'coherence_norm': ||ρ - diag(ρ)||₁,  # L1 coherence
    'uncertainty_x': ⟨Δx²⟩,               # Position variance
    'uncertainty_p': ⟨Δp²⟩,               # Momentum variance
    'uncertainty_product': Δx·Δp,         # Heisenberg product
    'fidelity_t0': F(ρ(t), ρ(0)),        # Fidelity vs initial
    'phase_coherence': |⟨a⟩|,             # Coherent amplitude
    'squeezing': ⟨Δx²⟩ - ⟨Δp²⟩,          # Quadrature squeezing
}
```

### Dataset Structure

```
qbm_10k.h5
├── inputs/              # [N, M, C, H, W] Wigner function at t=0
├── outputs/             # [N, M, T, C, H, W] Wigner evolution
├── parameters/
│   ├── params           # [N, 14] (γ, n̄, α₀, ω, etc.)
│   ├── gamma            # [N] dissipation rates
│   ├── nbar             # [N] thermal photon numbers
│   └── alpha_init       # [N] initial amplitudes
├── features/
│   ├── temporal/        # [N, T, 188] (178 standard + 10 quantum)
│   │   ├── standard/    # [N, T, 178]
│   │   └── quantum/     # [N, T, 10]
│   └── initial/         # [N, 426]
└── metadata/
    ├── hilbert_dim      # Truncation dimension
    ├── dt               # Time step
    └── operator_type    # "qbm"
```

---

## Meta Neural Operator (MNO) Dataset

### Purpose

Generate rollouts from a **pre-trained MNO model** to create diverse trajectory datasets for alignment tasks.

### Basic Usage

```bash
poetry run spinlock generate-mno-dataset \
  --mno-checkpoint checkpoints/mno/meta_operator_best.pt \
  --num-rollouts 100000 \
  --batch-size 128 \
  --output datasets/mno_rollouts_100k.h5
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--mno-checkpoint` | path | Pre-trained MNO model checkpoint |
| `--num-rollouts` | int | Number of trajectory rollouts to generate |
| `--batch-size` | int | Batch size for generation (GPU memory) |
| `--output` | path | Output HDF5 file |
| `--store-rollouts` | flag | Store full trajectories (default: False, features only) |

### Storage Modes

**Features-Only (Default):**
```bash
# Generate 100K rollouts, ~7GB storage
poetry run spinlock generate-mno-dataset \
  --mno-checkpoint checkpoints/mno/best.pt \
  --num-rollouts 100000 \
  --output datasets/mno_100k.h5
```

**With Full Trajectories:**
```bash
# Generate 100K rollouts, ~1.2TB storage (WARNING: large!)
poetry run spinlock generate-mno-dataset \
  --mno-checkpoint checkpoints/mno/best.pt \
  --num-rollouts 100000 \
  --store-rollouts \
  --output datasets/mno_100k_full.h5
```

### Dataset Structure

**Features-Only:**
```
mno_100k.h5
├── features/
│   ├── temporal/        # [N, T, 178] MNO trajectory features
│   └── initial/         # [N, 426] MNO initial condition features
└── metadata/
    ├── mno_checkpoint   # Source model path
    └── generation_date
```

**With Rollouts:**
```
mno_100k_full.h5
├── rollouts/
│   └── mno/             # [N, M, T, C, H, W] full trajectories
├── features/
│   ├── temporal/
│   └── initial/
└── metadata/
```

---

## Advanced Configuration

### Parallel Generation (Multi-GPU)

```bash
# Distribute across 4 GPUs
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu poetry run spinlock generate-cno-dataset \
    --num-samples 12500 \
    --output datasets/cno_gpu${gpu}.h5 \
    --seed $((42 + gpu)) &
done
wait

# Merge datasets
poetry run spinlock merge-datasets \
  --inputs datasets/cno_gpu*.h5 \
  --output datasets/cno_50k_merged.h5
```

### Custom Parameter Distributions

```python
# scripts/custom_generation.py
from spinlock.operators.convex import ConvexOperatorSampler

sampler = ConvexOperatorSampler(
    param_ranges={
        'diffusion': (0.1, 0.3),  # Narrow range for specific study
        'advection_x': (-0.5, 0.5),
        'reaction_rate': (0.01, 0.05),
    },
    operator_weights={
        'heat': 0.5,        # 50% heat equation
        'burgers': 0.3,     # 30% Burgers
        'reaction': 0.2,    # 20% reaction-diffusion
    }
)

# Generate with custom sampler
dataset = sampler.generate(num_samples=10000)
dataset.save('datasets/custom_distribution.h5')
```

### Quality Validation

```bash
# Validate generated dataset
poetry run spinlock validate-dataset \
  --dataset datasets/cno_50k.h5 \
  --checks all \
  --output validation_report.txt

# Expected output:
# ✓ Shape consistency: inputs/outputs match expected dimensions
# ✓ Parameter ranges: all within specified bounds
# ✓ Feature extraction: 178D temporal, 426D initial
# ✓ No NaN values: all features finite
# ✓ Operator diversity: 5 families represented
# ✓ Realization variance: >0.01 (sufficient diversity)
```

---

## Performance Benchmarks

### Generation Speed (NVIDIA A100)

| Operator | Samples/sec | 50K Dataset Time |
|----------|-------------|------------------|
| **CNO (mixed)** | ~120 | ~7 minutes |
| **QBM** | ~30 | ~28 minutes |
| **MNO rollouts** | ~250 | ~3.5 minutes |

### Storage Requirements

| Dataset | Size (features-only) | Size (with rollouts) |
|---------|---------------------|----------------------|
| CNO 50K | ~2.3 GB | ~45 GB |
| QBM 10K | ~1.1 GB | ~12 GB |
| MNO 100K | ~7 GB | ~1.2 TB |

---

## Troubleshooting

### OOM Errors

**Symptom:** CUDA out of memory during generation

**Solutions:**
```bash
# Reduce batch size
--batch-size 64  # Default is 128

# For QBM, reduce Hilbert dimension
--hilbert-dim 15  # Default is 20

# Generate in chunks
for i in {0..4}; do
  poetry run spinlock generate-cno-dataset \
    --num-samples 10000 \
    --seed $((42 + i)) \
    --output datasets/chunk_${i}.h5
done
```

### Slow Generation

**Symptom:** <10 samples/sec on GPU

**Solutions:**
```bash
# Check GPU utilization
nvidia-smi --loop=1

# Increase batch size if memory available
--batch-size 256

# Disable feature extraction during generation (extract later)
--skip-features

# Use torch.compile (requires PyTorch 2.0+)
--use-torch-compile
```

### Feature Extraction Failures

**Symptom:** NaN values in extracted features

**Solutions:**
```python
# Validate operator parameters
poetry run spinlock validate-parameters \
  --dataset datasets/problematic.h5

# Re-extract features with error handling
poetry run spinlock extract-features \
  --dataset datasets/raw.h5 \
  --output datasets/features.h5 \
  --handle-nans clip  # Clip to valid range instead of failing
```

---

## References

- LeVeque, "Finite Difference Methods for Ordinary and Partial Differential Equations" (2007)
- Breuer & Petruccione, "The Theory of Open Quantum Systems" (2002)
- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations" (2021)
