# Quantum Brownian Motion (QBM) Dataset Configurations

This directory contains configuration files for generating QBM datasets.

## Overview

Quantum Brownian Motion simulations model open quantum systems exhibiting:
- **Decoherence**: Quantum → classical transition via environmental coupling
- **Dissipation**: Energy loss to thermal bath
- **Tunneling**: Quantum barrier penetration
- **Interference**: Wave-like quantum behavior

## Physics Model: Caldeira-Leggett

The Caldeira-Leggett model describes a quantum particle coupled to a thermal bath:

```
H = p²/(2m) + V(x,y)  (System Hamiltonian)
dρ/dt = -i[H, ρ]/ℏ + L[ρ]  (Lindblad dissipator)
```

### Parameters

QBM simulations are controlled by 12 parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `gamma` | [0.0001, 0.1] | Bath coupling strength (log scale) |
| `kT` | [0.01, 10.0] | Temperature in energy units (log scale) |
| `mass` | [0.1, 10.0] | Particle mass |
| `potential_type` | {0,1,2,3} | Type: harmonic, double-well, quartic, random |
| `potential_params[0-3]` | [0, 1] | Potential-specific parameters (scaled) |
| `reserved[0-3]` | [0, 1] | Reserved for future extensions |

### Initial Conditions (Quantum)

Distribution of quantum states:
- **40%** Gaussian wavepackets: ψ(r) = exp(ik·r) exp(-|r-r₀|²/(4σ²))
- **30%** Coherent states: Minimum uncertainty states
- **15%** 2-component superpositions: Quantum interference patterns
- **15%** 3-component superpositions: Complex interference

### Potential Types

1. **Harmonic** (40%): V = ½mω²r²
   - Closed orbits, coherent oscillations

2. **Double-well** (30%): V = -½mω²x² + λx⁴
   - Tunneling, symmetry breaking, bistability

3. **Quartic** (20%): V = c₀ + c₂r² + c₄r⁴
   - Anharmonic oscillations, classical chaos

4. **Random disorder** (10%): V(r) via Gaussian Random Field
   - Anderson localization, disorder effects

## Usage

### Generate 100K Dataset

```bash
poetry run spinlock generate-qbm-dataset \
  --num-rollouts 100000 \
  --batch-size 2 \
  --output datasets/qbm_100k.h5
```

### Quick Test (100 samples)

```bash
poetry run spinlock generate-qbm-dataset \
  --num-rollouts 100 \
  --batch-size 2 \
  --output datasets/qbm_test.h5
```

### Custom Parameters

```bash
poetry run spinlock generate-qbm-dataset \
  --num-rollouts 10000 \
  --batch-size 2 \
  --grid-size 128 \
  --domain-size 15.0 \
  --rollout-steps 512 \
  --output datasets/qbm_10k_highres.h5
```

## Dataset Structure

Generated HDF5 files follow the SpinlockDataset schema:

```
qbm_100k.h5
├── inputs/
│   └── fields: [N, M, 2, H, W]  # Quantum ICs (Re/Im wavefunctions)
├── parameters/
│   └── params: [N, 12]  # QBM parameters
├── features/
│   ├── temporal/
│   │   └── features: [N, T, D_t]  # Temporal features from rollouts
│   └── initial/
│       └── aggregated/
│           └── features: [N, D_i]  # Initial features from ICs
└── rollouts/ # Optional (only if --store-rollouts)
    └── qbm: [N, M, T, 2, H, W]  # Full trajectories
```

Where:
- `N` = number of rollouts (e.g., 100K)
- `M` = realizations per rollout (default: 3)
- `T` = timesteps (default: 256)
- `H, W` = spatial grid (default: 64×64)
- `D_t` = temporal feature dimension (~345)
- `D_i` = initial feature dimension (~42)

## Performance

With `batch_size=2` on modern GPU:
- **100K rollouts**: ~2-4 hours (depending on GPU)
- **Dataset size**: ~10-15GB (features only, with compression)
- **With rollouts**: Significantly larger (~100-200GB)

**Recommendation**: Use features-only (default) for training. Only store full rollouts if needed for visualization or analysis.

## Training U-AFNO on QBM Data

Once generated, train U-AFNO:

```bash
poetry run spinlock train-meta-operator \
  --config configs/noa/qbm_u_afno.yaml \
  --dataset datasets/qbm_100k.h5
```

## Physics Validation

Verify physics correctness:

```bash
# Run physics tests
poetry run pytest scripts/validation/test_qbm_physics.py -v

# Visualize samples
poetry run python scripts/validation/visualize_qbm_samples.py \
  --dataset datasets/qbm_100k.h5 \
  --num-samples 10
```

## References

- Caldeira, A. O. & Leggett, A. J. (1983). Path integral approach to quantum Brownian motion. *Physica A*, 121(3), 587-616.
- Breuer, H.-P. & Petruccione, F. (2002). *The Theory of Open Quantum Systems*. Oxford University Press.
- Gardiner, C. & Zoller, P. (2004). *Quantum Noise*. Springer.
