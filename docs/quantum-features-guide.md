# Quantum Features Guide

## Overview

Spinlock supports **Quantum Brownian Motion (QBM)** as a specialized PDE operator family, extracting 10-11 dimensional quantum-specific features that extend the standard temporal feature set.

## Quantum Brownian Motion (QBM)

### Physics Background

QBM describes a quantum harmonic oscillator coupled to a thermal environment, exhibiting:
- **Decoherence**: Loss of quantum coherence over time
- **Dissipation**: Energy exchange with environment
- **Thermalization**: Evolution toward thermal equilibrium

**Lindblad Master Equation:**
```
dρ/dt = -i[H, ρ] + γ(n̄ + 1)D[a]ρ + γn̄D[a†]ρ
```

Where:
- ρ: Density matrix (quantum state)
- H: System Hamiltonian
- γ: Dissipation rate
- n̄: Thermal occupation number
- D[L]ρ = LρL† - ½{L†L, ρ}: Lindblad dissipator

### Dataset Generation

```bash
poetry run spinlock generate-qbm-dataset \
  --num-samples 10000 \
  --num-realizations 3 \
  --timesteps 128 \
  --grid-size 64 \
  --output datasets/qbm_10k.h5 \
  --device cuda
```

**Parameters Sampled:**
- Initial coherent state amplitude: α ∈ [0, 3]
- Dissipation rate: γ ∈ [0.01, 0.5]
- Thermal photon number: n̄ ∈ [0, 2]
- Temperature: T ∈ [0.1, 2.0] (in units of ℏω/kB)

## Quantum Features

### 1. Purity

**Definition:** Tr(ρ²) - measures quantum state mixedness

**Properties:**
- Pure state: Tr(ρ²) = 1
- Maximally mixed: Tr(ρ²) = 1/d (d = dimension)
- Monotonic: Decreases under decoherence

**Implementation:**
```python
purity = torch.einsum('...ij,...ji->...', rho, rho).real
```

### 2. Von Neumann Entropy

**Definition:** S = -Tr(ρ log ρ)

**Properties:**
- Pure state: S = 0
- Maximally mixed: S = log d
- Subadditivity: S(ρAB) ≤ S(ρA) + S(ρB)

**Implementation:**
```python
eigenvalues = torch.linalg.eigvalsh(rho)
entropy = -torch.sum(eigenvalues * torch.log(eigenvalues + eps), dim=-1)
```

### 3. Coherence Measures

**Quantum Coherence:** Off-diagonal density matrix elements

**Metrics Extracted:**
- **Mean coherence**: Average |ρᵢⱼ| for i≠j
- **Max coherence**: max |ρᵢⱼ| for i≠j
- **Coherence norm**: ||ρ - diag(ρ)||₁

**Implementation:**
```python
# Extract off-diagonal elements
mask = ~torch.eye(d, dtype=bool, device=rho.device)
off_diag = rho[..., mask].reshape(*rho.shape[:-2], d, d-1)

coherence_mean = torch.abs(off_diag).mean(dim=(-2, -1))
coherence_max = torch.abs(off_diag).max(dim=-1)[0].max(dim=-1)[0]
```

### 4. Uncertainty Relations

**Heisenberg Uncertainty:** ΔxΔp ≥ ℏ/2

**Metrics:**
- Position variance: Δx² = ⟨x²⟩ - ⟨x⟩²
- Momentum variance: Δp² = ⟨p²⟩ - ⟨p⟩²
- Uncertainty product: ΔxΔp

**Implementation:**
```python
x_var = torch.var(position_samples, dim=-1)
p_var = torch.var(momentum_samples, dim=-1)
uncertainty_product = torch.sqrt(x_var * p_var)
```

### 5. Quantum Trajectory Fidelity

**Fidelity:** F(ρ, σ) = Tr(√(√ρ σ √ρ))

**Use Case:** Measure distance between quantum states at different times

**Implementation:**
```python
def fidelity(rho1, rho2):
    sqrt_rho1 = matrix_sqrt(rho1)
    prod = sqrt_rho1 @ rho2 @ sqrt_rho1
    sqrt_prod = matrix_sqrt(prod)
    return torch.trace(sqrt_prod).real
```

## Feature Vector Structure

### Standard Temporal Features (178D)

- Energy, gradients, curl, divergence
- Statistical moments (mean, std, skew, kurtosis)
- Spatial structure (Fourier modes, correlation lengths)

### Quantum Extension (+10-11D)

```python
quantum_features = {
    'purity': [T],                    # 1D time series
    'entropy': [T],                   # 1D time series
    'coherence_mean': [T],            # 1D time series
    'coherence_max': [T],             # 1D time series
    'coherence_norm': [T],            # 1D time series
    'uncertainty_x': [T],             # 1D time series
    'uncertainty_p': [T],             # 1D time series
    'uncertainty_product': [T],       # 1D time series
    'fidelity_t0': [T],              # 1D time series (vs initial state)
    'phase_coherence': [T],          # 1D time series (optional)
    'squeezing': [T],                # 1D time series (optional)
}

# Total: 188-189D temporal features for QBM
```

## Training with Quantum Features

### Configuration

```yaml
# Generate QBM dataset with quantum features
dataset:
  operator_type: "qbm"
  extract_quantum: true
  num_samples: 50000

# VQ-VAE tokenizer config
encoder:
  temporal:
    variant: "pyramid"
    num_pyramid_levels: 3
    # Handles 188-189D input (178 standard + 10-11 quantum)
```

### Expected Behavior

**Quantum features enable:**
1. **Decoherence detection**: Tokenizer learns to distinguish coherent vs decoherent regimes
2. **Thermal state identification**: Tokens cluster by equilibrium vs transient dynamics
3. **Parameter inference**: Quantum observables strongly correlated with γ and n̄

**Performance:**
- Tokenizer reconstruction MSE: <0.02 (including quantum features)
- Quantum feature MSE: <0.015 (higher precision for sensitive metrics)
- Decoherence time prediction: R² > 0.85 from tokens alone

## Integration Example

```python
from spinlock.data import SpinlockDataset
from spinlock.tokens import VQTokenizer

# Load QBM dataset with quantum features
dataset = SpinlockDataset.from_file('datasets/qbm_50k.h5')

with dataset.open():
    # Extract temporal features (standard + quantum)
    temporal_feats = dataset.features.temporal.load_all()  # [N, T, 188]

    # Quantum subset
    quantum_start_idx = 178
    quantum_feats = temporal_feats[..., quantum_start_idx:]  # [N, T, 10-11]

    # Tokenize (quantum features included in temporal encoding)
    tokenizer = VQTokenizer.from_checkpoint('checkpoints/qbm_tokenizer.pt')
    tokens = tokenizer.encode(temporal_features=temporal_feats)

    # Decode and verify quantum feature reconstruction
    reconstructed = tokenizer.forward(temporal_features=temporal_feats)
    quantum_recon = reconstructed['reconstructed_split']['temporal'][..., quantum_start_idx:]

    quantum_mse = F.mse_loss(quantum_recon, quantum_feats)
    print(f"Quantum feature MSE: {quantum_mse:.6f}")
```

## Visualization

### Decoherence Trajectory

```python
import matplotlib.pyplot as plt

# Plot purity decay
time = np.arange(timesteps) * dt
purity = quantum_feats[:, :, 0]  # First quantum feature

plt.figure(figsize=(10, 6))
for i in range(10):  # Plot 10 samples
    plt.plot(time, purity[i], alpha=0.7)
plt.xlabel('Time (ℏ/γ)')
plt.ylabel('Purity Tr(ρ²)')
plt.title('Quantum Decoherence Trajectories')
plt.grid(True, alpha=0.3)
```

### Coherence vs Entropy Phase Diagram

```python
plt.figure(figsize=(8, 8))
plt.scatter(
    quantum_feats[:, -1, 1],  # Final entropy
    quantum_feats[:, -1, 2],  # Final coherence
    c=temperatures,           # Color by temperature
    cmap='coolwarm',
    alpha=0.6
)
plt.xlabel('Von Neumann Entropy')
plt.ylabel('Coherence (mean)')
plt.title('Quantum State Phase Diagram')
plt.colorbar(label='Temperature (ℏω/kB)')
```

## Limitations

1. **Density matrix reconstruction**: Requires full state vector (expensive for large systems)
2. **Hilbert space truncation**: Limited to finite-dimensional approximations
3. **Classical simulation**: Exponential scaling limits to ~20-30 qubits

## References

- Breuer & Petruccione, "The Theory of Open Quantum Systems" (2002)
- Wiseman & Milburn, "Quantum Measurement and Control" (2009)
- Johansson et al., "QuTiP: Quantum Toolbox in Python" (2012)
