# NOA Architecture: Two Paradigms of Learning

## Vision: NOA as Creative Observer

The Neural Operator Agent (NOA) is fundamentally a **creative observer** of dynamical systems, embedded within the continuous flow of physical change yet equipped to interpret and articulate that change through a symbolic lens.

Rather than merely simulating trajectories with rigid fidelity to ground-truth rollouts, NOA generates its own "ideas" of evolution—distinct pathways that may diverge from observed reality but remain **coherent and meaningful** when translated into the discrete behavioral tokens of the VQ-VAE vocabulary.

This vision enables two distinct training paradigms:

| Paradigm | Philosophy | Primary Loss |
|----------|------------|--------------|
| **MSE-led** | "Replicate ground truth exactly" | L_traj (trajectory MSE) |
| **VQ-led** | "Be expressible in symbolic vocabulary" | L_recon (VQ reconstruction) |

---

## Paradigm 1: MSE-Led Training (Physics First)

**Philosophy:** Match CNO ground-truth trajectories as closely as possible.

```
Loss = λ_traj × L_traj + λ_commit × L_commit + λ_latent × L_latent
       ═══════════════
          PRIMARY
```

### Loss Components

| Loss | Description | Default λ |
|------|-------------|-----------|
| **L_traj** (PRIMARY) | MSE between NOA and CNO trajectories | 1.0 |
| L_commit | VQ commitment (manifold adherence) | 0.5 |
| L_latent | NOA-VQ latent alignment | 0.1 |

### When to Use MSE-Led

- **Early-stage training** to establish physics grounding
- When **exact trajectory matching** is critical
- **Benchmarking** against CNO baselines
- Physics fidelity is the primary metric

### Code Example

```python
from spinlock.noa.losses import MSELedLoss

loss_fn = MSELedLoss(
    lambda_traj=1.0,      # Primary: trajectory matching
    lambda_commit=0.5,    # Auxiliary: VQ commitment
    lambda_latent=0.1,    # Auxiliary: latent alignment
    vqvae_alignment=alignment,
)
```

### CLI Usage

```bash
poetry run python scripts/dev/train_noa_unified.py \
    --loss-mode mse_led \
    --lambda-traj 1.0 --lambda-commit 0.5 --lambda-latent 0.1
```

---

## Paradigm 2: VQ-Led Training (Creative Observer)

**Philosophy:** Generate outputs that are meaningful in the VQ-VAE vocabulary, even if they diverge from ground truth.

```
Loss = λ_recon × L_recon + λ_commit × L_commit + λ_traj × L_traj
       ════════════════
          PRIMARY
```

### Loss Components

| Loss | Description | Default λ |
|------|-------------|-----------|
| **L_recon** (PRIMARY) | VQ reconstruction quality | 1.0 |
| **L_commit** | VQ commitment (embedding sharpness) | 0.5 |
| L_traj | Trajectory MSE (physics regularizer) | 0.3 |

### The Surprise Principle

In vq-led training, deviations from ground truth become **opportunities for discovery**:

| Traditional View | Creative Observer View |
|-----------------|----------------------|
| High MSE = bad model | High MSE = novel perspective |
| Match CNO exactly | Generate valid symbolic sequences |
| Penalize deviation | Embrace meaningful divergence |

A "wrong" rollout by traditional metrics (high L_traj) could represent a **novel perspective** on dynamics—much like how different neural activation patterns in human brains converge on shared concepts despite varying implementations.

### When to Use VQ-Led

- After **physics grounding is established** (not for cold start)
- When **symbolic interpretation** matters more than exact matching
- Training for **downstream reasoning** with tokens
- Encouraging **"creative" exploration** of dynamics
- When VQ token quality is the primary metric

### Code Example

```python
from spinlock.noa.losses import VQLedLoss

loss_fn = VQLedLoss(
    lambda_recon=1.0,     # Primary: VQ reconstruction
    lambda_commit=0.5,    # Primary: commitment sharpness
    lambda_traj=0.3,      # Auxiliary: physics regularizer
    vqvae_alignment=alignment,
)
```

### CLI Usage

```bash
poetry run python scripts/dev/train_noa_unified.py \
    --loss-mode vq_led \
    --lambda-recon 1.0 --lambda-commit 0.5 --lambda-traj 0.3
```

---

## Architecture Details

### U-AFNO Backbone

```
Input: θ (parameters) + u₀ (initial condition)
  ↓
U-AFNO Backbone (spectral mixing + U-Net skip connections)
  ↓
Autoregressive Rollout: u₀ → u₁ → u₂ → ... → uₜ
  ↓
Feature Extraction (INITIAL + SUMMARY + TEMPORAL)
  ↓
VQ-VAE Encoding → Quantization → Tokens
  ↓
Loss Computation (mode-dependent)
```

### Abstract Base Classes

The architecture uses OOP abstractions for modularity:

```python
# Abstract backbone interface
class BaseNOABackbone(nn.Module, ABC):
    @abstractmethod
    def single_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single timestep prediction."""
        pass

    @abstractmethod
    def get_intermediate_features(self, x, extract_from="bottleneck"):
        """Extract features for alignment losses."""
        pass

# Abstract loss interface
class BaseNOALoss(nn.Module, ABC):
    @abstractmethod
    def compute(self, pred, target, ic, noa) -> LossOutput:
        """Compute all loss components."""
        pass

    @property
    @abstractmethod
    def leading_loss_name(self) -> str:
        """Name of primary loss term."""
        pass
```

### LossOutput Dataclass

Standardized output format for all loss functions:

```python
@dataclass
class LossOutput:
    total: torch.Tensor              # For backprop
    components: Dict[str, Tensor]    # Individual losses
    metrics: Dict[str, float]        # For logging (detached)
```

---

## Hyperparameter Guidelines

### Default Weights by Mode

| Mode | λ_traj | λ_recon | λ_commit | λ_latent |
|------|--------|---------|----------|----------|
| MSE-led | 1.0 | N/A | 0.5 | 0.1-0.5 |
| VQ-led | 0.3 | 1.0 | 0.5 | N/A |

### λ_traj in VQ-Led

The default λ_traj=0.3 in vq-led mode provides **enough physics regularization** to prevent complete drift while allowing symbolic creativity:

- **0.1-0.2:** Very creative, may drift far from physics
- **0.3:** Balanced (recommended starting point)
- **0.5:** More conservative, closer to physics
- **1.0:** Essentially MSE-led behavior

### Tuning Recommendations

1. **Start with MSE-led** for initial physics grounding
2. **Switch to vq-led** after L_traj stabilizes
3. **Monitor both metrics** even when not optimizing for them
4. **VQ reconstruction quality** is the key indicator for vq-led

---

## Training Pipeline

### Two-Phase Curriculum (Recommended)

**Phase 1: Physics Grounding (MSE-led)**
```bash
poetry run python scripts/dev/train_noa_unified.py \
    --loss-mode mse_led \
    --epochs 5 --lambda-traj 1.0
```

**Phase 2: Creative Exploration (VQ-led)**
```bash
poetry run python scripts/dev/train_noa_unified.py \
    --loss-mode vq_led \
    --resume checkpoints/noa/mse_led_best.pt \
    --epochs 5 --lambda-recon 1.0 --lambda-traj 0.3
```

### Truncated BPTT

For long sequences (T > 32), use truncated backpropagation:

```bash
--timesteps 256 --bptt-window 32
```

This limits gradient flow to the last 32 steps while supervising the full trajectory.

---

## Config Files

Pre-configured YAML files for both modes:

```bash
# MSE-led baseline
configs/noa/mse_led_baseline.yaml

# VQ-led creative
configs/noa/vq_led_creative.yaml
```

---

## Comparison: What Each Mode Optimizes

### MSE-Led

✅ Trajectory accuracy (low MSE)
✅ Physics fidelity
✅ CNO baseline matching
❌ May produce trajectories that don't tokenize well

### VQ-Led

✅ VQ reconstruction quality
✅ Token sequence coherence
✅ Symbolic expressibility
❌ May deviate from physics ground truth

---

## References

- **NOA Backbone:** `src/spinlock/noa/backbone.py`
- **Abstract Bases:** `src/spinlock/noa/base_backbone.py`, `src/spinlock/noa/base_loss.py`
- **Loss Functions:** `src/spinlock/noa/losses/`
- **Training Script:** `scripts/dev/train_noa_unified.py`
- **VQ-VAE Alignment:** `src/spinlock/noa/vqvae_alignment.py`
