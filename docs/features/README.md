# Feature Families

Spinlock extracts **4 complementary feature families** that jointly capture neural operator behavior from different perspectives. This multi-modal representation provides a foundation for studying compositional reasoning, working memory dynamics, and episodic encoding in the controlled domain of dynamical systems.

## Overview

| Family | Dimensions | Captures | Granularity | Cognitive Role |
|--------|-----------|----------|-------------|----------------|
| **INITIAL** | 42D | Initial condition characteristics (spatial, spectral, information, morphology) | Per-realization | Input encoding |
| **ARCHITECTURE** | 21D+ | Operator parameters (architecture, stochastic, evolution) | Per-operator | Structural priors |
| **SUMMARY** | 420-520D | Aggregated behavioral statistics (spatial, spectral, temporal, causality) | Scalar summaries | Episodic compression |
| **TEMPORAL** | Variable | Full temporal trajectories preserving time-series structure | Per-timestep | Working memory |

## Feature Family Details

### Initial Condition (INITIAL) Features

**Location:** `src/spinlock/features/initial/`

42-dimensional hybrid features combining:
- **14 manual features:** Spatial, spectral, information-theoretic, and morphological characteristics
- **28 CNN embeddings:** ResNet-3 encoder extracting learned spatial patterns

See [Feature Reference](feature-reference.md) for complete feature specifications.

### Neural Operator Parameter (ARCHITECTURE) Features

**Location:** `src/spinlock/features/architecture/`

21+ dimensional features directly from the parameter space:
- Architecture features (6D): Layer counts, kernel sizes, channels
- Stochastic features (5D): Noise parameters, schedules
- Operator features (3D): Operator type identifiers
- Evolution features (3D): Update policies, integration parameters
- Stratification features (4D): Parameter space location

See [Feature Reference](feature-reference.md) for complete feature specifications.

### Summary Descriptor Features (SUMMARY)

**Location:** `src/spinlock/features/summary/`

420-520 dimensional aggregated behavioral statistics:

**v1.0 Features:**
- Spatial features (34D): Gradients, anisotropy, multiscale statistics
- Spectral features (31D): FFT power, dominant frequencies, spectral entropy
- Temporal features (44D): Autocorrelation, stationarity, regime transitions

**v2.0 Features:**
- Operator sensitivity (12D): Response to parameter perturbations
- Cross-channel correlations (12D): Multi-channel interactions
- Causality metrics (15D): Temporal information flow
- Invariant drift (64D): Long-term behavioral evolution

See [Feature Reference](feature-reference.md) for complete feature specifications.

### Temporal Dynamics (TEMPORAL) Features

**Location:** `src/spinlock/features/temporal/`

Variable-dimensional temporal features preserving time-series structure:
- Per-timestep feature vectors [N, M, T, D]
- Derived temporal curves (growth rates, curvature, oscillation)
- 1D CNN encoder for temporal pattern extraction

See [Feature Reference](feature-reference.md) for complete feature specifications.

## Joint Training

The VQ-VAE jointly trains on all 4 feature families simultaneously. Rather than treating each family independently, hierarchical clustering discovers natural groupings that span multiple feature types, enabling holistic operator behavioral patterns.

This multi-modal approach allows the model to learn representations that integrate:
- How initial conditions influence operator dynamics (INITIAL → input encoding)
- How architectural choices determine behavioral regimes (ARCHITECTURE → structural priors)
- Statistical signatures of emergent patterns (SUMMARY → episodic compression)
- Temporal evolution and regime transitions (TEMPORAL → working memory sequences)

### Cognitive Architecture Analogues

The four-family decomposition provides measurable analogues to cognitive processing:

**Input Encoding (INITIAL)**: Like sensory encoding in biological systems, INITIAL features characterize the "perceptual" properties of inputs that the operator will process. This enables studying how different input statistics influence downstream behavioral trajectories.

**Structural Priors (ARCHITECTURE)**: Analogous to innate biases or learned schemas, ARCHITECTURE features encode the operator's intrinsic computational structure—independent of specific inputs. This supports research into how architectural inductive biases shape behavioral regimes.

**Episodic Compression (SUMMARY)**: Similar to how episodic memory compresses experiences into gist representations, SUMMARY features aggregate temporal trajectories into statistical signatures. This compression is testable—can agents reconstruct or predict behaviors from compressed summaries?

**Working Memory (TEMPORAL)**: The distinction between SUMMARY (compressed) and TEMPORAL (sequential) mirrors the psychological distinction between long-term memory consolidation and working memory maintenance. Studying this trade-off reveals memory capacity constraints and what information is worth preserving vs. discarding.

These analogues are not metaphorical—they provide concrete, measurable frameworks for studying memory, compression, and representation in a domain without the confounds of natural perception.

## Feature Extraction Pipeline

```mermaid
flowchart LR
    Rollout[Neural Operator<br/>Rollout Data]
    INITIAL[INITIAL Extraction<br/>42D]
    ARCHITECTURE[ARCHITECTURE Extraction<br/>21D]
    SUMMARY[SUMMARY Extraction<br/>420-520D]
    TEMPORAL[TEMPORAL Extraction<br/>Variable]
    Concat[Feature<br/>Concatenation]
    Clean[Feature<br/>Cleaning]
    VQVAE[VQ-VAE<br/>Tokenization]

    Rollout --> INITIAL
    Rollout --> ARCHITECTURE
    Rollout --> SUMMARY
    Rollout --> TEMPORAL
    INITIAL --> Concat
    ARCHITECTURE --> Concat
    SUMMARY --> Concat
    TEMPORAL --> Concat
    Concat --> Clean
    Clean --> VQVAE
```

## Feature Cleaning

Before VQ-VAE training, all features undergo automatic cleaning:
1. **NaN removal:** Drop features with any NaN values
2. **Variance filtering:** Remove zero-variance features (threshold: 1e-8)
3. **Deduplication:** Remove highly correlated features (threshold: 0.99)
4. **Outlier capping:** Clip extreme values using MAD-based outliers (threshold: 5.0 MAD)

See [VQ-VAE Training Guide](../vqvae/training-guide.md) for details on the tokenization pipeline.

## Implementation

All feature extractors are GPU-accelerated and optimized for batch processing:
- Parallel extraction across multiple operators
- Efficient memory management for large datasets
- Inline feature computation during dataset generation

For usage examples, see [Getting Started](../getting-started.md).
