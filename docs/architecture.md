# Spinlock Architecture

**Stratified neural operator dataset generation and hierarchical VQ-VAE tokenization system.**

## System Overview

```mermaid
flowchart TD
    subgraph Row1[" "]
        direction LR
        subgraph Config["Configuration Layer"]
            direction TB
            YAML[YAML Config Files]
            Schema[Parameter Schema]
        end

        subgraph Sampling["Stratified Sampling"]
            direction TB
            Sobol[Sobol Sequences]
            Owen[Owen Scrambling]
            Stratify[Stratification]
        end

        subgraph Generation["Operator Generation"]
            direction TB
            Builder[Operator Builder]
            CNN[CNN Construction]
            Params[Parameter Mapping]
        end
    end

    subgraph Row2[" "]
        direction LR
        subgraph Execution["Rollout Execution"]
            direction TB
            INITIAL[Initial Conditions]
            Engine[Rollout Engine]
            Storage[HDF5 Storage]
        end

        subgraph Features["Feature Extraction"]
            direction TB
            ICExt[INITIAL: 42D]
            NOPExt[ARCHITECTURE: 21D]
            SDFExt[SUMMARY: 420-520D]
            TDExt[TEMPORAL: Variable]
        end

        subgraph Encoding["VQ-VAE Tokenization"]
            direction TB
            Clean[Feature Cleaning]
            Cluster[Category Discovery]
            Train[Hierarchical VQ-VAE]
            Tokens[Behavioral Tokens]
        end
    end

    Config --> Sampling
    Sampling --> Generation
    Generation --> Execution
    Execution --> Features
    Features --> Encoding

    style Row1 fill:none,stroke:none
    style Row2 fill:none,stroke:none
```

## Core Components

### 1. Configuration System
**Location:** `src/spinlock/config/`

- **Parameter schema:** Defines operator parameter spaces
- **Stratification:** Sobol-based low-discrepancy sampling
- **Validation:** Type checking and constraint enforcement

### 2. Operator Generation
**Location:** `src/spinlock/operators/`

- **Parameter mapping:** Maps Sobol samples to CNN architectures
- **Operator builder:** Constructs neural operators from parameters
- **Block composition:** Modular building blocks for operator design

### 3. Rollout Execution
**Location:** `src/spinlock/rollout/`

- **Initial conditions:** 28 INITIAL types across 5 diversity tiers
- **Stochastic rollouts:** 500 timesteps × 3 realizations per operator
- **Execution engine:** Batched GPU execution with memory optimization

### 4. Feature Extraction
**Location:** `src/spinlock/features/`

Four complementary feature families:
- **INITIAL (Initial Condition):** 42D hybrid features
- **ARCHITECTURE (Neural Operator Parameters):** 21D+ parameter features
- **SUMMARY (Summary Descriptor Features):** 420-520D aggregated statistics
- **TEMPORAL (Temporal Dynamics):** Variable temporal resolution features

See [Feature Families](features/README.md) for details.

## Multi-Modal Integration for Interpretability

### Why Four Feature Families?

The decomposition into INITIAL, ARCHITECTURE, SUMMARY, and TEMPORAL is not arbitrary—each family answers distinct questions about operator behavior:

| Family | Question Answered | Interpretability Value |
|--------|------------------|----------------------|
| **INITIAL** | How do input characteristics influence behavior? | Identifies sensitivity to initial conditions |
| **ARCHITECTURE** | Which design choices determine behavioral regimes? | Links structure to function explicitly |
| **SUMMARY** | What are the observable signatures of behavior? | Provides statistical evidence of patterns |
| **TEMPORAL** | How do behaviors evolve and transition? | Reveals dynamical mechanisms |

### Cross-Validation Through Multiple Perspectives

Multi-modal training enables **consistency checking**:
- If ARCHITECTURE suggests chaotic behavior, do SUMMARY entropy features confirm this?
- If TEMPORAL shows period-doubling bifurcations, do SUMMARY spectral features detect harmonics?
- If INITIAL indicates smooth inputs, does SUMMARY show expected spatial autocorrelation?

This cross-validation improves confidence that discovered categories reflect genuine behavioral differences, not statistical artifacts.

### Transparent Behavioral Taxonomy

The VQ-VAE codebook learns to compress behavior across all four perspectives simultaneously. This creates tokens that:
1. **Integrate evidence** from structure, statistics, and dynamics
2. **Are interpretable** through feature-space attribution
3. **Enable validation** by reconstructing interpretable features

Unlike end-to-end learned representations, this approach maintains a **transparent chain of reasoning**:
```
Raw dynamics → Interpretable features → Hierarchical clustering → Discrete tokens
     ↓              ↓                         ↓                        ↓
Observable    Statistical/    Data-driven      Discrete behavioral
behavior      structural      categories       vocabulary
              semantics       (inspectable)    (interpretable)
```

### 5. Data-Driven Behavioral Taxonomy
**Location:** `src/spinlock/encoding/`

- **Automatic feature cleaning:** NaN removal, variance filtering, deduplication
- **Category discovery:** Hierarchical clustering across all feature families
- **Hierarchical VQ:** 3-level discrete latent space (coarse → medium → fine)
- **Joint training:** Unified representations across INITIAL+ARCHITECTURE+SUMMARY+TEMPORAL

#### Interpretability Properties

The hierarchical clustering approach provides several transparency advantages:

1. **Inspectable Categories**: Unlike end-to-end learned embeddings, clusters can be characterized by:
   - Feature-space centroids (what defines each behavioral category?)
   - Feature attribution (which features distinguish categories?)
   - Hierarchical relationships (how do fine-grained behaviors relate to coarse categories?)

2. **Validation Pathways**:
   - Cluster quality metrics (silhouette score, Davies-Bouldin index)
   - Inter-cluster distances (are categories well-separated?)
   - Utilization rates (are all tokens meaningful, or are some unused?)

3. **Human-Interpretable Discoveries**:
   - Behavioral categories emerge from data, but can be validated by domain experts
   - Feature decomposition enables understanding *why* certain operators cluster together
   - Hierarchical structure reveals natural behavioral taxonomies

See [VQ-VAE Training Guide](vqvae/training-guide.md) for details.

### 6. Dataset Storage
**Location:** `src/spinlock/dataset/`

- **HDF5 format:** Efficient storage with compression
- **Metadata tracking:** INITIAL types, evolution policies, parameter stratification
- **Chunked I/O:** Optimized for large-scale dataset generation

### 7. Visualization
**Location:** `src/spinlock/visualization/`

- **Temporal evolution rendering:** Heatmap, RGB, PCA-based rendering
- **Aggregate statistics:** Mean, variance, FFT visualization
- **Video export:** MP4 and GIF generation

## Performance Characteristics

### Dataset Generation (Baseline 10K)
- **Operator count:** 10,000 neural operators
- **Rollout size:** 500 timesteps × 3 realizations
- **Grid resolution:** 128×128 (fixed for VQ-VAE compatibility)
- **Generation time:** ~12 hours (GPU-accelerated)
- **Dataset size:** ~50-100 GB (HDF5 compressed)

### Feature Extraction
- **Inline extraction:** Features computed during rollout generation
- **GPU acceleration:** Batch processing for INITIAL, SUMMARY, TEMPORAL features
- **Memory optimization:** Streaming computation for large datasets

### VQ-VAE Training
- **Input features:** ~500-600D after cleaning (INITIAL+ARCHITECTURE+SUMMARY+TEMPORAL concatenation)
- **Categories:** ~8-15 automatically discovered via clustering
- **Codebook sizes:** Configurable (typically 128-512 codes per level)
- **Training time:** ~2-6 hours for 10K dataset (GPU)

## Design Principles

### 1. Modularity
- Clean separation between config, sampling, generation, execution, features, encoding
- Composable building blocks for operators
- Extensible feature extraction system

### 2. DRY (Don't Repeat Yourself)
- Shared utilities across feature extractors
- Unified parameter mapping system
- Reusable GPU kernels for feature computation

### 3. Extensibility
- Easy addition of new feature families
- Pluggable operator architectures
- Configurable VQ-VAE architectures

### 4. Performance
- GPU-first design for all compute-intensive operations
- Batched processing throughout pipeline
- Memory-efficient streaming for large datasets

### 5. Reproducibility
- Stratified sampling for parameter space coverage
- Deterministic rollouts (seeded randomness)
- Comprehensive metadata tracking

## Data Flow

```mermaid
flowchart TD
    Config[Config YAML]
    Params[Parameter Vectors]
    Operators[Neural Operators]

    Rollouts[Stochastic Rollouts]
    Features[Multi-Modal Features]

    Cleaned[Cleaned Features]
    VQVAE[VQ-VAE Model]

    Tokens[Behavioral Tokens]
    NOA[Neural Operator Agent]

    Config --> Params --> Operators
    Operators --> Rollouts --> Features
    Features --> Cleaned --> VQVAE
    VQVAE --> Tokens --> NOA

    style Config fill:#e1f5e1,color:#000
    style NOA fill:#e1e8f5,color:#000
```

## References

- [NOA Roadmap](noa-roadmap.md) - 5-phase development plan
- [Feature Families](features/README.md) - INITIAL, ARCHITECTURE, SUMMARY, TEMPORAL documentation
- [Getting Started](getting-started.md) - Usage tutorials
- [Installation](installation.md) - Setup instructions
