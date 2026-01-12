# Spinlock

**Foundation for Neural Operator Agent Research**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pre-training infrastructure for **Neural Operator Agents (NOA)**—foundation models that learn to predict, classify, and reason about dynamical system behaviors. Generate large-scale operator datasets, extract multi-modal behavioral features, and train VQ-VAE tokenizers for downstream scientific ML applications.

---

## Table of Contents

- [🎯 What is Spinlock?](#-what-is-spinlock)
- [🔬 Design Philosophy: Bias-Minimizing Discovery](#-design-philosophy-bias-minimizing-discovery)
- [🧠 Neural Operator Agents (NOA)](#-neural-operator-agents-noa)
- [🏗️ Architecture](#️-architecture)
- [📊 Feature Families](#-feature-families)
- [🎛️ VQ-VAE Behavioral Tokenization](#️-vq-vae-behavioral-tokenization)
- [⚡ Quick Start](#-quick-start)
- [🚀 Installation](#-installation)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📄 Citation](#-citation)
- [📜 License](#-license)

---

## 🎯 What is Spinlock?

Spinlock provides infrastructure for training **Neural Operator Agents (NOA)**—foundation models that learn to predict, classify, and reason about dynamical system behaviors across diverse parameter regimes. Rather than studying individual operators in isolation, the system learns meta-operators by treating operator parameter space and initial conditions as structured domains for systematic exploration and discovery.

### Foundation Models for Dynamical Systems

The system enables pre-training on diverse dynamical behaviors for downstream application across physics, metaphysics, biochemistry, and engineering—analogous to language model pre-training on text corpora. The current implementation demonstrates this approach through independent optimization that trains meta-operators on 100K+ neural operator trajectories.

**Current Implementation:**
- **Data:** 100K+ stratified operator trajectories with provably optimal parameter space coverage
- **Features:** Multi-modal behavioral descriptors (INITIAL, SUMMARY, TEMPORAL)
- **Meta-Operator:** U-AFNO backbone (226M params) trained via **independent optimization**:
  - Stage 1: Pure MSE physics training (no VQ constraints)
  - Stage 2: Generate large-scale MNO rollout features (100K+ samples)
  - Stage 3: Train VQ-VAE on MNO's distribution (alignment by construction)
- **Tokenization:** VQ-VAE trained on MNO rollouts discovers 10 behavioral categories with hierarchical 3-level codebooks

**Research Directions:**

*Dynamical Systems:*
- Surrogate modeling for accelerated simulation
- Anomaly detection in real-time sensor data
- Transfer learning to domain-specific PDEs
- Discovery of universal patterns in computational physics

*Cognitive Capabilities:*
- Meta-learning from dynamics: few-shot adaptation via abstract behavioral principles
- Compositional reasoning: predict emergent behaviors from component interactions
- Working memory: temporal state maintenance and transformation
- Episodic encoding: consolidation and retrieval of dynamical event sequences
- Cross-domain abstraction: domain-invariant behavioral patterns

---

## 🔬 Design Philosophy: Bias-Minimizing Discovery

Spinlock operates on a foundational principle: **discovering novel computational structures requires minimizing human-imposed semantic bias**. Rather than pre-defining behavioral categories or task-specific objectives, the system treats neural operator space as alien territory to be explored without preconceptions.

**Core Approach:**
- **Stratified sampling:** Sobol sequences with Owen scrambling provide provably optimal space-filling coverage (discrepancy <0.01), eliminating blind spots in high-dimensional parameter spaces
- **Data-driven features:** Multi-modal extraction (INITIAL, SUMMARY, TEMPORAL) captures comprehensive behavioral signatures without predetermined "interesting" features
- **Unsupervised tokenization:** VQ-VAE discovers discrete behavioral vocabularies through compression, learning categories from empirical data rather than human labels
- **Physics of change:** Study computational dynamics as a fundamental object, not task-specific optimization

This approach enables discovery of universal patterns, phase transitions, and emergent taxonomies that reflect the true geometry of operator behavior space—structures potentially alien to human intuition but fundamental to understanding computation as a physical process.

**Name Origin:** The name draws from quantum field spinlocking—coherence emerging from chaotic fluctuations through spin alignment. Similarly, this system discovers order arising from apparent chaos by systematically exploring stochastic neural operator behaviors to uncover stable, reproducible patterns in high-dimensional parameter space.

---

## 🧠 Neural Operator Agents (NOA)

Spinlock provides the infrastructure for building **Neural Operator Agents (NOA)**—hybrid neural operator systems with discrete VQ-VAE perceptual loss that learn to understand, generate, and reason about dynamical behaviors.

The NOA uses a **U-AFNO backbone** that operates directly in continuous function space, generating rollouts whose behavioral features are encoded into discrete tokens via a frozen VQ-VAE. This physics-native architecture enables self-consistent self-modeling and law discovery in the same function space as the dynamics being studied.

**Key Innovations**:
- **Physics-native backbone:** U-AFNO neural operator (not transformer on tokens) operating in continuous function space
- **Discrete perceptual loss:** VQ-VAE encodes NOA rollouts → behavioral tokens for loss computation
- **Topological encoding:** Parameter-space distance (not chronological time) enables reasoning about functional similarity

### The NOA Vision: From Data to Systematic Discovery

**Phase 0: Foundation** 
- Stratified neural operator datasets with diverse parameter coverage
- Multi-modal feature extraction (INITIAL, ARCHITECTURE, SUMMARY, TEMPORAL)
- Data-driven behavioral taxonomy via hierarchical clustering

**Phase 1: MNO Training (Meta-Neural Operator)** (🔄 In Progress - Independent Optimization)

The **Meta-Neural Operator (MNO)** is a neural network that learns to predict dynamics across an entire family of operators. Unlike a standard neural operator that learns a single PDE or system, the MNO learns a *meta-mapping* from operator parameters θ (architecture, coefficients) to their forward dynamics. Given any (θ, u₀) pair, it predicts the full rollout trajectory—essentially learning "what any operator in this family will do."

Why "meta"? It operates one level above individual operators, learning the relationship between operator structure and behavior across thousands of distinct systems.

- U-AFNO backbone (226M parameters) with pure MSE training
- Trains the **MNO**: pure physics simulator (no agency, no reasoning)
- Later phases build **NOA** (Neural Operator Agent) on top of MNO + tokens
- **Three-Stage Independent Optimization**:

  | Stage | Component | Objective | Output |
  |-------|-----------|-----------|--------|
  | **1: High Fidelity Physics Baseline** | MNO backbone | L_traj only (pure MSE) | Trained MNO (L_traj < 1.0) |
  | **2: Feature Generation** | Trained MNO | Generate 100K+ rollouts | Large-scale MNO features |
  | **3: Tokenization** | VQ-VAE | L_recon + L_commit | VQ-VAE aligned to MNO |

- **Philosophy**: Train tokenizer on simulator's distribution (VQ-VAE adapts to MNO, not vice versa). This avoids competing gradients that plague joint optimization.
- **Stage 1 (High Fidelity Physics)**: Pure MSE training achieves optimal physics accuracy (L_traj < 1.0) without VQ interference. No token conditioning, no VQ constraints—single objective optimization. Output: trained MNO.
- **Stage 2**: Generate massive MNO rollout dataset with inline feature extraction (100K+ samples)
- **Stage 3**: Train VQ-VAE on MNO's distribution - guaranteed alignment by construction
- **Truncated BPTT**: Long-horizon training (256-step rollouts, 32-step backprop window)
- **Training flow**: Stage 1 (MNO) → Stage 2 (features) → Stage 3 (VQ-VAE) → Foundation for NOA agent

**Three-Stage Rationale**: This architecture eliminates the fundamental tension between physics accuracy and VQ quality. In two-stage curriculum approaches, joint optimization creates competing gradients where improving one objective degrades the other, resulting in equilibrium plateaus (neither optimal). Independent optimization achieves:
1. **Optimal physics**: Stage 1 pure MSE trains optimal MNO (L_traj < 1.0) without VQ interference
2. **Optimal tokenization**: Stage 3 VQ-VAE adapts to MNO's distribution, achieving L_recon < 0.05
3. **Massive scale**: 100K+ MNO samples vs 1K CNO samples improves VQ-VAE training quality
4. **Architectural simplicity**: No token conditioning, no loss balancing, no coupled debugging

See [Independent Optimization Guide](docs/noa-vqvae-independent.md) for complete implementation details and [Two-Stage Curriculum (Deprecated)](docs/two-stage-curriculum-architecture.md) for empirical analysis of competing gradient issues.

---

**MNO vs NOA Distinction:**
- **MNO (Meta-Neural Operator)**: Pure physics simulator (Phase 1). No agency, no reasoning—just accurate trajectory prediction.
- **NOA (Neural Operator Agent)**: Higher-level agent architecture (Phase 2+). Uses MNO as physics engine + operates over VQ tokens. Has working memory, curiosity, planning capabilities.

---

**Phase 2: Multi-Observation Context** (📋 Planned)
- Lightweight transformer/recurrent heads on VQ token sequences
- Capture higher-order dependencies and temporal correlations
- In-context learning of operator physics through attention mechanisms

**Phase 3: Curiosity-Driven Exploration** (📋 Planned)
- Adaptive refinement: Agent identifies high-variance regimes (prediction error/surprise) and autonomously re-parameterizes sampling
- World model uncertainty: Track which regions of operator space are poorly understood
- Directed discovery: Use prediction error as curiosity signal to guide exploration toward behavioral frontiers
- Validation: Does curiosity-driven sampling discover fundamentally new behavioral categories?

**Phase 4: Transparent Self-Modeling** (📋 Planned)
- Self-model learning: Agent develops interpretable internal model of its own behavioral prediction process
- Calibration validation: Measure alignment between what the agent predicts about itself vs. actual performance
- Distributional shift detection: Self-model enables identifying when the agent encounters truly novel operator regimes
- Transparency requirement: Self-models must be inspectable—understand what the system "believes" about its own capabilities

**Phase 5: Systematic Discovery of Computational Laws** (📋 Planned)
- Hypothesis generation: Identify potential universal patterns in operator behavior (e.g., "operators with high spatial gradients exhibit turbulent temporal dynamics")
- Rigorous testing: Validate hypotheses through directed sampling and statistical analysis
- Symbolic regression: Distill discovered patterns into interpretable mathematical relationships
- Falsifiability: Every discovered "law" must be testable and potentially refutable

**Current Status:** Phase 0 complete, Phase 1 Stage 1 training in progress

**Complete Training Workflow:**
```bash
# Stage 1: Train NOA with pure MSE (no VQ constraints)
spinlock train-meta-operator \
    --config configs/noa/experiments/phase2/exp_pure_mse.yaml

# Stage 2: Generate large-scale NOA rollout features
spinlock generate-noa-features \
    --noa-checkpoint checkpoints/noa/pure_mse_baseline/meta_operator_best.pt \
    --config configs/noa/experiments/phase2/exp_pure_mse.yaml \
    --output datasets/noa_features_100k.h5 \
    --n-samples 100000

# Stage 3: Train VQ-VAE on NOA's distribution
spinlock train-vqvae \
    --config configs/vqvae/noa_distribution_100k.yaml
```

**Configuration:** Edit YAML configs to adjust:
- Model capacity (`base_channels: 32-48` for 144M-226M params)
- Training scale (`n_samples`, `batch_size`, `epochs`)
- Feature generation (`n_samples` for NOA rollouts)
- BPTT parameters (`timesteps`, `bptt_window`)

See [docs/architecture.md](docs/architecture.md) for system overview and [docs/noa-vqvae-independent.md](docs/noa-vqvae-independent.md) for complete training guide. For the deprecated two-stage curriculum approach, see [docs/two-stage-curriculum-architecture.md](docs/two-stage-curriculum-architecture.md).

---

## 🏗️ Architecture

Spinlock implements an end-to-end pipeline from dataset generation through meta-operator training:

```mermaid
flowchart TB
    Config[YAML Config] --> Sampling[Stratified Sampling]
    Sampling --> CNOs[CNO Operators]
    CNOs --> Rollouts[Rollout Execution]
    Rollouts --> Extract[Feature Extraction]
    Extract --> CNOData[CNO Dataset]

    CNOData --> Stage1[Stage 1:<br/>High Fidelity<br/>Physics Baseline]
    Stage1 --> MNOModel[Trained<br/>MNO Model<br/>L_traj < 1.0]

    MNOModel --> Stage2[Stage 2:<br/>Generate Features<br/>100K+ Samples]
    Stage2 --> MNOFeatures[MNO Distribution<br/>Dataset]

    MNOFeatures --> Stage3[Stage 3:<br/>Independent VQ<br/>Training]
    Stage3 --> VQVAEModel[VQ-VAE<br/>Aligned to MNO]

    MNOModel -.-> Deployment[Deployment]
    VQVAEModel -.-> Deployment

    classDef phase0 fill:#b0bec5,stroke:#455a64,stroke-width:2px,color:#000
    classDef stage1 fill:#c8e6c9,stroke:#4caf50,stroke-width:2px,color:#000
    classDef stage2 fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000
    classDef stage3 fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef deployment fill:#b3e5fc,stroke:#0277bd,stroke-width:2px,color:#000

    class Config,Sampling,CNOs,Rollouts,Extract,CNOData phase0
    class Stage1,MNOModel stage1
    class Stage2,MNOFeatures stage2
    class Stage3,VQVAEModel stage3
    class Deployment deployment
```

### Pipeline Overview

**Stage 0: Foundation - CNO Dataset Generation** (blue-grey)
1. Stratified parameter sampling via Sobol sequences (provably optimal space-filling)
2. CNO operator construction and stochastic rollout execution (256 steps)
3. Multi-modal feature extraction (INITIAL, SUMMARY, TEMPORAL)
4. CNO dataset establishing ground truth physics (1K samples for Stage 1 training)

**Stage 1: High Fidelity Physics Baseline** (green)
- Train **MNO** (Meta-Neural Operator) with **pure MSE loss** - no VQ constraints, no token conditioning, single objective
- Architecture: U-AFNO backbone (226M params) with Truncated BPTT (256-step rollouts, 32-step window)
- Input: (θ, u₀) from CNO dataset
- Loss: L_traj only (MSE vs CNO trajectories)
- Target: L_traj < 1.0 (RMSE < field variation, research-grade physics accuracy)
- Output: Trained MNO ready for feature generation

**Stage 2: MNO Feature Generation** (yellow)
- **Independent sampling**: Generate 100K+ rollouts from trained MNO (10-100× larger than CNO dataset)
- Sample diverse (θ, u₀) from same parameter space as Stage 0
- Extract features inline with GPU-optimized pipeline (99.99% storage savings vs full trajectories)
- **Key**: Features represent MNO's actual distribution, not CNO's
- Output: Large-scale MNO feature dataset (~1 GB for 100K samples)

**Stage 3: Independent VQ Tokenization** (purple)
- Train VQ-VAE on MNO's distribution using standard VQ-VAE training pipeline (proven, simple)
- Loss: L_recon + L_commit (no physics loss, no competing objectives)
- **Alignment by construction**: VQ-VAE learns to compress what MNO actually produces
- Target: L_recon < 0.05 (better than CNO baseline of 0.067 due to distribution match)
- Output: VQ-VAE optimized for MNO's behavioral manifold, ready for NOA agent (Phase 2+)

**Three-Stage Rationale**: This architecture eliminates the fundamental tension between physics accuracy and VQ quality. In two-stage curriculum approaches, joint optimization creates competing gradients where improving one objective degrades the other, resulting in equilibrium plateaus (neither optimal). Independent optimization achieves:
1. **Optimal physics**: Stage 1 pure MSE trains optimal MNO (L_traj < 1.0) without VQ interference
2. **Optimal tokenization**: Stage 3 VQ-VAE adapts to MNO's distribution, achieving L_recon < 0.05
3. **Massive scale**: 100K+ MNO samples vs 1K CNO samples improves VQ-VAE training quality
4. **Architectural simplicity**: No token conditioning, no loss balancing, no coupled debugging

See [Independent Optimization Guide](docs/noa-vqvae-independent.md) for complete implementation details and [Two-Stage Curriculum (Deprecated)](docs/two-stage-curriculum-architecture.md) for empirical analysis of competing gradient issues.

### Key Components

- **Stratified Sampling**: Sobol sequences with Owen scrambling for uniform parameter space coverage
- **Multi-Modal Features**: INITIAL (42D), ARCHITECTURE (21D), SUMMARY (420-520D), TEMPORAL (variable)
- **VQ-VAE Tokenization**: Automatic category discovery, hierarchical 3-level encoding, adaptive compression
- **Independent Optimization**: Three-stage pipeline (pure physics → feature generation → VQ-VAE training)
- **CLI Commands**: `spinlock generate`, `spinlock train-meta-operator`, `spinlock generate-noa-features`, `spinlock train-vqvae`

See [docs/architecture.md](docs/architecture.md) for comprehensive system design and implementation details.

---

## 📊 Feature Families

Spinlock extracts **4 complementary feature families** that jointly capture neural operator behavior from different perspectives:

| Family | Captures | Granularity |
|--------|----------|-------------|
| **INITIAL** | Initial condition characteristics (spatial, spectral, information, morphology) | Per-realization |
| **ARCHITECTURE** | Operator parameters (architecture, stochastic, evolution) | Per-operator |
| **SUMMARY** | Aggregated behavioral statistics (spatial, spectral, temporal, causality) | Per-rollout (aggregated across timesteps and realizations) |
| **TEMPORAL** | Full temporal trajectories preserving time-series structure | Per-timestep |

### Joint Training

The VQ-VAE jointly trains on all 4 families simultaneously, learning unified representations that span:
- **INITIAL**: How initial conditions influence operator dynamics
- **ARCHITECTURE**: How architectural choices determine behavioral regimes
- **SUMMARY**: Statistical signatures of emergent patterns
- **TEMPORAL**: Temporal evolution and regime transitions

This multi-modal training enables the model to discover behavioral categories that integrate structural, dynamical, and temporal characteristics—essential for NOA systems that reason about operator behavior.

See [docs/features/](docs/features/) for detailed feature definitions and extraction methods.

---

## 🎛️ VQ-VAE Behavioral Tokenization

The VQ-VAE pipeline transforms continuous behavioral features into discrete tokens—a compositional vocabulary for describing neural operator dynamics.

### Why "Categories"?

The term **categories** for the top-level groupings produced by orthogonality-weighted clustering is deliberate. These are not mere statistical clusters but conceptual primitives through which continuous dynamical behavior is coarse-grained into interpretable structure.

| Term | Why Not | Categories Are Different |
|------|---------|--------------------------|
| **Clusters** | Too neutral—implies data density, not conceptual primacy | Categories are the basic "kinds" of behavior, not density modes |
| **Modes** | Suggests spectral/vibrational modes (overlaps with AFNO terminology) | Categories are perceptual, not physical |
| **Prototypes** | Feels exemplar-based (like k-means centers) | Categories are hierarchical lenses, not single points |
| **Factors** | Evokes latent variables without hierarchical structure | Categories have multi-level refinement |

**Philosophical grounding:** Categories function as fundamental ways of understanding emergent behavior—akin to Aristotelian/Kantian categories that structure perception of reality. In the NOA's "mind," categories are perceptual building blocks: the agent "sees" the world through these coarse filters first (top-level codebooks), then refines within them (lower levels). The orthogonality weighting explicitly encourages independence, reinforcing their role as distinct, non-overlapping modes of interpretation.

**Long-term vision:** These categories are seeds of an emergent "language of computation" that NOA may use for reasoning and discovery—the first step in turning continuous physics into symbolic thought.

📖 **See also:** [docs/baselines/100k-full-features-vqvae.md](docs/baselines/100k-full-features-vqvae.md) for the full terminology discussion.

### Production Baseline: 100K Full Features

Our production model achieves **98.4% quality** with **43.9% codebook utilization** on 100,000 operators:

| Metric | Value |
|--------|-------|
| Val Loss | **0.115** |
| Reconstruction Quality | **0.984** (98.4%) |
| Reconstruction Error | **0.016** |
| Input Features | ~200 (after cleaning from 298 encoded) |
| Categories Discovered | **10** (auto-discovered via clustering) |
| Hierarchical Levels | 3 (coarse → medium → fine) |
| Total Codebooks | 30 (10 categories × 3 levels) |
| Codebook Utilization | **43.9%** |
| Topographic Similarity | **0.997** (post-quantization) |

**Key design choices:**
- **Adaptive compression ratios**: Per-category ratios computed from feature characteristics (variance, dimensionality, information, correlation)
- **Hybrid INITIAL encoder** with end-to-end CNN training (14D manual + 28D learned)
- **Pure clustering** for category discovery with orphan reassignment (100% feature assignment)
- **Higher commitment cost** (0.35) for improved codebook utilization
- **Correlation > variance**: Clustering prioritizes correlation patterns over variance scale

### Visualization Dashboards

```bash
# Generate all three dashboards
poetry run spinlock visualize-vqvae \
    --checkpoint checkpoints/production/100k_with_initial/ \
    --output visualizations/ \
    --type all
```

| Dashboard | Purpose |
|-----------|---------|
| **Engineering** | Training curves, utilization heatmap, architecture schematic |
| **Topological** | t-SNE codebook embeddings, inter-codebook similarity |
| **Semantic** | Feature→category mapping, category sizes, correlation |

📖 **Detailed documentation:** [docs/baselines/100k-full-features-vqvae.md](docs/baselines/100k-full-features-vqvae.md)

---

## ⚡ Quick Start

### Generate Operator Dataset

```bash
# Generate with default fast configuration (v1.0-v2.0 features, 64×64, T=256, M=5)
poetry run spinlock generate \
    --config configs/experiments/baseline_10k.yaml

# Or with all v2.1 features enabled (slower, more comprehensive)
# Add to config YAML:
# features:
#   summary:
#     distributional: {enabled: true}
#     structural: {enabled: true}
#     physics: {enabled: true}
#     morphological: {enabled: true}
```

### Inspect Dataset

```bash
poetry run spinlock inspect datasets/my_operators.h5
```

### Visualize Operator Dynamics

Generate videos showing temporal evolution of operators with aggregate views (PCA, variance, mean):

```bash
# Visualize convex operators (more dynamic, amoeba-like behavior)
poetry run spinlock visualize-dataset \
    --dataset datasets/100k_full_features.h5 \
    --output visualizations/convex_operators.mp4 \
    --evolution-policy convex \
    --sampling-method diverse \
    --aggregates pca variance mean
```

![Convex Operator Evolution](docs/images/convex_operators_evolution.png)

*Convex evolution policy produces sustained, morphing dynamics. Each row is an operator; columns show realizations and aggregate statistics (PCA modes as RGB, variance map, mean field).*

### Train VQ-VAE Tokenizer

```bash
# Train on full dataset with ARCHITECTURE + SUMMARY features
poetry run spinlock train-vqvae \
    --config configs/vqvae/production/10k_arch_summary_400epochs.yaml \
    --verbose

# Or train on validation dataset (1K samples) for testing
poetry run spinlock train-vqvae \
    --config configs/vqvae/validation/1k_arch_summary.yaml \
    --verbose
```

### Extract Behavioral Tokens

```python
import torch
import yaml
from pathlib import Path
from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig

# Load VQ-VAE configuration
with open("checkpoints/vqvae/config.yaml") as f:
    config_dict = yaml.safe_load(f)

# Construct model from config
config = CategoricalVQVAEConfig(**config_dict["model"])
model = CategoricalHierarchicalVQVAE(config)

# Load trained weights
checkpoint = torch.load("checkpoints/vqvae/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Extract behavioral tokens from new operators
with torch.no_grad():
    # features: [N, D] tensor of operator features
    tokens = model.get_tokens(features)  # [N, num_categories * num_levels]
```

See [docs/getting-started.md](docs/getting-started.md) for tutorials and examples.

---

## 🚀 Installation

**Requirements:** Python 3.11+, CUDA 11.8+ (for GPU acceleration)

```bash
git clone https://github.com/yourusername/spinlock.git
cd spinlock
poetry install
```

**Docker:** See [docs/installation.md#docker](docs/installation.md#docker)

**From Source:** See [docs/installation.md#source](docs/installation.md#source)

For detailed installation instructions, platform-specific guides, and troubleshooting, see [docs/installation.md](docs/installation.md).

---

## 📚 Documentation

- [**NOA Roadmap**](docs/noa-roadmap.md) - 5-phase development plan for Neural Operator Agents
- [**Architecture**](docs/architecture.md) - Detailed system design and implementation
- [**Independent Optimization Guide**](docs/noa-vqvae-independent.md) - **Recommended approach:** Complete guide for 3-stage independent training. Stage 1: High fidelity physics baseline (pure MSE, L_traj < 1.0). Stage 2: Generate 100K+ NOA rollout features. Stage 3: Train VQ-VAE on NOA's distribution (alignment by construction). Includes implementation details, hyperparameters, troubleshooting, and performance benchmarks. Achieves optimal physics + optimal tokenization without competing gradients.
- [**NOA Training Guide**](docs/noa-training-guide.md) - Training configuration, loss functions, checkpointing, and troubleshooting
- [**Two-Stage Curriculum (Deprecated)**](docs/two-stage-curriculum-architecture.md) - Alternative approach with token-conditioned training (Stage 1) and VQ-led fine-tuning (Stage 2). Deprecated due to competing gradients between L_traj and L_recon causing equilibrium plateaus at batch 500 where neither objective optimizes fully. Empirical analysis shows NOA learns "VQ-friendly" dynamics at cost of physics accuracy. See independent optimization for superior approach.
- [**Feature Families**](docs/features/README.md) - INITIAL, ARCHITECTURE, SUMMARY, TEMPORAL feature definitions and extraction
- [**HDF5 Layout**](docs/features/hdf5-layout.md) - Dataset schema reference for VQ-VAE pipeline
- [**Baselines**](docs/baselines/README.md) - Production datasets and VQ-VAE tokenizers
  - [100K Dataset](docs/baselines/100k-full-features-dataset.md) - 100K operators with INITIAL+SUMMARY+TEMPORAL+ARCHITECTURE features
  - [100K VQ-VAE](docs/baselines/100k-full-features-vqvae.md) - Tokenizer (val_loss: 0.172, quality: 0.95, utilization: 67%)
- [**Getting Started**](docs/getting-started.md) - Tutorials and end-to-end examples
- [**Installation**](docs/installation.md) - Platform-specific installation guides

---

## 🔮 Future Directions

### Multi-Agent Token Communication

The independent optimization architecture enables a critical capability for collaborative discovery: **discrete symbolic communication between agents**. By operating over shared VQ-VAE token vocabularies, multiple NOA instances can engage in compositional reasoning, emergent communication protocols, and collaborative parameter space exploration.

**Key insight:** The independent optimization architecture naturally enables both symbolic and physics-accurate reasoning:

| VQ-VAE Tokens (System 1) | NOA Rollouts (System 2) |
|--------------------------|------------------------|
| Fast symbolic reasoning | Precise physics execution |
| Token-based communication | Continuous trajectories |
| Collaborative exploration | Ground-truth verification |
| Categorical classification | Quantitative prediction |

**Example workflow:**
```python
# Agent A: Fast symbolic screening
for theta in search_space:
    rollout = noa(theta, u0)
    features = extract_features(rollout, u0)
    tokens = vqvae.encode(features)
    if tokens match TARGET_CATEGORY:
        send_message(agent_b, tokens, theta)

# Agent B: Precise verification
for (tokens, theta) in messages:
    trajectory = noa(theta, u0)
    evaluate_exact_metrics(trajectory)
```

**Research directions:**
- Emergent compositional communication protocols
- Hierarchical multi-resolution discourse (L0/L1/L2 tokens)
- Token-based theory of mind
- Cross-domain behavioral transfer via shared vocabulary

📖 **Full documentation:** [docs/future/multiagent-token-communication.md](docs/future/multiagent-token-communication.md)

---

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines for:
- Code style and formatting
- Testing requirements
- Pull request process

For bugs and feature requests, please open an issue on GitHub.

---

## 📄 Citation

If you use Spinlock in your research, please cite:

```bibtex
@software{spinlock2024,
  title = {Spinlock: Foundation for Neural Operator Agent Research},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/spinlock}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Poetry](https://python-poetry.org/) - Dependency management
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) - Efficient data storage

Spinlock is part of ongoing research into meta-cognitive neural operator systems and autonomous scientific discovery.
