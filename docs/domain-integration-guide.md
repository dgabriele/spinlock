# Domain Integration Guide: Adding New Physics Domains

## Overview

This guide provides practical instructions for adding new physics domains to the multi-domain MNO architecture. It covers domain selection, dataset generation, MNO architecture considerations, VQ-VAE training, and cross-domain testing procedures.

**Target Audience:** Researchers and engineers implementing new domain integrations.

---

## Domain Selection Criteria

### Complementary Physics Families

When choosing the next domain to add, consider:

**1. Mathematical Diversity**

Choose domains with different mathematical structures:
- **Parabolic PDEs** (reaction-diffusion, heat equation) - already covered
- **Hyperbolic PDEs** (wave equations, acoustics) - fundamentally different propagation
- **Elliptic PDEs** (steady-state problems) - if time-dependent variants exist
- **Mixed-type equations** (fluid dynamics with multiple regimes)
- **Integro-differential equations** (non-local interactions)

**Rationale:** Testing vocabulary alignment across equation types probes deeper universality.

**2. Physical Mechanism Diversity**

Choose domains with different dominant physics:
- **Diffusion-dominated** (heat, chemical concentration) - ✓ RD covered
- **Advection-dominated** (fluids, transport phenomena) - next priority
- **Wave propagation** (acoustics, elasticity, EM waves)
- **Conservation laws** (traffic flow, shallow water)
- **Quantum dynamics** (Schrödinger, density functional theory)

**Rationale:** Different mechanisms test whether behavioral categories transcend specific physics.

**3. Observable Behaviors**

Choose domains with rich, interpretable behavioral regimes:
- Multiple steady states (laminar vs turbulent, different wave modes)
- Transitions and bifurcations (onset of instability, regime changes)
- Pattern formation (vortices, oscillations, coherent structures)
- Temporal dynamics (oscillatory, damping, growth, stationary)

**Rationale:** Rich behavior enables semantic correspondence analysis.

**4. Existing Knowledge**

Prefer domains with:
- Well-characterized parameter spaces (Reynolds number, wave number, etc.)
- Known behavioral regimes (turbulent cascade, wave dispersion)
- Established numerical methods (stable solvers, validated codes)
- Interpretable visualizations (vorticity, velocity, pressure fields)

**Rationale:** Enables validation and interpretation of learned categories.

### Recommended Domain Sequence

**Priority 1: 2D Navier-Stokes (Fluid Dynamics)**
- **Why:** Mathematically different from RD (advection vs diffusion), rich behaviors (laminar, transitional, turbulent), well-characterized (Reynolds number), excellent test case for vocabulary alignment
- **Expected:** ~10 categories (laminar, vortex shedding, turbulent, etc.)
- **Hypothesis Test:** Do "oscillatory" in RD ↔ "vortex shedding" in fluids?

**Priority 2: 2D Wave Equation**
- **Why:** Hyperbolic (vs parabolic RD/Navier-Stokes), fundamentally different (propagation vs diffusion), interpretable (standing waves, dispersion)
- **Expected:** ~10 categories (propagating, standing, dispersive, etc.)
- **Hypothesis Test:** Do wave categories align with RD/fluids?

**Priority 3: 2D Burgers' Equation**
- **Why:** Simpler than Navier-Stokes but captures shock formation, bridges diffusion and advection, analytically tractable
- **Expected:** Categories for smooth, shock, dispersive regimes
- **Hypothesis Test:** Shock formation ↔ discontinuity categories in other domains?

**Priority 4: Quantum 2D Schrödinger**
- **Why:** Complex-valued fields (fundamentally different), quantum tunneling and interference, ultimate test of universality (classical vs quantum)
- **Expected:** Categories for bound states, scattering, interference
- **Hypothesis Test:** Do quantum categories align with classical ones?

---

## Dataset Generation Workflow

### Step 1: Define Parameter Space

**Goal:** Sample diverse operators covering the domain's behavioral regimes.

**Example: 2D Navier-Stokes**

```python
# Define parameter space
parameter_space = {
    "reynolds_number": {
        "type": "log-uniform",
        "min": 10.0,      # Laminar regime
        "max": 1000.0,    # Turbulent regime
    },
    "forcing_amplitude": {
        "type": "uniform",
        "min": 0.1,
        "max": 2.0,
    },
    "forcing_wavenumber": {
        "type": "discrete",
        "values": [4, 6, 8, 12, 16],
    },
    "viscosity": {
        "type": "log-uniform",
        "min": 1e-4,
        "max": 1e-2,
    },
}

# Sample using Sobol sequence for coverage
n_operators = 10000
sampler = SobolSampler(parameter_space)
operators = sampler.sample(n_operators)
```

**Guidelines:**
- Cover full range of known behavioral regimes
- Use Sobol sampling for parameter space coverage
- Include edge cases and transitions
- Aim for 5K-10K operators (CNO dataset size)

### Step 2: Generate CNO Dataset

**Goal:** Create ground truth trajectories for each operator.

```bash
# Generate CNO dataset
spinlock generate-cno-dataset \
    --domain fluids \
    --equation navier_stokes_2d \
    --config configs/cno/fluids_10k.yaml \
    --output data/cno/fluids_10k/ \
    --num_operators 10000 \
    --trajectory_length 256 \
    --resolution 128 \
    --num_workers 32
```

**Configuration Example:**

```yaml
# configs/cno/fluids_10k.yaml
dataset:
  domain: fluids
  equation: navier_stokes_2d
  resolution: [128, 128]
  num_operators: 10000
  trajectory_length: 256
  dt: 0.01

parameter_sampling:
  reynolds_number:
    distribution: loguniform
    min: 10.0
    max: 1000.0
  forcing_amplitude:
    distribution: uniform
    min: 0.1
    max: 2.0
  forcing_wavenumber:
    distribution: discrete
    values: [4, 6, 8, 12, 16]
  viscosity:
    distribution: loguniform
    min: 1e-4
    max: 1e-2

initial_conditions:
  type: random_smooth
  smoothness_scale: 0.1
  num_per_operator: 5  # Generate 5 ICs per operator

solver:
  method: pseudospectral
  dealiasing: true
  time_integrator: rk4
  cfl_factor: 0.5

validation:
  energy_conservation_tolerance: 1e-3
  divergence_tolerance: 1e-6
```

**Guidelines:**
- Use stable, validated numerical solvers
- Verify physical constraints (energy conservation, incompressibility, etc.)
- Generate multiple ICs per operator for diversity
- Save in HDF5 format compatible with Spinlock

### Step 3: Validate Dataset Quality

**Goal:** Ensure dataset captures diverse behaviors without numerical artifacts.

```python
def validate_cno_dataset(dataset_path):
    """
    Validate CNO dataset quality before MNO training.
    """
    dataset = load_cno_dataset(dataset_path)

    # 1. Check for NaN/Inf
    assert not torch.isnan(dataset.trajectories).any()
    assert not torch.isinf(dataset.trajectories).any()

    # 2. Check parameter space coverage
    param_coverage = analyze_parameter_coverage(dataset.parameters)
    plot_parameter_distributions(param_coverage, save_path="param_coverage.pdf")

    # 3. Check behavioral diversity
    # Compute simple statistics: mean, variance, temporal autocorrelation
    stats = compute_trajectory_statistics(dataset.trajectories)
    plot_statistics_distributions(stats, save_path="behavior_diversity.pdf")

    # 4. Visualize sample trajectories
    sample_indices = np.random.choice(len(dataset), size=20)
    for idx in sample_indices:
        traj = dataset[idx]
        plot_spatiotemporal_evolution(traj, save_path=f"samples/traj_{idx}.pdf")

    # 5. Check for solver artifacts
    # Look for grid-aligned patterns, numerical instabilities
    check_numerical_artifacts(dataset.trajectories)

    print("Dataset validation complete.")
```

**Red Flags:**
- Narrow distribution of behaviors (adjust parameter space)
- Numerical instabilities (fix solver settings)
- Unphysical values (check boundary conditions)
- Grid artifacts (refine spatial resolution or dealiasing)

---

## MNO Architecture Considerations

### Domain-Specific Architecture Choices

Different equation types benefit from different architectures:

**Parabolic PDEs (Reaction-Diffusion, Heat)**
- **Architecture:** U-AFNO (U-Net + Adaptive Fourier Neural Operator)
- **Why:** Diffusion-dominated, spatially smooth, benefits from spectral methods
- **Proven:** 226M parameter MNO-RD achieves L_traj < 1.0

**Hyperbolic PDEs (Waves, Advection)**
- **Architecture:** U-AFNO with modifications for sharp features
- **Modifications:**
  - Reduce spectral filtering to preserve discontinuities
  - Add shock-capturing mechanisms if needed
  - Consider hybrid spectral-spatial approaches
- **Challenge:** Sharp fronts and discontinuities

**Incompressible Fluids (Navier-Stokes)**
- **Architecture:** U-AFNO with divergence-free constraints
- **Modifications:**
  - Enforce ∇·u = 0 via projection layer
  - Use vorticity-stream function formulation (naturally div-free)
  - Or use vector-valued AFNO with learned constraints
- **Challenge:** Physical constraints must be enforced

**Complex-Valued Fields (Schrödinger, EM Waves)**
- **Architecture:** Complex-valued U-AFNO
- **Modifications:**
  - Complex convolutions and activations
  - Phase-preserving nonlinearities
  - Energy/norm conservation layers
- **Challenge:** Complex number arithmetic, unitarity

### Configuration Template

```yaml
# configs/noa/fluids_pure_mse.yaml
model:
  architecture: u_afno
  type: meta_neural_operator
  domain: fluids  # NEW: specify domain

  # Domain-specific modifications
  divergence_free: true  # For incompressible fluids
  complex_valued: false  # For quantum/EM domains
  shock_capturing: false  # For hyperbolic equations

  # Architecture hyperparameters (tune per domain)
  encoder:
    channels: [64, 128, 256, 512]
    kernel_size: 3
    activation: gelu

  afno:
    num_blocks: 8
    num_heads: 8
    mlp_ratio: 4.0
    modes: 32  # May need adjustment for different domains
    spectral_filter: 0.8  # Reduce for sharp features

  decoder:
    channels: [512, 256, 128, 64]
    kernel_size: 3
    activation: gelu

  output:
    channels: 1  # Or 2 for vector fields (u, v)

training:
  loss:
    type: trajectory_mse  # Stage 1: pure physics
    trajectory_weight: 1.0
    prediction_horizons: [1, 2, 4, 8, 16, 32, 64, 128, 256]

  optimizer:
    type: adamw
    lr: 1e-4
    weight_decay: 1e-5
    betas: [0.9, 0.999]

  scheduler:
    type: cosine_annealing
    T_max: 100
    eta_min: 1e-6

  batch_size: 4  # Adjust based on GPU memory
  num_epochs: 100
  gradient_clip: 1.0

  # Validation
  val_frequency: 1
  val_metric: l_traj
  early_stopping_patience: 10

dataset:
  path: data/cno/fluids_10k/
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  num_workers: 8
```

### Training Guidelines

**Stage 1: Pure Physics MNO**

```bash
# Train MNO on new domain
spinlock train-meta-operator \
    --config configs/noa/fluids_pure_mse.yaml \
    --output checkpoints/noa/fluids/ \
    --experiment fluids_pure_mse_v1
```

**Target Metrics:**
- L_traj < 1.0 (physics-normalized trajectory loss)
- Validation loss converged
- No overfitting (train/val gap small)

**Debugging:**
- If loss doesn't decrease: Check solver correctness, dataset quality
- If overfitting: Increase regularization, add data augmentation
- If unstable: Reduce learning rate, add gradient clipping
- If slow convergence: Check architecture capacity, increase model size

---

## Feature Extraction Strategy

### Domain-Appropriate Features

Extract features that capture relevant physics for the domain:

**Reaction-Diffusion:**
- Raw state: u(x, y, t)
- Spatial gradients: ∇u
- Laplacian: ∇²u
- Temporal derivative: ∂u/∂t
- Statistical: mean, variance, skewness

**Fluid Dynamics:**
- Velocity field: u(x, y, t), v(x, y, t)
- Vorticity: ω = ∂v/∂x - ∂u/∂y
- Kinetic energy: ½(u² + v²)
- Enstrophy: ½ω²
- Divergence: ∇·u (should be ~0)

**Wave Equations:**
- Displacement: u(x, y, t)
- Velocity: ∂u/∂t
- Energy density: ½[(∂u/∂t)² + c²|∇u|²]
- Wave number spectra: FFT(u)
- Phase: arg(u) for complex waves

**Quantum (Schrödinger):**
- Wavefunction: ψ(x, y, t) (complex)
- Probability density: |ψ|²
- Phase: arg(ψ)
- Current density: Im(ψ*∇ψ)
- Energy expectation: ⟨ψ|H|ψ⟩

### Feature Generation

```bash
# Generate features from trained MNO
spinlock generate-noa-features \
    --noa-checkpoint checkpoints/noa/fluids/best.pt \
    --operator-dataset data/cno/fluids_10k/ \
    --output data/vqvae/fluids_100k/ \
    --num-samples 100000 \
    --feature-config configs/features/fluids.yaml \
    --device cuda \
    --batch-size 32 \
    --num-workers 8
```

**Feature Configuration:**

```yaml
# configs/features/fluids.yaml
features:
  domain: fluids

  # Raw state
  - type: raw_state
    channels: [u, v]  # Vector field

  # Derived quantities
  - type: vorticity
    method: finite_difference

  - type: kinetic_energy
    per_pixel: true

  - type: enstrophy
    per_pixel: true

  # Spatial derivatives
  - type: spatial_gradient
    field: u
    order: 1

  - type: spatial_gradient
    field: v
    order: 1

  # Temporal derivatives (from MNO predictions)
  - type: temporal_derivative
    field: u
    order: 1

  - type: temporal_derivative
    field: v
    order: 1

  # Statistical features
  - type: spatial_statistics
    fields: [u, v, vorticity]
    statistics: [mean, std, skewness, kurtosis]

  # Spectral features
  - type: fourier_spectrum
    fields: [u, v]
    num_modes: 32

sampling:
  # Sample diverse (θ, u₀) combinations
  num_samples: 100000
  operators_per_sample: uniform  # Sample operators uniformly
  ic_per_operator: 5  # Generate 5 ICs per operator
  trajectory_length: 64  # Shorter than CNO (256) for diversity

normalization:
  # Per-feature normalization
  method: standardize  # Zero mean, unit variance
  per_feature: true
```

**Guidelines:**
- Extract 100K+ samples for diversity
- Use domain-appropriate physical quantities
- Normalize features consistently
- Validate feature distributions (no NaN, reasonable ranges)

---

## VQ-VAE Training

### Domain-Specific VQ-VAE Configuration

```yaml
# configs/vqvae/fluids_100k.yaml
model:
  architecture: vqvae
  domain: fluids  # NEW: track domain

  encoder:
    type: conv_encoder
    input_channels: 20  # Total feature dimensionality
    hidden_channels: [64, 128, 256, 512]
    latent_dim: 256
    activation: gelu

  quantizer:
    type: vector_quantizer
    num_embeddings: 512  # Codebook size
    embedding_dim: 256
    commitment_cost: 0.25

    # Orthogonality-weighted clustering (critical!)
    use_orthogonality_loss: true
    orthogonality_weight: 0.1

  decoder:
    type: conv_decoder
    latent_dim: 256
    hidden_channels: [512, 256, 128, 64]
    output_channels: 20
    activation: gelu

training:
  loss:
    reconstruction_weight: 1.0
    vq_weight: 1.0
    orthogonality_weight: 0.1  # Encourages diverse categories

  optimizer:
    type: adamw
    lr: 1e-4
    weight_decay: 1e-5

  scheduler:
    type: cosine_annealing
    T_max: 50
    eta_min: 1e-6

  batch_size: 128
  num_epochs: 50
  gradient_clip: 1.0

dataset:
  path: data/vqvae/fluids_100k/
  train_split: 0.9
  val_split: 0.1
  shuffle: true
  num_workers: 8

evaluation:
  metrics:
    - reconstruction_loss
    - codebook_utilization
    - perplexity
  compute_frequency: 1
```

### Training VQ-VAE

```bash
# Train VQ-VAE on domain features
spinlock train-vqvae \
    --config configs/vqvae/fluids_100k.yaml \
    --output checkpoints/vqvae/fluids/ \
    --experiment fluids_vqvae_v1
```

**Target Metrics:**
- Reconstruction loss (L_recon) < 0.05
- Codebook utilization > 40% (>200 out of 512 codes used)
- Perplexity > 50 (indicates diversity)

**Interpretation:**
- Monitor which codes get used (category discovery)
- Visualize what each code represents (sample and plot)
- Check for mode collapse (some codes heavily used, others never)

### Category Discovery

```python
# Analyze discovered categories
def analyze_categories(vqvae_checkpoint, feature_dataset):
    """
    Analyze what each VQ-VAE category represents.
    """
    vqvae = load_vqvae(vqvae_checkpoint)
    dataset = load_feature_dataset(feature_dataset)

    # Tokenize all features
    tokens = []
    features = []
    for batch in dataloader(dataset):
        token = vqvae.encode(batch)  # [B]
        tokens.append(token)
        features.append(batch)

    tokens = torch.cat(tokens)
    features = torch.cat(features)

    # For each category, sample representative trajectories
    for category_idx in range(vqvae.num_embeddings):
        # Find samples assigned to this category
        mask = tokens == category_idx
        if mask.sum() == 0:
            print(f"Category {category_idx}: UNUSED")
            continue

        # Sample 10 representative trajectories
        category_features = features[mask]
        sample_indices = np.random.choice(len(category_features), size=10)
        samples = category_features[sample_indices]

        # Visualize
        plot_category_samples(
            samples, category_idx,
            save_path=f"analysis/fluids/category_{category_idx}.pdf"
        )

        # Compute statistics
        stats = {
            "count": mask.sum().item(),
            "percentage": (mask.sum() / len(tokens) * 100).item(),
            "mean_features": category_features.mean(dim=0),
            "std_features": category_features.std(dim=0),
        }

        print(f"Category {category_idx}: {stats['count']} samples ({stats['percentage']:.1f}%)")

    # Manual labeling
    # Researcher looks at visualizations and assigns semantic labels
    category_labels = {
        0: "laminar_flow",
        1: "vortex_shedding",
        2: "turbulent_cascade",
        3: "steady_state",
        4: "transitional",
        # ... etc
    }

    save_json(category_labels, "analysis/fluids/category_labels.json")
```

---

## Cross-Domain Testing

### Vocabulary Alignment Analysis

Once ≥2 domains are trained, test alignment:

```bash
# Run full cross-domain analysis
spinlock analyze-vocabulary-alignment \
    --domain1 reaction_diffusion \
    --domain2 fluid_dynamics \
    --vqvae1 checkpoints/vqvae/rd/best.pt \
    --vqvae2 checkpoints/vqvae/fluids/best.pt \
    --dataset1 data/vqvae/rd_100k/ \
    --dataset2 data/vqvae/fluids_100k/ \
    --output results/alignment/rd_fluids/ \
    --num-samples 10000
```

**This runs:**
1. Category count correspondence
2. Codebook embedding correlation
3. Semantic correspondence analysis (requires manual labeling)
4. Transfer learning test
5. Sequence structure analysis

**Output:**
- Quantitative metrics (correlation, transfer accuracy)
- Visualizations (correlation matrix, category samples)
- Statistical significance tests
- Summary report with interpretation

### Adding to Multi-Domain NOA

If alignment is strong, integrate into unified NOA:

```yaml
# configs/noa/multi_domain_phase2.yaml
model:
  architecture: noa
  type: neural_operator_agent

  domains:
    - name: reaction_diffusion
      vqvae_checkpoint: checkpoints/vqvae/rd/best.pt
      mno_checkpoint: checkpoints/noa/rd/best.pt
      num_categories: 10

    - name: fluid_dynamics
      vqvae_checkpoint: checkpoints/vqvae/fluids/best.pt
      mno_checkpoint: checkpoints/noa/fluids/best.pt
      num_categories: 10

    - name: waves  # Future domain
      vqvae_checkpoint: checkpoints/vqvae/waves/best.pt
      mno_checkpoint: checkpoints/noa/waves/best.pt
      num_categories: 10

  # Unified token embedding
  token_embedding:
    type: learned_alignment  # If categories align
    embedding_dim: 512
    align_codebooks: true  # Use vocabulary alignment

    # Or separate embeddings if no alignment
    # type: per_domain
    # embedding_dim: 512

  # Cross-domain attention
  transformer:
    num_layers: 12
    num_heads: 8
    hidden_dim: 512
    mlp_ratio: 4.0
    dropout: 0.1

  # Working memory (Phase 2)
  working_memory:
    capacity: 256  # Token sequence length
    num_slots: 16  # Attention slots

  # Symbolic reasoning
  reasoning:
    type: cross_domain  # Operates over all vocabularies
    attention_type: cross_attention  # Attend across domains
```

---

## Integration Checklist

### Before Starting

- [ ] Domain selected based on criteria (mathematical/physical diversity)
- [ ] Parameter space well-understood (known behavioral regimes)
- [ ] Numerical solver validated (stable, accurate, physically consistent)
- [ ] Computational resources allocated (GPU for MNO training)

### Dataset Generation

- [ ] CNO dataset generated (5K-10K operators, 256 timesteps)
- [ ] Parameter space well-sampled (Sobol or Latin hypercube)
- [ ] Multiple ICs per operator (5+ for diversity)
- [ ] Dataset validated (no NaN/Inf, diverse behaviors, no artifacts)
- [ ] Visualizations of sample trajectories inspected

### MNO Training (Stage 1)

- [ ] Architecture chosen appropriate for domain
- [ ] Configuration file created (based on template)
- [ ] Training launched with monitoring
- [ ] Target metric achieved (L_traj < 1.0)
- [ ] Model checkpointed and validated

### Feature Extraction (Stage 2)

- [ ] Domain-appropriate features defined
- [ ] Feature configuration file created
- [ ] 100K+ samples generated from MNO
- [ ] Features validated (distributions, no NaN, reasonable ranges)
- [ ] Feature dataset saved in compatible format

### VQ-VAE Training (Stage 3)

- [ ] VQ-VAE configuration created
- [ ] Training launched with orthogonality weighting
- [ ] Target metrics achieved (L_recon < 0.05, utilization > 40%)
- [ ] Categories analyzed and visualized
- [ ] Semantic labels assigned by domain expert

### Cross-Domain Analysis

- [ ] Vocabulary alignment analysis run
- [ ] Codebook correlation computed (with significance test)
- [ ] Semantic correspondence analyzed
- [ ] Transfer learning tested
- [ ] Results interpreted and documented
- [ ] Summary report generated

### Publication

- [ ] Results written up (regardless of alignment outcome)
- [ ] Figures and tables created (publication-ready)
- [ ] Code and data archived (reproducibility)
- [ ] Preprint posted (if alignment strong, major result)

---

## Troubleshooting

### Low MNO Accuracy (L_traj > 1.0)

**Possible Causes:**
- Insufficient model capacity → Increase parameters
- Poor dataset quality → Validate CNO solver
- Inappropriate architecture → Try domain-specific modifications
- Training instability → Reduce learning rate, add gradient clipping

### Low VQ-VAE Utilization (<20%)

**Possible Causes:**
- Mode collapse → Increase orthogonality weight
- Insufficient diversity → Generate more features
- Too many embedding codes → Reduce codebook size
- Training not converged → Train longer

### No Vocabulary Alignment (Correlation <0.3)

**Interpretation:**
- This is a valid scientific result!
- Domains have distinct behavioral geometries
- Proceed with domain-specific NOAs
- Document what makes domains different

### Partial Alignment (Correlation 0.5-0.7)

**Next Steps:**
- Identify which categories align and which don't
- Test partial transfer learning
- Build hybrid NOA architecture (shared + domain-specific)
- Refine universality hypothesis

---

## Conclusion

Domain integration follows a systematic pipeline:
1. **Select** complementary domain based on mathematical/physical diversity
2. **Generate** CNO dataset with diverse operators and behaviors
3. **Train** MNO with domain-appropriate architecture (Stage 1)
4. **Extract** domain-relevant features from MNO (Stage 2)
5. **Train** VQ-VAE to discover behavioral categories (Stage 3)
6. **Analyze** vocabulary alignment across domains
7. **Integrate** into multi-domain NOA if alignment is strong
8. **Publish** results regardless of outcome

**Key Principle:** Each domain is optimized independently for best performance. Cross-domain universality is discovered, not imposed.

The architecture is designed to test the computational universals hypothesis rigorously. Both positive and negative results advance our understanding of physics.
