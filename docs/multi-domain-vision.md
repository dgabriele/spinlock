# Multi-Domain MNO Architecture: Research Vision

## Overview

This document outlines the research vision for extending the Neural Operator Agent (NOA) architecture to multiple physics domains. The core hypothesis: **computational universals may exist across different physics families**, discoverable through symbolic/token-level transfer rather than trajectory-level prediction.

**Current Status:** Single domain (reaction-diffusion) complete. Multi-domain architecture is a research objective, not an implemented system.

---

## Research Motivation

### Why Multiple Domains?

Training a single MNO on reaction-diffusion operators demonstrates that learned physics engines can capture statistical regularities across thousands of operators. But this raises deeper questions:

**Are the behavioral patterns discovered domain-specific or universal?**

- Do the 10 categories discovered by VQ-VAE-RD (e.g., "oscillatory," "damping," "spreading") represent computational primitives that appear across ALL spatiotemporal dynamics?
- Or are they artifacts of reaction-diffusion's specific mathematical structure (parabolic PDEs, diffusion-reaction coupling)?

**Can symbolic knowledge transfer where trajectory predictions cannot?**

- Even when exact state predictions fail across domains, can behavioral categories align?
- If an MNO learns "oscillatory → damping transition" in reaction-diffusion, does that pattern appear in fluid vortex decay or wave interference?

### The Hypothesis

**Computational universals exist as substrate-independent patterns in the mathematics of spatiotemporal evolution.**

If true, this would mean:
- Certain behavioral categories emerge regardless of specific equations
- Token vocabularies from different physics domains align semantically
- Symbolic reasoning transfers across domains even when trajectory prediction doesn't
- Discovery happens at an abstract level above individual physics

If false, we learn:
- Different physics families have genuinely distinct behavioral geometries
- Domain boundaries are fundamental, not merely practical
- Each domain requires specialized treatment (which our architecture provides anyway)

**Either outcome advances scientific understanding.**

---

## Architecture Overview

### Core Design: Domain Independence + Shared Symbolic Layer

```
┌─────────────────────────────────────────────────────────────┐
│                     Physics Domains                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Reaction-Diffusion         Fluid Dynamics        Waves      │
│       Domain                    Domain            Domain     │
│         │                         │                 │        │
│         ▼                         ▼                 ▼        │
│   ┌──────────┐            ┌──────────┐      ┌──────────┐   │
│   │ MNO-RD   │            │MNO-Fluids│      │ MNO-Waves│   │
│   │ (U-AFNO) │            │ (U-AFNO) │      │ (variant)│   │
│   │  226M    │            │  ~200M   │      │  ~200M   │   │
│   └────┬─────┘            └────┬─────┘      └────┬─────┘   │
│        │                       │                  │         │
│        ▼                       ▼                  ▼         │
│   ┌──────────┐            ┌──────────┐      ┌──────────┐   │
│   │VQ-VAE-RD │            │VQ-Fluids │      │VQ-Waves  │   │
│   │10 tokens │            │~10 tokens│      │~10 tokens│   │
│   └────┬─────┘            └────┬─────┘      └────┬─────┘   │
│        │                       │                  │         │
│        └───────────┬───────────┴──────────────────┘         │
│                    ▼                                         │
│            ┌──────────────┐                                 │
│            │     NOA      │                                 │
│            │ Cross-Domain │                                 │
│            │  Reasoning   │                                 │
│            └──────┬───────┘                                 │
│                   │                                          │
│                   ▼                                          │
│         Computational Universals                            │
│            Discovery Layer                                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

**1. Domain Specialization**

Each physics family receives optimal treatment:
- **MNO Architecture**: Tailored to domain (U-AFNO for parabolic, variants for hyperbolic, complex-valued for quantum)
- **Training Distribution**: Domain-specific CNO datasets (RD operators, Navier-Stokes, wave equations)
- **Performance**: Each MNO achieves best possible accuracy for its domain (L_traj < 1.0)

**Why not a single universal MNO?**
- Different equation types benefit from different architectures
- Training on mixed distributions may compromise per-domain performance
- Domain-specific optimization is scientifically rigorous
- Enables clear hypothesis testing: Do independently optimized systems discover shared structure?

**2. Independent Tokenization**

Each MNO gets its own VQ-VAE trained exclusively on that domain's distribution:
- **VQ-VAE-RD**: Trained on 100K+ reaction-diffusion trajectories from MNO-RD
- **VQ-VAE-Fluids**: Trained on 100K+ fluid dynamics trajectories from MNO-Fluids
- **VQ-VAE-Waves**: Trained on 100K+ wave equation trajectories from MNO-Waves

**Why independent rather than shared tokenization?**
- Tests whether categories emerge naturally across domains (the key experiment)
- Avoids forcing alignment through shared architecture
- Allows each VQ-VAE to discover optimal categories for its distribution
- Makes vocabulary alignment a measurable outcome rather than architectural assumption

**3. Shared Symbolic Layer (NOA)**

Once domain-specific tokenizers exist, NOA operates over all vocabularies:
- Attention mechanisms process tokens from any domain
- Learns cross-domain correspondences if they exist
- Symbolic reasoning abstracts above trajectory-level details
- Working memory stores behavioral patterns as token sequences

**How NOA discovers universals:**
- If categories align, NOA recognizes equivalent patterns across domains
- Token sequences from Domain A map to semantic equivalents in Domain B
- Compositional structures transfer: "oscillatory + damping" in RD → "vortex decay" in fluids
- Symbolic transfer succeeds even when trajectory prediction fails

---

## Vocabulary Alignment: The Critical Experiment

### Hypothesis Testing Framework

**Prediction:** If computational universals exist, independently trained VQ-VAEs will discover aligned categories.

### Alignment Metrics

**1. Category Count Correspondence**
- Do all domains discover similar numbers of categories (~10)?
- Suggests consistent behavioral dimensionality across physics

**2. Codebook Embedding Correlation**
- Compute cosine similarity between codebook embeddings across domains
- High correlation (>0.7) suggests geometric alignment
- Low correlation (<0.3) suggests distinct behavioral spaces

**3. Semantic Correspondence**
- Manual interpretation: Do categories have equivalent meanings?
- "Oscillatory" in RD ↔ "Periodic vortex shedding" in fluids
- "Damping" in RD ↔ "Turbulent energy dissipation" in fluids
- "Spreading" in RD ↔ "Diffusive mixing" in fluids

**4. Transfer Learning Success**
- Train NOA on Domain A tokens
- Test classification accuracy on Domain B
- Success rate >80% indicates strong transfer
- Success rate <40% indicates domain boundaries

### Experimental Design

```python
# Phase 1: Independent Training
mno_rd = train_mno(domain="reaction_diffusion")
mno_fluids = train_mno(domain="fluid_dynamics")

vqvae_rd = train_vqvae(mno_distribution=mno_rd)
vqvae_fluids = train_vqvae(mno_distribution=mno_fluids)

# Phase 2: Vocabulary Alignment Analysis
alignment_metrics = {
    "category_counts": (vqvae_rd.num_categories, vqvae_fluids.num_categories),
    "embedding_correlation": cosine_similarity(
        vqvae_rd.codebook,
        vqvae_fluids.codebook
    ),
    "semantic_mapping": analyze_category_meanings(vqvae_rd, vqvae_fluids),
}

# Phase 3: Transfer Learning Test
noa = train_noa(tokens=vqvae_rd.tokenize(mno_rd.trajectories))
transfer_accuracy = evaluate_noa(
    noa,
    tokens=vqvae_fluids.tokenize(mno_fluids.trajectories)
)

# Phase 4: Interpretation
if alignment_metrics["embedding_correlation"] > 0.7 and transfer_accuracy > 0.8:
    conclusion = "Computational universals exist"
    publish_result = "Strong evidence for substrate-independent patterns"
elif alignment_metrics["embedding_correlation"] < 0.3 and transfer_accuracy < 0.4:
    conclusion = "Domain boundaries identified"
    publish_result = "Physics families have distinct behavioral geometries"
else:
    conclusion = "Partial alignment"
    publish_result = "Some categories transfer, others don't - investigate which"
```

### Interpretation of Outcomes

**Outcome A: Strong Alignment (correlation >0.7, transfer >80%)**

**Scientific Implications:**
- Computational universals DO exist across physics domains
- Behavioral categories represent substrate-independent patterns
- Symbolic reasoning captures abstract spatiotemporal structures
- Token vocabularies form a "language of dynamics"

**Practical Applications:**
- Single NOA can reason about multiple physics domains
- Transfer learning accelerates training on new domains
- Cross-domain analogical reasoning becomes possible
- Multi-agent systems communicate via universal symbolic protocol

**Next Steps:**
- Add third domain (waves) to test robustness
- Investigate which categories are most universal
- Develop theory of computational primitives
- Apply to scientific discovery (predict novel phenomena)

**Outcome B: Weak Alignment (correlation <0.3, transfer <40%)**

**Scientific Implications:**
- Different physics families have genuinely distinct behavioral geometries
- Parabolic PDEs (diffusion) ≠ Hyperbolic PDEs (waves) at fundamental level
- Domain boundaries are real, not just practical artifacts
- Specificity matters more than universality

**Practical Applications:**
- Domain-specific NOAs still valuable (we built them anyway)
- Transfer learning within-domain remains effective
- Multi-domain systems coordinate through explicit interfaces
- Each domain optimized independently for best performance

**Next Steps:**
- Focus on perfecting single-domain NOAs
- Develop domain-specific curiosity and planning
- Build ensemble systems with specialized agents
- Study what makes domains fundamentally different

**Outcome C: Partial Alignment (mixed metrics)**

**Scientific Implications:**
- Some computational patterns are universal, others domain-specific
- Universality exists at certain abstraction levels but not others
- Behavioral taxonomy has both general and specific components
- Physics exists on a spectrum of transferability

**Practical Applications:**
- Hybrid NOA architecture with shared + domain-specific components
- Selective transfer learning (only for aligned categories)
- Multi-level symbolic reasoning (universal + specialized)
- Adaptive models that learn domain boundaries

**Next Steps:**
- Identify which categories transfer and which don't
- Investigate why some patterns generalize
- Refine abstraction hierarchy
- Build theory of partial universality

---

## Transfer Learning at Symbolic Level

### Why Symbolic Rather Than Trajectory?

**Trajectory-Level Transfer Fails:**
- Exact state predictions require domain-specific physics
- Initial conditions, parameters, equations all different
- L_traj across domains would be meaningless
- No reason to expect state-space alignment

**Symbolic-Level Transfer Can Succeed:**
- Behavioral categories abstract above specific values
- "Oscillatory" describes patterns, not magnitudes
- Token sequences encode qualitative dynamics
- Compositional structure may be universal

### Example: Oscillatory Damping

**Reaction-Diffusion Domain:**
```
Initial State: Activator-inhibitor spots
MNO Rollout: Spots oscillate with decreasing amplitude
VQ-VAE Tokens: [7, 7, 7, 12, 12, 3, 3, 3, ...]
                │      │      │
           oscillatory │   stationary
                   damping
```

**Fluid Dynamics Domain:**
```
Initial State: Vortex pair
MNO Rollout: Vortices shed periodically, then dissipate
VQ-VAE Tokens: [9, 9, 9, 14, 14, 5, 5, 5, ...]
                │      │      │
           oscillatory │   stationary
                   damping
```

**Key Observation:**
Even though:
- Physical states are completely different (chemical concentrations vs velocity fields)
- Equations are different (reaction-diffusion vs Navier-Stokes)
- Parameter meanings differ (reaction rate vs Reynolds number)

The *behavioral pattern* may align:
- Token sequence structure similar: repeated → transition → constant
- Semantic interpretation equivalent: periodic → decay → equilibrium
- NOA trained on [7,7,12,3] might recognize [9,9,14,5] as same pattern

**If this happens, we've discovered computational universals.**

---

## Comparison with Alternative Approaches

### Why Not a Single Universal MNO?

**Hypothetical Universal Approach:**
- Train one MNO on mixed distribution (all physics combined)
- Single VQ-VAE tokenizes all domains
- Vocabulary is universal by construction

**Problems:**
1. **Compromised Performance**: Mixed training distribution may reduce per-domain accuracy
2. **Architectural Constraints**: Different equation types benefit from different architectures
3. **No Hypothesis Test**: Universality assumed, not discovered
4. **Scientific Rigor**: Doesn't test whether categories naturally align

**Our Multi-Domain Approach:**
- Each domain gets optimal treatment independently
- Universality is *discovered* through vocabulary alignment
- Clear experimental test with interpretable outcomes
- Both positive and negative results are scientifically valuable

### Why Not Trajectory-Level Transfer?

**Hypothetical Trajectory Approach:**
- Train MNO on Domain A
- Test prediction accuracy on Domain B trajectories
- High accuracy → universality exists

**Problems:**
1. **No Physical Basis**: Different domains have different state spaces, equations, scales
2. **Uninterpretable Failure**: Low accuracy expected, but doesn't disprove abstract universals
3. **No Abstraction**: Misses possibility of behavioral alignment without state alignment
4. **Impractical**: Can't use RD model to predict fluids trajectories anyway

**Our Symbolic Approach:**
- Abstraction layer (tokens) allows for domain differences
- Tests behavioral patterns, not exact states
- Success is interpretable: categories align semantically
- Failure is interpretable: domains have distinct geometries
- Practical: Even if universals exist, domain-specific MNOs still needed for accurate prediction

---

## Current Status and Roadmap

### Current Status (January 2026)

**Completed:**
- ✅ Reaction-diffusion domain fully implemented
  - MNO-RD trained to L_traj < 1.0 (Stage 1)
  - 100K+ diverse feature set generated (Stage 2)
  - VQ-VAE-RD trained with 10 categories, 47% utilization (Stage 3)
- ✅ Independent optimization architecture validated
- ✅ Stage 1-3 pipeline proven effective

**Current Capabilities:**
- Single-domain learned physics engine (reaction-diffusion)
- Behavioral tokenization within RD domain
- Foundation for NOA development (Phase 2 of roadmap)

**Not Yet Implemented:**
- Second physics domain (fluid dynamics)
- Cross-domain vocabulary alignment analysis
- Multi-domain NOA architecture
- Computational universals testing

### Near-Term Roadmap (Phase 1.5)

**Objective:** Add second domain to test multi-domain hypothesis

**Tasks:**
1. **Fluid Dynamics Dataset Generation**
   - 2D Navier-Stokes CNO operators
   - Reynolds number range: 10 → 1000 (laminar → turbulent)
   - 100K trajectories for diversity

2. **MNO-Fluids Training**
   - Architecture: U-AFNO (possibly modified for vector fields)
   - Pure MSE loss against CNO ground truth
   - Target: L_traj < 1.0

3. **VQ-VAE-Fluids Training**
   - Independent tokenization of MNO-Fluids distribution
   - Orthogonality-weighted clustering
   - Target: L_recon < 0.05, utilization >40%

4. **Vocabulary Alignment Analysis** ⭐ **CRITICAL EXPERIMENT**
   - Compare VQ-VAE-RD and VQ-VAE-Fluids codebooks
   - Compute alignment metrics
   - Test transfer learning
   - Interpret results → publish findings

5. **Cross-Domain NOA Architecture**
   - Unified attention over multiple token vocabularies
   - Embedding alignment (if categories correspond)
   - Symbolic reasoning across domains

**Success Criteria:**
- Vocabulary alignment hypothesis clearly supported or refuted
- Results publishable regardless of outcome
- Foundation laid for additional domains

### Long-Term Vision

**Phase 2-5 (NOA Development):**
- Working memory over multi-domain token sequences
- Curiosity-driven exploration across domains
- Self-modeling of domain-specific vs universal knowledge
- Discovery of cross-domain computational laws

**Additional Domains:**
- Wave equations (hyperbolic PDEs)
- Quantum dynamics (complex-valued fields)
- Solid mechanics (stress-strain, fracture)
- Climate modeling (multi-scale, coupled systems)

**Scientific Outcomes:**
- Map the landscape of computational universality
- Identify domain boundaries and transferable patterns
- Develop theory of spatiotemporal behavioral primitives
- Enable analogical reasoning across physics families

---

## Why This Matters

### Scientific Significance

**If computational universals exist:**
- We've discovered substrate-independent patterns in the mathematics of dynamics
- Certain behavioral structures emerge across ALL spatiotemporal evolution
- Symbolic reasoning captures something fundamental about physical law
- "Discovered physics" rather than "derived physics"

**If domains remain distinct:**
- We've identified fundamental boundaries in physics
- Different equation families have genuinely different behavioral geometries
- Specificity matters more than universality at the symbolic level
- Our domain-specific MNOs are the right architectural choice

**Either way:**
- We advance understanding of computational structure in physics
- We build high-performance learned physics engines
- We enable NOA development for autonomous scientific reasoning
- We create modular, extensible architecture for future domains

### Philosophical Implications

**The Nature of Discovery:**
- Traditional physics: Derive laws from first principles → Predict phenomena
- Learned physics: Observe statistical regularities → Discover patterns
- If patterns transfer across domains, they represent something "real" independent of human-derived equations

**Computational Universality:**
- Are there "atoms of dynamics" that appear everywhere?
- Does mathematics itself impose certain behavioral structures?
- Can we learn laws of physics that transcend specific equations?

**Alien Science:**
- If MNO discovers patterns we didn't design or expect, it's found something genuinely new
- If token vocabularies align across domains without our guidance, the alignment is "discovered"
- This is knowledge creation, not just knowledge representation

---

## Conclusion

The multi-domain MNO architecture represents a systematic approach to testing whether computational universals exist in physics. By training specialized MNOs independently on different physics families and analyzing vocabulary alignment, we can rigorously test fundamental hypotheses about the nature of spatiotemporal dynamics.

**The vision is both scientifically rigorous and practically valuable:**
- Each domain receives optimal treatment regardless of cross-domain results
- Vocabulary alignment is a clear, measurable experimental outcome
- Both positive and negative results advance scientific understanding
- The architecture scales to additional domains incrementally

**Current status: Research objective, not completed system.**

**Next critical milestone: Second domain integration (Phase 1.5) → Vocabulary alignment experiment.**

If categories align, we publish evidence for computational universals. If they don't, we publish evidence for domain boundaries. Either outcome is a contribution to science.

The architecture is designed to discover truth, not confirm assumptions.
