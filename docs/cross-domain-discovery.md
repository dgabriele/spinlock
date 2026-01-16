# Cross-Domain Discovery: Methodology and Analysis

## Overview

This document provides a rigorous methodology for testing the computational universals hypothesis through cross-domain vocabulary alignment. It defines what we mean by "computational universals," establishes quantitative metrics for measuring alignment, and provides a publication-ready analysis framework.

**Target Audience:** Researchers implementing multi-domain experiments, analyzing results, and preparing publications.

---

## What Are Computational Universals?

### Definition

**Computational Universal:** A behavioral pattern or structural feature that emerges across multiple physics domains despite differences in governing equations, state spaces, and parameters.

### Characteristics

**1. Substrate Independence**
- Pattern exists independent of specific physical quantities (concentration vs velocity vs pressure)
- Emerges from mathematics of spatiotemporal evolution, not particular equations
- Recognizable at behavioral/qualitative level even when quantitative details differ

**2. Spontaneous Emergence**
- Not imposed by shared architecture or training procedure
- Discovered independently in each domain
- Alignment is measured outcome, not design assumption

**3. Semantic Equivalence**
- Patterns have equivalent *meaning* across domains
- "Oscillatory" in reaction-diffusion corresponds to "periodic vortex shedding" in fluids
- Token sequences describe analogous behavioral trajectories

**4. Symbolic Transferability**
- Knowledge learned in one domain applies to another at token/category level
- NOA trained on Domain A recognizes patterns in Domain B
- Transfer succeeds even when trajectory-level prediction fails

### Examples of Potential Universals

**Oscillatory Patterns:**
- Reaction-diffusion: Activator-inhibitor limit cycles (Hopf bifurcations)
- Fluid dynamics: Vortex shedding (von Kármán street)
- Wave equations: Standing wave oscillations
- Quantum: Energy level transitions

**Damping/Dissipation:**
- Reaction-diffusion: Inhibitor diffusion stabilizing activator
- Fluid dynamics: Viscous dissipation of turbulent energy
- Wave equations: Amplitude decay due to medium absorption
- Quantum: Decoherence and relaxation

**Spreading/Diffusion:**
- Reaction-diffusion: Concentration gradients diffusing
- Fluid dynamics: Momentum diffusion, mixing
- Wave equations: Wave packet dispersion
- Quantum: Wavefunction spreading

**Symmetry Breaking:**
- Reaction-diffusion: Turing patterns from homogeneous state
- Fluid dynamics: Transition to turbulence
- Wave equations: Mode competition and selection
- Quantum: Spontaneous symmetry breaking

**Stationary/Equilibrium:**
- Reaction-diffusion: Fixed points, stable patterns
- Fluid dynamics: Laminar flow, steady circulation
- Wave equations: DC component, time-independent fields
- Quantum: Ground states, stationary states

### Non-Examples (Domain-Specific Features)

**Reaction-Diffusion Specific:**
- Activator-inhibitor coupling mechanism
- Specific reaction kinetics (FitzHugh-Nagumo, Gray-Scott)
- Chemical concentration magnitudes

**Fluid Dynamics Specific:**
- Incompressibility constraint (∇·u = 0)
- Reynolds number regimes
- Vorticity dynamics specific to 2D/3D

**Wave Equation Specific:**
- Group vs phase velocity distinction
- Dispersion relations
- Interference constructive/destructive patterns

**The Question:** Do the categories discovered by VQ-VAE capture universals or domain-specifics?

---

## Testing Framework

### Experimental Design

**Phase 1: Independent Domain Training**

For each physics domain D:
1. Generate CNO dataset of operators in domain D
2. Train MNO_D on domain D (pure MSE, Stage 1)
3. Generate 100K+ diverse features from MNO_D (Stage 2)
4. Train VQ-VAE_D on MNO_D distribution (Stage 3)
5. Extract codebook embeddings and category assignments

**Phase 2: Vocabulary Alignment Analysis**

For each domain pair (D1, D2):
1. Compute quantitative alignment metrics
2. Perform semantic correspondence analysis
3. Test transfer learning capability
4. Interpret results statistically

**Phase 3: Multi-Domain Integration**

If alignment is strong:
1. Train unified NOA over both vocabularies
2. Test cross-domain reasoning tasks
3. Measure emergent transfer capabilities

### Null Hypothesis

**H₀:** Independently trained VQ-VAEs discover domain-specific categories with no systematic alignment.

**Expected under H₀:**
- Codebook embedding correlation ≈ 0 (random)
- No semantic correspondence between categories
- Transfer learning accuracy ≈ random chance (1/K for K categories)
- Different numbers of discovered categories per domain

### Alternative Hypothesis

**H₁:** Independently trained VQ-VAEs discover aligned categories representing computational universals.

**Expected under H₁:**
- Codebook embedding correlation > 0.7
- Clear semantic correspondence (manual interpretation agrees)
- Transfer learning accuracy > 80%
- Similar numbers of categories per domain (~10)

---

## Quantitative Metrics

### 1. Category Count Correspondence

**Metric:** Compare number of utilized categories discovered by each VQ-VAE.

```python
def category_count_correspondence(vqvae_1, vqvae_2):
    """
    Compute whether domains discover similar numbers of categories.

    Returns:
        count_1: Number of utilized categories in domain 1
        count_2: Number of utilized categories in domain 2
        ratio: min(count_1, count_2) / max(count_1, count_2)
    """
    count_1 = (vqvae_1.category_usage > threshold).sum()
    count_2 = (vqvae_2.category_usage > threshold).sum()
    ratio = min(count_1, count_2) / max(count_1, count_2)

    return {
        "count_1": count_1,
        "count_2": count_2,
        "ratio": ratio,
        "interpretation": "similar" if ratio > 0.8 else "different"
    }
```

**Interpretation:**
- Ratio > 0.8: Domains have similar behavioral dimensionality
- Ratio < 0.5: Domains have fundamentally different complexity

**Significance:**
If computational universals exist, behavioral space should have similar dimensionality across domains (both discover ~10 categories, not 5 vs 50).

### 2. Codebook Embedding Correlation

**Metric:** Cosine similarity between codebook embeddings across domains.

```python
def codebook_correlation(vqvae_1, vqvae_2):
    """
    Compute optimal alignment between codebook embeddings.

    Uses Hungarian algorithm to find best category pairing,
    then computes correlation for aligned pairs.

    Returns:
        correlation_matrix: K1 x K2 pairwise cosine similarities
        optimal_pairing: List of (idx_1, idx_2) tuples for aligned categories
        mean_correlation: Average cosine similarity for aligned pairs
    """
    # Get codebook embeddings
    codebook_1 = vqvae_1.codebook.weight.detach()  # [K1, D]
    codebook_2 = vqvae_2.codebook.weight.detach()  # [K2, D]

    # Normalize to unit length
    codebook_1 = F.normalize(codebook_1, dim=-1)
    codebook_2 = F.normalize(codebook_2, dim=-1)

    # Compute all pairwise cosine similarities
    correlation_matrix = torch.mm(codebook_1, codebook_2.t())  # [K1, K2]

    # Find optimal pairing using Hungarian algorithm
    # (maximize sum of correlations)
    cost_matrix = -correlation_matrix.cpu().numpy()
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    optimal_pairing = list(zip(row_indices, col_indices))
    paired_correlations = correlation_matrix[row_indices, col_indices]
    mean_correlation = paired_correlations.mean().item()

    return {
        "correlation_matrix": correlation_matrix,
        "optimal_pairing": optimal_pairing,
        "mean_correlation": mean_correlation,
        "paired_correlations": paired_correlations,
        "interpretation": interpret_correlation(mean_correlation)
    }

def interpret_correlation(corr):
    """Interpret correlation strength."""
    if corr > 0.7:
        return "strong alignment - computational universals likely"
    elif corr > 0.5:
        return "moderate alignment - partial universality"
    elif corr > 0.3:
        return "weak alignment - some shared structure"
    else:
        return "no alignment - domain-specific categories"
```

**Interpretation:**
- Correlation > 0.7: Strong geometric alignment, categories correspond
- Correlation 0.5-0.7: Moderate alignment, partial correspondence
- Correlation 0.3-0.5: Weak alignment, limited shared structure
- Correlation < 0.3: No alignment, domain boundaries confirmed

**Statistical Significance:**
Compare to null distribution (random codebook correlations). Use permutation test:
```python
def significance_test(correlation, vqvae_1, vqvae_2, n_permutations=1000):
    """Test if correlation is significantly above random."""
    null_correlations = []
    for _ in range(n_permutations):
        # Permute codebook 2 randomly
        perm_indices = torch.randperm(vqvae_2.num_categories)
        perm_codebook_2 = vqvae_2.codebook.weight[perm_indices]

        # Compute correlation with permuted codebook
        null_corr = compute_correlation(vqvae_1.codebook.weight, perm_codebook_2)
        null_correlations.append(null_corr)

    p_value = (np.array(null_correlations) >= correlation).mean()
    return p_value
```

### 3. Semantic Correspondence Analysis

**Metric:** Manual interpretation agreement on category meanings.

```python
def semantic_correspondence(vqvae_1, vqvae_2, domain_1, domain_2):
    """
    Analyze semantic correspondence through visualization and manual interpretation.

    For each category in domain 1:
    1. Sample trajectories assigned to that category
    2. Visualize spatiotemporal patterns
    3. Manually assign semantic label (oscillatory, damping, spreading, etc.)

    Repeat for domain 2.

    Compare semantic labels for optimally paired categories.
    """
    # Step 1: Visualize and label domain 1 categories
    domain_1_labels = {}
    for cat_idx in range(vqvae_1.num_categories):
        samples = sample_category_trajectories(vqvae_1, domain_1, cat_idx, n=10)
        visualizations = plot_spatiotemporal_patterns(samples)

        # Manual labeling by domain expert
        semantic_label = interpret_pattern(visualizations, domain_1)
        domain_1_labels[cat_idx] = semantic_label

    # Step 2: Visualize and label domain 2 categories
    domain_2_labels = {}
    for cat_idx in range(vqvae_2.num_categories):
        samples = sample_category_trajectories(vqvae_2, domain_2, cat_idx, n=10)
        visualizations = plot_spatiotemporal_patterns(samples)

        semantic_label = interpret_pattern(visualizations, domain_2)
        domain_2_labels[cat_idx] = semantic_label

    # Step 3: Check correspondence for optimal pairing
    optimal_pairing = codebook_correlation(vqvae_1, vqvae_2)["optimal_pairing"]

    correspondences = []
    for idx_1, idx_2 in optimal_pairing:
        label_1 = domain_1_labels[idx_1]
        label_2 = domain_2_labels[idx_2]

        # Check if semantic labels match
        match = labels_correspond(label_1, label_2)
        correspondences.append({
            "category_1": idx_1,
            "category_2": idx_2,
            "label_1": label_1,
            "label_2": label_2,
            "match": match
        })

    agreement_rate = sum(c["match"] for c in correspondences) / len(correspondences)

    return {
        "domain_1_labels": domain_1_labels,
        "domain_2_labels": domain_2_labels,
        "correspondences": correspondences,
        "agreement_rate": agreement_rate
    }

def labels_correspond(label_1, label_2):
    """
    Determine if two semantic labels represent equivalent patterns.

    Examples:
        "oscillatory" (RD) ↔ "periodic vortex shedding" (fluids) → True
        "damping" (RD) ↔ "viscous dissipation" (fluids) → True
        "spreading" (RD) ↔ "laminar flow" (fluids) → False
    """
    # Define equivalence classes
    equivalences = {
        "oscillatory": ["oscillatory", "periodic", "vortex shedding", "limit cycle"],
        "damping": ["damping", "dissipation", "decay", "stabilization"],
        "spreading": ["spreading", "diffusion", "mixing", "dispersal"],
        "stationary": ["stationary", "equilibrium", "fixed point", "steady state"],
        "growth": ["growth", "amplification", "instability", "expansion"],
        # ... etc
    }

    for category, synonyms in equivalences.items():
        if label_1 in synonyms and label_2 in synonyms:
            return True
    return False
```

**Interpretation:**
- Agreement > 80%: Strong semantic correspondence, universals likely
- Agreement 50-80%: Moderate correspondence, partial universality
- Agreement < 50%: Weak correspondence, categories domain-specific

### 4. Transfer Learning Accuracy

**Metric:** Test if NOA trained on Domain A recognizes patterns in Domain B.

```python
def transfer_learning_test(vqvae_1, vqvae_2, domain_1_data, domain_2_data):
    """
    Test symbolic transfer learning capability.

    Train NOA classifier on domain 1 tokens, test on domain 2.
    """
    # Step 1: Generate token sequences for domain 1
    domain_1_tokens = []
    domain_1_labels = []
    for trajectory in domain_1_data:
        tokens = vqvae_1.tokenize(trajectory)  # [T]
        # Label is dominant category in trajectory
        label = tokens.mode().values.item()
        domain_1_tokens.append(tokens)
        domain_1_labels.append(label)

    # Step 2: Train simple NOA classifier on domain 1
    noa_classifier = TransformerClassifier(
        num_categories=vqvae_1.num_categories,
        num_classes=vqvae_1.num_categories
    )
    train_classifier(noa_classifier, domain_1_tokens, domain_1_labels)

    # Step 3: Generate token sequences for domain 2
    domain_2_tokens = []
    domain_2_labels = []
    for trajectory in domain_2_data:
        tokens = vqvae_2.tokenize(trajectory)
        label = tokens.mode().values.item()
        domain_2_tokens.append(tokens)
        domain_2_labels.append(label)

    # Step 4: Map domain 2 tokens through optimal pairing
    optimal_pairing = codebook_correlation(vqvae_1, vqvae_2)["optimal_pairing"]
    pairing_map = {idx_2: idx_1 for idx_1, idx_2 in optimal_pairing}

    domain_2_tokens_mapped = []
    for tokens in domain_2_tokens:
        mapped = torch.tensor([pairing_map[t.item()] for t in tokens])
        domain_2_tokens_mapped.append(mapped)

    # Step 5: Evaluate classifier on mapped domain 2 tokens
    predictions = []
    for tokens in domain_2_tokens_mapped:
        pred = noa_classifier(tokens).argmax()
        predictions.append(pred.item())

    # Step 6: Compute accuracy
    # Map predicted domain 1 labels back to domain 2 labels
    reverse_map = {idx_1: idx_2 for idx_1, idx_2 in optimal_pairing}
    mapped_predictions = [reverse_map[p] for p in predictions]

    accuracy = (np.array(mapped_predictions) == np.array(domain_2_labels)).mean()

    # Compare to random baseline
    random_accuracy = 1.0 / vqvae_2.num_categories

    return {
        "transfer_accuracy": accuracy,
        "random_baseline": random_accuracy,
        "improvement": accuracy - random_baseline,
        "interpretation": interpret_transfer_accuracy(accuracy, random_accuracy)
    }

def interpret_transfer_accuracy(accuracy, baseline):
    """Interpret transfer learning results."""
    if accuracy > 0.8:
        return "strong transfer - computational universals confirmed"
    elif accuracy > 0.6:
        return "moderate transfer - partial universality"
    elif accuracy > baseline + 0.2:
        return "weak transfer - some shared structure"
    else:
        return "no transfer - domain boundaries confirmed"
```

**Interpretation:**
- Accuracy > 80%: Strong transfer, symbolic knowledge generalizes
- Accuracy 60-80%: Moderate transfer, partial generalization
- Accuracy > baseline + 20%: Weak transfer, some structure shared
- Accuracy ≈ baseline: No transfer, domain-specific categories

### 5. Token Sequence Structure Analysis

**Metric:** Compare compositional structure of token sequences.

```python
def sequence_structure_analysis(vqvae_1, vqvae_2, domain_1_data, domain_2_data):
    """
    Analyze if token sequences have similar compositional structure.

    Metrics:
    - Transition matrix similarity
    - N-gram distribution overlap
    - Sequence motif correspondence
    """
    # Compute transition matrices
    T1 = compute_transition_matrix(vqvae_1, domain_1_data)  # [K1, K1]
    T2 = compute_transition_matrix(vqvae_2, domain_2_data)  # [K2, K2]

    # Align transition matrices using optimal pairing
    optimal_pairing = codebook_correlation(vqvae_1, vqvae_2)["optimal_pairing"]
    T2_aligned = reorder_matrix(T2, optimal_pairing)

    # Compute transition matrix correlation
    transition_correlation = np.corrcoef(T1.flatten(), T2_aligned.flatten())[0, 1]

    # Compute n-gram overlaps
    bigrams_1 = extract_ngrams(vqvae_1, domain_1_data, n=2)
    bigrams_2 = extract_ngrams(vqvae_2, domain_2_data, n=2)
    bigram_overlap = compute_ngram_overlap(bigrams_1, bigrams_2, optimal_pairing)

    trigrams_1 = extract_ngrams(vqvae_1, domain_1_data, n=3)
    trigrams_2 = extract_ngrams(vqvae_2, domain_2_data, n=3)
    trigram_overlap = compute_ngram_overlap(trigrams_1, trigrams_2, optimal_pairing)

    return {
        "transition_correlation": transition_correlation,
        "bigram_overlap": bigram_overlap,
        "trigram_overlap": trigram_overlap,
        "interpretation": interpret_sequence_structure(
            transition_correlation, bigram_overlap, trigram_overlap
        )
    }
```

**Interpretation:**
High sequence structure similarity suggests:
- Not just individual categories align, but their temporal relationships
- Compositional patterns transfer (e.g., "oscillatory → damping" sequence)
- Token sequences form a "grammar of dynamics"

---

## Statistical Analysis Framework

### Multiple Hypothesis Testing

When testing alignment across multiple domain pairs, correct for multiple comparisons:

```python
def multi_domain_analysis(vqvaes, domains, alpha=0.05):
    """
    Test vocabulary alignment across multiple domain pairs.
    Apply Bonferroni correction for multiple comparisons.
    """
    domain_pairs = list(itertools.combinations(range(len(domains)), 2))
    n_tests = len(domain_pairs)
    corrected_alpha = alpha / n_tests

    results = []
    for i, j in domain_pairs:
        metrics = compute_all_metrics(vqvaes[i], vqvaes[j], domains[i], domains[j])

        # Test significance
        p_value = significance_test(
            metrics["codebook_correlation"]["mean_correlation"],
            vqvaes[i], vqvaes[j]
        )

        significant = p_value < corrected_alpha

        results.append({
            "domain_1": domains[i],
            "domain_2": domains[j],
            "metrics": metrics,
            "p_value": p_value,
            "significant": significant
        })

    return results
```

### Effect Size Reporting

Report not just statistical significance but effect size:

```python
def effect_size_report(correlation, transfer_accuracy, random_baseline):
    """
    Compute standardized effect sizes for publication.
    """
    # Cohen's d for transfer accuracy
    cohen_d = (transfer_accuracy - random_baseline) / np.std([transfer_accuracy, random_baseline])

    # Correlation is already an effect size measure

    return {
        "correlation": correlation,
        "correlation_interpretation": interpret_correlation_effect_size(correlation),
        "transfer_improvement": transfer_accuracy - random_baseline,
        "cohen_d": cohen_d,
        "cohen_d_interpretation": interpret_cohen_d(cohen_d)
    }
```

---

## Publication-Ready Analysis Pipeline

### Complete Workflow

```python
def full_cross_domain_analysis(
    domain_1_name: str,
    domain_2_name: str,
    vqvae_1_path: str,
    vqvae_2_path: str,
    domain_1_data_path: str,
    domain_2_data_path: str,
    output_dir: str
):
    """
    Complete cross-domain analysis pipeline for publication.

    Generates:
    - Quantitative metrics with statistical significance
    - Visualizations of codebook alignment
    - Semantic correspondence tables
    - Transfer learning results
    - Publication-ready figures and tables
    """
    # Load models and data
    vqvae_1 = load_vqvae(vqvae_1_path)
    vqvae_2 = load_vqvae(vqvae_2_path)
    domain_1_data = load_dataset(domain_1_data_path)
    domain_2_data = load_dataset(domain_2_data_path)

    # 1. Category Count Correspondence
    print("1. Analyzing category counts...")
    category_counts = category_count_correspondence(vqvae_1, vqvae_2)
    save_json(category_counts, f"{output_dir}/category_counts.json")

    # 2. Codebook Embedding Correlation
    print("2. Computing codebook correlation...")
    correlation_results = codebook_correlation(vqvae_1, vqvae_2)
    save_json(correlation_results, f"{output_dir}/codebook_correlation.json")

    # Visualize correlation matrix
    plot_correlation_matrix(
        correlation_results["correlation_matrix"],
        domain_1_name, domain_2_name,
        save_path=f"{output_dir}/correlation_matrix.pdf"
    )

    # 3. Statistical Significance
    print("3. Testing statistical significance...")
    p_value = significance_test(
        correlation_results["mean_correlation"],
        vqvae_1, vqvae_2,
        n_permutations=1000
    )

    # 4. Semantic Correspondence
    print("4. Analyzing semantic correspondence...")
    semantic_results = semantic_correspondence(
        vqvae_1, vqvae_2, domain_1_data, domain_2_data
    )
    save_json(semantic_results, f"{output_dir}/semantic_correspondence.json")

    # Generate correspondence table for publication
    create_correspondence_table(
        semantic_results,
        domain_1_name, domain_2_name,
        save_path=f"{output_dir}/correspondence_table.tex"
    )

    # 5. Transfer Learning
    print("5. Testing transfer learning...")
    transfer_results = transfer_learning_test(
        vqvae_1, vqvae_2, domain_1_data, domain_2_data
    )
    save_json(transfer_results, f"{output_dir}/transfer_learning.json")

    # 6. Sequence Structure
    print("6. Analyzing sequence structure...")
    sequence_results = sequence_structure_analysis(
        vqvae_1, vqvae_2, domain_1_data, domain_2_data
    )
    save_json(sequence_results, f"{output_dir}/sequence_structure.json")

    # 7. Generate Summary Report
    print("7. Generating summary report...")
    summary = generate_summary_report(
        category_counts, correlation_results, p_value,
        semantic_results, transfer_results, sequence_results
    )

    with open(f"{output_dir}/SUMMARY.md", "w") as f:
        f.write(summary)

    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    print(f"\nKey Finding: {summary['conclusion']}")

    return summary
```

### Example Output Structure

```
results/rd_fluids_alignment/
├── category_counts.json
├── codebook_correlation.json
├── correlation_matrix.pdf
├── semantic_correspondence.json
├── correspondence_table.tex
├── transfer_learning.json
├── sequence_structure.json
├── visualizations/
│   ├── rd_category_0.pdf
│   ├── rd_category_1.pdf
│   ├── ...
│   ├── fluids_category_0.pdf
│   ├── fluids_category_1.pdf
│   └── ...
└── SUMMARY.md
```

---

## Interpretation Guidelines

### Strong Evidence for Computational Universals

**Criteria:**
- Codebook correlation > 0.7 (p < 0.001)
- Semantic agreement > 80%
- Transfer accuracy > 80%
- Sequence structure correlation > 0.6

**Interpretation:**
"Independently trained VQ-VAEs on reaction-diffusion and fluid dynamics discover aligned behavioral categories, providing strong evidence for computational universals. Token vocabularies transfer semantic meaning across domains, suggesting substrate-independent patterns in spatiotemporal evolution."

**Publication Angle:**
- Discovery of cross-domain behavioral primitives
- Symbolic transfer as evidence for mathematical universality
- Implications for unified theory of dynamics

### Moderate Evidence (Partial Universality)

**Criteria:**
- Codebook correlation 0.5-0.7 (p < 0.01)
- Semantic agreement 50-80%
- Transfer accuracy 60-80%
- Some categories align strongly, others don't

**Interpretation:**
"Partial vocabulary alignment suggests computational universals exist at certain abstraction levels. Some behavioral categories (oscillatory, damping) transfer robustly, while others (domain-specific features) remain distinct."

**Publication Angle:**
- Identification of transferable vs domain-specific patterns
- Hierarchy of universality in spatiotemporal dynamics
- Boundary conditions for cross-domain generalization

### Weak/No Evidence (Domain Boundaries)

**Criteria:**
- Codebook correlation < 0.3 (p > 0.05)
- Semantic agreement < 50%
- Transfer accuracy ≈ random baseline
- No systematic category correspondence

**Interpretation:**
"Independently trained VQ-VAEs discover domain-specific behavioral categories with no systematic alignment. Reaction-diffusion and fluid dynamics have fundamentally distinct behavioral geometries, suggesting domain boundaries are real rather than artifactual."

**Publication Angle:**
- Identification of fundamental domain boundaries in physics
- Evidence against naive universalism
- Validation of domain-specific modeling approaches

---

## Conclusion

Cross-domain discovery testing provides a rigorous framework for investigating computational universals. By quantifying vocabulary alignment through multiple complementary metrics and comparing against null hypotheses, we can rigorously test whether behavioral categories represent substrate-independent patterns or domain-specific artifacts.

**Key Principles:**

1. **Quantitative + Qualitative:** Combine correlation metrics with semantic interpretation
2. **Statistical Rigor:** Test significance against null distributions
3. **Multiple Metrics:** Converging evidence from codebook correlation, transfer learning, sequence structure
4. **Publication-Ready:** Standardized analysis pipeline generates reproducible results
5. **Theory-Neutral:** Framework works regardless of whether universals exist

**The methodology is designed to discover truth, not confirm assumptions.**

Both alignment and non-alignment are valuable scientific outcomes that advance our understanding of computational structure in physics.
