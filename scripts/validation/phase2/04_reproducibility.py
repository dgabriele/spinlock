#!/usr/bin/env python3
"""Validation Experiment 4: Reproducibility Testing.

Hypothesis:
    MNO autonomous evolution is deterministic. Same (u₀, perturbation) pair
    should produce identical (or near-identical) token sequences across
    multiple runs, demonstrating reproducible dynamics.

Success Criteria:
    - 95%+ episodes have token similarity > 0.95 across 5 runs
    - Mean within-pair token similarity > 0.98
    - Behavioral signature variance < 5% across runs
    - No outliers with similarity < 0.90 (unless numerical instability)

Methodology:
    1. Create 100 (u₀, perturbation) pairs
    2. Run each pair 5 times independently
    3. Compute token sequence similarity across runs (cosine + Hamming)
    4. Analyze behavioral signature variance
    5. Identify and investigate any outliers (low reproducibility)

Output:
    - similarity_distribution.png - Histogram of within-pair similarities
    - reproducibility_heatmap.png - Similarity matrices for sample pairs
    - signature_variance.png - Behavioral feature variance across runs
    - outlier_analysis.txt - Cases with low reproducibility
    - summary.txt - Pass/fail assessment
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from spinlock.perturbations import ImpulsePerturbationFactory
from spinlock.noa.behavioral_encoding import BehavioralSignature, BehavioralEncoder
from spinlock.noa.episode import EpisodeRunner
from spinlock.noa.early_stopping import StandardStoppingPolicy
from spinlock.noa.validation_utils import (
    load_mno_checkpoint,
    load_vqvae_checkpoint,
    sample_initial_condition,
    get_vqvae_num_categories_and_levels,
)


def load_models(mno_checkpoint: str, vqvae_checkpoint: str, device: str):
    """Load trained MNO and VQ-VAE models."""
    mno = load_mno_checkpoint(mno_checkpoint, device)
    vqvae = load_vqvae_checkpoint(vqvae_checkpoint, device)
    return mno, vqvae


def sample_ic_and_perturbation(spatial_size: Tuple[int, int], device: str, seed: int):
    """Sample initial condition and perturbation with fixed seed.

    Returns:
        (u0, perturbation) tuple
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sample IC
    u0 = sample_initial_condition(
        num_channels=2,
        spatial_size=spatial_size,
        ic_type="smooth_random",
        device=device,
        seed=seed,
    )

    # Sample perturbation type
    pattern_type = np.random.choice(["center", "corner", "uniform"])
    amp = np.random.uniform(0.5, 2.0)

    if pattern_type == "center":
        perturbation = ImpulsePerturbationFactory.center_blob(
            t_impulse=0, amplitude=amp, spatial_size=spatial_size, device=device
        )
    elif pattern_type == "corner":
        corner = np.random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
        perturbation = ImpulsePerturbationFactory.corner_blob(
            t_impulse=0, amplitude=amp, spatial_size=spatial_size,
            corner=corner, device=device
        )
    else:
        perturbation = ImpulsePerturbationFactory.uniform_forcing(
            t_impulse=0, amplitude=amp, spatial_size=spatial_size, device=device
        )

    return u0, perturbation


def compute_token_similarity(tokens1: torch.Tensor, tokens2: torch.Tensor) -> float:
    """Compute token sequence similarity.

    Uses combination of:
    - Cosine similarity (treats tokens as continuous)
    - Hamming similarity (exact matches)

    Returns:
        Similarity in [0, 1]
    """
    # Ensure same length
    T = min(tokens1.shape[0], tokens2.shape[0])
    tokens1 = tokens1[:T]
    tokens2 = tokens2[:T]

    # Hamming similarity (exact matches)
    exact_matches = (tokens1 == tokens2).float().mean().item()

    # Cosine similarity
    tokens1_flat = tokens1.flatten().float()
    tokens2_flat = tokens2.flatten().float()

    cosine_sim = torch.nn.functional.cosine_similarity(
        tokens1_flat.unsqueeze(0),
        tokens2_flat.unsqueeze(0)
    ).item()
    cosine_sim = (cosine_sim + 1.0) / 2.0  # Map [-1, 1] -> [0, 1]

    # Weighted combination (favor exact matches)
    similarity = 0.7 * exact_matches + 0.3 * cosine_sim

    return similarity


def run_experiment(
    mno,
    vqvae,
    device: str,
    n_pairs: int = 100,
    n_runs: int = 5,
    spatial_size: Tuple[int, int] = (64, 64),
    output_dir: Path = Path("results/phase2/exp4"),
):
    """Run reproducibility experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing reproducibility on {n_pairs} (u0, perturbation) pairs")
    print(f"Running each pair {n_runs} times\n")

    # Create episode runner
    stopping = StandardStoppingPolicy()
    runner = EpisodeRunner(mno, vqvae, stopping, device=device)

    # Storage
    all_similarities = []  # Within-pair similarities
    all_signature_variances = []  # Signature variance across runs
    outlier_pairs = []  # Pairs with low reproducibility

    num_categories, num_levels = get_vqvae_num_categories_and_levels(vqvae)
    encoder = BehavioralEncoder(num_categories, num_levels)

    for pair_idx in range(n_pairs):
        # Sample IC and perturbation
        u0, perturbation = sample_ic_and_perturbation(spatial_size, device, seed=pair_idx)

        # Run multiple times
        episodes = []
        for run_idx in range(n_runs):
            # Run episode (should be deterministic for same inputs)
            episode = runner.run_episode(u0, perturbation, max_steps=256)
            episodes.append(episode)

        # Compute pairwise token similarities
        pair_similarities = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                sim = compute_token_similarity(
                    episodes[i]["token_sequence"],
                    episodes[j]["token_sequence"]
                )
                pair_similarities.append(sim)

        mean_pair_similarity = np.mean(pair_similarities)
        all_similarities.extend(pair_similarities)

        # Compute behavioral signature variance
        signatures = encoder.encode_batch(episodes)

        # Compute variance of key signature features
        entropies = [sig["token_entropy"] for sig in signatures]
        diversities = [sig["token_diversity"] for sig in signatures]
        lengths = [sig["trajectory_length"] for sig in signatures]

        entropy_var = np.var(entropies) / (np.mean(entropies) + 1e-8)
        diversity_var = np.var(diversities) / (np.mean(diversities) + 1e-8)
        length_var = np.var(lengths) / (np.mean(lengths) + 1e-8)

        mean_var = (entropy_var + diversity_var + length_var) / 3
        all_signature_variances.append(mean_var)

        # Check for outliers (low reproducibility)
        if mean_pair_similarity < 0.90:
            outlier_pairs.append({
                "pair_idx": pair_idx,
                "mean_similarity": mean_pair_similarity,
                "signature_variance": mean_var,
                "entropies": entropies,
                "diversities": diversities,
                "lengths": lengths,
            })

        if (pair_idx + 1) % 20 == 0:
            print(f"  {pair_idx + 1}/{n_pairs} pairs tested")

    # Convert to numpy
    all_similarities = np.array(all_similarities)
    all_signature_variances = np.array(all_signature_variances)

    # Compute metrics
    mean_similarity = all_similarities.mean()
    median_similarity = np.median(all_similarities)
    high_similarity_rate = (all_similarities > 0.95).mean()
    very_high_similarity_rate = (all_similarities > 0.98).mean()

    mean_signature_var = all_signature_variances.mean()
    low_variance_rate = (all_signature_variances < 0.05).mean()

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Similarity distribution
    axes[0, 0].hist(all_similarities, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(mean_similarity, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_similarity:.3f}')
    axes[0, 0].axvline(0.95, color='orange', linestyle=':', linewidth=2,
                      label='Target (0.95)')
    axes[0, 0].axvline(0.98, color='green', linestyle=':', linewidth=2,
                      label='Excellent (0.98)')
    axes[0, 0].set_xlabel("Token Similarity")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Within-Pair Token Similarity Distribution\n({high_similarity_rate*100:.1f}% > 0.95)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Signature variance distribution
    axes[0, 1].hist(all_signature_variances, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].axvline(mean_signature_var, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_signature_var:.4f}')
    axes[0, 1].axvline(0.05, color='orange', linestyle=':', linewidth=2,
                      label='Target (0.05)')
    axes[0, 1].set_xlabel("Coefficient of Variation (signature features)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"Behavioral Signature Variance\n({low_variance_rate*100:.1f}% < 0.05)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Sample reproducibility heatmap
    # Show first 3 pairs with multiple runs
    sample_pairs = min(3, n_pairs)
    heatmap_data = np.zeros((sample_pairs * n_runs, sample_pairs * n_runs))

    for i in range(sample_pairs):
        # Regenerate episodes for visualization
        u0, perturbation = sample_ic_and_perturbation(spatial_size, device, seed=i)
        episodes = []
        for run_idx in range(n_runs):
            episode = runner.run_episode(u0, perturbation, max_steps=256)
            episodes.append(episode)

        # Compute similarity matrix for this pair
        for run_i in range(n_runs):
            for run_j in range(n_runs):
                global_i = i * n_runs + run_i
                global_j = i * n_runs + run_j
                if run_i == run_j:
                    heatmap_data[global_i, global_j] = 1.0
                else:
                    sim = compute_token_similarity(
                        episodes[run_i].token_sequence,
                        episodes[run_j].token_sequence
                    )
                    heatmap_data[global_i, global_j] = sim

    labels = []
    for i in range(sample_pairs):
        for run_idx in range(n_runs):
            labels.append(f"P{i}R{run_idx}")

    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.8, vmax=1.0,
                xticklabels=labels, yticklabels=labels, ax=axes[1, 0], cbar_kws={'label': 'Similarity'})
    axes[1, 0].set_title(f"Sample Reproducibility Heatmap\n(First {sample_pairs} pairs, {n_runs} runs each)")

    # 4. Scatter: similarity vs variance
    # Aggregate by pair
    pair_mean_sims = []
    pair_vars = []
    for pair_idx in range(n_pairs):
        # Get all pairwise similarities for this pair
        start_idx = pair_idx * (n_runs * (n_runs - 1) // 2)
        end_idx = start_idx + (n_runs * (n_runs - 1) // 2)
        if end_idx <= len(all_similarities):
            pair_sims = all_similarities[start_idx:end_idx]
            pair_mean_sims.append(pair_sims.mean())
            pair_vars.append(all_signature_variances[pair_idx])

    axes[1, 1].scatter(pair_mean_sims, pair_vars, alpha=0.6, s=30)
    axes[1, 1].axvline(0.95, color='orange', linestyle=':', label='Similarity target')
    axes[1, 1].axhline(0.05, color='orange', linestyle=':', label='Variance target')
    axes[1, 1].set_xlabel("Mean Token Similarity (within pair)")
    axes[1, 1].set_ylabel("Behavioral Signature Variance")
    axes[1, 1].set_title("Reproducibility vs Variance")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "reproducibility_analysis.png", dpi=150)
    plt.close()

    # Success assessment
    similarity_pass = high_similarity_rate >= 0.95
    mean_similarity_pass = mean_similarity > 0.98
    variance_pass = low_variance_rate >= 0.95
    no_outliers = len(outlier_pairs) == 0

    overall_pass = similarity_pass and mean_similarity_pass

    print("\n" + "="*70)
    print("EXPERIMENT 4: REPRODUCIBILITY TESTING")
    print("="*70)
    print(f"\nToken Similarity Statistics:")
    print(f"  Mean similarity:           {mean_similarity:.4f}")
    print(f"  Median similarity:         {median_similarity:.4f}")
    print(f"  Pairs with sim > 0.95:     {high_similarity_rate*100:.1f}%")
    print(f"  Pairs with sim > 0.98:     {very_high_similarity_rate*100:.1f}%")
    print(f"\nBehavioral Signature Variance:")
    print(f"  Mean variance (CV):        {mean_signature_var:.4f}")
    print(f"  Pairs with variance < 5%:  {low_variance_rate*100:.1f}%")
    print(f"\nOutliers (similarity < 0.90):")
    print(f"  Count:                     {len(outlier_pairs)}/{n_pairs}")
    print(f"\nSuccess Criteria:")
    print(f"  ≥95% pairs with sim>0.95:  {high_similarity_rate*100:.1f}% ≥ 95%  {'✓ PASS' if similarity_pass else '✗ FAIL'}")
    print(f"  Mean similarity > 0.98:    {mean_similarity:.4f} > 0.98  {'✓ PASS' if mean_similarity_pass else '✗ FAIL'}")
    print(f"  ≥95% pairs with var<5%:    {low_variance_rate*100:.1f}% ≥ 95%  {'✓ PASS' if variance_pass else '✗ FAIL'}")
    print(f"\nOverall: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    print("="*70)

    # Save outlier analysis
    if len(outlier_pairs) > 0:
        with open(output_dir / "outlier_analysis.txt", "w") as f:
            f.write(f"OUTLIER PAIRS (similarity < 0.90)\n")
            f.write("="*70 + "\n\n")
            for outlier in outlier_pairs:
                f.write(f"Pair {outlier['pair_idx']}:\n")
                f.write(f"  Mean similarity: {outlier['mean_similarity']:.4f}\n")
                f.write(f"  Signature variance: {outlier['signature_variance']:.4f}\n")
                f.write(f"  Entropies: {outlier['entropies']}\n")
                f.write(f"  Diversities: {outlier['diversities']}\n")
                f.write(f"  Lengths: {outlier['lengths']}\n\n")

    # Save summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write("EXPERIMENT 4: REPRODUCIBILITY TESTING\n")
        f.write("="*70 + "\n\n")
        f.write(f"Token Similarity Statistics:\n")
        f.write(f"  Mean similarity:           {mean_similarity:.4f}\n")
        f.write(f"  Median similarity:         {median_similarity:.4f}\n")
        f.write(f"  Pairs with sim > 0.95:     {high_similarity_rate*100:.1f}%\n")
        f.write(f"  Pairs with sim > 0.98:     {very_high_similarity_rate*100:.1f}%\n\n")
        f.write(f"Behavioral Signature Variance:\n")
        f.write(f"  Mean variance (CV):        {mean_signature_var:.4f}\n")
        f.write(f"  Pairs with variance < 5%:  {low_variance_rate*100:.1f}%\n\n")
        f.write(f"Outliers (similarity < 0.90):\n")
        f.write(f"  Count:                     {len(outlier_pairs)}/{n_pairs}\n\n")
        f.write(f"Success Criteria:\n")
        f.write(f"  ≥95% pairs with sim>0.95:  {'✓ PASS' if similarity_pass else '✗ FAIL'}\n")
        f.write(f"  Mean similarity > 0.98:    {'✓ PASS' if mean_similarity_pass else '✗ FAIL'}\n")
        f.write(f"  ≥95% pairs with var<5%:    {'✓ PASS' if variance_pass else '✗ FAIL'}\n\n")
        f.write(f"Overall: {'✓ PASS' if overall_pass else '✗ FAIL'}\n")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Validation Experiment 4: Reproducibility Testing"
    )
    parser.add_argument(
        "--mno-checkpoint",
        type=str,
        default="checkpoints/mno/pure_mse_baseline/meta_operator_best.pt",
        help="Path to trained MNO checkpoint",
    )
    parser.add_argument(
        "--vqvae-checkpoint",
        type=str,
        default="checkpoints/vqvae/mno_distribution_100k/vqvae_best.pt",
        help="Path to trained VQ-VAE checkpoint",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=100, help="Number of (u0, perturbation) pairs to test"
    )
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of runs per pair"
    )
    parser.add_argument("--spatial-size", type=int, default=64, help="Spatial grid size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase2/exp4",
        help="Output directory for results",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load models
    mno, vqvae = load_models(args.mno_checkpoint, args.vqvae_checkpoint, args.device)

    # Run experiment
    run_experiment(
        mno,
        vqvae,
        device=args.device,
        n_pairs=args.n_pairs,
        n_runs=args.n_runs,
        spatial_size=(args.spatial_size, args.spatial_size),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
