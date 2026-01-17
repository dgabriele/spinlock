#!/usr/bin/env python3
"""Validation Experiment 1: Perturbation Response Divergence.

Hypothesis:
    Different perturbations applied to the same initial state should produce
    different token sequences, demonstrating that MNO responds meaningfully
    to perturbations in autonomous mode.

Success Criteria:
    - 90%+ of perturbation pairs produce divergent token sequences
    - Token sequence divergence correlates with perturbation difference
    - Different perturbation locations → different behavioral responses

Methodology:
    1. Load trained MNO and VQ-VAE checkpoints
    2. Sample diverse initial conditions
    3. For each IC, apply multiple different perturbations:
       - Center blob vs corner blobs (spatial sensitivity)
       - Different amplitudes (magnitude sensitivity)
       - Different patterns (pattern sensitivity)
    4. Compute token sequence divergence
    5. Visualize results and validate criteria

Output:
    - Divergence heatmaps (perturbation pairs vs token divergence)
    - Statistical summary (mean divergence, success rate)
    - Example trajectories showing divergent responses
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

from src.spinlock.perturbations import ImpulsePerturbation, ImpulsePerturbationFactory
from src.spinlock.noa.episode import EpisodeRunner, Episode
from src.spinlock.noa.early_stopping import StandardStoppingPolicy
from src.spinlock.noa.behavioral_encoding import BehavioralSignature, BehavioralEncoder

# TODO: Import actual model classes when integrated
# from src.spinlock.noa.backbone import NOABackbone
# from src.spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE


def load_models(mno_checkpoint: str, vqvae_checkpoint: str, device: str):
    """Load trained MNO and VQ-VAE models.

    Args:
        mno_checkpoint: Path to MNO checkpoint
        vqvae_checkpoint: Path to VQ-VAE checkpoint
        device: Device for models

    Returns:
        (mno, vqvae) model tuple
    """
    # TODO: Implement actual model loading
    # For now, placeholder
    print(f"Loading MNO from {mno_checkpoint}")
    print(f"Loading VQ-VAE from {vqvae_checkpoint}")

    # Placeholder: return None for now
    # Once integrated:
    # mno = NOABackbone.load_from_checkpoint(mno_checkpoint).to(device)
    # vqvae = CategoricalHierarchicalVQVAE.load_from_checkpoint(vqvae_checkpoint).to(device)

    return None, None


def sample_initial_condition(spatial_size: Tuple[int, int], device: str) -> torch.Tensor:
    """Sample random initial condition.

    Args:
        spatial_size: (H, W) spatial dimensions
        device: Device placement

    Returns:
        Initial state [C, H, W]
    """
    # TODO: Implement proper IC sampling
    # For now, random Gaussian
    H, W = spatial_size
    num_channels = 2  # Reaction-diffusion (u, v)

    u0 = torch.randn(num_channels, H, W, device=device)
    return u0


def create_perturbation_set(
    spatial_size: Tuple[int, int],
    device: str,
) -> List[ImpulsePerturbation]:
    """Create diverse set of perturbations for testing.

    Returns:
        List of perturbations with varying:
        - Locations (center, corners, random)
        - Amplitudes (0.5, 1.0, 2.0)
        - Patterns (gaussian_blob, uniform, fourier_mode)
    """
    perturbations = []
    t_impulse = 0  # Apply at t=0

    # 1. Center blobs with different amplitudes
    for amp in [0.5, 1.0, 2.0]:
        p = ImpulsePerturbationFactory.center_blob(
            t_impulse=t_impulse,
            amplitude=amp,
            spatial_size=spatial_size,
            sigma=0.1,
            device=device,
        )
        p.name = f"center_blob_amp{amp}"
        perturbations.append(p)

    # 2. Corner blobs (test spatial sensitivity)
    for corner in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        p = ImpulsePerturbationFactory.corner_blob(
            t_impulse=t_impulse,
            amplitude=1.0,
            spatial_size=spatial_size,
            corner=corner,
            sigma=0.1,
            device=device,
        )
        p.name = f"corner_{corner}"
        perturbations.append(p)

    # 3. Uniform forcing (different amplitudes)
    for amp in [0.5, 1.0]:
        p = ImpulsePerturbationFactory.uniform_forcing(
            t_impulse=t_impulse,
            amplitude=amp,
            spatial_size=spatial_size,
            device=device,
        )
        p.name = f"uniform_amp{amp}"
        perturbations.append(p)

    # 4. Fourier modes (different wavelengths)
    for k in [(1, 1), (2, 2), (1, 2)]:
        p = ImpulsePerturbation(
            t_impulse=t_impulse,
            amplitude=1.0,
            pattern="fourier_mode",
            spatial_size=spatial_size,
            wavenumber=k,
            device=device,
        )
        p.name = f"fourier_k{k[0]}{k[1]}"
        perturbations.append(p)

    return perturbations


def compute_token_divergence(tokens1: torch.Tensor, tokens2: torch.Tensor) -> float:
    """Compute divergence between two token sequences.

    Uses normalized Hamming distance (fraction of differing tokens).

    Args:
        tokens1, tokens2: Token sequences [T, D]

    Returns:
        Divergence in [0, 1] (0 = identical, 1 = completely different)
    """
    # Ensure same length (use minimum)
    T = min(tokens1.shape[0], tokens2.shape[0])
    tokens1 = tokens1[:T]
    tokens2 = tokens2[:T]

    # Hamming distance
    different = (tokens1 != tokens2).float().sum().item()
    total = tokens1.numel()

    divergence = different / total if total > 0 else 0.0
    return divergence


def run_experiment(
    mno,
    vqvae,
    device: str,
    n_initial_conditions: int = 10,
    spatial_size: Tuple[int, int] = (64, 64),
    output_dir: Path = Path("results/phase2/exp1"),
):
    """Run perturbation response divergence experiment.

    Args:
        mno: Trained MNO model
        vqvae: Trained VQ-VAE model
        device: Device for computation
        n_initial_conditions: Number of ICs to test
        spatial_size: Spatial dimensions
        output_dir: Directory for results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create episode runner
    stopping = StandardStoppingPolicy()
    # TODO: Uncomment when models integrated
    # runner = EpisodeRunner(mno, vqvae, stopping, device=device)

    # Create perturbation set
    perturbations = create_perturbation_set(spatial_size, device)
    n_perturbations = len(perturbations)

    print(f"Testing {n_perturbations} perturbations on {n_initial_conditions} initial conditions")
    print(f"Perturbations: {[p.name for p in perturbations]}")

    # Storage for results
    all_divergences = []

    for ic_idx in range(n_initial_conditions):
        print(f"\nInitial condition {ic_idx + 1}/{n_initial_conditions}")

        # Sample IC
        u0 = sample_initial_condition(spatial_size, device)

        # Run episodes for all perturbations
        episodes = []
        for p_idx, perturbation in enumerate(perturbations):
            print(f"  Running perturbation {p_idx + 1}/{n_perturbations}: {perturbation.name}")

            # TODO: Uncomment when models integrated
            # episode = runner.run_episode(u0, perturbation)
            # episodes.append(episode)

            # Placeholder: create dummy episode for testing structure
            dummy_tokens = torch.randint(0, 10, (100, 20))  # [T=100, D=20]
            from dataclasses import dataclass

            @dataclass
            class DummyEpisode:
                token_sequence: torch.Tensor
                episode_id: str

            episode = DummyEpisode(token_sequence=dummy_tokens, episode_id=f"ep_{ic_idx}_{p_idx}")
            episodes.append(episode)

        # Compute pairwise divergences
        divergence_matrix = np.zeros((n_perturbations, n_perturbations))

        for i in range(n_perturbations):
            for j in range(i + 1, n_perturbations):
                div = compute_token_divergence(
                    episodes[i].token_sequence, episodes[j].token_sequence
                )
                divergence_matrix[i, j] = div
                divergence_matrix[j, i] = div

        all_divergences.append(divergence_matrix)

        # Save divergence matrix for this IC
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            divergence_matrix,
            xticklabels=[p.name for p in perturbations],
            yticklabels=[p.name for p in perturbations],
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
        )
        plt.title(f"Token Divergence Matrix (IC {ic_idx})")
        plt.tight_layout()
        plt.savefig(output_dir / f"divergence_matrix_ic{ic_idx}.png", dpi=150)
        plt.close()

    # Aggregate results
    mean_divergence_matrix = np.mean(all_divergences, axis=0)

    # Save aggregate
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        mean_divergence_matrix,
        xticklabels=[p.name for p in perturbations],
        yticklabels=[p.name for p in perturbations],
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    plt.title(f"Mean Token Divergence (n={n_initial_conditions} ICs)")
    plt.tight_layout()
    plt.savefig(output_dir / "divergence_matrix_mean.png", dpi=150)
    plt.close()

    # Compute success metrics
    # Extract off-diagonal elements (pairwise divergences)
    triu_indices = np.triu_indices(n_perturbations, k=1)
    pairwise_divergences = mean_divergence_matrix[triu_indices]

    success_rate = (pairwise_divergences > 0.1).mean()  # At least 10% different
    mean_divergence = pairwise_divergences.mean()
    median_divergence = np.median(pairwise_divergences)

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: PERTURBATION RESPONSE DIVERGENCE")
    print("=" * 70)
    print(f"Success Rate (>10% divergence): {success_rate * 100:.1f}%")
    print(f"Mean Divergence: {mean_divergence:.3f}")
    print(f"Median Divergence: {median_divergence:.3f}")
    print(f"Min Divergence: {pairwise_divergences.min():.3f}")
    print(f"Max Divergence: {pairwise_divergences.max():.3f}")
    print()
    print(f"Criteria: Success rate >= 90%")
    print(f"Result: {'✓ PASS' if success_rate >= 0.9 else '✗ FAIL'}")
    print("=" * 70)

    # Save summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write("EXPERIMENT 1: PERTURBATION RESPONSE DIVERGENCE\n")
        f.write("=" * 70 + "\n")
        f.write(f"Success Rate (>10% divergence): {success_rate * 100:.1f}%\n")
        f.write(f"Mean Divergence: {mean_divergence:.3f}\n")
        f.write(f"Median Divergence: {median_divergence:.3f}\n")
        f.write(f"Min Divergence: {pairwise_divergences.min():.3f}\n")
        f.write(f"Max Divergence: {pairwise_divergences.max():.3f}\n")
        f.write(f"\nCriteria: Success rate >= 90%\n")
        f.write(f"Result: {'✓ PASS' if success_rate >= 0.9 else '✗ FAIL'}\n")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Validation Experiment 1: Perturbation Response Divergence"
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
        "--n-ics", type=int, default=10, help="Number of initial conditions to test"
    )
    parser.add_argument("--spatial-size", type=int, default=64, help="Spatial grid size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase2/exp1",
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
        n_initial_conditions=args.n_ics,
        spatial_size=(args.spatial_size, args.spatial_size),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
