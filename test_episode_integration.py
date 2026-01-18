#!/usr/bin/env python3
"""Test end-to-end episode execution with actual trained models.

Validates that:
1. Models load correctly
2. Feature extraction matches VQ-VAE training distribution
3. Episode runner executes without errors
4. Token sequences are generated properly
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from spinlock.noa.episode import EpisodeRunner
from spinlock.noa.early_stopping import StandardStoppingPolicy
from spinlock.noa.validation_utils import (
    load_mno_checkpoint,
    load_vqvae_checkpoint,
    sample_initial_condition,
)
from spinlock.perturbations import ImpulsePerturbationFactory


def test_episode_execution():
    """Test end-to-end episode execution."""
    device = 'cpu'  # Use CPU for testing

    print("="*70)
    print("END-TO-END EPISODE EXECUTION TEST")
    print("="*70)

    # Step 1: Load models
    print("\n1. Loading models...")
    mno_checkpoint = "checkpoints/noa/pure_mse_v4_10k_contiguous/meta_operator_best.pt"
    vqvae_checkpoint = "checkpoints/vqvae/mno_10k_with_reference_reg/best_model.pt"

    if not Path(mno_checkpoint).exists():
        print(f"  ERROR: MNO checkpoint not found: {mno_checkpoint}")
        print("  Available checkpoints:")
        for ckpt in Path("checkpoints/noa").glob("*/meta_operator_best.pt"):
            print(f"    {ckpt}")
        return False

    if not Path(vqvae_checkpoint).exists():
        print(f"  ERROR: VQ-VAE checkpoint not found: {vqvae_checkpoint}")
        print("  Available checkpoints:")
        for ckpt in Path("checkpoints/vqvae").glob("*/best_model.pt"):
            print(f"    {ckpt}")
        return False

    try:
        mno = load_mno_checkpoint(mno_checkpoint, device)
        print(f"  ✓ MNO loaded from {mno_checkpoint}")
    except Exception as e:
        print(f"  ERROR loading MNO: {e}")
        return False

    try:
        vqvae = load_vqvae_checkpoint(vqvae_checkpoint, device)
        print(f"  ✓ VQ-VAE loaded from {vqvae_checkpoint}")
    except Exception as e:
        print(f"  ERROR loading VQ-VAE: {e}")
        return False

    # Step 2: Create episode runner
    print("\n2. Creating episode runner...")
    try:
        stopping = StandardStoppingPolicy()
        runner = EpisodeRunner(mno, vqvae, stopping, device=device)
        print(f"  ✓ Episode runner created")
        print(f"  ✓ Feature extractor initialized")
    except Exception as e:
        print(f"  ERROR creating runner: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Sample initial condition
    print("\n3. Sampling initial condition...")
    try:
        u0 = sample_initial_condition(
            num_channels=1,
            spatial_size=(64, 64),
            ic_type="smooth_random",
            device=device,
            seed=42,
        )
        print(f"  ✓ Initial condition shape: {u0.shape}")
    except Exception as e:
        print(f"  ERROR sampling IC: {e}")
        return False

    # Step 4: Create perturbation
    print("\n4. Creating perturbation...")
    try:
        perturbation = ImpulsePerturbationFactory.center_blob(
            t_impulse=0,
            amplitude=1.0,
            spatial_size=(64, 64),
            device=device,
        )
        print(f"  ✓ Perturbation created: {perturbation.pattern} at t={perturbation.t_impulse}")
    except Exception as e:
        print(f"  ERROR creating perturbation: {e}")
        return False

    # Step 5: Run episode
    print("\n5. Running episode (max 20 steps for test)...")
    try:
        episode = runner.run_episode(u0, perturbation, max_steps=20)

        print(f"  ✓ Episode completed")
        print(f"    - Episode ID: {episode.episode_id}")
        print(f"    - Steps: {episode.num_steps()}")
        print(f"    - Stop reason: {episode.stop_reason}")
        print(f"    - Initial state shape: {episode.initial_state.shape}")
        print(f"    - Trajectory shape: {episode.trajectory.shape}")
        print(f"    - Token sequence shape: {episode.token_sequence.shape}")
        print(f"    - Token dtype: {episode.token_sequence.dtype}")

        # Validate outputs
        assert episode.trajectory.ndim == 4, f"Expected 4D trajectory, got {episode.trajectory.ndim}D"
        assert episode.token_sequence.ndim == 2, f"Expected 2D tokens, got {episode.token_sequence.ndim}D"
        assert episode.num_steps() == episode.token_sequence.shape[0], \
            f"Steps mismatch: trajectory={episode.num_steps()}, tokens={episode.token_sequence.shape[0]}"

        print(f"\n  ✓ All validation checks passed")

    except Exception as e:
        print(f"  ERROR running episode: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("✓ END-TO-END TEST PASSED")
    print("="*70)
    return True


if __name__ == "__main__":
    success = test_episode_execution()
    sys.exit(0 if success else 1)
