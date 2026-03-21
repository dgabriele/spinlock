"""Calibrate denoising-step ↔ truncation-level mapping from empirical data.

Runs the D3PM sampling loop with full snapshot recording, then measures
agreement between each denoising step's token snapshot and each truncation
level's ground-truth tokens. The output is a JSON artifact with empirical
noise boundaries that can replace the uniform-spacing default in
DenoisingRoundtripLossConfig.noise_boundaries.

Algorithm:
    1. Load trained D3PM + denoiser + multi-truncation pretokenized dataset
    2. For N samples, run sample(snapshot_steps=list(range(T)))
    3. At each step t, compute per-position agreement against each truncation
       level's GT tokens (temporal keys only)
    4. Find the truncation level maximizing agreement at each step
    5. Output: JSON with calibrated boundaries + agreement curves

Usage:
    python experiments/diffusion/scripts/calibrate_trajectory.py \
        --checkpoint experiments/diffusion/results/v8_joint/v8_joint_d3pm_best.pt \
        --tokenized-path datasets/ds_lenia_fourier_10k_pretokenized.h5 \
        --tokenizer checkpoints/lenia/vq/v3_fourier_10k/vq_tokenizer_best.pt \
        --output experiments/diffusion/configs/calibrated_boundaries.json \
        --n-samples 50
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_from_checkpoint(checkpoint_path, tokenizer_path, device="cuda"):
    """Load D3PM + denoiser from checkpoint, vocab from tokenizer."""
    from spinlock.tokens.tokenizer import VQTokenizer
    from spinlock.tokens.schema import TokenSchema
    from spinlock.experimental.diffusion.models import (
        DiscreteD3PM,
        DiffusionSchedule,
        DenoisingNetwork,
    )

    tokenizer = VQTokenizer.from_checkpoint(tokenizer_path)
    schema = TokenSchema.from_tokenizer(tokenizer)
    vocab_sizes = schema.vocab_sizes_dict()
    cat_info = schema.category_level_info_dict()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    # Rebuild D3PM with graded schedule if configured
    graded_cfg = config.diffusion.graded_schedule
    scale_factors = graded_cfg.scale_factors or {}
    if graded_cfg.position_scale_factors_path:
        with open(graded_cfg.position_scale_factors_path) as f:
            scale_factors = json.load(f)

    diffusion = DiscreteD3PM(
        vocab_sizes,
        DiffusionSchedule(
            num_timesteps=config.diffusion.num_timesteps,
            schedule_type=config.diffusion.schedule_type,
        ),
        cat_info,
        transition_type=config.diffusion.transition_type,
        beta_scaling=config.diffusion.beta_scaling,
        graded_schedule_enabled=graded_cfg.enabled,
        graded_scale_factors=scale_factors,
        non_temporal_scale=graded_cfg.non_temporal_scale,
        family_scale_overrides=graded_cfg.family_scale_overrides,
    )
    diffusion.load_state_dict(ckpt["diffusion_state_dict"])

    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=cat_info,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        use_hierarchical_guidance=config.model.use_hierarchical_guidance,
        hierarchical_guidance_weight=config.model.hierarchical_guidance_weight,
        guidance_mode=config.model.hierarchical_guidance_mode,
        transition_type=config.diffusion.transition_type,
    )
    denoiser.load_state_dict(ckpt["denoiser_state_dict"])

    diffusion.to(device).eval()
    denoiser.to(device).eval()

    return diffusion, denoiser, vocab_sizes, cat_info, config


def load_multi_trunc_tokens(
    tokenized_path: str,
    tokenizer_path: str,
    truncation_lengths: List[int],
    n_samples: int,
    primary_truncation: int = 512,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Load ground-truth tokens at multiple truncation levels."""
    from spinlock.experimental.diffusion.data import (
        PretokenizedDiffusionDataset,
        collate_dict_batch,
    )
    from spinlock.experimental.diffusion.data.hierarchical_masking import (
        HierarchicalMaskGenerator,
        MaskingStrategy,
    )
    from spinlock.tokens.tokenizer import VQTokenizer
    from spinlock.tokens.schema import TokenSchema

    tokenizer = VQTokenizer.from_checkpoint(tokenizer_path)
    schema = TokenSchema.from_tokenizer(tokenizer)
    vocab_sizes = schema.vocab_sizes_dict()
    cat_info = schema.category_level_info_dict()

    mask_gen = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.RANDOM,
        vocab_sizes=vocab_sizes,
        category_level_info=cat_info,
        mask_probability=0.5,
    )

    ds = PretokenizedDiffusionDataset(
        tokenized_dataset_path=Path(tokenized_path),
        mask_generator=mask_gen,
        truncation_length=primary_truncation,
        aux_truncation_lengths=truncation_lengths,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=min(n_samples, len(ds)),
        shuffle=False,
        collate_fn=collate_dict_batch,
    )
    batch = next(iter(loader))

    # Collect aux truncation tokens + primary
    result = {}
    for tl, trunc_dict in batch.get("aux_trunc_tokens", {}).items():
        result[tl] = {k: v[:n_samples] for k, v in trunc_dict.items()}

    # Also include primary truncation tokens
    result[primary_truncation] = {
        k: v[:n_samples] for k, v in batch["tokens"].items()
    }

    return result


def calibrate_trajectory(
    checkpoint_path: str,
    tokenized_path: str,
    tokenizer_path: str,
    output_path: str,
    n_samples: int = 50,
    truncation_lengths: List[int] = None,
    device: str = "cuda",
) -> Dict:
    """Run calibration and produce empirical noise boundaries.

    Args:
        checkpoint_path: Path to trained D3PM checkpoint.
        tokenized_path: Path to multi-truncation pretokenized HDF5.
        tokenizer_path: Path to VQTokenizer checkpoint.
        output_path: Path to write JSON output.
        n_samples: Number of samples to run.
        truncation_lengths: Truncation levels to calibrate against.
        device: Compute device.

    Returns:
        Dict with calibrated boundaries and agreement curves.
    """
    if truncation_lengths is None:
        truncation_lengths = [32, 64, 128, 256, 512]

    logger.info("Loading model from checkpoint...")
    diffusion, denoiser, vocab_sizes, cat_info, config = load_from_checkpoint(
        checkpoint_path, tokenizer_path, device,
    )

    T = diffusion.schedule.num_timesteps
    logger.info(f"D3PM: T={T}, {len(vocab_sizes)} keys")

    # Identify temporal keys
    temporal_keys = [
        k for k, info in cat_info.items() if info.get("family") == "temporal"
    ]
    logger.info(f"Temporal keys for calibration: {len(temporal_keys)}")

    # Load multi-truncation GT
    logger.info(f"Loading GT tokens at truncation levels {truncation_lengths}...")
    gt_trunc = load_multi_trunc_tokens(
        tokenized_path, tokenizer_path, truncation_lengths, n_samples,
    )
    available_truncs = sorted(gt_trunc.keys())
    logger.info(f"Available truncation levels: {available_truncs}")

    # Move GT to device
    for tl in gt_trunc:
        gt_trunc[tl] = {k: v.to(device) for k, v in gt_trunc[tl].items()}

    # Run sample() with full snapshot recording
    all_steps = list(range(T))
    logger.info(f"Sampling {n_samples} trajectories with {T} snapshots each...")

    result = diffusion.sample(
        batch_size=n_samples,
        denoising_network=denoiser,
        device=device,
        snapshot_steps=all_steps,
    )

    if not isinstance(result, tuple):
        raise RuntimeError("sample() did not return trajectory — snapshot_steps not working")
    final_tokens, trajectory = result

    # For each step, compute agreement against each truncation level
    # agreement_curves[step][trunc_len] = mean agreement (temporal keys only)
    logger.info("Computing agreement curves...")
    agreement_curves: Dict[int, Dict[int, float]] = {}

    for step in sorted(trajectory.keys()):
        snapshot = trajectory[step]
        agreement_curves[step] = {}

        for tl in available_truncs:
            gt = gt_trunc[tl]
            n_agree = 0
            n_total = 0

            for key in temporal_keys:
                if key in snapshot and key in gt:
                    n_agree += (snapshot[key] == gt[key]).sum().item()
                    n_total += gt[key].numel()

            agreement_curves[step][tl] = n_agree / max(n_total, 1)

    # Also measure final (t=0) agreement
    agreement_curves[-1] = {}
    for tl in available_truncs:
        gt = gt_trunc[tl]
        n_agree = 0
        n_total = 0
        for key in temporal_keys:
            if key in final_tokens and key in gt:
                n_agree += (final_tokens[key] == gt[key]).sum().item()
                n_total += gt[key].numel()
        agreement_curves[-1][tl] = n_agree / max(n_total, 1)

    # Find best-matching truncation at each step
    best_trunc_per_step: Dict[int, int] = {}
    for step in sorted(agreement_curves.keys()):
        if step == -1:
            continue
        best_tl = max(
            agreement_curves[step],
            key=lambda tl: agreement_curves[step][tl],
        )
        best_trunc_per_step[step] = best_tl

    # Derive noise boundaries from the step→truncation mapping
    # Group steps by their best truncation, find boundary crossings
    # noise_frac = step / T, boundaries are at the transitions
    sorted_truncs = sorted(available_truncs)
    n_levels = len(sorted_truncs)

    boundaries = []
    if n_levels > 1:
        # Walk from high noise (step=T-1) to low noise (step=0)
        # Track when the best-matching truncation changes
        prev_trunc_idx = 0
        for step in reversed(range(T)):
            if step not in best_trunc_per_step:
                continue
            best_tl = best_trunc_per_step[step]
            trunc_idx = sorted_truncs.index(best_tl) if best_tl in sorted_truncs else 0

            if trunc_idx != prev_trunc_idx and trunc_idx > prev_trunc_idx:
                # Transition point: noise_frac where truncation level increases
                noise_frac = step / T
                inv_frac = 1.0 - noise_frac
                boundaries.append(round(inv_frac, 4))
                prev_trunc_idx = trunc_idx

        boundaries = sorted(boundaries)

    # If we didn't find enough boundaries, fall back to uniform
    if len(boundaries) < n_levels - 1:
        logger.warning(
            f"Found only {len(boundaries)} boundaries for {n_levels} levels, "
            "padding with uniform spacing"
        )
        uniform = [(i + 1) / n_levels for i in range(n_levels - 1)]
        # Merge: keep empirical where available, fill gaps from uniform
        if not boundaries:
            boundaries = uniform
        else:
            while len(boundaries) < n_levels - 1:
                # Add the most distant uniform boundary
                for ub in uniform:
                    if not any(abs(ub - b) < 0.05 for b in boundaries):
                        boundaries.append(round(ub, 4))
                        boundaries = sorted(boundaries)
                        break
                else:
                    break

    # Trim to exactly n_levels - 1
    boundaries = boundaries[: n_levels - 1]

    # Log results
    logger.info(f"\nCalibrated boundaries: {boundaries}")
    logger.info(f"Truncation levels: {sorted_truncs}")

    # Log agreement at representative steps
    for step in [int(T * f) for f in [0.8, 0.6, 0.4, 0.2, 0.0]]:
        step = max(0, min(step, T - 1))
        if step in agreement_curves:
            agrees = agreement_curves[step]
            parts = [f"T{tl}={a:.3f}" for tl, a in sorted(agrees.items())]
            best = best_trunc_per_step.get(step, "?")
            logger.info(f"  step={step:3d} (noise={step/T:.2f}): {', '.join(parts)}  best→T{best}")

    # Final step agreement
    if -1 in agreement_curves:
        agrees = agreement_curves[-1]
        parts = [f"T{tl}={a:.3f}" for tl, a in sorted(agrees.items())]
        logger.info(f"  final (t=0): {', '.join(parts)}")

    # Build output
    output = {
        "noise_boundaries": boundaries,
        "truncation_levels": sorted_truncs,
        "n_samples": n_samples,
        "num_timesteps": T,
        "best_trunc_per_step": {
            str(k): v for k, v in sorted(best_trunc_per_step.items())
        },
        "agreement_curves": {
            str(step): {str(tl): round(a, 4) for tl, a in agrees.items()}
            for step, agrees in sorted(agreement_curves.items())
        },
    }

    # Write JSON
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    logger.info(f"Wrote calibration to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate denoising-step to truncation-level mapping"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained D3PM checkpoint",
    )
    parser.add_argument(
        "--tokenized-path",
        type=str,
        required=True,
        help="Path to multi-truncation pretokenized HDF5",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to VQTokenizer checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write JSON output",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples for calibration (default: 50)",
    )
    parser.add_argument(
        "--truncation-lengths",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="Truncation levels (default: 32 64 128 256 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )

    args = parser.parse_args()
    calibrate_trajectory(
        checkpoint_path=args.checkpoint,
        tokenized_path=args.tokenized_path,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        n_samples=args.n_samples,
        truncation_lengths=args.truncation_lengths,
        device=args.device,
    )


if __name__ == "__main__":
    main()
