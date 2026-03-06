"""Offline hard-target refinement loop for D3PM.

Closed loop: D3PM predict → CVAE decode → MNO rollout → retokenize →
quality filter → fine-tune.

The D3PM generates temporal token completions from partial observations, but
has no mechanism to verify that predictions correspond to *realizable* operator
dynamics. This script closes the loop through the CVAE + MNO surrogate:

    1. Tokenize dataset temporal features → dataset_tokens (temporal only)
    2. Random mask → observed/target split
    3. D3PM inpainting → completed_tokens (temporal only)
    4. CVAE.sample(completed_tokens) → (theta_pred, IC_pred)
       The CVAE models P(theta, IC | temporal_tokens): given what the dynamics
       look like, generate plausible physical parameters and initial conditions.
    5. MNO rollout from CVAE-sampled params → realized trajectory
    6. Retokenize realized trajectory → realized_tokens (temporal only)
    7. Quality-filter: roundtrip self-consistency check — keep only samples
       where retokenized observed positions match the original dataset tokens
    8. Fine-tune D3PM on the accepted "realized" hard targets

E2E differentiability through the MNO was rejected: 256-512 step gradient chain
through chaotic dynamics is too noisy. Hard targets provide a clean,
non-differentiable supervision signal.

Usage:
    poetry run python experiments/diffusion/scripts/refine_d3pm.py \
        --config experiments/diffusion/configs/v7_refinement.yaml

    # Smoke test (CPU, 2 samples, 1 cycle):
    poetry run python experiments/diffusion/scripts/refine_d3pm.py \
        --config experiments/diffusion/configs/v7_refinement.yaml \
        --max-samples 2 --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from spinlock.data import SpinlockDataset
from spinlock.experimental.common.config.loader import load_experiment_config
from spinlock.experimental.diffusion.config import RefinementConfig
from spinlock.experimental.diffusion.models import (
    DenoisingNetwork,
    DiffusionSchedule,
    DiscreteD3PM,
)
from spinlock.tokens.cvae import TokenConditionedCVAE
from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.tokens.schema import TokenSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Model loading ────────────────────────────────────────────────────────────


def load_d3pm_and_denoiser(
    checkpoint_path: str, device: str
) -> Tuple[DiscreteD3PM, DenoisingNetwork, dict]:
    """Load trained D3PM + denoiser from checkpoint.

    Returns:
        (diffusion_model, denoising_network, checkpoint_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Extract vocab sizes from saved config's tokenizer
    tokenizer_ckpt = config.dataset.tokenizer_checkpoint
    if tokenizer_ckpt is not None and Path(str(tokenizer_ckpt)).exists():
        tokenizer = VQTokenizer.from_checkpoint(tokenizer_ckpt)
        schema = TokenSchema.from_tokenizer(tokenizer)
    elif config.dataset.use_pretokenized:
        schema = TokenSchema.from_pretokenized(config.dataset.tokenized_path)
    else:
        raise ValueError(
            "Cannot determine vocab sizes: no tokenizer_checkpoint or pretokenized path"
        )

    vocab_sizes = schema.vocab_sizes_dict()
    category_level_info = schema.category_level_info_dict()

    # Reconstruct diffusion model
    graded_cfg = config.diffusion.graded_schedule
    scale_factors = graded_cfg.scale_factors or {}
    if graded_cfg.position_scale_factors_path:
        import json

        with open(graded_cfg.position_scale_factors_path) as f:
            scale_factors = json.load(f)

    diffusion = DiscreteD3PM(
        vocab_sizes,
        DiffusionSchedule(
            num_timesteps=config.diffusion.num_timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            schedule_type=config.diffusion.schedule_type,
        ),
        category_level_info,
        transition_type=config.diffusion.transition_type,
        beta_scaling=config.diffusion.beta_scaling,
        graded_schedule_enabled=graded_cfg.enabled,
        graded_scale_factors=scale_factors,
        non_temporal_scale=graded_cfg.non_temporal_scale,
    )
    diffusion.load_state_dict(ckpt["diffusion_state_dict"])
    diffusion.to(device)

    # Reconstruct denoiser
    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
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
    denoiser.to(device)

    logger.info(
        f"Loaded D3PM + denoiser from {checkpoint_path} "
        f"(epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('best_val_loss', '?')})"
    )
    return diffusion, denoiser, ckpt


def load_mno(checkpoint_path: str, device: str):
    """Load trained V2MNO from checkpoint.

    Returns:
        V2MNO model instance (eval mode, on device)
    """
    from spinlock.mno.v2.model import V2MNO

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # V2MNO checkpoints store config + state dict
    config = ckpt["config"]
    dims = ckpt.get("dims", {})
    operator_type = ckpt.get("operator_type", None)

    mno = V2MNO.from_config(config, dims, device=device, operator_type=operator_type)
    mno.load_state_dict(ckpt["model_state_dict"])
    mno.to(device)
    mno.eval()

    logger.info(f"Loaded V2MNO from {checkpoint_path}")
    return mno


def load_tokenizer(checkpoint_path: str) -> VQTokenizer:
    """Load VQTokenizer with inverse decoders.

    Looks for theta_inverse and initial_inverse checkpoints alongside the main
    checkpoint (standard layout from VQTokenizer training).
    """
    ckpt_dir = Path(checkpoint_path).parent
    theta_inv = ckpt_dir / "theta_inverse_best.pt"
    initial_inv = ckpt_dir / "initial_inverse_best.pt"

    tokenizer = VQTokenizer.from_checkpoint(
        checkpoint_path,
        theta_inverse_path=str(theta_inv) if theta_inv.exists() else None,
        initial_inverse_path=str(initial_inv) if initial_inv.exists() else None,
    )
    logger.info(f"Loaded VQTokenizer from {checkpoint_path}")
    return tokenizer


def load_cvae(checkpoint_path: str, device: str) -> TokenConditionedCVAE:
    """Load trained Token-Conditioned CVAE.

    The CVAE models P(theta, IC | temporal_tokens) — given temporal tokens
    describing dynamics, generate plausible physical parameters and ICs.

    Args:
        checkpoint_path: Path to CVAE checkpoint
        device: Target device

    Returns:
        TokenConditionedCVAE in eval mode
    """
    cvae = TokenConditionedCVAE.from_checkpoint(
        Path(checkpoint_path), device=device
    )
    logger.info(f"Loaded CVAE from {checkpoint_path}")
    return cvae


# ── Hard target generation ───────────────────────────────────────────────────


def generate_random_mask(
    keys: List[str], batch_size: int, mask_probability: float, device: str
) -> Tuple[Dict[str, torch.BoolTensor], Dict[str, torch.BoolTensor]]:
    """Generate random observed/target masks for inpainting.

    Returns:
        (observed_dict, target_dict) — both map key → [B] boolean.
        observed[k] = True means position k is observed (kept).
        target[k] = True means position k is masked (to predict).
    """
    observed = {}
    target = {}
    for key in keys:
        mask = torch.rand(batch_size, device=device) < mask_probability
        target[key] = mask
        observed[key] = ~mask
    return observed, target


def generate_hard_targets(
    dataset: SpinlockDataset,
    diffusion: DiscreteD3PM,
    denoiser: DenoisingNetwork,
    mno,
    tokenizer: VQTokenizer,
    cvae: TokenConditionedCVAE,
    config: RefinementConfig,
) -> List[Dict]:
    """Generate hard targets from the full refinement pipeline.

    For each sample:
        1. Tokenize dataset temporal features → dataset_tokens (temporal only)
        2. Random mask → observed/target split
        3. D3PM inpainting → completed_tokens (temporal only)
        4. CVAE.sample(completed_tokens) → (theta_pred, IC_pred)
        5. MNO rollout from CVAE-sampled params → realized trajectory
        6. Retokenize realized trajectory → realized_tokens (temporal only)
        7. Quality filter: compare realized_tokens vs dataset_tokens at
           observed positions (roundtrip self-consistency check)
        8. If agreement >= threshold: accept realized tokens at target positions

    The quality gate checks whether the CVAE → MNO → retokenize roundtrip
    reproduces the original dataset tokens at observed positions. This validates
    that the CVAE+MNO pipeline is accurate for this sample before trusting
    its predictions at masked positions.

    Returns:
        List of accepted hard-target dicts, each with:
            tokens: full token dict (dataset tokens at observed, realized at target)
            observed: observed mask dict
            target: target mask dict
            agreement: float agreement rate
    """
    device = config.device
    threshold = config.quality_filter.min_observed_agreement
    hard_targets = []
    total_samples = 0
    total_accepted = 0
    agreement_sum = 0.0

    # Discover temporal-only keys from tokenizer
    schema = TokenSchema.from_tokenizer(tokenizer)
    temporal_keys = schema.keys_for_family("temporal")

    # Process samples individually (MNO rollout is memory-intensive)
    for idx in range(len(dataset)):
        sample = dataset[idx]

        # Step 1: Tokenize from pre-extracted temporal features
        gt_temporal = sample.get("gt_raw_temporal")
        if gt_temporal is not None:
            gt_temporal = gt_temporal.unsqueeze(0).to(device)  # [1, T, D_raw]
            with torch.no_grad():
                all_tokens = tokenizer.tokenize(
                    temporal_features=gt_temporal,
                )
        else:
            # Fallback: tokenize from IC + theta (if temporal features unavailable)
            ic = sample["ic"].unsqueeze(0).to(device)
            params = sample["params"].unsqueeze(0).to(device)
            with torch.no_grad():
                all_tokens = tokenizer.tokenize(
                    initial_raw=ic,
                    theta_features=params,
                )

        # Filter to temporal-only keys
        dataset_tokens = {
            k: v.to(device) for k, v in all_tokens.items()
            if k in temporal_keys
        }

        keys = sorted(dataset_tokens.keys())
        if not keys:
            logger.debug(f"Sample {idx}: no temporal keys found, skipping")
            total_samples += 1
            continue
        B = 1

        # Step 2: Random mask
        observed_dict, target_dict = generate_random_mask(
            keys, B, config.mask_probability, device
        )

        # Step 3: D3PM inpainting (observed positions held fixed via RePaint)
        with torch.no_grad():
            completed_tokens = diffusion.sample(
                batch_size=B,
                observed_dict=observed_dict,
                x_0_dict=dataset_tokens,
                denoising_network=denoiser,
                device=device,
                start_step=config.d3pm_start_step,
            )

        # Step 4: CVAE decode — sample (theta, IC) from temporal tokens
        with torch.no_grad():
            cvae_output = cvae.sample(completed_tokens, n_samples=1)
            theta_pred = cvae_output["theta"]  # [1, theta_dim]
            u0_pred = cvae_output["grids"]     # [1, C, H, W]

        # Step 5: MNO rollout from CVAE-sampled params
        with torch.no_grad():
            conditioning = {
                "theta": theta_pred,
                "ic": u0_pred,
                "token_indices": completed_tokens,
            }
            try:
                trajectory = mno.rollout(conditioning, steps=config.rollout_steps)
            except Exception as e:
                logger.debug(f"Sample {idx}: MNO rollout failed: {e}")
                total_samples += 1
                continue

        # Step 6: Retokenize realized trajectory (temporal only)
        with torch.no_grad():
            all_realized = tokenizer.tokenize(
                temporal_raw=trajectory,
            )
        realized_tokens = {
            k: v.to(device) for k, v in all_realized.items()
            if k in temporal_keys
        }

        # Step 7: Quality filter — roundtrip self-consistency check
        # Compare realized_tokens vs dataset_tokens at observed positions.
        # If the CVAE+MNO can't reproduce what we already know (the observed
        # tokens), we shouldn't trust its predictions at masked positions.
        num_observed = 0
        num_agree = 0
        for key in keys:
            if key not in realized_tokens:
                continue
            obs_mask = observed_dict[key]  # [1] bool
            if obs_mask.any():
                dataset_val = dataset_tokens[key][obs_mask]
                realized_val = realized_tokens[key][obs_mask]
                num_observed += dataset_val.numel()
                num_agree += (dataset_val == realized_val).sum().item()

        agreement = num_agree / max(num_observed, 1)
        agreement_sum += agreement
        total_samples += 1

        # Step 8: Accept or reject
        if agreement >= threshold:
            # Build hard target: dataset tokens at observed, realized at target
            hard_target_tokens = {}
            for key in keys:
                if key not in realized_tokens:
                    hard_target_tokens[key] = dataset_tokens[key].clone()
                    continue
                tgt_mask = target_dict[key]    # [1]
                merged = dataset_tokens[key].clone()
                merged[tgt_mask] = realized_tokens[key][tgt_mask]
                hard_target_tokens[key] = merged

            hard_targets.append({
                "tokens": hard_target_tokens,
                "observed": observed_dict,
                "target": target_dict,
                "agreement": agreement,
            })
            total_accepted += 1

        if (idx + 1) % 100 == 0:
            logger.info(
                f"  Processed {idx + 1}/{len(dataset)}: "
                f"accepted={total_accepted}/{total_samples} "
                f"({100 * total_accepted / max(total_samples, 1):.1f}%), "
                f"mean_agreement={agreement_sum / max(total_samples, 1):.3f}"
            )

    logger.info(
        f"Hard target generation complete: {total_accepted}/{total_samples} accepted "
        f"({100 * total_accepted / max(total_samples, 1):.1f}%), "
        f"mean_agreement={agreement_sum / max(total_samples, 1):.3f}"
    )
    return hard_targets


# ── Fine-tuning ──────────────────────────────────────────────────────────────


def fine_tune_d3pm(
    diffusion: DiscreteD3PM,
    denoiser: DenoisingNetwork,
    hard_targets: List[Dict],
    config: RefinementConfig,
) -> Dict[str, float]:
    """Fine-tune the D3PM denoiser on collected hard targets.

    Uses the same loss pattern as DiffusionTrainer._compute_loss() but with:
    - Targets at masked positions = realized tokens (not GT)
    - Lower learning rate (2e-5 vs 1e-4)
    - Fewer epochs (3)
    - Gradient clipping

    Returns:
        Dict with final training metrics.
    """
    ft_config = config.fine_tuning
    device = config.device

    if not hard_targets:
        logger.warning("No hard targets to fine-tune on. Skipping.")
        return {"loss": float("nan"), "num_samples": 0}

    logger.info(
        f"Fine-tuning on {len(hard_targets)} hard targets: "
        f"lr={ft_config.learning_rate}, epochs={ft_config.num_epochs}, "
        f"batch_size={ft_config.batch_size}"
    )

    # Set up optimizer (only denoiser is trainable)
    optimizer = AdamW(
        denoiser.parameters(),
        lr=ft_config.learning_rate,
        weight_decay=ft_config.weight_decay,
    )

    denoiser.train()

    keys = sorted(hard_targets[0]["tokens"].keys())
    total_loss = 0.0
    num_steps = 0

    for epoch in range(ft_config.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        # Simple batching over hard targets
        for batch_start in range(0, len(hard_targets), ft_config.batch_size):
            batch_items = hard_targets[batch_start : batch_start + ft_config.batch_size]
            B = len(batch_items)

            # Stack tokens into batched tensors
            tokens_batch = {
                key: torch.cat([item["tokens"][key] for item in batch_items], dim=0).to(device)
                for key in keys
            }
            observed_batch = {
                key: torch.cat([item["observed"][key] for item in batch_items], dim=0).to(device)
                for key in keys
            }
            target_batch = {
                key: torch.cat([item["target"][key] for item in batch_items], dim=0).to(device)
                for key in keys
            }

            # Sample random timesteps
            t = torch.randint(
                0, diffusion.schedule.num_timesteps, (B,), device=device
            )

            # Forward diffusion
            noisy_tokens, _ = diffusion.forward_process(
                tokens_batch, t, mask_dict=target_batch
            )

            # Predict clean tokens
            predicted_logits = denoiser(
                noisy_tokens, t, observed_dict=observed_batch
            )

            # Compute loss on target positions (same pattern as DiffusionTrainer)
            loss = _compute_refinement_loss(
                predicted_logits, tokens_batch, target_batch
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                denoiser.parameters(), ft_config.gradient_clip_norm
            )

            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            num_steps += 1

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        total_loss += avg_epoch_loss
        logger.info(f"  Fine-tune epoch {epoch + 1}/{ft_config.num_epochs}: loss={avg_epoch_loss:.4f}")

    return {
        "loss": total_loss / max(ft_config.num_epochs, 1),
        "num_samples": len(hard_targets),
        "num_steps": num_steps,
    }


def _compute_refinement_loss(
    predicted_logits: Dict[str, torch.Tensor],
    target_tokens: Dict[str, torch.Tensor],
    target_mask: Dict[str, torch.BoolTensor],
) -> torch.Tensor:
    """Cross-entropy loss on target (masked) positions.

    Simplified version of DiffusionTrainer._compute_loss() without SNR/vocab
    weighting — refinement is a focused fine-tuning step.
    """
    B = next(iter(predicted_logits.values())).shape[0]
    device = next(iter(predicted_logits.values())).device
    per_sample_loss = torch.zeros(B, device=device)
    per_sample_count = torch.zeros(B, device=device)

    for key in predicted_logits:
        logits = predicted_logits[key]      # [B, V]
        targets = target_tokens[key]        # [B]
        mask = target_mask[key].float()     # [B]

        loss = F.cross_entropy(logits, targets, reduction="none")  # [B]
        per_sample_loss = per_sample_loss + loss * mask
        per_sample_count = per_sample_count + mask

    per_sample_loss = per_sample_loss / per_sample_count.clamp(min=1.0)
    return per_sample_loss.mean()


# ── Checkpointing ────────────────────────────────────────────────────────────


def save_refinement_checkpoint(
    diffusion: DiscreteD3PM,
    denoiser: DenoisingNetwork,
    cycle: int,
    metrics: Dict,
    config: RefinementConfig,
    output_dir: Path,
):
    """Save checkpoint after a refinement cycle."""
    checkpoint = {
        "cycle": cycle,
        "denoiser_state_dict": denoiser.state_dict(),
        "diffusion_state_dict": diffusion.state_dict(),
        "metrics": metrics,
        "refinement_config": config.model_dump(),
    }
    path = output_dir / f"refinement_cycle_{cycle}.pt"
    torch.save(checkpoint, path)
    logger.info(f"Refinement checkpoint saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args):
    """Main refinement entry point."""
    # Load config
    config = load_experiment_config(args.config, RefinementConfig)

    # CLI overrides
    if args.max_samples is not None:
        config = config.model_copy(update={"max_samples": args.max_samples})
    if args.device is not None:
        config = config.model_copy(update={"device": args.device})

    # Set seed
    torch.manual_seed(config.seed)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("D3PM Offline Hard-Target Refinement")
    logger.info("=" * 60)
    logger.info(f"  D3PM checkpoint:     {config.d3pm_checkpoint}")
    logger.info(f"  MNO checkpoint:      {config.mno_checkpoint}")
    logger.info(f"  Tokenizer checkpoint:{config.tokenizer_checkpoint}")
    logger.info(f"  CVAE checkpoint:     {config.cvae_checkpoint}")
    logger.info(f"  Dataset:             {config.dataset_path}")
    logger.info(f"  Refinement cycles:   {config.num_refinement_cycles}")
    logger.info(f"  Mask probability:    {config.mask_probability}")
    logger.info(f"  Quality threshold:   {config.quality_filter.min_observed_agreement}")
    logger.info(f"  Device:              {config.device}")

    # Load models
    logger.info("\nLoading models...")
    diffusion, denoiser, _ = load_d3pm_and_denoiser(
        config.d3pm_checkpoint, config.device
    )
    mno = load_mno(config.mno_checkpoint, config.device)
    tokenizer = load_tokenizer(config.tokenizer_checkpoint)

    # Load CVAE for temporal-token → (theta, IC) decoding
    if config.cvae_checkpoint is None:
        raise ValueError(
            "cvae_checkpoint is required in RefinementConfig. "
            "Train a CVAE first: spinlock train-cvae --config configs/token_conditioned_cvae.yaml"
        )
    cvae = load_cvae(config.cvae_checkpoint, config.device)

    # Load dataset with GT temporal features for temporal-only tokenization
    logger.info("\nLoading dataset...")
    dataset = SpinlockDataset(
        config.dataset_path,
        max_samples=config.max_samples,
        load_gt_temporal_features=True,
    )
    logger.info(f"Dataset: {len(dataset)} samples")

    # Refinement loop
    all_metrics = []
    for cycle in range(config.num_refinement_cycles):
        logger.info(f"\n{'='*60}")
        logger.info(f"Refinement Cycle {cycle + 1}/{config.num_refinement_cycles}")
        logger.info(f"{'='*60}")

        # Generate hard targets
        logger.info("\nGenerating hard targets...")
        hard_targets = generate_hard_targets(
            dataset, diffusion, denoiser, mno, tokenizer, cvae, config
        )

        # Fine-tune
        logger.info("\nFine-tuning D3PM on hard targets...")
        ft_metrics = fine_tune_d3pm(diffusion, denoiser, hard_targets, config)
        all_metrics.append({
            "cycle": cycle + 1,
            "num_accepted": len(hard_targets),
            **ft_metrics,
        })

        # Save checkpoint
        save_refinement_checkpoint(
            diffusion, denoiser, cycle + 1, ft_metrics, config, output_dir
        )

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Refinement Complete")
    logger.info(f"{'='*60}")
    for m in all_metrics:
        logger.info(
            f"  Cycle {m['cycle']}: accepted={m['num_accepted']}, "
            f"loss={m['loss']:.4f}"
        )

    # Save final checkpoint
    final_path = output_dir / "refinement_final.pt"
    torch.save(
        {
            "denoiser_state_dict": denoiser.state_dict(),
            "diffusion_state_dict": diffusion.state_dict(),
            "all_metrics": all_metrics,
            "refinement_config": config.model_dump(),
        },
        final_path,
    )
    logger.info(f"\nFinal refined model saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline hard-target refinement for D3PM"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to refinement config YAML",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max_samples (for smoke tests)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., 'cpu' for smoke tests)",
    )
    args = parser.parse_args()
    main(args)
