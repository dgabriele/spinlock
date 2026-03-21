"""Offline hard-target refinement loop for D3PM (inverse generation).

Generates novel Sobol parameters, rolls out GT trajectories, then tests the
D3PM's ability to infer (theta, IC) from observed temporal tokens. Accepted
proposals (where D3PM's dreamed params reproduce the dynamics) become hard
targets for fine-tuning.

Pipeline per proposal:
    1. Sample new Sobol params → Fourier IC → GT rollout → tokenize all families
    2. Give temporal tokens to D3PM as observed, mask theta+IC
    3. D3PM inpaints theta+IC (inverse problem: dynamics → causes)
    4. Decode predicted theta+IC → rollout → retokenize temporal
    5. Quality filter: temporal agreement ≥ threshold
    6. Fine-tune D3PM on accepted hard targets

Usage:
    poetry run python experiments/diffusion/scripts/refine_d3pm.py \
        --config experiments/diffusion/configs/v8_refinement.yaml

    # Resume from last completed cycle:
    poetry run python experiments/diffusion/scripts/refine_d3pm.py \
        --config experiments/diffusion/configs/v8_refinement.yaml --resume
"""

import argparse
import copy
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from spinlock.experimental.common.config.loader import load_experiment_config
from spinlock.experimental.diffusion.config import RefinementConfig
from spinlock.experimental.diffusion.refinement import (
    AdaptiveRefinementSearch,
    PrioritizedReplayBuffer,
)
from spinlock.experimental.diffusion.models import (
    DenoisingNetwork,
    DiffusionSchedule,
    DiscreteD3PM,
)
from spinlock.rollout.provider import build_rollout_provider, RolloutProvider
from spinlock.tokens.token_decoder import IntegratedTokenDecoder
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
) -> Tuple[DiscreteD3PM, DenoisingNetwork, dict, "Optional[TokenFilter]"]:
    """Load trained D3PM + denoiser from checkpoint.

    Returns:
        (diffusion_model, denoising_network, checkpoint_dict, token_filter)
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

    # Apply entropy filter if training used it
    token_filter = None
    if getattr(config.dataset, 'entropy_filter', False):
        from spinlock.experimental.diffusion.data.token_filter import TokenFilter
        token_filter = TokenFilter.from_pretokenized(
            str(config.dataset.tokenized_path),
            truncation_length=config.dataset.truncation_length,
        )
        vocab_sizes = token_filter.filter_vocab_sizes(vocab_sizes)
        category_level_info = token_filter.filter_category_level_info(category_level_info)

    # Reconstruct diffusion model
    graded_cfg = config.diffusion.graded_schedule
    scale_factors = graded_cfg.scale_factors or {}
    if graded_cfg.position_scale_factors_path:
        import json

        with open(graded_cfg.position_scale_factors_path) as f:
            scale_factors = json.load(f)

    if token_filter is not None:
        scale_factors = {k: v for k, v in scale_factors.items() if k in token_filter._active_set}

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
        family_scale_overrides=graded_cfg.family_scale_overrides,
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
        + (f", entropy_filter={len(token_filter.active_keys)} active" if token_filter else "")
    )
    return diffusion, denoiser, ckpt, token_filter


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


# ── Fine-tuning ──────────────────────────────────────────────────────────────


def fine_tune_d3pm(
    diffusion: DiscreteD3PM,
    denoiser: DenoisingNetwork,
    hard_targets: List[Dict],
    config: RefinementConfig,
    anchor_params: Optional[Dict[str, torch.Tensor]] = None,
    replay_buffer: Optional[PrioritizedReplayBuffer] = None,
    cycle: int = 0,
) -> Dict[str, float]:
    """Fine-tune the D3PM denoiser on collected hard targets.

    All token positions (temporal + initial + theta) are trained.

    v12 surprise-driven training:
    - Three-level surprise hierarchy:
      Level 1: PrioritizedReplayBuffer — hard targets replayed more often
      Level 2: Surprise-weighted loss — geometric mean of agreement-surprise
               and CE-loss-surprise per sample
      Level 3: Focal loss (per-position, unchanged)
    - Plus v11 stabilization: anchor, cosine LR, per-cycle decay

    Returns:
        Dict with final training metrics.
    """
    ft_config = config.fine_tuning
    device = config.device

    if not hard_targets:
        logger.warning("No hard targets to fine-tune on. Skipping.")
        return {"loss": float("nan"), "anchor_loss": 0.0, "num_samples": 0}

    # ── Replay mixing ─────────────────────────────────────────────────────
    if replay_buffer and len(replay_buffer) > 0 and ft_config.replay_fraction > 0:
        n_replay = int(
            len(hard_targets) * ft_config.replay_fraction / (1 - ft_config.replay_fraction)
        )
        replay_samples = replay_buffer.sample(n_replay)
        training_targets = hard_targets + replay_samples
        random.shuffle(training_targets)
        logger.info(
            f"  Replay mixing: {len(hard_targets)} new + {len(replay_samples)} replay "
            f"= {len(training_targets)} total"
        )
    else:
        training_targets = hard_targets

    # ── Per-cycle LR decay ────────────────────────────────────────────────
    effective_lr = ft_config.learning_rate * (ft_config.per_cycle_lr_decay ** cycle)

    logger.info(
        f"Fine-tuning on {len(training_targets)} targets (cycle {cycle + 1}): "
        f"lr={effective_lr:.2e}, epochs={ft_config.num_epochs}, "
        f"batch_size={ft_config.batch_size}"
        + (f", anchor_weight={ft_config.anchor_weight}" if ft_config.anchor_weight > 0 else "")
    )

    optimizer = AdamW(
        denoiser.parameters(),
        lr=effective_lr,
        weight_decay=ft_config.weight_decay,
    )

    # ── Cosine schedule with warmup ───────────────────────────────────────
    scheduler = None
    if ft_config.use_cosine_schedule:
        steps_per_epoch = max(1, len(training_targets) // ft_config.batch_size)
        total_steps = steps_per_epoch * ft_config.num_epochs
        warmup_steps = int(total_steps * ft_config.warmup_fraction)
        cosine_steps = max(1, total_steps - warmup_steps)
        min_lr_frac = ft_config.min_lr_fraction

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return 0.1 + 0.9 * (step / max(1, warmup_steps))
            progress = (step - warmup_steps) / cosine_steps
            return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

    denoiser.train()

    keys = sorted(training_targets[0]["tokens"].keys())
    total_ce_loss = 0.0
    total_anchor_loss = 0.0
    total_mean_surprise = 0.0
    num_steps = 0

    for epoch in range(ft_config.num_epochs):
        epoch_ce_loss = 0.0
        epoch_anchor_loss = 0.0
        epoch_mean_surprise = 0.0
        epoch_steps = 0

        # Re-shuffle each epoch
        epoch_targets = training_targets.copy()
        random.shuffle(epoch_targets)

        for batch_start in range(0, len(epoch_targets), ft_config.batch_size):
            batch_items = epoch_targets[batch_start : batch_start + ft_config.batch_size]
            B = len(batch_items)

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

            t = torch.randint(
                0, diffusion.schedule.num_timesteps, (B,), device=device
            )

            noisy_tokens, _ = diffusion.forward_process(
                tokens_batch, t, mask_dict=target_batch
            )

            predicted_logits = denoiser(
                noisy_tokens, t, observed_dict=observed_batch
            )

            per_sample_loss = _compute_refinement_loss(
                predicted_logits, tokens_batch, target_batch,
            )

            # ── Level 2: Agreement-surprise sample weighting ─────────────
            # Uses static agreement signal only. The dynamic CE loss signal
            # is confounded by the random diffusion timestep (t~U[0,T]) and
            # adds noise rather than useful difficulty information.
            with torch.no_grad():
                agreements = torch.tensor(
                    [item["agreement"] for item in batch_items],
                    device=device, dtype=torch.float32,
                )
                agreement_surprise = 1.0 - agreements  # [B]
                sample_weights = agreement_surprise / agreement_surprise.mean().clamp(min=1e-6)
                sample_weights = sample_weights.clamp(max=ft_config.max_surprise_weight)

            ce_loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()

            # ── Anchor regularization ─────────────────────────────────────
            anchor_loss_val = 0.0
            if anchor_params is not None and ft_config.anchor_weight > 0:
                anchor_loss = sum(
                    (p - anchor_params[name]).pow(2).sum()
                    for name, p in denoiser.named_parameters()
                    if p.requires_grad and name in anchor_params
                )
                loss = ce_loss + ft_config.anchor_weight * anchor_loss
                anchor_loss_val = anchor_loss.item()
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                denoiser.parameters(), ft_config.gradient_clip_norm
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_ce_loss += ce_loss.item()
            epoch_anchor_loss += anchor_loss_val
            epoch_mean_surprise += sample_weights.mean().item()
            epoch_steps += 1
            num_steps += 1

        avg_ce = epoch_ce_loss / max(epoch_steps, 1)
        avg_anchor = epoch_anchor_loss / max(epoch_steps, 1)
        avg_surprise = epoch_mean_surprise / max(epoch_steps, 1)
        total_ce_loss += avg_ce
        total_anchor_loss += avg_anchor
        total_mean_surprise += avg_surprise

        current_lr = optimizer.param_groups[0]["lr"]
        anchor_str = f", anchor={avg_anchor:.4f}" if ft_config.anchor_weight > 0 else ""
        logger.info(
            f"  Fine-tune epoch {epoch + 1}/{ft_config.num_epochs}: "
            f"ce={avg_ce:.4f}{anchor_str}, surprise_w={avg_surprise:.2f}, lr={current_lr:.2e}"
        )

    return {
        "loss": total_ce_loss / max(ft_config.num_epochs, 1),
        "anchor_loss": total_anchor_loss / max(ft_config.num_epochs, 1),
        "mean_surprise_weight": total_mean_surprise / max(ft_config.num_epochs, 1),
        "effective_lr": effective_lr,
        "num_samples": len(training_targets),
        "num_new_samples": len(hard_targets),
        "num_replay_samples": len(training_targets) - len(hard_targets),
        "num_steps": num_steps,
    }


def _compute_refinement_loss(
    predicted_logits: Dict[str, torch.Tensor],
    target_tokens: Dict[str, torch.Tensor],
    target_mask: Dict[str, torch.BoolTensor],
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Per-sample focal cross-entropy loss on target (masked) positions.

    Returns per-sample losses [B] — caller applies surprise weighting
    and reduces. Uses focal loss (Level 3 of the surprise hierarchy)
    to avoid wasting gradient on positions the model already predicts
    with high confidence.
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

        # Focal weighting: down-weight easy predictions (Level 3)
        if focal_gamma > 0:
            with torch.no_grad():
                p_t = F.softmax(logits, dim=-1)
                p_correct = p_t.gather(1, targets.unsqueeze(1)).squeeze(1)
                focal_weight = (1 - p_correct) ** focal_gamma
            loss = loss * focal_weight

        per_sample_loss = per_sample_loss + loss * mask
        per_sample_count = per_sample_count + mask

    per_sample_loss = per_sample_loss / per_sample_count.clamp(min=1.0)
    return per_sample_loss  # [B] — caller applies Level 2 weighting


# ── Checkpointing ────────────────────────────────────────────────────────────


def save_refinement_checkpoint(
    diffusion: DiscreteD3PM,
    denoiser: DenoisingNetwork,
    cycle: int,
    metrics: Dict,
    config: RefinementConfig,
    output_dir: Path,
    replay_buffer: Optional[PrioritizedReplayBuffer] = None,
    best_eval_agreement: float = 0.0,
    patience_counter: int = 0,
    best_cycle: int = 0,
):
    """Save checkpoint after a refinement cycle (includes early stopping + replay state)."""
    checkpoint = {
        "cycle": cycle,
        "denoiser_state_dict": denoiser.state_dict(),
        "diffusion_state_dict": diffusion.state_dict(),
        "metrics": metrics,
        "refinement_config": config.model_dump(),
        # v11 early stopping state
        "best_eval_agreement": best_eval_agreement,
        "patience_counter": patience_counter,
        "best_cycle": best_cycle,
    }
    if replay_buffer is not None:
        checkpoint["replay_buffer"] = replay_buffer.state_dict()
    path = output_dir / f"refinement_cycle_{cycle}.pt"
    torch.save(checkpoint, path)
    logger.info(f"Refinement checkpoint saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args):
    """Main refinement entry point."""
    config = load_experiment_config(args.config, RefinementConfig)

    if args.max_samples is not None:
        config = config.model_copy(update={"max_samples": args.max_samples})
    if args.device is not None:
        config = config.model_copy(update={"device": args.device})

    torch.manual_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ft_config = config.fine_tuning

    logger.info("=" * 60)
    logger.info("D3PM Offline Hard-Target Refinement (v12 Pool-Based Retrain)")
    logger.info("=" * 60)
    logger.info(f"  D3PM checkpoint:     {config.d3pm_checkpoint}")
    logger.info(f"  Rollout source:      {config.mno_checkpoint or 'GT simulator'}")
    logger.info(f"  Tokenizer checkpoint:{config.tokenizer_checkpoint}")
    logger.info(f"  Dataset:             {config.dataset_path}")
    logger.info(f"  Refinement cycles:   {config.num_refinement_cycles}")
    logger.info(f"  Mask probability:    {config.mask_probability}")
    logger.info(f"  Quality threshold:   {config.quality_filter.min_observed_agreement}")
    logger.info(f"  D3PM candidates:     {config.adaptive.initial_d3pm_candidates} (adaptive initial)")
    logger.info(f"  Sampling temperature:{config.sampling_temperature}")
    logger.info(f"  Adaptive:            perturbation={'ON' if config.adaptive.perturbation.enabled else 'OFF'}, "
                f"max_rounds={config.adaptive.stopping.max_rounds_per_sample}, "
                f"max_extra={config.adaptive.budget.max_extra_candidates}")
    # v11 stabilization log
    logger.info(f"  Anchor weight:       {ft_config.anchor_weight}")
    logger.info(f"  Replay fraction:     {ft_config.replay_fraction} (max {ft_config.max_replay_size})")
    logger.info(f"  Cosine schedule:     {ft_config.use_cosine_schedule}")
    logger.info(f"  Per-cycle LR decay:  {ft_config.per_cycle_lr_decay}")
    logger.info(f"  Early stopping:      patience={config.early_stopping_patience}")
    # v12 surprise log
    logger.info(f"  Surprise alpha:      {ft_config.surprise_alpha}")
    logger.info(f"  Max surprise weight: {ft_config.max_surprise_weight}")
    logger.info(f"  Device:              {config.device}")

    # Load models
    logger.info("\nLoading models...")
    diffusion, denoiser, _, token_filter = load_d3pm_and_denoiser(
        config.d3pm_checkpoint, config.device
    )
    rollout_provider = build_rollout_provider(
        mno_checkpoint=config.mno_checkpoint,
        tokenizer_checkpoint=config.tokenizer_checkpoint,
        device=config.device,
        dataset_config_path=config.dataset_config_path,
    )
    tokenizer = load_tokenizer(config.tokenizer_checkpoint)
    tokenizer.model.to(config.device)
    decoder = IntegratedTokenDecoder(tokenizer)

    # ── Base weights: snapshot for retrain-from-base each cycle ─────────
    # Instead of incremental fine-tuning (which causes compounding drift),
    # we reset to the base checkpoint before each fine-tuning round and
    # train on the full accumulated target pool.
    base_denoiser_state = {
        k: v.detach().clone() for k, v in denoiser.state_dict().items()
    }
    base_diffusion_state = {
        k: v.detach().clone() for k, v in diffusion.state_dict().items()
    }
    logger.info(f"Base weights captured for retrain-from-base")

    # ── Target pool (Level 1: prioritized by agreement-surprise) ─────────
    replay_buffer = PrioritizedReplayBuffer(
        ft_config.max_replay_size, alpha=ft_config.surprise_alpha,
    )

    # ── Early stopping state ──────────────────────────────────────────────
    best_eval_agreement = 0.0
    patience_counter = 0
    best_cycle = 0

    # Detect completed cycles for resume
    start_cycle = 0
    if args.resume:
        existing = sorted(output_dir.glob("refinement_cycle_*.pt"))
        if existing:
            last_cycle_ckpt = existing[-1]
            last_cycle_num = int(last_cycle_ckpt.stem.split("_")[-1])
            logger.info(f"Resuming: found {len(existing)} cycle checkpoints, last=cycle_{last_cycle_num}")
            ckpt = torch.load(last_cycle_ckpt, map_location=config.device, weights_only=False)
            # Load fine-tuned model for inference in next cycle
            denoiser.load_state_dict(ckpt["denoiser_state_dict"])
            diffusion.load_state_dict(ckpt["diffusion_state_dict"])
            # Restore target pool
            if "replay_buffer" in ckpt:
                replay_buffer.load_state_dict(ckpt["replay_buffer"])
                logger.info(f"  Restored target pool: {len(replay_buffer)} items")
            if "best_eval_agreement" in ckpt:
                best_eval_agreement = ckpt["best_eval_agreement"]
                patience_counter = ckpt["patience_counter"]
                best_cycle = ckpt["best_cycle"]
                logger.info(
                    f"  Restored early stopping: best={best_eval_agreement:.4f} "
                    f"(cycle {best_cycle}), patience={patience_counter}"
                )
            start_cycle = last_cycle_num
            logger.info(f"Loaded weights from {last_cycle_ckpt}, starting at cycle {start_cycle + 1}")

    # Create adaptive search (reused across cycles for improvement tracking)
    search = AdaptiveRefinementSearch(
        diffusion, denoiser, rollout_provider, tokenizer, decoder, config,
        token_filter=token_filter,
    )

    # Held-out evaluation set (generated once, reused across cycles)
    eval_gt_data = []
    if config.eval_samples > 0:
        eval_gt_data = search.generate_eval_set(config.eval_samples)
        if eval_gt_data:
            logger.info("\nBaseline evaluation (before any fine-tuning):")
            baseline_metrics = search.evaluate_model(eval_gt_data)
            # Initialize best from baseline if resuming from scratch
            if best_eval_agreement == 0.0:
                best_eval_agreement = baseline_metrics.get("mean_agreement", 0.0)
        else:
            baseline_metrics = {}
    else:
        baseline_metrics = {}

    # Refinement loop
    all_metrics = []
    for cycle in range(start_cycle, config.num_refinement_cycles):
        logger.info(f"\n{'='*60}")
        logger.info(f"Refinement Cycle {cycle + 1}/{config.num_refinement_cycles}")
        logger.info(f"{'='*60}")

        logger.info("\nGenerating hard targets (adaptive)...")
        hard_targets = search.generate_targets(cycle=cycle)

        # Accumulate in target pool
        replay_buffer.add(hard_targets)
        logger.info(f"Target pool: {len(replay_buffer)} items total")

        # ── Retrain from base on accumulated pool ─────────────────────
        # Reset to v8 base weights — eliminates compounding drift.
        # Each cycle trains from scratch on the full pool, so later cycles
        # benefit from more data without inheriting errors from earlier cycles.
        denoiser.load_state_dict(base_denoiser_state)
        diffusion.load_state_dict(base_diffusion_state)

        # Sample from pool (priority-weighted: hard targets overrepresented)
        n_train = min(len(replay_buffer), 2000)
        training_targets = replay_buffer.sample(n_train)
        logger.info(
            f"  Retrain from base: {n_train} targets sampled from pool "
            f"({len(hard_targets)} new this cycle)"
        )

        logger.info("\nFine-tuning D3PM on target pool...")
        ft_metrics = fine_tune_d3pm(
            diffusion, denoiser, training_targets, config,
            anchor_params=None,   # no anchor needed — always starting from base
            replay_buffer=None,   # no replay mixing — pool IS the training set
            cycle=0,              # always cycle 0 LR (no decay, fresh start)
        )

        # Held-out evaluation
        eval_metrics = {}
        if eval_gt_data and (cycle + 1) % config.eval_frequency == 0:
            logger.info("\nEvaluating on held-out set...")
            eval_metrics = search.evaluate_model(eval_gt_data)

        all_metrics.append({
            "cycle": cycle + 1,
            "num_targets": len(hard_targets),
            **ft_metrics,
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
        })

        # ── Early stopping ────────────────────────────────────────────────
        current_agreement = eval_metrics.get("mean_agreement", 0.0)
        if eval_metrics and config.early_stopping_patience > 0:
            if current_agreement > best_eval_agreement:
                best_eval_agreement = current_agreement
                patience_counter = 0
                best_cycle = cycle + 1
                logger.info(
                    f"  New best eval agreement: {best_eval_agreement:.4f} (cycle {best_cycle})"
                )
            else:
                patience_counter += 1
                logger.info(
                    f"  No improvement: {current_agreement:.4f} <= {best_eval_agreement:.4f} "
                    f"(patience {patience_counter}/{config.early_stopping_patience})"
                )

        save_refinement_checkpoint(
            diffusion, denoiser, cycle + 1, ft_metrics, config, output_dir,
            replay_buffer=replay_buffer,
            best_eval_agreement=best_eval_agreement,
            patience_counter=patience_counter,
            best_cycle=best_cycle,
        )

        # Check early stopping after saving (so we always have the latest checkpoint)
        if (
            config.early_stopping_patience > 0
            and patience_counter >= config.early_stopping_patience
        ):
            logger.info(
                f"\nEarly stopping triggered: no improvement for {patience_counter} cycles. "
                f"Best was cycle {best_cycle} with agreement {best_eval_agreement:.4f}"
            )
            # Load best cycle checkpoint
            best_ckpt_path = output_dir / f"refinement_cycle_{best_cycle}.pt"
            if best_ckpt_path.exists():
                best_ckpt = torch.load(best_ckpt_path, map_location=config.device, weights_only=False)
                denoiser.load_state_dict(best_ckpt["denoiser_state_dict"])
                diffusion.load_state_dict(best_ckpt["diffusion_state_dict"])
                logger.info(f"Restored best model from cycle {best_cycle}")
            break

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Refinement Complete")
    logger.info(f"{'='*60}")
    if baseline_metrics:
        logger.info(
            f"  Baseline: eval_agreement={baseline_metrics.get('mean_agreement', 0):.4f}, "
            f"eval_accept={baseline_metrics.get('acceptance_rate', 0):.3f}"
        )
    for m in all_metrics:
        eval_str = ""
        if "eval_mean_agreement" in m:
            eval_str = (
                f", eval_agreement={m['eval_mean_agreement']:.4f}"
                f", eval_accept={m['eval_acceptance_rate']:.3f}"
            )
        anchor_str = ""
        if m.get("anchor_loss", 0) > 0:
            anchor_str = f", anchor={m['anchor_loss']:.4f}"
        diversity_str = ""
        if "eval_mean_pairwise_hamming" in m:
            diversity_str = (
                f", hamming={m['eval_mean_pairwise_hamming']:.3f}"
                f", unique={m['eval_mean_unique_candidates']:.1f}"
                f", collapsed={m['eval_frac_fully_collapsed']:.2f}"
            )
        logger.info(
            f"  Cycle {m['cycle']}: targets={m.get('num_new_samples', m['num_targets'])}"
            f"+{m.get('num_replay_samples', 0)}replay, "
            f"loss={m['loss']:.4f}{anchor_str}"
            f", lr={m.get('effective_lr', ft_config.learning_rate):.2e}"
            f"{eval_str}{diversity_str}"
        )
    if config.early_stopping_patience > 0 and best_cycle > 0:
        logger.info(f"  Best cycle: {best_cycle} (agreement={best_eval_agreement:.4f})")

    final_path = output_dir / "refinement_final.pt"
    torch.save({
        "denoiser_state_dict": denoiser.state_dict(),
        "diffusion_state_dict": diffusion.state_dict(),
        "all_metrics": all_metrics,
        "baseline_eval_metrics": baseline_metrics,
        "refinement_config": config.model_dump(),
    }, final_path)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed cycle checkpoint in output_dir.",
    )
    args = parser.parse_args()
    main(args)
