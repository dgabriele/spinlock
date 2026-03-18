"""Offline hard-target refinement loop for D3PM.

Closed loop: D3PM inpaint → IntegratedTokenDecoder → rollout → retokenize →
quality filter → fine-tune.

The D3PM generates ALL token positions (temporal + initial + theta). Observed
temporal tokens are fixed; initial + theta positions are inpainted. The
IntegratedTokenDecoder (codebook lookup + inverse heads) replaces the CVAE.
Diversity comes from D3PM's stochastic denoising trajectories.

    1. Tokenize dataset → ALL tokens (temporal + initial + theta)
    2. Mask: fix temporal positions, mask initial + theta positions
    3. D3PM inpaint masked positions
    4. IntegratedTokenDecoder.decode(completed_tokens) → (theta, IC)
    5. Rollout from decoded params → realized trajectory
    6. Retokenize realized trajectory → realized_tokens
    7. Quality filter: observed-position agreement check
    8. Fine-tune D3PM on accepted hard targets (all positions)

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from spinlock.data import SpinlockDataset
from spinlock.experimental.common.config.loader import load_experiment_config
from spinlock.experimental.diffusion.config import RefinementConfig
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


# ── Hard target generation ───────────────────────────────────────────────────


def generate_random_mask(
    keys: List[str],
    temporal_keys: set,
    batch_size: int,
    mask_probability: float,
    device: str,
) -> Tuple[Dict[str, torch.BoolTensor], Dict[str, torch.BoolTensor]]:
    """Generate observed/target masks for inpainting.

    Temporal positions: randomly masked (some observed, some targets).
    Initial + theta positions: always masked (always targets for inpainting).

    Returns:
        (observed_dict, target_dict) — both map key → [B] boolean.
    """
    observed = {}
    target = {}
    for key in keys:
        if key in temporal_keys:
            # Temporal: random masking
            mask = torch.rand(batch_size, device=device) < mask_probability
            target[key] = mask
            observed[key] = ~mask
        else:
            # Initial/theta: always masked (target for inpainting)
            target[key] = torch.ones(batch_size, dtype=torch.bool, device=device)
            observed[key] = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return observed, target


def generate_hard_targets(
    dataset: SpinlockDataset,
    diffusion: DiscreteD3PM,
    denoiser: DenoisingNetwork,
    rollout_provider: RolloutProvider,
    tokenizer: VQTokenizer,
    decoder: IntegratedTokenDecoder,
    config: RefinementConfig,
) -> List[Dict]:
    """Generate hard targets from the full refinement pipeline.

    D3PM generates ALL token positions (temporal + initial + theta).
    Multiple denoising passes provide diversity (different trajectories →
    different completions). Best candidate per sample is kept.

    Returns:
        List of accepted hard-target dicts.
    """
    device = config.device
    threshold = config.quality_filter.min_observed_agreement
    hard_targets = []
    total_samples = 0
    total_accepted = 0
    agreement_sum = 0.0
    max_accept = config.max_accepted_targets
    gen_bs = config.generation_batch_size

    # Discover all keys and temporal keys from tokenizer
    schema = TokenSchema.from_tokenizer(tokenizer)
    temporal_keys = set(schema.keys_for_family("temporal"))
    all_keys = sorted(schema.vocab_sizes_dict().keys())

    N = len(dataset)
    done = False

    D3PM_BS = 512

    # ── Prefetch helper: CPU-only data collation ────────────────

    def _collate_batch(start: int, end: int):
        """Load samples from HDF5 and stack tensors on CPU."""
        gt_temporals = []
        ics = []
        params_list = []
        has_gt_features = True

        for idx in range(start, end):
            sample = dataset[idx]
            ics.append(sample["ic"])
            params_list.append(sample["params"])
            gt = sample.get("gt_raw_temporal")
            if gt is None:
                has_gt_features = False
            gt_temporals.append(gt)

        ics_cpu = torch.stack(ics)
        params_cpu = torch.stack(params_list)
        gt_cpu = torch.stack(gt_temporals) if has_gt_features else None
        return ics_cpu, params_cpu, gt_cpu, has_gt_features, end - start

    # ── Super-batch loop ──────────────────────────────────────────
    super_ranges = [
        (s, min(s + D3PM_BS, N)) for s in range(0, N, D3PM_BS)
    ]

    with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        first_end = min(gen_bs, N)
        pending_future = prefetch_pool.submit(_collate_batch, 0, first_end)

        for si, (super_start, super_end) in enumerate(super_ranges):
            if done:
                break

            # ══ Phase 1a: Tokenize super-batch in mini-batches ════
            mini_ranges = [
                (s, min(s + gen_bs, super_end))
                for s in range(super_start, super_end, gen_bs)
            ]
            accumulated = []

            for mi, (mb_start, mb_end) in enumerate(mini_ranges):
                ics_cpu, params_cpu, gt_cpu, has_gt, B = pending_future.result()

                next_global = mb_end
                if next_global < N:
                    next_mb_end = min(next_global + gen_bs, N)
                    pending_future = prefetch_pool.submit(
                        _collate_batch, next_global, next_mb_end,
                    )

                ics_t = ics_cpu.to(device, non_blocking=True)
                params_t = params_cpu.to(device, non_blocking=True)

                # Step 1: Tokenize ALL families (temporal + initial + theta)
                with torch.no_grad():
                    if has_gt:
                        gt_batch = gt_cpu.to(device, non_blocking=True)
                        all_tokens = tokenizer.tokenize(
                            temporal_features=gt_batch,
                            theta_features=params_t if tokenizer.model.theta_dim > 0 else None,
                            initial_raw=ics_t if tokenizer.model.initial_dim > 0 else None,
                        )
                        del gt_batch
                    else:
                        conditioning = {"theta": params_t, "ic": ics_t}
                        trajectories = rollout_provider.rollout(
                            conditioning, steps=config.rollout_steps,
                        )
                        all_tokens = tokenizer.tokenize(
                            temporal_raw=trajectories,
                            theta_features=params_t if tokenizer.model.theta_dim > 0 else None,
                            initial_raw=ics_t if tokenizer.model.initial_dim > 0 else None,
                        )
                        del trajectories

                dataset_tokens = {
                    k: v.to(device) for k, v in all_tokens.items()
                    if k in all_keys
                }
                if dataset_tokens:
                    accumulated.append(dataset_tokens)

            if not accumulated:
                continue

            keys = sorted(accumulated[0].keys())
            chunk_tokens = {
                k: torch.cat([bt[k] for bt in accumulated])
                for k in keys
            }
            del accumulated
            chunk_N = chunk_tokens[keys[0]].shape[0]

            # ══ Phase 1b: Bulk mask generation ════════════════════
            chunk_observed, chunk_target = generate_random_mask(
                keys, temporal_keys, chunk_N, config.mask_probability, device,
            )

            # ══ Phase 2: D3PM diversity ensemble + decode + quality filter ═
            n_candidates = config.d3pm_n_candidates

            for p2_start in range(0, chunk_N, gen_bs):
                if done:
                    break
                p2_end = min(p2_start + gen_bs, chunk_N)
                B = p2_end - p2_start

                dataset_tokens_mb = {
                    k: v[p2_start:p2_end] for k, v in chunk_tokens.items()
                }
                observed_dict = {
                    k: v[p2_start:p2_end] for k, v in chunk_observed.items()
                }
                target_dict = {
                    k: v[p2_start:p2_end] for k, v in chunk_target.items()
                }

                best_agreement = [-1.0] * B
                best_realized: list = [None] * B
                first_agreement = [0.0] * B

                for candidate_idx in range(n_candidates):
                    # D3PM inpainting (stochastic — different each call)
                    with torch.no_grad():
                        completed_tokens = diffusion.sample(
                            batch_size=B,
                            observed_dict=observed_dict,
                            x_0_dict=dataset_tokens_mb,
                            denoising_network=denoiser,
                            device=device,
                            start_step=config.d3pm_start_step,
                        )

                    # Decode tokens → (theta, IC) via inverse heads
                    with torch.no_grad():
                        decoded = decoder.decode(completed_tokens)
                        theta_pred = decoded.get("theta")
                        u0_pred = decoded.get("grids")

                    if theta_pred is None or u0_pred is None:
                        logger.debug(
                            f"Candidate {candidate_idx}: decoder returned "
                            f"theta={theta_pred is not None}, grids={u0_pred is not None}"
                        )
                        continue

                    # Rollout from decoded params
                    with torch.no_grad():
                        conditioning = {
                            "theta": theta_pred,
                            "ic": u0_pred,
                            "token_indices": completed_tokens,
                        }
                        try:
                            trajectories = rollout_provider.rollout(
                                conditioning, steps=config.rollout_steps,
                            )
                        except Exception as e:
                            logger.debug(f"Candidate {candidate_idx}: rollout failed: {e}")
                            continue

                    # Retokenize all families from rollout
                    # (IC+theta passed to satisfy encoder, but only temporal used for scoring)
                    with torch.no_grad():
                        all_realized = tokenizer.tokenize(
                            temporal_raw=trajectories,
                            theta_features=theta_pred,
                            initial_raw=u0_pred,
                        )
                    del trajectories
                    realized_tokens = {
                        k: v.to(device) for k, v in all_realized.items()
                        if k in all_keys
                    }

                    # Score: temporal observed-position agreement ONLY
                    # IC/theta roundtrip is NOT scored — theta→behavior is many-to-many
                    # and IC is the diversity source. The right question is: "does
                    # the D3PM's (theta, IC) produce dynamics matching observed tokens?"
                    for b in range(B):
                        num_observed = 0
                        num_agree = 0
                        for key in keys:
                            if key not in temporal_keys or key not in realized_tokens:
                                continue
                            if observed_dict[key][b]:
                                num_observed += 1
                                if dataset_tokens_mb[key][b] == realized_tokens[key][b]:
                                    num_agree += 1
                        agreement = num_agree / max(num_observed, 1)

                        if candidate_idx == 0:
                            first_agreement[b] = agreement
                        if agreement > best_agreement[b]:
                            best_agreement[b] = agreement
                            # Store the complete token set (all families)
                            best_realized[b] = {
                                k: v[b:b+1].clone()
                                for k, v in completed_tokens.items()
                            }

                # Ensemble improvement logging
                if n_candidates > 1:
                    improvements = sum(
                        1 for b in range(B)
                        if best_agreement[b] > first_agreement[b]
                    )
                    logger.debug(
                        f"  Ensemble: {improvements}/{B} samples "
                        f"improved by best-of-{n_candidates}"
                    )

                # ── Quality filter + accept ───────────────────────
                for b in range(B):
                    if best_realized[b] is None:
                        total_samples += 1
                        continue

                    agreement = best_agreement[b]
                    agreement_sum += agreement
                    total_samples += 1

                    if agreement >= threshold:
                        # Build hard target: dataset tokens at observed, best at target
                        hard_target_tokens = {}
                        for key in keys:
                            if key in best_realized[b]:
                                # For target positions, use D3PM completion
                                # For observed positions, keep dataset values
                                if target_dict[key][b]:
                                    hard_target_tokens[key] = best_realized[b][key]
                                else:
                                    hard_target_tokens[key] = dataset_tokens_mb[key][b:b+1].clone()
                            else:
                                hard_target_tokens[key] = dataset_tokens_mb[key][b:b+1].clone()

                        hard_targets.append({
                            "tokens": hard_target_tokens,
                            "observed": {k: v[b:b+1] for k, v in observed_dict.items()},
                            "target": {k: v[b:b+1] for k, v in target_dict.items()},
                            "agreement": agreement,
                        })
                        total_accepted += 1

                        if max_accept is not None and total_accepted >= max_accept:
                            logger.info(
                                f"  Early stop: reached {max_accept} accepted targets "
                                f"after {super_start + p2_start + b + 1}/{N} samples"
                            )
                            done = True
                            break

                # Periodic logging
                global_processed = super_start + p2_end
                if global_processed % 100 < gen_bs or global_processed >= N:
                    target_str = f"/{max_accept}" if max_accept else ""
                    logger.info(
                        f"  Processed {global_processed}/{N}: "
                        f"accepted={total_accepted}{target_str}/{total_samples}"
                        f" ({100 * total_accepted / max(total_samples, 1):.1f}%"
                        f"), mean_agreement="
                        f"{agreement_sum / max(total_samples, 1):.3f}"
                    )

            del chunk_tokens, chunk_observed, chunk_target

    logger.info(
        f"Hard target generation complete: "
        f"{total_accepted}/{total_samples} accepted "
        f"({100 * total_accepted / max(total_samples, 1):.1f}%), "
        f"mean_agreement={agreement_sum / max(total_samples, 1):.3f}"
    )
    return hard_targets


# ── Novel-Sobol self-consistency generation ──────────────────────────────────


def generate_novel_sobol_targets(
    diffusion: DiscreteD3PM,
    denoiser: DenoisingNetwork,
    rollout_provider: RolloutProvider,
    tokenizer: VQTokenizer,
    decoder: IntegratedTokenDecoder,
    config: RefinementConfig,
    token_filter=None,
    cycle: int = 0,
) -> List[Dict]:
    """Generate hard targets from novel Sobol parameters + D3PM temporal completion.

    Samples new (theta, IC) from the same Sobol parameter space as the training
    dataset (different seed), tokenizes them, then has the D3PM predict temporal
    tokens via inpainting. Physics verifies: rollout from the real (theta, IC) →
    retokenize → check if D3PM's temporal prediction matches reality.

    This expands the D3PM's training distribution to novel configurations without
    reusing training data.

    Returns:
        List of accepted hard-target dicts.
    """
    from spinlock.lenia.params import (
        DEFAULT_RANGES, sobol_expected_dims, sobol_batch_to_tensors,
    )
    from spinlock.lenia.fourier_ic import FourierICGenerator, FourierICConfig
    from spinlock.sampling.sobol import StratifiedSobolSampler

    device = config.device
    threshold = config.quality_filter.min_observed_agreement
    hard_targets = []
    total_samples = 0
    total_accepted = 0
    all_agreements: List[float] = []
    max_accept = config.max_accepted_targets
    gen_bs = config.generation_batch_size
    n_candidates = config.d3pm_n_candidates

    schema = TokenSchema.from_tokenizer(tokenizer)
    full_temporal_keys = set(schema.keys_for_family("temporal"))
    full_keys = sorted(schema.vocab_sizes_dict().keys())

    # D3PM operates on active keys only when entropy-filtered
    if token_filter is not None:
        all_keys = sorted(token_filter.active_keys)
        temporal_keys = set(k for k in all_keys if k in full_temporal_keys)
    else:
        all_keys = full_keys
        temporal_keys = full_temporal_keys

    # Sobol sampler with different seed per cycle (avoids resampling same configs)
    n_channels = 3  # Lenia channels
    sobol_dim = sobol_expected_dims(n_channels, DEFAULT_RANGES)
    cycle_seed = config.seed + 95 + cycle * 1000
    sobol_sampler = StratifiedSobolSampler(
        dimensionality=sobol_dim, scramble=True, seed=cycle_seed,
    )

    # Fourier IC generator (theta-coherent ICs)
    ic_gen = FourierICGenerator(FourierICConfig())

    n_attempts = (max_accept or 5000) * 2
    done = False

    logger.info(
        f"Novel-Sobol generation: cycle={cycle}, seed={cycle_seed}, "
        f"up to {n_attempts} proposals, target {max_accept} accepted"
    )

    for batch_start in range(0, n_attempts, gen_bs):
        if done:
            break
        B = min(gen_bs, n_attempts - batch_start)

        # Step 1: Sample new Sobol unit vectors [0,1]^D
        # Dataset stores raw unit vectors as theta — tokenizer was trained on these
        unit_vecs = sobol_sampler.sample(B)  # [B, D] numpy
        unit_vecs_t = torch.from_numpy(unit_vecs).float()

        # Raw [0,1] Sobol vectors for tokenizer (matches dataset format)
        theta_for_tokenizer = unit_vecs_t.to(device)  # [B, 34]

        # Physical params for rollout (radii, mu, sigma, dt, coupling, etc.)
        batch_tensors = sobol_batch_to_tensors(
            unit_vecs, n_channels, device=device, ranges=DEFAULT_RANGES,
        )

        # Step 2: Generate Fourier ICs (theta-coherent, using physical radii)
        ics = ic_gen.generate_batch(
            batch_size=B,
            n_channels=n_channels,
            grid_size=128,
            seed=cycle_seed + batch_start,
            device=torch.device(device),
            kernel_radii=batch_tensors.radii,
        )  # [B, C, H, W]

        # Step 3: Rollout from physical params, tokenize with raw Sobol theta
        with torch.no_grad():
            conditioning = {"theta": theta_for_tokenizer, "ic": ics}
            try:
                trajectories = rollout_provider.rollout(
                    conditioning, steps=config.rollout_steps,
                )
            except Exception as e:
                logger.debug(f"Batch {batch_start}: rollout failed: {e}")
                total_samples += B
                continue

            # Tokenize everything (GT tokens for all 3 families)
            # theta_for_tokenizer is raw [0,1] Sobol vector (matches training format)
            gt_tokens = tokenizer.tokenize(
                temporal_raw=trajectories,
                theta_features=theta_for_tokenizer,
                initial_raw=ics,
            )
            del trajectories

        gt_tokens_device = {
            k: v.to(device) for k, v in gt_tokens.items() if k in full_keys
        }
        # Contract to active keys for D3PM
        if token_filter is not None:
            gt_active = token_filter.contract(gt_tokens_device)
        else:
            gt_active = gt_tokens_device

        # Step 4: D3PM inpaints theta+IC given temporal (observed)
        # Uses active keys only (entropy-filtered)
        observed_dict = {}
        target_dict = {}
        for key in all_keys:
            if key in temporal_keys:
                observed_dict[key] = torch.ones(B, dtype=torch.bool, device=device)
                target_dict[key] = torch.zeros(B, dtype=torch.bool, device=device)
            else:
                observed_dict[key] = torch.zeros(B, dtype=torch.bool, device=device)
                target_dict[key] = torch.ones(B, dtype=torch.bool, device=device)

        best_agreement = [-1.0] * B
        best_tokens: list = [None] * B

        for candidate_idx in range(n_candidates):
            # D3PM inpaints theta+IC given temporal observations (active keys only)
            with torch.no_grad():
                completed = diffusion.sample(
                    batch_size=B,
                    observed_dict=observed_dict,
                    x_0_dict=gt_active,
                    denoising_network=denoiser,
                    device=device,
                )

            # Expand to full keys for decoder (which needs all families)
            if token_filter is not None:
                completed_full = token_filter.expand(completed)
            else:
                completed_full = completed

            # Decode D3PM's predicted theta+IC → continuous params + grid
            with torch.no_grad():
                decoded = decoder.decode(completed_full)
                theta_pred = decoded.get("theta")
                u0_pred = decoded.get("grids")

            if theta_pred is None or u0_pred is None:
                logger.debug(f"Candidate {candidate_idx}: decode failed")
                continue

            # Rollout from D3PM's predicted (theta, IC) → realized trajectory
            with torch.no_grad():
                pred_conditioning = {
                    "theta": theta_pred,
                    "ic": u0_pred,
                    "token_indices": completed_full,
                }
                try:
                    pred_trajectories = rollout_provider.rollout(
                        pred_conditioning, steps=config.rollout_steps,
                    )
                except Exception as e:
                    logger.debug(f"Candidate {candidate_idx}: rollout failed: {e}")
                    continue

            # Retokenize realized trajectory
            with torch.no_grad():
                realized = tokenizer.tokenize(
                    temporal_raw=pred_trajectories,
                    theta_features=theta_pred,
                    initial_raw=u0_pred,
                )
            del pred_trajectories
            realized_temporal = {
                k: v.to(device) for k, v in realized.items()
                if k in temporal_keys
            }

            # Score: do realized temporal tokens match GT temporal?
            for b in range(B):
                n_checked = 0
                n_agree = 0
                for key in temporal_keys:
                    if key not in realized_temporal or key not in gt_active:
                        continue
                    n_checked += 1
                    if realized_temporal[key][b] == gt_active[key][b]:
                        n_agree += 1
                agreement = n_agree / max(n_checked, 1)

                if agreement > best_agreement[b]:
                    best_agreement[b] = agreement
                    best_tokens[b] = {
                        k: v[b:b+1].clone() for k, v in completed.items()
                    }

        # Accept/reject
        for b in range(B):
            total_samples += 1
            if best_tokens[b] is None:
                continue

            agreement = best_agreement[b]
            all_agreements.append(agreement)

            if agreement >= threshold:
                hard_targets.append({
                    "tokens": best_tokens[b],
                    "observed": {k: v[b:b+1] for k, v in observed_dict.items()},
                    "target": {k: v[b:b+1] for k, v in target_dict.items()},
                    "agreement": agreement,
                })
                total_accepted += 1

                if max_accept is not None and total_accepted >= max_accept:
                    logger.info(
                        f"  Reached {max_accept} accepted targets "
                        f"after {total_samples} proposals"
                    )
                    done = True
                    break

        # Periodic logging
        if total_samples % 100 < gen_bs or done:
            target_str = f"/{max_accept}" if max_accept else ""
            ag = np.array(all_agreements) if all_agreements else np.array([0.0])
            logger.info(
                f"  Proposals: {total_samples}, "
                f"accepted={total_accepted}{target_str} "
                f"({100 * total_accepted / max(total_samples, 1):.1f}%), "
                f"agreement: mean={ag.mean():.3f} std={ag.std():.3f} "
                f"min={ag.min():.3f} max={ag.max():.3f}"
            )

    ag = np.array(all_agreements) if all_agreements else np.array([0.0])
    logger.info(
        f"Novel-Sobol generation complete: "
        f"{total_accepted}/{total_samples} accepted "
        f"({100 * total_accepted / max(total_samples, 1):.1f}%), "
        f"agreement: mean={ag.mean():.3f} std={ag.std():.3f} "
        f"min={ag.min():.3f} max={ag.max():.3f}"
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

    All token positions (temporal + initial + theta) are trained.

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

        for batch_start in range(0, len(hard_targets), ft_config.batch_size):
            batch_items = hard_targets[batch_start : batch_start + ft_config.batch_size]
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

            loss = _compute_refinement_loss(
                predicted_logits, tokens_batch, target_batch,
            )

            optimizer.zero_grad()
            loss.backward()
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
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy loss on target (masked) positions.

    Uses the same focal loss as training to avoid wasting gradient on
    shared tokens that the model already predicts with high confidence.
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

        # Focal weighting: down-weight easy predictions
        if focal_gamma > 0:
            with torch.no_grad():
                p_t = F.softmax(logits, dim=-1)
                p_correct = p_t.gather(1, targets.unsqueeze(1)).squeeze(1)
                focal_weight = (1 - p_correct) ** focal_gamma
            loss = loss * focal_weight

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
    config = load_experiment_config(args.config, RefinementConfig)

    if args.max_samples is not None:
        config = config.model_copy(update={"max_samples": args.max_samples})
    if args.device is not None:
        config = config.model_copy(update={"device": args.device})

    torch.manual_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("D3PM Offline Hard-Target Refinement (Integrated)")
    logger.info("=" * 60)
    logger.info(f"  D3PM checkpoint:     {config.d3pm_checkpoint}")
    logger.info(f"  Rollout source:      {config.mno_checkpoint or 'GT simulator'}")
    logger.info(f"  Tokenizer checkpoint:{config.tokenizer_checkpoint}")
    logger.info(f"  Dataset:             {config.dataset_path}")
    logger.info(f"  Refinement cycles:   {config.num_refinement_cycles}")
    logger.info(f"  Mask probability:    {config.mask_probability}")
    logger.info(f"  Quality threshold:   {config.quality_filter.min_observed_agreement}")
    logger.info(f"  D3PM candidates:     {config.d3pm_n_candidates}")
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

    # Choose generation mode
    unconditional = getattr(args, 'unconditional', False)

    dataset = None
    if not unconditional:
        logger.info("\nLoading dataset...")
        dataset = SpinlockDataset(
            config.dataset_path,
            max_samples=config.max_samples,
            load_gt_temporal_features=True,
        )
        logger.info(f"Dataset: {len(dataset)} samples")
    else:
        logger.info("\nUnconditional mode: no dataset needed")

    # Refinement loop
    all_metrics = []
    for cycle in range(config.num_refinement_cycles):
        logger.info(f"\n{'='*60}")
        logger.info(f"Refinement Cycle {cycle + 1}/{config.num_refinement_cycles}")
        logger.info(f"{'='*60}")

        logger.info("\nGenerating hard targets...")
        if unconditional:
            hard_targets = generate_novel_sobol_targets(
                diffusion, denoiser, rollout_provider, tokenizer, decoder, config,
                token_filter=token_filter, cycle=cycle,
            )
        else:
            hard_targets = generate_hard_targets(
                dataset, diffusion, denoiser, rollout_provider, tokenizer, decoder, config
            )

        logger.info("\nFine-tuning D3PM on hard targets...")
        ft_metrics = fine_tune_d3pm(diffusion, denoiser, hard_targets, config)

        all_metrics.append({
            "cycle": cycle + 1,
            "num_accepted": len(hard_targets),
            **ft_metrics,
        })

        save_refinement_checkpoint(
            diffusion, denoiser, cycle + 1, ft_metrics, config, output_dir,
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

    final_path = output_dir / "refinement_final.pt"
    torch.save({
        "denoiser_state_dict": denoiser.state_dict(),
        "diffusion_state_dict": diffusion.state_dict(),
        "all_metrics": all_metrics,
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
        "--unconditional",
        action="store_true",
        help="Novel-Sobol mode: sample new (theta, IC) from Sobol space (different seed), "
             "D3PM predicts temporal via inpainting, physics verifies. No dataset reuse.",
    )
    args = parser.parse_args()
    main(args)
