"""Adaptive guided refinement search for D3PM inverse generation.

Replaces fixed-candidate generation with difficulty-proportional search:
1. Cheap initial D3PM round (2 candidates by default)
2. Budget allocation based on distance to acceptance threshold
3. Progressive refinement: D3PM re-sampling + local theta perturbation
4. Dynamic stopping when marginal improvement plateaus

The D3PM acts as an approximate posterior over (theta, IC) given temporal
observations. Local Sobol perturbation explores the physics-space
neighborhood around the D3PM's best proposal, treating it as MCMC
refinement around a warm start.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from spinlock.experimental.diffusion.config import RefinementConfig
from spinlock.experimental.diffusion.models import (
    DenoisingNetwork,
    DiscreteD3PM,
)
from spinlock.lenia.fourier_ic import FourierICConfig, FourierICGenerator
from spinlock.lenia.params import (
    DEFAULT_RANGES,
    sobol_batch_to_tensors,
    sobol_expected_dims,
)
from spinlock.rollout.provider import RolloutProvider
from spinlock.sampling.sobol import StratifiedSobolSampler
from spinlock.tokens.schema import TokenSchema
from spinlock.tokens.token_decoder import IntegratedTokenDecoder
from spinlock.tokens.tokenizer import VQTokenizer

from .candidate_budget import CandidateBudgetAllocator
from .ic_perturber import FourierICPerturber
from .local_perturber import LocalParameterPerturber

logger = logging.getLogger(__name__)


class AdaptiveRefinementSearch:
    """Adaptive difficulty-proportional search for D3PM inverse generation.

    Easy samples (>80% agreement on first try) cost 1-2 candidates.
    Hard samples get progressively wider local search around the D3PM's
    best proposal: more D3PM re-samples exploring the token-space posterior,
    plus local Sobol perturbation exploring the physics-space neighborhood.
    """

    def __init__(
        self,
        diffusion: DiscreteD3PM,
        denoiser: DenoisingNetwork,
        rollout_provider: RolloutProvider,
        tokenizer: VQTokenizer,
        decoder: IntegratedTokenDecoder,
        config: RefinementConfig,
        token_filter=None,
    ) -> None:
        self.diffusion = diffusion
        self.denoiser = denoiser
        self.rollout_provider = rollout_provider
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.config = config
        self.token_filter = token_filter

        # Schema / key setup
        schema = TokenSchema.from_tokenizer(tokenizer)
        self.full_temporal_keys = set(schema.keys_for_family("temporal"))
        full_theta_keys = set(schema.keys_for_family("theta"))
        full_initial_keys = set(schema.keys_for_family("initial"))
        self.full_keys = sorted(schema.vocab_sizes_dict().keys())

        if token_filter is not None:
            self.all_keys = sorted(token_filter.active_keys)
            self.temporal_keys = {
                k for k in self.all_keys if k in self.full_temporal_keys
            }
            self.theta_keys = {
                k for k in self.all_keys if k in full_theta_keys
            }
            self.initial_keys = {
                k for k in self.all_keys if k in full_initial_keys
            }
        else:
            self.all_keys = self.full_keys
            self.temporal_keys = self.full_temporal_keys
            self.theta_keys = full_theta_keys
            self.initial_keys = full_initial_keys

        self.non_temporal_keys = sorted(
            k for k in self.all_keys if k not in self.temporal_keys
        )

        # Subcomponents
        n_channels = 3
        sobol_dim = sobol_expected_dims(n_channels, DEFAULT_RANGES)
        self.sobol_dim = sobol_dim
        self.n_channels = n_channels

        self.perturber = LocalParameterPerturber(
            config.adaptive.perturbation, sobol_dim=sobol_dim,
        )
        self.ic_perturber = FourierICPerturber(
            ic_config=FourierICConfig(),
        )
        self.budget_allocator = CandidateBudgetAllocator(
            config.adaptive.budget,
            threshold=config.quality_filter.min_observed_agreement,
        )

        # Per-cycle acceptance rate (logged, not used for stopping)
        self.cycle_acceptance_rates: List[float] = []

        # Cosine-embedding agreement: cache codebook embeddings per temporal key
        self._temporal_codebooks: Dict[str, Tensor] = {}
        if config.quality_filter.agreement_metric == "cosine_embedding":
            for key in self.temporal_keys:
                if key in tokenizer.model.quantizers:
                    self._temporal_codebooks[key] = (
                        tokenizer.model.quantizers[key].embedding.weight.detach()
                    )

        # GPU memory phasing: tokenizer is swapped CPU↔GPU around rollouts
        # to free ~226MB VRAM for Lenia trajectory tensors.
        self._tokenizer_on_gpu = True

    # ── GPU memory phasing ──────────────────────────────────────────────────

    def _tokenizer_to_cpu(self) -> None:
        """Move tokenizer model to CPU to free VRAM for rollouts."""
        if self._tokenizer_on_gpu:
            self.tokenizer.model.to("cpu")
            torch.cuda.empty_cache()
            self._tokenizer_on_gpu = False

    def _tokenizer_to_gpu(self) -> None:
        """Move tokenizer model back to GPU for tokenization/decoding."""
        if not self._tokenizer_on_gpu:
            self.tokenizer.model.to(self.config.device)
            self._tokenizer_on_gpu = True

    # ── GT-token training ─────────────────────────────────────────────────

    def generate_gt_targets(self, cycle: int) -> List[Dict]:
        """Generate training targets from GT tokens directly (no D3PM inference).

        Every novel Sobol sample becomes a valid training target with agreement=1.0,
        since the tokens are ground-truth by construction. This eliminates the
        acceptance filter and the expensive D3PM + adaptive refinement phases.
        """
        max_accept = self.config.max_accepted_targets or 1000
        gen_bs = self.config.generation_batch_size

        cycle_seed = self.config.seed + 95 + cycle * 1000
        sobol_sampler = StratifiedSobolSampler(
            dimensionality=self.sobol_dim, scramble=True, seed=cycle_seed,
        )
        ic_gen = FourierICGenerator(FourierICConfig())

        hard_targets: List[Dict] = []

        logger.info(
            f"GT target generation: cycle={cycle}, seed={cycle_seed}, "
            f"target={max_accept}"
        )

        n_batches = (max_accept + gen_bs - 1) // gen_bs
        for batch_idx, batch_start in enumerate(range(0, max_accept, gen_bs)):
            B = min(gen_bs, max_accept - batch_start)
            logger.info(
                f"  GT batch {batch_idx + 1}/{n_batches}: "
                f"generating {B} samples (total so far: {len(hard_targets)})"
            )
            gt_result = self._generate_gt(
                sobol_sampler, ic_gen, B, cycle_seed, batch_start,
            )
            if gt_result is None:
                logger.warning(f"  GT batch {batch_idx + 1} failed, skipping")
                continue
            gt_active, observed_dict, target_dict, sobol_vectors = gt_result

            for b in range(B):
                tokens_single = {k: v[b : b + 1].clone() for k, v in gt_active.items()}
                hard_targets.append({
                    "tokens": tokens_single,
                    "observed": {k: v[b : b + 1] for k, v in observed_dict.items()},
                    "target": {k: v[b : b + 1] for k, v in target_dict.items()},
                    "agreement": 1.0,
                    "sobol_vector": sobol_vectors[b].cpu(),
                })

            if len(hard_targets) >= max_accept:
                break

        logger.info(f"GT target generation complete: {len(hard_targets)} targets")
        return hard_targets[:max_accept]

    def generate_eval_set(
        self, n_samples: int,
    ) -> List[Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]]:
        """Generate a fixed held-out evaluation set (called once before the loop).

        Uses a deterministic seed that never overlaps with training cycle seeds.
        Returns per-sample (gt_active, observed, target) tuples for reuse.
        """
        gen_bs = self.config.generation_batch_size
        eval_seed = self.config.seed + 77777

        sobol_sampler = StratifiedSobolSampler(
            dimensionality=self.sobol_dim, scramble=True, seed=eval_seed,
        )
        ic_gen = FourierICGenerator(FourierICConfig())

        eval_data: List[Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]] = []

        logger.info(f"Generating held-out eval set: n={n_samples}, seed={eval_seed}")

        n_batches = (n_samples + gen_bs - 1) // gen_bs
        for batch_idx, batch_start in enumerate(range(0, n_samples, gen_bs)):
            B = min(gen_bs, n_samples - batch_start)
            logger.info(
                f"  Eval batch {batch_idx + 1}/{n_batches}: "
                f"generating {B} samples (total so far: {len(eval_data)})"
            )
            gt_result = self._generate_gt(
                sobol_sampler, ic_gen, B, eval_seed, batch_start,
            )
            if gt_result is None:
                logger.warning(f"  Eval batch {batch_idx + 1} failed, skipping")
                continue
            gt_active, observed_dict, target_dict, _sobol = gt_result

            for b in range(B):
                eval_data.append((
                    {k: v[b : b + 1].clone() for k, v in gt_active.items()},
                    {k: v[b : b + 1] for k, v in observed_dict.items()},
                    {k: v[b : b + 1] for k, v in target_dict.items()},
                ))

            if len(eval_data) >= n_samples:
                break

        eval_data = eval_data[:n_samples]
        logger.info(f"Eval set ready: {len(eval_data)} samples")
        return eval_data

    def evaluate_model(
        self,
        eval_data: List[Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]],
    ) -> Dict[str, float]:
        """Run D3PM inference + roundtrip on the held-out eval set.

        Batches samples, runs initial D3PM round (no adaptive refinement),
        and returns mean agreement, acceptance rate, and diversity metrics.
        """
        if not eval_data:
            return {}

        device = self.config.device
        threshold = self.config.quality_filter.min_observed_agreement
        n_candidates = self.config.adaptive.initial_d3pm_candidates
        gen_bs = self.config.generation_batch_size

        all_agreements: List[float] = []
        all_candidates_accumulated: List[List[Dict[str, Tensor]]] = []

        for batch_start in range(0, len(eval_data), gen_bs):
            batch_items = eval_data[batch_start : batch_start + gen_bs]
            B = len(batch_items)

            # Stack per-sample dicts into batched tensors
            keys = sorted(batch_items[0][0].keys())
            gt_active = {
                k: torch.cat([item[0][k] for item in batch_items], dim=0).to(device)
                for k in keys
            }
            observed_dict = {
                k: torch.cat([item[1][k] for item in batch_items], dim=0).to(device)
                for k in keys
            }
            gt_temporal = {
                k: v for k, v in gt_active.items() if k in self.temporal_keys
            }

            best_agreement, _, batch_candidates = self._initial_d3pm_round(
                gt_active, observed_dict, gt_temporal, B,
            )

            for b in range(B):
                ag = best_agreement[b].item()
                all_agreements.append(ag if ag >= 0 else 0.0)
            all_candidates_accumulated.extend(batch_candidates)

        agreements_arr = np.array(all_agreements)
        mean_ag = float(agreements_arr.mean())
        std_ag = float(agreements_arr.std())
        min_ag = float(agreements_arr.min())
        max_ag = float(agreements_arr.max())
        accept_rate = float((agreements_arr >= threshold).mean())

        # Diversity metrics
        diversity = self._compute_candidate_diversity(all_candidates_accumulated)

        diversity_str = ""
        if diversity:
            diversity_str = (
                f", diversity: hamming={diversity['mean_pairwise_hamming']:.3f} "
                f"unique={diversity['mean_unique_candidates']:.1f}/{n_candidates} "
                f"collapsed={diversity['frac_fully_collapsed']:.2f}"
            )

        logger.info(
            f"  Eval ({len(all_agreements)} samples): "
            f"agreement mean={mean_ag:.4f} std={std_ag:.4f} "
            f"min={min_ag:.4f} max={max_ag:.4f}, "
            f"accepted={accept_rate:.3f} (threshold={threshold})"
            f"{diversity_str}"
        )

        result = {
            "mean_agreement": mean_ag,
            "std_agreement": std_ag,
            "min_agreement": min_ag,
            "max_agreement": max_ag,
            "acceptance_rate": accept_rate,
        }
        result.update(diversity)
        return result

    # ── Main entry point ────────────────────────────────────────────────────

    def generate_targets(self, cycle: int) -> List[Dict]:
        """Generate hard targets using adaptive difficulty-proportional search.

        Replaces ``generate_novel_sobol_targets()`` with adaptive refinement.
        """
        threshold = self.config.quality_filter.min_observed_agreement
        adaptive_cfg = self.config.adaptive
        stopping_cfg = adaptive_cfg.stopping

        hard_targets: List[Dict] = []
        total_samples = 0
        total_accepted = 0
        all_agreements: List[float] = []
        max_accept = self.config.max_accepted_targets
        gen_bs = self.config.generation_batch_size

        # Sobol sampler with cycle-specific seed
        cycle_seed = self.config.seed + 95 + cycle * 1000
        sobol_sampler = StratifiedSobolSampler(
            dimensionality=self.sobol_dim, scramble=True, seed=cycle_seed,
        )
        ic_gen = FourierICGenerator(FourierICConfig())

        n_attempts = (max_accept or 5000) * 2
        done = False

        # Dynamic stopping window
        window_accepted = 0
        window_total = 0

        logger.info(
            f"Adaptive generation: cycle={cycle}, seed={cycle_seed}, "
            f"initial_candidates={adaptive_cfg.initial_d3pm_candidates}, "
            f"max_rounds={stopping_cfg.max_rounds_per_sample}, "
            f"perturbation={'ON' if adaptive_cfg.perturbation.enabled else 'OFF'}, "
            f"target {max_accept} accepted"
        )

        for batch_start in range(0, n_attempts, gen_bs):
            if done:
                break
            B = min(gen_bs, n_attempts - batch_start)

            batch_targets, batch_agreements = self._process_batch(
                sobol_sampler, ic_gen, B, cycle_seed, batch_start,
            )

            hard_targets.extend(batch_targets)
            all_agreements.extend(batch_agreements)
            n_batch_accepted = len(batch_targets)
            total_accepted += n_batch_accepted
            total_samples += B
            window_accepted += n_batch_accepted
            window_total += B

            # Check acceptance rate window for early stopping
            if window_total >= stopping_cfg.acceptance_rate_window:
                rate = window_accepted / window_total
                if rate < stopping_cfg.min_acceptance_rate:
                    logger.info(
                        f"  Dynamic stop: acceptance rate {rate:.3f} < "
                        f"{stopping_cfg.min_acceptance_rate} over last "
                        f"{window_total} proposals"
                    )
                    done = True
                window_accepted = 0
                window_total = 0

            if max_accept is not None and total_accepted >= max_accept:
                logger.info(
                    f"  Reached {max_accept} accepted targets "
                    f"after {total_samples} proposals"
                )
                done = True

            # Periodic logging
            if total_samples % 100 < gen_bs or done:
                ag = np.array(all_agreements) if all_agreements else np.array([0.0])
                target_str = f"/{max_accept}" if max_accept else ""
                logger.info(
                    f"  Proposals: {total_samples}, "
                    f"accepted={total_accepted}{target_str} "
                    f"({100 * total_accepted / max(total_samples, 1):.1f}%), "
                    f"agreement: mean={ag.mean():.3f} std={ag.std():.3f} "
                    f"min={ag.min():.3f} max={ag.max():.3f}"
                )
                _log_agreement_distribution(ag, threshold)

        # Final summary
        ag = np.array(all_agreements) if all_agreements else np.array([0.0])
        logger.info(
            f"Adaptive generation complete: "
            f"{total_accepted}/{total_samples} accepted "
            f"({100 * total_accepted / max(total_samples, 1):.1f}%), "
            f"agreement: mean={ag.mean():.3f} std={ag.std():.3f}"
        )
        _log_agreement_distribution(ag, threshold)

        # Log acceptance rate (not used for stopping — sample difficulty varies)
        acceptance_rate = total_accepted / max(total_samples, 1)
        self.cycle_acceptance_rates.append(acceptance_rate)

        return hard_targets

    # ── Batch processing ────────────────────────────────────────────────────

    def _process_batch(
        self,
        sobol_sampler: StratifiedSobolSampler,
        ic_gen: FourierICGenerator,
        B: int,
        cycle_seed: int,
        batch_start: int,
    ) -> Tuple[List[Dict], List[float]]:
        """Process one batch: GT generation → initial D3PM → adaptive refinement."""
        threshold = self.config.quality_filter.min_observed_agreement

        # 1. Generate GT: Sobol → physical → IC → rollout → tokenize
        gt_result = self._generate_gt(
            sobol_sampler, ic_gen, B, cycle_seed, batch_start,
        )
        if gt_result is None:
            return [], []
        gt_active, observed_dict, target_dict, sobol_vectors = gt_result
        gt_temporal = {
            k: v for k, v in gt_active.items() if k in self.temporal_keys
        }

        # 2. Initial D3PM round (cheap: 2 candidates by default)
        best_agreement, best_tokens, _ = self._initial_d3pm_round(
            gt_active, observed_dict, gt_temporal, B,
        )

        # Count initial acceptances before adaptive refinement
        initial_accepted = sum(
            1 for b in range(B)
            if best_tokens[b] is not None and best_agreement[b] >= threshold
        )

        # 3. Adaptive refinement for rejected samples
        #    Modifies best_agreement and best_tokens in-place.
        rejected_indices = [
            b for b in range(B)
            if best_tokens[b] is not None and best_agreement[b] < threshold
        ]
        if rejected_indices:
            self._adaptive_rounds(
                rejected_indices, best_agreement, best_tokens,
                gt_active, gt_temporal, observed_dict, target_dict,
                cycle_seed, batch_start,
            )

        # 4. Collect all results (initial successes + adaptive recoveries)
        hard_targets: List[Dict] = []
        all_agreements: List[float] = []

        for b in range(B):
            if best_tokens[b] is None:
                continue
            all_agreements.append(best_agreement[b].item())
            if best_agreement[b] >= threshold:
                hard_targets.append(self._make_hard_target(
                    best_tokens[b], observed_dict, target_dict,
                    best_agreement[b].item(), b,
                    sobol_vector=sobol_vectors[b].cpu(),
                ))

        n_recovered = len(hard_targets) - initial_accepted
        if rejected_indices:
            logger.info(
                f"    Batch: {initial_accepted}/{B} initial, "
                f"{n_recovered}/{len(rejected_indices)} recovered by adaptive"
            )

        return hard_targets, all_agreements

    # ── GT generation ───────────────────────────────────────────────────────

    def _generate_gt(
        self,
        sobol_sampler: StratifiedSobolSampler,
        ic_gen: FourierICGenerator,
        B: int,
        cycle_seed: int,
        batch_start: int,
    ) -> Optional[Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Tensor]]:
        """Sample Sobol → physical → IC → rollout → tokenize → masks.

        Returns (gt_active, observed_dict, target_dict, sobol_vectors) or None on failure.
        The sobol_vectors [B, D] are stored in targets for future analysis.
        """
        device = self.config.device

        unit_vecs = sobol_sampler.sample(B)
        unit_vecs_t = torch.from_numpy(unit_vecs).float()
        theta_for_tokenizer = unit_vecs_t.to(device)

        batch_tensors = sobol_batch_to_tensors(
            unit_vecs, self.n_channels, device=device, ranges=DEFAULT_RANGES,
        )

        ics = ic_gen.generate_batch(
            batch_size=B,
            n_channels=self.n_channels,
            grid_size=128,
            seed=cycle_seed + batch_start,
            device=torch.device(device),
            kernel_radii=batch_tensors.radii,
        )

        with torch.no_grad():
            self._tokenizer_to_cpu()
            try:
                trajectories = self.rollout_provider.rollout(
                    {"theta": theta_for_tokenizer, "ic": ics},
                    steps=self.config.rollout_steps,
                )
            except Exception as e:
                logger.debug(f"Batch {batch_start}: GT rollout failed: {e}")
                self._tokenizer_to_gpu()
                return None

            self._tokenizer_to_gpu()
            gt_tokens = self.tokenizer.tokenize(
                temporal_raw=trajectories,
                theta_features=theta_for_tokenizer,
                initial_raw=ics,
            )
            del trajectories

        gt_tokens_device = {
            k: v.to(device) for k, v in gt_tokens.items() if k in self.full_keys
        }
        if self.token_filter is not None:
            gt_active = self.token_filter.contract(gt_tokens_device)
        else:
            gt_active = gt_tokens_device

        # Build observed/target masks
        observed_dict: Dict[str, Tensor] = {}
        target_dict: Dict[str, Tensor] = {}
        for key in self.all_keys:
            if key in self.temporal_keys:
                observed_dict[key] = torch.ones(B, dtype=torch.bool, device=device)
                target_dict[key] = torch.zeros(B, dtype=torch.bool, device=device)
            else:
                observed_dict[key] = torch.zeros(B, dtype=torch.bool, device=device)
                target_dict[key] = torch.ones(B, dtype=torch.bool, device=device)

        return gt_active, observed_dict, target_dict, unit_vecs_t

    # ── D3PM rounds ─────────────────────────────────────────────────────────

    def _initial_d3pm_round(
        self,
        gt_active: Dict[str, Tensor],
        observed_dict: Dict[str, Tensor],
        gt_temporal: Dict[str, Tensor],
        B: int,
    ) -> Tuple[Tensor, List[Optional[Dict]], List[List[Dict[str, Tensor]]]]:
        """Run initial D3PM candidates, return per-sample best + all candidates.

        Returns:
            best_agreement: [B] best agreement per sample
            best_tokens: [B] best token dict per sample
            all_candidates: [B][N] list of all candidate token dicts (for diversity)
        """
        device = self.config.device
        n_candidates = self.config.adaptive.initial_d3pm_candidates

        best_agreement = torch.full((B,), -1.0, device=device)
        best_tokens: List[Optional[Dict]] = [None] * B
        all_candidates: List[List[Dict[str, Tensor]]] = [[] for _ in range(B)]

        for _ in range(n_candidates):
            self._tokenizer_to_cpu()
            with torch.no_grad():
                completed = self.diffusion.sample(
                    batch_size=B,
                    observed_dict=observed_dict,
                    x_0_dict=gt_active,
                    denoising_network=self.denoiser,
                    device=device,
                    temperature=self.config.sampling_temperature,
                )

            agreements, tokens_list = self._evaluate_completion(
                completed, gt_temporal,
            )

            for b in range(B):
                # Store candidate for diversity analysis
                all_candidates[b].append(
                    {k: v[b : b + 1].clone() for k, v in completed.items()}
                )
                if tokens_list[b] is not None and agreements[b] > best_agreement[b]:
                    best_agreement[b] = agreements[b]
                    best_tokens[b] = tokens_list[b]

        return best_agreement, best_tokens, all_candidates

    def _evaluate_completion(
        self,
        completed: Dict[str, Tensor],
        gt_temporal: Dict[str, Tensor],
    ) -> Tuple[Tensor, List[Optional[Dict]]]:
        """Evaluate one D3PM completion: expand → decode → rollout → retokenize → score.

        Returns (agreements [B], tokens_list [B]).
        """
        device = self.config.device
        B = next(iter(completed.values())).shape[0]

        # Expand to full keys for decoder
        if self.token_filter is not None:
            completed_full = self.token_filter.expand(completed)
        else:
            completed_full = completed

        with torch.no_grad():
            self._tokenizer_to_gpu()
            decoded = self.decoder.decode(completed_full)
            theta_pred = decoded.get("theta")
            u0_pred = decoded.get("grids")

        if theta_pred is None or u0_pred is None:
            return torch.full((B,), -1.0, device=device), [None] * B

        with torch.no_grad():
            self._tokenizer_to_cpu()
            try:
                pred_trajectories = self.rollout_provider.rollout(
                    {"theta": theta_pred, "ic": u0_pred, "token_indices": completed_full},
                    steps=self.config.rollout_steps,
                )
            except Exception:
                self._tokenizer_to_gpu()
                return torch.full((B,), -1.0, device=device), [None] * B

        with torch.no_grad():
            self._tokenizer_to_gpu()
            realized = self.tokenizer.tokenize(
                temporal_raw=pred_trajectories,
                theta_features=theta_pred,
                initial_raw=u0_pred,
            )
        del pred_trajectories

        realized_temporal = {
            k: v.to(device) for k, v in realized.items() if k in self.temporal_keys
        }

        agreements = self._score_temporal_agreement(realized_temporal, gt_temporal, B)
        tokens_list = [
            {k: v[b : b + 1].clone() for k, v in completed.items()}
            for b in range(B)
        ]

        return agreements, tokens_list

    # ── Adaptive refinement ─────────────────────────────────────────────────

    def _adaptive_rounds(
        self,
        rejected_indices: List[int],
        best_agreement: Tensor,
        best_tokens: List[Optional[Dict]],
        gt_active: Dict[str, Tensor],
        gt_temporal: Dict[str, Tensor],
        observed_dict: Dict[str, Tensor],
        target_dict: Dict[str, Tensor],
        cycle_seed: int,
        batch_start: int,
    ) -> None:
        """Run adaptive refinement rounds, updating best_agreement/best_tokens in-place."""
        device = self.config.device
        adaptive_cfg = self.config.adaptive
        threshold = self.config.quality_filter.min_observed_agreement
        P = adaptive_cfg.perturbation.perturbations_per_round

        R = len(rejected_indices)

        # Allocate per-sample budgets based on difficulty
        rej_agreements = torch.tensor(
            [best_agreement[i].item() for i in rejected_indices], device=device,
        )
        d3pm_budgets, perturb_budgets = self.budget_allocator.allocate(rej_agreements)

        d3pm_remaining = d3pm_budgets.tolist()
        perturb_remaining = perturb_budgets.tolist()
        still_active = [True] * R

        sigma = adaptive_cfg.perturbation.initial_sigma

        for round_idx in range(adaptive_cfg.stopping.max_rounds_per_sample):
            # Active = below threshold AND has remaining budget
            active = [
                i for i in range(R)
                if still_active[i]
                and (d3pm_remaining[i] > 0 or perturb_remaining[i] > 0)
            ]
            if not active:
                break

            # --- D3PM re-sampling ---
            d3pm_eligible = [i for i in active if d3pm_remaining[i] > 0]
            if d3pm_eligible:
                self._d3pm_resample_round(
                    d3pm_eligible, rejected_indices,
                    best_agreement, best_tokens,
                    gt_active, observed_dict, gt_temporal,
                )
                for i in d3pm_eligible:
                    d3pm_remaining[i] -= 1

            # --- Perturbation ---
            perturb_eligible = [i for i in active if perturb_remaining[i] > 0]
            if perturb_eligible and adaptive_cfg.perturbation.enabled:
                perturb_orig = [rejected_indices[i] for i in perturb_eligible]
                self._perturbation_round(
                    perturb_eligible, perturb_orig,
                    best_tokens, gt_active, gt_temporal, best_agreement,
                    sigma, cycle_seed, batch_start + (round_idx + 1) * 10000,
                )
                for i in perturb_eligible:
                    perturb_remaining[i] -= P

            # Mark newly accepted as inactive
            for i in active:
                orig_idx = rejected_indices[i]
                if best_agreement[orig_idx] >= threshold:
                    still_active[i] = False

            # Widen perturbation radius
            sigma = min(
                sigma * adaptive_cfg.perturbation.sigma_growth_factor,
                adaptive_cfg.perturbation.max_sigma,
            )

            n_still = sum(still_active)
            if n_still > 0:
                logger.debug(
                    f"    Round {round_idx}: {n_still}/{R} still below threshold, "
                    f"sigma={sigma:.4f}"
                )

    def _d3pm_resample_round(
        self,
        d3pm_eligible: List[int],
        rejected_indices: List[int],
        best_agreement: Tensor,
        best_tokens: List[Optional[Dict]],
        gt_active: Dict[str, Tensor],
        observed_dict: Dict[str, Tensor],
        gt_temporal: Dict[str, Tensor],
    ) -> None:
        """One D3PM re-sample for each eligible rejected sample."""
        device = self.config.device
        d3pm_orig = [rejected_indices[i] for i in d3pm_eligible]

        gt_active_sub = {k: v[d3pm_orig] for k, v in gt_active.items()}
        observed_sub = {k: v[d3pm_orig] for k, v in observed_dict.items()}
        gt_temporal_sub = {k: v[d3pm_orig] for k, v in gt_temporal.items()}

        self._tokenizer_to_cpu()
        with torch.no_grad():
            completed = self.diffusion.sample(
                batch_size=len(d3pm_orig),
                observed_dict=observed_sub,
                x_0_dict=gt_active_sub,
                denoising_network=self.denoiser,
                device=device,
                temperature=self.config.sampling_temperature,
            )

        agreements, tokens_list = self._evaluate_completion(
            completed, gt_temporal_sub,
        )

        for j, local_idx in enumerate(d3pm_eligible):
            orig_idx = rejected_indices[local_idx]
            if tokens_list[j] is not None and agreements[j] > best_agreement[orig_idx]:
                best_agreement[orig_idx] = agreements[j]
                best_tokens[orig_idx] = tokens_list[j]

    def _perturbation_round(
        self,
        perturb_eligible: List[int],
        perturb_orig: List[int],
        best_tokens: List[Optional[Dict]],
        gt_active: Dict[str, Tensor],
        gt_temporal: Dict[str, Tensor],
        best_agreement: Tensor,
        sigma: float,
        cycle_seed: int,
        seed_offset: int,
    ) -> None:
        """One round of perturbation: decode best → perturb → rollout → score → update.

        Both theta and IC are perturbed in their respective parameter spaces:
        - Theta: clipped Gaussian in [0,1]^34 Sobol space (LocalParameterPerturber)
        - IC: Sobol quasi-random offsets in [0,1]^36 Fourier param space
              (FourierICPerturber), centered on FFT-extracted params from the
              D3PM's decoded IC grids, with frequencies from perturbed theta's radii.
        """
        device = self.config.device
        P = self.config.adaptive.perturbation.perturbations_per_round
        R_prime = len(perturb_orig)

        # 1. Decode best tokens → theta centers [R', sobol_dim] + IC grids [R', C, H, W]
        self._tokenizer_to_gpu()
        theta_centers = []
        ic_grids = []
        for orig_idx in perturb_orig:
            tokens_single = best_tokens[orig_idx]
            if self.token_filter is not None:
                expanded = self.token_filter.expand(tokens_single)
            else:
                expanded = tokens_single
            with torch.no_grad():
                decoded = self.decoder.decode(expanded)
            theta = decoded.get("theta")
            grids = decoded.get("grids")
            if theta is None:
                theta_centers.append(torch.full((1, self.sobol_dim), 0.5, device=device))
            else:
                theta_centers.append(theta)
            if grids is None:
                ic_grids.append(torch.full((1, self.n_channels, 128, 128), 0.5, device=device))
            else:
                ic_grids.append(grids)

        theta_centers_t = torch.cat(theta_centers, dim=0)  # [R', D_theta]
        ic_grids_t = torch.cat(ic_grids, dim=0)            # [R', C, H, W]

        # 2. Perturb theta: [R'*P, D_theta] clamped to [0,1]
        theta_perturbed = self.perturber.perturb(theta_centers_t, sigma, P)

        # 3. Physical params from perturbed theta (need radii for IC frequencies)
        unit_vecs_np = theta_perturbed.cpu().detach().numpy()
        batch_tensors = sobol_batch_to_tensors(
            unit_vecs_np, self.n_channels, device=device, ranges=DEFAULT_RANGES,
        )

        # 4. Perturb ICs in Fourier parameter space, using perturbed radii
        #    for frequency coupling (same sigma as theta perturbation)
        ic_seed = cycle_seed + seed_offset + 7777
        ics = self.ic_perturber.perturb_from_decoded(
            decoded_ics=ic_grids_t,           # [R', C, H, W]
            perturbed_radii=batch_tensors.radii,  # [R'*P, C]
            sigma=sigma,
            n_per_center=P,
            seed=ic_seed,
        )  # [R'*P, C, H, W]

        # 5. Rollout from perturbed params
        with torch.no_grad():
            self._tokenizer_to_cpu()
            try:
                trajectories = self.rollout_provider.rollout(
                    {"theta": theta_perturbed, "ic": ics},
                    steps=self.config.rollout_steps,
                )
            except Exception as e:
                logger.debug(f"    Perturbation rollout failed: {e}")
                self._tokenizer_to_gpu()
                return

        # 6. Tokenize
        with torch.no_grad():
            self._tokenizer_to_gpu()
            perturbed_full = self.tokenizer.tokenize(
                temporal_raw=trajectories,
                theta_features=theta_perturbed,
                initial_raw=ics,
            )
        del trajectories

        # 7. Score: expand GT temporal to match perturbation batch [R'*P]
        gt_temporal_expanded: Dict[str, Tensor] = {}
        for key in self.temporal_keys:
            vals = torch.stack([gt_temporal[key][idx] for idx in perturb_orig])
            gt_temporal_expanded[key] = vals.repeat_interleave(P)

        realized_temporal = {
            k: v.to(device) for k, v in perturbed_full.items()
            if k in self.temporal_keys
        }
        agreements = self._score_temporal_agreement(
            realized_temporal, gt_temporal_expanded, R_prime * P,
        )

        # 8. Contract perturbed tokens to active space
        perturbed_device = {
            k: v.to(device) for k, v in perturbed_full.items()
            if k in self.full_keys
        }
        if self.token_filter is not None:
            perturbed_active = self.token_filter.contract(perturbed_device)
        else:
            perturbed_active = perturbed_device

        # 9. Update best per rejected sample
        for j, (local_idx, orig_idx) in enumerate(
            zip(perturb_eligible, perturb_orig)
        ):
            best_in_group = best_agreement[orig_idx].item()
            best_flat_idx = -1

            for p in range(P):
                flat_idx = j * P + p
                ag = agreements[flat_idx].item()
                if ag > best_in_group:
                    best_in_group = ag
                    best_flat_idx = flat_idx

            if best_flat_idx >= 0:
                # Combine GT temporal + perturbed non-temporal
                combined: Dict[str, Tensor] = {}
                for key in self.all_keys:
                    if key in self.temporal_keys:
                        combined[key] = gt_active[key][orig_idx : orig_idx + 1].clone()
                    else:
                        combined[key] = perturbed_active[key][
                            best_flat_idx : best_flat_idx + 1
                        ].clone()

                best_agreement[orig_idx] = best_in_group
                best_tokens[orig_idx] = combined

    # ── Diversity metrics ─────────────────────────────────────────────────────

    def _compute_candidate_diversity(
        self,
        all_candidates: List[List[Dict[str, Tensor]]],
    ) -> Dict[str, float]:
        """Compute diversity metrics across candidate sets.

        For each sample, compares all N candidates on non-temporal positions
        (theta + IC) using pairwise Hamming distance.

        Args:
            all_candidates: [num_samples][N] list of candidate token dicts.

        Returns:
            Dict with diversity metrics, or empty dict if insufficient data.
        """
        if not all_candidates or not all_candidates[0]:
            return {}

        nt_keys = self.non_temporal_keys
        theta_keys_sorted = sorted(self.theta_keys)
        initial_keys_sorted = sorted(self.initial_keys)

        if not nt_keys:
            return {}

        total_hamming = 0.0
        theta_hamming = 0.0
        ic_hamming = 0.0
        total_unique = 0.0
        total_collapsed = 0
        n_samples = 0
        n_pairs = 0
        n_theta_pairs = 0
        n_ic_pairs = 0

        for candidates in all_candidates:
            N = len(candidates)
            if N < 2:
                total_unique += 1.0
                n_samples += 1
                continue

            # Stack non-temporal tokens into [N, K] matrix
            nt_vecs = []
            theta_vecs = []
            ic_vecs = []
            for cand in candidates:
                nt_vecs.append(torch.cat([cand[k].reshape(-1) for k in nt_keys]))
                if theta_keys_sorted:
                    theta_vecs.append(
                        torch.cat([cand[k].reshape(-1) for k in theta_keys_sorted])
                    )
                if initial_keys_sorted:
                    ic_vecs.append(
                        torch.cat([cand[k].reshape(-1) for k in initial_keys_sorted])
                    )

            nt_mat = torch.stack(nt_vecs)  # [N, K]

            # Pairwise Hamming: broadcast [N, 1, K] vs [1, N, K]
            K = nt_mat.shape[1]
            diff = (nt_mat.unsqueeze(1) != nt_mat.unsqueeze(0))  # [N, N, K]
            pw_hamming = diff.float().mean(dim=-1)  # [N, N]
            # Extract upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
            total_hamming += pw_hamming[mask].sum().item()
            n_pairs += mask.sum().item()

            # Theta Hamming
            if theta_vecs:
                theta_mat = torch.stack(theta_vecs)
                K_t = theta_mat.shape[1]
                if K_t > 0:
                    t_diff = (theta_mat.unsqueeze(1) != theta_mat.unsqueeze(0))
                    t_ham = t_diff.float().mean(dim=-1)
                    theta_hamming += t_ham[mask].sum().item()
                    n_theta_pairs += mask.sum().item()

            # IC Hamming
            if ic_vecs:
                ic_mat = torch.stack(ic_vecs)
                K_i = ic_mat.shape[1]
                if K_i > 0:
                    i_diff = (ic_mat.unsqueeze(1) != ic_mat.unsqueeze(0))
                    i_ham = i_diff.float().mean(dim=-1)
                    ic_hamming += i_ham[mask].sum().item()
                    n_ic_pairs += mask.sum().item()

            # Count unique candidates (by full non-temporal vector)
            unique_count = len(set(
                tuple(v.tolist()) for v in nt_vecs
            ))
            total_unique += unique_count

            # Fully collapsed = all candidates identical
            if unique_count == 1:
                total_collapsed += 1

            n_samples += 1

        if n_samples == 0:
            return {}

        result = {
            "mean_pairwise_hamming": total_hamming / max(n_pairs, 1),
            "mean_unique_candidates": total_unique / n_samples,
            "frac_fully_collapsed": total_collapsed / n_samples,
        }
        if n_theta_pairs > 0:
            result["theta_pairwise_hamming"] = theta_hamming / n_theta_pairs
        if n_ic_pairs > 0:
            result["ic_pairwise_hamming"] = ic_hamming / n_ic_pairs

        return result

    # ── Scoring helpers ─────────────────────────────────────────────────────

    def _score_temporal_agreement(
        self,
        realized: Dict[str, Tensor],
        gt: Dict[str, Tensor],
        B: int,
    ) -> Tensor:
        """Score temporal agreement between realized and GT tokens.

        When ``agreement_metric == "token_match"``, returns fraction of exact
        matches (binary, in [0, 1]).

        When ``agreement_metric == "cosine_embedding"``, looks up codebook
        embeddings for each position and returns mean cosine similarity across
        temporal keys.  This is geometry-aware: near-miss codes (close in
        embedding space) score high, consistent with weighted-Hamming training.
        """
        device = next(iter(gt.values())).device
        n_agree = torch.zeros(B, device=device)
        n_checked = 0

        if self.config.quality_filter.agreement_metric == "cosine_embedding":
            for key in self.temporal_keys:
                if key not in realized or key not in gt:
                    continue
                cb = self._temporal_codebooks.get(key)
                if cb is None:
                    continue
                cb_dev = cb.to(device)
                r_emb = cb_dev[realized[key][:B]]  # [B, D_k]
                g_emb = cb_dev[gt[key][:B]]        # [B, D_k]
                n_agree += F.cosine_similarity(r_emb, g_emb, dim=-1)
                n_checked += 1
        else:
            for key in self.temporal_keys:
                if key not in realized or key not in gt:
                    continue
                n_checked += 1
                n_agree += (realized[key][:B] == gt[key][:B]).float()

        return n_agree / max(n_checked, 1)

    def _make_hard_target(
        self,
        tokens: Dict[str, Tensor],
        observed_dict: Dict[str, Tensor],
        target_dict: Dict[str, Tensor],
        agreement: float,
        b: int,
        sobol_vector: Optional[Tensor] = None,
    ) -> Dict:
        """Construct a hard target dict for D3PM fine-tuning."""
        target = {
            "tokens": tokens,
            "observed": {k: v[b : b + 1] for k, v in observed_dict.items()},
            "target": {k: v[b : b + 1] for k, v in target_dict.items()},
            "agreement": agreement,
        }
        if sobol_vector is not None:
            target["sobol_vector"] = sobol_vector
        return target


# ── Logging utility ─────────────────────────────────────────────────────────


def _log_agreement_distribution(
    agreements: np.ndarray, threshold: float, n_bins: int = 5,
) -> None:
    """Log a data-driven bucketed frequency distribution of agreement values.

    Uses Jenks natural breaks (1D k-means) to find optimal bin edges
    that minimize within-class variance, then logs counts per bin.
    """
    if len(agreements) < n_bins:
        return

    sorted_vals = np.sort(agreements)

    # Quantile-based initialization then 1D k-means refinement
    breaks = [sorted_vals[0]]
    for b in range(1, n_bins):
        breaks.append(np.quantile(sorted_vals, b / n_bins))
    breaks.append(sorted_vals[-1] + 1e-6)

    for _ in range(20):
        assignments = np.digitize(sorted_vals, breaks[1:-1])
        new_breaks = [sorted_vals[0]]
        for b in range(n_bins):
            cluster = sorted_vals[assignments == b]
            if len(cluster) > 0:
                new_breaks.append(cluster.max() + 1e-6)
            else:
                new_breaks.append(new_breaks[-1])
        if len(new_breaks) == len(breaks) and all(
            abs(a - b_val) < 1e-6 for a, b_val in zip(new_breaks, breaks)
        ):
            break
        breaks = new_breaks

    bin_edges = sorted(set(breaks))
    if len(bin_edges) < 2:
        return

    logger.info("  Agreement distribution:")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        count = np.sum((agreements >= lo) & (agreements < hi))
        pct = 100 * count / len(agreements)
        bar = "#" * int(pct / 2)
        accepted = "+" if lo >= threshold else " "
        logger.info(
            f"    {accepted} [{lo:.3f}-{hi:.3f}): {count:5d} ({pct:5.1f}%) {bar}"
        )
