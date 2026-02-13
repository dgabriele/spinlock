"""Surprisal computation for token synthesis verification.

Measures how well dreamed token sequences survive the physical roundtrip:
    generated tokens → decode → QBM rollout → retokenize → retokenized tokens

Surprisal captures the distance between generated and retokenized tokens,
weighted by the entropy of the retokenized distribution.

Both generated and retokenized token dicts share the same key set (all 96
category-level keys for QBM). Jaccard over {(key, value)} tuples simplifies to
normalized Hamming similarity: match_count / total_keys.
"""

import logging
import random
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F

from spinlock.experimental.token_synthesis.config import (
    CounterfactualConfig,
    SurprisalConfig,
)

logger = logging.getLogger(__name__)


class SurprisalComputer:
    """Compute surprisal scores for generated vs retokenized token sequences.

    Surprisal = (1 - Jaccard) + lambda * entropy(retokenized)

    Higher surprisal → the physical grounding loop changed more tokens,
    meaning the generated sequence was less physically coherent.

    Args:
        config: SurprisalConfig with lambda_entropy and other parameters
        vocab_sizes: Dict mapping quantizer key → vocab size
    """

    def __init__(self, config: SurprisalConfig, vocab_sizes: Dict[str, int]):
        self.config = config
        self.vocab_sizes = vocab_sizes
        self._keys = sorted(vocab_sizes.keys())
        self._num_keys = len(self._keys)

        # Precompute max entropy per key (log2 of vocab_size)
        self._max_entropy_per_key = {
            k: torch.log2(torch.tensor(float(v))) for k, v in vocab_sizes.items()
        }
        self._total_max_entropy = sum(e.item() for e in self._max_entropy_per_key.values())

    def compute_jaccard(
        self,
        tokens_a: Dict[str, torch.Tensor],
        tokens_b: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Per-sample Jaccard similarity — fraction of matching (key, value) pairs.

        Since both dicts have identical key sets, this is equivalent to
        normalized Hamming similarity.

        Args:
            tokens_a: Dict mapping key → token indices [B]
            tokens_b: Dict mapping key → token indices [B]

        Returns:
            Jaccard similarity [B] in [0, 1]
        """
        # Use intersection of available keys (graceful if some families were skipped)
        common_keys = sorted(set(tokens_a.keys()) & set(tokens_b.keys()))
        if not common_keys:
            batch_size = next(iter(tokens_a.values())).shape[0]
            return torch.zeros(batch_size)

        matches = torch.zeros(next(iter(tokens_a.values())).shape[0])
        for key in common_keys:
            a = tokens_a[key]
            b = tokens_b[key]
            matches = matches + (a.cpu() == b.cpu()).float()

        return matches / len(common_keys)

    def compute_entropy(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample normalized entropy of token assignments.

        For each sample, computes how uniformly distributed the token values
        are within each codebook, normalized by the maximum possible entropy.

        This measures token diversity within a single sample — higher entropy
        means the sample uses a wider range of codebook entries.

        Args:
            tokens: Dict mapping key → token indices [B]

        Returns:
            Normalized entropy [B] in [0, 1]
        """
        batch_size = next(iter(tokens.values())).shape[0]
        total_entropy = torch.zeros(batch_size)

        for key in self._keys:
            if key not in tokens:
                continue

            t = tokens[key].cpu().long()
            vocab_size = self.vocab_sizes[key]

            # One-hot encode and compute per-sample "distribution"
            # For a single token per codebook, entropy is 0 (deterministic).
            # But across the batch, we can still measure diversity.
            # Here we treat each sample's token as a point mass → entropy = 0.
            # The meaningful entropy comes from comparing ACROSS codebooks within a sample:
            # we compute how many distinct token values appear vs how many could appear.
            #
            # Alternative: normalized token value / vocab_size as "spread" measure.
            # We use fraction of vocab used across all codebooks for this sample.
            unique_ratio = t.float() / max(vocab_size - 1, 1)  # [B] in [0, 1]
            total_entropy = total_entropy + unique_ratio

        # Normalize by number of keys
        if self._num_keys > 0:
            total_entropy = total_entropy / self._num_keys

        return total_entropy.clamp(0.0, 1.0)

    def compute_surprisal(
        self,
        generated: Dict[str, torch.Tensor],
        retokenized: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute surprisal score for generated vs retokenized tokens.

        surprisal = (1 - Jaccard(gen, retok)) + lambda * entropy(retok)

        Args:
            generated: Dict mapping key → generated token indices [B]
            retokenized: Dict mapping key → retokenized token indices [B]

        Returns:
            Surprisal scores [B] (higher = more novel/surprising)
        """
        jaccard = self.compute_jaccard(generated, retokenized)
        entropy = self.compute_entropy(retokenized)

        surprisal = (1.0 - jaccard) + self.config.lambda_entropy * entropy
        return surprisal

    def verify_with_multiple_samples(
        self,
        generated: Dict[str, torch.Tensor],
        retokenized_samples: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance of surprisal across K verification runs.

        Running QBM simulation multiple times with different random seeds produces
        slightly different retokenized results due to stochastic realizations.
        This captures the consistency of the physical grounding.

        Args:
            generated: Dict mapping key → generated token indices [B]
            retokenized_samples: List of K retokenized token dicts [B]

        Returns:
            Tuple of (mean_surprisal [B], variance [B])
        """
        surprisals = torch.stack([
            self.compute_surprisal(generated, retok)
            for retok in retokenized_samples
        ])  # [K, B]

        mean_surprisal = surprisals.mean(dim=0)   # [B]
        variance = surprisals.var(dim=0)           # [B]

        return mean_surprisal, variance


class CounterfactualSurprisal:
    """Novelty via masked prediction cross-entropy.

    Masks a subset of token keys, forward-diffuses masked positions to
    t_probe (default: T, full noise), and has the denoiser predict them
    from observed context. Per-sample mean cross-entropy on masked
    positions measures how surprising the token combination is.

    High CE = denoiser can't recover masked tokens from context → genuinely novel.
    Low CE = easily predicted from context → boring/redundant.

    Masking strategies are mixed per call:
    - Category-level: mask all levels of 1+ random categories (structured)
    - Random: each key independently masked with probability p (unstructured)

    Uses existing D3PM forward_process(mask_dict) and DenoisingNetwork(observed_dict)
    infrastructure — no new model changes needed.

    Args:
        config: CounterfactualConfig with masking parameters
        d3pm: DiscreteD3PM instance for forward process
        denoiser: DenoisingNetwork instance for prediction
        schema: TokenSchema for category structure
        device: torch device
    """

    def __init__(self, config, d3pm, denoiser, schema, device):
        self._config = config
        self._d3pm = d3pm
        self._denoiser = denoiser
        self._device = device

        # All token keys
        self._all_keys = sorted(schema.keys)

        # Precompute category structure from schema
        self._categories = schema.unique_categories()
        self._category_keys = {
            cat: schema.keys_for_category(*cat)
            for cat in self._categories
        }

        # Probe timestep: default to T-1 (max noise)
        self._t_probe = (
            config.t_probe
            if config.t_probe is not None
            else d3pm.schedule.num_timesteps - 1
        )

        logger.info(
            f"CounterfactualSurprisal: {len(self._all_keys)} keys, "
            f"{len(self._categories)} categories, t_probe={self._t_probe}"
        )

    def compute(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample counterfactual surprisal.

        1. Select keys to mask via mixed strategy
        2. Forward-diffuse masked positions to t_probe (full noise)
        3. Denoiser predicts masked positions from observed context
        4. Return mean cross-entropy on masked positions per sample

        Args:
            tokens: Dict mapping key → token indices [B]

        Returns:
            Per-sample surprisal [B]
        """
        batch_size = next(iter(tokens.values())).shape[0]
        masked_keys = self._select_masked_keys()

        if not masked_keys:
            return torch.zeros(batch_size, device=self._device)

        # Build mask dicts: True = apply noise (masked), False = keep (observed)
        mask_dict = {}
        observed_dict = {}
        for k in self._all_keys:
            if k not in tokens:
                continue
            is_masked = k in masked_keys
            mask_dict[k] = torch.full(
                (batch_size,), is_masked, dtype=torch.bool, device=self._device,
            )
            observed_dict[k] = torch.full(
                (batch_size,), not is_masked, dtype=torch.bool, device=self._device,
            )

        # Forward-diffuse masked positions to t_probe
        # Move tokens to device
        tokens_device = {
            k: v.to(self._device) for k, v in tokens.items()
            if k in self._all_keys
        }

        with torch.no_grad():
            noisy, _ = self._d3pm.forward_process(
                tokens_device, self._t_probe, mask_dict=mask_dict,
            )

            # Denoiser predicts from observed context
            t_tensor = torch.full(
                (batch_size,), self._t_probe, dtype=torch.long, device=self._device,
            )
            logits = self._denoiser(noisy, t_tensor, observed_dict=observed_dict)

        # Cross-entropy on masked positions, averaged per sample
        total_ce = torch.zeros(batch_size, device=self._device)
        n_masked = 0

        for k in masked_keys:
            if k not in logits or k not in tokens_device:
                continue
            ce = F.cross_entropy(
                logits[k], tokens_device[k], reduction='none',
            )  # [B]
            total_ce = total_ce + ce
            n_masked += 1

        if n_masked > 0:
            total_ce = total_ce / n_masked

        return total_ce

    def _select_masked_keys(self) -> Set[str]:
        """Select keys to mask via mixed strategy.

        With probability category_mask_prob → category-level masking,
        otherwise → random per-key masking.
        """
        if random.random() < self._config.category_mask_prob:
            return self._select_category_mask()
        else:
            return self._select_random_mask()

    def _select_category_mask(self) -> Set[str]:
        """Mask all levels of 1-N randomly selected categories.

        Ensures at least 1 key remains observed for context.
        """
        n_cats = min(
            random.randint(1, self._config.max_categories_to_mask),
            len(self._categories) - 1,  # Keep at least 1 category observed
        )
        if n_cats <= 0:
            return set()

        selected = random.sample(self._categories, n_cats)
        masked = set()
        for cat in selected:
            masked.update(self._category_keys[cat])

        # Safety: ensure at least 1 key remains observed
        if len(masked) >= len(self._all_keys):
            # Remove one key from masked set
            masked.discard(next(iter(masked)))

        return masked

    def _select_random_mask(self) -> Set[str]:
        """Each key masked independently with probability mask_fraction.

        Ensures at least 1 masked and at least 1 observed.
        """
        masked = {
            k for k in self._all_keys
            if random.random() < self._config.mask_fraction
        }

        # Ensure at least 1 masked
        if not masked:
            masked.add(random.choice(self._all_keys))

        # Ensure at least 1 observed
        if len(masked) >= len(self._all_keys):
            masked.discard(random.choice(list(masked)))

        return masked
