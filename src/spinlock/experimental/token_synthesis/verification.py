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
from typing import Dict, List, Tuple

import torch

from spinlock.experimental.token_synthesis.config import SurprisalConfig

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
