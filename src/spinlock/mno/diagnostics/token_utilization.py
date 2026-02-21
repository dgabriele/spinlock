"""Token utilization diagnostics for MNO training.

Measures actual token diversity in VQ space to complement contrastive loss metrics.
The token_contrastive loss can be misleading - it measures similarity in feature
space, not actual token code diversity. These diagnostics directly measure:

1. Per-codebook utilization - which codes are actually used?
2. Token set diversity - how many unique token combinations?
3. Token entropy - peaked (collapsed) or uniform (diverse)?
4. Comparison to baseline - does MNO span same distribution as CNO?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class TokenUtilizationDiagnostic:
    """Measures token diversity in VQ space."""

    def __init__(self, num_samples: int = 100):
        """Initialize diagnostic.

        Args:
            num_samples: Number of samples to collect per diagnostic run
        """
        self.num_samples = num_samples
        self.reset()

    def reset(self):
        """Reset accumulators for new diagnostic period."""
        self.token_indices: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.num_collected = 0

    def accumulate(self, hard_tokens: Dict[str, torch.Tensor]):
        """Accumulate token indices from a batch.

        Args:
            hard_tokens: Dict[quantizer_key → [B]] token indices
        """
        if self.num_collected >= self.num_samples:
            return

        for key, tokens in hard_tokens.items():
            self.token_indices[key].append(tokens.cpu())

        self.num_collected += tokens.shape[0]

    def compute_diagnostics(self) -> Dict[str, float]:
        """Compute token utilization metrics.

        Returns:
            Dict with:
                - per_codebook_utilization: fraction of codes used per quantizer
                - mean_codebook_utilization: average across all quantizers
                - unique_token_combinations: number of unique full token sets
                - token_entropy: average entropy across quantizers (bits)
                - num_samples: number of samples analyzed
        """
        if self.num_collected == 0:
            return {}

        # Concatenate all batches
        concatenated = {}
        for key, token_list in self.token_indices.items():
            concatenated[key] = torch.cat(token_list, dim=0)  # [N]

        # Per-codebook utilization
        utilizations = {}
        entropies = {}

        for key, tokens in concatenated.items():
            # Unique codes used
            unique_codes = tokens.unique()
            vocab_size = tokens.max().item() + 1  # Approximate
            utilization = len(unique_codes) / vocab_size
            utilizations[key] = utilization

            # Entropy (in bits)
            counts = torch.bincount(tokens, minlength=vocab_size).float()
            probs = counts / counts.sum()
            probs = probs[probs > 0]  # Remove zeros
            entropy = -(probs * torch.log2(probs)).sum().item()
            entropies[key] = entropy

        # Token set diversity (unique combinations)
        # Stack all quantizer tokens: [N, num_quantizers]
        stacked = torch.stack([concatenated[k] for k in sorted(concatenated.keys())], dim=1)

        # Convert to tuples and count unique
        token_tuples = [tuple(row.tolist()) for row in stacked]
        unique_combinations = len(set(token_tuples))

        return {
            'per_codebook_utilization': utilizations,
            'mean_codebook_utilization': np.mean(list(utilizations.values())),
            'unique_token_combinations': unique_combinations,
            'mean_token_entropy': np.mean(list(entropies.values())),
            'num_samples': self.num_collected,
        }

    def format_report(self, diagnostics: Dict) -> str:
        """Format diagnostics as a readable report.

        Args:
            diagnostics: Output from compute_diagnostics()

        Returns:
            Formatted string report
        """
        if not diagnostics:
            return "No token data collected"

        lines = [
            f"Token Utilization Diagnostic (n={diagnostics['num_samples']})",
            f"  Mean codebook utilization: {diagnostics['mean_codebook_utilization']:.3f}",
            f"  Unique token combinations: {diagnostics['unique_token_combinations']}",
            f"  Mean token entropy: {diagnostics['mean_token_entropy']:.2f} bits",
        ]

        # Show top 5 worst utilized codebooks
        utils = diagnostics['per_codebook_utilization']
        sorted_utils = sorted(utils.items(), key=lambda x: x[1])[:5]
        if sorted_utils:
            lines.append("  Bottom 5 codebook utilizations:")
            for key, util in sorted_utils:
                lines.append(f"    {key}: {util:.3f}")

        return "\n".join(lines)


def compare_token_distributions(
    mno_tokens: Dict[str, torch.Tensor],
    cno_tokens: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compare token distributions between MNO and CNO rollouts.

    Measures how well MNO spans the same token space as CNO.

    Args:
        mno_tokens: MNO token indices, Dict[quantizer_key → [N]]
        cno_tokens: CNO token indices, Dict[quantizer_key → [N]]

    Returns:
        Dict with:
            - js_divergence: Jensen-Shannon divergence (0=identical, 1=completely different)
            - overlap_coefficient: Jaccard similarity of used codes
            - kl_divergence: KL(CNO || MNO) - how much info lost
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy

    results = {}

    for key in mno_tokens.keys():
        if key not in cno_tokens:
            continue

        mno = mno_tokens[key].numpy()
        cno = cno_tokens[key].numpy()

        # Get vocabulary size (max of both)
        vocab_size = max(mno.max(), cno.max()) + 1

        # Compute distributions
        mno_counts = np.bincount(mno, minlength=vocab_size)
        cno_counts = np.bincount(cno, minlength=vocab_size)

        mno_probs = mno_counts / mno_counts.sum()
        cno_probs = cno_counts / cno_counts.sum()

        # JS divergence
        js_div = jensenshannon(mno_probs, cno_probs)

        # Overlap coefficient (Jaccard of used codes)
        mno_used = set(mno.tolist())
        cno_used = set(cno.tolist())
        jaccard = len(mno_used & cno_used) / len(mno_used | cno_used)

        # KL divergence CNO || MNO (how much info lost)
        # Add small epsilon to avoid log(0)
        kl_div = entropy(cno_probs + 1e-10, mno_probs + 1e-10)

        results[key] = {
            'js_divergence': js_div,
            'overlap_coefficient': jaccard,
            'kl_divergence': kl_div,
        }

    # Average across all quantizers
    if results:
        avg_js = np.mean([v['js_divergence'] for v in results.values()])
        avg_overlap = np.mean([v['overlap_coefficient'] for v in results.values()])
        avg_kl = np.mean([v['kl_divergence'] for v in results.values()])

        return {
            'mean_js_divergence': avg_js,
            'mean_overlap_coefficient': avg_overlap,
            'mean_kl_divergence': avg_kl,
            'per_quantizer': results,
        }

    return {}
