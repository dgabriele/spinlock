#!/usr/bin/env python3
"""Check ground truth token diversity in the pretokenized dataset.

This tells us if the GT dataset has inherently low or high token diversity.
If GT has low diversity, then token_contrastive is fighting the wrong battle.
"""

import h5py
import torch
import numpy as np
from pathlib import Path
import sys

# Add spinlock to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinlock.mno.diagnostics.token_utilization import TokenUtilizationDiagnostic


def main():
    pretokenized_path = "datasets/50k_cno_v3_tokenized_temporal_res.h5"
    num_samples = 1000  # Analyze first 1K samples

    print(f"Analyzing GT token diversity: {pretokenized_path}")
    print(f"Samples: {num_samples}\n")

    # Load GT tokens
    with h5py.File(pretokenized_path, 'r') as f:
        print("Available keys:", list(f.keys()))
        print()

        # Assuming structure: tokens/{quantizer_key} -> [N, ...]
        tokens_group = f['tokens']
        quantizer_keys = list(tokens_group.keys())
        print(f"Quantizers: {len(quantizer_keys)}")
        print(f"First few: {quantizer_keys[:5]}\n")

        # Load all quantizer tokens for first N samples
        gt_tokens = {}
        for key in quantizer_keys:
            tokens_array = tokens_group[key][:num_samples]  # [N, ...] or [N]
            # Flatten to [N] if multi-dim
            if len(tokens_array.shape) > 1:
                tokens_array = tokens_array.reshape(num_samples, -1)[:, 0]  # Take first timestep if temporal
            gt_tokens[key] = torch.from_numpy(tokens_array.astype(np.int64))

    # Run diagnostic
    diagnostic = TokenUtilizationDiagnostic(num_samples=num_samples)
    diagnostic.accumulate(gt_tokens)

    results = diagnostic.compute_diagnostics()
    report = diagnostic.format_report(results)

    print("=" * 70)
    print("GROUND TRUTH TOKEN DIVERSITY")
    print("=" * 70)
    print(report)
    print("=" * 70)

    # Compare to random chance
    if results:
        mean_util = results['mean_codebook_utilization']
        mean_entropy = results['mean_token_entropy']
        unique_combos = results['unique_token_combinations']

        print("\nInterpretation:")
        if mean_util < 0.3:
            print(f"  ⚠️  LOW codebook utilization ({mean_util:.3f}) - GT uses few codes per quantizer")
        else:
            print(f"  ✓  MODERATE/HIGH utilization ({mean_util:.3f}) - GT spans many codes")

        if unique_combos < num_samples * 0.5:
            print(f"  ⚠️  LOW combination diversity ({unique_combos}/{num_samples}) - many duplicates")
        else:
            print(f"  ✓  HIGH combination diversity ({unique_combos}/{num_samples}) - mostly unique")

        print("\nConclusion:")
        if mean_util < 0.3 or unique_combos < num_samples * 0.5:
            print("  → GT has LOW token diversity")
            print("  → token_contrastive is FIGHTING the GT distribution")
            print("  → RECOMMENDATION: Disable token_contrastive (lambda=0.0)")
            print("  → Keep roundtrip loss only (enforces MNO matches GT distribution)")
        else:
            print("  → GT has MODERATE/HIGH token diversity")
            print("  → token_contrastive is ALIGNED with GT distribution")
            print("  → RECOMMENDATION: Keep token_contrastive (investigate other issues)")


if __name__ == "__main__":
    main()
