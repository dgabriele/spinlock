#!/usr/bin/env python3
"""
Compare pretokenized dataset token assignments to EMA cluster sizes.

If EMA shows 100% utilization but pretokenized dataset shows only 1 token,
then EMA is misleading (accumulated transients from early training).
"""

import torch
import h5py
import numpy as np
from pathlib import Path

checkpoint_path = Path("checkpoints/vqvae/theta_baseline_50k/vq_tokenizer_final.pt")
tokenized_path = Path("datasets/50k_baseline_tokenized_theta.h5")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']

print("="*80)
print("COMPARING PRETOKENIZED TOKENS TO EMA CLUSTER SIZES")
print("="*80)

# Focus on collapsed groups
test_groups = [
    ('temporal_group_5', 'L0'),
    ('temporal_group_10', 'L0'),
    ('temporal_group_19', 'L0'),
]

with h5py.File(tokenized_path, 'r') as f:
    tokens_group = f['tokens']

    for group_name, level in test_groups:
        print(f"\n{'='*80}")
        print(f"{group_name} {level}")
        print("="*80)

        # Get EMA cluster sizes
        ema_key = f"quantizers.{group_name}_{level}.ema_cluster_size"
        if ema_key in state_dict:
            ema_cluster = state_dict[ema_key].numpy()
            n_codes = len(ema_cluster)

            print(f"\nEMA Cluster Sizes:")
            print(f"  Total codes: {n_codes}")
            print(f"  Active (EMA > 0): {(ema_cluster > 0).sum()}/{n_codes}")
            print(f"  Unique EMA values: {len(np.unique(ema_cluster))}/{n_codes}")

            # Show distribution
            sorted_ema = np.sort(ema_cluster)[::-1]
            print(f"  Top 5 EMA values: {sorted_ema[:5].tolist()}")
            print(f"  Bottom 5 EMA values: {sorted_ema[-5:].tolist()}")

        # Get actual token assignments from pretokenized dataset
        token_key = f"{group_name}_{level}"
        if token_key in tokens_group:
            tokens = tokens_group[token_key][:]
            unique_tokens = np.unique(tokens)

            print(f"\nActual Token Assignments (from pretokenized dataset):")
            print(f"  Unique tokens used: {len(unique_tokens)}/{n_codes}")
            print(f"  Token IDs: {unique_tokens.tolist()}")

            # Count frequency of each token
            token_counts = np.bincount(tokens, minlength=n_codes)
            print(f"\nToken Usage Frequency:")
            for token_id in unique_tokens:
                count = token_counts[token_id]
                pct = 100 * count / len(tokens)
                ema_val = ema_cluster[token_id]
                print(f"  Token {token_id}: {count:5d} times ({pct:5.2f}%) | EMA={ema_val:.2f}")

            # Check unused tokens
            unused_tokens = np.setdiff1d(np.arange(n_codes), unique_tokens)
            if len(unused_tokens) > 0:
                print(f"\nUnused tokens ({len(unused_tokens)}/{n_codes}):")
                print(f"  IDs: {unused_tokens[:20].tolist()}")
                print(f"  Their EMA values: {ema_cluster[unused_tokens[:20]].tolist()}")

                # Check if unused tokens have non-zero EMA
                unused_with_ema = unused_tokens[ema_cluster[unused_tokens] > 0]
                if len(unused_with_ema) > 0:
                    print(f"\n  ⚠️  {len(unused_with_ema)} unused tokens have non-zero EMA!")
                    print(f"  This proves EMA includes training transients, not final usage.")
                    print(f"  Sample unused tokens with EMA > 0:")
                    for token_id in unused_with_ema[:5]:
                        print(f"    Token {token_id}: EMA={ema_cluster[token_id]:.2f} (never used in final tokenization)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
If we see unused tokens with non-zero EMA, it proves:

1. EMA cluster size tracks CUMULATIVE assignments during training
2. Early in training (before encoder converges), random/varied code assignments
3. As training progresses, encoder learns to map similar inputs to same embedding
4. Final tokenization uses only 1-2 codes, but EMA still remembers early assignments
5. EMA with decay 0.99 means old assignments decay very slowly

VERDICT:
- EMA cluster size is NOT a good metric for final model diversity
- Pretokenized dataset analysis is the GROUND TRUTH
- The "100% utilization" is an artifact of EMA accumulation over training
- Real utilization (after convergence) is much lower (1-3 codes per collapsed group)

This is NOT a bug - it's how EMA-based VQ-VAE works.
But it means we should NOT rely on EMA for diversity assessment.
""")
