#!/usr/bin/env python3
"""Debug script to check actual feature dimensions."""

import torch
from spinlock.features.temporal.temporal import TemporalFeatureExtractor

device = torch.device('cpu')
extractor = TemporalFeatureExtractor(device=device, window_size=5, medium_window=20)

B, C, H, W = 4, 3, 64, 64

# Feed enough timesteps
for t in range(25):
    u = torch.randn(B, C, H, W, device=device)

    # Patch extract to print dimensions
    inst = extractor._extract_instantaneous(u)
    local = extractor._extract_local_temporal(u)
    stab = extractor._extract_local_stability(u)
    phase = extractor._extract_phase_space(u)
    multi = extractor._extract_multiscale(u)

    if t == 24:  # Last iteration
        print(f"Instantaneous: {inst.shape} (expected [4, 24])")
        print(f"Local Temporal: {local.shape} (expected [4, 39])")
        print(f"Local Stability: {stab.shape} (expected [4, 25])")
        print(f"Phase Space: {phase.shape} (expected [4, 26])")
        print(f"Multi-scale: {multi.shape} (expected [4, 30])")

        total = inst.shape[1] + local.shape[1] + stab.shape[1] + phase.shape[1] + multi.shape[1]
        print(f"\nTotal: {total}D")
        print(f"Expected: 144D (24+39+25+26+30)")
        print(f"Difference: {total - 144}D")
