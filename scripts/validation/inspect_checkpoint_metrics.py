#!/usr/bin/env python3
"""Inspect checkpoint metrics to understand utilization computation."""

import torch
import json
from pathlib import Path

# Load checkpoint
checkpoint_path = Path("checkpoints/vqvae/theta_baseline_50k/vq_tokenizer_final.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("="*80)
print("CHECKPOINT KEYS:")
print("="*80)
for key in sorted(checkpoint.keys()):
    value = checkpoint[key]
    if isinstance(value, (int, float, str)):
        print(f"  {key}: {value}")
    elif isinstance(value, dict):
        print(f"  {key}: <dict with {len(value)} keys>")
    elif isinstance(value, list):
        print(f"  {key}: <list with {len(value)} items>")
    else:
        print(f"  {key}: {type(value)}")

print("\n" + "="*80)
print("FINAL METRICS KEYS (if exists):")
print("="*80)
if 'metrics' in checkpoint:
    metrics = checkpoint['metrics']
    print(f"Total metrics: {len(metrics)}")
    print("\nSample metrics (first 20):")
    for i, key in enumerate(sorted(metrics.keys())[:20]):
        print(f"  {key}: {metrics[key]}")

    # Look for utilization metrics
    print("\n" + "="*80)
    print("UTILIZATION METRICS:")
    print("="*80)
    util_metrics = {k: v for k, v in metrics.items() if 'utilization' in k.lower()}
    if util_metrics:
        for key, value in sorted(util_metrics.items())[:10]:
            print(f"  {key}: {value}")
    else:
        print("  NO UTILIZATION METRICS FOUND!")

# Check training history file
history_path = Path("checkpoints/vqvae/theta_baseline_50k/training_history.json")
if history_path.exists():
    print("\n" + "="*80)
    print("TRAINING HISTORY:")
    print("="*80)
    with open(history_path) as f:
        history = json.load(f)

    print(f"Keys in history: {list(history.keys())}")

    if 'final_metrics' in history:
        final_metrics = history['final_metrics']
        print(f"\nFinal metrics count: {len(final_metrics)}")

        # Check for utilization
        util_in_history = {k: v for k, v in final_metrics.items() if 'utilization' in k.lower()}
        if util_in_history:
            print(f"\nUtilization metrics in history: {len(util_in_history)}")
            for key, value in sorted(util_in_history.items())[:10]:
                print(f"  {key}: {value}")
        else:
            print("\nNO UTILIZATION METRICS IN HISTORY!")

# Check codebook EMA cluster sizes (this is the REAL utilization)
print("\n" + "="*80)
print("CODEBOOK EMA CLUSTER SIZES (Real Utilization):")
print("="*80)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']

    # Find quantizer EMA cluster sizes
    ema_keys = [k for k in state_dict.keys() if 'ema_cluster_size' in k]
    print(f"\nFound {len(ema_keys)} quantizer codebooks")

    print("\nSample EMA cluster sizes (first 10 quantizers):")
    for key in sorted(ema_keys)[:10]:
        ema_cluster = state_dict[key]
        total_codes = len(ema_cluster)
        active_codes = (ema_cluster > 0).sum().item()
        utilization = active_codes / total_codes if total_codes > 0 else 0
        print(f"  {key}:")
        print(f"    Active: {active_codes}/{total_codes} ({100*utilization:.1f}%)")
        print(f"    Values: {ema_cluster.tolist()}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
print("The dashboard shows 0.00% because the training code doesn't compute")
print("and save per-quantizer utilization metrics in final_metrics.")
print("\nThe REAL utilization is in the ema_cluster_size tensors in model_state_dict.")
print("The visualization code needs to read from there, not from final_metrics.")
