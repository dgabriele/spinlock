#!/usr/bin/env python3
"""
Validate PCA input quality for v4 OPQ grouping.

Uses VQTokenizer.clean_features() (now public) to apply the exact same
preprocessing pipeline as training, then compares:

  A. mean(dim=1)                       — old path (expected rank-1 collapse)
  B. cat([mean, std])                  — new v4 path, raw scale
  C. cat([mean, std]) standardized     — standard PCA preprocessing
  D. std(dim=1)                        — dynamics channel alone

"Effective rank" for the 30-group setup: want PCs-for-90% > 10.

Usage:
    poetry run python3 scripts/analysis/validate_pca_grouping_input.py \
        --config configs/qbm/vqvae_v4_opq.yaml \
        --dataset datasets/qbm_50k.h5 \
        [--samples 5000]
"""

import sys
sys.path.insert(0, 'src')

import argparse
import warnings
import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import yaml
from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.tokens.config import TokenizerConfig


def load_raw_features(h5_path: str, max_samples: int) -> dict:
    """Load raw feature dict suitable for VQTokenizer.clean_features()."""
    with h5py.File(h5_path, "r") as f:
        ds_t = f["features/temporal/features"]
        N_total = ds_t.shape[0]
        N = min(max_samples, N_total)

        temporal = torch.from_numpy(ds_t[:N].astype(np.float32))  # [N, T, D_t]

        # Load initial_raw for semantic feature name detection
        initial_raw = None
        if "inputs/fields" in f:
            initial_raw = torch.from_numpy(
                f["inputs/fields"][:N].astype(np.float32)
            )

    print(f"Loaded temporal: {tuple(temporal.shape)}  (from {N_total} total)")
    if initial_raw is not None:
        print(f"Loaded initial_raw: {tuple(initial_raw.shape)}")

    return {
        "temporal": temporal,
        "initial_raw": initial_raw,
        "initial_manual": None,
        "temporal_mask": None,
        "temporal_lengths": None,
        "theta": None,
    }


def pca_top5(X: np.ndarray, label: str) -> dict:
    """Fit PCA and print/return top-5 explained variance ratios."""
    N, D = X.shape
    n_components = min(D, 60, N - 1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)

    ev = pca.explained_variance_ratio_
    cumulative = np.cumsum(ev)
    pcs_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    pcs_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    quality = "GOOD" if pcs_90 > 5 else ("OK" if pcs_90 > 2 else "BAD — collapsed")

    print(f"\n{label}  [{N} × {D}]")
    print(f"  Top-5 explained variance: {[f'{v:.4f}' for v in ev[:5]]}")
    print(f"  PCs for 90% variance:  {pcs_90}  ({quality} for 30-group setup)")
    print(f"  PCs for 99% variance:  {pcs_99}")

    return {"pcs_90": pcs_90, "pcs_99": pcs_99, "top5": ev[:5].tolist()}


def main():
    parser = argparse.ArgumentParser(description="Validate PCA grouping input quality")
    parser.add_argument("--config", default="configs/qbm/vqvae_v4_opq.yaml")
    parser.add_argument("--dataset", default="datasets/qbm_50k.h5")
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()

    print("=" * 65)
    print("PCA Grouping Input Validation  (uses VQTokenizer.clean_features)")
    print("=" * 65)

    # Build a minimal tokenizer instance just for clean_features()
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = TokenizerConfig(**cfg_dict)
    tokenizer = VQTokenizer(config=cfg)

    # Load raw data
    raw_features = load_raw_features(args.dataset, args.samples)
    temporal_raw = raw_features["temporal"]  # [N, T, D_t]
    N, T, D_t = temporal_raw.shape

    # Quick NaN summary on raw input
    nan_total = torch.isnan(temporal_raw).sum().item()
    nan_cols = torch.isnan(temporal_raw).any(dim=(0, 1))
    print(f"\nRaw temporal: {nan_total} NaN values across {nan_cols.sum().item()} feature columns")

    # Apply the EXACT same cleaning as VQTokenizer.fit()
    print("\nRunning VQTokenizer.clean_features()...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cleaned = tokenizer.clean_features(raw_features)

    temporal_clean = cleaned["temporal"]  # [N, T, D_kept]
    D_kept = temporal_clean.shape[2]

    # Convert to numpy for PCA
    t = temporal_clean.numpy()  # [N, T, D_kept]

    mean_np = t.mean(axis=1)              # [N, D_kept]
    std_np  = t.std(axis=1)              # [N, D_kept]
    cat_np  = np.concatenate([mean_np, std_np], axis=1)  # [N, 2*D_kept]

    # Standardized version of cat (zero-mean, unit-var per feature)
    scaler = StandardScaler()
    cat_std = scaler.fit_transform(cat_np)  # [N, 2*D_kept]

    print("\n" + "=" * 65)
    print("PCA Variance Profiles (on cleaned features)")
    print("=" * 65)
    results = {}
    results["A"] = pca_top5(mean_np, "A: mean(dim=1)          [OLD — expected rank-1]")
    results["B"] = pca_top5(cat_np,  "B: cat([mean,std])       [NEW v4, raw scale]")
    results["C"] = pca_top5(cat_std, "C: cat([mean,std]) std'd [standardized PCA]")
    results["D"] = pca_top5(std_np,  "D: std(dim=1)            [dynamics only]")

    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  Config:               {args.config}")
    print(f"  Dataset:              {args.dataset}")
    print(f"  Samples:              {N}")
    print(f"  D_t (original):       {D_t}")
    print(f"  D_kept (after clean): {D_kept}")
    print(f"  cat([m,s]) dim:       {2 * D_kept}")
    print(f"  Groups M:             30")
    print(f"  Dim per group:        {2*D_kept}/30 ≈ {2*D_kept/30:.1f}")
    print()
    print("  Effective rank (PCs for 90%):")
    for k, v in results.items():
        bar = "#" * v["pcs_90"]
        print(f"    {k}: {v['pcs_90']:3d}  {bar}")
    print()
    if results["C"]["pcs_90"] > results["B"]["pcs_90"]:
        print("  ★ Standardization helps — consider adding StandardScaler in PCAGrouper.")
    elif results["B"]["pcs_90"] > results["A"]["pcs_90"]:
        print("  ★ cat([mean,std]) improves rank over mean-only.")
    else:
        print("  ★ Dataset appears near-rank-1 regardless of temporal summary.")
        print("    Consider: wider parameter ranges, more diverse dataset, or")
        print("    temporally-stratified features (early/late trajectory stats).")


if __name__ == "__main__":
    main()
