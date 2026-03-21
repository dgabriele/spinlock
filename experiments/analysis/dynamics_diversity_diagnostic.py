#!/usr/bin/env python3
"""Phase 0 Diagnostic: Quantify token diversity and dynamics in Lenia datasets.

Analyses:
  1. Per-position token entropy — positions with near-zero entropy = many-to-one collapse
  2. Dynamics class distribution — FIXED_POINT : PERIODIC : APERIODIC : TRANSIENT ratios
  3. Token collision rate — identical temporal token vectors with different theta params

Usage:
    python experiments/analysis/dynamics_diversity_diagnostic.py \
        --pretokenized datasets/ds_lenia_fourier_10k_pretokenized.h5 \
        --raw datasets/ds_lenia_fourier_10k.h5 \
        [--truncation T256] [--level L0]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Dynamics diversity diagnostic (Phase 0)")
    p.add_argument("--pretokenized", required=True, help="Pretokenized HDF5 path")
    p.add_argument("--raw", required=True, help="Raw dataset HDF5 path")
    p.add_argument("--truncation", default="T256", help="Truncation length to analyze (default: T256)")
    p.add_argument("--level", default="L0", help="Hierarchy level (default: L0)")
    return p.parse_args()


# ── 1. Per-position Token Entropy ──


def compute_token_entropy(f: h5py.File, truncation: str, level: str):
    """Compute Shannon entropy per token position across all samples.

    Returns dict mapping position name → (entropy, num_unique, codebook_size).
    """
    tokens_group = f["tokens"]
    results = {}

    # Gather temporal, initial, theta position keys
    families = {"temporal": [], "initial": [], "theta": []}
    for key in sorted(tokens_group.keys()):
        if key.startswith("temporal_group_") and f"trunc_{truncation}" in key and key.endswith(f"_{level}"):
            families["temporal"].append(key)
        elif key.startswith("initial_spatial_") and key.endswith(f"_{level}"):
            families["initial"].append(key)
        elif key.startswith("theta_param_") and key.endswith(f"_{level}"):
            families["theta"].append(key)

    for family, keys in families.items():
        for key in keys:
            tokens = tokens_group[key][:]  # [N]
            unique, counts = np.unique(tokens, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-20))
            max_entropy = np.log2(len(unique)) if len(unique) > 1 else 0.0
            results[key] = {
                "family": family,
                "entropy": entropy,
                "max_entropy": max_entropy,
                "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
                "num_unique": len(unique),
                "total_samples": len(tokens),
                "top_token_frac": counts.max() / counts.sum(),
            }

    return results


def report_entropy(results: dict):
    """Print entropy analysis grouped by family."""
    print("\n" + "=" * 70)
    print("1. PER-POSITION TOKEN ENTROPY")
    print("=" * 70)

    for family in ["temporal", "initial", "theta"]:
        family_results = {k: v for k, v in results.items() if v["family"] == family}
        if not family_results:
            continue

        entropies = [v["entropy"] for v in family_results.values()]
        norm_entropies = [v["normalized_entropy"] for v in family_results.values()]
        top_fracs = [v["top_token_frac"] for v in family_results.values()]

        print(f"\n  {family.upper()} ({len(family_results)} positions):")
        print(f"    Entropy:      mean={np.mean(entropies):.3f}, "
              f"min={np.min(entropies):.3f}, max={np.max(entropies):.3f}")
        print(f"    Norm entropy: mean={np.mean(norm_entropies):.3f}, "
              f"min={np.min(norm_entropies):.3f}, max={np.max(norm_entropies):.3f}")
        print(f"    Top token %:  mean={np.mean(top_fracs):.1%}, "
              f"max={np.max(top_fracs):.1%}")

        # Flag low-entropy positions (< 50% of max possible entropy)
        low_ent = [(k, v) for k, v in family_results.items() if v["normalized_entropy"] < 0.5]
        if low_ent:
            print(f"    WARNING: {len(low_ent)} low-entropy positions (norm < 0.5):")
            for k, v in sorted(low_ent, key=lambda x: x[1]["normalized_entropy"]):
                print(f"      {k}: H={v['entropy']:.3f} (norm={v['normalized_entropy']:.3f}, "
                      f"top={v['top_token_frac']:.1%}, unique={v['num_unique']})")


# ── 2. Dynamics Class Distribution ──


def report_dynamics_classes(f: h5py.File):
    """Report dynamics class distribution from raw dataset metadata."""
    print("\n" + "=" * 70)
    print("2. DYNAMICS CLASS DISTRIBUTION")
    print("=" * 70)

    if "metadata" not in f or "dynamics_classes" not in f["metadata"]:
        print("  [No dynamics_classes in metadata — was classify_dynamics enabled?]")
        return {}

    classes = f["metadata"]["dynamics_classes"][:]
    # Decode bytes if needed
    if hasattr(classes[0], 'decode'):
        classes = np.array([c.decode() for c in classes])

    counts = Counter(classes)
    total = len(classes)

    print(f"\n  Total samples: {total}")
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:15s}: {count:6d}  ({count/total:6.2%})")

    return dict(counts)


# ── 3. Token Collision Rate ──


def analyze_token_collisions(
    pretok_f: h5py.File,
    raw_f: h5py.File,
    truncation: str,
    level: str,
):
    """Find samples with identical temporal tokens but different theta params.

    A collision means the tokenizer maps different parameter configs to the
    same temporal representation — the inverse mapping is ill-conditioned there.
    """
    print("\n" + "=" * 70)
    print("3. TOKEN COLLISION ANALYSIS")
    print("=" * 70)

    tokens_group = pretok_f["tokens"]

    # Build temporal token matrix [N, num_temporal_positions]
    temporal_keys = sorted([
        k for k in tokens_group.keys()
        if k.startswith("temporal_group_") and f"trunc_{truncation}" in k and k.endswith(f"_{level}")
    ])

    if not temporal_keys:
        print(f"  No temporal tokens found for {truncation}/{level}")
        return

    N = tokens_group[temporal_keys[0]].shape[0]
    num_pos = len(temporal_keys)
    print(f"\n  Building temporal token matrix: [{N}, {num_pos}]...")

    token_matrix = np.zeros((N, num_pos), dtype=np.int32)
    for i, key in enumerate(temporal_keys):
        token_matrix[:, i] = tokens_group[key][:]

    # Find unique token vectors and their indices
    # Convert to tuple-hashable form
    print("  Finding unique token vectors...")
    token_to_indices: dict[tuple, list[int]] = {}
    for idx in range(N):
        key = tuple(token_matrix[idx])
        if key not in token_to_indices:
            token_to_indices[key] = []
        token_to_indices[key].append(idx)

    num_unique = len(token_to_indices)
    collision_groups = {k: v for k, v in token_to_indices.items() if len(v) > 1}
    num_collision_groups = len(collision_groups)
    num_colliding_samples = sum(len(v) for v in collision_groups.values())

    print(f"\n  Unique temporal token vectors: {num_unique} / {N} ({num_unique/N:.1%})")
    print(f"  Collision groups (>1 sample with same tokens): {num_collision_groups}")
    print(f"  Samples involved in collisions: {num_colliding_samples} ({num_colliding_samples/N:.1%})")

    if not collision_groups:
        print("  No collisions — each sample has a unique temporal token vector.")
        return

    # Analyze: are collisions between same-param realizations or different params?
    # In the pretokenized file, samples are expanded: idx = param_idx * M + realization_idx
    # We need to figure out M (num_realizations)
    if "parameters" in raw_f and "params" in raw_f["parameters"]:
        num_params = raw_f["parameters"]["params"].shape[0]
        M = N // num_params  # realizations per param
        print(f"\n  Realizations per param config: {M} (N={N}, params={num_params})")

        # Classify collision types
        same_param_collisions = 0
        cross_param_collisions = 0
        cross_param_collision_groups = 0

        params = raw_f["parameters"]["params"][:]  # [num_params, D]

        for token_vec, indices in collision_groups.items():
            param_indices = set(idx // M for idx in indices)
            if len(param_indices) == 1:
                same_param_collisions += len(indices)
            else:
                cross_param_collisions += len(indices)
                cross_param_collision_groups += 1

                # Compute theta L2 distances for cross-param collision pairs
                if cross_param_collision_groups <= 5:
                    pidxs = sorted(param_indices)
                    dists = []
                    for i_p in range(len(pidxs)):
                        for j_p in range(i_p + 1, len(pidxs)):
                            d = np.linalg.norm(params[pidxs[i_p]] - params[pidxs[j_p]])
                            dists.append(d)
                    print(f"    Cross-param collision: {len(pidxs)} params, "
                          f"L2 distances: {[f'{d:.3f}' for d in dists[:5]]}")

        print(f"\n  Same-param collisions (realizations): {same_param_collisions} samples")
        print(f"  Cross-param collisions (PROBLEM): {cross_param_collisions} samples "
              f"in {cross_param_collision_groups} groups")

        if cross_param_collisions > 0:
            print(f"\n  ** {cross_param_collision_groups} groups of different parameter configs "
                  f"produce identical temporal tokens — ill-conditioned inverse. **")
        else:
            print("\n  All collisions are between realizations of the same param config — "
                  "the tokenizer correctly distinguishes different params!")

    # Distribution of collision group sizes
    sizes = sorted([len(v) for v in collision_groups.values()], reverse=True)
    print(f"\n  Collision group size distribution:")
    print(f"    Top 10: {sizes[:10]}")
    print(f"    Mean: {np.mean(sizes):.1f}, Median: {np.median(sizes):.0f}, Max: {max(sizes)}")


# ── Main ──


def main():
    args = parse_args()

    pretok_path = Path(args.pretokenized)
    raw_path = Path(args.raw)

    if not pretok_path.exists():
        print(f"Error: pretokenized file not found: {pretok_path}")
        sys.exit(1)
    if not raw_path.exists():
        print(f"Error: raw dataset not found: {raw_path}")
        sys.exit(1)

    print(f"Dynamics Diversity Diagnostic (Phase 0)")
    print(f"  Pretokenized: {pretok_path}")
    print(f"  Raw dataset:  {raw_path}")
    print(f"  Truncation:   {args.truncation}")
    print(f"  Level:        {args.level}")

    with h5py.File(pretok_path, "r") as pretok_f, h5py.File(raw_path, "r") as raw_f:
        # 1. Token entropy
        entropy_results = compute_token_entropy(pretok_f, args.truncation, args.level)
        report_entropy(entropy_results)

        # 2. Dynamics class distribution
        dynamics_counts = report_dynamics_classes(raw_f)

        # 3. Token collision rate
        analyze_token_collisions(pretok_f, raw_f, args.truncation, args.level)

    # Summary
    print("\n" + "=" * 70)
    print("DECISION GATE")
    print("=" * 70)
    print("""
  If cross-param token collisions are RARE (< 1%):
    → Tokenizer discriminability is adequate; perturbation probing may
      still improve D3PM training by enriching the feature landscape.

  If cross-param token collisions are COMMON (> 5%):
    → Tokenizer is losing information; perturbation probing (Phase 1)
      is critical to improve discriminability.

  If most samples are FIXED_POINT with near-zero temporal entropy:
    → Dataset is dominated by convergent dynamics; the D3PM is spending
      most capacity on a flat plateau. Perturbation will help reveal
      hidden dynamical repertoire.
""")


if __name__ == "__main__":
    main()
