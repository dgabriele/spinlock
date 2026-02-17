#!/usr/bin/env python3
"""
Thin wrapper — binned token similarity visualization.

The full implementation lives in:
    spinlock.visualization.vqvae.binned_similarity_dashboard

Supports two metrics:
  --metric jaccard   Set overlap |A∩B|/|A∪B|  (presence/absence)
  --metric js        Jensen-Shannon divergence  (frequency-sensitive)
"""

import sys
sys.path.insert(0, 'src')

import argparse
from pathlib import Path
from spinlock.visualization.vqvae import create_binned_similarity_dashboard


def main():
    parser = argparse.ArgumentParser(
        description="Binned token similarity visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  jaccard  Set overlap |A∩B|/|A∪B| — presence/absence only
  js       Jensen-Shannon divergence — sensitive to code frequency distributions
        """,
    )
    parser.add_argument(
        "--dataset",
        default="datasets/qbm_50k_pretokenized.h5",
        help="Input pretokenized HDF5 dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/vqvae_fixed_metric_50k_binned",
        help="Output directory for visualizations",
    )
    parser.add_argument("--n-bins", type=int, default=5000)
    parser.add_argument(
        "--metric",
        choices=["jaccard", "js"],
        default="jaccard",
        help="Similarity metric (default: jaccard)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        metavar="FAMILY",
        help=(
            "Token families to include (e.g. --families temporal). "
            "Omit to include all families (temporal + initial + theta). "
            "Use 'temporal' to isolate the temporal encoder's diversity, "
            "excluding parameter/IC tokens that inflate baseline similarity."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    families_tag = "_" + "+".join(sorted(args.families)) if args.families else ""
    metric_tag = "jaccard" if args.metric == "jaccard" else "js"
    stem = f"rollout_similarity_{metric_tag}{families_tag}"
    output_path = output_dir / f"{stem}.png"

    create_binned_similarity_dashboard(
        tokenized_h5_path=args.dataset,
        output_path=output_path,
        n_bins=args.n_bins,
        metric=args.metric,
        families=args.families,
    )


if __name__ == "__main__":
    main()
