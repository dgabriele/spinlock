#!/usr/bin/env python3
"""
Thin wrapper — tokenizer physics-alignment diagnostics.

The full implementation lives in:
    spinlock.visualization.vqvae.physics_alignment_dashboard

Answers: does the VQTokenizer discriminate physically distinct QBM samples?

Usage:
    poetry run python3 scripts/analysis/diagnose_tokenizer_physics_alignment.py \\
        --tokens datasets/qbm_50k_tokenized_multiresolution.h5 \\
        --params datasets/qbm_50k.h5 \\
        --output visualizations/qbm_physics_alignment/physics_alignment.png
"""

import sys
sys.path.insert(0, 'src')

import argparse
from pathlib import Path
from spinlock.visualization.vqvae import create_physics_alignment_dashboard


def main():
    parser = argparse.ArgumentParser(
        description="VQTokenizer physics-alignment diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interpretation:
  disc_ratio < 1.0  -> tokenizer discriminates that parameter
  knn_R2 > 0.1      -> tokens are predictive of that parameter
  MI > 0.05         -> category carries physics signal
        """,
    )
    parser.add_argument(
        "--tokens",
        required=True,
        help="Path to pretokenized HDF5 dataset",
    )
    parser.add_argument(
        "--params",
        required=True,
        help="Path to HDF5 with physical parameters (/parameters/params)",
    )
    parser.add_argument(
        "--output",
        default="visualizations/physics_alignment/physics_alignment.png",
        help="Output path for PNG figure",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=5000,
        help="Max samples for discrimination ratio computation (default: 5000)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display figure, only save",
    )
    args = parser.parse_args()

    if args.no_display:
        import matplotlib
        matplotlib.use("Agg")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_physics_alignment_dashboard(
        tokens_h5_path=args.tokens,
        params_h5_path=args.params,
        output_path=output_path,
        subsample=args.subsample,
    )


if __name__ == "__main__":
    main()
