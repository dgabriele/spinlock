"""Physics-alignment diagnostics dashboard.

Answers the central question: does the VQTokenizer discriminate physically
distinct QBM samples, or does high token similarity reflect genuine physics
invariants?

Four panels:
  1. Discrimination ratios per parameter (bar chart)
  2. MI heatmap — token categories × physical parameters
  3. KNN R² parameter recovery per parameter (bar chart)
  4. Category scatter — utilization × max MI × classification

Usage::

    from spinlock.visualization.vqvae import create_physics_alignment_dashboard

    fig = create_physics_alignment_dashboard(
        tokens_h5_path="datasets/qbm_50k_tokenized.h5",
        params_h5_path="datasets/qbm_50k.h5",
        output_path="visualizations/physics_alignment.png",
    )
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def create_physics_alignment_dashboard(
    tokens_h5_path: str,
    params_h5_path: str,
    output_path: Optional[str | Path] = None,
    param_names: Optional[List[str]] = None,
    subsample: int = 5000,
    figsize: Tuple[int, int] = (20, 16),
    dpi: int = 150,
) -> Figure:
    """2×2 dashboard: does the tokenizer capture physical variation?

    Panel 1 (top-left):    Discrimination ratios per parameter (bar chart)
    Panel 2 (top-right):   MI heatmap — n_categories × n_params
    Panel 3 (bottom-left): KNN R² parameter recovery per parameter (bar chart)
    Panel 4 (bottom-right):Category scatter — utilization × max MI × classification

    Prints summary metrics to stdout.

    Args:
        tokens_h5_path: Path to pretokenized HDF5 dataset
        params_h5_path: Path to HDF5 with physical parameters (/parameters/params)
        output_path: If given, save PNG to this path
        param_names: Optional override for parameter names (auto-detected if None)
        subsample: Max samples to use for discrimination ratios (speed)
        figsize: Figure size in inches
        dpi: Figure resolution

    Returns:
        Matplotlib Figure object
    """
    from .utils import (
        load_token_matrix,
        load_physics_params,
        compute_discrimination_ratios,
        compute_category_param_mi,
        compute_knn_param_recovery,
        classify_token_categories,
    )
    from .components import (
        plot_discrimination_bars,
        plot_metric_heatmap,
        plot_metric_bars,
        plot_category_scatter,
    )
    from spinlock.visualization.vqvae.utils import get_utilization_cmap

    print("=" * 70)
    print("PHYSICS-ALIGNMENT DIAGNOSTIC DASHBOARD")
    print("=" * 70)

    # ── Data loading ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading tokens...")
    tokens, token_keys = load_token_matrix(tokens_h5_path)

    print("\n[2/6] Loading physical parameters...")
    params, p_names = load_physics_params(params_h5_path, param_names)
    n_cats = tokens.shape[1]
    n_params = params.shape[1]

    # ── Analysis ──────────────────────────────────────────────────────────────
    print(f"\n[3/6] Computing discrimination ratios (subsample={subsample:,})...")
    ratios = compute_discrimination_ratios(
        tokens, params, p_names, subsample=subsample
    )

    print(f"\n[4/6] Computing MI matrix ({n_cats} cats × {n_params} params)...")
    mi_raw, mi_norm = compute_category_param_mi(tokens, params, p_names)

    print(f"\n[5/6] KNN parameter recovery (R²)...")
    r2_per_param = compute_knn_param_recovery(tokens, params)

    print(f"\n[6/6] Classifying token categories...")
    classifications = classify_token_categories(tokens, mi_norm)
    n_unique_per_cat = np.array([len(np.unique(tokens[:, c])) for c in range(n_cats)])
    max_mi_per_cat = mi_norm.max(axis=1)

    # ── Summary stats ─────────────────────────────────────────────────────────
    from collections import Counter
    cls_counts = Counter(classifications)
    print("\n" + "─" * 70)
    print("SUMMARY")
    print("─" * 70)
    print(f"  Total categories: {n_cats}")
    print(f"  Collapsed (1 code):          {cls_counts['collapsed']:3d} ({100*cls_counts['collapsed']/n_cats:.1f}%)")
    print(f"  Physics-informative (MI>0.05):{cls_counts['physics-informative']:3d} ({100*cls_counts['physics-informative']/n_cats:.1f}%)")
    print(f"  Conserved (multi-code, low MI):{cls_counts['conserved']:3d} ({100*cls_counts['conserved']/n_cats:.1f}%)")
    print()
    for p_name, r2 in zip(p_names, r2_per_param):
        ratio = ratios.get(p_name, 1.0)
        print(f"  {p_name:20s}  disc_ratio={ratio:.3f}  knn_R²={r2:.3f}")
    print("─" * 70)
    print("Interpretation:")
    print("  disc_ratio < 1.0  → tokenizer discriminates this parameter")
    print("  knn_R² > 0.1      → tokens are predictive of this parameter")
    print("  MI > 0.05         → category carries physics signal")
    if cls_counts["physics-informative"] > 0:
        print(f"\n  ✅ {cls_counts['physics-informative']} physics-informative categories found — tokenizer captures physics signal")
    else:
        print(f"\n  ⚠  No physics-informative categories — tokenizer may be physics-blind")
    print("─" * 70)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "VQTokenizer Physics-Alignment Diagnostics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Discrimination ratios
    ax1 = axes[0, 0]
    plot_discrimination_bars(ax1, p_names, ratios)

    # Panel 2: MI heatmap
    ax2 = axes[0, 1]
    # For readability: show max MI per category (condensed row) if many categories
    if n_cats > 60:
        # Show aggregate: max MI across all categories for each param as a bar
        max_mi_per_param = mi_norm.max(axis=0)
        plot_metric_bars(
            ax2,
            labels=p_names,
            values=max_mi_per_param.tolist(),
            title=f"Max Normalized MI per Parameter\n(across {n_cats} categories)",
            ylabel="Max Normalized MI",
            colormap="plasma",
            vmin=0.0,
            vmax=max(0.01, float(max_mi_per_param.max())),
        )
    else:
        # Full heatmap
        cmap = get_utilization_cmap()
        plot_metric_heatmap(
            ax2,
            matrix=mi_norm,
            row_labels=[f"cat{c}" for c in range(n_cats)],
            col_labels=p_names,
            title="Normalized MI: Categories × Parameters",
            cmap="plasma",
            vmin=0.0,
            vmax=max(0.01, float(mi_norm.max())),
            annotations=None,
            colorbar_label="Normalized MI",
            fmt=".2f",
        )

    # Panel 3: KNN R²
    ax3 = axes[1, 0]
    plot_metric_bars(
        ax3,
        labels=p_names,
        values=r2_per_param.tolist(),
        title="KNN Parameter Recovery (R²)\nTokens → Physics Parameters",
        ylabel="R² Score",
        colormap="RdYlGn",
        vmin=0.0,
        vmax=max(0.01, float(r2_per_param.max())),
    )
    ax3.axhline(y=0.1, color="blue", linestyle="--", linewidth=1.2, alpha=0.7, label="R²=0.1 threshold")
    ax3.legend(fontsize=8)

    # Panel 4: Category scatter
    ax4 = axes[1, 1]
    plot_category_scatter(
        ax4,
        n_unique=n_unique_per_cat,
        max_mi=max_mi_per_cat,
        classifications=classifications,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        png_path = output_path if output_path.suffix == ".png" else output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        print(f"\n  ✓ Saved: {png_path}")

    print("=" * 70)
    print("✓ Physics-alignment dashboard complete")
    print("=" * 70)
    return fig
