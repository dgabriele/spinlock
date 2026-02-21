#!/usr/bin/env python3
"""Publication-quality VQTokenizer pipeline visualizations.

Generates 7 figures documenting the CNO v1 VQTokenizer pipeline.

Usage:
    poetry run python scripts/visualize_vqtokenizer_pipeline.py
    poetry run python scripts/visualize_vqtokenizer_pipeline.py \
        --checkpoint checkpoints/cno/v1/vq_tokenizer_best.pt \
        --dataset datasets/50k_baseline.h5 \
        --output visualizations/vqtokenizer_pipeline
"""

import sys
sys.path.insert(0, 'src')

import argparse
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.decomposition import PCA
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})
DPI = 200
C_TEMPORAL = '#2471a3'
C_INITIAL  = '#27ae60'
C_THETA    = '#8e44ad'
C_REMOVED  = '#c0392b'
C_DEDUP    = '#e67e22'
C_KEPT     = '#27ae60'
GRID_ALPHA = 0.3


# ── Data loading ───────────────────────────────────────────────────────────────

def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_dataset_subset(path, n=5000):
    with h5py.File(path, 'r') as f:
        temporal = f['features/temporal/features'][:n]   # [N, T, D_raw]
        initial  = f['features/initial/aggregated/features'][:n]
    return temporal, initial


# ── Fig 1 — Feature Extraction ────────────────────────────────────────────────

def fig1_feature_extraction(ckpt, dataset_path, out_dir):
    """Horizontal stacked bar of sub-extractor composition + I/O schematic."""
    import json as _json
    fm = ckpt['feature_metadata']['families']['temporal']
    n_raw = fm['original_feature_count']  # 345 for 3ch CNO

    # Load summary feature registry from HDF5 for semantic category breakdown
    sub_extractors = None
    try:
        with h5py.File(dataset_path, 'r') as f:
            reg_json = f['features/summary'].attrs.get('feature_registry', None)
            if reg_json is not None:
                reg = _json.loads(reg_json)
                # Map category names → human-readable labels + colors
                cat_style = {
                    'spatial':       ('Spatial statistics',   '#aed6f1'),
                    'spectral':      ('Spectral & entropy',   '#a9dfbf'),
                    'cross_channel': ('Cross-channel',        '#f9e79f'),
                    'temporal':      ('Temporal dynamics',    '#f5cba7'),
                }
                sub_extractors = []
                for cat, (label, color) in cat_style.items():
                    count = len(reg.get(cat, {}))
                    if count > 0:
                        sub_extractors.append((label, count, color))
    except Exception:
        pass

    # Fallback: approximate breakdown
    if not sub_extractors:
        sub_extractors = [
            ('Spatial statistics', int(n_raw * 0.26), '#aed6f1'),
            ('Spectral & entropy', int(n_raw * 0.30), '#a9dfbf'),
            ('Cross-channel',      int(n_raw * 0.17), '#f9e79f'),
            ('Temporal dynamics',  n_raw - int(n_raw * 0.73), '#f5cba7'),
        ]

    # Determine if using real registry data
    registry_total = sum(c for _, c, _ in sub_extractors)
    using_registry = abs(registry_total - n_raw) > 10  # True if summary registry ≠ raw count

    fig, (ax_bar, ax_schema) = plt.subplots(1, 2, figsize=(10, 5),
                                             gridspec_kw={'width_ratios': [1.6, 1]})
    fig.suptitle('Fig 1 — Feature Extraction: TemporalFeatureOrchestrator',
                 fontsize=12, fontweight='bold', y=1.01)

    # Left: stacked horizontal bar
    left = 0
    for name, count, color in sub_extractors:
        ax_bar.barh(0, count, left=left, height=0.5, color=color,
                    edgecolor='#555', linewidth=0.8, label=f'{name} ({count})')
        if count >= 12:
            ax_bar.text(left + count / 2, 0, f'{count}',
                        ha='center', va='center', fontsize=9, fontweight='bold')
        left += count

    ax_bar.set_xlim(0, max(n_raw, registry_total) + 5)
    ax_bar.set_ylim(-0.6, 0.8)
    ax_bar.set_xlabel('Feature count', fontsize=9)
    ax_bar.set_yticks([])
    title_note = f'({registry_total} summary features, HDF5 registry)' if using_registry else f'({n_raw} per-timestep features, 3-channel CNO)'
    ax_bar.set_title(f'Sub-extractor composition  {title_note}', fontsize=10, fontweight='bold')
    ax_bar.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax_bar.grid(axis='x', alpha=GRID_ALPHA)
    if using_registry:
        ax_bar.axvline(n_raw, color='#c0392b', linestyle='--', linewidth=1.2, alpha=0.7)
        ax_bar.text(n_raw + 2, -0.45, f'Per-timestep: {n_raw}D', fontsize=7.5,
                    color='#c0392b', style='italic')
        ax_bar.text(max(n_raw, registry_total) / 2, -0.52,
                    '* Breakdown from HDF5 summary feature registry (aggregated features)',
                    ha='center', fontsize=7, color='#888', style='italic')
    else:
        ax_bar.text(n_raw / 2, -0.5, '* Breakdown estimated — semantic names require retrain',
                    ha='center', fontsize=7, color='#888', style='italic')

    # Right: I/O schema with FancyBboxPatch
    ax_schema.axis('off')
    ax_schema.set_xlim(0, 1)
    ax_schema.set_ylim(0, 1)
    ax_schema.set_title('Data flow', fontsize=10, fontweight='bold')

    def schematic_box(ax, x, y, w, h, label, sublabel, facecolor, edgecolor):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                              facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
                              transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.65, label,
                ha='center', va='center', fontsize=9, fontweight='bold',
                transform=ax.transAxes)
        ax.text(x + w / 2, y + h * 0.3, sublabel,
                ha='center', va='center', fontsize=7.5, color='#444',
                transform=ax.transAxes)

    arrow_kw = dict(arrowstyle='->', color='#555', lw=1.5,
                    connectionstyle='arc3,rad=0.0')

    schematic_box(ax_schema, 0.1, 0.72, 0.8, 0.18,
                  'CNO Rollout Fields', '[T, 3, 64, 64] per timestep',
                  '#d6eaf8', C_TEMPORAL)
    ax_schema.annotate('', xy=(0.5, 0.70), xytext=(0.5, 0.72),
                       arrowprops=arrow_kw, xycoords='axes fraction', textcoords='axes fraction')
    schematic_box(ax_schema, 0.1, 0.47, 0.8, 0.20,
                  'TemporalFeatureOrchestrator',
                  'spatial · spectral · cross-channel · dynamics',
                  '#fef9e7', '#d4ac0d')
    ax_schema.annotate('', xy=(0.5, 0.45), xytext=(0.5, 0.47),
                       arrowprops=arrow_kw, xycoords='axes fraction', textcoords='axes fraction')
    schematic_box(ax_schema, 0.1, 0.27, 0.8, 0.16,
                  'Temporal Features', f'[T, {n_raw}D] per sample',
                  '#d6eaf8', C_TEMPORAL)

    plt.tight_layout()
    out = out_dir / '01_feature_extraction.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


# ── Fig 2 — Feature Cleaning ──────────────────────────────────────────────────

def fig2_feature_cleaning(ckpt, temporal_raw, out_dir):
    """Cleaning funnel + variance histogram."""
    fm = ckpt['feature_metadata']['families']['temporal']
    ops = fm['cleaning_operations']

    n_raw      = fm['original_feature_count']  # 345
    n_final    = fm['cleaned_feature_count']     # 257

    # Reconstruct per-op counts
    stage_counts = [n_raw]
    stage_labels = [f'Raw\n{n_raw}']
    stage_colors = ['#5d6d7e']

    removed_by_op = {}
    for op in ops:
        if op['num_removed'] > 0:
            removed_by_op[op['operation_type']] = set(op['features_removed'])
            n_after = op['num_kept']
            stage_counts.append(n_after)
            op_name = {'variance_filter': 'Variance\nfilter', 'deduplication': 'Deduplication'}.get(
                op['operation_type'], op['operation_type'])
            label_color = C_REMOVED if op['operation_type'] == 'variance_filter' else C_DEDUP
            stage_labels.append(f'{op_name}\n−{op["num_removed"]}  →  {n_after}')
            stage_colors.append(label_color)

    removed_variance = removed_by_op.get('variance_filter', set())
    removed_dedup    = removed_by_op.get('deduplication', set())
    kept_set         = set(fm['kept_feature_indices'])

    fig, (ax_funnel, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Fig 2 — Feature Cleaning Pipeline', fontsize=12, fontweight='bold', y=1.01)

    # Left: vertical flow funnel
    ax_funnel.axis('off')
    ax_funnel.set_xlim(0, 1)
    ax_funnel.set_ylim(0, 1)
    ax_funnel.set_title('Cleaning funnel', fontsize=10, fontweight='bold')

    n_stages = len(stage_counts)
    heights  = np.linspace(0.88, 0.38, n_stages)   # keep flow in upper 60% of axes
    max_bar  = max(stage_counts)
    bar_h    = 0.09

    for i, (cnt, lbl, col) in enumerate(zip(stage_counts, stage_labels, stage_colors)):
        w = 0.7 * cnt / max_bar
        x0 = 0.5 - w / 2
        box = FancyBboxPatch((x0, heights[i] - bar_h / 2), w, bar_h,
                              boxstyle='round,pad=0.01',
                              facecolor=col, edgecolor='white', linewidth=1.5,
                              transform=ax_funnel.transAxes, alpha=0.88)
        ax_funnel.add_patch(box)
        # Number label
        ax_funnel.text(0.5, heights[i], lbl.replace('\n', ' '),
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white', transform=ax_funnel.transAxes)
        # Arrow between stages
        if i < n_stages - 1:
            ax_funnel.annotate('', xy=(0.5, heights[i + 1] + bar_h / 2 + 0.01),
                               xytext=(0.5, heights[i] - bar_h / 2 - 0.01),
                               xycoords='axes fraction', textcoords='axes fraction',
                               arrowprops=dict(arrowstyle='->', color='#777', lw=1.5))

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#5d6d7e', label=f'Raw ({n_raw})'),
        Patch(facecolor=C_REMOVED, label=f'Variance-filtered ({len(removed_variance)})'),
        Patch(facecolor=C_DEDUP,   label=f'Deduplicated ({len(removed_dedup)})'),
        Patch(facecolor=C_KEPT,    label=f'Kept ({n_final})'),
    ]
    ax_funnel.legend(handles=legend_elems, loc='upper center',
                     bbox_to_anchor=(0.5, 0.22),   # place in the cleared lower strip
                     fontsize=8, framealpha=0.9, ncol=2)

    # Right: per-feature variance histogram
    # Compute variance over samples × timesteps
    N, T, D = temporal_raw.shape
    flat = temporal_raw.reshape(-1, D)  # [N*T, 345]
    variances = flat.var(axis=0)        # [345]

    var_threshold = ops[0]['parameters']['variance_threshold']

    # Classify each feature
    var_removed  = variances[list(removed_variance)]
    dedup_kept   = variances[[i for i in range(D) if i in kept_set or i in removed_dedup]]
    var_kept     = variances[list(kept_set)]
    var_dedup    = variances[list(removed_dedup)]

    bins_log = np.logspace(np.floor(np.log10(variances[variances > 0].min()) - 0.5),
                           np.ceil(np.log10(variances.max()) + 0.5), 60)

    ax_hist.hist(variances[list(kept_set)], bins=bins_log, color=C_KEPT,
                 alpha=0.75, label=f'Kept ({len(kept_set)})', zorder=3)
    ax_hist.hist(variances[list(removed_dedup)], bins=bins_log, color=C_DEDUP,
                 alpha=0.75, label=f'Dedup-removed ({len(removed_dedup)})', zorder=2)
    ax_hist.hist(variances[list(removed_variance)], bins=bins_log, color=C_REMOVED,
                 alpha=0.9, label=f'Variance-removed ({len(removed_variance)})', zorder=4)

    ax_hist.axvline(var_threshold, color='#c0392b', linestyle='--', linewidth=1.5,
                    label=f'Variance threshold ({var_threshold:.0e})')
    ax_hist.set_xscale('log')
    ax_hist.set_xlabel('Per-feature variance (log scale)', fontsize=9)
    ax_hist.set_ylabel('Feature count', fontsize=9)
    ax_hist.set_title(f'Variance distribution across {N:,} samples × {T} timesteps', fontsize=10)
    ax_hist.legend(fontsize=8, loc='upper left')
    ax_hist.grid(alpha=GRID_ALPHA)

    plt.tight_layout()
    out = out_dir / '02_feature_cleaning.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


# ── Fig 3 — PCA Grouping ──────────────────────────────────────────────────────

def fig3_pca_grouping(ckpt, temporal_raw, out_dir):
    """PCA scree + group size distribution + feature-group assignment matrix."""
    fm = ckpt['feature_metadata']['families']['temporal']
    kept_indices = fm['kept_feature_indices']  # indices into raw 345-dim
    gi = ckpt['group_indices']
    temporal_groups = {k: v for k, v in gi.items() if k.startswith('temporal_group_')}
    n_groups = len(temporal_groups)

    # Select kept features from raw
    temporal_clean = temporal_raw[:, :, kept_indices]   # [N, T, 257]
    N, T, D_clean = temporal_clean.shape

    # Aggregate per sample: mean + std over timesteps → [N, 2*D_clean]
    mu  = temporal_clean.mean(axis=1)   # [N, D_clean]
    sig = temporal_clean.std(axis=1)    # [N, D_clean]
    X   = np.concatenate([mu, sig], axis=1)  # [N, 2*D_clean]

    # PCA
    n_components = min(40, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    ev = pca.explained_variance_ratio_

    # Determine if group_indices are into cleaned or raw space
    all_group_indices = [idx for v in temporal_groups.values() for idx in v]
    max_idx = max(all_group_indices) if all_group_indices else 0
    # If max index < D_clean (257), they're in cleaned space; else raw space
    in_cleaned_space = max_idx < D_clean

    # Group sizes
    group_sizes = [len(temporal_groups.get(f'temporal_group_{g}', [])) for g in range(n_groups)]

    # Build assignment matrix [D_clean, n_groups]
    assign = np.zeros((D_clean, n_groups), dtype=np.uint8)
    for g in range(n_groups):
        indices = temporal_groups.get(f'temporal_group_{g}', [])
        for idx in indices:
            if in_cleaned_space and idx < D_clean:
                assign[idx, g] = 1
            elif not in_cleaned_space:
                # Map raw index → cleaned index
                if idx in kept_indices:
                    cleaned_idx = kept_indices.index(idx)
                    assign[cleaned_idx, g] = 1

    # Sort features to show diagonal block structure
    group_of_feature = assign.argmax(axis=1)  # which group each feature belongs to
    sort_order = np.argsort(group_of_feature, kind='stable')
    assign_sorted = assign[sort_order, :]

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle('Fig 3 — PCA-Based Feature Grouping', fontsize=12, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_scree  = fig.add_subplot(gs[0, 0])
    ax_sizes  = fig.add_subplot(gs[0, 1:])
    ax_matrix = fig.add_subplot(gs[1, :])

    # Scree plot — log y for individual bars (variance collapses to near-0 for later PCs)
    k_show = min(30, n_components)
    cumev  = np.cumsum(ev[:k_show])
    ax_scree.bar(range(1, k_show + 1), ev[:k_show] * 100, color='#aed6f1',
                 edgecolor='#2471a3', linewidth=0.5, label='Individual')
    ax_scree.set_yscale('log')
    ax2 = ax_scree.twinx()
    ax2.plot(range(1, k_show + 1), cumev * 100, color='#c0392b', linewidth=1.5,
             marker='o', markersize=2.5, label='Cumulative')
    ax_scree.set_xlabel('Principal component')
    ax_scree.set_ylabel('Explained variance (%, log)')
    ax2.set_ylabel('Cumulative (%)', color='#c0392b')
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='y', colors='#c0392b')
    ax_scree.set_title(f'PCA scree  ({N:,} samples)', fontsize=9, fontweight='bold')
    ax_scree.grid(alpha=GRID_ALPHA, axis='y', which='both')
    # Mark 90% and 99% cumulative thresholds
    for pct, ls in [(90, ':'), (99, '--')]:
        idx_t = np.searchsorted(cumev, pct / 100)
        if idx_t < k_show:
            ax2.axhline(pct, color='#c0392b', linestyle=ls, linewidth=0.9, alpha=0.6)
            ax2.text(k_show * 0.98, pct + 1, f'{pct}% @ PC{idx_t + 1}', ha='right',
                     fontsize=6.5, color='#c0392b')

    # Group size distribution
    cmap_groups = plt.get_cmap('tab20')
    colors_g = [cmap_groups(g / n_groups) for g in range(n_groups)]
    bars = ax_sizes.bar(range(n_groups), group_sizes, color=colors_g, edgecolor='white', linewidth=0.3)
    ax_sizes.set_xlabel('Group index')
    ax_sizes.set_ylabel('Feature count per group')
    ax_sizes.set_title(f'Group size distribution  ({n_groups} groups, {D_clean} features total)',
                       fontsize=9, fontweight='bold')
    ax_sizes.set_xticks(range(0, n_groups, 5))
    ax_sizes.axhline(np.mean(group_sizes), color='#555', linestyle='--', linewidth=1,
                     label=f'Mean: {np.mean(group_sizes):.1f}')
    ax_sizes.legend(fontsize=8)
    ax_sizes.grid(alpha=GRID_ALPHA, axis='y')

    # Assignment matrix
    im = ax_matrix.imshow(assign_sorted.T, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                           interpolation='nearest')
    ax_matrix.set_xlabel(f'Feature index (sorted by group,  {D_clean} total)', fontsize=8)
    ax_matrix.set_ylabel('Group index', fontsize=8)
    ax_matrix.set_title('Feature-group assignment  (sorted to show diagonal blocks)',
                         fontsize=9, fontweight='bold')
    ax_matrix.set_yticks(range(0, n_groups, 5))
    ax_matrix.grid(False)

    # Group boundary lines on matrix
    cumsum = 0
    for g in range(n_groups):
        cumsum += group_sizes[g]
        if g < n_groups - 1:
            ax_matrix.axvline(cumsum - 0.5, color='white', linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    out = out_dir / '03_pca_grouping.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


# ── Fig 4 — Pyramid Encoder ───────────────────────────────────────────────────

def fig4_pyramid_encoder(ckpt, out_dir):
    """PyramidTemporalEncoder structure for one illustrative group."""
    enc_cfg       = ckpt['config']['encoder']['temporal']
    level_dims    = enc_cfg['level_dims']          # [6, 12, 20, 26]
    ds_factors    = enc_cfg['downsample_factors']  # [1, 2, 4, 8]
    embedding_dim = ckpt['config']['encoder']['embedding_dim']  # 64

    gi = ckpt['group_indices']
    # Pick the group with median size for illustration
    temporal_groups = {k: v for k, v in gi.items() if k.startswith('temporal_group_')}
    sizes = [(k, len(v)) for k, v in temporal_groups.items()]
    sizes.sort(key=lambda x: x[1])
    ex_key, ex_size = sizes[len(sizes) // 2]  # median group
    ex_group_idx = int(ex_key.split('_')[-1])

    n_levels   = len(level_dims)
    T_example  = 256

    fig, axes = plt.subplots(n_levels + 1, 1, figsize=(10, 6),
                              gridspec_kw={'height_ratios': [1] * n_levels + [0.7]})
    fig.suptitle(f'Fig 4 — PyramidTemporalEncoder  '
                 f'(Group {ex_group_idx}, G_k={ex_size} features)',
                 fontsize=12, fontweight='bold', y=1.02)

    level_colors = ['#2471a3', '#1a9e73', '#7b5ea7', '#c0392b']

    for l, ax in enumerate(axes[:n_levels]):
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)

        ds = ds_factors[l]
        T_l = T_example // ds
        out_d = level_dims[l]
        color = level_colors[l]

        # Input box
        in_box = FancyBboxPatch((0.2, 0.15), 1.8, 0.7, boxstyle='round,pad=0.05',
                                 facecolor='#eaf2fb', edgecolor=color, linewidth=1.5)
        ax.add_patch(in_box)
        ax.text(1.1, 0.5, f'[T/{ds}, G_k]\n[{T_l}, {ex_size}]',
                ha='center', va='center', fontsize=8, fontweight='bold')

        # Arrow
        ax.annotate('', xy=(3.0, 0.5), xytext=(2.0, 0.5),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

        # Conv box
        conv_box = FancyBboxPatch((3.0, 0.15), 2.2, 0.7, boxstyle='round,pad=0.05',
                                   facecolor='#fdebd0', edgecolor='#e67e22', linewidth=1.5)
        ax.add_patch(conv_box)
        ax.text(4.1, 0.55, f'Conv1d → ResBlocks', ha='center', va='center', fontsize=8)
        ax.text(4.1, 0.32, f'→ GAP → 256D', ha='center', va='center', fontsize=7.5, color='#555')

        # Arrow
        ax.annotate('', xy=(6.2, 0.5), xytext=(5.2, 0.5),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

        # Linear projection
        proj_box = FancyBboxPatch((6.2, 0.15), 1.8, 0.7, boxstyle='round,pad=0.05',
                                   facecolor='#e8daef', edgecolor=C_THETA, linewidth=1.5)
        ax.add_patch(proj_box)
        ax.text(7.1, 0.5, f'Linear\n256D→{out_d}D',
                ha='center', va='center', fontsize=8, fontweight='bold')

        # Level label
        ax.text(9.5, 0.5, f'L{l}  ×{ds}', ha='center', va='center',
                fontsize=9, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, lw=1))

        # Waveform schematic (shrinking sine)
        x_wave = np.linspace(0, 1, max(8, T_l // 4))
        y_wave = 0.5 + 0.18 * np.sin(2 * np.pi * x_wave * (5 - l)) * (1.0 - l * 0.1)
        ax_twin = ax.inset_axes([0.02, 0.0, 0.17, 1.0])
        ax_twin.plot(x_wave, y_wave, color=color, linewidth=1.0)
        ax_twin.set_xlim(0, 1); ax_twin.set_ylim(0.1, 0.9)
        ax_twin.axis('off')

    # Bottom: concatenation bar
    ax_cat = axes[-1]
    ax_cat.axis('off')
    ax_cat.set_xlim(0, 10)
    ax_cat.set_ylim(0, 1)
    left = 1.5
    total = sum(level_dims)
    for l, (d, c) in enumerate(zip(level_dims, level_colors)):
        w = 7.0 * d / total
        cat_box = FancyBboxPatch((left, 0.15), w, 0.7, boxstyle='round,pad=0.02',
                                  facecolor=c, edgecolor='white', linewidth=1.2, alpha=0.85)
        ax_cat.add_patch(cat_box)
        ax_cat.text(left + w / 2, 0.5, f'L{l}\n{d}D',
                    ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
        left += w
    ax_cat.text(0.9, 0.5, 'Concat:', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_cat.text(8.7, 0.5, f'= {embedding_dim}D embedding', ha='left', va='center',
                fontsize=9, fontweight='bold', color='#555')

    plt.tight_layout()
    out = out_dir / '04_pyramid_encoder.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


# ── Fig 5 — Quantization Hierarchy ────────────────────────────────────────────

def fig5_quantization_hierarchy(ckpt, out_dir):
    """30×3 vocab-size heatmap + per-level total codes bar chart."""
    msd = ckpt['model_state_dict']
    gi  = ckpt['group_indices']
    temporal_groups = sorted(
        [k for k in gi if k.startswith('temporal_group_')],
        key=lambda k: int(k.split('_')[-1])
    )
    n_groups = len(temporal_groups)
    n_levels = 3

    # Extract vocab sizes [n_groups, n_levels]
    vocab = np.zeros((n_groups, n_levels), dtype=int)
    for g, gname in enumerate(temporal_groups):
        for l in range(n_levels):
            key = f'quantizers.{gname}_L{l}.embedding.weight'
            if key in msd:
                vocab[g, l] = msd[key].shape[0]

    embedding_dims = np.zeros((n_groups, n_levels), dtype=int)
    for g, gname in enumerate(temporal_groups):
        for l in range(n_levels):
            key = f'quantizers.{gname}_L{l}.embedding.weight'
            if key in msd:
                embedding_dims[g, l] = msd[key].shape[1]

    fig, (ax_hmap, ax_bar) = plt.subplots(1, 2, figsize=(12, 6),
                                           gridspec_kw={'width_ratios': [4, 1]})
    fig.suptitle('Fig 5 — Quantization Hierarchy: 90 Codebooks (30 groups × 3 levels)',
                 fontsize=12, fontweight='bold', y=1.02)

    # Heatmap
    im = ax_hmap.imshow(vocab, aspect='auto', cmap='Blues',
                         vmin=vocab.min() - 1, vmax=vocab.max())
    plt.colorbar(im, ax=ax_hmap, fraction=0.04, pad=0.02, label='Vocab size (# codes)')

    # Annotate cells
    for g in range(n_groups):
        for l in range(n_levels):
            v = vocab[g, l]
            d = embedding_dims[g, l]
            text_color = 'white' if v > (vocab.max() * 0.6) else '#222'
            ax_hmap.text(l, g, f'{v}', ha='center', va='center',
                         fontsize=7, fontweight='bold', color=text_color)

    ax_hmap.set_xticks([0, 1, 2])
    ax_hmap.set_xticklabels(['L0 (coarse)', 'L1 (medium)', 'L2 (fine)'], fontsize=9)
    ax_hmap.set_yticks(range(0, n_groups, 5))
    ax_hmap.set_yticklabels([f'G{g}' for g in range(0, n_groups, 5)], fontsize=8)
    ax_hmap.set_xlabel('Quantization level')
    ax_hmap.set_ylabel('Group index')
    ax_hmap.set_title(f'Vocab sizes per codebook  '
                      f'(emb_dim≈{embedding_dims[:, 0].mean():.0f}/{embedding_dims[:, 1].mean():.0f}/'
                      f'{embedding_dims[:, 2].mean():.0f}D for L0/L1/L2)',
                      fontsize=9, fontweight='bold')

    # Right: total codes per level
    total_per_level = vocab.sum(axis=0)
    bar_colors = ['#2471a3', '#1a9e73', '#7b5ea7']
    bars = ax_bar.barh([2, 1, 0], total_per_level, color=bar_colors, edgecolor='white', linewidth=1)
    ax_bar.set_yticks([0, 1, 2])
    ax_bar.set_yticklabels(['L2', 'L1', 'L0'], fontsize=9)
    ax_bar.set_xlabel('Total codes across\nall groups', fontsize=8)
    ax_bar.set_title('Total\nper level', fontsize=9, fontweight='bold')
    for bar, val in zip(bars, total_per_level[::-1]):
        ax_bar.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    str(val), va='center', fontsize=9, fontweight='bold')
    ax_bar.grid(alpha=GRID_ALPHA, axis='x')
    ax_bar.set_xlim(0, total_per_level.max() * 1.25)

    # Caption
    total_quantizers = n_groups * n_levels
    total_codes      = vocab.sum()
    fig.text(0.5, -0.01,
             f'{total_quantizers} quantizers  ·  {total_codes:,} total codebook entries  '
             f'·  EMA decay=0.99  ·  commitment_cost=0.25',
             ha='center', fontsize=8, color='#555', style='italic')

    plt.tight_layout()
    out = out_dir / '05_quantization_hierarchy.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


# ── Fig 6 — Training Curves ───────────────────────────────────────────────────

def fig6_training_curves(ckpt, out_dir):
    """Training and validation loss curves with individual component subplots."""
    hist       = ckpt['metadata']['training_history']
    train_ms   = hist['train_metrics']   # list of dicts per epoch
    val_ms     = hist['val_metrics']     # list of dicts per val step
    best_epoch = ckpt['epoch']
    best_val   = ckpt['val_loss']

    n_train = len(train_ms)
    n_val   = len(val_ms)

    # Components present in both train and val (skip complex nested keys)
    components = ['loss', 'reconstruction', 'vq', 'orthogonality', 'informativeness', 'roundtrip']
    comp_labels = {
        'loss':            'Total loss',
        'reconstruction':  'Reconstruction',
        'vq':              'VQ commitment',
        'orthogonality':   'Orthogonality',
        'informativeness': 'Informativeness',
        'roundtrip':       'Round-trip',
    }
    comp_colors = ['#2471a3', '#27ae60', '#e67e22', '#8e44ad', '#c0392b', '#17a589']

    train_epochs = np.arange(1, n_train + 1)
    val_freq     = max(1, n_train // n_val) if n_val > 0 else 1
    val_epochs   = np.arange(val_freq, n_train + 1, val_freq)[:n_val]

    n_comp = len(components)
    ncols  = 3
    nrows  = (n_comp + ncols - 1) // ncols  # 2 rows of 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    fig.suptitle(f'Fig 6 — Training Dynamics  (CNO v1 VQTokenizer, {n_train} epochs)',
                 fontsize=12, fontweight='bold', y=1.01)

    for idx, (comp, color) in enumerate(zip(components, comp_colors)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        t_vals = [m.get(comp, np.nan) for m in train_ms]
        v_vals = [m.get(comp, np.nan) for m in val_ms]

        ax.plot(train_epochs, t_vals, color=color, linewidth=1.6,
                label='Train', alpha=0.9)
        if n_val > 0 and len(val_epochs) == n_val:
            ax.plot(val_epochs, v_vals, color=color, linewidth=1.6,
                    linestyle='--', label='Val', alpha=0.9, marker='o', markersize=3)

        ax.set_yscale('log')
        ax.set_title(comp_labels[comp], fontsize=9, fontweight='bold', color=color)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.grid(alpha=GRID_ALPHA, which='both')
        ax.legend(fontsize=7.5, loc='upper right')

        # Mark best epoch
        ax.axvline(best_epoch, color='#aaa', linestyle=':', linewidth=1.2, alpha=0.8)

        # Annotation on total-loss subplot only
        if comp == 'loss':
            info_text = (f'Best ep: {best_epoch}\n'
                         f'Val: {best_val:.5f}')
            ax.text(0.97, 0.97, info_text, transform=ax.transAxes,
                    fontsize=7.5, va='top', ha='right', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7',
                              edgecolor='#d4ac0d', alpha=0.9))

    # Hide any unused subplots
    for idx in range(n_comp, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    plt.tight_layout()
    out = out_dir / '06_training_curves.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


# ── Fig 7 — Architecture Diagram ──────────────────────────────────────────────

def fig7_architecture_diagram(ckpt, out_dir):
    """End-to-end pipeline schematic drawn with matplotlib patches."""
    fm          = ckpt['feature_metadata']['families']['temporal']
    gi          = ckpt['group_indices']
    enc_cfg     = ckpt['config']['encoder']
    hier_cfg    = ckpt['config']['hierarchy']
    n_raw       = fm['original_feature_count']      # 345
    n_clean     = fm['cleaned_feature_count']        # 257
    n_groups    = len([k for k in gi if k.startswith('temporal_group_')])
    n_levels    = hier_cfg['num_levels']             # 3
    emb_dim     = enc_cfg['embedding_dim']           # 64
    level_dims  = enc_cfg['temporal']['level_dims']  # [6,12,20,26]
    ds_factors  = enc_cfg['temporal']['downsample_factors']

    # Infer initial and theta enc dims from checkpoint config
    initial_manual_dim = enc_cfg['initial']['manual_dim']       # 42
    initial_cnn_dim    = enc_cfg['initial']['cnn_embedding_dim'] # 384
    theta_out_dim      = enc_cfg['theta']['output_dim']          # 32
    theta_param_dim    = enc_cfg['theta']['param_dim']           # 14

    fig = plt.figure(figsize=(22, 8.0))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 22)
    ax.set_ylim(2.8, 8.8)   # more compact vertical space
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Column x-centres — increased spacing throughout for clarity
    col_xs = [1.4, 3.6, 5.8, 8.5, 11.5, 15.5, 19.8]
    col_titles = ['Inputs', 'Feature\nextraction', 'Cleaning', 'Grouping',
                  'Per-group\nencoders', 'VQ hierarchy', 'Outputs']
    col_bg_color = '#f8f9fa'

    # Column widths — increased for better breathing room
    col_widths = [2.2, 2.2, 2.2, 2.4, 3.0, 3.6, 2.4]

    # Column separator backgrounds — fully containing labels with proper padding
    for i, (cx, cw) in enumerate(zip(col_xs, col_widths)):
        bg = FancyBboxPatch((cx - cw / 2, 2.95), cw, 5.55,
                             boxstyle='round,pad=0.06',
                             facecolor=col_bg_color, edgecolor='#aaa',
                             linewidth=1.4, zorder=0)
        ax.add_patch(bg)
        ax.text(cx, 8.35, col_titles[i], ha='center', va='center', fontsize=11,
                fontweight='bold', color='#444', zorder=5)

    def box(cx, cy, w, h, label, sublabel='', fc='#d6eaf8', ec=C_TEMPORAL, fs=11):
        b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                            boxstyle='round,pad=0.10',
                            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
        ax.add_patch(b)
        ax.text(cx, cy + (0.10 if sublabel else 0), label,
                ha='center', va='center', fontsize=fs, fontweight='bold', zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.24, sublabel, ha='center', va='center',
                    fontsize=9.5, color='#333', zorder=4)

    def arrow(x1, y1, x2, y2, dim_label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2.0,
                                   connectionstyle='arc3,rad=0', shrinkA=0, shrinkB=0), zorder=6)
        if dim_label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.18, dim_label, fontsize=9.5, color='#333',
                    style='italic', va='bottom', ha='center', fontweight='semibold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.10', facecolor='white',
                              edgecolor='#d0d0d0', linewidth=0.6, alpha=0.96))

    # Row y-positions: compact but clear spacing
    Y_TEMPORAL = 6.8
    Y_INITIAL  = 4.8
    Y_THETA    = 3.5

    # ── Col 1: Inputs ──
    box(col_xs[0], Y_TEMPORAL, 1.7, 0.75, 'CNO Fields', f'[T, 3, 64, 64]',
        fc='#d6eaf8', ec=C_TEMPORAL)
    box(col_xs[0], Y_INITIAL, 1.7, 0.75, 'Initial Condition', f'[3, 64, 64]',
        fc='#d5f5e3', ec=C_INITIAL)
    box(col_xs[0], Y_THETA, 1.7, 0.75, f'θ ∈ ℝ{theta_param_dim}', f'params ({theta_param_dim}D)',
        fc='#e8daef', ec=C_THETA)

    # ── Col 2: Feature extraction ──
    box(col_xs[1], Y_TEMPORAL, 1.7, 0.75, 'Temporal Feature\nExtraction', f'→ [T, {n_raw}D]',
        fc='#d6eaf8', ec=C_TEMPORAL)
    box(col_xs[1], Y_INITIAL, 1.7, 0.75, 'HybridInitial\nEncoder',
        f'manual {initial_manual_dim}D+CNN {initial_cnn_dim}D',
        fc='#d5f5e3', ec=C_INITIAL)
    box(col_xs[1], Y_THETA, 1.7, 0.75, 'ThetaMLP',
        f'→ {theta_out_dim}D',
        fc='#e8daef', ec=C_THETA)

    # Arrows Col1 → Col2 (box half-width = 1.7/2 = 0.85)
    arrow(col_xs[0] + 0.85, Y_TEMPORAL, col_xs[1] - 0.85, Y_TEMPORAL)
    arrow(col_xs[0] + 0.85, Y_INITIAL,  col_xs[1] - 0.85, Y_INITIAL)
    arrow(col_xs[0] + 0.85, Y_THETA,    col_xs[1] - 0.85, Y_THETA)

    # ── Col 3: Cleaning ──
    box(col_xs[2], Y_TEMPORAL, 1.5, 0.75,
        'Variance filter\n+ Dedup',
        f'{n_raw}→{n_clean}D',
        fc='#fef9e7', ec='#d4ac0d')
    # Initial and theta pass-through text floats above the long arrow line
    for y_pt in [Y_INITIAL, Y_THETA]:
        ax.text(col_xs[2], y_pt + 0.32, '(pass-through)', ha='center', va='center',
                fontsize=9.5, color='#666', style='italic', zorder=7,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='#ddd', linewidth=0.5, alpha=0.92))

    arrow(col_xs[1] + 0.85, Y_TEMPORAL, col_xs[2] - 0.75, Y_TEMPORAL, f'{n_raw}D')
    # Long pass-through arrows: start right edge of Cleaning col, end left edge of Encoder col
    # (they span Cleaning→Grouping→Encoder as single arrows through the pass-through text)
    arrow(col_xs[1] + 0.85, Y_INITIAL, col_xs[2] + 0.75, Y_INITIAL)
    arrow(col_xs[1] + 0.85, Y_THETA,   col_xs[2] + 0.75, Y_THETA)

    # ── Col 4: Grouping ──
    box(col_xs[3], Y_TEMPORAL, 1.4, 0.75, 'PCA dominant-PC\ngrouping',
        f'{n_groups} slices',
        fc='#fef9e7', ec='#d4ac0d')

    arrow(col_xs[2] + 0.75, Y_TEMPORAL, col_xs[3] - 0.7, Y_TEMPORAL, f'{n_clean}D')
    # Initial & theta pass-through spans Cleaning→Encoder (over Grouping col)
    arrow(col_xs[2] + 0.75, Y_INITIAL, col_xs[4] - 0.85, Y_INITIAL)
    arrow(col_xs[2] + 0.75, Y_THETA,   col_xs[4] - 0.85, Y_THETA)

    # ── Col 5: Encoders ──
    # 30× temporal pyramid encoders with internal level breakdown
    pyr_cx   = col_xs[4]
    pyr_w    = 2.6   # outer box width
    pyr_bot  = 5.50   # adjusted for proper clearance from column label
    pyr_top  = 8.10   # lowered to avoid overlap with column header
    pyr_h    = pyr_top - pyr_bot

    pyramid_box = FancyBboxPatch((pyr_cx - pyr_w / 2, pyr_bot), pyr_w, pyr_h,
                                  boxstyle='round,pad=0.10',
                                  facecolor='#eaf4fb', edgecolor=C_TEMPORAL, linewidth=2.0, zorder=3)
    ax.add_patch(pyramid_box)

    # Header text
    ax.text(pyr_cx, 7.95, f'{n_groups}×  PyramidEncoder', ha='center', va='center',
            fontsize=11.5, fontweight='bold', color=C_TEMPORAL, zorder=4)
    ax.text(pyr_cx, 7.73, f'[B, T, Gₖ]  →  {emb_dim}D', ha='center', va='center',
            fontsize=9.5, color='#444', zorder=4)

    # Per-level sub-boxes — adjusted to fit within new pyramid bounds
    level_fill   = ['#aed6f1', '#a9dfbf', '#d7bde2', '#fadbd8']
    level_border = ['#2471a3', '#1a9e73', '#7b5ea7', '#c0392b']
    ds_labels    = [f'×{ds}' for ds in ds_factors]
    sub_ys    = [7.35, 6.92, 6.49, 6.06]   # adjusted for new pyramid height
    sub_h     = 0.40   # slightly smaller boxes
    sub_w     = pyr_w - 0.32

    for l, (sy, ld, lf, lb, ds_lbl) in enumerate(
            zip(sub_ys, level_dims, level_fill, level_border, ds_labels)):
        sub = FancyBboxPatch((pyr_cx - sub_w / 2, sy - sub_h / 2), sub_w, sub_h,
                              boxstyle='round,pad=0.02',
                              facecolor=lf, edgecolor=lb, linewidth=1.4, zorder=4)
        ax.add_patch(sub)
        ax.text(pyr_cx - sub_w / 2 + 0.24, sy, f'L{l}',
                ha='center', va='center', fontsize=10.5, fontweight='bold', color=lb, zorder=5)
        ax.text(pyr_cx, sy, f'T/{ds_factors[l]}  →  {ld}D',
                ha='center', va='center', fontsize=10, color='#222', fontweight='semibold', zorder=5)
        ax.text(pyr_cx + sub_w / 2 - 0.14, sy, ds_lbl,
                ha='right', va='center', fontsize=9.5, color='#555', style='italic', zorder=5)

    # Concat bar — positioned with clear separation from L3
    concat_y = 5.70   # adjusted for new pyramid height
    concat_h = 0.32
    concat_box = FancyBboxPatch((pyr_cx - sub_w / 2, concat_y - concat_h / 2), sub_w, concat_h,
                                 boxstyle='round,pad=0.03',
                                 facecolor='#d6eaf8', edgecolor=C_TEMPORAL, linewidth=1.2, zorder=4)
    ax.add_patch(concat_box)
    ax.text(pyr_cx, concat_y, f'concat  →  {emb_dim}D',
            ha='center', va='center', fontsize=10, fontweight='bold', color=C_TEMPORAL, zorder=5)

    # Col 5: Initial/Theta encoders — matching height with VQ and output boxes
    box(col_xs[4], Y_INITIAL, 1.8, 0.70, 'InitialHybrid\nEncoder',
        f'→ {initial_manual_dim + initial_cnn_dim}D',
        fc='#d5f5e3', ec=C_INITIAL, fs=9.5)
    box(col_xs[4], Y_THETA, 1.8, 0.70, 'ThetaMLP',
        f'→ {theta_out_dim}D',
        fc='#e8daef', ec=C_THETA, fs=9.5)

    # Fan-out arrows from grouping into pyramid encoders (placed after pyr_w is defined)
    for g_y in [7.5, 7.0, 6.5]:
        ax.annotate('', xy=(col_xs[4] - pyr_w/2 - 0.05, g_y), xytext=(col_xs[3] + 0.7, Y_TEMPORAL),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.2, alpha=0.85), zorder=2)

    # ── Col 6: VQ hierarchy ──
    _all_vq_colors = ['#aed6f1', '#a9dfbf', '#d7bde2', '#fadbd8']
    _all_vq_ec     = [C_TEMPORAL, C_INITIAL, C_THETA, '#7f8c8d']
    vq_levels    = [f'L{i}' for i in range(n_levels)]
    vq_colors    = _all_vq_colors[:n_levels]
    vq_ec        = _all_vq_ec[:n_levels]
    vq_y_centers = np.linspace(7.5, 6.5, n_levels).tolist()   # compact positioning

    # 30×3 VQ grid — increased spacing between boxes
    n_show  = 6
    w_vq    = 0.42   # wider boxes
    gap_vq  = 0.10   # increased gap
    x_start = col_xs[5] - (n_show * (w_vq + gap_vq) - gap_vq) / 2
    for li, (lvl, vc, vce, vy) in enumerate(zip(vq_levels, vq_colors, vq_ec, vq_y_centers)):
        for g_show in range(n_show):
            x_vq = x_start + g_show * (w_vq + gap_vq)
            vq_box = FancyBboxPatch((x_vq, vy - 0.24), w_vq, 0.48,
                                     boxstyle='round,pad=0.05',
                                     facecolor=vc, edgecolor=vce, linewidth=1.2, zorder=3)
            ax.add_patch(vq_box)
            if g_show == 3:
                ax.text(x_vq + w_vq / 2, vy, '···', ha='center', va='center',
                        fontsize=12, color='#555', fontweight='bold', zorder=4)
            else:
                ax.text(x_vq + w_vq / 2, vy, f'VQ\n{lvl}', ha='center', va='center',
                        fontsize=8.5, fontweight='bold', zorder=4)
        ax.text(x_start - 0.20, vy, lvl, ha='right', va='center',
                fontsize=12, fontweight='bold', color=vce, zorder=5)

    # Initial + theta VQ — matching height with other boxes in rows
    box(col_xs[5], Y_INITIAL, 1.8, 0.70, 'Initial VQ\n(groups×3)',
        fc='#d5f5e3', ec=C_INITIAL, fs=9.5)
    box(col_xs[5], Y_THETA, 1.8, 0.70, 'Theta VQ\n(groups×3)',
        fc='#e8daef', ec=C_THETA, fs=9.5)

    # VQ grid right edge
    vq_row_half_w = (n_show * (w_vq + gap_vq) - gap_vq) / 2
    vq_right_x    = col_xs[5] + vq_row_half_w + 0.08

    # Arrows Col5 → Col6
    # Temporal: pyramid right edge → VQ grid left edge (above L1 to avoid overlap)
    temporal_arrow_y = 7.2   # positioned between L0 and L1 to avoid label overlap
    arrow(pyr_cx + pyr_w / 2 + 0.05, temporal_arrow_y, x_start - 0.25, temporal_arrow_y, f'{emb_dim}D')
    # Initial/Theta: use box half-width (0.85)
    arrow(col_xs[4] + 0.85, Y_INITIAL, col_xs[5] - vq_row_half_w - 0.15, Y_INITIAL)
    arrow(col_xs[4] + 0.85, Y_THETA,   col_xs[5] - vq_row_half_w - 0.15, Y_THETA)

    # ── Col 7: Outputs ──
    box(col_xs[6], temporal_arrow_y, 1.8, 1.25, 'Token dict', '{cat_level: int}',
        fc='#fdebd0', ec='#e67e22', fs=10.5)
    box(col_xs[6], Y_INITIAL, 1.8, 0.70, 'IĈ', 'inverse CNN',
        fc='#d5f5e3', ec=C_INITIAL, fs=9.5)
    box(col_xs[6], Y_THETA, 1.8, 0.70, 'θ̂', 'inverse MLP',
        fc='#e8daef', ec=C_THETA, fs=9.5)

    # Arrows Col6 → Col7
    arrow(vq_right_x + 0.05, temporal_arrow_y, col_xs[6] - 0.85, temporal_arrow_y)
    arrow(col_xs[5] + vq_row_half_w + 0.12, Y_INITIAL, col_xs[6] - 0.85, Y_INITIAL)
    arrow(col_xs[5] + vq_row_half_w + 0.12, Y_THETA,   col_xs[6] - 0.85, Y_THETA)

    # Title / caption
    ax.text(11.0, 2.80,
            f'VQTokenizer v5 (CNO, 3ch)  ·  {n_groups} groups × {n_levels} levels = '
            f'{n_groups * n_levels} temporal quantizers  ·  emb_dim={emb_dim}D',
            ha='center', va='top', fontsize=10, color='#444', style='italic')

    out = out_dir / '07_architecture_diagram.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='VQTokenizer pipeline visualizations')
    ap.add_argument('--checkpoint', default='checkpoints/cno/v1/vq_tokenizer_best.pt')
    ap.add_argument('--dataset',    default='datasets/50k_baseline.h5')
    ap.add_argument('--output',     default='visualizations/vqtokenizer_pipeline')
    ap.add_argument('--n-samples',  type=int, default=5000,
                    help='Dataset samples to load for Figs 2-3 (default 5000)')
    ap.add_argument('--figs',       nargs='+', type=int, default=list(range(1, 8)),
                    metavar='N', help='Which figures to generate (default: 1 2 3 4 5 6 7)')
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = load_checkpoint(args.checkpoint)
    print(f'  ✓ epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.6f}')

    # Only load dataset if needed for Figs 2 or 3
    temporal_raw = initial_raw = None
    if 2 in args.figs or 3 in args.figs:
        print(f'Loading {args.n_samples:,} samples from {args.dataset}')
        temporal_raw, initial_raw = load_dataset_subset(args.dataset, args.n_samples)
        print(f'  ✓ temporal {temporal_raw.shape}, initial {initial_raw.shape}')

    print(f'\nGenerating figures → {out_dir}/')
    if 1 in args.figs:
        fig1_feature_extraction(ckpt, args.dataset, out_dir)
    if 2 in args.figs:
        fig2_feature_cleaning(ckpt, temporal_raw, out_dir)
    if 3 in args.figs:
        fig3_pca_grouping(ckpt, temporal_raw, out_dir)
    if 4 in args.figs:
        fig4_pyramid_encoder(ckpt, out_dir)
    if 5 in args.figs:
        fig5_quantization_hierarchy(ckpt, out_dir)
    if 6 in args.figs:
        fig6_training_curves(ckpt, out_dir)
    if 7 in args.figs:
        fig7_architecture_diagram(ckpt, out_dir)

    print(f'\nAll done. Open {out_dir}/ to inspect.')


if __name__ == '__main__':
    main()
