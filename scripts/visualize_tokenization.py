#!/usr/bin/env python3
"""
Visualize VQTokenizer variable-length tokenization for a single sample.

TOP:    Wavefunction |ψ|² frames (every N steps)
BOTTOM: Full token index matrix — one row per truncation T, one column per
        codebook. Uses ellipsis (…) when all 183 columns won't fit readably.

Usage:
    poetry run python scripts/visualize_tokenization.py
    poetry run python scripts/visualize_tokenization.py --sample-index 7
    poetry run python scripts/visualize_tokenization.py --no-replay
"""

import sys
sys.path.insert(0, 'src')

import argparse
import re
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.qbm.replayer import QBMReplayer


# ── Data ──────────────────────────────────────────────────────────────────────

def load_sample(path, idx):
    with h5py.File(path, 'r') as f:
        return (
            f['parameters/params'][idx],
            f['inputs/fields'][idx],
            f['features/temporal/features'][idx],
            f['features/initial/aggregated/features'][idx],
        )


def replay_rollout(params, config, T=256, seed=0):
    replayer = QBMReplayer.from_config(config, device='cuda')
    traj = replayer.rollout(params_vector=params, num_realizations=1,
                            num_timesteps=T, seed=seed)
    r, i = traj[0, :, 0].cpu().numpy(), traj[0, :, 1].cpu().numpy()
    return r**2 + i**2


def ic_only(ic, T=256):
    prob = (ic[:, 0]**2 + ic[:, 1]**2).mean(axis=0)
    return np.broadcast_to(prob[None], (T, *prob.shape)).copy()


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_bins(tokenizer, temporal, initial, ic, params, bins, device):
    dev   = torch.device(device)
    init  = torch.tensor(initial,         dtype=torch.float32).unsqueeze(0).to(dev)
    raw   = torch.tensor(ic.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(dev)
    theta = torch.tensor(params,          dtype=torch.float32).unsqueeze(0).to(dev)
    out   = {}
    for T in bins:
        feat = torch.tensor(temporal[:T], dtype=torch.float32).unsqueeze(0).to(dev)
        mask = torch.ones(1, T, dtype=torch.bool, device=dev)
        lens = torch.tensor([T], dtype=torch.long, device=dev)
        with torch.no_grad():
            toks = tokenizer.tokenize(temporal_features=feat, initial_manual=init,
                                      initial_raw=raw, theta_features=theta,
                                      temporal_mask=mask, temporal_lengths=lens)
        out[T] = {k: int(v.item()) for k, v in toks.items()}
    return out


def sort_all_keys(keys):
    def order(k):
        m = re.match(r'(temporal|initial|theta)_group_(\d+)_L(\d+)', k)
        if not m: return (9, 0, 0)
        fam = {'temporal': 0, 'initial': 1, 'theta': 2}[m.group(1)]
        return (fam, int(m.group(3)), int(m.group(2)))
    return sorted(keys, key=order)


def family_spans(sorted_keys):
    spans, cur, start = [], None, 0
    for i, k in enumerate(sorted_keys):
        m = re.match(r'(temporal|initial|theta)_group_\d+_(L\d+)', k)
        label = f"{m.group(1)} {m.group(2)}" if m else 'other'
        if label != cur:
            if cur: spans.append((cur, start, i - 1))
            cur, start = label, i
    if cur: spans.append((cur, start, len(sorted_keys) - 1))
    return spans


# ── Ellipsis truncation ───────────────────────────────────────────────────────

def apply_ellipsis(keys, matrix, max_display=80):
    """If n_cols > max_display, keep first half and last half with a gap column."""
    n = len(keys)
    if n <= max_display:
        return keys, matrix, None  # no ellipsis

    half  = max_display // 2
    left  = half
    right = max_display - left

    kept_keys   = list(keys[:left]) + ['…'] + list(keys[n - right:])
    # build display matrix: left cols | NaN col | right cols
    nan_col     = np.full((matrix.shape[0], 1), np.nan)
    disp_matrix = np.concatenate([matrix[:, :left], nan_col, matrix[:, n - right:]], axis=1)
    ellipsis_col = left   # column index of the '…' column

    return kept_keys, disp_matrix, ellipsis_col


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(prob_frames, tokens_by_bin, bins, sample_idx, params, max_cols=90):
    import warnings

    all_keys = sort_all_keys(list(next(iter(tokens_by_bin.values())).keys()))
    all_keys = [k for k in all_keys if k.startswith('temporal_')]
    n_cols   = len(all_keys)
    n_rows   = len(bins)

    matrix = np.array([[tokens_by_bin[T][k] for k in all_keys] for T in bins],
                      dtype=float)

    disp_keys, disp_mat, ell_col = apply_ellipsis(all_keys, matrix, max_cols)
    n_disp = len(disp_keys)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_max = np.nanmax(disp_mat, axis=0, keepdims=True)
    col_max  = np.where(np.isnan(col_max) | (col_max == 0), 1.0, col_max)
    mat_norm = disp_mat / col_max

    # ── Absolute layout (all sizes in inches) ─────────────────────────────────
    row_h    = 0.62          # height of each matrix row = height of each frame thumb
    cell_w   = 0.115         # width per token column
    frame_w  = row_h         # square thumbnail
    gap      = 0.07          # gap between thumbnail and matrix
    lmargin  = 0.08          # left of figure to left of thumbnails
    rmargin  = 0.05
    title_h  = 0.44
    note_h   = 0.18          # bottom note line
    bmargin  = 0.10

    mat_w  = n_disp * cell_w
    header_h = 0.16          # family label header strip above matrix
    fig_w  = lmargin + frame_w + gap + mat_w + rmargin
    fig_h  = bmargin + n_rows * row_h + header_h + title_h

    # helper: inches → figure fraction
    def fx(x): return x / fig_w
    def fy(y): return y / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.rcParams.update({'font.family': 'sans-serif'})

    # Title
    theta_str = ', '.join(f'{p:.3f}' for p in params)
    n_cb_note = f'{n_cols} codebooks'
    if ell_col is not None:
        n_cb_note = f'{n_cols} codebooks (showing {n_disp - 1})'
    fig.suptitle(
        f'Variable-Length VQ Tokenization  —  Sample #{sample_idx}  ({n_cb_note})\n'
        f'θ = [{theta_str}]',
        fontsize=9, fontweight='bold', y=0.998, va='top',
    )

    # ── Per-row frame thumbnails + matrix ─────────────────────────────────────
    p_lo = np.percentile(prob_frames, 1)
    p_hi = np.percentile(prob_frames, 99)

    mat_bottom = bmargin   # bottom of matrix block in inches

    for i, T in enumerate(bins):
        # y position: row 0 (T=bins[0]) is at the top of the matrix block
        row_bot = mat_bottom + (n_rows - 1 - i) * row_h

        # Frame thumbnail
        ax_f = fig.add_axes([fx(lmargin), fy(row_bot), fx(frame_w), fy(row_h)])
        t_idx = min(T - 1, len(prob_frames) - 1)
        fr = np.clip((prob_frames[t_idx] - p_lo) / max(p_hi - p_lo, 1e-9), 0, 1)
        ax_f.imshow(fr, cmap='inferno', vmin=0, vmax=1,
                    interpolation='bilinear', aspect='equal')
        ax_f.set_xticks([]); ax_f.set_yticks([])
        for sp in ax_f.spines.values(): sp.set_linewidth(0.4)
        # T label inside top-left of frame
        ax_f.text(0.05, 0.95, f't={T}', transform=ax_f.transAxes,
                  fontsize=5.5, color='white', va='top', ha='left',
                  fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.35, lw=0))

    # ── Family color metadata (used for header, matrix dividers, footer) ────────
    mat_x = lmargin + frame_w + gap

    fam_colors = {'temporal L0': '#2471a3', 'temporal L1': '#1a9e73',
                  'temporal L2': '#7b5ea7',
                  'initial L0':  '#27ae60', 'initial L1':  '#1abc9c',
                  'initial L2':  '#2980b9', 'theta L0':    '#8e44ad',
                  'theta L1':    '#a93226', 'theta L2':    '#6c3483'}

    orig_spans = family_spans(all_keys)
    if ell_col is not None:
        left_n  = ell_col
        right_n = n_disp - ell_col - 1
        orig_n  = n_cols
        shift   = ell_col + 1 - (orig_n - right_n)
        disp_spans = []
        for fam, s, e in orig_spans:
            ls, le = min(s, left_n - 1), min(e, left_n - 1)
            if ls <= le:
                disp_spans.append((fam, ls, le))
            rs  = max(s + shift, ell_col + 1)
            re_ = min(e + shift, n_disp - 1)
            if rs <= re_:
                disp_spans.append((fam, rs, re_))
    else:
        disp_spans = list(orig_spans)

    # Family boundary x-positions in display coords (between consecutive families)
    fam_div_x = []
    for j in range(len(disp_spans) - 1):
        _, _, e1 = disp_spans[j]
        _, s2, _ = disp_spans[j + 1]
        if e1 + 1 == s2:           # adjacent — boundary between them
            fam_div_x.append(e1 + 0.5)

    # ── Family header strip (above matrix) — neutral dark, clearly a label ──────
    header_bot = mat_bottom + n_rows * row_h
    hdr = fig.add_axes([fx(mat_x), fy(header_bot), fx(mat_w), fy(header_h)])
    hdr.set_xlim(-0.5, n_disp - 0.5)
    hdr.set_ylim(0, 1)
    hdr.axis('off')
    hdr.set_facecolor('#2d2d2d')
    hdr.patch.set_visible(True)
    # Fill entire background dark grey
    hdr.barh(0.5, n_disp, left=-0.5, height=1.0, color='#2d2d2d', linewidth=0)
    # White dividers at family boundaries
    for xb in fam_div_x:
        hdr.axvline(xb, color='white', linewidth=1.5)
    # Family name labels in white — centred with vertical breathing room
    for fam, s, e in disp_spans:
        if e - s + 1 >= 4:
            hdr.text((s + e) / 2, 0.5, fam,
                     ha='center', va='center', fontsize=6.5,
                     color='white', fontweight='bold', clip_on=True)

    # ── Token matrix (single imshow for all rows) ──────────────────────────────
    ax = fig.add_axes([fx(mat_x), fy(mat_bottom), fx(mat_w), fy(n_rows * row_h)])

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#d8d8d8')

    ax.imshow(mat_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1,
              interpolation='nearest')

    # 1px cell separators (0.5pt ≈ 1px at 200 dpi)
    for x in np.arange(-0.5, n_disp, 1):
        ax.axvline(x, color='white', linewidth=0.5)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color='white', linewidth=0.5)

    # Thick family boundary dividers
    for xb in fam_div_x:
        ax.axvline(xb, color='white', linewidth=2.0, zorder=5)

    # Integer labels
    fs = max(4.5, min(7.5, 95 / n_disp))
    for r in range(n_rows):
        for c in range(n_disp):
            v = disp_mat[r, c]
            if np.isnan(v):
                ax.text(c, r, '…', ha='center', va='center',
                        fontsize=8, color='#aaa')
                continue
            bg  = mat_norm[r, c]
            col = '#111' if bg > 0.65 else 'white'
            ax.text(c, r, str(int(v)), ha='center', va='center',
                    fontsize=fs, color=col, fontweight='bold')

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-0.5, n_disp - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for sp in ax.spines.values(): sp.set_linewidth(0.0)

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',        default='datasets/qbm_50k.h5')
    ap.add_argument('--checkpoint',     default='checkpoints/v5/pyramid/vq_tokenizer_best.pt')
    ap.add_argument('--config',         default='configs/qbm/substrate.yaml')
    ap.add_argument('--sample-index',   type=int, default=0)
    ap.add_argument('--max-cols',       type=int, default=90,
                    help='Max token columns before ellipsis (default 90)')
    ap.add_argument('--output',         default=None)
    ap.add_argument('--device',         default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--no-replay',      action='store_true')
    ap.add_argument('--seed',           type=int, default=0)
    args = ap.parse_args()

    out = args.output or f'visualizations/tokenization/sample{args.sample_index}.png'
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading sample {args.sample_index}...')
    params, ic, temporal, initial = load_sample(args.dataset, args.sample_index)

    if args.no_replay:
        prob_frames = ic_only(ic)
    else:
        print('Replaying QBM trajectory...')
        try:
            prob_frames = replay_rollout(params, args.config, T=256, seed=args.seed)
            print(f'  ✓ {prob_frames.shape}')
        except Exception as e:
            print(f'  ⚠ Replay failed ({e}), using IC only')
            prob_frames = ic_only(ic)

    print('Loading tokenizer...')
    tokenizer = VQTokenizer.from_checkpoint(args.checkpoint)
    tokenizer.model.to(args.device).eval()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    bins = ckpt.get('variable_length_metadata', {}).get('bins', [32, 64, 128, 256])
    print(f'  Bins: {bins}')

    print('Tokenizing...')
    tokens_by_bin = tokenize_bins(
        tokenizer, temporal, initial, ic, params, bins, args.device)
    print(f'  ✓ {len(next(iter(tokens_by_bin.values())))} tokens per bin')

    print('Rendering...')
    fig = make_figure(
        prob_frames=prob_frames,
        tokens_by_bin=tokens_by_bin,
        bins=bins,
        sample_idx=args.sample_index,
        params=params,
        max_cols=args.max_cols,
    )
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Saved: {out}')


if __name__ == '__main__':
    main()
