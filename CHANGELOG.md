# Changelog

All notable changes to the Spinlock project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-01-08

### Removed
- **CUDA infrastructure** (92 MB) - Abandoned custom kernels (80× slower than PyTorch)
  - Pivoted to torch.compile() strategy (see `docs/decisions/PIVOT_TO_TORCH_COMPILE.md`)
  - Custom Conv2D, InstanceNorm, and Activation kernels were inefficient
  - Decision made December 29, 2025
- **Dataset backups** (12 GB) - Development snapshots now obsolete
  - `datasets/baseline_10k.h5.backup`
  - `datasets/100k_full_features.h5.backup`
- **Empty NOA directories** - training/ and evaluation/ subdirectories had no implementation
- **__pycache__ directories** (2.7 GB) - Auto-generated Python cache cleaned

### Archived
- **35 obsolete dev scripts** (~2.5K LOC) - One-off experiments and bug fixes
  - 17 scripts from Jan 1: test utilities, benchmarks, validation scripts
  - 12 nested test scripts from `scripts/dev/tests/`
  - 4 tier validation scripts from `scripts/dev/validation/`
  - 2 training scripts: `train_noa_state_supervised.py` and `train_noa_real_data.py`
  - Moved to `scripts/archived/` for historical reference
  - Superseded by `train_noa_unified.py` (supports both MSE-led and VQ-led training modes)
- **6 outdated experiment configs** - Small datasets (10k/50k) superseded by 100k
  - `test_realizations_50/`, `test_2k_phase1_phase2/`, `vqvae_baseline_10k_temporal/`
  - `baseline_10k/`, `50k_max_stratified/`, `vqvae_baseline_10k/`
  - Moved to `configs/archived/experiments/` for reproducibility

### Changed
- **.gitignore** - Added `*.o` and `*.a` patterns to prevent compiled artifact commits

### Total Impact
- **Disk space saved:** ~14.8 GB (31% reduction)
  - Dataset backups: 12.0 GB
  - `__pycache__`: 2.7 GB
  - CUDA directory: 92 MB
- **Code cleanup:** ~2.5K LOC archived
- **Repository structure:** Cleaner working tree for active development
- **Canonical training:** `train_noa_unified.py` is now the single training script

### Verification
- ✅ All critical imports verified working
- ✅ No broken dependencies
- ✅ Backup branch created: `backup-pre-cleanup-2026-01-08`
- ✅ 4 logical git commits with clear rationale
