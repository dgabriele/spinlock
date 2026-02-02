# Changelog

All notable changes to the Spinlock project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-02-02

### Added
- **torch.compile support for variable-length models** - Selective compilation architecture
  - Fixed-length: ~30-40% speedup (full model compilation)
  - Variable-length: ~15-25% speedup (encoder/decoder only, skip dynamic temporal)
  - Compilation disabled by default for variable-length (limited benefit)
  - See `docs/vqvae/torch-compile.md` for optimization guide
- **Learnable category assignment** - Gradient-based category optimization with Gumbel-Softmax
  - End-to-end optimization of feature-to-category mappings
  - Temperature annealing (1.0 → 0.1) for soft-to-hard assignments
  - Assignment losses (orthogonality, balance) for stable training
  - Supports both standard and hybrid initial encoders
  - See `docs/vqvae/learnable-assignments.md` for implementation details
- **Hybrid initial encoder integration** - End-to-end CNN training for initial conditions
  - Gradients flow from VQ-VAE losses back through CNN encoder
  - Compatible with both static and learnable assignments
  - 14D raw ICs → 128D CNN features during category discovery
  - See `docs/vqvae/architecture.md#hybrid-initial-path` for architecture
- **Comprehensive VQ-VAE documentation**
  - `docs/vqvae/README.md` - Complete guide with workflows and troubleshooting
  - `docs/vqvae/architecture.md` - Encoding paths, components, training process
  - `docs/vqvae/assignment-strategies.md` - Static vs learnable comparison
  - `docs/vqvae/learnable-assignments.md` - Integration architecture details
  - `docs/vqvae/learnable-mode-guide.md` - Complete usage guide
  - `docs/vqvae/variable-length-encoding.md` - Temporal pyramid integration
  - `docs/vqvae/torch-compile.md` - Performance optimization
  - `docs/decisions/2026-02-learnable-integration.md` - Implementation decision record

### Changed
- **Documentation reorganization** - Root directory cleanup
  - Moved implementation docs from root to `docs/vqvae/`
  - Updated README with VQ-VAE architecture overview
  - Added VQ-VAE section to documentation index
  - Root now contains only standard files (README.md, CHANGELOG.md)
- **VQ-VAE compilation defaults** - Variable-length models no longer compile by default
  - Limited speedup (~15-25%) not worth compilation overhead
  - Can still enable with `--compile` flag
  - Fixed-length models benefit significantly (~30-40% speedup)

### Fixed
- **Device mismatch in dead code reset** - CPU/CUDA tensor errors
  - Proper device handling for assignment matrix in reset logic
  - Fixes crash when using learnable assignments with dead code reset
- **Missing raw_ics support in compilation wrapper** - Variable-length compilation errors
  - Added `raw_ics` parameter to compiled forward signature
  - Enables hybrid encoder with compilation
- **Model unwrapping in metrics computation** - Compiled model compatibility
  - Extract underlying model from `OptimizedModule` wrapper
  - Fixes metric extraction from compiled models
- **Feature masking for learnable models in dead code reset**
  - Proper handling of soft assignments during reset
  - Prevents incorrect feature grouping in learnable mode

### Documentation
- Added `docs/vqvae/architecture.md` (~400 lines)
- Added `docs/vqvae/assignment-strategies.md` (~300 lines)
- Added `docs/vqvae/torch-compile.md` (moved from root)
- Added `docs/vqvae/learnable-assignments.md` (merged from 2 root files)
- Added `docs/vqvae/learnable-mode-guide.md` (moved from root, merged content)
- Added `docs/vqvae/variable-length-encoding.md` (moved from root)
- Added `docs/vqvae/README.md` (~50 lines)
- Added `docs/decisions/2026-02-learnable-integration.md` (moved from root)
- Updated `README.md` with VQ-VAE architecture section
- Updated `README.md` documentation index with VQ-VAE links

## [Unreleased] - 2026-01-30

### Added
- **Per-Channel Independent Initial Conditions (v3.2)** - Major enhancement for VQ-VAE diversity
  - Each of the 3 channels can now have different IC types, parameters, or characteristics
  - New `method: "per_channel"` with channel-specific configurations
  - Creates richer behavioral diversity for VQ-VAE category discovery
  - Enables cross-channel interaction pattern analysis
  - Supports NOA compositional reasoning experiments

  **Implementation:**
  - `ChannelICConfig` dataclass for per-channel configuration
  - `ICTypeSampler` for probabilistic IC type selection
  - `PerChannelICGenerator` with efficient batching (groups by IC type)
  - Pipeline integration in `DatasetGenerationPipeline`
  - New `ic_types` format: `"ch0:grf|ch1:struct|ch2:mgrf"`

  **Files:**
  - `src/spinlock/config/schema.py` - Config classes
  - `src/spinlock/dataset/generators.py` - Generator implementation
  - `src/spinlock/dataset/pipeline.py` - Pipeline integration
  - `tests/test_per_channel_ics.py` - 13 unit tests (all passing)
  - `configs/experiments/test_per_channel_100.yaml` - Test config
  - `configs/experiments/cno_50k_per_channel.yaml` - Production config
  - `docs/per_channel_ic_implementation.md` - Complete documentation

  **Backward Compatible:**
  - Existing configs and datasets work unchanged
  - VQ-VAE training requires no updates (only uses field tensors)
  - Single IC type format still supported

### Changed
- **HDF5 Layout Documentation** (`docs/features/hdf5-layout.md`)
  - Updated to v3.2 with per-channel IC format
  - Added IC Type Format section explaining both formats
  - Added migration notes for v3.2
  - Example configurations for per-channel ICs

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
