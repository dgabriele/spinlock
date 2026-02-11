# QBM Visualization Implementation

## Summary

Implemented a complete visualization system for Quantum Brownian Motion (QBM) datasets following the existing Spinlock visualization architecture. The implementation achieves **~80% code reuse** from existing infrastructure while adding **~1,000 LOC** of QBM-specific components.

## Components Implemented

### 1. Data Loader (`src/spinlock/visualization/quantum/data_loader.py`)
- **Purpose**: Load QBM datasets with complex wavefunction handling
- **Features**:
  - Context manager for safe HDF5 file access
  - Multiple output formats: probability density |ψ|², complex tensors, real/imag channels
  - Quantum feature extraction (purity, entropy, coherence)
  - Parameter loading (gamma, nbar, alpha_init)
- **LOC**: ~150

### 2. QBM Renderer (`src/spinlock/visualization/quantum/renderer.py`)
- **Purpose**: Render probability density |ψ|² from complex wavefunctions
- **Features**:
  - Supports pre-computed probability or on-the-fly conversion from real/imag
  - Multiple normalization modes: global, percentile, per-frame
  - Extends base `RenderStrategy` interface
  - GPU-accelerated colormap application
- **LOC**: ~200

### 3. Wigner Renderer (`src/spinlock/visualization/quantum/wigner_renderer.py`)
- **Purpose**: Compute and render Wigner phase-space distribution W(x,p)
- **Features**:
  - FFT-based Wigner function computation
  - Diverging colormap for quasi-probability (can be negative)
  - Symmetric normalization around zero
  - GPU-accelerated when device is CUDA
- **LOC**: ~200

### 4. Quantum Observable Overlay (`src/spinlock/visualization/quantum/aggregates.py`)
- **Purpose**: Overlay quantum observables as text on rendered frames
- **Features**:
  - Supports purity, entropy, coherence metrics
  - PIL-based text rendering with outline for visibility
  - Configurable position (top-left, top-right, bottom-left, bottom-right)
- **LOC**: ~200

### 5. CLI Command (`src/spinlock/cli/visualize_qbm.py`)
- **Purpose**: CLI integration matching existing `visualize-dataset` pattern
- **Features**:
  - Three rendering modes: probability, wigner, both (side-by-side)
  - Multiple sampling strategies: diverse, sobol, random
  - Quantum observable overlays
  - Frame-by-frame GPU memory management
  - Automatic GPU/CPU fallback for video encoding
- **LOC**: ~400

## Code Reuse

### Reused Infrastructure (~4,000 LOC)
- **GPUColormap**: Efficient color mapping with GPU-resident LUTs
- **VideoExporter**: ffmpeg integration with NVENC GPU acceleration support
- **Base RenderStrategy**: Normalization utilities and renderer interface
- **CLI Framework**: Argument parsing, base command pattern
- **Frame-by-frame Pattern**: Memory-efficient rendering pipeline

### New Code (~1,000 LOC)
- QBM-specific data handling: 150 LOC
- Probability density renderer: 200 LOC
- Wigner phase-space renderer: 200 LOC
- Quantum observable overlay: 200 LOC
- CLI integration: 400 LOC

## Usage Examples

### Basic Probability Density
```bash
poetry run spinlock visualize-qbm \
  --dataset datasets/qbm_50k.h5 \
  --output visualizations/qbm_probability.mp4 \
  --n-rollouts 4 \
  --renderer probability \
  --overlay-observable purity
```

### Wigner Phase-Space
```bash
poetry run spinlock visualize-qbm \
  --dataset datasets/qbm_50k.h5 \
  --output visualizations/qbm_wigner.mp4 \
  --n-rollouts 4 \
  --renderer wigner \
  --overlay-observable entropy
```

### Side-by-Side Comparison
```bash
poetry run spinlock visualize-qbm \
  --dataset datasets/qbm_50k.h5 \
  --output visualizations/qbm_both.mp4 \
  --n-rollouts 2 \
  --renderer both \
  --overlay-observable coherence_mean
```

## Testing

### Unit Tests
- **Renderer tests**: `tests/visualization/quantum/test_qbm_renderer.py` (8 tests, all passing)
- **Data loader tests**: `tests/visualization/quantum/test_qbm_data_loader.py` (11 tests, all passing)
- **Total coverage**: 19 unit tests covering core functionality

### Integration Test
- **Direct test**: `scripts/test_qbm_visualization_direct.py`
- **Validates**: Full rendering pipeline, video export, quantum overlays
- **Status**: All tests passing

## Architecture Decisions

### 1. Output Format: [B, H, W, 3] vs [B, 3, H, W]
- **Decision**: Renderers output [B, H, W, 3] format
- **Rationale**: Easier grid stacking (concatenate along spatial dimensions)
- **Conversion**: CLI transposes to [T, 3, H, W] for video export at the end

### 2. Normalization Strategy
- **Global mode**: Computes percentile-based bounds across full dataset for consistent coloring
- **Percentile mode**: Per-batch percentile clipping for robust visualization
- **Per-frame mode**: Independent normalization per timestep

### 3. Grid Layout
- **Probability/Wigner**: N rollouts × M realizations grid
- **Both mode**: N rollouts × (2×M) - side-by-side for each realization
- **Stacking**: Manual concatenation to avoid dependency on VisualizationGrid (which expects different format)

### 4. Memory Management
- **Frame-by-frame rendering**: Only one timestep on GPU at a time
- **Aggressive cleanup**: `torch.cuda.empty_cache()` after each frame
- **Batch export**: Collect all frames in CPU memory, then export once

## Design Improvements Made

### 1. Fixed Broken Import in `visualize_diffusion_inpainting.py`
- **Problem**: Hard import of `experiments.diffusion` module caused CLI to fail loading
- **Solution**: Wrapped experimental imports in try-except with `DIFFUSION_AVAILABLE` flag
- **Impact**: CLI now loads successfully even without experimental dependencies

### 2. Clean Functional Decomposition
- Each component has single responsibility
- Data loader: I/O and format conversion
- Renderers: Visualization strategy
- Overlays: Metadata display
- CLI: Orchestration and user interface

### 3. Consistent API Design
- All renderers implement `RenderStrategy` interface
- `supports_channels()` method for validation
- `render()` with optional `global_stats` parameter
- `compute_global_stats()` for dataset-wide normalization

## Performance Characteristics

### Memory Usage
- **Frame-by-frame**: ~0.6 MB/frame for 4 rollouts × 3 realizations × 64×64
- **Total video**: ~60 MB for 100 timesteps (before compression)
- **Output video**: ~5-10 MB for 100 timesteps (H.264 compressed)

### Speed Estimates
- **GPU (NVIDIA)**: ~30-100 frames/sec (Wigner is slower due to FFT)
- **CPU**: ~1-5 frames/sec
- **128 timesteps (4 rollouts)**: ~5-15 seconds on GPU, ~2-5 minutes on CPU

## Files Modified/Created

### New Files
- `src/spinlock/visualization/quantum/__init__.py`
- `src/spinlock/visualization/quantum/data_loader.py`
- `src/spinlock/visualization/quantum/renderer.py`
- `src/spinlock/visualization/quantum/wigner_renderer.py`
- `src/spinlock/visualization/quantum/aggregates.py`
- `src/spinlock/cli/visualize_qbm.py`
- `tests/visualization/quantum/__init__.py`
- `tests/visualization/quantum/test_qbm_renderer.py`
- `tests/visualization/quantum/test_qbm_data_loader.py`
- `scripts/test_qbm_visualization_direct.py`
- `docs/qbm-visualization-implementation.md`

### Modified Files
- `src/spinlock/cli/__init__.py` - Registered VisualizeQBMCommand
- `src/spinlock/cli/visualize_diffusion_inpainting.py` - Fixed import issues

## Next Steps

1. **Test with Real Dataset**: Run visualization on actual QBM dataset
2. **Optimization**: Profile Wigner computation for large datasets
3. **Documentation**: Add to README and getting-started guide
4. **Advanced Features** (optional):
   - Multi-observable overlay (show multiple metrics)
   - Interactive HTML export with hover tooltips
   - Animation effects (smooth transitions, zoom)
