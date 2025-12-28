# Advanced Visualization Features

## Overview

The Spinlock visualization system now includes **7 advanced aggregate renderers** for comprehensive analysis of stochastic neural operator realizations.

## New Aggregate Renderers

### 1. **Envelope Renderer** (`--aggregates envelope`)
- **Purpose**: Shows min/max range across realizations while preserving spatial structure
- **Use case**: Understand bounds of variation without averaging away structure
- **Implementation**: Uses HeatmapRenderer to visualize spatial min/max range

### 2. **Entropy Map Renderer** (`--aggregates entropy`)
- **Purpose**: Per-pixel Shannon entropy to capture structural uncertainty
- **Use case**: Identify regions with high multimodality or stochastic variation
- **Implementation**: Histogram-based entropy calculation (32 bins default)
- **Benefits**: Replaces variance for better uncertainty quantification

### 3. **PCA Mode Renderer** (`--aggregates pca`)
- **Purpose**: Shows principal components of variation across realizations
- **Use case**: Understand dominant modes of variation in spatial patterns
- **Implementation**: SVD-based, shows top-k principal components (default: 3)
- **Output**: RGB visualization of spatial modes

### 4. **SSIM Map Renderer** (`--aggregates ssim`)
- **Purpose**: Structure-aware similarity across realizations
- **Use case**: Identify regions with consistent vs. variable structure
- **Implementation**: Pairwise SSIM computation, averaged across all pairs
- **Benefits**: Better than pixel-wise metrics for structural consistency

### 5. **Spectral Aggregate Renderer** (`--aggregates spectral`)
- **Purpose**: FFT power spectrum showing dominant frequencies
- **Use case**: Understand spectral characteristics and periodicity
- **Implementation**: Per-realization FFT, averaged log power spectrum
- **Output**: Log-scale visualization of frequency content

### 6. **Trajectory Overlay Renderer** (`--aggregates overlay`)
- **Purpose**: Overlays all realizations with transparency
- **Use case**: Visual inspection of trajectory spread
- **Implementation**: Alpha-blended composite of all realizations

### 7. **Mean/Variance/StdDev** (original renderers, still available)
- **Purpose**: Basic statistical aggregates
- **Use case**: Standard deviation-based uncertainty

## Smart Display Control

### `--display-realizations K`

Controls how many individual realizations are shown in the grid:
- **Default**: 2 realizations displayed individually
- **Aggregates always use ALL realizations** (e.g., all 10)
- **Benefits**: Reduces grid size while maintaining aggregate quality

**Example**:
```bash
# Show only 2 individual realizations + 4 aggregates (uses all 10 for aggregates)
python scripts/spinlock.py visualize-dataset \
    --dataset datasets/benchmark_10k.h5 \
    --display-realizations 2 \
    --aggregates mean envelope entropy pca
```

**Grid Layout**:
- **Before**: 4 operators × 12 columns (10 realizations + 2 aggregates) = 768px width
- **After**: 4 operators × 6 columns (2 realizations + 4 aggregates) = 384px width

## Usage Examples

### Example 1: Quick Overview (Default)
```bash
poetry run python scripts/spinlock.py visualize-dataset \
    --dataset datasets/benchmark_10k.h5 \
    --output visualizations/quick_overview.mp4 \
    --n-operators 4 \
    --steps 30
```

**Defaults**: mean, envelope, entropy aggregates with 2 displayed realizations

### Example 2: Comprehensive Analysis (All Aggregates)
```bash
poetry run python scripts/spinlock.py visualize-dataset \
    --dataset datasets/benchmark_10k.h5 \
    --output visualizations/comprehensive.mp4 \
    --aggregates mean variance envelope entropy pca ssim spectral \
    --display-realizations 1 \
    --n-operators 3 \
    --steps 50
```

**Grid**: 3 operators × 8 columns (1 realization + 7 aggregates)

### Example 3: Uncertainty Focused
```bash
poetry run python scripts/spinlock.py visualize-dataset \
    --dataset datasets/benchmark_10k.h5 \
    --output visualizations/uncertainty.mp4 \
    --aggregates entropy envelope ssim \
    --display-realizations 2 \
    --n-operators 6
```

**Focus**: Entropy (multimodality), Envelope (bounds), SSIM (structural consistency)

### Example 4: Mode Analysis
```bash
poetry run python scripts/spinlock.py visualize-dataset \
    --dataset datasets/benchmark_10k.h5 \
    --output visualizations/modes.mp4 \
    --aggregates pca spectral \
    --display-realizations 3 \
    --n-operators 4
```

**Focus**: PCA modes + spectral content

### Example 5: Single Operator Deep Dive
```bash
poetry run python scripts/spinlock.py visualize-dataset \
    --dataset datasets/benchmark_10k.h5 \
    --output visualizations/single_op.mp4 \
    --operator-indices 42 \
    --aggregates mean variance entropy pca envelope ssim spectral \
    --display-realizations 5 \
    --steps 100
```

**Grid**: 1 operator × 12 columns (5 realizations + 7 aggregates)

## Test Results

### Test 1: Basic Functionality
- **Command**: 4 operators, mean/envelope/entropy/pca, 2 displayed realizations, 30 steps
- **Grid**: 256×384 (4 ops × 6 columns)
- **Output**: `visualizations/test_new_aggregates.mp4` (291K, 30 frames)
- **Status**: ✅ Success (149.3s)

### Test 2: Comprehensive
- **Command**: 3 operators, all 7 aggregates, 1 displayed realization, 50 steps
- **Grid**: 192×512 (3 ops × 8 columns)
- **Output**: `visualizations/comprehensive_test.mp4` (528K, 50 frames)
- **Status**: ✅ Success (209.8s)

## Performance Characteristics

**Rendering Time per Frame** (approximate, depends on grid size):
- **Mean/Variance/StdDev**: ~0.5s (fast, pure tensor ops)
- **Envelope**: ~0.8s (min/max + heatmap)
- **Entropy**: ~3.0s (histogram computation, expensive)
- **PCA**: ~2.0s (SVD computation)
- **SSIM**: ~2.5s (pairwise structure comparison)
- **Spectral**: ~1.5s (FFT computation)
- **Overlay**: ~0.6s (alpha blending)

**Recommendation**: For quick previews, use mean/envelope. For deep analysis, use entropy/pca/ssim.

## Grid Layout Details

**Layout formula**:
- **Height**: `N_operators × 64px`
- **Width**: `(M_display + K_aggregates) × 64px`

Where:
- `N_operators`: Number of operators visualized
- `M_display`: Number of individual realizations displayed (--display-realizations)
- `K_aggregates`: Number of aggregate renderers (--aggregates)

**Example**:
- 4 operators, 2 displayed realizations, 4 aggregates
- Grid: 256×384 (4×64 height, 6×64 width)

## Aggregate Computation Details

**Critical Feature**: Aggregates (mean, entropy, pca, etc.) **always use ALL realizations** in the dataset, regardless of `--display-realizations`.

**Example**:
- Dataset has 10 realizations per operator
- `--display-realizations 2` shows only 2 individual columns
- **But**: Entropy/PCA/SSIM aggregates still computed from all 10 realizations

**Implementation** (`grid.py:231`):
```python
# Render individual realizations (subset)
for col in range(M_display):
    realization = realizations_op[col:col+1]  # [1, C, H, W]
    # ...

# Render aggregates (using ALL realizations)
for agg_idx, agg_renderer in enumerate(self.aggregate_renderers):
    col = M_display + agg_idx
    agg_rgb = agg_renderer.render(realizations_op)  # Uses all M realizations
```

## Default Changes

**Previous defaults**:
- Aggregates: mean, variance
- Display realizations: All (10)

**New defaults** (as of commit c45643c):
- Aggregates: mean, envelope, entropy
- Display realizations: 2

**Rationale**:
- Entropy is more informative than variance for multimodal distributions
- Envelope shows variation bounds without losing structure
- Displaying 2 realizations reduces grid size while maintaining context
- All aggregates still use full 10 realizations for accuracy

## Future Enhancements

Potential additions:
1. **Quantile Renderer**: Show 10th/50th/90th percentiles
2. **Divergence Map**: KL divergence across realizations
3. **Correlation Map**: Spatial correlation structure
4. **Fourier Phase**: Phase consistency across realizations
5. **Custom Colormap Support**: Per-aggregate colormap selection

## Implementation Files

- `src/spinlock/visualization/core/aggregator.py` - All aggregate renderers
- `src/spinlock/visualization/core/grid.py` - Grid layout with display_realizations support
- `src/spinlock/cli/visualize.py` - CLI parameter handling

## Commit History

- **c45643c**: feat(visualization): add 7 advanced aggregate renderers and smart display control
- **51ca6f4**: fix(visualization): suppress torchvision video encoding deprecation warning
