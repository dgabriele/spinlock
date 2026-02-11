# Initial Feature Extraction Refactoring

## Summary

Refactored initial feature extraction from ad-hoc scripts into a unified, well-engineered pipeline following DRY and OOP principles.

## What Changed

### Before (Problems)
- ❌ Hardcoded logic in `pipeline.py` assumed single-channel PDEs `[N, M, H, W]`
- ❌ Standalone script duplicated extraction logic
- ❌ Failed on multi-channel/multi-species inputs (QBM: `[N, 3, 2, 64, 64]`)
- ❌ No automatic shape adaptation
- ❌ Manual intervention required after dataset generation

### After (Solution)
- ✅ **Unified `InitialFeatureExtractionPipeline`** in `src/spinlock/features/initial/extraction_pipeline.py`
- ✅ **Automatic shape detection** and normalization
- ✅ **Adaptive to dataset structure** (channels, species, realizations)
- ✅ **Configurable extractor types** (statistical or manual)
- ✅ **DRY**: Single source of truth for extraction logic
- ✅ **OOP**: Clean, reusable, testable components

## Architecture

### New Module Structure

```
src/spinlock/features/initial/
├── extraction_pipeline.py       # NEW: Unified extraction pipeline
│   ├── InitialFeatureExtractionPipeline (class)
│   ├── ExtractorType (enum)
│   └── extract_initial_features() (convenience function)
├── ic_feature_extractors.py    # Statistical extractor
├── manual_extractors.py         # Manual/pattern extractor
└── __init__.py                  # Exports all components
```

### Shape Adaptation Logic

The pipeline intelligently handles various input shapes:

| Input Shape | Interpretation | Normalization |
|-------------|----------------|---------------|
| `[N, H, W]` | Single-channel | Add channel dim → `[N, 1, H, W]` |
| `[N, C, H, W]` | Multi-channel | Use as-is |
| `[N, M, H, W]` | Realizations | Select first → `[N, 1, H, W]` |
| `[N, M, C, H, W]` | Realizations + channels | Select first → `[N, C, H, W]` |
| `[N, C, S, H, W]` | **Channels × Species** | **Flatten → `[N, C*S, H, W]`** |

The last case is critical for quantum systems like QBM where we have:
- C=3 channels (Re(ψ), Im(ψ), density)
- S=2 species
- Result: 6 effective channels

### Feature Dimensions

The pipeline automatically computes feature dimensions based on channel count:

**Statistical Extractor** (recommended):
- Distributional: 11 features/channel (mean, std, min, max, median, 4 percentiles, skew, kurt)
- Energy: 2 norms/channel + cross-correlations
- Spatial: 8 features/channel (optional, disabled by default)

**Example**: QBM with 6 channels (3×2) → 93D features
- Distributional: 6 × 11 = 66D
- Energy: 6 × 2 + (6×5)/2 = 12 + 15 = 27D
- Total: 93D

**Manual Extractor** (legacy):
- 14 features per channel
- Example: 6 channels → 84D features

## Updated Components

### 1. Dataset Generation Pipeline

**File**: `src/spinlock/dataset/pipeline.py`

```python
def _extract_initial_features(self) -> None:
    """Extract INITIAL features using unified pipeline."""
    from ..features.initial import extract_initial_features, ExtractorType

    extract_initial_features(
        dataset_path=self.config.dataset.output_path,
        extractor_type=ExtractorType.STATISTICAL,  # Better than manual
        device='cpu',
        batch_size=100,
        include_spatial=False,  # Collapsed variance on small grids
        overwrite=True,
        verbose=True,
    )
```

**Benefits**:
- ✅ Automatic shape detection
- ✅ Handles multi-channel/species
- ✅ No more hardcoded assumptions
- ✅ Works for PDE and quantum systems

### 2. Standalone Script

**File**: `scripts/extract_initial_manual_features.py`

Simplified to use the pipeline:

```python
from spinlock.features.initial import extract_initial_features, ExtractorType

def main(dataset_path: str, use_statistical: bool = True) -> int:
    extractor_type = ExtractorType.STATISTICAL if use_statistical else ExtractorType.MANUAL

    extract_initial_features(
        dataset_path=dataset_path,
        extractor_type=extractor_type,
        device='cpu',
        batch_size=100,
        include_spatial=False,
        overwrite=True,
        verbose=True,
    )
    return 0
```

**Before**: 106 lines of duplicated logic
**After**: 42 lines using the pipeline

### 3. Python API

Can now be used programmatically:

```python
from spinlock.features.initial import InitialFeatureExtractionPipeline, ExtractorType

# Direct extraction from arrays
pipeline = InitialFeatureExtractionPipeline(
    extractor_type=ExtractorType.STATISTICAL,
    device='cuda',
    batch_size=128
)
features = pipeline.extract_from_array(my_data)  # [N, D]

# Or extract from HDF5 and save
pipeline.extract_and_save(dataset_path='data.h5')

# Or use convenience function
from spinlock.features.initial import extract_initial_features
extract_initial_features('data.h5', extractor_type='statistical')
```

## Example: QBM Dataset

### Original Extraction (Old Script)
```bash
# Took only first channel dimension → 2 channels → 27D
poetry run python scripts/extract_initial_manual_features.py datasets/qbm_50k.h5
```
Result: 27D features (incomplete - missing Re(ψ) and Im(ψ) channels)

### New Extraction (Unified Pipeline)
```bash
# Automatically detects all channels × species → 6 channels → 93D
poetry run python scripts/extract_initial_manual_features.py datasets/qbm_50k.h5
```
Result: 93D features (complete - uses all quantum information)

**Shape handling**:
```
Input: [50000, 3, 2, 64, 64]  # N=50K, C=3 channels, S=2 species
       ↓
Detected: 3 channels × 2 species → flatten
       ↓
Normalized: [50000, 6, 64, 64]  # N=50K, C*S=6 effective channels
       ↓
Extracted: [50000, 93]  # 93D statistical features
```

## Design Principles Applied

### 1. **DRY (Don't Repeat Yourself)**
- Single extraction pipeline used by:
  - Dataset generation pipeline
  - Standalone script
  - Direct Python API

### 2. **OOP (Object-Oriented Programming)**
- `InitialFeatureExtractionPipeline` class encapsulates:
  - Configuration
  - Shape normalization logic
  - Batch processing
  - Storage management

### 3. **Separation of Concerns**
- Shape detection: `_normalize_input_shape()`
- Feature extraction: Delegated to specific extractors
- Storage: `extract_and_save()`
- Processing: `extract_from_array()`

### 4. **Open/Closed Principle**
- Open for extension: Easy to add new extractor types
- Closed for modification: Existing code doesn't change

### 5. **Configuration over Convention**
- Extractor type: Configurable (statistical vs manual)
- Device: Configurable (CPU vs GPU)
- Spatial features: Configurable (on/off)
- Batch size: Configurable

## Testing

```python
# Test with synthetic data
pipeline = InitialFeatureExtractionPipeline(
    extractor_type=ExtractorType.STATISTICAL,
    device='cpu'
)

# Test various shapes
test_cases = [
    np.random.randn(100, 64, 64),        # Single channel
    np.random.randn(100, 3, 64, 64),      # Multi-channel
    np.random.randn(100, 3, 2, 64, 64),   # Channels × species
]

for data in test_cases:
    features = pipeline.extract_from_array(data)
    print(f"Input: {data.shape} → Output: {features.shape}")
```

## Migration Guide

### For Dataset Generation
No action needed! The pipeline automatically uses the new extraction during generation.

### For Existing Datasets
Re-extract features to get complete information:
```bash
poetry run python scripts/extract_initial_manual_features.py your_dataset.h5
```

### For VQTokenizer Configs
Update `encoder.initial.manual_dim` to match extracted feature dimension:
```python
# Check extracted dimension
import h5py
with h5py.File('your_dataset.h5', 'r') as f:
    dim = f['features/initial/aggregated/features'].shape[1]
    print(f"Feature dimension: {dim}")
```

Then update config:
```yaml
encoder:
  initial:
    manual_dim: 93  # Use actual dimension from above
```

## Future Improvements

1. **Auto-detect feature dimension** in VQTokenizer config
2. **Add more extractor types** (e.g., learned embeddings)
3. **Parallel processing** for very large datasets
4. **Progress bars** for long extractions
5. **Feature validation** (check for NaN, Inf, collapsed variance)

## References

- **Pipeline Code**: `src/spinlock/features/initial/extraction_pipeline.py`
- **Statistical Extractor**: `src/spinlock/features/initial/ic_feature_extractors.py`
- **Manual Extractor**: `src/spinlock/features/initial/manual_extractors.py`
- **Usage Examples**: `docs/FEATURE_EXTRACTION_WORKFLOW.md`

