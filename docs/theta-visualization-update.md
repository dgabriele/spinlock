# VQVAE Dashboard Theta Family Support - Update Summary

## Overview

Updated the VQVAE visualization dashboards to support the theta (architectural/parameter) feature family. The VQTokenizer now supports three families:
1. **temporal** - trajectory temporal features
2. **initial** - initial condition features
3. **theta** - operator parameters (14D parameter vectors in [0,1] unit hypercube)

## Files Modified

### 1. `/home/daniel/projects/spinlock/src/spinlock/cli/visualize_vqvae.py`

**Changes:**
- **Checkpoint detection** (lines 116-128): Updated to support multiple checkpoint naming conventions
  - Now checks for: `final_model.pt`, `best_model.pt`, `vq_tokenizer_final.pt`, `vq_tokenizer_best.pt`
  - Handles both legacy and newer checkpoint naming patterns
  - Provides clear error messages listing all supported filenames

**Impact:** CLI now works with both legacy checkpoints and newer theta-enabled checkpoints that use the `vq_tokenizer_*.pt` naming convention.

### 2. `/home/daniel/projects/spinlock/src/spinlock/visualization/vqvae/utils.py`

**Changes:**
- **New helper function** `find_checkpoint_file()` (lines 247-275): Centralized checkpoint file discovery
  - Tries candidates in order: `final_model.pt`, `vq_tokenizer_final.pt`, `best_model.pt`, `vq_tokenizer_best.pt`
  - Provides single source of truth for checkpoint naming conventions
  - Used by all visualization modules for consistency

- **Feature family parsing** (lines 304-334): Rewrote the family detection logic to parse from `group_indices` keys
  - Now detects families from keys like `"temporal_group_1"`, `"theta_group_1"`, `"initial_group_2"`
  - Properly maps feature indices to their families
  - Maintains backward compatibility with legacy checkpoints

- **Feature name abbreviations** (lines 468-493): Added theta abbreviation
  - `"theta"` → `"θ"` (Greek theta symbol)
  - Also added `"initial"` → `"ic"` for completeness

**Key improvement:** The new approach is more robust because it derives families directly from the group structure rather than relying on config or feature name conventions.

### 3. `/home/daniel/projects/spinlock/src/spinlock/visualization/vqvae/semantic_dashboard.py`

**Changes:**
- **Feature ordering** (line 45): Updated `family_order` list
  - Old: `["summary", "temporal", "architecture"]`
  - New: `["summary", "temporal", "theta", "initial", "architecture"]`
  - This controls the vertical ordering in the feature-category matrix

**Impact:** Theta features now appear in their own section in the feature-category matrix, between temporal and initial families.

### 4. `/home/daniel/projects/spinlock/src/spinlock/visualization/vqvae/engineering_dashboard.py`

**Changes:**
- **Architecture diagram** (lines 69-71): Enhanced family display in encoder box
  - Added family abbreviation mapping: `{"temporal": "temp", "theta": "θ", "initial": "init", "architecture": "arch"}`
  - Increased display limit from 3 to 4 families
  - Shows abbreviated names for better readability

**Impact:** The architecture diagram now clearly shows when theta encoder is present in the model.

### 5. `/home/daniel/projects/spinlock/src/spinlock/visualization/vqvae/topological_dashboard.py`

**Changes:**
- **Checkpoint loading** (line 42): Updated to use `find_checkpoint_file()` helper
  - Supports both legacy and new checkpoint naming conventions

- **Codebook extraction** (lines 52-95): Enhanced to support new quantizer structure
  - Legacy structure: `vq_layers.{idx}.embedding.weight` and `ema_cluster_size`
  - New structure: `quantizers.{group}_{level}.embedding.weight` and `ema_cluster_size`
  - Automatically detects which structure is present in checkpoint
  - Handles theta, temporal, and initial quantizer groups

- **t-SNE label generation** (lines 271-282): Updated to handle new codebook key format
  - Legacy format: `cb_0`, `cb_1`, etc.
  - New format: `temporal_group_1_L0`, `theta_group_1_L0`, etc.
  - Uses stable hash for consistent coloring with new format

**Impact:** Topological dashboard now works with both legacy flat VQ structures and newer hierarchical quantizer structures, correctly displaying theta-related codebooks.

## Testing

Tested with production theta-enabled checkpoint (`checkpoints/vqvae/theta_topo_50k/`) containing:
- 14 theta features (operator parameters)
- 345 temporal features (trajectory dynamics)
- 42 initial features (initial conditions)
- 23 total categories (1 theta group, 20 temporal groups, 2 initial groups)
- 3 quantization levels per category

### Test Results

✅ **All Dashboards Generated Successfully**
```bash
poetry run spinlock visualize-vqvae \
  --checkpoint checkpoints/vqvae/theta_topo_50k/ \
  --output visualizations/test_theta/ \
  --type all --no-display --dpi 100
```

Output files created:
- `theta_topo_50k_engineering.png` (93.1 KB)
- `theta_topo_50k_topological.png` (163.7 KB)
- `theta_topo_50k_semantic.png` (191.8 KB)

✅ **Semantic Dashboard** - Successfully displays:
- Feature-category matrix with theta row section
- Family legend includes theta (θ symbol)
- Codebook utilization for theta categories
- Category correlation including theta

✅ **Engineering Dashboard** - Successfully displays:
- Architecture diagram shows "θ" in encoder list
- Training curves (family-agnostic, no changes needed)
- Loss components (family-agnostic, no changes needed)

✅ **Topological Dashboard** - Successfully displays:
- t-SNE embeddings for all codebooks including theta quantizers
- Usage heatmap includes theta categories (theta_group_1_L0, L1, L2)
- Similarity matrix includes theta categories
- Correctly handles new `quantizers.*` structure

✅ **Backward Compatibility Test**
```bash
poetry run spinlock visualize-vqvae \
  --checkpoint checkpoints/vqvae/50k_baseline/ \
  --output visualizations/test_backward_compat/ \
  --type all --no-display --dpi 100
```

Legacy checkpoint (without theta) continues to work correctly:
- Loads with older `vq_layers.*` structure
- Displays temporal and initial families only
- No errors or warnings related to theta support

## Usage

### Command Line
```bash
# Visualize checkpoint with theta support
poetry run spinlock visualize-vqvae \
  --checkpoint checkpoints/vqvae/theta_enabled_model/ \
  --output visualizations/ \
  --type all

# Just semantic dashboard
poetry run spinlock visualize-vqvae \
  --checkpoint checkpoints/vqvae/theta_enabled_model/ \
  --type semantic
```

### Python API
```python
from spinlock.visualization.vqvae import (
    create_semantic_dashboard,
    create_engineering_dashboard,
    create_topological_dashboard,
)

# Create semantic dashboard
fig = create_semantic_dashboard(
    checkpoint_path="checkpoints/vqvae/theta_enabled_model/",
    output_path="visualizations/semantic.png",
    compute_topo=True,  # Include topographic metrics
    dpi=150,
)
```

## Backward Compatibility

All changes are backward compatible:
- Checkpoints without theta family continue to work as before
- The family detection has multiple fallback strategies
- Legacy feature naming (family::name format) still supported
- No breaking changes to existing visualization APIs

## Future Enhancements

Potential improvements for future work:
1. Add theta-specific visualizations (e.g., parameter space coverage)
2. Show per-family reconstruction quality metrics
3. Visualize how theta features cluster compared to temporal/initial
4. Add parameter sensitivity analysis to dashboards

## Verification Checklist

- [x] Family parsing correctly identifies theta from group_indices keys
- [x] Semantic dashboard displays theta in feature-category matrix
- [x] Semantic dashboard includes theta in family legend (θ symbol)
- [x] Engineering dashboard shows theta in architecture diagram (θ symbol)
- [x] Topological dashboard includes theta categories
- [x] Topological dashboard handles new `quantizers.*` structure
- [x] Topological dashboard handles legacy `vq_layers.*` structure
- [x] CLI command works end-to-end with theta checkpoints
- [x] CLI supports both checkpoint naming conventions
- [x] Backward compatibility maintained for non-theta checkpoints
- [x] No breaking changes to existing APIs
- [x] Feature name abbreviations use Greek theta symbol (θ)
- [x] All three dashboards generate successfully for theta checkpoint
- [x] All three dashboards generate successfully for legacy checkpoint

## Related Files

For reference, theta implementation in the tokenizer:
- `/home/daniel/projects/spinlock/src/spinlock/tokens/encoders/theta.py` - ThetaMLPEncoder (14D → 64D → 32D)
- `/home/daniel/projects/spinlock/src/spinlock/features/grouping/theta.py` - ThetaFeatureGrouper
- `/home/daniel/projects/spinlock/src/spinlock/features/grouping/factory.py` - Factory with theta support

## Key Technical Improvements

### 1. Unified Checkpoint Discovery
Previously, each visualization module independently looked for checkpoint files. Now:
- Single `find_checkpoint_file()` helper function in `utils.py`
- Consistent behavior across all dashboards
- Clear error messages when checkpoint not found

### 2. Multi-Structure Quantizer Support
The topological dashboard now handles two fundamentally different checkpoint structures:
- **Legacy structure**: Flat `vq_layers.{idx}` with sequential indexing
- **New structure**: Hierarchical `quantizers.{group}_{level}` with semantic naming

This enables visualizing both:
- Older checkpoints trained before feature grouping refactor
- Newer checkpoints with explicit family/group organization

### 3. Robust Family Detection
Family detection now uses multiple fallback strategies:
1. Parse from `group_indices` keys (primary, most reliable)
2. Check `families` config in checkpoint
3. Parse from feature names with `::` delimiter
4. Default to "behavioral" family as fallback

This ensures visualizations work even with incomplete checkpoint metadata.

## Summary

The VQVAE visualization system now fully supports the theta family. All three dashboard types (semantic, engineering, topological) correctly display theta-related information when present in checkpoints. The implementation is backward compatible and follows the existing visualization patterns.

**Key achievements:**
- ✅ Theta features visualized in all three dashboard types
- ✅ Supports both legacy and new checkpoint structures
- ✅ Supports both legacy and new naming conventions
- ✅ Zero breaking changes to existing code
- ✅ Production-tested with real theta checkpoint (50K samples, 401 total features)
- ✅ Backward compatibility verified with legacy checkpoint
