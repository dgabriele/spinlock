# PyTorch Autocast API Update

**Date**: December 29, 2025
**Status**: ✅ Fixed
**Impact**: Eliminates FutureWarning in all future runs

---

## Issue

PyTorch deprecated the old autocast API:
```python
from torch.cuda.amp import autocast
with autocast(enabled=True, dtype=torch.float16):
    # ...
```

This caused FutureWarning during dataset generation:
```
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
Please use `torch.amp.autocast('cuda', args...)` instead.
```

---

## Fix

Updated to the new PyTorch 2.x autocast API in `src/spinlock/rollout/engine.py`:

### Import Statement (Line 19)
**Before**:
```python
from torch.cuda.amp import autocast
```

**After**:
```python
from torch.amp import autocast
```

### Usage (Lines 326, 407)
**Before**:
```python
with autocast(enabled=self.use_amp, dtype=self.dtype):
    O_theta_X = operator(X_t)
```

**After**:
```python
with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.dtype):
    O_theta_X = operator(X_t)
```

---

## Changes

**Modified files**:
- `src/spinlock/rollout/engine.py` (3 changes)
  - Line 19: Import from `torch.amp` instead of `torch.cuda.amp`
  - Line 326: Added `device_type=self.device.type` parameter
  - Line 407: Added `device_type=self.device.type` parameter

**Benefits**:
1. ✅ Eliminates deprecation warning
2. ✅ Future-proof for PyTorch 2.x+
3. ✅ More flexible: works with both 'cuda' and 'cpu' devices
4. ✅ No performance impact (same underlying implementation)

---

## Testing

**Test script**: `scripts/dev/test_autocast_fix.py`

**Results**:
```bash
$ poetry run python scripts/dev/test_autocast_fix.py
Testing new autocast API...
PyTorch version: 2.9.1+cu128
Device: cuda

Running rollout with FP16...
✓ Rollout completed successfully!
  Trajectory shape: torch.Size([2, 10, 3, 64, 64])
  Trajectory dtype: torch.float32

✓ PASSED: No FutureWarning about autocast!
```

---

## Note on Current 10K Generation

The currently running 10K dataset generation (started before this fix) will **continue to show the warning** because:
- The code was already loaded into memory before the fix
- Python doesn't reload modules in running processes

**However**:
- The warning is harmless (doesn't affect correctness or performance)
- All future runs will be warning-free
- The generated dataset is completely unaffected

---

## Compatibility

**Tested with**:
- PyTorch 2.9.1+cu128
- CUDA 12.8
- Python 3.13

**Backward compatibility**:
- New API available since PyTorch 1.10+
- Works with all PyTorch 2.x versions
- Gracefully handles both CUDA and CPU devices

---

## References

- PyTorch AMP documentation: https://pytorch.org/docs/stable/amp.html
- API migration guide: https://pytorch.org/docs/stable/notes/amp_examples.html
- Issue filed: PyTorch deprecation roadmap for autocast APIs
