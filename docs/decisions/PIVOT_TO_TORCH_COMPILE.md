# Pivot from Custom CUDA Kernels to torch.compile()

**Date**: December 29, 2025
**Decision**: Abandon custom CUDA kernel approach, pivot to torch.compile() optimization

## Why We Pivoted

### Performance Results (Custom CUDA Kernels)
- **Target**: 3-5× speedup over PyTorch
- **Actual**: **80× SLOWER** than PyTorch baseline
- **Details**:
  - PyTorch Conv→InstanceNorm→GELU: 1.04 ms/iteration (5×64×128×128)
  - Custom CUDA pipeline: 81.5 ms/iteration
  - Slowdown factor: 78-80×

### What We Tried
1. ✅ Implemented three-kernel pipeline (Conv2D → InstanceNorm → Activation)
2. ✅ Fixed instance normalization bug (global statistics)
3. ✅ Eliminated cudaMalloc/Free overhead (PyTorch-managed workspace)
4. ✅ All numerical accuracy tests passing (max error: 1.36e-03)
5. ❌ Performance remained catastrophically poor

### Root Cause
Fundamental kernel inefficiencies:
- Custom Conv2D kernel likely has poor memory access patterns
- Missing optimizations that cuDNN provides (Winograd, im2col, etc.)
- Suboptimal occupancy and thread utilization
- Would require weeks of low-level optimization work

## New Approach: torch.compile() + PyTorch Optimizations

### Strategy
Instead of custom CUDA, leverage PyTorch's built-in optimization infrastructure:

1. **torch.compile()** - JIT compilation with automatic kernel fusion
2. **Mixed precision (FP16/BF16)** - 2× throughput improvement
3. **Operator-level optimizations** - Replace inefficient layers
4. **Better batching** - Pre-allocate tensors, eliminate Python overhead
5. **torch.jit.script** - TorchScript for critical paths

### Expected Results
- ✅ Proven 2-5× speedups in production
- ✅ Zero custom CUDA code to maintain
- ✅ Works across all GPU architectures
- ✅ Can be implemented in days vs weeks
- ✅ Automatic optimization updates with PyTorch releases

## Files Removed
Dead code from failed CUDA kernel approach:
- `src/fused_ops/fused_convblock.cu` - 80× slower than PyTorch
- `src/fused_ops/conv2d.cu` - Inefficient convolution kernel
- `src/fused_ops/instance_norm.cu` - Normalization implementation
- `src/fused_ops/activation_dropout.cu` - Activation functions
- `src/fused_ops/activations.cu` - GELU approximation
- `include/fused_ops.h` - Dead API definitions
- `tests/test_fused_convblock_accuracy.py` - Tests for removed kernels
- `tests/benchmark_fused_convblock.py` - Benchmark showing failure
- `tests/profile_kernel_breakdown.py` - Profiling script
- `KNOWN_ISSUES.md` - Bug documentation

## Files Retained
Basic CUDA infrastructure kept for potential future use:
- `src/common/dummy_kernel.cu` - Testing infrastructure
- `src/bindings/torch_bindings.cpp` - PyTorch C++ extension (minimal)
- `setup.py` - Build system
- Directory structure for future CUDA work if needed

## Next Steps
See implementation plan in: `docs/optimization/torch_compile_strategy.md`

## Lessons Learned
1. **Don't reinvent cuDNN** - PyTorch's kernels are heavily optimized
2. **Measure first** - Custom kernels need profiling at every stage
3. **Use PyTorch's tools** - torch.compile() is the modern optimization path
4. **Focus on algorithms** - Better ROI than low-level kernel tuning
