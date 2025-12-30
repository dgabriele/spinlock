# torch.compile() Performance Findings

**Date**: December 29, 2025
**Status**: Limited effectiveness for stochastic neural operators

## Results

### Benchmark Configuration
- Operator: 4-layer CNN, 64 base channels
- Input: 5×3×128×128 (batch size 5)
- Device: RTX 3060 Ti
- PyTorch: 2.9.1+cu128

### Performance

| Mode | Time (ms/iter) | Speedup |
|------|----------------|---------|
| Eager (baseline) | 38.4 ms | 1.0× |
| torch.compile(max-autotune) | 35.2 ms | **1.09×** |

**Result**: Only 1.09× speedup instead of expected 1.5-2×

### Analysis

**Why so slow?**

1. **Stochastic Operations Break Graph**
   - Dropout layers use random number generation
   - StochasticBlock adds Gaussian noise
   - These create graph breaks that prevent full optimization

2. **cuDNN Already Optimized**
   - PyTorch's eager mode uses cuDNN for Conv2d, InstanceNorm
   - These are already highly optimized
   - torch.compile() can't improve much on top of cuDNN

3. **Small Operator**
   - 4 layers is relatively shallow
   - Most time spent in individual cuDNN calls
   - Little opportunity for inter-layer fusion

### Recommendations

**For Spinlock Dataset Generation:**

❌ **torch.compile() alone**: Not worth the complexity
- Only 9% speedup
- 15-18 second compilation overhead per operator
- Breaks with stochastic operations

✅ **Mixed Precision (BF16)**: Much better approach
- Expected 2× speedup (100% improvement vs 9%)
- No compilation overhead
- Works perfectly with stochastic operations
- Simple to implement

✅ **Combined Strategy**:
1. **Primary**: Mixed precision (BF16) → 2× speedup
2. **Secondary**: Batching optimizations → 1.5× speedup
3. **Tertiary**: Operator-level changes (GroupNorm) → 1.2× speedup
4. **Skip**: torch.compile() (minimal benefit for our use case)

**Expected Total**: 3.6× speedup without torch.compile()

### When torch.compile() WOULD Help

torch.compile() is most effective for:
- **Pure feedforward networks** (no stochastic operations)
- **Deep networks** (10+ layers) where fusion matters
- **Custom operations** not in cuDNN
- **Training workloads** (backward pass fusion)

For our use case (shallow stochastic operators, inference only), **mixed precision is the better path**.

## Next Steps

1. ✅ Implement automatic mixed precision (BF16)
2. ✅ Benchmark BF16 speedup (expecting 2×)
3. ✅ Add batching optimizations
4. ❌ Skip torch.compile() integration (keep code for potential future use, but don't enable by default)

