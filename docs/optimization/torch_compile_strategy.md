# PyTorch Optimization Strategy for Spinlock Dataset Generation

**Goal**: Achieve 10× speedup for temporal dataset generation through PyTorch-native optimizations

**Current Baseline**: 74 hours for 10K operators (T=500, 128×128, RTX 3060 Ti)
**Target**: <7.4 hours (10× speedup minimum)

---

## Phase 1: torch.compile() with Kernel Fusion (Week 1)

### Overview
PyTorch 2.0+ JIT compilation with automatic kernel fusion eliminates Python overhead and fuses operations at the CUDA level.

### Implementation

#### 1.1 Basic torch.compile() Integration

**File**: `src/spinlock/operators/cnn.py`

```python
import torch

class OptimizedCNN(torch.nn.Module):
    """CNN operator with torch.compile() optimization"""

    def __init__(self, params, use_compile=True, compile_mode="max-autotune"):
        super().__init__()

        # Build operator as usual
        self.layers = self._build_layers(params)

        # Apply torch.compile() to forward pass
        if use_compile and torch.__version__ >= "2.0.0":
            # max-autotune: Aggressive optimization, tries multiple kernel variants
            # reduce-overhead: Minimize Python overhead
            # default: Balanced approach
            self.forward = torch.compile(
                self.forward,
                mode=compile_mode,
                fullgraph=True,  # Require single graph (no Python breaks)
                dynamic=False     # Static shapes for better optimization
            )

    def forward(self, x):
        # This entire forward pass will be JIT compiled and fused
        for layer in self.layers:
            x = layer(x)
        return x
```

#### 1.2 Compile Entire Operator Pipeline

**File**: `src/spinlock/rollout/engine.py`

```python
def _prepare_operator_for_rollout(self, operator):
    """Prepare operator with torch.compile() optimization"""

    # Compile the full forward pass (policy update included)
    if hasattr(operator, '_compiled_forward'):
        return operator._compiled_forward

    # Create compiled version with policy
    @torch.compile(mode="max-autotune", fullgraph=True)
    def compiled_forward_with_policy(x_t, policy_params):
        # Operator forward
        x_raw = operator(x_t)

        # Policy update (convex/residual/autoregressive)
        if policy_params['type'] == 'convex':
            alpha = policy_params['alpha']
            x_next = alpha * x_raw + (1 - alpha) * x_t
        elif policy_params['type'] == 'residual':
            dt = policy_params['dt']
            x_next = x_t + dt * x_raw
        else:  # autoregressive
            x_next = x_raw

        return x_next

    operator._compiled_forward = compiled_forward_with_policy
    return compiled_forward_with_policy
```

### Expected Speedup: 1.5-2×
- Eliminates Python interpreter overhead (~20-30% of runtime)
- Fuses Conv→Norm→Activation chains automatically
- Optimizes memory layouts

---

## Phase 2: Mixed Precision Training (Week 1)

### Overview
FP16/BF16 computation provides 2× throughput improvement on modern GPUs with negligible accuracy loss for neural operators.

### Implementation

#### 2.1 Automatic Mixed Precision (AMP)

**File**: `src/spinlock/rollout/engine.py`

```python
from torch.cuda.amp import autocast

class OperatorRollout:
    def __init__(self, operator, policy, num_timesteps, device='cuda', use_amp=True):
        self.operator = operator
        self.use_amp = use_amp and torch.cuda.is_available()
        self.dtype = torch.bfloat16 if self.use_amp else torch.float32

    def evolve_operator(self, initial_condition, num_realizations):
        """Rollout with automatic mixed precision"""

        # Convert IC to appropriate dtype
        X_t = initial_condition.to(dtype=self.dtype)

        with autocast(enabled=self.use_amp, dtype=torch.bfloat16):
            for t in range(self.num_timesteps):
                # All operator computations in BF16
                X_next = self.operator(X_t)

                # Policy update also in BF16
                X_t = self.policy.update(X_next, X_t)

        # Convert back to FP32 for feature extraction (if needed)
        return X_t.to(dtype=torch.float32)
```

#### 2.2 BF16 vs FP16 Selection

```python
def select_precision_dtype(device):
    """Choose best dtype based on GPU architecture"""

    if not torch.cuda.is_available():
        return torch.float32

    # Check compute capability
    capability = torch.cuda.get_device_capability(device)
    major, minor = capability

    # Ampere (RTX 30xx) and newer: BF16 preferred
    if major >= 8:
        return torch.bfloat16  # Better dynamic range, no loss scaling needed

    # Turing (RTX 20xx): FP16 with loss scaling
    elif major >= 7:
        return torch.float16   # Requires gradient scaling

    # Older: FP32 only
    else:
        return torch.float32
```

### Expected Speedup: 1.8-2×
- 2× memory bandwidth (64-bit → 32-bit)
- 2× compute throughput on tensor cores
- Minimal accuracy impact for PDE operators

---

## Phase 3: Operator-Level Optimizations (Week 2)

### 3.1 Replace InstanceNorm with GroupNorm

**Motivation**: InstanceNorm requires global reductions per sample, GroupNorm is faster

```python
# BEFORE (slow):
self.norm = nn.InstanceNorm2d(channels, affine=True)

# AFTER (faster):
self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
```

**Speedup**: 1.1-1.2× (GroupNorm is 15-20% faster)

### 3.2 Use Fused Activation Functions

```python
# BEFORE (separate kernels):
x = self.conv(x)
x = self.norm(x)
x = F.gelu(x)

# AFTER (fused with torch.compile):
# torch.compile() will automatically fuse these!
# But we can help by using inplace operations where safe
x = self.conv(x)
x = self.norm(x)
x = F.gelu(x, approximate='tanh')  # Tanh approximation is faster
```

### 3.3 Pre-allocate Reusable Buffers

```python
class OperatorRollout:
    def __init__(self, ...):
        # Pre-allocate workspace tensors (avoid allocations in hot loop)
        self.workspace = {
            'x_buffer_1': None,
            'x_buffer_2': None,
        }

    def evolve_operator(self, initial_condition, num_realizations):
        B, C, H, W = num_realizations, *initial_condition.shape

        # Allocate once
        if self.workspace['x_buffer_1'] is None:
            self.workspace['x_buffer_1'] = torch.empty(
                B, C, H, W, device=self.device, dtype=self.dtype
            )
            self.workspace['x_buffer_2'] = torch.empty_like(
                self.workspace['x_buffer_1']
            )

        # Ping-pong between buffers (avoid allocations)
        x_current = self.workspace['x_buffer_1']
        x_next = self.workspace['x_buffer_2']

        x_current.copy_(initial_condition)

        for t in range(self.num_timesteps):
            # In-place operations where possible
            self.operator(x_current, out=x_next)

            # Swap buffers (no allocation!)
            x_current, x_next = x_next, x_current

        return x_current
```

### Expected Speedup: 1.2-1.3×

---

## Phase 4: Batching Optimizations (Week 2)

### 4.1 Increase Batch Size with Gradient Checkpointing

**Problem**: Memory limits batch size to N=2-3

**Solution**: Gradient checkpointing (for training) or streaming (for inference)

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # Trade compute for memory (recompute activations during backward)
        return checkpoint(self.layer, x, use_reentrant=False)
```

**For Inference** (dataset generation):
```python
# Process in micro-batches, accumulate results
def batch_inference(operator, inputs, micro_batch_size=10):
    results = []

    with torch.no_grad():  # No gradients = 2× memory reduction
        for i in range(0, len(inputs), micro_batch_size):
            batch = inputs[i:i+micro_batch_size]
            output = operator(batch)
            results.append(output.cpu())  # Move to CPU to free GPU memory

    return torch.cat(results)
```

### 4.2 Optimize Dataloader Pipeline

```python
# Multi-process data loading for IC generation
ic_loader = DataLoader(
    ic_dataset,
    batch_size=batch_size,
    num_workers=4,        # Parallel IC generation
    pin_memory=True,      # Faster CPU→GPU transfer
    prefetch_factor=2,    # Pre-load next batch
    persistent_workers=True  # Keep workers alive
)
```

### Expected Speedup: 1.5-2× (from larger batch sizes)

---

## Phase 5: Profile-Guided Optimization (Week 3)

### 5.1 Identify Bottlenecks

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    for t in range(100):
        x = operator(x)

# Analyze results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export for visualization
prof.export_chrome_trace("operator_trace.json")
```

### 5.2 Optimize Based on Profile

**Common bottlenecks**:
1. **Small kernel launches** → Batch operations
2. **CPU→GPU transfers** → Pre-allocate on GPU
3. **Synchronization points** → Use asynchronous ops
4. **Memory allocations** → Pre-allocate buffers

---

## Expected Total Speedup

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| Baseline | 1.0× | 1.0× |
| torch.compile() | 1.8× | 1.8× |
| Mixed precision (BF16) | 2.0× | 3.6× |
| Operator optimizations | 1.2× | 4.3× |
| Batching improvements | 1.8× | 7.7× |
| Profile-guided tuning | 1.3× | **10×** |

**Conservative estimate**: 8-12× speedup
**Optimistic estimate**: 12-15× speedup

**10K operator generation time**: 74h → **6-9 hours** ✓

---

## Implementation Checklist

### Week 1: Core Optimizations
- [ ] Add `torch.compile()` to CNN operator forward pass
- [ ] Compile policy update with operator
- [ ] Implement automatic mixed precision (BF16)
- [ ] Benchmark: measure speedup on 100-operator test set

### Week 2: Refinement
- [ ] Replace InstanceNorm with GroupNorm
- [ ] Pre-allocate workspace buffers
- [ ] Implement micro-batching for larger effective batch size
- [ ] Add dataloader optimizations
- [ ] Benchmark: measure cumulative speedup

### Week 3: Profiling & Tuning
- [ ] Run torch.profiler on full pipeline
- [ ] Identify and fix top 5 bottlenecks
- [ ] Test on full 10K operator generation
- [ ] Document final speedup results

---

## Success Criteria

**Minimum** (Week 1):
- ✅ 3× speedup from torch.compile() + mixed precision
- ✅ 10K operators in <25 hours

**Target** (Week 2):
- ✅ 8× speedup from all optimizations
- ✅ 10K operators in <10 hours

**Stretch** (Week 3):
- ✅ 12× speedup with profile-guided tuning
- ✅ 10K operators in <6 hours

---

## Rollback Plan

If torch.compile() causes issues:
1. **Fallback to eager mode**: Just disable compile flag
2. **Selective compilation**: Only compile critical paths
3. **Use torch.jit.script**: More predictable than compile
4. **Keep all other optimizations**: Mixed precision, batching, etc.

---

## References

- PyTorch 2.0 torch.compile() docs: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- Mixed precision training: https://pytorch.org/docs/stable/amp.html
- Profiler guide: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Performance tuning: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
