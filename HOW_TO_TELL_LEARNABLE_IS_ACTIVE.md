# How to Tell if Learnable Assignment Mode is Active

## ✅ Quick Checklist

When you run training, look for these indicators:

### 1. **Config Shows Learnable** ✓
```
Category Discovery:
  Assignment: learnable    ← Should say "learnable" not "auto"
```

### 2. **Model Building Shows Learnable Mode** ✓
```
[LEARNABLE ASSIGNMENT MODE]
Creating learnable categorical VQ-VAE with end-to-end assignment learning

Initializing assignment matrix from clustering...
  Assignment matrix initialized: <PerFamilyAssignmentMatrix or SoftAssignmentMatrix>
  Per-family mode: 5 families  ← If using per-family
```

### 3. **Trainer Initialization Shows Learnable** ✓
```
Learnable assignment training enabled:
  Assignment LR: 0.001
  Temperature schedule: linear
  Gradient clip norm: 1.0
```

### 4. **Training Epochs Show Assignment Metrics** ✓
```
Epoch 10/500 (12.3s): train=0.0234 val=0.0289
  Components: recon=0.0180 vq=0.0024 ortho=0.0015 info=0.0008 topo=0.0007
              assign_orthogonality=0.0003   ← NEW METRIC
              assign_balance=0.0001         ← NEW METRIC
              assign_total=0.0004           ← NEW METRIC
  Temperature: 0.90                         ← NEW METRIC (annealing 1.0→0.1)
  Utilization: 23.4%
```

## ❌ Signs It's NOT Using Learnable Mode

### Warning Message
If you see this, it fell back to static mode:
```
[WARNING] Learnable assignment mode not yet supported with hybrid INITIAL encoding
  Falling back to standard categorical VQ-VAE with static assignments
```

### Missing Metrics
If epoch logs look like this (NO assignment metrics):
```
Epoch 10/500 (12.3s): train=0.0234 val=0.0289
  Components: recon=0.0180 vq=0.0024 ortho=0.0015 info=0.0008 topo=0.0007
  Utilization: 23.4%
```
→ **Static mode** (no `assign_*` or `Temperature` metrics)

## 🔧 Common Issues & Fixes

### Issue 1: Hybrid INITIAL Encoding

**Problem:**
```yaml
families:
  initial:
    encoder: initial_hybrid  # ← Triggers fallback
```

**Fix:**
Use identity encoder instead:
```yaml
families:
  initial:
    encoder: identity  # ← Compatible with learnable
```

**Or** wait for future enhancement to support hybrid with learnable.

### Issue 2: Config Still Shows "auto"

**Problem:**
```yaml
training:
  category_assignment: "auto"  # ← Wrong!
```

**Fix:**
```yaml
training:
  category_assignment: "learnable"  # ← Correct!
```

### Issue 3: Missing learnable_assignment Section

**Problem:** Config has `category_assignment: learnable` but no parameters.

**Fix:** Add learnable_assignment section:
```yaml
learnable_assignment:
  temperature_start: 1.0
  temperature_end: 0.1
  temperature_schedule: "linear"
  orthogonality_weight: 0.1
  balance_weight: 0.05
  assignment_lr: 0.001
```

## 📊 Full Example Output (Learnable Active)

```
======================================================================
VQ-VAE TRAINING CONFIGURATION
======================================================================
Category Discovery:
  Assignment: learnable   ✓

Training:
  Epochs:        100
  ...
======================================================================

...

Building VQ-VAE model...

[LEARNABLE ASSIGNMENT MODE]                           ✓
Creating learnable categorical VQ-VAE with end-to-end assignment learning

Initializing assignment matrix from clustering...
  Assignment matrix initialized: PerFamilyAssignmentMatrix(...)

Model created:
  Total parameters:     43,601,340
  ...

Initializing trainer...

Learnable assignment training enabled:                ✓
  Assignment LR: 0.001
  Temperature schedule: linear
  Gradient clip norm: 1.0

======================================================================
VQ-VAE TRAINING
======================================================================
Epochs: 100
...

Epoch 1/100 (15.2s): train=15.234 val=16.891
  Components: recon=12.180 vq=1.024 ortho=0.815 info=0.508 topo=0.707
              assign_orthogonality=0.423   ✓
              assign_balance=0.156         ✓
              assign_total=0.579           ✓
  Temperature: 1.00                        ✓
  Utilization: 11.2%

Epoch 10/100 (14.8s): train=0.523 val=0.689
  Components: recon=0.380 vq=0.024 ortho=0.015 info=0.008 topo=0.007
              assign_orthogonality=0.003   ✓
              assign_balance=0.001         ✓
              assign_total=0.004           ✓
  Temperature: 0.91                        ✓ (annealing)
  Utilization: 23.4%

...
```

## 🎯 Quick Test Command

To test with a simple config that definitely uses learnable mode:

```bash
poetry run spinlock train-vqvae \
  --config configs/vqvae/learnable_simple_test.yaml \
  --epochs 5 \
  --verbose
```

Then look for the ✓ indicators listed above!
