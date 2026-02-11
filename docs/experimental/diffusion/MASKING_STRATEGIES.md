# Masking Strategy Training Guide

Comparison of three approaches for training with multiple masking strategies.

---

## TL;DR Recommendation

**For baseline:** Start with **single strategy (RANDOM)**
- Simplest to debug, provides baseline metrics
- Config: `configs/baseline_diffusion.yaml`

**If baseline works well:** Try **curriculum learning** to potentially improve quality via coarse-to-fine progression
- Config: `configs/curriculum_learning.yaml`

**If baseline struggles:** Try **mixed training** for better generalization
- Config: `configs/mixed_training.yaml`

---

## Option 1: Single Strategy (Baseline) ✅ RECOMMENDED START

### What It Is
Train with one masking strategy throughout (typically RANDOM at 50% mask probability).

### Pros
- ✅ Simplest to implement and debug
- ✅ Clear baseline metrics
- ✅ RANDOM is most general case
- ✅ Fastest experimentation cycle

### Cons
- ❌ Less robust to distribution shift
- ❌ No coarse-to-fine learning signal

### When to Use
- **First experiment**: Always start here
- **Debugging**: Isolate masking from other issues
- **Quick validation**: Test architecture changes

### Config
```yaml
# configs/baseline_diffusion.yaml
masking:
  strategy: "random"  # Single strategy
  mask_probability: 0.5
```

### Command
```bash
python experiments/diffusion/scripts/train.py \
  --config experiments/diffusion/configs/baseline_diffusion.yaml
```

---

## Option 2: Curriculum Learning (Staged) 🎓 BETTER STRUCTURE

### What It Is
Train in stages with increasing difficulty:
1. **Stage 1**: COARSE_ONLY (L0 only) - Learn hierarchical structure
2. **Stage 2**: HIERARCHICAL (L0+L1) - Refine to medium detail
3. **Stage 3**: RANDOM (all patterns) - Generalize

### Pros
- ✅ Mimics hierarchical token structure (coarse-to-fine)
- ✅ Better interpretability (know what's learned at each stage)
- ✅ Can tune hyperparameters per stage
- ✅ Prevents early confusion from hard patterns
- ✅ Natural alignment with VQTokenizer v2 design

### Cons
- ❌ More complex training pipeline
- ❌ Need to decide stage transitions (how many epochs?)
- ❌ Risk of catastrophic forgetting
- ❌ Longer total training time (stage overhead)

### When to Use
- **After baseline works**: Use as potential quality improvement
- **Hierarchical emphasis**: When L0→L1→L2 emergence is critical
- **Debugging hierarchical issues**: Isolate which level fails
- **Research**: Study how diffusion learns coarse-to-fine

### Config
```yaml
# configs/curriculum_learning.yaml
curriculum:
  stages:
    - name: "stage1_coarse"
      strategy: "coarse_only"
      num_epochs: 10
    - name: "stage2_hierarchical"
      strategy: "hierarchical"
      num_epochs: 10
    - name: "stage3_random"
      strategy: "random"
      num_epochs: 10
```

### Predefined Curricula
```python
from experiments.diffusion.training import (
    get_coarse_to_fine_curriculum,  # Standard 10+10+10
    get_fast_curriculum,             # Quick 5+10
    get_fine_tuning_curriculum,      # With decreasing LR
)
```

### Command
```bash
# Requires custom script (see below)
python experiments/diffusion/scripts/train_curriculum.py \
  --config experiments/diffusion/configs/curriculum_learning.yaml
```

---

## Option 3: Mixed Training (Multi-Task) 🎲 BETTER ROBUSTNESS

### What It Is
Each batch randomly samples a masking strategy with configured weights:
- 50% RANDOM (general)
- 30% COARSE_ONLY (hierarchical)
- 20% HIERARCHICAL (refinement)

### Pros
- ✅ More robust model (handles diverse patterns)
- ✅ Natural data augmentation
- ✅ Single training run (simpler than curriculum)
- ✅ No catastrophic forgetting risk
- ✅ Model learns to adapt to different mask distributions

### Cons
- ❌ Harder to debug (which strategy is failing?)
- ❌ May confuse model early in training
- ❌ No control over difficulty progression
- ❌ Potentially slower convergence

### When to Use
- **After baseline works**: Use for robustness improvement
- **Deployment**: When inference masking is unpredictable
- **Data augmentation**: Prevent overfitting to one pattern
- **Multi-task learning**: Want single model for all patterns

### Config
```yaml
# configs/mixed_training.yaml
masking:
  type: "mixed"
  strategies:
    - strategy: "random"
      weight: 0.5  # 50% of batches
    - strategy: "coarse_only"
      weight: 0.3  # 30% of batches
    - strategy: "hierarchical"
      weight: 0.2  # 20% of batches
```

### Command
```bash
# Requires modified train.py to support mixed masking
python experiments/diffusion/scripts/train.py \
  --config experiments/diffusion/configs/mixed_training.yaml
```

---

## Comparison Table

| Aspect | Single (Baseline) | Curriculum | Mixed |
|--------|------------------|------------|-------|
| **Complexity** | ⭐ Simple | ⭐⭐⭐ Complex | ⭐⭐ Moderate |
| **Training Time** | Fast | Slow (stages) | Fast |
| **Debugging** | ⭐⭐⭐ Easy | ⭐⭐ Medium | ⭐ Hard |
| **Robustness** | ⭐ Low | ⭐⭐ Medium | ⭐⭐⭐ High |
| **Hierarchical Learning** | ⭐ Implicit | ⭐⭐⭐ Explicit | ⭐⭐ Mixed |
| **Interpretability** | ⭐⭐⭐ Clear | ⭐⭐⭐ Clear | ⭐ Unclear |
| **Use Case** | Baseline | Research | Production |

---

## Recommended Workflow

### Phase 1: Baseline (Week 1)
```bash
# 1. Train baseline RANDOM
python train.py --config baseline_diffusion.yaml

# 2. Evaluate
python evaluate.py --checkpoint results/baseline_50steps/best.pt

# 3. Check metrics:
#    - Overall accuracy > 60%?
#    - L0 accuracy > 75%?
#    - Training stable?
```

**Decision Point:**
- ✅ **Baseline works well** → Try curriculum or mixed for improvement
- ❌ **Baseline fails** → Debug architecture/hyperparameters first

### Phase 2: Advanced Strategies (Week 2)

**If hierarchical emergence is important:**
```bash
# Try curriculum learning
python train_curriculum.py --config curriculum_learning.yaml

# Expected benefit: Better L0→L1→L2 progression
```

**If robustness is important:**
```bash
# Try mixed training
python train.py --config mixed_training.yaml

# Expected benefit: More stable across masking distributions
```

### Phase 3: Ablation (Week 3)

Compare all three:
```python
results = {
    'baseline': evaluate('baseline_50steps/best.pt'),
    'curriculum': evaluate('curriculum_learning/best.pt'),
    'mixed': evaluate('mixed_training/best.pt'),
}

# Metrics to compare:
# - Overall token accuracy
# - Per-level accuracy (L0, L1, L2)
# - Robustness across masking strategies
# - Training time and stability
```

---

## Implementation Notes

### Mixed Training Setup
The train.py script needs minor modification to support mixed masking:

```python
# In train.py, replace mask generator creation:
if config['masking'].get('type') == 'mixed':
    from experiments.diffusion.data import MixedMaskGenerator
    strategies = [
        (MaskingStrategy(s['strategy']), s['weight'])
        for s in config['masking']['strategies']
    ]
    mask_generator = MixedMaskGenerator(
        strategies=strategies,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        seed=config['masking'].get('seed', 42),
    )
else:
    # Single strategy (existing code)
    mask_generator = HierarchicalMaskGenerator(...)
```

### Curriculum Training Setup
Create `scripts/train_curriculum.py`:

```python
from experiments.diffusion.training import (
    CurriculumDiffusionTrainer,
    CurriculumStage,
)

# Parse stages from config
stages = [
    CurriculumStage(
        name=s['name'],
        strategy=MaskingStrategy(s['strategy']),
        num_epochs=s['num_epochs'],
        learning_rate=s.get('learning_rate'),
        mask_probability=s.get('mask_probability'),
    )
    for s in config['curriculum']['stages']
]

# Create curriculum trainer
trainer = CurriculumDiffusionTrainer(
    curriculum_stages=stages,
    vocab_sizes=vocab_sizes,
    category_level_info=category_level_info,
    ...
)

# Train through all stages
history = trainer.train_curriculum()
```

---

## Expected Results

### Baseline (RANDOM only)
- Overall accuracy: **~60%** (target)
- L0 accuracy: **~75%** (coarse)
- L1 accuracy: **~55%** (medium)
- L2 accuracy: **~45%** (fine)
- Training: Stable, fast convergence

### Curriculum Learning
- Overall accuracy: **~65%** (+5% over baseline)
- L0 accuracy: **~80%** (+5%, better coarse learning)
- L1 accuracy: **~60%** (+5%, better progression)
- L2 accuracy: **~50%** (+5%, better refinement)
- Training: Staged, longer but more interpretable

### Mixed Training
- Overall accuracy: **~63%** (+3% over baseline)
- More robust across different masking patterns
- Training: Slightly slower convergence, more stable
- Better generalization to unseen masking distributions

---

## FAQ

**Q: Can I combine curriculum and mixed?**
A: Yes! Use curriculum for stages, but within each stage use mixed masking:
```yaml
curriculum:
  stages:
    - name: "stage1"
      masking_type: "mixed"
      strategies: [...]  # Mix of COARSE_ONLY + HIERARCHICAL
      num_epochs: 10
```

**Q: What if curriculum causes catastrophic forgetting?**
A: Try:
1. Longer stages (more epochs per stage)
2. Add replay buffer (keep samples from previous stages)
3. Use mixed training within stages
4. Decrease LR between stages (fine-tuning)

**Q: Which strategy for production?**
A: Depends:
- **Known masking pattern**: Single strategy (fastest)
- **Variable masking**: Mixed training (most robust)
- **Research/interpretability**: Curriculum (most insights)

**Q: Can I start from a checkpoint?**
A: Yes, all approaches support `--resume`:
```bash
python train.py --config config.yaml --resume checkpoint.pt
```

---

## Summary

| Goal | Recommended Approach |
|------|---------------------|
| **Quick baseline** | Single (RANDOM) |
| **Best L0→L1→L2 emergence** | Curriculum |
| **Most robust model** | Mixed |
| **Production deployment** | Mixed or Single (depending on use case) |
| **Research/interpretability** | Curriculum |
| **Debugging** | Single (simplest) |

**Start simple, add complexity only if needed.**
