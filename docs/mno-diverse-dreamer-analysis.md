# MNO "Diverse Dreamer" — Token Diversity Training Mode

## Context

MNO is currently trained with `mse_led_param_sensitive` mode, which prioritizes
trajectory MSE (physics fidelity). But the downstream consumer of MNO trajectories
is the VQ tokenizer — what matters isn't exact match to QBM, but whether MNO
produces **diverse, tokenizable** trajectories that cover the VQ vocabulary well.

## Core Insight: VQ Codebook as Physical Ontology

The VQ codebook was trained on real QBM physics. Any trajectory that tokenizes
cleanly (roundtrip-consistent through VQ encode/decode) is *implicitly physically
grounded* — the codebook IS the physical ontology. We don't need high trajectory
accuracy if we have:

1. **Behavioral diversity** — different params produce different token sets (contrastive)
2. **VQ coherence** — MNO outputs live on the VQ manifold (commitment + reconstruction)
3. **Light physics regularization** — prevent total drift (low-weight traj + IC MSE)

### Why rel_l2 Matters Less Than Token Diversity

In the traditional `mse_led` paradigm, relative L2 error measures how close MNO
trajectories are to QBM ground truth. This is the right metric when the goal is
physics emulation.

But for the token synthesis pipeline, what matters is:
- Do different parameter configurations produce **distinguishable** token sequences?
- Do MNO trajectories lie on the VQ manifold (tokenize cleanly)?
- Is the token vocabulary **well-covered** (not collapsed to a few codes)?

A trajectory with rel_l2 = 0.3 that tokenizes to a unique, valid token set is
*more useful* than one with rel_l2 = 0.05 that maps to the same tokens as every
other trajectory.

## Physics-Space Counterfactual

The key experiment that motivates this mode:

1. Take two parameter vectors θ_a and θ_b
2. Run MNO: traj_a = MNO(ic, θ_a), traj_b = MNO(ic, θ_b)
3. Tokenize: tokens_a = VQ(features(traj_a)), tokens_b = VQ(features(traj_b))
4. Measure: Are tokens_a ≠ tokens_b? (Jaccard distance)

If the contrastive loss succeeds, different parameters will produce different
rollouts → different features → different token sets. This is the *token-level
parameter sensitivity* that the downstream D3PM needs.

## Contrastive + VQ Roundtrip as Sufficient Grounding

The claim: **contrastive diversity + VQ coherence is sufficient for downstream
token synthesis**, without exact trajectory matching.

### Why VQ Roundtrip Implies Physical Plausibility

The VQ codebook was trained on features extracted from *real QBM simulations*.
Each codebook entry corresponds to a cluster center in feature space that was
observed in actual physics. When MNO produces a trajectory that VQ-encodes cleanly:

- The features fall near a learned cluster center (commitment loss is low)
- The decoded features closely match the encoded ones (reconstruction loss is low)
- This means the trajectory *behaves like a real QBM simulation* in feature space

### Why Contrastive Loss Implies Parameter Sensitivity

InfoNCE contrastive loss ensures:
- rollout(ic, θ_i) is most similar to θ_i in embedding space (positive pair)
- rollout(ic, θ_i) is dissimilar to θ_j for j ≠ i (negative pairs)

This is exactly what the downstream D3PM needs: given a parameter vector, it must
predict a *specific* token sequence, not a generic one. The contrastive loss
directly trains for this discrimination.

## Training Speed Advantage

When `lambda_traj=0` and `lambda_ic=0`, the `needs_target_trajectory` property
returns `False`. The training loop skips the QBM replayer rollout entirely — the
dominant per-batch cost. Pure diversity mode trains significantly faster because:

- QBM replayer: ~200ms/batch (GPU simulation of quantum system)
- Contrastive + VQ forward: ~50ms/batch (pure neural net)
- Speedup: ~4x per batch

## Loss Function Design

```
Loss = λ_contrastive * L_contrastive    ← PRIMARY: behavioral diversity
     + λ_recon * L_recon                ← PRIMARY: VQ feature reconstruction
     + λ_commit * L_commit              ← PRIMARY: VQ manifold adherence
     + λ_traj * L_traj                  ← OPTIONAL: physics regularizer (0 = skip)
     + λ_ic * L_ic                      ← OPTIONAL: IC regularizer (0 = skip)
```

### Default Configuration (Pure Diversity Mode)

| Loss | Weight | Role |
|------|--------|------|
| L_contrastive | 1.0 | Different params → different trajectories |
| L_recon | 1.0 | VQ reconstruction quality |
| L_commit | 0.5 | VQ manifold adherence |
| L_traj | 0.0 | OFF — no physics MSE needed |
| L_ic | 0.0 | OFF — no IC preservation needed |

### Hybrid Configuration (Diversity + Light Physics)

| Loss | Weight | Role |
|------|--------|------|
| L_contrastive | 1.0 | Different params → different trajectories |
| L_recon | 0.5 | VQ reconstruction quality |
| L_commit | 0.3 | VQ manifold adherence |
| L_traj | 0.1 | Light physics regularizer |
| L_ic | 0.1 | Light IC preservation |

## Relationship to Other Loss Modes

| Aspect | MSE-led | VQ-led | Token Diversity |
|--------|---------|--------|-----------------|
| Primary objective | Physics fidelity | VQ coherence | Behavioral diversity |
| Contrastive | None | None | InfoNCE (primary) |
| L_traj weight | 1.0 (primary) | 0.3 (regularizer) | 0.0 (off by default) |
| QBM replayer needed | Always | Always | No (when traj=0, ic=0) |
| Token metrics | None | None | Set diversity, accuracy |
| Parameter sensitivity | Implicit | None | Explicit (contrastive) |

## Expected Outcomes

After training with token diversity mode:
1. Contrastive accuracy > 0.8 (params distinguish rollouts)
2. Token set diversity > 0.5 (batch produces varied tokens)
3. VQ reconstruction loss converging (trajectories stay on manifold)
4. rel_l2 may be higher than MSE-led (acceptable — not the objective)

## Future Directions

- **Curriculum**: Start with MSE-led for physics grounding, then switch to token
  diversity for vocabulary coverage
- **Adaptive weighting**: Increase lambda_contrastive as contrastive accuracy
  plateaus, shift weight to VQ losses
- **Token coverage metric**: Track fraction of VQ vocabulary used across training
  set, not just per-batch diversity
