# Multi-Agent Token Communication: VQ-Led Systems for Collaborative Discovery

**Status:** Future Research Direction
**Date:** January 2026
**Prerequisites:** VQ-VAE tokenization, VQ-led NOA training

---

## Executive Summary

The VQ-led training paradigm enables a critical capability that MSE-led models cannot provide: **discrete symbolic communication between agents**. By operating over shared VQ-VAE token vocabularies, multiple NOA instances can engage in compositional reasoning, emergent communication protocols, and collaborative parameter space exploration—grounded in behavioral semantics rather than arbitrary symbols.

This document outlines the architecture for multi-agent systems that use VQ-led models for symbolic reasoning while leveraging MSE-led models for precise physics execution.

---

## The Communication Problem

### Why Continuous Representations Fail for Multi-Agent Systems

| Continuous (MSE-led) | Discrete (VQ-led) |
|---------------------|------------------|
| Agent 1: "Here's a 256×64×64 trajectory" | Agent 1: "Token sequence: [7, 12, 3, 7]" |
| Agent 2: "How do I reason about 1M floats?" | Agent 2: "Ah, category 7 → 12 transition" |
| **No shared symbolic basis** | **Shared categorical vocabulary** |
| Communication requires full trajectory transfer | Communication via compact token sequences |
| No compositional structure | Natural temporal compositionality |

**Key insight:** Discrete tokens provide the symbolic substrate necessary for inter-agent communication, while continuous trajectories do not.

---

## Dual-Model Architecture: Division of Labor

The optimal architecture uses **both** VQ-led and MSE-led models in complementary roles:

```
VQ-led (Symbolic Reasoner)          MSE-led (Physics Executor)
         ↓                                    ↓
   "What kind?"                         "Exactly what?"
   Fast, categorical                    Precise, continuous
   ~0.1 VQ resolution                   ~0.4 MSE accuracy
   Discrete tokens                      Full trajectories
   Communication-ready                  Execution-ready
```

### Task Allocation

| Task | Model | Rationale |
|------|-------|-----------|
| **Symbolic reasoning** | VQ-led | Operates in discrete token space, compositional |
| **Fast screening** | VQ-led | Categorical classification without full rollout |
| **Multi-agent communication** | VQ-led | Shared vocabulary of behavioral tokens |
| **Planning & search** | VQ-led | Fast symbolic exploration over categories |
| **Precise execution** | MSE-led | Physics-accurate rollouts for downstream use |
| **Verification** | MSE-led | Ground truth when exact values needed |
| **Engineering constraints** | MSE-led | Quantitative predictions (amplitude, period, etc.) |

---

## System 1 / System 2 Analogy

The dual-model architecture mirrors dual-process theories of cognition:

| | VQ-led (System 1) | MSE-led (System 2) |
|---|-------------------|-------------------|
| **Speed** | Fast (symbolic) | Slow (analytical) |
| **Output** | Categorical ("stable oscillation") | Quantitative (MSE=0.41) |
| **Cost** | Cheap (discrete reasoning) | Expensive (full rollout) |
| **Communication** | ✅ Token sequences | ❌ Raw trajectories |
| **Compositionality** | ✅ Natural (temporal sequences) | ❌ Continuous manifold |
| **Use case** | Exploration, filtering, reasoning | Verification, precise prediction |

---

## Compositional Token Semantics

### Token Embeddings Enable Grounded Communication

VQ-VAE tokens are not arbitrary symbols—they are **grounded in behavioral categories** discovered from data.

```python
# VQ-VAE produces token embeddings
token_7_embedding = vqvae.codebook[category=7, level=0]  # [256-dim vector]

# Agents can reason about token relationships:
"Token 7 is similar to token 12" (cosine similarity)
"Token sequence [7, 12, 3] describes oscillation → damping → stable"
"If I see [7, 12, ...], expect category 3 next"
```

### Hierarchical Multi-Resolution Discourse

The **3-level VQ-VAE hierarchy** enables progressive refinement through dialogue:

```
Agent A: "Coarse category: 7 (L0 token)"
Agent B: "Understood. Refinement?"
Agent A: "Medium: 12 (L1 token)"
Agent B: "Got it. Exact variant?"
Agent A: "Fine: 3 (L2 token)"
Agent B: "Ah - damped oscillation, 0.5Hz, amplitude 2.3"
```

**Progressive refinement through multi-turn dialogue**—impossible with continuous representations.

---

## Multi-Agent Communication Protocols

### Example 1: Collaborative Parameter Search

```python
# Phase 1: Fast symbolic screening (VQ-led)
for theta in candidate_parameters:
    token_seq = vq_led_noa(theta, u0)
    category = vqvae.classify(token_seq)

    if category in desired_behaviors:
        promising_params.append(theta)

# Phase 2: Precise evaluation (MSE-led)
for theta in promising_params:
    exact_trajectory = mse_led_noa(theta, u0)
    compute_exact_metrics(exact_trajectory)
```

**Agent dialogue:**
```
Agent A: "I'm exploring region R1. Seeing mostly tokens [2, 5, 8]."
Agent B: "That's the stable fixed-point region. Try R2 for oscillations."
Agent A: "Found token sequence [7, 12, 3] in R2!"
Agent B: "Perfect. That's the target. Send me θ for MSE-led verification."
```

### Example 2: Compositional Reasoning

```
Agent A: "If I combine operator θ₁ (tokens [7, 12]) with θ₂ (tokens [3, 5])..."
Agent B: "...you get token sequence [7, 12, 3, 5]?"
Agent A: "No! Nonlinear composition → [9, 1, 4]"
Agent B: "Emergent category 9. Fascinating. That's a new behavioral regime."
```

### Example 3: Emergent Shorthand Protocols

```
# Initial communication (verbose)
Agent A: "Observed token sequence [7, 12, 3, 7, 12, 3, 7, 12, 3]"
Agent B: "Periodic oscillation, category 7-12 limit cycle"

# After repeated interactions (emergent shorthand)
Agent A: "Type-7 cycle"
Agent B: "Confirmed. Period 3. Checking for bifurcation."
```

**Protocol emergence:** Agents develop efficient communication conventions grounded in the VQ vocabulary structure, similar to emergent communication in Lewis signaling games.

---

## Connection to Language Games

### Traditional Grounded Language Games

| Traditional Approach | VQ-Led NOA Approach |
|---------------------|---------------------|
| Arbitrary symbols (no intrinsic semantics) | **Grounded tokens** (behavioral categories) |
| Meaning emerges from scratch | **Bootstrap from VQ-VAE** (pre-learned semantics) |
| Compositional pressure from task | **Natural compositionality** (temporal sequences) |
| Hard to evaluate meaning | **Interpretable** (tokens map to behaviors) |
| Sender/receiver architecture | Multi-agent token exchange |

### Lewis Signaling Game with VQ-Led NOAs

**Classic setup:** Sender observes state → sends discrete symbol → Receiver acts

**VQ-led variant:**
```
Agent A observes (θ, u₀) → VQ-led encodes to tokens → Agent B reasons about behavior
                                     ↓
                        Tokens are grounded in VQ-VAE categories
                        (not arbitrary - semantically meaningful)
```

**Key advantage:** Communication bootstraps from pre-learned behavioral vocabulary rather than emerging from random initialization.

---

## Implementation Architecture

### Hybrid NOA System

```python
class HybridNOA:
    """Dual-model system: VQ-led for reasoning, MSE-led for execution."""

    def __init__(self, vq_led_checkpoint, mse_led_checkpoint, vqvae):
        self.reasoner = load_noa(vq_led_checkpoint)  # VQ-led
        self.executor = load_noa(mse_led_checkpoint)  # MSE-led
        self.vqvae = vqvae  # Shared vocabulary

    def reason(self, theta, u0) -> List[int]:
        """Fast symbolic classification via VQ tokens."""
        trajectory = self.reasoner(theta, u0)
        features = extract_features(trajectory)
        tokens = self.vqvae.encode(features)
        return tokens

    def execute(self, theta, u0) -> torch.Tensor:
        """Precise physics simulation."""
        return self.executor(theta, u0)

    def get_token_embeddings(self, tokens: List[int]) -> torch.Tensor:
        """Get continuous embeddings for discrete tokens."""
        return self.vqvae.get_embeddings(tokens)

    def communicate(self, tokens: List[int]) -> str:
        """Convert tokens to human-interpretable message."""
        categories = self.vqvae.decode_categories(tokens)
        return f"Behavior: {categories}"
```

### Multi-Agent Communication Layer

```python
class MultiAgentSystem:
    """Manages communication between multiple VQ-led NOA agents."""

    def __init__(self, num_agents: int, vqvae):
        self.agents = [HybridNOA(...) for _ in range(num_agents)]
        self.vqvae = vqvae  # Shared vocabulary
        self.message_history = []

    def send_message(self, sender_id: int, receiver_id: int, tokens: List[int]):
        """Send discrete token message between agents."""
        message = {
            'sender': sender_id,
            'receiver': receiver_id,
            'tokens': tokens,
            'embeddings': self.agents[sender_id].get_token_embeddings(tokens),
            'timestamp': time.time(),
        }
        self.message_history.append(message)
        return message

    def collaborative_search(self, param_space, target_category: int):
        """Agents collaborate via token exchange to find target behavior."""

        # Divide search space among agents
        regions = partition_space(param_space, len(self.agents))

        # Phase 1: Parallel symbolic screening
        results = []
        for agent_id, region in enumerate(regions):
            for theta in region:
                tokens = self.agents[agent_id].reason(theta, u0)

                # Communicate findings
                if matches_target(tokens, target_category):
                    self.send_message(
                        sender_id=agent_id,
                        receiver_id='all',
                        tokens=tokens
                    )
                    results.append(theta)

        # Phase 2: MSE-led verification (any agent can execute)
        verified = [self.agents[0].execute(theta, u0) for theta in results]

        return verified
```

---

## The Killer Feature: Continuous Semantics + Discrete Communication

**VQ token embeddings are continuous vectors** → agents can:

1. **Compute similarity** - `cosine_similarity(token_7_emb, token_12_emb)`
2. **Interpolate** - "behavior between category 7 and 12"
3. **Cluster** - "these 5 token sequences are related"
4. **Learn transformations** - "adding noise shifts embeddings this way"
5. **Attend over tokens** - Multi-head attention over token sequences

**All while maintaining discrete symbolic communication.**

This is the bridge between:
- **Symbolic reasoning** (discrete tokens for communication)
- **Continuous semantics** (embeddings for similarity/learning)

### Example: Token Similarity Reasoning

```python
# Agent A observes token sequence from exploration
tokens_a = [7, 12, 3]
embeddings_a = vqvae.get_embeddings(tokens_a)

# Agent B has seen similar sequence before
tokens_b = [7, 12, 5]
embeddings_b = vqvae.get_embeddings(tokens_b)

# Compute semantic similarity
similarity = cosine_similarity(embeddings_a, embeddings_b)
# High similarity → Agent B can transfer knowledge

if similarity > 0.9:
    agent_b_says("I've seen something similar. Token 3 and 5 differ only in fine detail.")
    agent_b_says("Expect damped oscillation, slight amplitude difference.")
```

---

## Research Directions

### 1. Emergent Communication Protocols

**Question:** What compositional languages emerge when agents optimize for collaborative discovery?

**Setup:**
- Multiple VQ-led agents explore parameter space
- Reward for efficiently communicating discoveries
- Token sequences can be arbitrary (not just VQ-VAE outputs)

**Hypothesis:** Agents develop compositional protocols grounded in VQ vocabulary structure (e.g., "token 7 at position 1 means oscillatory, position 2 means amplitude").

### 2. Meta-Learning Communication Efficiency

**Question:** Can agents learn to compress token sequences into higher-level abstractions?

**Example:**
```
Initial: [7, 12, 3, 7, 12, 3, 7, 12, 3]  (9 tokens)
Learned: [CYCLE(7, 12, 3), period=3]     (meta-token + parameter)
```

### 3. Cross-Domain Transfer via Tokens

**Question:** Do VQ tokens learned on one operator family transfer to others?

**Setup:**
- Train VQ-VAE on reaction-diffusion operators
- Test token communication on fluid dynamics
- Measure semantic preservation

**Hypothesis:** Abstract behavioral categories (stability, oscillation, chaos) transfer across domains.

### 4. Hierarchical Multi-Agent Planning

**Question:** Can agents plan collaboratively using hierarchical token sequences?

**Setup:**
- Coarse planning at L0 tokens (categorical regions)
- Medium refinement at L1 tokens (sub-regions)
- Fine execution at L2 tokens (exact parameters)

**Hypothesis:** Hierarchical tokens enable efficient multi-resolution planning.

### 5. Token-Based Theory of Mind

**Question:** Can Agent A predict Agent B's future tokens from past communication?

**Setup:**
- Agent B explores parameter space, sends token observations
- Agent A builds model of Agent B's token distribution
- Test: Does Agent A accurately predict Agent B's next message?

**Application:** Anticipatory collaboration, resource allocation.

---

## Comparison to Existing Approaches

| Approach | Communication Medium | Grounding | Compositionality |
|----------|---------------------|-----------|------------------|
| **Traditional Multi-Agent RL** | Continuous observations | Task-specific | Limited |
| **Emergent Communication (MARL)** | Learned discrete symbols | Arbitrary | Emergent |
| **Graph Neural Networks** | Node/edge features | Structural | Fixed |
| **VQ-Led Multi-Agent (This Work)** | **VQ-VAE tokens** | **Behavioral categories** | **Natural (temporal)** |

**Key distinction:** VQ tokens are grounded in pre-learned behavioral semantics, not arbitrary or task-specific.

---

## Prerequisites for Implementation

### Technical Requirements

1. **Trained VQ-VAE** - Production tokenizer (e.g., `100k_3family_v1`)
2. **VQ-led NOA checkpoint** - Trained with VQ reconstruction as primary loss
3. **MSE-led NOA checkpoint** - Trained with trajectory MSE as primary loss
4. **Shared dataset** - Same operator families for both models

### Minimal Viable Prototype

**Two-agent collaborative search:**
```python
# Agent A: Explorer (VQ-led)
for theta in search_space_A:
    tokens = agent_a.reason(theta, u0)
    if is_interesting(tokens):
        send_to_agent_b(tokens, theta)

# Agent B: Verifier (MSE-led)
for (tokens, theta) in received_messages:
    if confirm_interest(tokens):
        trajectory = agent_b.execute(theta, u0)
        evaluate_trajectory(trajectory)
```

**Success metric:** Agent B spends less time on uninteresting parameters thanks to Agent A's symbolic filtering.

---

## Limitations and Open Questions

### Current Limitations

1. **VQ-VAE vocabulary size** - Limited to ~370 active codes (53% utilization)
   - May need larger codebooks for richer communication

2. **Token sequence length** - Current rollouts → single token sequence
   - Multi-turn communication requires temporal segmentation

3. **Cross-domain transfer** - Tokens learned on specific operator families
   - Generalization to unseen domains untested

### Open Questions

1. **Optimal token granularity?** - How many categories needed for effective communication?

2. **Syntax emergence?** - Do agents develop grammatical structure over token sequences?

3. **Theory of mind?** - Can Agent A model Agent B's token generation process?

4. **Scalability?** - Does communication efficiency scale to >2 agents?

5. **Adversarial robustness?** - Can agents detect/correct miscommunication?

---

## Conclusion

VQ-led NOA training enables **multi-agent systems with grounded symbolic communication**. By operating over shared VQ-VAE token vocabularies, agents can:

- **Communicate efficiently** via discrete token sequences
- **Reason compositionally** over temporal behavioral patterns
- **Collaborate** on parameter space exploration and discovery
- **Develop emergent protocols** grounded in behavioral semantics

The dual-model architecture (VQ-led for reasoning, MSE-led for execution) provides the optimal division of labor: symbolic communication for exploration, precise physics for verification.

**For multi-agent language games and collaborative discovery, VQ-led is not just useful—it's the right architecture.**

---

## References

- VQ-VAE Production Baseline: [docs/baselines/100k-full-features-vqvae.md](../baselines/100k-full-features-vqvae.md)
- NOA Architecture: [docs/noa-architecture.md](../noa-architecture.md)
- Lewis Signaling Games: Classic emergent communication framework
- VQ-led Training: Prioritizes symbolic coherence over physics fidelity

---

**Status:** Conceptual framework ready for implementation
**MVP Idea:** Implement two-agent prototype, measure communication efficiency gains
**Timeline:** Research direction for 2026+
