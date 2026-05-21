# V12: Cooperative RL for Long-Horizon GUI Agents

## Motivation

### Lessons from Previous Versions

| Version | Architecture | Training | TSR | Key Insight |
|---------|-------------|----------|-----|-------------|
| V6.5 | Cooperative LoRA (V+A, hard routing, h-space comm) | SFT | ~30%+ | Dual experts + communication works, but SFT has ceiling |
| V8 | Single model | SP+GiGPO trajectory RL | 15.4% | Trajectory RL from raw base works; SP reward + GiGPO > step-level GRPO |
| V10 | Grounder+Actor (text comm, two-stage gen) | Step-level GRPO | 6.5% | Two-stage generation kills exploration; can't discover open/system_button |
| V10.1 | Same + grounder format reward | Step-level GRPO | 6.5% | Format reward doesn't improve accuracy |
| V11 | Cooperative LoRA + learned router | Trajectory GRPO | TBD | Cooperative + RL, but standard GRPO, not SP+GiGPO |

### Core Observations

1. **RL exploration failure (V10)**: GRPO from raw base model can never discover `open` (0% acc) and `system_button` (1.2% acc) — cold-start problem. V8 also starts from raw base but achieves 15.4% TSR because it uses single-model output (no two-stage exploration bottleneck).

2. **SFT provides action vocabulary, RL provides optimization**: UI-S1's original RL pipeline starts from SFT checkpoint and succeeds. V10 skips SFT and fails on rare actions. V6.5's SFT knows all actions but can't go beyond imitation.

3. **SP+GiGPO >> step-level GRPO**: V8's trajectory-level SP reward (`first_error/total_steps`) + cross-trajectory GiGPO comparison gives much denser and more effective learning signal than V10's binary step reward.

4. **Architecture matters**: V10's text-based two-stage communication is non-differentiable and doubles the exploration space. V6.5's h-space communication is differentiable but static. Neither is optimal.

---

## Design: Soft Cooperative LoRA + Trajectory RL

### Key Idea

Two generic LoRA "skills" with **soft, dynamic routing** per token per layer, where cooperation emerges through the **attention mechanism** and routing patterns are optimized by **trajectory-level RL** (SP+GiGPO).

### Architecture: Soft-Routed Cooperative LoRA

**Drop the hard V/T routing and side-channel communication.** Replace with:

```python
# At layer l, for each token i with hidden state x:

# Dynamic routing: how much of each skill for this token?
r = sigmoid(w_route[l] @ x)          # scalar in [0,1], input-dependent

# Two generic LoRA skills (NOT labeled vision/text)
delta_1 = B_1[l] @ A_1[l] @ x        # skill 1's modification
delta_2 = B_2[l] @ A_2[l] @ x        # skill 2's modification

# Soft blend
delta = r * delta_1 + (1-r) * delta_2

# Applied to all projections: Q, K, V, O, gate, up, down
x_new = W @ x + delta
```

**Why this works:**
- **No hard V/T split**: Each token dynamically decides its skill blend. A text token describing "blue button on top right" can use more spatial-skill; a vision token showing text on screen can use more language-skill.
- **Cooperation through attention**: Skills modify Q, K, V projections → different routing patterns produce different attention patterns → cooperation emerges from how differently-routed tokens interact through attention.
- **Trivial inference**: Just `sigmoid` + weighted LoRA. Same complexity as standard LoRA.

### Why NOT Standard MoE

| | Standard MoE | Soft Cooperative LoRA |
|--|-------------|----------------------|
| **Where** | FFN layers only | All attention projections (Q,K,V,O) + FFN |
| **What it controls** | What information each token contains | How tokens communicate (attention patterns) |
| **Expert coupling** | Independent: removing one expert doesn't affect the other | Coupled through attention: changing one skill's routing changes all tokens' attention patterns |
| **Temporal** | Stateless per-step routing | Routing depends on hidden state encoding trajectory history |
| **Optimization** | Task loss + load balance | Trajectory-level RL (SP+GiGPO) optimizes cooperation across steps |

**Core distinction**: MoE = independent experts + weighted mixture. Ours = coupled experts through attention dynamics + trajectory-level cooperation.

### Two Levels of Dynamic Cooperation

The cooperation pattern is a **step x layer** dynamic tensor:

1. **Layer dimension** (within one forward pass): Different layers have different routing patterns. Lower layers may route more to spatial-skill for visual grounding; upper layers may route more to planning-skill for action decisions.

2. **Step dimension** (across trajectory): At step 1 (open app), routing pattern differs from step 5 (type in field) or step 10 (verify completion). The hidden state encodes trajectory history → routing naturally adapts.

Both dimensions emerge automatically from input-dependent routing — no external controller needed.

### RL Algorithm: SP+GiGPO with Routing-Based Structured Exploration

**Main objective**: SP+GiGPO trajectory reward (from V8)
- SP reward: `first_error_step / total_steps` — dense trajectory-level signal
- GiGPO: cross-trajectory comparison at each step — better advantage estimation
- SPWA: step-level advantage weighted by marginal contribution to SP

**Routing regularization** (minimal):
```
L = L_SP+GiGPO - lambda * H(r_bar)
```
Maximize entropy of average routing to prevent skill collapse. No other auxiliary losses.

**Structured exploration via routing perturbation**:
- For K rollouts in GiGPO, sample K different routing configurations (add noise/perturbation to routing weights)
- Each rollout has a different cooperation strategy
- GiGPO cross-trajectory comparison directly evaluates: "which cooperation pattern works better at this step?"
- This provides **structured diversity** along the cooperation dimension, more efficient than random temperature sampling

### Training Pipeline

1. **Phase 1: SFT warm-start**
   - Train soft cooperative LoRA with SFT on AC/GUI-360 data
   - Let initial skill specialization emerge (no forced V/T assignment)
   - Solves cold-start: model learns all action types (open, system_button, etc.)

2. **Phase 2: Trajectory RL**
   - Fine-tune with SP+GiGPO on trajectory data
   - RL optimizes both skill weights AND routing patterns for trajectory completion
   - Routing-based structured exploration for efficient learning
   - Cooperation patterns evolve: "SFT teaches skills WHAT to do, RL teaches them HOW to cooperate"

### Analysis Plan

The architecture produces rich analyzable outputs:

1. **Routing heatmap** (step x layer): Visualize how cooperation evolves across trajectory steps and model depth
2. **Per-action-type routing**: Do different actions (click, swipe, open, type) activate different routing patterns?
3. **SFT vs RL routing comparison**: Does RL discover fundamentally different cooperation strategies than SFT?
4. **Skill specialization analysis**: What does each skill learn to do? (via activation analysis or probing)
5. **Ablation**: Soft routing vs hard routing vs single LoRA — quantify the benefit of dynamic cooperation

---

## Implementation Notes

### Compatibility with vLLM
- Soft routing is just `sigmoid(w @ x) * LoRA_1(x) + (1-sigmoid(w @ x)) * LoRA_2(x)`
- Can be pre-merged for vLLM inference: `delta = B_merged(r) @ A_merged(r) @ x`
- Or implement as custom vLLM LoRA layer (lightweight modification)

### Parameter Count
- Two LoRA adapters: 2x standard LoRA params
- Routing weights: 1 vector of dim `hidden_size` per layer per module = negligible
- No communication matrices (W_av, W_va) — removed, cooperation through attention
- Overall: ~2x LoRA params, much less than v6.5 (which has LoRA + comm matrices + gates)

### Key Hyperparameters
- `lora_r`: rank per skill (e.g., 128 or 256)
- `lambda_balance`: routing entropy regularization weight
- `routing_noise_std`: perturbation scale for structured exploration during RL
- SP+GiGPO params: `gamma`, `spwa_decay`, `K` (rollouts)
