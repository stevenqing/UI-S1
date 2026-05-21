# Cooperative LoRA for GUI Agents: Research Roadmap

## Problem
Train a VLM (Qwen2.5-VL-7B) for multi-step mobile GUI navigation.
Base model achieves 15.88% Task Success Rate (TSR) on AndroidControl.
Goal: learn **two cooperative skills** (perception + action planning) via LoRA.

--

## V10: Two-Pass Sequential Architecture

**Idea**: Separate Grounder (LoRA\_G) and Actor (LoRA\_A), communicate via natural language.

```
Screenshot → [LoRA_G] → "<target>Blue submit button</target>"
                              ↓
Screenshot + Description → [LoRA_A] → {"action":"click","coordinate":[920,1800]}
```

**Training**: Single-step GRPO, K=8 rollouts, dual rewards (grounder + actor).

**Result**: 6.55% TSR (worse than base 15.88%)

**Failure mode**: Action type collapse — grounder always describes UI elements
→ actor always predicts `click` (83% vs GT 60%). Rare actions (`open`: 3.3%, `system_button`: 4.1%) nearly zero.

**Lesson**: Explicit text communication contaminates action type distribution. Sequential two-pass doubles exploration space → cold-start failure on rare actions.

---

## V11: Soft-Routed Cooperative LoRA + Trajectory RL

**Idea**: Replace two-pass with **soft dynamic routing** — single forward pass, two generic LoRA experts blended per-token.

```
r(x) = sigmoid(w_route · x)    (learned, per-layer)
delta = r · A₁B₁(x) + (1-r) · A₂B₂(x)
```

**Key innovations**:
- No explicit V/A labels — cooperation patterns emerge from RL
- Trajectory-level reward: **SP** (Sequential Progress) = first\_correct\_steps / total\_steps
- **GiGPO** cross-rollout advantage normalization
- **SPWA** step weighting (decay after first error)

**Training**: SFT warmup → Trajectory RL (SP+GiGPO+SPWA)

**Status**: Framework validated, episode data prepared. Served as architectural foundation for V12.

---

## V12: Soft Cooperative LoRA — Direct RL from Raw Model

**Idea**: Skip SFT (which failed due to B=0 init → zero gradients to A), run trajectory RL directly from base Qwen2.5-VL-7B.

**Architecture**: Same soft routing as V11, but simplified:
- `lora_r=128`, target: q/k/v/o projections, 132M trainable params
- B=0 init → delta=0 → starts as exact base model (15.88% TSR policy)
- RL exploration grows B from step 1 (gradients don't depend on B)

**Key improvements over V11**:
1. **Batched K=8 generation** (single `model.generate()` call, ~5x faster)
2. **Hybrid advantage** = 0.5 × trajectory\_adv + 0.5 × step\_adv
   - Trajectory: SP+GiGPO+SPWA (rewards sequential correctness)
   - Step-level: per-step cross-K normalized reward (rewards action precision)
   - Solves nz=0% problem: even when SP identical across K, step rewards differ

**Step-level reward** (dense, continuous):
```
R = 0.1 × format + 0.2 × type_match + 0.7 × content
    (parseable?)    (click vs swipe?)    (coord distance / text similarity)
```

**Training progress** (Epoch 0, ongoing):

| Step | Reward | SP | KL | nz% | routing\_w |
|------|--------|------|------|------|-----------|
| S1   | 0.345  | 0.000 | 0.000 | 77% | 0.500 |
| S7   | 0.350  | 0.052 | 0.015 | 75% | 0.488 |
| S13  | 0.366  | 0.001 | 0.009 | 81% | 0.477 |

- **routing\_w diverging** from 0.5 → experts differentiating
- **nz consistently >75%** (was 0% without step-level advantage)
- KL growing → policy exploring away from base model
- ~4 min/step, 29 steps/epoch, ~2h/epoch, 24h budget fits ~6 epochs

---

## Evolution Summary

| | V10 | V11 | V12 |
|---|---|---|---|
| **Architecture** | Two-pass (G→A) | Soft routing, single pass | Soft routing, single pass |
| **Communication** | Text (explicit) | Routing weights (implicit) | Routing weights (implicit) |
| **Training** | Step GRPO | Traj RL (SP+GiGPO) | Traj RL (SP+GiGPO+SPWA) + **step adv** |
| **Start point** | SFT checkpoint | SFT → RL | **Raw base model** |
| **Generation** | Batch K=8 | Serial K=8 | **Batch K=8** |
| **Advantage** | Step-only | Traj-only | **Traj + Step hybrid** |
| **Rare actions** | Collapsed (open 3%) | — | Preserved (base model) |

## Key Insight

> **Direct RL from a working base model** avoids both SFT's dead-gradient problem
> and two-pass exploration failure. Hybrid traj+step advantage ensures every
> training step has gradient signal, while trajectory reward shapes long-horizon behavior.
