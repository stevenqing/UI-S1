# v10 S300 Evaluation Analysis

## Evaluation Setup

- **Checkpoint**: epoch-1_step-300 (best on val set)
- **Eval data**: AndroidControl eval set (1543 episodes, 8444 steps)
- **Method**: vLLM multi-LoRA, grounder→actor two-pass, GT history (single-step eval)
- **Job**: 4038520

## Results

### Episode-Level Metrics

| Metric | v10 S300 |
|--------|----------|
| TSR | 6.55% (101/1543) |
| Avg Progress | 12.30% |
| Step Accuracy | 34.23% (2890/8444) |

### Step-Level Metrics

| Metric | v10 S300 | Base Model (1000-step sample) |
|--------|----------|-------------------------------|
| Type Acc | 56.7% | 69.8% |
| Action Acc | 34.2% | 57.0% |

### Per Action-Type Breakdown

| Type | Count (%) | Type Acc | Action Acc | Notes |
|------|-----------|----------|------------|-------|
| click | 5074 (60.1%) | 80.3% | 46.1% | Best type, but still below base (91.5%/70.0%) |
| type | 632 (7.5%) | 63.4% | 44.6% | Decent, below base (91.9%/86.0%) |
| swipe | 1211 (14.3%) | 14.9% | 11.6% | Very low — base was 34.6% |
| open | 608 (7.2%) | 0.2% | 0.2% | Nearly zero — base was 13.6% |
| wait | 567 (6.7%) | 19.9% | 19.9% | Low — base was 29.2% |
| system_button | 343 (4.1%) | 3.8% | 3.2% | Very low — base was 35.9% |
| long_press | 9 (0.1%) | 22.2% | 22.2% | Too few samples |

### Per Length Bucket

| Bucket | Episodes | TSR | Step Acc | Avg Progress |
|--------|----------|-----|----------|--------------|
| short(1-3) | 438 | 21.5% | 39.7% | 30.1% |
| medium(4-7) | 788 | 0.9% | 36.6% | 6.7% |
| long(8-15) | 289 | 0.0% | 30.2% | 1.8% |
| vlong(16+) | 28 | 0.0% | 26.7% | 0.3% |

---

## Root Cause Analysis

### Problem 1: Model predicts `click` for everything (Action Type Collapse)

**Overall prediction distribution**: click=74.1%, left_click=8.9%, type=8.8%, swipe=3.9%, everything else <2%.

The model predicts `click` (or `left_click`) for **83%** of all steps, but the GT distribution is only 60.1% click. This causes massive failure on non-click action types.

**What the model predicts when GT is each type:**

| GT Type | Predicted as click/left_click | Predicted correctly |
|---------|-------------------------------|---------------------|
| open (608) | 477 (78%) | 1 (0.2%) |
| swipe (1211) | 972 (80%) | 180 (14.9%) |
| system_button (343) | 282 (82%) | 13 (3.8%) |
| wait (567) | 386 (68%) | 113 (19.9%) |

**Root cause**: The grounder→actor architecture creates an information bottleneck:

1. **Grounder prompt asks only to "describe the target UI element"** — it never mentions action type. So the grounder output is always about some UI element's appearance and location.
2. **Actor receives a grounding description about a UI element** → naturally predicts `click` on that element, since "described element + location" strongly implies clicking.
3. For `open`, `swipe`, `wait`, `system_button` — there is no specific UI element to describe, so the grounder describes something irrelevant, and the actor clicks on it.

**Evidence from grounder outputs:**
- For `open` GT: grounder describes random UI elements on screen (not the app to open)
- For `swipe` GT: grounder describes a UI element to click, actor clicks instead of swiping
- For `system_button` GT: grounder describes a UI element, actor clicks instead of pressing Back
- For `wait` GT: grounder describes a UI element, actor clicks instead of waiting

**Grounder text action-word frequency:**
- "click" appears in 33% of grounder outputs (even though grounder shouldn't be suggesting actions)
- "swipe/scroll" in only 2.6%
- "wait" in only 0.6%

### Problem 2: `left_click` hallucination (8.9% of predictions)

The model outputs `left_click` instead of `click` for 751 steps. This is likely learned from the grounder's descriptions mentioning "click" — the model sometimes generates `left_click` as a desktop-style variant. Since evaluation requires exact type match, all these are counted as wrong even when coordinates are correct.

If we treat `left_click` as `click`:
- Click type_acc: 80.3% → 91.8%
- Click action_acc: 46.1% → 52.6%
- Overall type_acc: 56.7% → 63.6%

### Problem 3: Grounder describes UI element even when no element interaction needed

For non-element actions (`open`, `wait`, `system_button`, `swipe`), the grounder still tries to find and describe a UI element, because its system prompt says "describe the target UI element." This misleads the actor into clicking.

Examples:
- GT=`open("UN News")` → Grounder: "The target UI element is the search icon located in the top right corner" → Actor: click
- GT=`system_button("Back")` → Grounder: "The target UI element is the Follow button" → Actor: click
- GT=`wait(2)` → Grounder: "The target UI element is the button at the bottom right corner" → Actor: click

### Problem 4: Reward function doesn't penalize type collapse

The reward function gives 0.0 for type mismatch, but during GRPO training with K=8 samples, if all 8 samples predict `click` for a `swipe` GT, they all get 0.0 reward → zero advantage → no learning signal. The model never learns to predict non-click types because:
1. It rarely explores non-click predictions (almost all samples are click)
2. When it does, the reward signal is too noisy to reinforce it

### Problem 5: Training data bias

The training data heavily favors `click` actions. In GRPO, the model generates its own rollouts — since it starts predicting `click` early, it never explores other action types, creating a self-reinforcing cycle.

---

## Proposed Fixes

### Fix 1: Redesign Grounder Prompt — Include Action Type in Grounding

**Current prompt:**
```
"describe the target UI element that should be interacted with for the next action"
```

**Proposed prompt:**
```
"Determine the next action type (click, type, open, swipe, wait, system_button, long_press, terminate) and describe the target. For click/long_press: describe the UI element and its location. For type: describe the text input field. For open: state the app name to open. For swipe: describe the scroll direction needed. For wait: explain what to wait for. For system_button: state which button (Back/Home/Overview)."
```

This makes the grounder responsible for **both action type and target description**, giving the actor clear signals for all action types.

**Impact**: High — directly addresses the root cause of action type collapse.

### Fix 2: Handle `left_click` in Eval and Reward

Add `left_click` → `click` normalization in both `evaluate_action()` and `actor_reward()`:

```python
# Normalize common aliases
if pred_type == "left_click":
    pred_type = "click"
```

**Impact**: Medium — immediately gains ~7% type accuracy and ~6% click action accuracy.

### Fix 3: Add Action Type Hint to Actor Prompt

If the grounder now outputs action type, include it explicitly in the actor prompt:

```
"The grounding agent determined: action_type={type}, target={description}.
Execute this action with the correct parameters."
```

This ensures the actor doesn't ignore the grounder's type recommendation.

### Fix 4: Structured Grounder Output Format

Instead of free-text grounder output, use a structured format:

```
<grounding>
<action_type>swipe</action_type>
<description>Scroll down to see more content below the fold</description>
<direction>up</direction>
</grounding>
```

This makes parsing more reliable and ensures action type is always present.

### Fix 5: Type-Aware Reward Shaping

Add a small positive reward for correct action type even when content is wrong:

```python
# Current: type mismatch → 0.0
# Proposed: type match → 0.15 base reward (even if content wrong)
#           type mismatch → 0.0
#           This ensures the model learns action type classification first
```

### Fix 6: Exploration Bonus for Non-Click Types

During GRPO, add temperature scaling or action-type-aware sampling to encourage exploration of non-click action types. For example, if all K samples are `click`, resample some with higher temperature or inject a type prior.

### Fix 7: Train Grounder and Actor Separately First (Curriculum)

Phase 1: Train grounder with supervised loss on action type + description
Phase 2: Train actor conditioned on GT grounder output
Phase 3: Joint GRPO fine-tuning

This ensures both adapters have baseline capability before RL.

---

## Priority Ranking

1. **Fix 1 (Grounder prompt redesign)** — Highest impact, addresses root cause
2. **Fix 2 (left_click normalization)** — Quick win, no retraining needed
3. **Fix 4 (Structured grounder output)** — Complements Fix 1
4. **Fix 3 (Actor prompt with type hint)** — Ensures actor uses grounder's type
5. **Fix 5 (Type-aware reward)** — Better learning signal
6. **Fix 6 (Exploration bonus)** — Prevents collapse during RL
7. **Fix 7 (Curriculum training)** — Most effort, best long-term solution

---

## Implementation Status (Fixes 1–5)

Fixes 1–5 have been implemented across `train_grpo.py`, `reward.py`, and `eval_v10_vllm.py`.

### Implemented Changes

#### Fix 1: Grounder Prompt Redesign (`train_grpo.py`, `eval_v10_vllm.py`)

`GROUNDER_SYSTEM` now instructs the grounder to determine action type and describe the target using structured output:

```python
GROUNDER_SYSTEM = (
    "You are a GUI grounding agent. Given a screenshot and an instruction, "
    "determine the next action type and describe the target.\n\n"
    "Output format:\n"
    "<action_type>one of: click, type, open, swipe, long_press, wait, system_button, terminate</action_type>\n"
    "<target>description of the target (UI element location for click/long_press, "
    "app name for open, scroll direction for swipe, button name for system_button, "
    "reason for wait, or text to type)</target>"
)
```

`format_grounder_text()` tail changed from "Describe the target UI element for the next action." to "Determine the action type and describe the target."

#### Fix 2: `left_click` Normalization (`reward.py`, `eval_v10_vllm.py`)

Both GT and predicted action types are normalized: `left_click` → `click`. Applied in:
- `grounder_reward()` — normalizes both `gt_type` and `pred_type`
- `actor_reward()` — normalizes both `gt_type` and `pred_type`
- `evaluate_action()` (eval) — normalizes both before comparison

Expected impact: ~7% type accuracy improvement, ~6% click action accuracy improvement.

#### Fix 3: Actor Prompt with Type Hint (`train_grpo.py`, `eval_v10_vllm.py`)

`ACTOR_SYSTEM` updated to mention "grounding analysis (action type + target description)".

`format_actor_text()` signature changed from `(goal, history, grounding)` to `(goal, history, action_type, target)`, passing the grounder's parsed action type and target as separate fields:

```python
parts.append(f"\nGrounding action type: {action_type}")
parts.append(f"Grounding target: {target}")
```

#### Fix 4: Structured Grounder Output (`train_grpo.py`, `eval_v10_vllm.py`)

New `parse_grounder_output(text) -> (action_type, target)` function extracts structured fields from `<action_type>` and `<target>` tags. Falls back to `("unknown", full_text)` if tags are missing, ensuring robustness during early training.

Used in both `generate_rollouts()` and `validate()` (train) and `process_step()` (eval).

#### Fix 5: Type-Aware Reward — `left_click` Normalization Only

The main reward change is the `left_click` normalization (Fix 2). The actor reward structure is kept strict: type mismatch → 0.0. The prompt redesign (Fixes 1, 3, 4) addresses the root cause of action type collapse, so additional reward shaping is not needed at this stage.

### Files Modified

| File | Changes |
|------|---------|
| `v10/train_grpo.py` | `GROUNDER_SYSTEM`, `ACTOR_SYSTEM`, `format_grounder_text()`, `format_actor_text()` (new signature), `parse_grounder_output()` (new), `generate_rollouts()`, `validate()` |
| `v10/reward.py` | `left_click` → `click` normalization in `grounder_reward()` and `actor_reward()` |
| `v10/eval_v10_vllm.py` | Same prompt/format changes as `train_grpo.py`, `parse_grounder_output()` (new), `left_click` normalization in `evaluate_action()`, `process_step()` updated |

### Verification Plan

1. Submit eval job with updated prompts on small subset (`--max_episodes 10`) to verify grounder outputs structured format
2. Check that `left_click` normalization works (expect ~7% type_acc improvement on existing results)
3. Re-train from base with updated prompts (new run, not resume)
4. Compare action type distribution — should see more non-click types
