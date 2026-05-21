# V10 Cooperative LoRA GRPO — Journey Summary

## Architecture
- **Two separate LoRA adapters** on Qwen2.5-VL-7B: Grounder (LoRA_G) + Actor (LoRA_A)
- **Sequential generation**: Grounder generates text description → Actor reads it + screenshot → generates action
- **Communication**: Text-based (non-differentiable), grounder output injected into actor prompt
- **Training**: GRPO with K=8 rollouts, DDP on 4 nodes × 4 GPUs

## Eval Results Comparison

### Step-Level Accuracy

| Checkpoint | TSR | Step Acc | Type Acc |
|------------|-----|----------|----------|
| V10.0 s300 (early) | 6.55% | **34.2%** | 56.7% |
| V10.0 s200 (8node) | 6.35% | 42.2% | 71.3% |
| V10.0 epoch-2 | 6.48% | 42.5% | 71.2% |
| V10.0 epoch-3 | 5.90% | 42.2% | 71.2% |
| V10.1 ep0 (format rw) | 6.16% | 42.4% | 71.3% |
| V10.1 ep1 (format rw) | 6.55% | 42.5% | 71.2% |

**Key finding**: V10.0 converges to the same step acc (~42%) and type acc (~71%) given enough training. Grounder format reward did NOT help — V10.0 epoch-2 matches V10.1 ep1 exactly. The early eval (s300) was just an undertrained snapshot.

### Per Action-Type Breakdown (V10.1 ep1 — representative of all converged checkpoints)

| Action | Total | Type Acc | Action Acc | Notes |
|--------|-------|----------|------------|-------|
| click | 5074 | 92.4% (4687) | 50.4% (2557) | Good type, but 2130 wrong-element clicks |
| type | 632 | 94.1% (595) | 65.8% (416) | Best performing |
| swipe | 1211 | 38.2% (462) | 31.1% (377) | 89.5% of errors → click |
| wait | 567 | 41.3% (234) | 41.3% (234) | If type correct, action always correct |
| open | 608 | 3.3% (20) | 0.0% (0) | **Broken** — model doesn't know this action |
| system_button | 343 | 4.1% (14) | 1.2% (4) | **Broken** — all GT = "Back" |
| long_press | 9 | 11.1% (1) | 0.0% (0) | Too few samples |

### Per Trajectory Length (V10.1 ep1)

| Bucket | Episodes | TSR | Step Acc | Avg Progress |
|--------|----------|-----|----------|--------------|
| short (1-3) | 438 | 17.6% | 39.8% | 0.209 |
| medium (4-7) | 788 | 2.9% | 45.4% | 0.066 |
| long (8-15) | 289 | 0.35% | 41.0% | 0.016 |
| vlong (16+) | 28 | 0.0% | 32.8% | 0.002 |

## Bad Case Analysis (V10.1 ep1)

### 1. `open` — 0% action accuracy (608 samples)

Grounder predicts instead: click 56.5%, swipe 20.1%, terminate 6.8%.

**Root cause**: The model doesn't know `open` is an action type. It converts "Open X app" into "click on X icon" or interacts with whatever is currently on screen. Often the target app isn't even visible — the model should issue `open(app_name)` but instead clicks random elements.

### 2. `system_button` — 1.2% action accuracy (343 samples, all GT="Back")

Grounder predicts instead: click 75.4%, swipe 5.8%, terminate 5.2%.

**Root cause**: Model doesn't distinguish Android system Back button from in-app back arrow. ~15% of errors correctly identify "go back" intent but click the visual back arrow instead of issuing `system_button(Back)`. The other 85% ignore the navigation intent entirely.

### 3. `click` — 50.4% action accuracy (2130 wrong out of 4687 type-correct)

Coordinate error distribution (1918 with coords):
- 50-100px: 6.3% (borderline)
- 100-500px: 43.8% (wrong area)
- 500+px: 50.0% (completely wrong)
- **Median error: 499px, Mean: 716px**

**Root cause**: NOT localization errors — the model clicks the **wrong element entirely**. 83% of errors are >200px away. The grounder describes the wrong target, not an imprecise location.

### 4. `swipe` — 31.1% action accuracy (749 wrong out of 1211)

Grounder predicts instead: click 89.5%, terminate 6.5%.

**Root cause**: Grounder converts scroll/swipe instructions into click. "Scroll down to see more" → click on content below. "Swipe left on carousel" → click navigation arrow. The grounder is the bottleneck — it doesn't understand swipe as a gesture.

## Key Takeaways

1. **TSR unchanged despite 6× step acc improvement** — because `open` (608 steps, 0% acc) and `system_button` (343 steps, 1.2% acc) are in many trajectories and break the chain
2. **Model's action vocabulary is click-dominated** — click is 60% of GT but model predicts click ~85%+ of the time
3. **Two epochs give same results** — grounder format reward converges in epoch 0, no further gain
4. **Click errors are wrong-element, not localization** — median 499px error means fundamentally wrong target
5. **Grounder is the bottleneck for non-click actions** — it converts everything to click before actor sees it
6. **To improve TSR**: must fix `open` and `system_button` recognition; these action types don't exist in the model's learned vocabulary

## V10.1 Fix: Grounder Format Reward
```
grounder_reward = 0.5 × format_reward + 0.5 × downstream_reward

format_reward:
  +0.3  has <action_type> tag
  +0.5  action type matches GT
  +0.2  has <target> tag
```

Originally thought to improve type_acc from 56.7% → 71.2%, but V10.0 reaches the same numbers with more training. **The format reward provided no additional benefit** — the 56.7% was just from an undertrained checkpoint (s300). All converged checkpoints plateau at ~71% type acc, ~42% step acc, ~6.5% TSR regardless of format reward.
