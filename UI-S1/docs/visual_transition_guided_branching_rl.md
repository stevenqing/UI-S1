# Offline Visual Transition Optimization

Date: 2026-06-25

## Motivation

GUI-360 offline data is teacher-forced in state space: every screenshot is the ground-truth screen from the expert trajectory. This is useful for learning action selection on expert states, but it does not expose the model to the states induced by its own actions.

The full real-world objective would be closed-loop control:

```text
maximize E_{s_t ~ d_pi, a_t ~ pi(.|s_t, g)} [R]
```

Offline GUI-360 mostly gives:

```text
(s_t*, a_t*, s_{t+1}*) ~ d_expert
```

Therefore, pure offline training should not be expected to solve real closed-loop control. However, if the target is explicitly the current offline GUI-360 evaluation, then the objective is narrower and tractable: optimize action survival on ground-truth screenshots under `stop_on_error`.

## Core Hypothesis

To beat the previous best results, optimize visual state alignment rather than textual history repair.

Evidence so far:

- Text-history error injection under GT-screen rescue is nearly flat, so text history is not the main causal error horizon.
- Full SFT changes the visual side mostly through `visual.merger`; visual encoder blocks are effectively unchanged.
- Static SVD merge recovered only 17.4 TSR / 64.6 StepSR, so SVD compression alone is not enough.
- V15 SVD plus cooperative RL remains the best non-full-param direction.

## Design Principle

There are two distinct objectives:

1. Offline evaluation: maximize prefix survival on expert states `s_t*`.
2. Real online control: maximize reward on policy-induced states `s_t ~ d_pi`.

This document now focuses on the first objective. Do not claim this solves online recovery; use it to beat the offline benchmark cleanly.

## Offline Evaluation Objective

With GT screenshots and `stop_on_error`, an episode succeeds only if every prefix action is correct. For an episode with `T` steps, the effective target is approximately:

```text
P(success) = product_t P(match(a_t, a_t*) | s_t*, goal, correct prefix)
```

Uniform step accuracy is not the right training objective. A mistake at early depth destroys the whole episode, and a small per-step gain compounds across the prefix. The offline-only strategy should therefore optimize survival-weighted action matching:

```text
L = sum_t w_t * CE(a_t*, pi(. | s_t*, goal, history_t))
```

where `w_t` should upweight high-leverage steps:

- early steps, because all later credit depends on them
- long-episode prefixes, because long tasks dominate TSR difficulty
- historically fragile families: click grounding, text value carry, swipe/scroll, premature terminate
- steps where the current model has low margin or high disagreement under sampling

The important pivot: offline improvement is not recovery. It is better expert-state decision making under the exact evaluator distribution.

## Minimal Algorithm

### Stage 0: Stable Starting Point

Use `checkpoints/gui360-fullparam-sft-step250` as the base policy, not the raw base model and not the static SVD merge.

Train only a small post-training surface:

- language LoRA or cooperative LoRA on `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- route and communication parameters if cooperative LoRA is used
- `visual.merger` or a merger LoRA with a smaller learning rate

Keep the visual encoder frozen.

### Stage 1: Offline Transition Prior

Build transition examples from expert trajectories:

```text
input:  goal, s_t, a_t, s_{t+1}
target: transition_is_plausible, action_type, target_region, progress_delta, terminate_ok
```

This can be trained as an auxiliary loss on the same model or as a lightweight verifier. The key is action-conditioned visual alignment: the model should know what visual change an action is supposed to produce.

Suggested first losses:

- action reconstruction from `(goal, s_t, s_{t+1})`
- terminate validity from `(goal, s_t, s_{t+1})`
- progress classification from `(goal, s_t, s_{t+1})`
- optional target-region consistency for click actions

This stage is not meant to improve TSR by itself. It is meant to provide dense reward and representation pressure for Stage 2.

### Stage 2: Offline Hard-State Mining

Run the current policy on GT screenshots over the training split and mine hard states:

```text
hardness(s_t*) = 1 - P(match) or vote_disagreement or first-error frequency
```

Build a replay set that over-samples:

- first-error states
- one step before first-error states
- long-task early prefixes
- false `terminate` predictions
- click-coordinate near misses
- text-entry value mismatches

The replay examples still use GT screenshots. This is intentional because the offline evaluator also uses GT screenshots.

Use the existing matcher as an offline oracle for candidate-set distillation:

```text
for each GT state s_t*:
  sample K actions from pi(. | s_t*)
  score each action with compute_step_reward / matcher against a_t*
  if greedy is wrong but at least one sampled action is correct:
    train preference or SFT toward the best correct candidate
  if all samples are wrong:
    keep the GT action as a high-weight hard example
```

This is the offline analogue of exploration: no new screens are needed, but the model learns to put probability mass on actions it already sometimes knows how to produce.

### Stage 3: Visual Transition Prior Without Online Rollout

Use adjacent GT screenshots only as an auxiliary signal, not as simulated environment feedback.

Train the model to answer action-conditioned visual questions:

```text
given goal, s_t*, a_t*, s_{t+1}*: was this transition plausible and goal-progressing?
given goal, s_t*, s_{t+1}*: reconstruct a_t*
given goal, s_t*: is terminate valid now?
```

This helps the offline model bind actions to visual consequences, but the evaluation target remains the current-step action.

### Future Only: Short Online Branching Rollouts

This section is not part of the current offline-only plan. It is kept only to mark the boundary between benchmark optimization and real closed-loop control.

Start from an expert prefix, then let the current policy control the emulator for a short branch.

```text
expert prefix: s_0*, a_0*, ..., s_t*
branch:        a_t ~ pi, s_{t+1} ~ env, a_{t+1} ~ pi, ... for H steps
```

Use small horizons first:

```text
H in {1, 2, 4}
```

Collect real off-expert screenshots from AndroidWorld or the available Android emulator stack. Store every branch as:

```json
{
  "episode_id": "...",
  "goal": "...",
  "prefix_step": 7,
  "branch_step": 1,
  "screen_before": "...png",
  "action": {...},
  "screen_after": "...png",
  "matched_gt": false,
  "reward": 0.0,
  "done": false,
  "failure_reason": "wrong_click|bad_text|premature_terminate|off_task|..."
}
```

### Stage 4: Offline Reward / Loss Shaping

For offline-only optimization, use loss shaping rather than online reward.

Recommended components:

```text
L = L_action_ce
  + alpha * L_transition_aux
  + beta  * L_terminate_validity
  + gamma * L_margin_hard_states
```

Important: `terminate` should be treated as a calibrated binary decision. Most non-final states should create negative examples for `terminate`, because premature termination is disproportionately harmful under TSR.

### Stage 5: Offline Post-Training Update

Use the existing V15/V13 data and LoRA infrastructure as the first training skeleton. Do not redesign the architecture before the offline hard-state and transition-prior baseline is measured.

First training variants:

| Variant | Base | Trainable params | Hard mining | Transition aux | Goal |
|---|---|---|---|---|---|
| A | full SFT step250 | language LoRA | no | no | post-SFT offline baseline |
| B | full SFT step250 | language LoRA + visual.merger | no | yes | test transition pressure |
| C | full SFT step250 | language LoRA + visual.merger | yes | yes | main offline improvement test |
| D | full SFT step250 | cooperative LoRA + visual.merger | yes | yes | V15-style offline mainline |
| E | D | same | yes, stronger long-task weights | yes | push TSR on long episodes |

## Evaluation Protocol

Use the balanced 1000-episode GUI-360 eval for the final metric:

```text
stop_on_error = true
match_threshold = 0.5
history_mode = full
gt_history = false
```

Use cheap gates before full eval:

- 50 or 100 episode slice after every short training interval
- 200 episode slice before any 1000 episode run
- report TSR, StepSR, progress, premature terminate rate, and average completed depth

Success criteria:

```text
minimum useful: > 22.2 TSR on full 1000
strong result:  23-24 TSR with StepSR not below full SFT
diagnostic win: progress improves even if TSR is flat
```

## Implementation Order

1. Add offline transition dataset builder from GUI-360 episodes.
2. Add transition verifier or auxiliary head/loss.
3. Add hard-state mining by running the current model on GT screenshots and recording first-error states.
4. Add K-sample matcher distillation on hard states.
5. Add survival-weighted supervised/RL-style loss over the mined offline states.
6. Add premature-terminate diagnostics and loss weighting.
7. Run variants A-D with 100/200 episode eval gates.

## First Concrete Experiment

Do not start with the full online system. Start with the smallest disconfirming check:

```text
Can transition-prior auxiliary loss plus hard-state replay on full SFT step250 reduce first-error rate and premature terminate on a 200-episode eval slice?
```

If no, the transition prior is not helping the offline evaluator and should be dropped. If yes, scale to cooperative LoRA plus stronger long-task weighting.

## Why This Is Simpler Than V17-Style Aux Generation

This plan does not depend on emergent textual reasoning or extra prompt phases. For offline evaluation, it directly optimizes the benchmark distribution:

```text
offline eval: s ~ d_expert, terminate at first wrong action
```

The model improves by learning what actions do to the visual world, then applying that pressure to the exact GT-screen action decisions used by the evaluator.