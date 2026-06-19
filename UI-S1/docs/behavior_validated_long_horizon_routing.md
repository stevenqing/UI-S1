# Behavior-Validated Long-Horizon Routing

## Core Research Idea

Long-horizon reasoning should not be treated as a default prompting style. It should be treated as a selective intervention that is activated only when the current screen is insufficient and a compact segment memory is behaviorally useful.

The key idea is to turn model behavior itself into supervision:

```text
Run the same next-action prediction under multiple context interventions.
Compare which context succeeds or fails.
Use the outcome pattern to label whether this step needs no history, segment memory, full history, or replanning.
```

This changes long-horizon routing from a hand-written rule into a data-discovered decision problem.

## Why This Is Needed

The Qwen3-VL all-sample result shows a strong asymmetry:

```text
Most steps are solved by current-screen grounding.
Only a small hard subset needs memory.
When memory helps, it is usually associated with carried values, multi-segment progress, or non-obvious actions.
```

So an always-on long-context policy is wasteful and can introduce stale-history regressions. The better objective is selective memory:

```text
default: current screen + local instruction
activate segment memory: only when segment state carries useful information
activate full history: only when compact segment memory fails but raw history rescues
escalate/replan: when all context variants fail
```

## How The Samples Are Generated

The routing samples are generated from completed model-behavior validation runs, not from hand labels.

### Step 1: Build Cases

Start from segmented trajectories:

```text
datasets/segmentation_train/gui_odyssey_segments.jsonl
```

For each episode, construct two kinds of cases:

```text
real_boundary: current step is a predicted segment start
random_control: current step is a sampled non-boundary position
```

For the Qwen3-VL all-sample run, this produced:

```text
real_boundary cases: 15182
random_control cases: 6598
total cases: 21780
```

Each case is evaluated under both thinking modes:

```text
non_thinking
thinking
```

and four context interventions:

```text
no_history       current screen + goal/current step only
segment_summary  previous segment summaries + current step/screenshot
full_history     recent raw action history + current step/screenshot
wrong_summary    unrelated segment memory + current step/screenshot
```

### Step 2: Run Counterfactual Model Interventions

For every `(case, thinking_mode)`, run the same model on all four context variants. The ground-truth next action is fixed.

This gives a behavior vector:

```json
{
  "no_history": false,
  "segment_summary": true,
  "full_history": true,
  "wrong_summary": false
}
```

The vector is more informative than a single accuracy number because it tells us which context actually changed the model's decision.

### Step 3: Convert Behavior Vectors Into Routing Labels

The label is derived from counterfactual success/failure patterns:

| Behavior pattern | Route label | Meaning |
|---|---|---|
| no_history succeeds | `use_no_history` | current screen is sufficient |
| no_history fails, segment_summary succeeds | `use_segment_summary` | compact segment memory rescues |
| no_history fails, segment_summary fails, full_history succeeds | `use_full_history` | raw history has useful details missing from segment summary |
| segment_summary succeeds, wrong_summary fails | `use_segment_summary` | memory is specific, not generic history benefit |
| no_history succeeds, segment_summary fails | `use_no_history` | memory can hurt; avoid segment memory |
| wrong_summary succeeds when segment_summary fails | `avoid_segment_summary` | memory signal is unstable or misleading |
| all variants fail | `escalate_or_replan` | need another candidate source/verifier or better perception |

The implementation is:

```text
scripts/build_long_horizon_routing_data.py
```

Current output:

```text
datasets/long_horizon_routing_data_qwen3_qwen35/routing_examples.jsonl
datasets/long_horizon_routing_data_qwen3_qwen35/routing_report.md
```

### Step 4: Add Long-Horizon Features

Each routing example includes segment-derived features:

```json
{
  "step_index": 19,
  "prev_segments": 3,
  "segment_len_so_far": 12,
  "carried_values": ["vanilla extract", "plain chocolate chips"],
  "memory_strength": "high",
  "dominant_capability": "interact",
  "gt_action_type": "swipe",
  "is_long_horizon": true
}
```

Long-horizon is not defined only by trajectory length. A sample is long-horizon if any of these hold:

```text
step_index is high
multiple previous segments exist
segment carries values/entities
memory_strength is medium/high
```

This is important because some early steps are already memory-sensitive when the goal carries a specific target value.

## Current Dataset Statistics

The combined Qwen3-VL all-sample + Qwen3.5 8192-context routing dataset has:

```text
examples: 45560
long_horizon examples: 44000
use_memory positives: 779
```

Route distribution:

| route | n | rate |
|---|---:|---:|
| use_no_history | 41555 | 0.912 |
| escalate_or_replan | 3197 | 0.070 |
| use_segment_summary | 554 | 0.012 |
| use_full_history | 225 | 0.005 |
| avoid_segment_summary | 29 | 0.001 |

The important finding is not that memory helps often. It is that memory helps rarely but specifically. This is exactly the regime where a router is useful.

## Training Objective

Train a lightweight routing head:

```text
input: current step, screen/grounding features, segment state, carried values, capability, memory features
output: route_label
```

Possible labels:

```text
use_no_history
use_segment_summary
use_full_history
escalate_or_replan
avoid_segment_summary
```

Because positives are sparse, training should be cost-sensitive:

```text
high precision for use_segment_summary
high recall for escalate_or_replan on all-condition failures
low false-positive rate for memory activation
```

The router should optimize utility, not raw class accuracy. A useful loss can weight examples as:

```text
segment_rescue > full_history_rescue_only > segment_beats_wrong > current_screen_sufficient > segment_regression
```

## Inference Policy

At inference time:

1. Run the segmenter on the current trajectory prefix.
2. Extract current segment summary, carried values, capability, and memory features.
3. Router predicts one of the context policies.
4. Generate action with that context policy.
5. Verify with capability-specific checks.
6. If route is `escalate_or_replan`, request another candidate source or stronger verifier rather than blindly adding more history.

This yields a selective long-horizon agent:

```text
screen-grounded by default
memory-aware when behaviorally justified
full-history only for rare raw-history rescues
replan when all context variants fail
```

## Research Claim

The research contribution is behavior-validated memory routing:

```text
We can discover when long-horizon memory is useful by counterfactually intervening on context and observing model success, then train a router to predict those contexts from trajectory/segment features.
```

This is stronger than manually declaring that long-horizon reasoning is useful. It produces measurable labels:

```text
current-screen sufficient
segment-memory rescue
full-history rescue
memory regression
unresolved hard case
```

and directly connects segmentation quality to downstream action prediction.

## Next Experiments

1. Train a small classifier on `routing_examples.jsonl`.
2. Evaluate on held-out episodes with route utility metrics:
   - memory activation precision,
   - segment rescue recall,
   - stale-memory regression rate,
   - overall next-action value accuracy.
3. Compare against static policies:
   - always no_history,
   - always segment_summary,
   - always full_history.
4. Re-run behavior validation using the learned router as the context selector.
5. Mine false positives and false negatives to revise segment features and memory summaries.

## First Router Training Result

Initial lightweight routers were trained from the behavior-derived labels:

```text
scripts/train_long_horizon_router.py
docs/long_horizon_memory_router_research_method.md
datasets/long_horizon_router_logistic
datasets/long_horizon_router_forest
datasets/long_horizon_router_forest_with_action
datasets/long_horizon_router_qwen3vl_forest
datasets/long_horizon_router_training_summary.md
```

The result is informative but not yet sufficient for deployment:

```text
RandomForest with structural features reaches high overall accuracy because most examples are use_no_history.
However, memory-positive precision/recall is low on held-out episodes.
Adding oracle gt_action_type improves only slightly.
```

Current best action-aware forest test metrics:

```text
accuracy: 0.8840
macro_f1: 0.3200
memory precision/recall/f1: 0.0621 / 0.1196 / 0.0818
```

This is a useful negative result: the sparse memory-rescue subset cannot be recovered reliably from shallow structural features alone. The next router needs semantic/screen/candidate features such as carried-value overlap, OCR/current-screen availability, and no-history-vs-segment candidate disagreement.

## Practical Caution

The current labels are model-dependent. Qwen3-VL and Qwen3.5 do not fail on exactly the same examples. The right target is therefore not an absolute memory-needed oracle, but a model-conditional router:

```text
route = f(model_family, thinking_mode, current_state, segment_state)
```

This is useful because the deployed agent also knows which model/route it is using.