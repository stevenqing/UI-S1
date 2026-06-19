# Long-Horizon Memory Router: First-Principles Research Method

Minimal next-step version: `docs/minimal_counterfactual_memory_utility.md`.

## Motivation

The all-sample Qwen3-VL and Qwen3.5 behavior runs gave us two facts:

```text
1. Current-screen grounding solves most GUI steps, even many long-horizon-looking steps.
2. Segment memory helps a small but meaningful hard subset, especially carried-value and multi-segment cases.
```

The first router baselines gave a negative result:

```text
Shallow structural features can predict the majority no-history class and many replan cases,
but they cannot reliably predict rare segment-memory rescue cases.
```

So the next step should not be "add a few more scalar features." The real research problem is to learn when memory contains causal information missing from the current screen.

## First-Principles Formulation

At step `t`, an agent chooses action `a_t` from:

```text
goal g
current observation o_t  = screenshot / UI / OCR / local instruction
history h_<t
segment memory m_t       = compact summaries and carried values
```

The current screen is sufficient when:

```text
p(a_t | g, o_t) is already sharp and correct
```

Memory is useful only when it carries information about a latent task state `z_t` that is not recoverable from the current observation:

```text
I(a_t ; m_t | g, o_t) > cost(memory)
```

In words:

```text
Use memory only if it changes the predicted action in a correct, specific, and non-stale way.
```

The central object is not a generic long-horizon label. It is conditional memory utility:

```text
U_memory(g, o_t, m_t) = expected action utility with memory - expected action utility without memory
```

Our model-behavior validation estimates this utility through interventions.

## Counterfactual Intervention View

For each case, run the same model under four contexts:

```text
C0 = no_history
C1 = segment_summary
C2 = full_history
C3 = wrong_summary
```

This gives a behavior vector:

```json
{
  "no_history": false,
  "segment_summary": true,
  "full_history": true,
  "wrong_summary": false
}
```

This vector is a causal probe:

```text
segment_summary true while no_history false      => compact memory rescue
segment_summary true while wrong_summary false   => memory is specific, not generic extra context
full_history true while segment_summary false    => summary missing needed detail
segment_summary false while no_history true      => memory regression / stale summary risk
all false                                       => perception/reasoning failure, not a memory route problem
```

## Research Hypothesis

The next-stage hypothesis is:

```text
A selective memory router can learn memory utility if it sees semantic alignment between
the current action need, current screen evidence, carried entities, segment summaries,
and candidate disagreement patterns.
```

The first router failed because it only saw shallow structure:

```text
step_index
prev_segments
segment_len_so_far
carried_value_count
memory_strength
dominant_capability
case_kind
```

Those features say "this step might be long-horizon", but not "this exact action needs this exact memory."

The missing variables are semantic and visual:

```text
Does the current screen contain the carried value?
Does the current instruction refer to the carried value?
Does the next action need an entity selected earlier?
Does no_history predict a plausible action with the wrong exact value?
Does segment_summary repair that value?
Does wrong_summary pull the model toward a stale or unrelated entity?
```

## Method: Behavior-Validated Counterfactual Memory Router

The proposed method is a two-stage router trained from behavior-derived labels.

### Stage 1: Hard-State Detector

Predict whether current-screen action prediction is likely to fail or be ambiguous.

Target positives:

```text
no_history wrong
all_conditions_wrong
non_obvious_no_history_wrong
low no-history confidence if available
```

This stage should have high recall. Missing a hard state prevents any memory rescue.

### Stage 2: Memory Utility Ranker

Given a hard state, rank candidate context policies:

```text
no_history
segment_summary
full_history
replan/escalate
avoid_segment_summary
```

Instead of a flat classifier, train pairwise preferences from interventions:

```text
segment_summary > no_history       when segment_rescue
segment_summary > wrong_summary    when segment_beats_wrong
full_history > segment_summary     when full_history_rescue_only
no_history > segment_summary       when segment_regression
replan > all contexts              when all_conditions_wrong
```

Pairwise/ranking is better than ordinary class prediction because the labels are sparse and utility-ordered.

## Data Construction

The current behavior-derived routing data is:

```text
datasets/long_horizon_routing_data_qwen3_qwen35/routing_examples.jsonl
```

Existing fields:

```text
model_key
thinking_mode
case_kind
step_index
prev_segments
segment_len_so_far
carried_values
carried_value_count
memory_strength
dominant_capability
gt_action_type
route_label
route_reason
condition_value_match
```

Needed new fields:

```text
ocr_tokens
screen_caption
visible_text_contains_carried_value
visible_text_goal_overlap
keyboard_or_text_field_visible
list_or_detail_state
app_surface_signature
carried_value_in_goal
carried_value_in_current_instruction
carried_value_in_screen
segment_summary_current_instruction_overlap
segment_summary_screen_overlap
segment_summary_goal_slot_overlap
no_history_candidate_type
segment_candidate_type
no_history_vs_segment_action_type_agree
no_history_text_value_overlap_with_carried_values
segment_text_value_overlap_with_carried_values
no_history_is_semantically_plausible_but_exact_value_wrong
```

These features can be generated automatically from existing fields plus OCR/visual captioning. They are closer to the first-principles question: whether memory contains missing action-relevant information.

## Training Objectives

### Objective A: Route Classification

Predict:

```text
use_no_history
use_segment_summary
use_full_history
escalate_or_replan
avoid_segment_summary
```

This is useful for reporting, but should not be the only objective.

### Objective B: Memory Utility Ranking

Train scores:

```text
s_no_history
s_segment_summary
s_full_history
s_replan
s_avoid_memory
```

Use pairwise losses:

```text
loss = max(0, margin - s_positive + s_negative)
```

Examples:

```text
segment_rescue:          s_segment_summary > s_no_history
segment_beats_wrong:     s_segment_summary > s_wrong_summary_proxy
full_history_rescue:     s_full_history > s_segment_summary and s_full_history > s_no_history
segment_regression:      s_no_history > s_segment_summary
all_conditions_wrong:    s_replan > all context scores
```

### Objective C: High-Precision Memory Activation

Because memory positives are rare, optimize a constrained objective:

```text
maximize recall(use_memory)
subject to precision(use_memory) >= target_precision
```

Recommended target operating points:

```text
precision >= 0.50 for early experiments
precision >= 0.70 for deployment-like settings
```

This is more appropriate than optimizing accuracy, because `use_no_history` is over 90% of examples.

## Model Families To Try

### Rich Feature Gradient Boosting

Use engineered semantic/screen/candidate features with a calibrated model:

```text
HistGradientBoostingClassifier
XGBoost/LightGBM if available
calibrated probability threshold for memory activation
```

This is fast and interpretable.

### Cross-Encoder Memory Utility Model

Input:

```text
[goal]
[current instruction]
[screen OCR/caption]
[segment summary]
[carried values]
```

Output:

```text
memory utility score
route scores
```

This directly tests whether semantic alignment is the missing signal.

### Candidate-Aware Router

Generate no-history and segment-summary candidate actions first, then route based on disagreement:

```text
if candidates agree and pass verifier: use cheaper route
if segment candidate repairs entity/value/action-type: use segment memory
if both fail: replan/escalate
```

This is likely the strongest practical method because many memory-rescue cases are visible as candidate disagreement.

## Evaluation Protocol

### Offline Router Metrics

Episode-level held-out split.

Report:

```text
route accuracy
macro F1
memory activation precision/recall/F1
segment_rescue recall
segment_regression false activation rate
escalate_or_replan recall
```

### Counterfactual Utility Metrics

Use the original behavior table to estimate routed accuracy without re-querying the model:

```text
accuracy_if_router_selects_context
memory_activation_rate
stale_memory_regression_rate
oracle_gap_to_best_context
cost-adjusted utility
```

### Prospective Routed Evaluation

Actually run the model using router-selected contexts on a held-out split.

Compare against static policies:

```text
always no_history
always segment_summary
always full_history
oracle best of four
learned router
```

The learned router is useful only if it beats `always no_history` without introducing many stale-memory regressions.

## Expected Outcomes

The expected shape is:

```text
always no_history: high baseline, cheap, misses memory rescues
always segment_summary: small average gain or neutral, possible stale regressions
always full_history: expensive, can help hard cases, can overfit/stale
learned router: similar average accuracy to no_history, but higher hard-subset accuracy and lower stale regression than always memory
```

The target is not a huge full-set accuracy jump. The target is selective improvement:

```text
more segment_rescue recovered
fewer segment_regressions
better exact carried-value actions
better non-obvious swipe/type/system actions
```

## Why This Is A Research Method

The method is not just feature engineering. It proposes a way to discover memory utility from model behavior:

```text
1. Treat context as an intervention.
2. Measure counterfactual success/failure under context variants.
3. Convert behavior vectors into memory utility labels and pairwise preferences.
4. Train a router to predict those utilities from segment/current-state features.
5. Validate by prospective routed action generation.
```

This makes the segmentation system self-improving:

```text
better segment summaries -> stronger memory rescues
router errors -> reveal missing segment features
hard-case clusters -> suggest new capability/memory schema refinements
```

## Immediate Next Implementation

1. Add OCR/screen-caption extraction for routing examples.
2. Add carried-value overlap features.
3. Add candidate-disagreement features using stored no_history and segment_summary predictions.
4. Train a two-stage router:
   - hard-state detector,
   - memory utility ranker.
5. Evaluate routed utility offline before running another expensive vLLM pass.

The next artifact should be:

```text
datasets/long_horizon_routing_data_qwen3_qwen35/routing_examples_enriched.jsonl
datasets/long_horizon_router_enriched/router_training_report.md
datasets/long_horizon_router_enriched/routed_utility_report.md
```