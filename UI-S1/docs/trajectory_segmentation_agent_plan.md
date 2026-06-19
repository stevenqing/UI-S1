# Cross-Benchmark Trajectory Segmentation Agent Plan

## Goal

Build a trajectory-understanding agent that can automatically split GUI-agent trajectories into meaningful subtasks and sub-capabilities across benchmarks, not only on GUI-Odyssey.

The desired output is a structured hierarchy:

```text
benchmark episode
  -> macro segment / subtask
      -> capability phase
          -> atomic action
```

This should support three downstream uses:

1. Selective memory: decide when history is actually needed.
2. Candidate routing: choose which prompt/model/route should generate candidates for this step.
3. Verification: reject stale-history, wrong-target, wrong-text, and wrong-action-type candidates.

The central claim is that v2/v3/v4-style routes should be treated as candidate sources. The segmentation and verifier are the mechanism that decides when to trust them.

## Why This Matters

The current evidence says route design alone is unstable.

- Prompt/template diversity has real oracle headroom, but a single template does not dominate.
- Role splitting can improve both semantic and type accuracy for stronger models, but the gain is modest.
- Long-horizon reasoning gives the largest semantic rescue, but also introduces stale-history regressions.
- Most failures are grounding and exact-value carry failures, not output-format failures.

Therefore, the next useful system is not another hand-written route. It is a boundary-aware agent that knows:

- which subtask the current step belongs to,
- what capability is needed now,
- whether history is locally necessary,
- which entity/value must be carried forward,
- whether a candidate action is grounded on the current screen.

## Data Observations From GUI-Odyssey

From the downloaded GUI-Odyssey random split test data:

- Episodes: 1666
- Steps: 25807
- Trajectory length: min 3, median 14, mean 15.49, p90 24, p95 28, max 51
- Categories: General_Tool, Multi_Apps, Information_Management, Social_Sharing, Media_Entertainment, Web_Shopping
- Actions:
  - CLICK: 18764, 72.7%
  - TEXT: 2666, 10.3%
  - SCROLL: 2622, 10.2%
  - COMPLETE: 1572, 6.1%
  - INCOMPLETE: 94, 0.4%
  - LONG_PRESS: 89, 0.3%
- System buttons inside CLICK:
  - KEY_HOME: 1964
  - KEY_BACK: 80
  - KEY_APPSELECT: 62
- `sam2_bbox` is present for 16747 steps, about 64.9%.

Each GUI-Odyssey step includes unusually useful supervision:

```text
description             current-screen visual description
intention               action rationale / local goal
low_level_instruction   short next-step instruction
context                 progress summary up to this step
action/info             ground-truth action and parameters
sam2_bbox               target bbox when available
```

Important lesson: direct step-level capability tagging is too fine-grained. The median run length of a naive capability label is only about 1 step. A useful segmentation must first find macro boundaries, then label micro capabilities within each macro segment.

## Cross-Benchmark Common Abstraction

The segmentation method must work across GUI-Odyssey, GUI-360, AndroidControl, and future mobile/desktop GUI datasets. The shared representation should avoid benchmark-specific field names.

### Canonical Episode Schema

```json
{
  "benchmark": "gui_odyssey | gui360 | android_control | ...",
  "episode_id": "...",
  "task_goal": "...",
  "task_metadata": {
    "apps": [],
    "category": "...",
    "device_or_platform": "..."
  },
  "steps": [
    {
      "step_index": 0,
      "screenshot": "...",
      "action": {"type": "click | type | scroll | open | wait | system_button | terminate", "args": {}},
      "text_fields": {
        "instruction": "...",
        "thought": "...",
        "observation": "...",
        "context": "...",
        "history": "..."
      },
      "grounding": {
        "bbox": null,
        "coordinate": null,
        "ui_element_text": null,
        "a11y": null
      }
    }
  ]
}
```

### Field Mapping

| Canonical field | GUI-Odyssey | GUI-360 | AndroidControl |
|---|---|---|---|
| task_goal | `task_info.instruction` / converted `goal` | `request` | `goal` |
| step screenshot | `steps[].screenshot` | `step.screenshot_clean` / `screenshot_desktop` | `steps[].screenshot` |
| action | `action` + `info` | `step.action` | `action_content` |
| step instruction | `low_level_instruction` | sometimes `thought` / generated prompt context | absent or derivable from action/history |
| thought/rationale | `intention` | `thought` | absent unless augmented |
| visual observation | `description` | `observation` / a11y in some prepared data | absent unless generated |
| progress/history | `context` | prompt history / trajectory order | step history / trajectory order |
| bbox/grounding | `sam2_bbox` | coordinate/control/a11y fields | `bbox` for click when present |

The segmentation algorithm should operate on canonical fields, with benchmark-specific adapters only at ingestion time.

## What Counts As A Segment

A segment is not just a consecutive run of the same action type. A segment should represent a local objective that can be summarized as:

```text
In surface/app X, accomplish Y, producing artifact/entity Z for later steps.
```

Examples:

- Open a browser and search for information.
- Select a product/result/entity from a list.
- Compose and send a message to a recipient.
- Transfer a value from one app to another.
- Configure a setting/reminder/event.
- Return home or switch apps after completing the current local objective.

The segment boundary should be placed when the local objective, UI surface, or carried state changes, not every time the action primitive changes.

## Boundary Signals That Generalize Across Benchmarks

### 1. Explicit Navigation And Surface Transition

These are the strongest benchmark-agnostic signals.

- `open` action in AndroidControl
- `Open ... app/browser` instruction in GUI-Odyssey and GUI-360-style data
- system buttons: Home, Back, AppSelect/Menu
- app switcher or launcher interactions
- screenshot-level UI state change after a system navigation

This should produce macro segments such as:

```text
open browser -> search web -> Home
open shopping app -> search/add item -> Home
open messaging app -> send message -> Home
open clock/calendar -> set reminder -> terminate
```

### 2. Task Slot Completion

Most long GUI tasks can be decomposed into slots from the global goal.

Example goal:

```text
Choose a movie, add snacks, invite Victor James, set a reminder.
```

Slots:

```json
[
  {"slot": "movie", "status": "find/select"},
  {"slot": "snacks", "status": "add_to_cart"},
  {"slot": "Victor James", "status": "send_invitation"},
  {"slot": "reminder", "status": "set"}
]
```

Boundary signal: the current segment produces or consumes a slot value.

This generalizes beyond app names because every benchmark has a task goal, and long tasks often mention multiple objects, recipients, files, settings, or target states.

### 3. Capability Phase Change

Inside a macro segment, use capability phases rather than macro boundaries.

Recommended capability taxonomy:

```text
open_surface
search_or_query
enter_text
browse_or_scroll
select_target
inspect_or_compare
copy_or_transfer
communicate_or_share
commit_or_create
system_navigation
wait_or_load
finish_or_status
```

These phases should help route candidates and verifiers, but should not always split a new subtask.

### 4. Screen-State Discontinuity

When text metadata is weak, use screenshot state changes.

Signals:

- large OCR/a11y vocabulary change,
- app chrome change,
- list/detail view transition,
- keyboard opened/closed,
- modal/dialog appeared/dismissed,
- launcher/home screen visible,
- target bbox distribution changes.

This is important for AndroidControl, where natural-language step-level annotations are sparse.

### 5. Memory Necessity

Use history only when the current action depends on a value introduced earlier.

Positive signals:

- type text copied from earlier search/result,
- recipient/product/file selected earlier,
- current screen lacks the target value but goal/history contains it,
- cross-app transfer is underway.

Negative signals:

- current screen contains all necessary target information,
- action is local navigation or local click,
- history mentions a different app/surface than the current one,
- the route proposes goal-level relevance without local necessity.

This is the main guardrail against v4 stale-history regressions.

## Proposed Agent Architecture

### Stage A: Benchmark Adapter

Normalize raw benchmark examples into the canonical schema.

Adapters:

- `GUIOdysseyAdapter`
- `GUI360Adapter`
- `AndroidControlAdapter`

Each adapter should produce canonical `Episode` and `Step` records. The downstream segmenter should not know benchmark-specific field names.

### Stage B: Boundary Proposer

Generate candidate boundary positions with high recall.

Signals:

- explicit open/home/back/app-switch actions,
- task-slot transition,
- screen-state discontinuity,
- capability-phase transition,
- long repeated scroll/search subchains that end in selection,
- terminal commit actions before navigation.

Output:

```json
{
  "candidate_boundaries": [
    {"after_step": 22, "source": "system_home", "confidence": 0.95},
    {"after_step": 31, "source": "slot_completion+system_home", "confidence": 0.91}
  ]
}
```

### Stage C: Segment Verifier

Merge or reject proposed boundaries.

Questions:

- Does the local objective change after the boundary?
- Does the app/surface change?
- Does a carried entity/value get produced before the boundary?
- Would splitting here create a single-step meaningless segment?
- Would not splitting hide a memory dependency?

This verifier should be conservative: over-segmentation is bad because it makes memory fragmented and creates noisy training labels.

### Stage D: Segment Summarizer

For each accepted segment, produce a compact memory object.

Output schema:

```json
{
  "segment_id": 2,
  "step_range": [23, 31],
  "surface": "Amazon Shopping",
  "subtask": "Search for snacks and add one item to the cart",
  "capability_phases": ["open_surface", "search_or_query", "enter_text", "browse_or_scroll", "commit_or_create"],
  "inputs_needed": ["movie night plan"],
  "outputs_produced": ["snacks added to cart"],
  "memory_for_future": "Snack item was added in Amazon; no exact text needs to be carried unless later asked."
}
```

### Stage E: Capability Router

Use the current segment and current step capability to choose candidate generation and verification strategy.

Routing examples:

- `enter_text`: exact value carry verifier, compare with goal/segment memory.
- `select_target`: grounding verifier using bbox/OCR/a11y/screenshot.
- `browse_or_scroll`: progress verifier; avoid premature termination.
- `system_navigation`: require evidence that local segment is complete.
- `communicate_or_share`: verify recipient and message payload.
- `commit_or_create`: verify final state or target button before accepting.

## Relationship To v2 / v3 / v4 Routes

The segmenter turns the three routes into controlled tools.

### v2: Diverse Prompt/Memory Agents

Use v2-style candidates when capability uncertainty is high.

- Good for: alternative target hypotheses, exact text carry variants, non-overlapping prompt wins.
- Risk: type/action drift and stale memory.
- Segmenter guardrail: only allow memory-heavy v2 candidates when `memory_needed=true`.

### v3: Reasoner + Tool Caller

Use v3 when the current step needs a concise local rationale before action.

- Good for: action-type sanity, local tool-call formatting, modest balanced improvements.
- Risk: can still pick wrong primitive or target.
- Segmenter guardrail: reason only over the current segment, not the full trajectory.

### v4: Long-Horizon Reasoning

Use v4 selectively at macro boundaries or cross-app transfer points.

- Good for: long-horizon entity/state dependencies.
- Risk: stale-history regressions.
- Segmenter guardrail: retrieve only relevant segment summaries, never dump raw full history by default.

## Training Plan

### Phase 0: Rule-Based Weak Labels

Create weak segment labels across benchmarks.

High-precision macro boundaries:

- `open` action following a previous local objective,
- Home/AppSelect/Back followed by a different surface,
- `terminate`,
- goal slot transition,
- strong screen-state discontinuity.

Weak capability labels:

- action type,
- low-level instruction verb,
- thought/intention keywords,
- UI/a11y/OCR state.

Expected output files:

```text
datasets/segmentation/gui_odyssey_segments.jsonl
datasets/segmentation/android_control_segments.jsonl
datasets/segmentation/gui360_segments.jsonl
```

### Phase 1: LLM Adjudication

Use a stronger model as a boundary adjudicator on a sampled subset.

Prompt contract:

```text
Given goal, actions, step instructions, optional observations, and proposed boundaries,
return accepted segments with subtask summaries and memory-needed flags.
Do not split only because action type changes.
Do split when local objective, surface, carried entity, or output artifact changes.
```

Use this to clean the weak labels and create an evaluation set.

### Phase 2: Train A Lightweight Segmenter

Inputs:

- goal,
- previous segment summary,
- recent step instructions/actions,
- current screenshot embedding or OCR/a11y summary,
- current action candidate when available.

Outputs:

```json
{
  "boundary_before_step": false,
  "macro_segment_label": "search web for movie",
  "capability": "enter_text",
  "memory_needed": true,
  "carried_entities": ["adventure movie", "Victor James"],
  "confidence": 0.86
}
```

### Phase 3: Integrate With Action Candidate Selection

At inference time:

1. Run segmenter on current trajectory prefix.
2. Select relevant segment memory.
3. Generate candidates from chosen prompt/model/route pool.
4. Verify using capability-specific checks.
5. Commit only if confidence is high; otherwise ask for another candidate source.

## Cross-Benchmark Spotting Method

The key method should be benchmark-agnostic and use a two-level detector:

### Level 1: Universal Boundary Score

For each potential boundary after step `t`, compute:

```text
boundary_score(t) =
  w1 * navigation_transition
  + w2 * surface_change
  + w3 * goal_slot_change
  + w4 * produced_artifact
  + w5 * memory_scope_change
  + w6 * terminal_or_commit_transition
  - w7 * short_fragment_penalty
  - w8 * same_screen_same_object_penalty
```

This is universal because it depends on fields all benchmarks can provide or derive:

- action sequence,
- screenshots,
- task goal,
- optional OCR/a11y/observation,
- optional bbox/coordinates,
- previous and next local instructions when available.

### Level 2: Segment Validity Verifier

For each proposed segment, check:

```text
coherence:       do steps pursue the same local objective?
completeness:    does the segment produce/consume a meaningful state?
non-triviality:  is it more than a formatting/action-type artifact?
memory utility:  would this segment summary help future steps?
screen locality: does current action depend on current screen or earlier segment?
```

This turns segmentation into a decision problem rather than a pure clustering problem.

## Automatic Design Discovery

The current annotation layer is a manually seeded design, but it should not stay purely hand-written. A better long-term method is to treat the design as a latent structure discovery problem: find recurring boundary patterns, recurring local objectives, and recurring capability phases from the trajectories themselves, then use human/LLM labels only to name and validate the discovered clusters.

### Discovery Principle

Do not start by deciding the final taxonomy. Start with many candidate signals and ask which hidden states best explain the observed trajectory transitions.

For every step and every adjacent transition, build a benchmark-agnostic feature record:

```json
{
  "goal_tokens": "...",
  "action_type": "click | type | swipe | open | wait | system_button | terminate",
  "action_args": {...},
  "step_text": "instruction/thought/observation/context if available",
  "screen_signature": "OCR/a11y/image embedding if available",
  "grounding_signature": "bbox/coordinate region",
  "prev_to_current_delta": {
    "action_change": true,
    "text_shift": 0.73,
    "screen_shift": 0.64,
    "goal_slot_shift": false,
    "system_nav": false
  }
}
```

Then learn two latent variables:

```text
z_segment(t):     which macro subtask the current step belongs to
z_capability(t):  which local capability is required at this step
```

The hand-written labels such as `search`, `browse_scan`, `navigate_system`, and `configure_edit` should be treated as temporary names for clusters, not as the final ontology.

### Practical Algorithm

1. Generate high-recall candidate boundaries using the weak proposer.
2. Embed each transition with action, text, goal-slot, screen, and grounding delta features.
3. Cluster transitions into boundary archetypes.
4. Cluster within-segment step windows into capability archetypes.
5. Ask an LLM or human to name only the stable clusters, not every example.
6. Convert cluster names into the first taxonomy.
7. Train a lightweight segmenter to predict the discovered boundary/capability labels.
8. Re-run discovery on model errors and add or merge clusters.

This gives an iterative loop:

```text
weak signals -> latent clusters -> LLM/human names -> trained segmenter -> error clusters -> revised schema
```

### How To Know The Design Is Real

A discovered segment/capability design is useful only if it is stable and predictive.

Use these checks:

- Cross-benchmark stability: the same cluster appears in GUI-Odyssey, AndroidControl, and GUI-360.
- Route predictiveness: cluster identity predicts which candidate route/verifier works best.
- Memory predictiveness: cluster identity predicts whether history helps or hurts.
- Boundary usefulness: segment summaries improve downstream action accuracy or reduce stale-history errors.
- Compression: the schema explains long trajectories with fewer, coherent states than action-type runs.

If a cluster exists only in one benchmark and does not improve routing, memory, or verification, it should not become part of the shared taxonomy. It can remain a benchmark-specific feature.

### Why This Can Discover The Schema Automatically

GUI tasks have repeated hidden structure even when apps and benchmarks differ:

```text
open/switch surface -> search/input -> browse/select -> commit/create -> navigate/finish
```

The exact app names change, but the transition signatures are similar. For example, a search phase often has goal-token overlap, type/click actions, keyboard or search-bar state, then a shift into result browsing. A cross-app transfer often has a value introduced in one segment, a system navigation boundary, and later a type/select action that consumes that value. These patterns can be discovered from transition statistics before they are named.

So the intended method is not to freeze today's hand-made labels. The intended method is to use today's labels as weak initialization, then let cross-benchmark clustering and downstream route/memory/verifier gains decide the final segmentation ontology.

### Current Prototype

Implemented an offline discovery script:

```text
scripts/discover_segmentation_schema.py
```

It does not read the weak segment JSONL as supervision. It starts from raw train trajectories, runs the benchmark adapters only to normalize field names, extracts transition/window features, and clusters them into candidate archetypes.

Current train run:

```text
datasets/schema_discovery_train/discovered_schema.json
datasets/schema_discovery_train/discovered_schema_report.md
```

The first discovered boundary archetypes include:

```text
type -> click              query/value entered, then execute/select
click -> type              focus field/search bar, then enter value
system_button -> click     after surface navigation, open/select next app
click -> terminate         commit/final action, then task finish
click -> system_button     local objective done, navigate away
swipe -> click             browse/scan, then select target
```

These were recovered from raw goal/action/text/grounding transition statistics, not from the hand-written `dominant_capability` labels. The next version should replace the greedy feature clustering with learned embeddings and evaluate which discovered clusters best predict routing, memory use, and verifier success.

### Boundary Signal Mining Validation

Implemented a second diagnostic script:

```text
scripts/mine_boundary_signals.py
```

This script treats weak segment starts as provisional boundary labels, but computes predictors only from raw transition fields. It answers: which raw features actually predict boundaries?

Current output:

```text
datasets/schema_discovery_train/boundary_signal_scores.jsonl
datasets/schema_discovery_train/boundary_signal_report.md
```

Top raw boundary predictors on train:

```text
action_bigram:system_button->click          precision 1.000, lift 6.49
action_bigram:system_button->swipe          precision 1.000, lift 6.49
prev_system_nav                             precision 0.995, lift 6.46
action_bigram:swipe->system_button          precision 0.966, lift 6.27
action_bigram:type->system_button           precision 0.949, lift 6.16
curr_action:system_button                   precision 0.849, lift 5.51
action_bigram:click->system_button          precision 0.828, lift 5.37
```

Interpretation: surface transitions and local-objective completion can be recovered almost directly from raw action transitions. The less obvious semantic boundaries still require text/screen/goal-slot features or LLM adjudication, but the strongest part of the schema is already data-supported.

### Bottleneck Validation Status

Implemented an offline proxy validator:

```text
scripts/validate_segmentation_bottlenecks.py
```

Current output:

```text
datasets/bottleneck_validation_train/summary.json
datasets/bottleneck_validation_train/bottleneck_validation_report.md
datasets/bottleneck_validation_train/boundary_metrics.jsonl
```

This compares real segment boundaries against random non-boundary positions from the same episodes.

Current train result:

```text
real boundaries:   15459
random controls:   15456

prev_system_nav lift:        605.67
curr_system_nav lift:         22.63
changed_action_type lift:      2.00
value_consumed_after:         24.1% of real boundaries
route_shift at real boundary:  0.94 average
memory_strength after boundary: 1.45 average
```

Category breakdown:

```text
surface_navigation real_count: 14862
after_value_entry real_count:    218
browse_to_interact real_count:   164
capability_shift real_count:     124
same_action real_count:           91
```

Interpretation:

- Surface-navigation boundaries are strongly bottleneck-like under offline proxies: they are rare at random positions, dominate real boundaries, and align with route/memory changes.
- Non-surface semantic boundaries are not yet proven as causal bottlenecks. They appear in the data, but their action/text shift advantage over random controls is modest.
- Therefore the current validated claim should be: **surface-transition boundaries are bottleneck-like; semantic/local-objective boundaries remain candidates requiring model-error or intervention validation.**

Next validation should use model behavior:

```text
boundary-aware memory vs full history vs no previous segment summary
boundary-aware route selection vs fixed route
merge/split/random boundary perturbation
error concentration within +/-2 steps of predicted boundaries
```

### Qwen Model-Behavior Validation Protocol

Implemented a model-facing validation harness:

```text
scripts/eval_model_bottleneck_behavior.py
scripts/run_qwen3_bottleneck_validation.sh
```

Target models:

```text
Qwen/Qwen3-VL-8B-Instruct   visual model, current screenshot included
Qwen/Qwen3.5-9B             multimodal model, current screenshot included
```

For each sampled case, evaluate the same next-action prediction under four context interventions:

```text
no_history       goal + current step/screenshot only
segment_summary  goal + previous segment summaries + current step/screenshot
full_history     goal + raw previous action history + current step/screenshot
wrong_summary    goal + unrelated segment summary + current step/screenshot
```

Cases are split into:

```text
real_boundary     current step is a segment start
random_control    current step is a non-boundary position in the same episodes
```

Behavioral bottleneck evidence criterion:

```text
segment_summary improves over no_history at real boundaries
wrong_summary hurts at real boundaries
full_history is a high-cost upper baseline
the same gains are weaker or absent on random controls
```

Expected outputs:

```text
datasets/model_bottleneck_validation/model_behavior_results.jsonl
datasets/model_bottleneck_validation/summary.json
datasets/model_bottleneck_validation/model_behavior_report.md
```

Current runtime status:

```text
Qwen validation now uses uv-managed vLLM environments only; no conda environment is required.
Qwen3-VL env: .venv-qwen3-vllm, vLLM 0.11.0, transformers 4.57.1, torch 2.8.0.
Qwen3.5 env: .venv-qwen35-vllm, vLLM 0.23.0, transformers 5.12.1, torch 2.11.0.
Local checkpoints are present under checkpoints/Qwen3-VL-8B-Instruct and checkpoints/Qwen3.5-9B.
The project .venv still has the repo training stack and should not be used to serve Qwen3/Qwen3.5.
```

Operational notes:

```text
Qwen3-VL image prompts can exceed 4096 decoder tokens, so VL_MAX_MODEL_LEN should be 8192.
Qwen3.5 image prompts can also exceed 4096 tokens, so TEXT_MAX_MODEL_LEN should be 8192 for fair multimodal evaluation.
In this shared GPU container, dynamic vLLM memory profiling can fail when other processes release memory during startup.
Sequential startup avoids most profiling failures while still letting vLLM allocate by gpu_memory_utilization.
For Qwen3-VL all-sample throughput, four independent single-GPU vLLM servers are faster and more stable than one over-aggressive server.
Both Qwen3-VL and Qwen3.5 should be evaluated in non_thinking and thinking modes via chat_template_kwargs={"enable_thinking": false/true}.
```

Validated smoke run:

```text
output: datasets/model_bottleneck_validation_smoke_final_fixedkv
rows: 32
errors: 0
Qwen3-VL parse/type/value: 16/16, 16/16, 16/16
Qwen3.5 parse/type/value: 16/16, 14/16, 14/16
```

Current small-sample interpretation:

```text
The vLLM/uv infrastructure is validated for both target models.
The 4-case smoke is too small to establish bottleneck causality.
It does show that the context-intervention protocol runs cleanly and that Qwen3.5 can be sensitive to segment_summary/wrong_summary at real boundaries.
Next step is a larger run, e.g. MAX_CASES=40 or higher, before drawing bottleneck conclusions.
```

Completed Qwen3-VL all-sample validation:

```text
script: scripts/run_qwen3_vl_overnight_all_samples.sh
execution: four independent Qwen3-VL vLLM servers on GPUs 4/5/6/7, one shard per GPU
output: datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_20260618_4gpu_sharded/merged
rows: 174240
complete paired cases: 43560
errors: 0
thinking modes: non_thinking, thinking
conditions: no_history, segment_summary, full_history, wrong_summary
```

Merged reports:

```text
datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_20260618_4gpu_sharded/merged/model_behavior_results.jsonl
datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_20260618_4gpu_sharded/merged/model_behavior_report.md
datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_20260618_4gpu_sharded/merged/hard_case_analysis/qwen3_vl_8b_hard_case_report.md
datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_20260618_4gpu_sharded/merged/hard_case_analysis/qwen3_vl_8b_hard_cases.jsonl
```

All-sample Qwen3-VL value accuracy:

| mode | case | no_history | segment_summary | full_history | wrong_summary |
|---|---|---:|---:|---:|---:|
| non_thinking | real_boundary | 0.958 | 0.963 | 0.962 | 0.959 |
| non_thinking | random_control | 0.826 | 0.829 | 0.835 | 0.823 |
| thinking | real_boundary | 0.956 | 0.958 | 0.958 | 0.953 |
| thinking | random_control | 0.832 | 0.834 | 0.838 | 0.828 |

Hard-subset counts from the merged all-sample run:

```text
segment_rescue: 284
long_horizon_segment_rescue: 281
memory_specific_segment_over_wrong: 348
segment_regression: 139
wrong_beats_segment: 124
long_horizon_no_history_wrong: 3454
all_conditions_wrong: 3066
```

Current all-sample interpretation:

```text
Qwen3-VL strongly validates the evaluation harness: 174240 requests, parse 1.0, errors 0.
The aggregate bottleneck effect is positive but small because current-screen no_history is already strong.
The main signal is concentrated in hard/long-horizon cases: 281 of 284 segment_rescue cases are long-horizon-tagged.
Correct segment summaries beat wrong summaries more often than the reverse, but the margin is not huge.
Therefore the most defensible claim is: segment memory is not globally necessary for easy current-screen actions, but it is useful for a meaningful hard subset involving carried values, multi-segment progress, and non-obvious swipe/type/system transitions.
This supports using the segmenter as a selective memory/router, not as an always-on replacement for current-screen grounding.
```

Selective long-horizon routing data:

```text
script: scripts/build_long_horizon_routing_data.py
research note: docs/behavior_validated_long_horizon_routing.md
next research method: docs/long_horizon_memory_router_research_method.md
minimal next method: docs/minimal_counterfactual_memory_utility.md
step-by-step research log: docs/long_horizon_memory_research_steps.md
output: datasets/long_horizon_routing_data_qwen3_qwen35
examples: 45560
long_horizon examples: 44000
use_memory positives: 779
```

Generated files:

```text
datasets/long_horizon_routing_data_qwen3_qwen35/routing_examples.jsonl
datasets/long_horizon_routing_data_qwen3_qwen35/routing_report.md
datasets/long_horizon_router_training_summary.md
```

Routing labels from Qwen3-VL all-sample plus Qwen3.5 8192-context validation:

| route | n | rate |
|---|---:|---:|
| use_no_history | 41555 | 0.912 |
| escalate_or_replan | 3197 | 0.070 |
| use_segment_summary | 554 | 0.012 |
| use_full_history | 225 | 0.005 |
| avoid_segment_summary | 29 | 0.001 |

Interpretation for long-horizon strengthening:

```text
The long-horizon route should be selective, not default-on.
Current screen is sufficient for most cases, even many long-horizon-tagged ones.
The valuable positives are small but high-signal: segment_rescue, segment_beats_wrong, and full_history_rescue_only.
Train a router head to predict use_no_history / use_segment_summary / use_full_history / escalate_or_replan / avoid_segment_summary from segment state, carried values, step index, previous segment count, dominant capability, and current action type.
At inference time, default to no_history for current-screen grounding; activate segment_summary only when the router predicts memory need; reserve full_history for rare cases where segment summaries fail but raw history rescues.
Escalate/replan cases should trigger another candidate source or verifier rather than blindly adding more history.
The next research path is candidate-level repair/verifier features, not OCR-first feature expansion.
OCR or visual text can remain optional supporting evidence, but the primary signal to model is whether segment-memory candidates repair no-history candidate errors.
First repair-feature ablation supports this as a verifier path: candidate + repair reaches test precision 0.3922 / recall 0.9524 at threshold 0.90 with 7 regressions, and precision 0.5833 / recall 0.6667 at threshold 0.97 with 3 regressions. This does not solve rare-positive routing yet, but it improves the high-confidence operating point without OCR.
Error mining then shows that counterfactual specificity is the next strongest non-OCR signal: rejecting cases where segment_summary and wrong_summary produce the same candidate raises precision at threshold 0.90 from 0.3922 to 0.4651 without reducing recall, and rejecting same-type segment/wrong candidates reaches precision 0.6400 at threshold 0.90. At threshold 0.99, same-type rejection reaches precision 1.0000 with recall 0.5238.
Training specificity features directly is stronger than using only a post-hoc filter: candidate + repair + specificity reaches AP 0.8318, precision 0.5714 / recall 0.9524 at threshold 0.90, and precision 0.6129 / recall 0.9048 at threshold 0.99. This meets the initial precision >= 0.50 target for memory-specific bottleneck detection without OCR.
Adding instruction task-progress features shows the best next-stage scorer should stay small: specificity + progress reaches AP 0.8443, precision 0.5676 / recall 1.0000 at threshold 0.70, and precision 0.5714 / recall 0.9524 at threshold 0.90. Full candidate + repair + specificity + progress is worse (AP 0.8050), so the method should use two clean tests rather than a broad feature stack: memory-specificity, then instruction-progress compatibility.
Cross-benchmark audit reframes the contribution: the method is a benchmark-agnostic context-intervention protocol, but scorer transfer is not proven yet. GUI-Odyssey and AndroidControl eval both pass structural readiness for segmentation, context interventions, specificity, and progress tests. GUI-360 remains an adapter target rather than a validated transfer result. See `docs/cross_benchmark_memory_router_research_protocol.md`.
On GUI-Odyssey, per-capability thresholding is currently a negative result: sparse memory-positive dev support causes per-capability thresholds to overfit. The selected policy should stay global for now. A dev target of 0.60 gives test precision 0.5250 / recall 1.0000; a dev target of 0.70 gives test precision 0.7778 / recall 0.6667.
At the high-recall GUI-Odyssey operating point, remaining false positives are mostly not wrong-memory specificity failures: 10/16 are unresolved all-condition failures, 5/16 are true segment regressions, and 1/16 is summary-insufficient/full-history-only. The next GUI-Odyssey component should therefore be an unresolved/replan detector layered after specificity + progress, not another memory feature family.
Full-history consistency gives the first candidate-validity verifier: at threshold 0.70, requiring segment_summary and full_history to share action type improves test precision from 0.5676 to 0.6207 and reduces regressions from 5 to 2, while recall drops from 1.0000 to 0.8571. This supports a three-stage policy: propose memory with specificity+progress, verify with full-history consistency, then accept memory / reject to no_history / escalate to full_history or replan.
Multi-route policy evaluation shows the current best prospective GUI-Odyssey policy should remain selective: use specificity+progress at threshold 0.70, commit segment memory only when full-history consistency supports the segment candidate, otherwise return to no_history. Always full_history has higher static action accuracy on this split (0.9221) but doubles context cost and is not selective; full-history fallback after segment rejection recovers only one test case. A separate hard-state detector is needed before full_history/replan becomes a real route.
The verifier should now be treated as an agent, not a scalar classifier. The multi-agent framing is: Local Context Agent, Segment Memory Agent, Full History Agent, Distractor Memory Probe Agent, Verifier Agent, and Execution Coordinator. The verifier receives structured candidate packets and outputs a JSON route decision with reason codes, avoiding the OOD low-level-action generation issue from earlier two-agent experiments. See `docs/multi_agent_memory_router_framework.md`.
```

Recommended run:

```bash
STAMP=20260618_overnight_qwen3vl_all_samples \
bash scripts/run_qwen3_vl_overnight_all_samples.sh
```

For fastest all-sample Qwen3-VL reruns, split into four shards and run four single-GPU vLLM servers with:

```text
CASE_SHARD_INDEX=0..3
CASE_SHARD_COUNT=4
VL_CUDA_VISIBLE_DEVICES=4/5/6/7
VL_PORT=8000/8001/8002/8003
REQUEST_WORKERS=64
VL_EXTRA_ARGS='--max-num-seqs 128 --max-num-batched-tokens 65536'
```

## Evaluation Plan

### Intrinsic Metrics

- Boundary precision/recall/F1 against LLM-adjudicated or human labels.
- Segment purity: one app/surface/local objective per segment.
- Segment completeness: segment has a meaningful output or state transition.
- Capability accuracy: capability labels match action/function needs.
- Memory-needed precision: history is used only when necessary.

### Downstream Metrics

- Semantic action accuracy.
- Type/action accuracy.
- Stale-history regression rate.
- Exact text carry accuracy.
- Target grounding accuracy for click/long_press.
- Non-click recall for scroll/type/system_button/terminate.

### Critical Ablations

```text
best single prompt
+ capability router only
+ segment summaries
+ selective memory
+ v2/v3/v4 candidate pool
+ verifier commit
```

## Implementation Milestones

### Milestone 1: Data Adapters And Weak Segmenter

- Implement canonical schema adapters for GUI-Odyssey and AndroidControl first.
- Add GUI-360 adapter once raw data path is confirmed.
- Generate rule-based segment JSONL.
- Visualize 20 long trajectories with proposed boundaries.

### Milestone 2: Boundary Adjudication Set

- Sample across categories, lengths, action types, and benchmarks.
- Ask LLM/human to approve or edit boundaries.
- Build a small gold set for boundary and segment quality.

### Milestone 3: Segmenter Model

- Train a text-only baseline first.
- Add screenshot/OCR/a11y features later.
- Predict boundary, capability, memory-needed, and segment summary.

### Milestone 4: Verifier Integration

- Use capability labels to choose verification checks.
- Use segment memory to restrict history retrieval.
- Compare against v2/v3/v4 routes without segmentation.

## Initial Design Decision

Start with a hybrid method, not a pure neural method.

Reason:

- Navigation and terminal actions give high-precision boundaries.
- Text fields give useful but noisy local intent.
- Screenshot/a11y changes are needed for sparse benchmarks.
- A verifier can reject over-segmentation and stale-history use.

The first version should therefore be:

```text
rules propose -> LLM/verifier adjudicates -> train lightweight segmenter -> integrate with action verifier
```

## Open Questions

1. Should segment summaries be generated from gold trajectories only, or also from predicted trajectories during rollout?
2. How much visual information is needed for boundary detection beyond text/action fields?
3. Can a single capability taxonomy cover mobile and desktop GUI tasks without becoming too coarse?
4. How should we score memory-needed labels when the action can be solved from both current screen and history?
5. Should the verifier be trained jointly with the segmenter or separately as a candidate-ranking model?

## Immediate Next Step

Implement a small offline prototype:

```text
scripts/analyze_trajectory_segments.py
```

It should:

1. Load GUI-Odyssey converted or raw annotations.
2. Load AndroidControl JSONL.
3. Convert both to canonical schema.
4. Produce weak macro segments and capability labels.
5. Write JSONL and a markdown report with examples.

Once this works on GUI-Odyssey and AndroidControl, add GUI-360 support and start LLM adjudication.