# Cross-Benchmark Memory Router Research Protocol

## Core Question

The current method should be judged by two standards:

```text
1. Is it a research method, not just a set of benchmark-specific features?
2. Can the method be instantiated across GUI benchmarks with different schemas and domains?
```

The answer is conditional:

```text
Yes for the core method: context intervention -> candidate specificity -> instruction-progress compatibility.
Not yet fully proven for all benchmarks: GUI-Odyssey is validated; AndroidControl is structurally ready; GUI-360 needs a canonical adapter and behavior run.
```

## What Makes It A Research Method

The method is not tied to OCR, a particular app, or a handcrafted long-horizon rule. It defines a causal object:

```text
Does memory change the model's next-action candidate in a correct and memory-specific way?
```

The method has four benchmark-independent operations:

```text
1. Canonicalize trajectory steps into goal, screenshot, action, instruction, and optional local observation.
2. Build segment memory from previous macro segments and carried values.
3. Intervene on context: no_history, segment_summary, full_history, wrong_summary.
4. Learn when the segment-memory candidate is specific and task-progressing relative to no_history and wrong_summary candidates.
```

This makes the unit of analysis a counterfactual behavior vector, not a benchmark label:

```text
no_history wrong
segment_summary correct
wrong_summary wrong
```

That vector means memory-specific rescue regardless of whether the source is GUI-Odyssey, AndroidControl, or GUI-360.

## Cross-Benchmark Invariants

A benchmark can support the method if it exposes these invariants:

| requirement | needed for | benchmark-specific? |
|---|---|---|
| task goal | all context prompts | no |
| screenshot/current state | next-action prediction | no |
| ground-truth next action | behavior validation | no |
| action schema | candidate comparison | mostly no, requires normalization |
| trajectory order | segmentation and memory | no |
| previous segment summaries | memory intervention | generated, not native |
| wrong memory from other episodes | specificity intervention | generated, not native |
| current step instruction | progress compatibility | sometimes native, sometimes adapter-derived |

The strongest part of the method is that wrong memory is generated from the same canonical segment pool. That makes specificity a portable causal test:

```text
If true memory and wrong memory induce the same candidate, the candidate change is not memory-specific.
```

## Current Cross-Benchmark Audit

Implemented audit script:

```text
scripts/audit_cross_benchmark_memory_method.py
```

Adapter fix:

```text
scripts/analyze_trajectory_segments.py now preserves AndroidControl step_instruction in canonical text_fields.instruction.
```

Temporary audit inputs:

```text
datasets/segmentation_train/gui_odyssey_segments.jsonl
tmp_cross_benchmark_segments/android_eval_std/android_control_segments.jsonl
```

Audit result:

| benchmark | episodes | steps | instruction rate | screenshot rate | segmentation | interventions | specificity | progress | core ready | full ready |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| AndroidControl eval | 200 | 1067 | 99.5% | 100.0% | yes | yes | yes | yes | yes | yes |
| GUI-Odyssey train sample | 500 | 7705 | 100.0% | 100.0% | yes | yes | yes | yes | yes | yes |

Interpretation:

```text
The method is structurally portable from GUI-Odyssey to AndroidControl.
AndroidControl is not yet behavior-validated with Qwen under the four interventions, but the data supports the same experiment.
```

## GUI-360 Status

GUI-360 is promising but not yet audited through the same canonical adapter.

Known facts from existing docs and scripts:

```text
- GUI-360 has trajectories, screenshots, and normalized GUI actions.
- Existing train/eval scripts use GUI-360 JSONL-style action prediction data.
- Current segmentation adapter supports GUI-Odyssey and AndroidControl, not GUI-360 yet.
```

Therefore:

```text
GUI-360 does not falsify cross-benchmark portability.
It is the next adapter target.
```

The right next step is not to claim transfer on GUI-360. It is to implement the adapter and run the same audit and intervention protocol.

## What Would Count As Cross-Benchmark Evidence

### Level 0: Structural Portability

A benchmark passes if canonicalized episodes support:

```text
goal + screenshot + ground-truth action + trajectory order + normalized action schema
```

Current status:

```text
GUI-Odyssey: pass
AndroidControl: pass
GUI-360: pending adapter
```

### Level 1: Intervention Portability

Run the same model under:

```text
no_history
segment_summary
full_history
wrong_summary
```

Report behavior-vector distributions per benchmark:

```text
current-screen sufficient
segment-memory rescue
full-history rescue
wrong-memory nonspecific success
segment regression
all-condition failure
```

This tests whether memory bottlenecks exist across benchmarks.

### Level 2: Feature Portability

Train on one benchmark and test on another:

```text
train GUI-Odyssey -> test AndroidControl
train AndroidControl -> test GUI-Odyssey
train mixed -> leave-one-benchmark-out
```

Use the same feature families:

```text
specificity features
progress features
optional repair/candidate features
```

A real cross-benchmark result should report:

```text
memory-positive AP
precision/recall at fixed thresholds
threshold selected on source dev, evaluated unchanged on target test
per-capability breakdown
```

### Level 3: Prospective Transfer

Actually route contexts on held-out target benchmark examples and compare:

```text
always no_history
always segment_summary
always full_history
learned router
oracle best context
```

The method is useful if it improves hard-subset memory rescue without increasing stale-memory regressions.

## Why The Current Best Signal Is Research-Relevant

The current best non-OCR model is:

```text
specificity + progress
```

It uses only portable objects:

```text
candidate under no_history
candidate under segment_summary
candidate under wrong_summary
current instruction intent
normalized action type / value / button / coordinate shape
```

It does not require:

```text
GUI-Odyssey categories
OCR text
app-specific rules
benchmark-specific action templates beyond normalization
```

This is why it is a plausible cross-benchmark method.

## Current Limitations

1. Existing positive evidence is still mostly GUI-Odyssey behavior validation.
2. AndroidControl is structurally ready but needs a behavior-intervention run.
3. GUI-360 needs a canonical adapter before the method can be tested there.
4. Progress features depend on step-level instruction availability; when absent, we need a goal/action-derived intent proxy.
5. The learned thresholds may not transfer directly; leave-one-benchmark-out thresholding is required.

## Next Experiments

1. Implement GUI-360 canonical adapter.
2. Run AndroidControl behavior validation under the four context interventions.
3. Build AndroidControl CMU rows with the same script.
4. Train/evaluate cross-benchmark splits:

```text
GUI-Odyssey -> AndroidControl
AndroidControl -> GUI-Odyssey
mixed train -> held-out benchmark
```

5. Report both:

```text
method existence: do memory-specific bottlenecks appear across benchmarks?
method transfer: does a learned specificity/progress scorer generalize across benchmarks?
```

## Research Claim To Use Carefully

The current defensible claim is:

```text
Counterfactual context intervention is a benchmark-agnostic way to discover memory-specific action bottlenecks.
Specificity and instruction-progress compatibility are portable candidate-level tests for those bottlenecks.
```

The claim we should not make yet is:

```text
A scorer trained on GUI-Odyssey already transfers to every GUI benchmark.
```

That requires the Level 2 and Level 3 experiments above.
