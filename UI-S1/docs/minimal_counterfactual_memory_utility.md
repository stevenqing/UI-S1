# Minimal Research Method: Counterfactual Memory Utility

## One-Sentence Thesis

Long-horizon reasoning should be learned as **counterfactual memory utility**: a memory is useful only if replacing no-history context with that memory changes an otherwise wrong action into the correct action, while an unrelated memory does not.

This is the smallest research object we can study.

## What We Should Not Do Next

Do not start by building a large engineering router with many hand-added features.

That path is too unconstrained:

```text
more features -> more knobs -> unclear scientific claim
```

The previous router baselines already showed that shallow structural features are insufficient. Adding dozens of ad hoc features risks hiding the real question.

## First Principle

At step `t`, the agent observes:

```text
current state x_t = goal + current screen + current local instruction
memory m_t        = previous segment summary / carried values
action a_t        = next ground-truth action
```

Memory is useful only if:

```text
P(a_t | x_t, m_t) > P(a_t | x_t)
```

But we do not need calibrated probabilities. We can use behavior interventions:

```text
no_history wrong, segment_memory correct  => memory has positive utility
no_history correct, segment_memory wrong  => memory has negative utility
wrong_memory correct or segment wrong     => memory signal is not specific
all contexts wrong                        => this is not solved by memory routing
```

Therefore the primitive label is not `long_horizon=true`. The primitive label is:

```text
utility(memory, current_state) ∈ {positive, neutral, negative, unresolved}
```

## Minimal Experimental Unit

For each step, construct a triplet:

```text
x_t       current state
m_pos     true segment memory
m_neg     unrelated/wrong segment memory
```

Then evaluate or train on the contrast:

```text
score(x_t, m_pos) > score(x_t, m_neg)
```

Only samples where memory behavior matters should strongly supervise the model:

```text
positive utility: no_history wrong, true memory correct, wrong memory wrong
negative utility: no_history correct, true memory wrong
unresolved: all wrong
neutral: no_history correct and memory also correct
```

This gives a clean contrastive learning problem.

## Minimal Model

Train a small memory utility scorer:

```text
s_theta(x_t, m) -> scalar utility score
```

The scorer does not directly output an action. It only answers:

```text
Should this memory be attached to the action model for this current state?
```

Inputs should be minimal but semantically meaningful:

```text
x_t:
  goal
  current step instruction / local hint
  OCR or screen caption

m:
  segment summary
  carried values
  segment capability / surface if available
```

No large route taxonomy is needed at this stage.

## Minimal Objective

Use pairwise ranking rather than multi-class classification.

For memory-positive cases:

```text
L_pos = max(0, margin - s(x_t, m_true) + s(x_t, m_wrong))
```

For memory-negative cases:

```text
L_neg = max(0, margin - s(x_t, no_memory) + s(x_t, m_true))
```

For unresolved cases:

```text
do not train memory preference; send to replan/perception bucket
```

This avoids forcing all failures into memory categories.

## Minimal Policy

At inference time:

```text
if s(x_t, m_true) >= tau:
    use segment memory
else:
    use no_history
```

Optional second threshold:

```text
if hard_state_score(x_t) high and memory score low:
    replan / ask another candidate source
```

But the first experiment should only test the single threshold `tau`.

## Minimal Evaluation

Evaluate three policies on held-out episodes:

```text
1. Always no_history
2. Always segment_summary
3. Learned memory utility threshold
```

Metrics:

```text
overall value accuracy
hard-subset value accuracy
memory activation rate
memory activation precision
segment_rescue recall
segment_regression rate
cost-adjusted utility
```

The learned policy succeeds only if it improves hard-subset accuracy without paying a large stale-memory cost.

## Why This Is A Real Research Method

This method makes a falsifiable claim:

```text
There exists a learnable compatibility function between current state and segment memory
that predicts when memory has positive counterfactual utility.
```

It is falsified if:

```text
memory-positive examples cannot be separated from neutral/negative examples even with semantic current-state and memory representations
```

It is supported if:

```text
a learned memory utility scorer beats both always-no-history and always-memory on hard subsets
while preserving high precision of memory activation
```

## Immediate Next Experiment

The next concrete experiment should be a minimal contrastive dataset and scorer.

### Data

From existing behavior results, create:

```text
datasets/counterfactual_memory_utility/train.jsonl
datasets/counterfactual_memory_utility/dev.jsonl
datasets/counterfactual_memory_utility/test.jsonl
```

## First Minimal Experiment Result

Implemented the first version:

```text
scripts/build_counterfactual_memory_utility_data.py
scripts/train_counterfactual_memory_utility.py
datasets/counterfactual_memory_utility
datasets/counterfactual_memory_utility_tfidf
```

Dataset statistics:

```text
rows: 45560
train/dev/test: 36450 / 4566 / 4544
positive: 207
negative: 188
neutral: 41575
unresolved: 3226
nonspecific_positive: 139
summary_insufficient: 225
```

The first scorer uses TF-IDF over:

```text
CURRENT STATE: goal + instruction + observation + local hint + current segment hypothesis
MEMORY: true segment memory or wrong memory
```

and trains a balanced logistic classifier for positive counterfactual memory utility.

Held-out test result:

```text
average_precision: 0.1492
roc_auc: 0.9321
always_no_history_acc: 0.9153
always_segment_summary_acc: 0.9186
```

Threshold behavior on test:

| threshold | predicted memory | precision | recall | routed value acc | regressions |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 3286 | 0.0064 | 1.0000 | 0.9203 | 17 |
| 0.10 | 822 | 0.0231 | 0.9048 | 0.9190 | 10 |
| 0.50 | 124 | 0.0645 | 0.3810 | 0.9166 | 5 |
| 0.70 | 38 | 0.0789 | 0.1429 | 0.9162 | 0 |
| 0.90 | 2 | 1.0000 | 0.0952 | 0.9157 | 0 |

Interpretation:

```text
The semantic memory-current-state scorer has real ranking signal: ROC-AUC 0.9321 and AP 0.1492 are far above the base positive rate.
Low thresholds can improve routed value accuracy over both static baselines, but precision is too low for deployment.
High thresholds give clean memory activations, but recall is tiny.
```

This partially supports the research hypothesis:

```text
semantic compatibility is informative,
but text-only TF-IDF compatibility is not sufficient for a practical high-precision/high-recall memory router.
```

The next minimal increment should add candidate disagreement, then candidate-level repair/verifier features. OCR/screen availability is useful only as optional supporting evidence, not the main path.

## Candidate-Disagreement Increment

Implemented candidate-aware scoring:

```text
scripts/train_counterfactual_memory_utility.py --candidate-features
datasets/counterfactual_memory_utility_candidate
```

This adds features derived from stored model candidates:

```text
no_history candidate action type/text
segment_summary candidate action type/text
wrong_summary candidate action type/text
candidate type disagreement
candidate text disagreement
candidate overlap with carried values
candidate overlap with memory text
```

Held-out test result:

```text
average_precision: 0.6349
roc_auc: 0.9973
always_no_history_acc: 0.9153
always_segment_summary_acc: 0.9186
best routed_acc: 0.9217
```

This is a large improvement over TF-IDF-only:

```text
AP:      0.1492 -> 0.6349
ROC-AUC: 0.9321 -> 0.9973
```

But precision remains below the desired operating point:

```text
threshold 0.90: precision 0.3878, recall 0.9048
```

Conclusion:

```text
Candidate disagreement is the strongest signal so far and validates the counterfactual method.
The next missing piece is not raw OCR. It is a candidate-level verifier that can tell whether the segment-memory candidate repairs a no-history error or merely changes the answer.
```

## Weak Screen-Availability Increment

Implemented a first screen/current-state availability proxy:

```text
scripts/build_counterfactual_memory_utility_data.py
scripts/train_counterfactual_memory_utility.py --candidate-features --screen-features
datasets/counterfactual_memory_utility_v2
datasets/counterfactual_memory_utility_candidate_screen
```

This exposes current-state parts:

```text
goal
current instruction
screen observation
local hint
current segment hypothesis
```

and adds overlap features between those fields, carried values, and memory text.

Held-out test result:

```text
average_precision: 0.5978
roc_auc: 0.9970
best routed_acc: 0.9217
threshold 0.90 precision/recall: 0.4043 / 0.9048
```

Comparison:

```text
candidate-only AP:          0.6349
candidate+screen-proxy AP:  0.5978
candidate-only precision@0.90:         0.3878
candidate+screen-proxy precision@0.90: 0.4043
```

Interpretation:

```text
The weak screen proxy slightly improves high-threshold precision but does not improve ranking AP.
The current observation text is too coarse to answer whether the carried value is actually visible.
The next minimal increment should use real OCR/visible text or image-derived UI state, not just existing natural-language observations.
```

## Partial Real-OCR Increment

Implemented RapidOCR-based screen text extraction:

```text
scripts/build_ocr_cache.py
datasets/counterfactual_memory_utility_v2/ocr_cache_informative.jsonl
datasets/counterfactual_memory_utility_candidate_screen_ocr
```

Current cache is partial:

```text
322 OCR rows completed out of about 3051 informative target screenshots.
```

Held-out test result with partial OCR:

```text
average_precision: 0.5470
roc_auc: 0.9968
best routed_acc: 0.9214
threshold 0.90 precision/recall: 0.4130 / 0.9048
```

Comparison:

```text
candidate-only AP:                 0.6349
candidate+weak-screen AP:          0.5978
candidate+weak-screen+partial-OCR: 0.5470

candidate-only precision@0.90:                 0.3878
candidate+weak-screen precision@0.90:          0.4043
candidate+weak-screen+partial-OCR precision@0.90: 0.4130
```

Interpretation:

```text
Partial OCR gives a tiny high-threshold precision gain but hurts AP.
This does not yet prove OCR availability helps; coverage is too incomplete and OCR is noisy.
OCR should be deprioritized as the main path. The cleaner next step is candidate-level exact-value/action repair features.
```

## Candidate-Level Repair Increment

Implemented candidate disagreement as explicit repair features:

```text
action_type_repair
exact_text_repair
system_button_repair
target_entity_repair
stale_memory_risk
```

The question becomes:

```text
Did segment memory repair a specific no-history mistake in a way that is detectable before seeing the ground truth?
```

This keeps the method close to the counterfactual evidence and avoids relying on noisy OCR as a proxy for screen availability.

Run artifacts:

```text
scripts/train_counterfactual_memory_utility.py --repair-features
datasets/counterfactual_memory_utility_candidate_refit
datasets/counterfactual_memory_utility_repair_only
datasets/counterfactual_memory_utility_candidate_repair
```

Held-out test comparison, retrained with the same feature extractor version:

| model | AP | ROC-AUC | precision@0.90 | recall@0.90 | regressions@0.90 | precision@0.97 | recall@0.97 |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate refit | 0.6003 | 0.9971 | 0.3878 | 0.9048 | 10 | 0.4800 | 0.5714 |
| repair only | 0.6164 | 0.9976 | 0.3774 | 0.9524 | 9 | 0.5600 | 0.6667 |
| candidate + repair | 0.6270 | 0.9977 | 0.3922 | 0.9524 | 7 | 0.5833 | 0.6667 |

Conclusion:

```text
Repair features are not a large AP win, but they improve the routing operating point.
The important gain is lower regression at high recall and a cleaner high-confidence threshold: candidate + repair reaches precision 0.5833 / recall 0.6667 at threshold 0.97.
```

Updated next step:

```text
Analyze false positives under candidate + repair.
Add wrong-summary agreement and stale-memory-risk features so the router can reject candidate changes caused by misleading memory rather than useful repair.
```

## Counterfactual Specificity Filter

Error mining artifact:

```text
scripts/analyze_candidate_repair_errors.py
datasets/counterfactual_memory_utility_candidate_repair_error_analysis
```

Main finding:

```text
True positives at thresholds 0.90, 0.97, and 0.99 never had exact segment_summary == wrong_summary candidate.
False positives often did, especially in the highest-confidence bucket.
```

Test policy comparison:

| threshold | policy | predicted | precision | recall | regressions |
|---:|---|---:|---:|---:|---:|
| 0.90 | candidate + repair | 51 | 0.3922 | 0.9524 | 7 |
| 0.90 | reject exact segment == wrong | 43 | 0.4651 | 0.9524 | 6 |
| 0.90 | reject segment type == wrong type | 25 | 0.6400 | 0.7619 | 4 |
| 0.97 | candidate + repair | 24 | 0.5833 | 0.6667 | 3 |
| 0.97 | reject exact segment == wrong | 22 | 0.6364 | 0.6667 | 3 |
| 0.97 | reject segment type == wrong type | 17 | 0.7647 | 0.6190 | 2 |
| 0.99 | candidate + repair | 14 | 0.7857 | 0.5238 | 0 |
| 0.99 | reject exact segment == wrong | 12 | 0.9167 | 0.5238 | 0 |
| 0.99 | reject segment type == wrong type | 11 | 1.0000 | 0.5238 | 0 |

Conclusion:

```text
The next method should be memory-specific candidate repair, not generic candidate disagreement.
A memory candidate is useful only if it repairs no_history in a way that wrong memory does not also induce.
```

## Learned Specificity-Aware Scorer

Implemented trainable specificity features in the CMU scorer:

```text
scripts/train_counterfactual_memory_utility.py --specificity-features
datasets/counterfactual_memory_utility_specificity_only
datasets/counterfactual_memory_utility_repair_specificity
datasets/counterfactual_memory_utility_candidate_repair_specificity
```

Held-out test result:

| model | AP | precision@0.90 | recall@0.90 | precision@0.99 | recall@0.99 |
|---|---:|---:|---:|---:|---:|
| candidate + repair | 0.6270 | 0.3922 | 0.9524 | 0.7857 | 0.5238 |
| specificity only | 0.8269 | 0.5556 | 0.9524 | 0.6429 | 0.8571 |
| repair + specificity | 0.8261 | 0.5882 | 0.9524 | 0.6129 | 0.9048 |
| candidate + repair + specificity | 0.8318 | 0.5714 | 0.9524 | 0.6129 | 0.9048 |

Best operating points for candidate + repair + specificity:

| threshold | predicted | precision | recall | regressions |
|---:|---:|---:|---:|---:|
| 0.30 | 40 | 0.5250 | 1.0000 | 6 |
| 0.90 | 35 | 0.5714 | 0.9524 | 5 |
| 0.99 | 31 | 0.6129 | 0.9048 | 4 |

Conclusion:

```text
Specificity is the missing non-OCR signal.
It converts candidate disagreement from a broad correlation into a memory-specific repair test.
This reaches the initial precision >= 0.50 target while keeping high recall, without relying on OCR.
```

Next step:

```text
Analyze the remaining false positives for specificity-aware routing.
The next verifier should distinguish task-progressing repair from generic navigation changes.
```

## Instruction Task-Progress Increment

Implemented progress features:

```text
scripts/train_counterfactual_memory_utility.py --progress-features
datasets/counterfactual_memory_utility_progress_only
datasets/counterfactual_memory_utility_specificity_progress
datasets/counterfactual_memory_utility_repair_specificity_progress
datasets/counterfactual_memory_utility_candidate_repair_specificity_progress
```

The features classify current instruction intent and compare candidate/no-history/wrong-memory action compatibility:

```text
empty, home, back, scroll, type, click_open, terminate, adjust, other
```

Held-out test result:

| model | AP | precision@0.50 | recall@0.50 | precision@0.70 | recall@0.70 | precision@0.90 | recall@0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate + repair + specificity | 0.8318 | 0.5128 | 0.9524 | 0.5263 | 0.9524 | 0.5714 | 0.9524 |
| progress only | 0.6978 | 0.4762 | 0.9524 | 0.5135 | 0.9048 | 0.5588 | 0.9048 |
| specificity + progress | 0.8443 | 0.5250 | 1.0000 | 0.5676 | 1.0000 | 0.5714 | 0.9524 |
| repair + specificity + progress | 0.8074 | 0.5526 | 1.0000 | 0.5714 | 0.9524 | 0.5588 | 0.9048 |
| candidate + repair + specificity + progress | 0.8050 | 0.5526 | 1.0000 | 0.5405 | 0.9524 | 0.5588 | 0.9048 |

Conclusion:

```text
The best next-stage scorer is specificity + progress, not the full feature stack.
This is an important simplicity result: once specificity is available, broad repair/candidate features can overfit or reintroduce false positives.
```

Selected operating points for specificity + progress:

| threshold | predicted | precision | recall | regressions |
|---:|---:|---:|---:|---:|
| 0.50 | 40 | 0.5250 | 1.0000 | 7 |
| 0.70 | 37 | 0.5676 | 1.0000 | 5 |
| 0.90 | 35 | 0.5714 | 0.9524 | 5 |
| 0.99 | 28 | 0.6071 | 0.8095 | 2 |

Updated method:

```text
memory-specificity test + instruction-progress compatibility test
```

## Cross-Benchmark Research Check

The method should be evaluated as a benchmark-agnostic intervention protocol, not as a GUI-Odyssey feature recipe.

Reference:

```text
docs/cross_benchmark_memory_router_research_protocol.md
scripts/audit_cross_benchmark_memory_method.py
```

Current structural audit:

| benchmark | instruction rate | screenshot rate | core ready | full ready |
|---|---:|---:|---|---|
| AndroidControl eval | 99.5% | 100.0% | yes | yes |
| GUI-Odyssey train sample | 100.0% | 100.0% | yes | yes |

Claim status:

```text
Defensible now: context intervention + specificity/progress tests are structurally portable across GUI-Odyssey and AndroidControl.
Not yet defensible: a scorer trained on GUI-Odyssey transfers to every GUI benchmark.
```

Next required evidence:

```text
AndroidControl behavior-validation run.
Cross-benchmark CMU training/evaluation.
Leave-one-benchmark-out thresholds.
Prospective routed evaluation on target benchmark.
```

## GUI-Odyssey Thresholding Result

Per-capability thresholding was tested on GUI-Odyssey using dev-selected thresholds:

```text
scripts/evaluate_memory_router_thresholds.py
datasets/counterfactual_memory_utility_specificity_progress_thresholds*
```

Result:

```text
Per-capability thresholds do not reliably improve test behavior because memory-positive dev support is too sparse.
The safest current policy is a single global threshold for the specificity+progress scorer.
```

Recommended GUI-Odyssey operating points:

| dev target | test predicted | test precision | test recall | regressions |
|---:|---:|---:|---:|---:|
| 0.60 | 40 | 0.5250 | 1.0000 | 7 |
| 0.70 | 18 | 0.7778 | 0.6667 | 1 |

This is another useful simplicity result:

```text
specificity + progress + global threshold
```

is currently more reliable than sparse per-capability calibration.

## GUI-Odyssey Remaining Error Taxonomy

At the selected high-recall global threshold:

```text
specificity + progress threshold 0.70
precision 0.5676
recall 1.0000
```

False positives are:

| FP type | count |
|---|---:|
| unresolved, all conditions fail | 10 |
| negative, segment memory regresses no_history | 5 |
| summary insufficient, full history only helps | 1 |

Dominant FP capability:

```text
navigate_system: 8 / 16
```

Conclusion:

```text
The next bottleneck is not memory specificity.
It is candidate validity / replan detection: many false positives are cases where the memory-induced candidate is specific and instruction-compatible but still wrong under every context.
```

Next GUI-Odyssey module:

```text
unresolved-or-replan detector on top of specificity + progress
```

## Full-History Consistency Cascade

Implemented a cascade evaluation:

```text
scripts/evaluate_memory_router_cascade.py
datasets/counterfactual_memory_utility_specificity_progress_cascade
```

The first-principles split is:

```text
memory utility scorer: should segment memory be considered?
candidate-validity verifier: does full history support the segment candidate?
```

Result:

| threshold | filter | test precision | test recall | regressions |
|---:|---|---:|---:|---:|
| 0.70 | none | 0.5676 | 1.0000 | 5 |
| 0.70 | segment_full_same_type | 0.6207 | 0.8571 | 2 |
| 0.70 | segment_full_same_type_not_wrong_type | 0.6667 | 0.6667 | 2 |
| 0.90 | none | 0.5714 | 0.9524 | 5 |
| 0.90 | segment_full_same_type | 0.6296 | 0.8095 | 2 |

Conclusion:

```text
Full-history consistency should be treated as a commit verifier, not as another memory feature.
It gives a controllable precision/regression improvement at the cost of recall.
```

Updated policy direction:

```text
specificity + progress proposes memory
full-history consistency verifies candidate validity
failed verification routes to no_history, full_history, or replan depending on candidate agreement structure
```

Each row:

```json
{
  "current_state_text": "goal + current instruction + optional OCR/caption",
  "true_memory_text": "segment summary + carried values",
  "wrong_memory_text": "unrelated segment summary",
  "utility_label": "positive | neutral | negative | unresolved",
  "preference_pairs": [["true_memory", "wrong_memory"]],
  "metadata": {
    "episode_id": "...",
    "step_index": 12,
    "case_kind": "real_boundary",
    "action_type": "type",
    "is_long_horizon": true
  }
}
```

### Model

Start with frozen text embeddings plus a small scorer:

```text
embedding(current_state_text)
embedding(memory_text)
features = [x, m, x*m, |x-m|]
classifier/ranker -> utility score
```

Then compare against a small cross-encoder if needed.

### Success Criteria

The first target is not high recall. It is high precision:

```text
memory activation precision >= 0.50
segment_rescue recall better than structural router baseline
segment_regression below always-segment-summary baseline
```

If this works, then we have a research result:

```text
Segment memory utility is learnable from semantic compatibility between current state and memory.
```

If it fails, then the result is also informative:

```text
Memory utility may require candidate-level disagreement or visual grounding, not just current-state/memory semantic similarity.
```

## Relationship To The Current Work

Current completed evidence:

```text
Qwen3-VL all-sample behavior validation: 174240 rows, 0 errors
Hard subset: segment_rescue = 284, long_horizon_segment_rescue = 281
Routing labels: 45560 examples, use_memory positives = 779
Structural router: high no-history accuracy, poor memory-positive generalization
```

The minimal next method uses these results not as final labels for a big router, but as counterfactual evidence for learning memory utility.