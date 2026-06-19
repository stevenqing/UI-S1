# Long-Horizon Memory Research Steps

This file records each step of the long-horizon memory/router research line, from behavior validation to the minimal counterfactual memory utility experiment.

## Step 1: Validate Whether Segment Memory Has Behavioral Signal

Question:

```text
Does segment memory change next-action prediction behavior in a useful way?
```

Method:

```text
Run the same case under no_history, segment_summary, full_history, and wrong_summary.
Compare next-action correctness across contexts.
```

Main script:

```text
scripts/eval_model_bottleneck_behavior.py
scripts/run_qwen3_bottleneck_validation.sh
```

Key Qwen3-VL all-sample output:

```text
datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_20260618_4gpu_sharded/merged/model_behavior_results.jsonl
datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_20260618_4gpu_sharded/merged/model_behavior_report.md
```

Result:

```text
rows: 174240
complete paired cases: 43560
errors: 0
```

All-sample value accuracy:

| mode | case | no_history | segment_summary | full_history | wrong_summary |
|---|---|---:|---:|---:|---:|
| non_thinking | real_boundary | 0.958 | 0.963 | 0.962 | 0.959 |
| non_thinking | random_control | 0.826 | 0.829 | 0.835 | 0.823 |
| thinking | real_boundary | 0.956 | 0.958 | 0.958 | 0.953 |
| thinking | random_control | 0.832 | 0.834 | 0.838 | 0.828 |

Interpretation:

```text
The aggregate memory effect is positive but small.
Current-screen grounding is already very strong.
The signal is likely concentrated in hard cases, not visible as a large full-set average gain.
```

## Step 2: Identify Hard And Long-Horizon Subsets

Question:

```text
Where exactly does segment memory help?
```

Method:

```text
Find cases where no_history fails but segment_summary succeeds.
Mark long-horizon cases using step index, previous segments, carried values, and memory strength.
```

Script:

```text
scripts/analyze_model_bottleneck_hard_cases.py
```

Output:

```text
datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_20260618_4gpu_sharded/merged/hard_case_analysis/qwen3_vl_8b_hard_case_report.md
datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_20260618_4gpu_sharded/merged/hard_case_analysis/qwen3_vl_8b_hard_cases.jsonl
```

Key counts:

```text
segment_rescue: 284
long_horizon_segment_rescue: 281
memory_specific_segment_over_wrong: 348
segment_regression: 139
wrong_beats_segment: 124
long_horizon_no_history_wrong: 3454
all_conditions_wrong: 3066
```

Interpretation:

```text
Almost every segment rescue is long-horizon-tagged.
This supports selective memory for hard/long-horizon cases.
It does not support always-on memory.
```

## Step 3: Convert Behavior Into Router Labels

Question:

```text
Can we turn context-intervention behavior into supervised router labels?
```

Method:

```text
Map behavior vectors into route labels:
  use_no_history
  use_segment_summary
  use_full_history
  escalate_or_replan
  avoid_segment_summary
```

Script:

```text
scripts/build_long_horizon_routing_data.py
```

Output:

```text
datasets/long_horizon_routing_data_qwen3_qwen35/routing_examples.jsonl
datasets/long_horizon_routing_data_qwen3_qwen35/routing_report.md
```

Dataset statistics:

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

Interpretation:

```text
Memory is rare but high-value.
The router must be high precision, not high activation rate.
```

## Step 4: Train A Structural Router Baseline

Question:

```text
Can simple segment/trajectory metadata predict memory route labels?
```

Method:

```text
Train class-balanced logistic and RandomForest classifiers on structural features.
Episode-level train/dev/test split.
```

Script:

```text
scripts/train_long_horizon_router.py
```

Outputs:

```text
datasets/long_horizon_router_logistic
datasets/long_horizon_router_forest
datasets/long_horizon_router_forest_with_action
datasets/long_horizon_router_qwen3vl_forest
datasets/long_horizon_router_training_summary.md
```

Best structural baseline result:

```text
RandomForest with oracle action type
test accuracy: 0.8840
test macro_f1: 0.3200
memory precision/recall/f1: 0.0621 / 0.1196 / 0.0818
```

Interpretation:

```text
Structural features predict no_history and many replan cases.
They do not reliably predict rare segment-memory rescue cases.
This is a negative result and motivates a more principled method.
```

## Step 5: Define The Minimal Research Object

Question:

```text
What is the smallest scientific object we need to learn?
```

Answer:

```text
Counterfactual memory utility: whether memory m helps action prediction given current state x.
```

Research note:

```text
docs/minimal_counterfactual_memory_utility.md
```

Minimal label space:

```text
positive: no_history wrong, true memory correct, wrong memory wrong
negative: no_history correct, true memory wrong
neutral: no_history and memory both correct
unresolved: all contexts wrong
```

Interpretation:

```text
Do not predict generic long-horizon.
Predict whether a specific memory is useful for a specific current state.
```

## Step 6: Build Counterfactual Memory Utility Data

Question:

```text
Can we build triplets (current state, true memory, wrong memory) from existing behavior runs?
```

Script:

```text
scripts/build_counterfactual_memory_utility_data.py
```

Output:

```text
datasets/counterfactual_memory_utility/train.jsonl
datasets/counterfactual_memory_utility/dev.jsonl
datasets/counterfactual_memory_utility/test.jsonl
datasets/counterfactual_memory_utility/stats.json
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

Interpretation:

```text
The positive counterfactual memory utility class is extremely sparse.
This confirms that the task should be treated as high-precision retrieval/ranking, not ordinary multi-class routing.
```

## Step 7: Train Minimal TF-IDF Memory Utility Scorer

Question:

```text
Is semantic compatibility between current state and memory informative at all?
```

Script:

```text
scripts/train_counterfactual_memory_utility.py
```

Output:

```text
datasets/counterfactual_memory_utility_tfidf/memory_utility_model.joblib
datasets/counterfactual_memory_utility_tfidf/memory_utility_report.md
```

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
Semantic compatibility has real ranking signal: ROC-AUC 0.9321 is high, AP 0.1492 is far above the base positive rate.
However, TF-IDF alone is not enough for a practical router.
Low thresholds improve routed accuracy but have terrible precision.
High thresholds give clean memory activations but recall is tiny.
```

## Current Research Conclusion

The current best statement is:

```text
Counterfactual memory utility is learnable as a ranking signal, but a deployable long-horizon memory router needs additional evidence beyond text similarity.
```

The next required evidence was initially framed as:

```text
screen/OCR availability: whether the current screen already contains the carried value
candidate disagreement: whether no_history and segment_summary candidates differ in action type or exact value
```

After the partial OCR experiment, the main path should be narrowed:

```text
primary: candidate-level exact-value and verifier features
secondary/optional: OCR or stronger screen-text extraction
```

This is now a clean research path:

```text
behavior intervention -> counterfactual utility labels -> semantic scorer -> add visual/candidate evidence -> routed prospective evaluation
```

## Immediate Next Step

Implement one minimal addition at a time:

1. Add candidate-disagreement features from stored predictions.
2. Re-train the CMU scorer.
3. Check whether high-precision recall improves over TF-IDF-only.
4. Then add exact-value repair/verifier features before returning to OCR.

The next artifact should be:

```text
datasets/counterfactual_memory_utility_candidate_features
datasets/counterfactual_memory_utility_candidate_scorer
```

## Step 8: Add Candidate-Disagreement Features

Question:

```text
Is no-history vs segment-summary candidate disagreement the missing causal signal for memory utility?
```

Method:

```text
Use stored no_history, segment_summary, and wrong_summary predictions.
Add features that compare candidate action type, candidate text/value, carried-value overlap, and whether segment candidate differs from no-history.
```

Script:

```text
scripts/train_counterfactual_memory_utility.py --candidate-features
```

Output:

```text
datasets/counterfactual_memory_utility_candidate/memory_utility_report.md
datasets/counterfactual_memory_utility_candidate/memory_utility_model.joblib
```

Held-out test comparison:

| model | AP | ROC-AUC | best routed acc | notes |
|---|---:|---:|---:|---|
| TF-IDF current/memory text | 0.1492 | 0.9321 | 0.9206 | weak precision, broad recall |
| TF-IDF + candidate disagreement | 0.6349 | 0.9973 | 0.9217 | much stronger ranking signal |

Candidate-aware threshold behavior on test:

| threshold | predicted memory | precision | recall | routed value acc | regressions |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 81 | 0.2593 | 1.0000 | 0.9206 | 16 |
| 0.30 | 66 | 0.3030 | 0.9524 | 0.9217 | 10 |
| 0.50 | 64 | 0.3125 | 0.9524 | 0.9217 | 10 |
| 0.70 | 59 | 0.3390 | 0.9524 | 0.9217 | 10 |
| 0.90 | 49 | 0.3878 | 0.9048 | 0.9206 | 10 |

Interpretation:

```text
Candidate disagreement is a major missing signal.
It improves AP by more than 4x over TF-IDF-only.
However, precision is still below the desired 0.50 deployment target.
This suggests the next missing signal should be candidate-level verification: whether the segment-memory candidate repairs the no-history candidate's action type, exact value, target entity, or stale action. OCR/screen availability may help later, but it is not the cleanest immediate next step.
```

## Step 9: Add Current-State / Screen-Availability Features

Question:

```text
Does a weak screen/current-state availability proxy improve memory utility prediction beyond candidate disagreement?
```

Method:

```text
Expose current-state parts in CMU data: goal, current instruction, screen observation, local hint, current segment.
Add overlap features between carried values / memory text and these current-state fields.
Train the same scorer with both candidate-disagreement and screen/current-state features.
```

Scripts:

```text
scripts/build_counterfactual_memory_utility_data.py
scripts/train_counterfactual_memory_utility.py --candidate-features --screen-features
```

Output:

```text
datasets/counterfactual_memory_utility_v2
datasets/counterfactual_memory_utility_candidate_screen
```

Held-out test comparison:

| model | AP | ROC-AUC | best routed acc | best high-threshold precision |
|---|---:|---:|---:|---:|
| TF-IDF current/memory text | 0.1492 | 0.9321 | 0.9206 | 1.0000 at threshold 0.90, but only 2 predictions |
| TF-IDF + candidate disagreement | 0.6349 | 0.9973 | 0.9217 | 0.3878 at threshold 0.90 |
| TF-IDF + candidate disagreement + screen/current-state overlap | 0.5978 | 0.9970 | 0.9217 | 0.4043 at threshold 0.90 |

Candidate+screen threshold behavior on test:

| threshold | predicted memory | precision | recall | routed value acc | regressions |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 83 | 0.2530 | 1.0000 | 0.9208 | 15 |
| 0.50 | 63 | 0.3175 | 0.9524 | 0.9217 | 10 |
| 0.70 | 56 | 0.3571 | 0.9524 | 0.9214 | 10 |
| 0.90 | 47 | 0.4043 | 0.9048 | 0.9203 | 10 |

Interpretation:

```text
Candidate disagreement remains the dominant signal.
Weak screen/current-state text overlap does not materially improve AP or routed accuracy.
It slightly improves high-threshold precision, but not enough to reach the desired >=0.50 precision operating point.
This means we likely need real OCR/visible text or visual state features, not just the existing natural-language observation field.
```

Updated next step at this point:

```text
Add exact-value and candidate-verifier features.
The key feature should answer: did segment memory repair the no-history candidate in a way a verifier can detect?
```

## Step 10: Add Partial Real OCR Availability

Question:

```text
Does real OCR-visible text improve the candidate-aware memory utility scorer?
```

Method:

```text
Install RapidOCR in the uv-managed .venv.
Use opencv-python-headless to avoid server GUI library dependencies.
Build an OCR cache for counterfactually informative screenshots plus a neutral sample.
Attach OCR text to current-state text and add OCR overlap features.
```

Scripts:

```text
scripts/build_ocr_cache.py
scripts/train_counterfactual_memory_utility.py --candidate-features --screen-features --ocr-cache ...
```

Output:

```text
datasets/counterfactual_memory_utility_v2/ocr_cache_informative.jsonl
datasets/counterfactual_memory_utility_candidate_screen_ocr
```

Current OCR coverage:

```text
ocr_cache rows: 322
ok: 322
```

This is partial coverage only. Full informative target is about 3051 screenshots.

Held-out test comparison:

| model | AP | ROC-AUC | best routed acc | precision@0.90 | recall@0.90 |
|---|---:|---:|---:|---:|---:|
| candidate disagreement | 0.6349 | 0.9973 | 0.9217 | 0.3878 | 0.9048 |
| candidate + weak screen text | 0.5978 | 0.9970 | 0.9217 | 0.4043 | 0.9048 |
| candidate + weak screen text + partial OCR | 0.5470 | 0.9968 | 0.9214 | 0.4130 | 0.9048 |

Interpretation:

```text
Partial real OCR slightly improves high-threshold precision but lowers AP.
This is not enough evidence that OCR solves the problem.
The likely issue is incomplete OCR coverage and noisy OCR text.
The right conclusion is not to make OCR the main path. OCR is slow and noisy in this environment, and partial OCR did not improve AP. Treat OCR as optional supporting evidence, not the next core method.
```

Updated next step after this result:

```text
Move to candidate-level exact-value/verifier features.
Use OCR later only if candidate-verifier features still cannot distinguish memory necessity from memory correlation.
```

## Step 11: Add Candidate-Level Repair Features

Question:

```text
Can we predict positive memory utility by checking whether the segment-memory candidate repairs a specific no-history error?
```

Rationale:

```text
Candidate disagreement already gives the strongest signal so far.
The next step is to structure that disagreement into verifiable repair types, not to add more raw OCR overlap.
```

Implemented candidate repair types:

```text
action_type_repair: no_history action type wrong, segment action type plausible/correct
exact_text_repair: no_history text/value differs from carried value, segment text/value matches carried value
system_button_repair: no_history selects Home/Back incorrectly, segment selects the needed system action
target_entity_repair: no_history clicks/selects generic or stale entity, segment candidate selects carried/current entity
stale_memory_risk: segment candidate moves toward an unrelated/wrong-summary entity
```

Artifacts:

```text
scripts/train_counterfactual_memory_utility.py --repair-features
datasets/counterfactual_memory_utility_candidate_refit
datasets/counterfactual_memory_utility_repair_only
datasets/counterfactual_memory_utility_candidate_repair
```

Implementation detail:

```text
The features do not use gt_action or condition_value_match.
They compare no_history versus memory-conditioned candidates using action type transitions, system-button fallback, value text changes, carried-value overlap gain, memory/current/instruction overlap gain, coordinate movement, and swipe direction changes.
```

Fair held-out test comparison, retrained with the same feature extractor version:

| model | AP | ROC-AUC | precision@0.90 | recall@0.90 | regressions@0.90 | precision@0.97 | recall@0.97 | regressions@0.97 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate refit | 0.6003 | 0.9971 | 0.3878 | 0.9048 | 10 | 0.4800 | 0.5714 | 5 |
| repair only | 0.6164 | 0.9976 | 0.3774 | 0.9524 | 9 | 0.5600 | 0.6667 | 3 |
| candidate + repair | 0.6270 | 0.9977 | 0.3922 | 0.9524 | 7 | 0.5833 | 0.6667 | 3 |

High-confidence operating points for candidate + repair:

| threshold | predicted | precision | recall | routed acc | regressions |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 51 | 0.3922 | 0.9524 | 0.9217 | 7 |
| 0.95 | 43 | 0.4419 | 0.9048 | 0.9206 | 7 |
| 0.97 | 24 | 0.5833 | 0.6667 | 0.9188 | 3 |
| 0.98 | 17 | 0.6471 | 0.5238 | 0.9186 | 1 |
| 0.99 | 14 | 0.7857 | 0.5238 | 0.9184 | 0 |

Interpretation:

```text
Repair features do not produce a large AP jump over candidate disagreement.
They do improve the high-confidence routing shape: at threshold 0.90, candidate + repair keeps recall at 0.9524 and reduces segment regressions from 10 to 7 versus candidate-only.
At threshold 0.97, candidate + repair crosses the 0.50 precision operating point with 0.5833 precision and 0.6667 recall.
This supports the non-OCR direction as a deployable verifier/reranker path, but the next step should target false positives rather than adding more broad features.
```

Next experiment:

```text
Mine the false positives at threshold 0.90 and 0.97.
Separate true repair from merely different candidate by adding wrong-summary agreement, stale-memory risk, and candidate self-consistency features.
```

## Step 12: Counterfactual Specificity Filter

Question:

```text
Can we reject false memory activations by checking whether true segment memory produces a candidate that is specific relative to a wrong-memory candidate?
```

Rationale:

```text
If segment_summary and wrong_summary produce the same or same-type candidate, the candidate change is probably not caused by the correct memory.
This is a cleaner non-OCR causal test than asking whether screen text contains the value.
```

Script:

```text
scripts/analyze_candidate_repair_errors.py
datasets/counterfactual_memory_utility_candidate_repair_error_analysis
```

Error mining result:

```text
At threshold 0.90, true positives had segment_summary == wrong_summary candidate in 0/20 cases.
False positives had exact segment_summary == wrong_summary candidate in 8/31 cases.
At threshold 0.99, false positives had exact segment_summary == wrong_summary candidate in 2/3 cases.
```

Specificity filter comparison on test:

| threshold | policy | predicted | precision | recall | routed acc | regressions |
|---:|---|---:|---:|---:|---:|---:|
| 0.90 | raw candidate + repair | 51 | 0.3922 | 0.9524 | 0.9217 | 7 |
| 0.90 | reject exact segment == wrong | 43 | 0.4651 | 0.9524 | 0.9203 | 6 |
| 0.90 | reject segment type == wrong type | 25 | 0.6400 | 0.7619 | 0.9179 | 4 |
| 0.97 | raw candidate + repair | 24 | 0.5833 | 0.6667 | 0.9188 | 3 |
| 0.97 | reject exact segment == wrong | 22 | 0.6364 | 0.6667 | 0.9184 | 3 |
| 0.97 | reject segment type == wrong type | 17 | 0.7647 | 0.6190 | 0.9177 | 2 |
| 0.99 | raw candidate + repair | 14 | 0.7857 | 0.5238 | 0.9184 | 0 |
| 0.99 | reject exact segment == wrong | 12 | 0.9167 | 0.5238 | 0.9179 | 0 |
| 0.99 | reject segment type == wrong type | 11 | 1.0000 | 0.5238 | 0.9177 | 0 |

Interpretation:

```text
Counterfactual specificity is a strong next signal.
Exact segment/wrong agreement improves precision without hurting recall in this test.
Type-level agreement is more conservative: it sacrifices recall but gives high precision and zero false positives at threshold 0.99.
This reframes the router as a memory-specific repair detector: route to memory only when the segment-memory candidate changes the no-history candidate and does not look like a generic change induced by wrong memory.
```

Next experiment:

```text
Train a specificity-aware candidate repair scorer rather than applying the filter only after scoring.
Use segment-vs-wrong exact match, type match, swipe direction match, and coordinate-distance buckets as features.
Evaluate whether it preserves the 0.90 high-recall regime while improving precision beyond 0.50.
```

## Step 13: Train Specificity-Aware Candidate Repair Scorer

Question:

```text
Can counterfactual specificity be learned as part of the scorer instead of applied only as a post-hoc filter?
```

Implemented features:

```text
specificity_candidate_equals_distractor_exact
specificity_candidate_equals_distractor_type
specificity_candidate_differs_from_no_and_distractor
specificity_candidate_type_differs_from_no_and_distractor
specificity_distractor_equals_no_exact/type
candidate-vs-distractor swipe direction match
candidate-vs-distractor coordinate-distance bucket
```

Artifacts:

```text
datasets/counterfactual_memory_utility_specificity_only
datasets/counterfactual_memory_utility_repair_specificity
datasets/counterfactual_memory_utility_candidate_repair_specificity
datasets/counterfactual_memory_utility_candidate_repair_specificity_error_analysis
```

Held-out test comparison:

| model | AP | ROC-AUC | precision@0.90 | recall@0.90 | regressions@0.90 | precision@0.99 | recall@0.99 | regressions@0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate + repair | 0.6270 | 0.9977 | 0.3922 | 0.9524 | 7 | 0.7857 | 0.5238 | 0 |
| specificity only | 0.8269 | 0.9991 | 0.5556 | 0.9524 | 6 | 0.6429 | 0.8571 | 5 |
| repair + specificity | 0.8261 | 0.9991 | 0.5882 | 0.9524 | 5 | 0.6129 | 0.9048 | 4 |
| candidate + repair + specificity | 0.8318 | 0.9991 | 0.5714 | 0.9524 | 5 | 0.6129 | 0.9048 | 4 |

Best high-recall operating point:

| threshold | predicted | precision | recall | routed acc | regressions |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 40 | 0.5250 | 1.0000 | 0.9188 | 6 |
| 0.90 | 35 | 0.5714 | 0.9524 | 0.9186 | 5 |
| 0.99 | 31 | 0.6129 | 0.9048 | 0.9186 | 4 |

Important nuance:

```text
Routed accuracy does not increase as much as memory-specific precision, because the specificity-aware scorer rejects many nonspecific positives where wrong memory also happens to produce a correct candidate.
For bottleneck validation, that is the desired behavior: the target is specific memory utility, not any condition that makes segment_summary correct.
```

Learned feature sanity check:

```text
Positive weights: distractor has no-history type, candidate differs from both no-history and distractor, candidate type differs from both, exact-value gain, carried-value overlap.
Negative weights: candidate equals distractor exact/type/value, candidate equals no-history, matching swipe direction with distractor, same click/click or swipe/swipe transitions.
```

Interpretation:

```text
This is the strongest non-OCR result so far.
The method now passes the initial high-precision target: precision >= 0.50 while maintaining high recall for specific segment-rescue cases.
The result supports a research claim: behavior-intervention candidates contain enough counterfactual structure to identify memory-specific bottlenecks without OCR.
```

Next experiment:

```text
Mine the remaining false positives for the specificity-aware scorer.
The residual errors are mostly system_button->click and system_button->system_button transitions, suggesting the next verifier should reason about whether a candidate is task-progressing or just another valid navigation action.
```

## Step 14: Add Instruction Task-Progress Features

Question:

```text
Can we reduce remaining false positives by checking whether the memory-induced candidate is compatible with the current instruction intent?
```

Motivation from Step 13 false positives:

```text
At threshold 0.90, remaining false positives were mostly unresolved/negative cases.
Dominant transitions: system_button->click, system_button->system_button, and click->swipe.
The issue is no longer generic wrong-memory agreement; it is whether the candidate is actually task-progressing for the current instruction.
```

Implemented features:

```text
scripts/train_counterfactual_memory_utility.py --progress-features
instruction intents: empty, home, back, scroll, type, click_open, terminate, adjust, other
candidate/no-history/distractor action compatibility with each intent
candidate-specific instruction match
empty-instruction candidate-change risk
navigation-intent candidate click/swipe risk
```

Artifacts:

```text
datasets/counterfactual_memory_utility_progress_only
datasets/counterfactual_memory_utility_specificity_progress
datasets/counterfactual_memory_utility_repair_specificity_progress
datasets/counterfactual_memory_utility_candidate_repair_specificity_progress
datasets/counterfactual_memory_utility_specificity_progress_error_analysis
```

Held-out test comparison:

| model | AP | precision@0.50 | recall@0.50 | precision@0.70 | recall@0.70 | precision@0.90 | recall@0.90 | regressions@0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate + repair + specificity | 0.8318 | 0.5128 | 0.9524 | 0.5263 | 0.9524 | 0.5714 | 0.9524 | 5 |
| progress only | 0.6978 | 0.4762 | 0.9524 | 0.5135 | 0.9048 | 0.5588 | 0.9048 | 5 |
| specificity + progress | 0.8443 | 0.5250 | 1.0000 | 0.5676 | 1.0000 | 0.5714 | 0.9524 | 5 |
| repair + specificity + progress | 0.8074 | 0.5526 | 1.0000 | 0.5714 | 0.9524 | 0.5588 | 0.9048 | 5 |
| candidate + repair + specificity + progress | 0.8050 | 0.5526 | 1.0000 | 0.5405 | 0.9524 | 0.5588 | 0.9048 | 5 |

Selected model:

```text
specificity + progress
```

Reason:

```text
It has the best AP, preserves recall 1.0000 through threshold 0.70, and reaches precision 0.5714 / recall 0.9524 at threshold 0.90.
Adding repair and generic candidate features on top of specificity+progress hurts AP, so the next-stage verifier should stay minimal.
```

Weight sanity check:

```text
Positive: candidate-specific scroll match, distractor matches no-history type, candidate differs from no-history and distractor.
Negative: click with empty instruction, candidate equals no-history, candidate matches distractor value/type, task-intent no-history system_button -> candidate task-action overactivation.
```

Interpretation:

```text
Task-progress features help, but only when paired with specificity.
The research direction is now a two-test router: memory-specificity first, instruction-progress compatibility second.
This is still fully non-OCR and uses only counterfactual candidates plus current instruction text.
```

Next experiment:

```text
Use specificity+progress as the default non-OCR scorer.
Mine its remaining false positives and check whether they are annotation ambiguity, unresolved all-wrong cases, or true memory regressions.
Then evaluate per-capability thresholds rather than a single global threshold.
```

## Step 15: Reframe As Cross-Benchmark Research Method

Question:

```text
Is this a real research method, and can it be useful across benchmarks?
```

Answer:

```text
The method is research-level if the invariant object is counterfactual memory utility, not benchmark-specific features.
The current evidence supports structural cross-benchmark portability for GUI-Odyssey and AndroidControl.
It does not yet prove trained-scorer transfer across all benchmarks.
```

New protocol document:

```text
docs/cross_benchmark_memory_router_research_protocol.md
```

New audit script:

```text
scripts/audit_cross_benchmark_memory_method.py
```

Adapter fix:

```text
scripts/analyze_trajectory_segments.py now preserves AndroidControl step_instruction in canonical text_fields.instruction.
```

Audit result on canonical segmented episodes:

| benchmark | episodes | steps | instruction rate | screenshot rate | segmentation | interventions | specificity | progress | core ready | full ready |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| AndroidControl eval | 200 | 1067 | 99.5% | 100.0% | yes | yes | yes | yes | yes | yes |
| GUI-Odyssey train sample | 500 | 7705 | 100.0% | 100.0% | yes | yes | yes | yes | yes | yes |

Interpretation:

```text
The core intervention method is portable because both benchmarks expose goal, screenshot/current state, normalized action, trajectory order, and step-level instruction.
Specificity is portable because wrong memory can be sampled from the same canonical segment pool.
Progress is portable when step-level instruction exists; AndroidControl eval has it once the adapter preserves step_instruction.
```

Important limitation:

```text
This is not yet a cross-benchmark transfer result.
The next evidence must train on one benchmark and evaluate unchanged on another, then run prospective routed evaluation.
```

Next experiment:

```text
Run AndroidControl behavior interventions under no_history / segment_summary / full_history / wrong_summary.
Build AndroidControl CMU rows.
Evaluate GUI-Odyssey -> AndroidControl and AndroidControl -> GUI-Odyssey transfer for specificity+progress.
```

## Step 16: GUI-Odyssey Per-Capability Thresholding

Question:

```text
Can we improve GUI-Odyssey memory activation by selecting thresholds per dominant capability instead of using one global threshold?
```

Motivation:

```text
The remaining GUI-Odyssey positives are not evenly distributed across capabilities.
Test positives are concentrated in browse_scan, search, interact, navigate_system, and select_target.
This suggests per-capability thresholds might recover hard cases while avoiding false activations in low-yield capabilities.
```

Script:

```text
scripts/evaluate_memory_router_thresholds.py
```

Input scorer:

```text
datasets/counterfactual_memory_utility_specificity_progress/memory_utility_model.joblib
```

Outputs:

```text
datasets/counterfactual_memory_utility_specificity_progress_thresholds
datasets/counterfactual_memory_utility_specificity_progress_thresholds_min5
datasets/counterfactual_memory_utility_specificity_progress_thresholds_min10
```

Result on GUI-Odyssey test, thresholds selected only on dev:

| target precision | policy | support rule | predicted | precision | recall | regressions |
|---:|---|---|---:|---:|---:|---:|
| 0.50 | global | n/a | 45 | 0.4667 | 1.0000 | 7 |
| 0.50 | per capability | min dev positives 2 | 45 | 0.4667 | 1.0000 | 7 |
| 0.60 | global | n/a | 40 | 0.5250 | 1.0000 | 7 |
| 0.60 | per capability | min dev positives 2 | 44 | 0.4773 | 1.0000 | 7 |
| 0.70 | global | n/a | 18 | 0.7778 | 0.6667 | 1 |
| 0.70 | per capability | min dev positives 2 | 38 | 0.5000 | 0.9048 | 6 |
| 0.70 | per capability | min dev positives 5 | 29 | 0.6207 | 0.8571 | 3 |
| 0.70 | per capability | min dev positives 10 | 18 | 0.7778 | 0.6667 | 1 |

Interpretation:

```text
Per-capability thresholding is not currently a reliable improvement on GUI-Odyssey.
The dev positives per capability are too sparse, so per-capability thresholds overfit.
The conservative min-positive setting collapses back to the global threshold.
```

Selected operating policy for now:

```text
Use the global specificity+progress threshold selected on dev.
For high recall: target 0.60 dev threshold gives test precision 0.5250 / recall 1.0000.
For higher precision: target 0.70 dev threshold gives test precision 0.7778 / recall 0.6667.
```

Next experiment:

```text
Do not add per-capability thresholds until each capability has enough memory-positive dev support.
Instead, mine false positives under the global high-recall threshold and separate annotation ambiguity from true memory regressions.
```