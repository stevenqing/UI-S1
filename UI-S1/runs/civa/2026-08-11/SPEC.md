# CIVA: Conditional Incremental Value Admission

Date frozen: 2026-08-11

Status: `PREREGISTERED_AFTER_DELTA_BEFORE_CIVA_FIT`

## 1. Scope and claim boundary

CIVA is a new post-DELTA discovery study. VUS-SR, CARE, RAVEL, DELTA, and the descriptive switch headroom in `KNOWN_DIAGNOSTICS.json` are known. CIVA cannot confirm a method on Mind2Web or ScreenSpot-Pro; any positive result still requires a separately frozen policy-level study and an untouched third benchmark.

The first-principles question is narrower than fusion: can information available before an extra evidence channel is admitted predict whether that channel will improve or damage the VUS-binding direct decision?

For frozen baseline policy $\pi_0$ and channel policy $\pi_m$:

$$
\Delta_m(x)=Y_{\pi_m}(x)-Y_{\pi_0}(x)\in\{-1,0,+1\}.
$$

`+1` is a rescue, `-1` is a harm, and `0` is neutral. CIVA estimates rescue and harm separately and admits a channel only when estimated incremental utility is positive under a development-selected hard threshold.

## 2. Frozen policies

All policies use already locked, fallback-agnostic Qwen3-VL label logits. No new VLM inference is allowed in CIVA-A0.

- baseline: candidate argmax from `vus_binding`;
- real experts: candidate argmax from `global_semantic`, `fine_local`, and `context_local`;
- placebo expert: candidate argmax from `random_placebo`.

Display-label logits are restored to original candidate order before argmax. Ties use the lowest original candidate index. No learned DELTA/VUS policy output is used as a training feature or target, avoiding second-level stacking across incompatible outer folds.

## 3. Pre-admission features

The selector may use only information available before acquiring a real expert:

- fixed hashed instruction unigram/bigram representation;
- VUS-binding logits, probabilities, entropy, top margins, normalized ranks, and score dispersion;
- public candidate action categories, normalized coordinates, duplicate/agreement summaries, and pairwise geometric dispersion;
- screenshot aspect ratio and candidate count;
- benchmark and arm indicators.

The selector may not use any global/fine/context/random logits, source/model/slot identity, website/application identity, evaluator outputs, target box, target area, private labels, frozen-policy success, or outer-fold identity. Website/application remain grouping variables only. Mind2Web history-only fields are excluded so the feature contract is shared across benchmarks.

## 4. Frozen learner

For each expert and feature variant, train independent rescue and harm `HistGradientBoostingClassifier` heads. The admission score is

$$
s_m(x)=P(\Delta_m=+1\mid x)-P(\Delta_m=-1\mid x).
$$

Hyperparameters are fixed in `configs/civa_prereg.yaml`. No search, early stopping, calibration-method selection, or expert warm start is allowed.

Primary `REAL_FULL` chooses the real expert with maximum score and switches only when that score exceeds a frozen threshold. Mandatory learned controls are:

- `REAL_NO_TEXT`: same learner without instruction features;
- `REAL_TEXT_ONLY`: instruction plus benchmark/arm state only;
- `PLACEBO_FULL`: same learner and features for the random-center expert.

`MATCHED_RANDOM` switches at the same outer-test benchmark/arm coverage as `REAL_FULL`, using a frozen hash order and hash-selected real expert. It cannot influence threshold selection.

## 5. Nested protocol

For each of five outer folds:

1. keep the physical outer-label file sealed;
2. use the other four label folds only;
3. produce four-fold development OOF scores, each fit on the other three development folds;
4. select one threshold per benchmark/arm from infinity, zero, and positive score deciles;
5. require each development cell to lose at most 0.5 MDE and each benchmark equal-arm mean to lose at most 0.25 MDE; infinity reproduces baseline;
6. refit every frozen learner on all four development folds;
7. atomically fsync feature hashes, thresholds, learner contract, and opened development-label hashes to `outer-k.pretest.json`;
8. only then open outer-fold labels and evaluate once.

Training weights give each benchmark mass one, each unique row equal mass within benchmark, and each arm equal mass within row. Statistics use the existing grouped 10,000-resample paired bootstrap and 99% percentile intervals.

## 6. Gates

| Gate | Requirement |
| --- | --- |
| CIVA-1 | `REAL_FULL` equal-benchmark standardized 99% CI is positive versus VUS-binding direct baseline |
| CIVA-2 | At least one benchmark equal-arm 99% CI is positive; every arm on the other benchmark has CI lower bound above negative MDE |
| CIVA-3 | `REAL_FULL` balanced 99% CI is positive versus `MATCHED_RANDOM` |
| CIVA-4 | `REAL_FULL` balanced 99% CI is positive versus `PLACEBO_FULL` |
| CIVA-5 | `REAL_FULL` balanced 99% CI is positive versus `REAL_NO_TEXT` |
| CIVA-6 | Every benchmark/arm cell is noninferior to baseline with 99% CI lower bound above negative MDE |

CIVA-A0 passes only if all six gates pass. A pass supports learnable task-conditioned evidence admission, not a deployable method.

## 7. Kill conditions

- `CIVA-K1`: any locked input hash or row identity mismatch;
- `CIVA-K2`: any prohibited post-evidence, identity, target, evaluator, or label field enters features;
- `CIVA-K3`: matched random switching explains the gain;
- `CIVA-K4`: random-center expert explains the gain;
- `CIVA-K5`: instruction text adds no held-out utility beyond no-text state;
- `CIVA-K6`: any outer label opens before the pretest record is fsynced;
- `CIVA-K7`: any threshold, learner, feature, gate, or expert set changes after a formal fit.

If CIVA-A0 fails, no policy-level admission model, contrastive verifier, VLM fine-tuning, or distillation is authorized from this branch.