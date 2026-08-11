# RAVEL: Relational Aggregation with Visual Evidence and Lower-Bound Safety

Date frozen: 2026-08-11

Status: `FROZEN_AFTER_CARE_A1_BEFORE_RAVEL_RESULTS`

## 1. Scope and disclosure

RAVEL is a separate post-A1 study. It was frozen after the preliminary A1 direction was known but before the corrected cross-fitted-reliability rerun. The corrected run preserves and strengthens the same conclusion: stage1-only structural routing does not improve counterfactual candidate coverage and can hurt final Step-SR. RAVEL therefore fixes candidate acquisition rather than adapting it.

All VUS-SR results, CARE diagnostics, and CARE A1 outcomes are known. RAVEL is discovery on the same two benchmarks. A third untouched benchmark is required for confirmation.

## 2. First-principles claim

With fixed acquisition, the dominant residual loss is not lack of candidate coverage but failure to identify a semantically correct minority candidate from a visually compressed set.

RAVEL tests one claim:

> Candidate selection improves when each candidate receives explicit local multi-scale evidence, candidates are compared relationally to a strong fallback in net-utility space, and override risk is controlled by a lower confidence bound.

This differs from:

- density voting, which favors repeated geometric modes;
- VUS-SR, which uses one downsampled full-screen A--L overlay;
- Q2b verification, which independently filters candidates using crop-presence YES/NO labels;
- actor pairwise training, which changes the generator and has previously damaged behavior.

## 3. Fixed acquisition

Primary acquisition is C-cond for both benchmarks because every CARE A1 outer-development split selected C-cond as nested best-static pass@12. All four arms remain mandatory evaluation cells; no adaptive routing is used.

Candidate budget remains 12 forwards. Selector evidence uses one Qwen3-VL invocation per row-arm, matching the number of VLM selector calls in VUS.

## 4. Token-matched local evidence

For each row-arm create three images:

1. unmarked global screenshot;
2. 12-tile fine candidate mosaic;
3. 12-tile context candidate mosaic.

Each candidate tile contains one crosshair and A--L label. Fine and context crop short-edge fractions inherit CARE's pre-A1 frozen values `0.07` and `0.21`; they are not tuned after A1.

Total processed visual pixels must not exceed the row's frozen VUS full-screen visual pixel count by more than 2%. Allocate 50% global, 25% fine mosaic, and 25% context mosaic. A compute-matched full-screen control uses the same total pixel budget and one VLM invocation.

Prompt contains task/history and action/parameter legend. It prohibits target boxes, positive DOM, candidate success, evaluator fields, source/model/slot identity, and fallback identity.

Qwen3-VL emits A--L candidate logits. All logits are locked and hashed before labels are opened.

## 5. E0 evidence gate

Compare local evidence to the frozen full-screen VUS logits on:

- utility-positive candidate AUROC;
- direct recall when exactly one candidate is correct;
- direct recall in the smallest target-area quartile;
- nested safe Step-SR using the unchanged VUS-SR architecture and training protocol.

Required controls:

- compute-matched full-screen;
- candidate mosaics with random centers;
- global-only at matched total pixels;
- fine-only and context-only descriptive ablations.

E0 passes if either:

1. one benchmark gains at least 0.03 utility-positive AUROC and the other loses at most 0.01; or
2. one benchmark's safe Step-SR has positive 99% CI versus VUS-SR and every cell on the other benchmark is noninferior under its MDE.

If E0 fails, stop RAVEL. Full VLM LoRA is prohibited because the proposed evidence representation itself lacks value.

## 6. Relational utility model

Only after E0 passes, train a permutation-equivariant relational set model. For candidate $i$ and exact CEV-A fallback $b$:

$$
y_{ib}\in\{\mathrm{REPAIR},\mathrm{SAME},\mathrm{BREAK}\},
$$

and

$$
\Delta_{ib}=P(\mathrm{REPAIR})-P(\mathrm{BREAK}).
$$

Pair score is antisymmetric:

$$
d(i,b)=h(e_i,e_b)-h(e_b,e_i).
$$

Frozen losses:

- listwise repair-or-KEEP CE: 1.0;
- Bradley–Terry loss on discordant pairs: 1.0;
- fallback-correct BCE: 0.5;
- expected U-GRPO utility: 0.1.

Candidate deletion is prohibited. The model selects from all 12 real candidates or KEEP.

## 7. Lower-bound safety

Calibration rows estimate grouped net utility for a fixed conservative-to-permissive threshold sequence. Select the most permissive threshold whose one-sided 99% lower confidence bound is nonnegative. Apply family-wise correction over the fixed sequence.

Threshold selection metric is net repair-minus-break utility, never verifier accuracy or AUROC.

## 8. Nested protocol

- five grouped outer folds;
- physically fold-sealed labels;
- inner checkpoint fold;
- atomic pretest selection record before any outer label access;
- one outer-test evaluation;
- paired 10,000-resample grouped bootstrap with 99% CIs;
- same rows/arms/MDEs as VUS-SR.

## 9. Main gates

| Gate | Requirement |
| --- | --- |
| RAVEL-1 | E0 local evidence passes its frozen representation gate |
| RAVEL-2 | Mind2Web equal-arm safe gain vs VUS-SR has 99% CI lower bound positive |
| RAVEL-3 | Every ScreenSpot cell is noninferior to VUS-SR under 0.70 pp MDE |
| RAVEL-4 | Equal-benchmark standardized 99% CI is positive vs VUS-SR |
| RAVEL-5 | One-sided calibrated net-utility lower bound is nonnegative in every outer fold |
| RAVEL-6 | Frozen transfer passes on an untouched third benchmark |

RAVEL is a confirmed method only if RAVEL-1--RAVEL-6 pass. RAVEL-1--RAVEL-5 is discovery.

## 10. Kill conditions

- `RAVEL-K1`: visual pixel budget exceeds VUS by more than 2%;
- `RAVEL-K2`: any target/evaluator field enters evidence prompt;
- `RAVEL-K3`: random-center control explains the local-evidence gain;
- `RAVEL-K4`: E0 fails;
- `RAVEL-K5`: relational training improves verifier metrics but not final utility;
- `RAVEL-K6`: any outer-test label opens before pretest fsync;
- `RAVEL-K7`: third benchmark influences method or threshold selection.

Failed gates stop the branch; they do not authorize crop-scale, prompt, or loss tuning.
