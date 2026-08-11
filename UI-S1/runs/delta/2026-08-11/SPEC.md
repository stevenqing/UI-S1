# DELTA: Decision-Level Evidence Late-fusion with Testable Attribution

Date frozen: 2026-08-11

Status: `FROZEN_AFTER_RAVEL_K4_BEFORE_DELTA_RESULTS`

## 1. Scope

DELTA is a new post-RAVEL study. CARE routing and RAVEL pixel-level early fusion are closed. All VUS-SR, CARE, and RAVEL E0 results are known.

RAVEL established two facts:

1. candidate-centered local pixels contain signal: local utility AUROC exceeds token-identical random centers by about 0.046 on both benchmarks;
2. putting global/fine/context images into one fixed-token prompt loses Mind2Web global/unique-candidate information and reduces final Step-SR by 2.19 pp.

DELTA tests whether independently encoded evidence channels are complementary when fused only at candidate-decision level.

## 2. Frozen inputs

No new VLM inference is allowed in the viability study. Inputs are already blind-locked candidate logits:

- VUS marked full-screen logits: binding-rich channel;
- RAVEL global-only logits: unmarked global-semantic channel;
- RAVEL fine-only logits: local appearance channel;
- RAVEL context-only logits: local-context channel;
- frozen structural/fallback-pair candidate features.

Random-center logits are prohibited as an input and retained only as a placebo control.

Every channel must match its locked manifest SHA-256. Source/model/slot identity, target boxes, evaluator fields, and test labels are prohibited model inputs.

## 3. Model

For candidate $i$ and channel $m$, map channel score/rank/entropy into an embedding $e_i^m$. Fuse channels with per-candidate simplex weights:

$$
\alpha_i=\mathrm{softmax}(g([e_i^1,\ldots,e_i^M,x_i])),
\qquad
e_i=\sum_m\alpha_i^m e_i^m.
$$

The channel gate is candidate-wise and permutation equivariant. It cannot attend to candidate slot position. A set encoder then predicts candidate/KEEP utility and fallback correctness, matching VUS-SR's deployment surface.

Frozen architecture:

- channel encoder width 32, shared across channels;
- channel gate width 32;
- fused candidate width 64;
- two Transformer set layers, four heads, no position embeddings;
- dropout 0.1.

Frozen losses:

- repair-or-KEEP listwise CE: 1.0;
- fallback-correct BCE: 0.5;
- channel consistency loss on candidate permutations: 0.1;
- expected U-GRPO utility: 0.1.

## 4. Required controls

- VUS-SR frozen output;
- same model with only VUS channel;
- same model with VUS + global;
- same model with VUS + fine + context but no global;
- same model replacing fine/context with random-center logits;
- fixed equal-weight logit average;
- channel-dropout robustness.

Controls are selected before outer-test access. The full model must beat both VUS-only same-capacity and random-channel controls; otherwise gains are architecture/capacity effects rather than evidence complementarity.

## 5. Nested protocol

- physically fold-sealed labels;
- five grouped outer folds;
- two training folds, one checkpoint fold, one OOF selection fold;
- atomic pretest record before outer label access;
- fixed model/controls, no post-result hyperparameter changes;
- 10,000 paired grouped bootstrap resamples and 99% CIs.

## 6. Gates

| Gate | Requirement |
| --- | --- |
| DELTA-1 | Full late fusion beats VUS-SR on Mind2Web with 99% CI lower bound positive |
| DELTA-2 | Every ScreenSpot arm is noninferior to VUS-SR under 0.70 pp MDE |
| DELTA-3 | Equal-benchmark standardized 99% CI is positive versus VUS-SR |
| DELTA-4 | Full model beats VUS-only same-capacity control with positive balanced 99% CI |
| DELTA-5 | Full model beats random-channel placebo with positive balanced 99% CI |
| DELTA-6 | At least two real channels receive nontrivial gate mass and permutation attribution is stable across folds |

Passing DELTA-1--DELTA-6 supports evidence complementarity but not deployment efficiency.

## 7. Distillation and confirmation boundary

Late fusion uses four independently encoded channels and is therefore a multi-call research oracle. Only if DELTA-1--DELTA-6 pass may a separate student distill it into one selector invocation under the VUS token budget. The student must retain the teacher's gains within one MDE.

No confirmed method claim is allowed until the distilled student is evaluated once on an untouched third benchmark. GUI-Odyssey app-split is preferred, but its dataset is not currently mounted. AndroidControl is only a robustness set because it has prior-study and evaluator confounds.

## 8. Kill conditions

- `DELTA-K1`: any channel manifest or row identity mismatch;
- `DELTA-K2`: VUS-only same-capacity explains the full gain;
- `DELTA-K3`: random channels explain the full gain;
- `DELTA-K4`: Mind2Web does not improve or ScreenSpot loses one MDE;
- `DELTA-K5`: channel attribution collapses to one channel in at least four folds;
- `DELTA-K6`: any outer label opens before pretest fsync;
- `DELTA-K7`: student or confirmation protocol is tuned after third-benchmark access.
