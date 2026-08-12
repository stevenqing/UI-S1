# DELTA Decision-Level Late-Fusion Report

Date: 2026-08-11

Outcome: `DELTA_NOT_SUPPORTED`

## 1. Protocol and integrity anchors

The result-free implementation was committed and pushed as `de6a716` before any formal outer-fold fit or outer-label access. Five grouped outer folds then trained `FULL`, `VUS_ONLY`, `VUS_GLOBAL`, `VUS_LOCAL`, and `RANDOM_PLACEBO` independently with paired seeds. `FIXED_AVERAGE` remained untrained.

Every fold has an atomically written pretest seal covering thresholds, epochs, development-label hashes, and the five locked channel hashes. All outer outputs have exact 2,080-row Mind2Web and 1,581-row ScreenSpot-Pro coverage in every arm and variant. Fallback mismatches are zero. All variants selected 30 final epochs. FULL candidate-permutation maximum errors are `4.77e-7`--`7.15e-7`, below the frozen `1e-5` limit.

## 2. Primary and control results

All entries are FULL minus the named control. Benchmark columns are equal-arm Step-SR percentage-point differences with paired grouped 99% confidence intervals. The balanced column averages benchmark effects after MDE standardization.

| Control | Mind2Web delta, 99% CI | ScreenSpot-Pro delta, 99% CI | Balanced MDE units, 99% CI |
| --- | ---: | ---: | ---: |
| frozen VUS-SR | -0.41 `[-1.20,+0.40]` | +0.11 `[-0.20,+0.41]` | -0.257 `[-0.949,+0.429]` |
| VUS_ONLY | **-0.85 `[-1.56,-0.16]`** | +0.25 `[-0.08,+0.57]` | -0.523 `[-1.155,+0.101]` |
| VUS_GLOBAL | **-1.06 `[-1.72,-0.39]`** | +0.27 `[-0.02,+0.58]` | **-0.671 `[-1.257,-0.074]`** |
| VUS_LOCAL | **+0.91 `[+0.38,+1.44]`** | +0.03 `[-0.17,+0.24]` | **+0.771 `[+0.304,+1.238]`** |
| RANDOM_PLACEBO | -0.42 `[-1.08,+0.24]` | +0.21 `[-0.06,+0.48]` | -0.200 `[-0.772,+0.367]` |
| FIXED_AVERAGE | **+1.33 `[+0.46,+2.19]`** | +0.22 `[-0.15,+0.57]` | **+1.252 `[+0.486,+1.997]`** |

FULL safe Step-SR is 34.51% on Mind2Web and 64.37% on ScreenSpot-Pro. Frozen VUS-SR remains 34.92% and 64.26%. Every ScreenSpot arm meets the preregistered 99% noninferiority bound, but Mind2Web does not improve.

## 3. Frozen gates

| Gate | Result | Decision |
| --- | --- | --- |
| DELTA-1 | Mind2Web CI lower bound is -1.20 pp | FAIL |
| DELTA-2 | all ScreenSpot arm CI lower bounds exceed -0.70 pp | PASS |
| DELTA-3 | balanced CI versus VUS-SR crosses zero | FAIL |
| DELTA-4 | balanced CI versus VUS_ONLY crosses zero | FAIL |
| DELTA-5 | balanced CI versus RANDOM_PLACEBO crosses zero | FAIL |
| DELTA-6 | four channels exceed 0.10 mass in 5/5 folds; equivariance passes | PASS |

Only DELTA-2 and DELTA-6 pass. Evidence complementarity requires all six gates, so the method is not supported.

## 4. Failure mechanism

The failure is in candidate ranking and channel admission, not merely the safe threshold:

- on Mind2Web, FULL direct accuracy is 34.87%, below VUS_ONLY at 35.36% and VUS_GLOBAL at 35.83%; the safe policy still improves FULL over its 31.92% fallback, but cannot recover the ranking loss;
- FULL significantly beats VUS_LOCAL, showing that global/binding evidence rescues local-only evidence;
- FULL is significantly worse than VUS_GLOBAL, showing that adding fine/context evidence back into the useful global/binding pair causes negative transfer;
- without retraining or recalibration, dropping fine or context from FULL raises Mind2Web safe Step-SR by +0.83 and +0.56 pp; dropping VUS or global lowers it by -1.31 and -1.05 pp. ScreenSpot dropout effects are near zero;
- mean FULL gate mass is close to 25% for every real channel in every fold. Its marginal normalized entropy is 0.9996--1.0000. This does not prove per-example gate collapse, but it shows that DELTA-6's mass criterion cannot distinguish selective utility routing from balanced channel use.

The mandatory controls therefore reject both desired claims: FULL does not beat the same-capacity VUS_ONLY model, and it does not beat the RANDOM_PLACEBO model. The supported interpretation is narrow: independently locked channels contain different signals, but this shared simplex-gate/listwise objective does not learn utility-preserving channel admission; local channels continue to dilute Mind2Web global/binding evidence even after decision-level separation.

## 5. Decision boundary

DELTA is closed without post-result tuning. One-call distillation and GUI-Odyssey confirmation are cancelled because they were authorized only after DELTA-1 through DELTA-6 all passed. VUS-SR remains the strongest defensible learned aggregator from this sequence.

`VUS_GLOBAL` is a diagnostic control, not a post-hoc selected method. Any future evidence-admission study requires a new result-free preregistration and independent confirmation; it cannot tune DELTA masks, losses, thresholds, or gate regularization on these results.