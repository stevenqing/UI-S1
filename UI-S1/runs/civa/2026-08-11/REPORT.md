# CIVA-A0 Conditional Incremental Value Admission Report

Date: 2026-08-11

Outcome: `CIVA_ADMISSION_NOT_SUPPORTED`

## 1. Protocol and integrity

CIVA-A0 tests whether information available before acquiring an additional evidence channel predicts that channel's incremental utility over the raw VUS-binding direct candidate. It is not a comparison against VUS-SR safe Step-SR.

The preregistration, result-free implementation, and runtime-only correction were published before valid model fitting as commits `e948157`, `fc402bb`, and `ca81746`. The first launcher invocation selected the wrong interpreter and all five workers failed at `import numpy`, before public-data import, private-label access, model fitting, or pretest creation. Those logs are retained under `invalid_runtime_001/`.

The corrected run completed five outer folds. Every fold has an atomically written pretest seal containing four development-label hashes, the sealed outer-label hash, five locked channel hashes, public-data hash, five feature/index hashes, implementation hashes, thresholds, and the learner contract. Matched-random switch counts equal REAL_FULL exactly in every outer benchmark/arm cell. Merged coverage is 2,080 Mind2Web rows and 1,581 ScreenSpot-Pro rows per arm and variant. Adjudication is byte-deterministic with SHA-256 `2f1cdea6c2a28de0c89557117d8c32e57cdfc040f3fda75b32ab36722070add0`.

## 2. Raw-direct admission results

| Policy | Mind2Web accuracy | Delta vs baseline | ScreenSpot-Pro accuracy | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| VUS-binding direct baseline | 30.49% | -- | 45.21% | -- |
| REAL_FULL | **32.07%** | **+1.57 pp** | **50.62%** | **+5.41 pp** |
| REAL_NO_TEXT | **32.37%** | +1.88 pp | **50.87%** | +5.66 pp |
| REAL_TEXT_ONLY | 31.15% | +0.66 pp | 48.73% | +3.53 pp |
| PLACEBO_FULL | 31.27% | +0.78 pp | 46.98% | +1.77 pp |
| MATCHED_RANDOM | 28.00% | -2.49 pp | 45.64% | +0.43 pp |

REAL_FULL minus baseline is:

- Mind2Web: +1.57 pp, 99% CI `[+0.79,+2.39]`;
- ScreenSpot-Pro: +5.41 pp, 99% CI `[+4.12,+6.81]`;
- balanced standardized effect: +5.163 MDE, 99% CI `[+4.020,+6.334]`.

REAL_FULL also beats matched-random and placebo in the preregistered balanced tests. It switches 21.98% of Mind2Web rows and 46.87% of ScreenSpot-Pro rows. Global semantic is selected most often: 1,606/1,829 Mind2Web switches and 2,012/2,964 ScreenSpot switches.

## 3. Failed gates

| Gate | Result | Decision |
| --- | --- | --- |
| CIVA-1 | balanced CI positive versus baseline | PASS |
| CIVA-2 | both benchmark equal-arm CIs positive | PASS |
| CIVA-3 | balanced CI positive versus matched random | PASS |
| CIVA-4 | balanced CI positive versus placebo | PASS |
| CIVA-5 | FULL versus NO_TEXT CI `[-1.161,+0.305]` MDE | **FAIL** |
| CIVA-6 | Mind2Web C-uni lower bound -1.01 pp < -0.61 pp margin | **FAIL** |

All six gates were required. The result therefore does not support CIVA-A0 promotion.

## 4. Mechanism

The positive part is real and specific: VUS-binding uncertainty plus public candidate geometry/action structure identifies many rows where a different independently locked channel is useful. REAL_FULL significantly beats both matched-random switching and a learned random-center placebo.

The preregistered semantic mechanism is not supported. Removing instruction features improves the point estimate by +0.30 pp on Mind2Web and +0.25 pp on ScreenSpot-Pro; FULL minus NO_TEXT has a balanced 99% CI crossing zero. Instruction hashing therefore adds no demonstrated held-out incremental utility beyond VUS state and public structure.

Safety is also not fully established. Mind2Web C-uni has a positive +0.29 pp point estimate, but its 99% CI `[-1.01,+1.59]` is wider than the frozen -0.61 pp noninferiority margin. The study cannot replace this uncertainty with a favorable aggregate result.

Finally, REAL_FULL remains far below frozen VUS-SR safe Step-SR because A0 deliberately operates on raw channel-direct policies: 32.07% versus VUS-SR 34.92% on Mind2Web and 50.62% versus 64.26% on ScreenSpot-Pro. It is a learnability diagnostic, not a stronger aggregator.

## 5. Decision

CIVA closes without post-result feature, learner, threshold, or gate changes. `REAL_NO_TEXT` is a diagnostic control and is not promoted post hoc. Per the frozen kill boundary, policy-level admission over VUS-SR, contrastive verification, VLM fine-tuning, and distillation are not run.

The defensible scientific conclusion is narrower: **pre-admission VUS uncertainty and candidate structure carry substantial information about raw expert utility, but instruction-conditioned uplift and uniform cell safety were not established.** VUS-SR remains the strongest defensible learned aggregator.