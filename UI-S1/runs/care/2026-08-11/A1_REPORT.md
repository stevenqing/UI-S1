# CARE A1 Structural Acquisition Router Report

Date: 2026-08-11

Outcome: `CLOSE_ROUTING`

## Result

Five outer folds all selected C-cond as the nested best-static acquisition on both benchmarks. The learned structural router used only the six candidates shared across all four arms.

| Selected router − nested static C-cond | Mind2Web | 99% CI | ScreenSpot-Pro | 99% CI |
| --- | ---: | ---: | ---: | ---: |
| pass@12 | **−1.01 pp** | **[−2.10,0.00]** | +0.06 pp | [−0.81,+1.05] |
| frozen VUS-SR safe Step-SR | −0.53 pp | [−2.04,+0.97] | **−1.27 pp** | [−3.28,+0.41] |

All three frozen A1 gates fail:

- no benchmark has a positive 99% pass@12 lower bound;
- the other-benchmark noninferiority condition is not satisfied;
- ScreenSpot safe loss exceeds its 0.70 pp MDE.

The corrected router captured −16.7% of oracle routing coverage gain on Mind2Web and 1.7% on ScreenSpot. Choice frequencies were diffuse rather than collapsing to the robust C-cond arm. Final epoch selections were 50/50/46/50/46; no post-result capacity or hyperparameter search is authorized.

The first A1 implementation accidentally omitted the frozen cross-fitted source-reliability scalar and was invalidated before commit. Correction 002 added public source metadata, fold-local Beta-smoothed reliability, training-row leave-one values, and fixed validation/test values. The corrected run reported here supersedes it. The invalid preliminary adjudication is retained outside git with SHA-256 `e60f54ff6a3e6e0aa8a7774bede08d55068e67c0fa5e6fa09a49c34339ff8360`.

## Interpretation

The retained counterfactual bank proves that row-wise arm routing has oracle value, but six-candidate structural state does not identify that value out of sample. The useful arm appears to depend on latent task/image semantics or on information not available until stage 2. A structural value-of-information router is therefore not a supported contribution.

CARE's acquisition-routing branch is closed. This result does not support adding stage-2 information to a stage-1 router, because that would violate the decision boundary. It also does not support selecting arms by downstream VUS test outcomes.

The independent evidence diagnosis remains valid: candidate-ranking gaps are 18.52 pp and 14.60 pp, small targets are harder, and unique-correct recall is poor. That mechanism is separated into the post-A1 RAVEL protocol rather than silently modifying CARE.
