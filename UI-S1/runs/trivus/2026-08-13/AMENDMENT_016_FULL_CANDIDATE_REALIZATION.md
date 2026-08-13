# Amendment 016: Full-Candidate Realization

Date: 2026-08-13

Timing: after commit `2f615f3e` and after the post-hoc headroom atlas. Existing formal labels are open, so this is an exploratory method-development protocol and cannot authorize confirmation or promotion.

## 1. Finding

The full candidate pool exceeds the strongest fallback by 24.29 percentage points on Mind2Web, 15.31 points on ScreenSpot-Pro, and 6.88 points on AndroidControl. The oracle union of all existing formal policy direct choices recovers only 3.17, 0.16, and 1.48 points respectively. The dominant remaining gap is candidate ranking, not final fallback override.

The prior target assigns all mass to KEEP whenever fallback is correct. It therefore does not train absolute candidate correctness on those rows, including rows with other correct candidates. The new primary objective must consume every valid candidate-success label.

## 2. Unified method

Each benchmark fits its own contextual candidate verifier using the same architecture, objective, cross-fitting, calibration, and evaluation protocol.

For every row, the verifier emits one success logit per valid candidate. Its loss is:

- row-normalized binary cross entropy over every valid candidate;
- plus `0.5` times the mean within-row positive-versus-negative softplus ranking loss.

Rows with all-negative or all-positive candidates contribute absolute BCE but no pairwise term. Candidate count is normalized within each row so 12-candidate rows do not receive four times the mass of 3-candidate rows. Existing benchmark/cell/group weights are applied after row normalization.

The direct candidate is the valid candidate with maximum calibrated success score. A separately cross-fitted incremental-utility gate decides whether to override the benchmark's strongest frozen fallback.

## 3. Required cross-fitting

Candidate verifier predictions used to train the override gate must be OOF. The override gate may use candidate scores, score entropy, top-two margin, fallback score, candidate-set disagreement, and frozen public context features. It may not use in-sample candidate predictions or success labels at inference.

## 4. Development boundary

The loss primitives may be synthetic-tested now. Before any real-data optimizer step, a separate frozen config must bind exact folds, features, architecture, optimizer, seeds, class calibration, threshold grid, strongest baseline identities, artifacts, and the untouched confirmation source.