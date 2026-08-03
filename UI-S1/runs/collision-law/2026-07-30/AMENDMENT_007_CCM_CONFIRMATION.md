# Preregistration Amendment 007: Collision-Calibrated Mode

Date: 2026-07-31

Frozen after commit `015fd5e`, after the A5a leave-one-out K3 retrial, and before any calibrated A5 result or W4 inference.

## Motivation and status of prior results

The collision law states that aggregation value is governed by truth concentration relative to error collision in the evaluator metric space. W1 used collision only as a descriptive and predictive statistic. Collision-Calibrated Mode (CCM) promotes the same quantity to a decision statistic and to the fixed-budget allocation objective.

Original A3 included a non-comparable self-vote: an AndroidControl coordinate candidate received self-kernel `1 / rho0 = 3.256`, while a parameterless candidate received self-kernel `1`. A5a removes only `i = j` terms. Its frozen-fold retrial is already complete in `a5a_retrial.json`: A5a improves over A3 by 2.44 pp on AndroidControl Low and 5.95 pp on AndroidControl High, but remains below sequential A2 by 1.37 and 3.16 pp respectively. Mind2Web is unchanged. K3 therefore remains triggered after implementation correction. The separate A2-versus-A0 collision tax remains -7.03 pp on AndroidControl Low and -1.32 pp on AndroidControl High.

W1 and W2 are discovery data for A5. W4 AndroidControl-Curated is the frozen post-definition confirmation set. MM-Mind2Web-v2 corrected labels will be sought as the Mind2Web confirmation set. If they cannot be obtained with auditable identities and evaluator semantics, all Mind2Web A5 evidence remains discovery-stage and the paper states this limitation.

## Candidate and source definitions

The candidate set is the parsed model/view predictions themselves. CCM never synthesizes a coordinate or string. Candidate order and lexical tie breaks remain those of W1.

Each action has exactly one candidate class, with precedence:

1. `string-bearing` if the evaluator requires a string;
2. `coordinate-bearing` if it requires a coordinate and not a string;
3. `parameterless` otherwise.

This precedence assigns Mind2Web `TYPE` and `SELECT` to `string-bearing`, because their success requires both element membership and string correctness. The parameterless class is kept separate so hedge actions such as AndroidControl `wait` cannot be averaged with coordinate-bearing actions.

Source family is the released lineage prefix: TongUI, UI-TARS, UI-AGILE, GUI-R1, UI-R1-E, CogAgent, and any later confirmation source's declared released lineage. Views retain the base model family and model identity.

For an ordered candidate-voter pair `(i, j)`, excluding `i = j`, pair type is:

1. `same-model-diff-view` if base model identity is equal and view identity differs;
2. `same-family` if family is equal but base model identity differs;
3. `cross-family` otherwise.

## Fold-local calibration

For candidate `i`, let `C_i` be whether that exact candidate succeeds under the locked evaluator and let `u_ij = k(y_i, y_j)` be the existing product-kernel similarity. Labels are read only on calibration folds. Test-time scoring uses only predictions, source identities, and frozen calibration tables.

Source reliability uses a Laplace-smoothed MAP prior:

`r_s = (successes_s + 1) / (rows_s + 2)`.

Each likelihood-ratio table estimates:

`ell(u) = log p(bin(u) | C_i = 1) - log p(bin(u) | C_i = 0)`.

The table has eight equal-frequency bins whose boundaries are empirical similarity quantiles on the calibration rows. Bin probabilities use add-one smoothing separately for successful and failed candidates. A cell is eligible only with at least 32 successful and 32 failed directed pair observations. An ineligible nine-cell table backs off first to the same candidate class pooled across pair types, then to the pool-wide table. The same eight-bin/add-one rule and minimum counts apply at every level.

Quantile bin membership is invariant under every strictly monotone reparameterization of `u`; therefore the calibrated decision is invariant to such kernel transformations. This, not an uncalibrated zero-tuning claim, is the method property. The honest method description is: kernel-zero-tuning, no model training, fold-local collision calibration of at most nine one-dimensional density ratios, with decisions invariant to strictly monotone kernel reparameterization.

CCM candidate score is:

`S_i = logit(r_source(i)) + evidence_i`.

The best-single source's prediction is already a candidate. If pairwise log likelihood ratios carry no net evidence, the reliability prior returns the best source without an external fallback rule.

## Frozen ablation ladder

- `A5a_LOO`: leave-one-out observed kernel density, already evaluated solely to retrial K3.
- `A5b_MAP_pooled_LR`: source MAP prior plus one pool-wide likelihood-ratio table; no pair-type or candidate-class conditioning and no family de-duplication.
- `A5c_MAP_nine_LR`: source MAP prior plus the fixed `3 pair types x 3 candidate classes` likelihood-ratio tables and frozen backoff; every voter contributes separately.
- `A5d_MAP_nine_LR_family`: A5c, except log-LR evidence from voters in the same family is averaged within family before family contributions are summed. Pair-type conditioning and family de-duplication are distinct ablations.
- `A5d-risk`: A5d plus an explicit override threshold on `S_gap = S_winner - S_best-source-candidate`.

Main MAP-only A5b-A5d calibration uses all four non-test grouped folds. A5d-risk uses a nested split: for outer test fold `f`, threshold-dev is `(f + 1) mod 5`, and the remaining three folds fit priors and LR tables. Candidate thresholds are all observed nonnegative threshold-dev `S_gap` values plus infinity. Select the smallest threshold whose threshold-dev Step SR is at least best-single Step SR; ties in empirical gap values are inclusive. Infinity always reproduces best single, so the constraint is feasible. No non-inferiority margin or manually chosen threshold is introduced.

## Capacity comparison with E2

E2 T2 used all model outputs, generic consistency features, and gradient-boosted heads with thousands of effective parameters. CCM receives a subset of that information. Its sufficient statistic is fixed by the collision law, its additive log-LR form is fixed by Bayes, and it estimates at most nine one-dimensional ratios plus source priors. Comparison to E2 is therefore an inductive-bias and capacity comparison, not a claim that CCM has access to new information.

## P3-CCM allocation

The original kappa-only P3 remains a valid negative preregistered result. Its replacement is exploratory until confirmation.

At the same five-forward budget, fit frozen A5d CCM on each outer development set. Initialize with the highest development Step SR unit. At each step, simulate frozen CCM on the development rows and add the unit producing the greatest development Step SR increment. Break ties by higher standalone development Step SR, then unit key. Stop only at five units, even if all remaining increments are nonpositive, to preserve equal compute. Evaluate the selected five exactly once on the held-out fold. No handcrafted `r*T/(epsilon+E)` utility is used.

## K4 and success criteria

K4 triggers and removes the LR component from the method claim if either:

1. A5c does not strictly exceed A5a on Mind2Web discovery Step SR; or
2. A5c is significantly inferior to the poolwise-best frozen baseline on either AndroidControl pool under the paired one-sided exact McNemar test with Holm correction over the three pools at family-wise `alpha = 0.05`.

If K4 triggers, the method falls back to the law plus A5a/MAP selective aggregation; calibrated LR remains diagnostic only.

Full method success requires:

1. no pool is significantly inferior to its poolwise-best frozen baseline under the paired one-sided exact McNemar/Holm test;
2. at least two of three discovery pools are significantly superior under the paired one-sided exact McNemar/Holm test; and
3. the direction is preserved on each available frozen confirmation benchmark.

Frozen poolwise-best discovery baselines are A0 on AndroidControl Low and High and original A3 on Mind2Web visual.

## Required diagnostics

Every A5 result reports:

1. an override curve: fraction of rows where CCM changes the best-source prediction and conditional Step SR after override;
2. `S_gap` AUROC for selected-candidate correctness, compared with the existing negative-dispersion AUROC of 0.660 on Mind2Web;
3. paired wins and losses against A0, A2, A3, and A5a;
4. results for all candidate classes and pair types, without post-hoc class omission;
5. effective sample counts and backoff level for every LR table.

## Confirmation gate

W4 inference may begin only after this amendment is committed. W4 labels and predictions must not alter A5 definitions, bins, smoothing, backoff, source taxonomy, threshold selection, K4, or success criteria.