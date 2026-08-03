# Amendment 004: X6 Held-Out Pool Prediction

Date: 2026-08-02

Status: frozen while X2 inference is running and before any X2 trace is loaded, scored, or summarized. Supersedes the X6 `BLOCKED_NO_HELDOUT_POOLS` status only if X2 Q2/Q4 complete their fixed-12 integrity gate. X5 remains unavailable.

## Observation unit

Use pool-by-outer-fold observations. Training consists of the eight frozen L2 pools times five folds, for 40 observations. Held-out validation consists of X2 Q2 and Q4 times the same five folds, for 10 observations. Q1/Q3 are excluded from validation because they are already among the L2 training pools.

## Unlabeled feature

For every row, compute Euclidean distance for all 66 unordered pairs among the 12 predicted points and divide each distance by the original image diagonal. Average over pairs, then average over rows in the outer development split. This `mean_pairwise_normalized_distance` uses no target bbox, success label, model identity, or test-fold outcome.

## Predictor

Fit one ordinary least-squares model with intercept on the 40 L2 observations:

`heldout_pass_at_12 = intercept + coefficient * dev_mean_pairwise_normalized_distance`.

No feature selection, transformation, regularization, pool fixed effect, clipping, or sign constraint is used. Freeze the fitted coefficient before calculating X2 held-out outcomes. The fit report includes training R-squared and training Spearman as descriptive quantities only.

## Validation

For each of the 10 X2 observations, compute the feature on its outer development rows, obtain the frozen OLS prediction, and pair it with pass@12 on that fold's held-out rows. X6 passes only when Spearman between predicted and actual held-out pass@12 is strictly greater than 0.7. Report p-value and all 10 records.

Ten observations from only two new pools are a minimal validation and must be described as low-power. If either X2 pool is incomplete, if fewer than 10 finite observations exist, or if the L2/X2 fold mappings differ, X6 remains blocked rather than falling back to in-sample correlation.
