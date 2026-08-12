# Amendment 001: Exact CIVA-A0 Execution

Date: 2026-08-11

Timing: frozen after result-free data/model unit tests and full public-data assembly; before loading any private label for a CIVA fit.

## Learner details

Each variant and expert has two independent binary heads for rescue and harm. Split weights retain the preregistered benchmark/row/arm ratios and are multiplied to mean one before `HistGradientBoostingClassifier.fit`, so regularization is not changed by the number of folds in a split. A single-class head returns its weighted empirical probability and does not fit a tree.

All variants in the same split use the same base seed and are independently initialized. There is no checkpoint selection, early stopping, probability calibration, class reweighting, hyperparameter search, or model sharing.

## Hard policy and threshold ties

For real variants, select the expert with maximum predicted `P(rescue)-P(harm)`; ties use frozen expert order global, fine, context. Switch only if its candidate differs from the VUS-binding candidate and its score is at least the selected threshold.

Candidate thresholds are infinity, zero, and the 0.0--1.0 deciles of positive scores among changed development OOF rows. Select maximum development net accuracy delta. Exact ties choose the largest threshold, so infinity preserves baseline whenever switching has no measured advantage.

## Matched-random control

Within each outer-test benchmark/arm cell, `MATCHED_RANDOM` switches exactly as many rows as `REAL_FULL`. Rows are ordered by SHA-256 of the frozen seed and sample key. On selected rows, one real expert whose candidate differs from baseline is chosen by a second SHA-256 value. No label, learner score, expert success, website/application identity, or outer-fold feature enters this control.

## Result boundary

CIVA-A0 evaluates raw channel-direct policies, not frozen VUS-SR or trained DELTA variants. A positive result would authorize a separate policy-level protocol; it would not itself establish improvement over VUS-SR.