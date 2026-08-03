# Amendment 001: NOA-Stop Operations

Date: 2026-08-03

Status: result-blind, frozen after N5 passed and before any N4 NOA result.

N5 shows that high-disagreement rows retain positive pass@N headroom, so NOA-stop is eligible for the accuracy-plus-compute-saving claim.

## Static sequence

NOA-static uses the development failure-kappa matrix and generalized effective sample size

`N_eff(S) = |S|^2 / (1^T R_S 1)`.

It greedily selects the largest marginal N_eff action. Ties use higher development individual accuracy and frozen action order. The sequence is nested to N=16.

## Realized stopping signal

Before executing an action, its prediction coordinate is unavailable. After each selected action is executed, construct a realized correlation matrix from only the observed selected coordinates:

- diagonal entries are one;
- off-diagonal entries are the 14-pixel Gaussian evaluator-kernel similarities.

Compute realized generalized N_eff and its increment over the previous selected prefix. NOA-stop executes at least four actions. Starting after action four, it stops including the just-observed action when that action's realized marginal N_eff is strictly below the frozen threshold. It executes at most 16 actions.

This is a one-step delayed stopping signal, not a prediction of an unexecuted action. No unselected coordinate, confidence or label is read.

## Fold-local threshold

For each outer fold, candidate thresholds are all finite realized marginal values observed on development rows, plus negative infinity and positive infinity. For each threshold, simulate stopping on development rows and evaluate unchanged B3.

Eligible thresholds have mean development forwards at most eight. Select by:

1. highest development B3 accuracy;
2. lower mean forwards;
3. higher threshold.

The outer test fold uses that threshold without modification. Report test mean, median, p10/p90 and the full forward-count histogram.

The strong stopping claim compares NOA-stop B3 to Uniform Mixed N12 and requires mean forwards at most eight and an accuracy deficit no larger than the frozen 0.70 pp MDE. Paired confidence intervals are reported but do not replace this frozen tolerance rule.