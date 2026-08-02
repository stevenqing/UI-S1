# Amendment 001: X3 Statistical Operations

Date: 2026-08-02

Status: frozen after Diversity-Axis preregistration and X1's incomplete N4-only result, before reconstructing or inspecting any X3 per-row output or bootstrap slope.

The common budgets are N=4, 8, 12, and 16. For each pool and aggregation rule, fit ordinary least squares accuracy on forward count with an intercept. The slope unit is accuracy per forward. M1 CCM is the primary rule because it is the Allocation-Law primary selector; B3 is a strengthening analysis.

The primary X3 prediction passes when the 99% application-group bootstrap CI has V-only upper bound below zero and Mixed lower bound above zero under M1. B3 is reported separately and does not alter X-K2. X-K2 triggers when either primary strict inequality fails.

Bootstrap uses 10,000 replicates, seed 20260802, and the frozen outer-fold-stratified application resampling from Allocation-Law Amendment 004. Fold membership remains fixed. Per-row held-out B3 and fold-local M1 outputs are reconstructed by the frozen Allocation-Law loader and evaluator. Before bootstrap, every reconstructed point estimate must equal `L1_RESULTS.json` within 1e-15.

Area strata are equal-count quintiles of all 1,581 rows sorted by normalized GT bbox area `(bbox_width * bbox_height) / (screen_width * screen_height)`, with row ID as deterministic tie-breaker. Report Mixed minus V-only accuracy for every common budget under B3 and M1. Area is used only after inference and never enters candidate generation or selection.

V-only N=24 remains structurally unavailable because GTA1 has only 16-19 unique candidates per row. Mixed N=24 is displayed as a one-sided point only and is excluded from slopes, bootstrap, and bilateral pool comparisons.
