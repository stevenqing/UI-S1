# Amendment 004: Fold-Stratified Group Bootstrap

Date: 2026-08-01

Status: frozen after the first L2 execution stopped at its bootstrap validity gate, before any L2 result was emitted or written. L1 results were already available. No L2 metric or correlation was inspected when this amendment was made.

The dataset has 26 application groups allocated 5/5/6/5/5 across the frozen outer folds. A global 26-group bootstrap has approximately 1.6% probability of omitting at least one entire held-out fold. Such replicates cannot form the preregistered 40 fold-pool observations and caused fewer than 99% finite primary replicates.

The L2 application-group bootstrap is therefore stratified by frozen outer fold. Within each replicate and fold, sample that fold's application groups with replacement using the original number of groups in the fold. Concatenating the five strata preserves 26 sampled groups per replicate and guarantees that every development/held-out fold observation exists. Pool statistics, fold assignment, selectors, outcomes, seed, 10,000 replicate count, and 99% finite gate are unchanged.