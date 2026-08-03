# Source Bias and Lineage-Normalized Aggregation

Date: 2026-08-03

Status: result-blind preregistration. No B1, B2, B3x or B4 result existed when this protocol was frozen. No new model inference is used.

## B1 source-bias gate

For every pool/rule/stratum enumerated in `configs/b1_pools.yaml`, recover a winner-source label and compare its observed distribution with the pool's candidate-source proportions using a chi-square goodness-of-fit test. Report Cramer's V and multinomial standardized residuals.

Winner source attribution:

- B3: the actual candidate selected by official MVP group scoring and coverage tie-break.
- M1: the actual candidate index returned by fold-local CCM.
- graph centroid: compute the frozen graph-centroid output, then attribute it to the nearest real candidate by Euclidean distance; ties use original candidate order. The winning graph-component source composition is reported separately.

Correct and incorrect rows are tested separately using each rule's own prediction correctness.

B1 passes when 7B Uniform Mixed N12 B3 incorrect rows have p<0.001 and positive GTA1 standardized residual, and 72B Uniform Mixed N8 has the same direction. If only 72B passes, B2 remains executable but the mechanism scope is 72B-local.

## B4 attribution

Run three indirect, zero-GPU diagnostics:

1. Compare winner-source distributions for the three-lineage full-image-only pool (view 0, three candidates) and the views 1-3 crop-only pool (nine candidates).
2. Within each lineage, report median pairwise distance among its four view0-3 candidates, normalized by image diagonal, with paired application-group bootstrap comparisons.
3. For Uniform Mixed pools where lineage candidate counts differ by at most one, deterministically downsample overrepresented GTA1 candidates to the minimum lineage count, preserving earliest view order, and recompute B3. This is descriptive and is not a method result.

Proposal-source attribution is supported only when GTA overrepresentation is weaker on view 0 than views 1-3 and GTA within-lineage distance is lower than both alternatives. Otherwise source bias is reported as a heterogeneous-pool aggregation effect, not specifically caused by the proposer.

## B2 lineage-normalized aggregation

Every lineage is reduced to one representative; three lineage representatives then receive one lineage-level vote each. All 21 reduction/decision combinations are frozen in `configs/b2_variants.yaml`.

Outer folds are the existing five application-group folds. For outer fold `f`:

- inner validation is original fold `(f+1) mod 5`;
- inner training is the remaining three folds;
- fit reduction/decision reliability parameters on inner training;
- evaluate all 21 variants on inner validation and select by B3 accuracy, then M1-style point accuracy is not used, then frozen variant order;
- refit selected variant parameters on all four outer-development folds;
- evaluate exactly one selected variant on outer held-out fold.

Thus the claim-bearing nested result has one prediction per held-out row. The 21 variants are also evaluated cross-fitted without nesting as a descriptive sensitivity grid and are never used as the headline maximum.

### Reduction definitions

- R1: centroid of the largest official complete-link 14-pixel group; tie uses first group.
- R2: Euclidean geometric median by Weiszfeld iterations, tolerance 1e-6, maximum 100 iterations; exact-data-point singularity returns that point.
- R3: medoid minimizing summed Euclidean distance; tie uses lower view index.
- R4: centroid of the largest 14-pixel graph-connected component; tie uses lower minimum view index.
- R5: candidate from the lineage's development-strongest view; tie uses lower view index.
- R6: view-0 candidate.
- R7: within the largest official group, centroid weighted by development `(lineage,view)` accuracy; tie/group semantics match R1.

### Decision definitions

- D1: centroid of the largest 14-pixel graph-connected component among lineage representatives; each lineage weight is one.
- D2: choose the representative with maximum sum of development lineage-reliability weights from representatives within 14 pixels; ties use higher own reliability then frozen lineage order. The output is a real representative.
- D3: if all three representative pair distances exceed 14 pixels, output the development-most-reliable lineage representative; otherwise apply D1.

If a lineage has no finite candidate point, it abstains. Existing traces use finite failure sentinels; sentinels remain candidates and are not silently removed.

B2 primary comparisons use unchanged candidate pools:

- 7B Uniform Mixed N12 against B3 63.69 and M1 63.82;
- 72B Uniform Mixed N8 against B3 41.24 and M1 52.12;
- best-single mandatory context: 7B Qwen3 bare 54.65 and 72B Qwen3.5 bare 71.41.

Primary success requires the nested result to exceed B3 by more than 0.007043345177520599 with positive 99% paired-bootstrap CI lower bound at both scales. Exceeding B3 but not M1 is reported as B3-bias correction only. B-K4 triggers when 72B nested LN remains below 71.41 best-single.

## B3x reclaim

B3x runs only if B2 primary success passes. Apply each outer fold's B2-selected/refit rule to CALA-S N12, NOA-static N12 and the R1 highest-disagreement Uniform Mixed N4/N24 pools. Report paired grouped intervals. B3x unified-mechanism success requires all three original negative deltas to become nonnegative in point estimate.

## Statistics and boundaries

- 10,000 fold-stratified application-group bootstrap replicates, seed 20260803.
- 99% percentile intervals and plus-one one-sided p-values.
- best-single is mandatory in every main table.
- No new reduction or decision variant is added after results.
- Full-grid maxima are descriptive only.
- Existing R4 selective-accuracy results are preserved unchanged.
