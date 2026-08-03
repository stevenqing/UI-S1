# Amendment 001: Full Source-Bias Spec Extension

Date: 2026-08-03

Status: frozen after the original 21-variant study and before computing any result added by this amendment. Existing results remain immutable historical evidence. This amendment closes the gap to the full requested specification.

## B1 two-stage decomposition

For B3 and graph centroid, decompose source amplification on each stratum into:

1. cluster formation amplification = winning-set source share / full-pool source share;
2. within-cluster representation amplification = final-winner source share / winning-set source share.

The winning set is the official B3 winning complete-link group or the graph-centroid winning connected component. M1 has no winning-set abstraction and is marked `NOT_APPLICABLE` for this decomposition; its final source goodness-of-fit test remains required.

The 72B B3 incorrect N8 anchor must reproduce winning-group counts GTA/Venus/Qwen3.5 = 1374/1000/370 and final winners = 872/52/5.

Add 72B Uniform Mixed N9 and N12 in view-major, model-minor order. Both have equal lineage slots, 3/3/3 and 4/4/4. The primary divisible-slot controls are 7B N12/N24 and 72B N9/N12.

## B4 random count balancing

For every Uniform Mixed pool whose lineage counts differ by one, perform 10,000 seeded random global action-subset draws. Each draw retains the minimum count per lineage, samples without replacement within each overrepresented lineage, and evaluates B3 over all 1,581 rows. Report the accuracy distribution and source-bias distribution. The previously reported earliest-view deterministic balance remains a sensitivity check, not the random-discard result.

## R0 minimal intervention

R0 preserves B3 official group formation and group scoring. Only the output point from the winning group changes:

- R0a: centroid of all winning-group members;
- R0b: compute one centroid per lineage present in the winning group, then average lineage centroids equally;
- R0c: compute one centroid per lineage present in the winning group, then average lineage centroids using outer-development lineage reliability.

R0b/R0c therefore do not restore repeated votes through lineage member count. Finite failure sentinels are retained exactly as in the original study.

## Nested selection

The combined nested selector evaluates 24 methods in frozen order: R0a, R0b, R0c, followed by the existing R1_D1 through R7_D3 order. It uses the already frozen outer/inner split and tie breaks.

A second R0-only nested selector chooses among R0a/R0b/R0c under the same split. It tests whether changing only the within-cluster representative is sufficient.

Both selectors refit development reliability on full outer development and evaluate once on outer test. All 24 cross-fitted scores are descriptive only. Report min, median, max, nested-result percentile position, and outer-fold selection frequencies.

## Gates

Original B2 success criteria remain unchanged and apply to the combined 24-method nested result. R0-only is a separate minimal-intervention claim. B3x runs only if combined B2 passes both scales. B-K5 triggers when R0-only fails but combined B2 succeeds.
