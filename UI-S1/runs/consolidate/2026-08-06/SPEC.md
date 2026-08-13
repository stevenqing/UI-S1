# ScreenSpot-Pro Consolidation Spec

Date: 2026-08-06

Status: execution freeze. S-series analyses use existing banks only. Q-series designs are frozen before their result-producing inference. Mandatory controls may not be removed.

Upstream: `runs/dominance/2026-08-06/`, where D0 completed, D1 found directional negative association, and D2 is `BLOCKED_MISSING_ROW_LEVEL_TRACES`.

## Scope

The run has two halves:

1. S1--S6 strengthen and correct ScreenSpot-Pro claims using the existing action bank and D1 pools.
2. Q1/Q2 explore sequential candidate generation and a genuinely new verification channel. Q-series failure does not invalidate the diagnostic paper.

## S1: pool distribution

Compare every two- and three-lineage action pool with the strongest same-budget single-lineage pool. Report positive share, median, quartiles, empirical CDF, and extrema separately by pool size.

If positive share is below 60%, S-K1 triggers and the claim becomes: only some cross-lineage configurations outperform single-lineage allocation. The 12-forward reported pool must not be assigned a percentile in a 2/3-forward distribution.

## S2: held-out pool selector

Features may use prediction geometry, lineage composition, and development-fold member reliability only. Held-out and training pools must share no actions. Report primary, geometry-only, quality-only, and random-pool controls.

Success requires mean held-out Spearman above 0.7 and top-decile B3 above random with a positive 99% CI lower bound.

## S3: leave-one-lineage ablation

At 12 forwards, compare the complete three-lineage 4x3 pool against each two-lineage 6x2 pool on identical folds. If the full pool does not significantly outperform every ablation, S-K2 triggers and the weak-model claim is replaced by lineage diversity with third-lineage saturation.

## S4: slope hardening

Evaluate every integer budget N=2--16 for V-only and Mixed. Report dense slopes, paired N16-minus-N4 differences, and a Page-style global trend test. If dense V-only slope intervals cross zero, S-K3 triggers: paired endpoint differences become primary and slopes become supplementary.

Mixed-minus-V-only at N16 and N16-minus-N4 are distinct contrasts and must not be conflated.

## S5: decline attribution

Evaluate GTA1 B3 curves under 1,000 random permutations of views 0--15. Compare observed and randomized slope distributions. Persistent random-order decline supports correlation dominance; disappearance or reversal supports proposer rank decay.

## S6: anchors and dependence

Retain the SafeGround numerical anchor outcome exactly. Replace pool bootstrap for D1 with action-cluster bootstrap using the 36 frozen actions. Report raw and controlled correlations without causal variance language.

## Q1: sequential candidate generation

The result-blind arm definition is frozen in `configs/q1_arms.yaml`.

- Stage 1: three lineages by views 0/1, six forwards.
- Stage 2: two crops by three lineages, six forwards.
- C-uni: original Uniform Mixed views 2/3.
- C-cond: largest and second-largest cross-lineage consensus RoIs.
- C-rand: seeded random crops.
- C-self: GTA1 view0/view1-centered crops.

Primary success requires C-cond minus C-uni B3 above MDE `0.007043345177520599` and positive 99% CI lower bound. C-rand and C-self comparisons are mandatory. Pass@N is not a gate.

Q-K1 triggers if C-cond does not beat C-rand. Q-K2 triggers if C-cond does not beat C-self.

## Q2a: element-space mode

The result-blind variant is frozen in `configs/q2a_variant.yaml`. Add patch-28 element-cell clustering as method 25, preserve the original combined-24 result, and rerun the same nested selection. Do not replace the original result unless paired improvement has a positive interval.

## Q2b: cross-lineage verification

The result-blind mapping and prompt are frozen in `configs/q2b_verification.yaml`.

- Stage 1: six generation forwards.
- Stage 2: six non-self binary verification forwards.
- Positive candidates are passed to frozen B3; all-negative rows fall back to frozen B3 over all stage-1 candidates.
- Baseline: Uniform Mixed N12.
- Mandatory diagnostics: verification accuracy, yes precision, yes recall, parse failures, and seeded 50% random filter reference.

Primary success requires verified B3 minus Uniform N12 above MDE and positive 99% CI lower bound. Q-K3 triggers if verification accuracy does not exceed 50%.

## Execution order

1. Complete S1/S3/S5/S6.
2. Complete S2/S4/Q2a.
3. Adjudicate S-series wording.
4. Run Q1 sequentially by model, never loading different model families concurrently.
5. Run Q2b smoke then full verification sequentially by model.
6. Generate final tables only after all mandatory JSON outputs are `PASS`.

## General constraints

- Preserve frozen/recovery provenance.
- Use fail-closed identity, source, model-index, region, and output hash checks.
- No target fields may enter Q1 RoI generation or Q2b crop/prompt generation.
- No mandatory control may be omitted.
- External PID `2274` must never be signaled, killed, paused, or reprioritized.
