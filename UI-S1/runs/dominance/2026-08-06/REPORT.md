# Post-B2 Dominance and Freeze Report

Date: 2026-08-06

## Outcome

The run completes D0 and the available ScreenSpot portion of D1. D2 is fail-closed because frozen row-level cross-benchmark traces are missing. No GPU inference was launched and no new aggregation method was introduced.

The final paper remains path B: a ScreenSpot-Pro diagnostic paper. The dominance-gap pattern is directionally present but does not satisfy the frozen law criterion and is not promoted to a main-text law.

## D0: R7 implementation fault

The historical weighted centroid omitted normalization by total weight. With the probe points `(0,0)` and `(10,0)` and weights `(2,1)`, the broken implementation returns `(10,0)` while the correct weighted centroid is `(3.33,0)`. This coordinate scaling explains the two-scale 2% R7 signature.

| Scale | Historical R7 D1/D2/D3 | Repaired R7 D1/D2/D3 |
|---|---|---|
| 7B | 2.28 / 2.15 / 2.21% | 62.62 / 61.99 / 61.80% |
| 72B | 2.66 / 2.40 / 2.34% | 51.61 / 64.64 / 64.83% |

The controlled comparison on the recovered bank reruns the repaired frozen 21-method grid and the same grid with R7 removed. Nested predictions and selected variants are identical at both scales. R7 is never selected. D-K1 does not trigger and the B2 conclusion does not change.

The pre/post numbers are nevertheless both retained because the historical and recovered studies also differ in bank bytes and, for combined-24, method set.

## D1: dominance-gap analysis

The ScreenSpot bank is frozen to views 0--11, yielding 36 actions. Exhaustive enumeration evaluates:

- 432 two-lineage pools;
- 1,728 three-lineage pools;
- 2,160 pools total;
- one action per retained lineage;
- 1,581 identities per pool;
- frozen B3 and fold-local M1.

Each pool records best and second member, dominance gap, mean member quality, mean pairwise failure kappa, and aggregation delta over best.

| Outcome | Raw Spearman | 99% CI | Controlled rank correlation | 99% CI |
|---|---:|---:|---:|---:|
| B3 minus best | -0.388 | [-0.430, -0.347] | -0.367 | [-0.410, -0.323] |
| M1 minus best | -0.482 | [-0.530, -0.434] | -0.499 | [-0.547, -0.450] |

Controls are mean member quality and mean pairwise failure kappa. Intervals use 10,000 pool-bootstrap replicates stratified by pool size.

The negative direction is robust, but neither raw coefficient reaches the frozen `rho < -0.6` threshold. More importantly, the combined three-benchmark gate cannot be computed. The result is `INCONCLUSIVE_BLOCKED_CROSS_BENCHMARK_ROWS`, not a positive law.

The motivating 7B `63.69%` point is B3 while the 72B `70.52%` point is nested LN. They are explicitly excluded from a shared correlation because the aggregators differ.

## D2: cross-benchmark preflight

The frozen runner requires `runs/complementarity/2026-07-30/rows.parquet`. The manifest records 102,054 tidy rows and an expected parquet SHA, but the file is absent. A deterministic rebuild was attempted from locked summaries and failed at the first absent row trace:

`runs/androidcontrol-rft/2026-07-29/artifacts/ui-agile-3b/low/predictions.jsonl`

The other lane directories likewise retain only aggregate score/audit files. Those files cannot reconstruct joint correctness, candidate coordinates, pairwise failure kappa, or mixed-pool predictions.

The frozen member-quality anchors and confounds are preserved in `d2_cross_benchmark_status.json`, but mixed metrics remain null. D-K3 is not adjudicated. No Mind2Web or AndroidControl transfer direction is claimed.

## Paper decision

The main positive evidence remains:

1. opposite-sign single-lineage and cross-lineage budget slopes;
2. a frozen weak lineage coexisting with a stronger mixed pool;
3. severe source bias at both scales;
4. a strong but scale-specific 72B lineage-normalization repair;
5. R4's narrowed selective-prediction signal strengthening.

The main negative evidence remains:

1. no absolute superiority over Qwen3.5 best-single;
2. no meaningful selector gain;
3. no cross-scale aggregation repair;
4. no shared-proposer causal attribution;
5. no general count-balancing result;
6. no dominance law under the frozen criterion;
7. no completed cross-benchmark transfer.

## Kill conditions

| Kill condition | Status | Consequence |
|---|---|---|
| D-K1: repaired R7 changes B2 conclusion | FALSE | retain B2 gate; report old and repaired grids |
| D-K2: controlled dominance effect disappears | NOT TRIGGERED ON SCREENSPOT; COMBINED BLOCKED | no law claim because magnitude gate and benchmark coverage fail |
| D-K3: Mind2Web direction opposes ScreenSpot | NOT ADJUDICATED | constrain claims to ScreenSpot-Pro |

## Deliverables

- `d0_r7_audit.py` -> `d0_r7_audit.json`;
- `d1_dominance_law.py` -> `d1_dominance_law.json`, `fig_dominance.pdf`;
- `d2_cross_benchmark_audit.py` -> `d2_cross_benchmark_status.json`;
- `FREEZE.md` is the sole writing authority;
- `STATUS.json` is the machine-readable decision state.
