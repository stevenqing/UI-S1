# Closing Main Tables

## Primary ScreenSpot-Pro table

| Configuration | Forwards | B3 | M1 / reported selector | pass@N | Source |
|---|---:|---:|---:|---:|---|
| GTA1 bare single view | 1 | — | 49.40 | — | local reproduction; model-card anchor 50.1 |
| GTA1 + official MVP candidates | 12 | 60.09 | 60.40 | 72.80 | local, fixed GTA1 lineage |
| Qwen3 single lineage | 12 | 56.93 | 56.80 | 74.19 | local, shared GTA1 geometry |
| UI-TARS single lineage | 12 | 51.87 | 52.44 | 70.02 | local, shared GTA1 geometry |
| MVP GRPO selector (4B, trained) | — | — | 62.80 | — | published paper only; different environment; excluded from calculations |
| Mixed lineage | 12 | 63.69 | 63.82 | 79.19 | local, 3 lineages x 4 views |
| Mixed lineage | 24 | 62.56 | 64.07 | 81.72 | local, one-sided budget extension |

Only the local N=12 rows are direct fixed-budget comparisons on the same 1,581 examples and candidate geometry. The bare row uses one forward, Mixed N24 uses 24 forwards, and the published 62.8 row is not from our environment; none enters a paired comparison with those rows.

The two additional lineages are not stronger hidden substitutes. At displayed two-decimal table precision, Qwen3 and UI-TARS individually trail GTA1 M1 by 3.60 and 7.96 percentage points (exact unrounded gaps: 3.61 and 7.97 points), yet the mixed N12 pool reaches 63.82.

The published MVP anchor 61.7 is audited through the separate official-code reproduction at 61.35 (0.35 points lower). It is not an anchor for the N12 GTA1 row at 60.09/60.40, whose candidate budget and evaluation are different.

We do not claim absolute ScreenSpot-Pro SOTA. The supported statement is: under the same local backbone inventory, shared candidate geometry, examples, and 12-forward test-time budget, the mixed pool exceeds every internally evaluated single-lineage configuration, including unchanged B3 and fold-local M1 selectors.

## Lineage-count and composition table

| N12 pool | B3 | M1 | pass@12 | Mean dev failure kappa |
|---|---:|---:|---:|---:|
| GTA1 + Qwen3 | 63.76 | 63.88 | 78.56 | 0.647 |
| GTA1 + UI-TARS | 62.05 | 62.05 | 75.65 | 0.606 |
| Qwen3 + UI-TARS | 59.27 | 60.40 | 76.72 | 0.630 |
| GTA1 + Qwen3 + UI-TARS | 63.69 | 63.82 | 79.19 | 0.594 |

The best two-lineage M1 pool (GTA1 + Qwen3, 63.88) is statistically close to the three-lineage point (63.82). This table supports a correlation/composition interpretation rather than a monotonic model-count claim.
