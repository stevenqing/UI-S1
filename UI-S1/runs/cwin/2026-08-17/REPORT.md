# CWIN Complementary-Window Stage-0 Report

Date: 2026-08-17

Outcome: `CWIN_STAGE0_W_G1_PASS_W_K5_ENDPOINT_AMENDMENT_REQUIRED_GPU_UNAUTHORIZED`

CWIN Stage 0 is a zero-GPU, post-selection ScreenSpot-Pro diagnostic. It changes no prior result and establishes no method gain.

## Gate result

All five outer folds selected K=4, the upper endpoint of the frozen grid. The nested outer-test micro-average strict oracle upper bound is **+20.62 pp**, above the W-G1 threshold of +0.70 pp. W-G1 passes. W-K5 also triggers because every fold selected an endpoint.

| Outer fold | Rows | Selected K | Drop-only B3 | Drop-only M1 | L4 upper | L4 conservative |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 314 | 4 | +0.637 pp | +0.637 pp | +25.796 pp | +5.903 pp |
| 1 | 311 | 4 | +0.322 pp | +0.643 pp | +16.077 pp | +2.736 pp |
| 2 | 337 | 4 | +1.780 pp | +2.077 pp | +17.804 pp | +3.315 pp |
| 3 | 306 | 4 | +0.000 pp | +1.634 pp | +10.131 pp | +0.000 pp |
| 4 | 313 | 4 | +0.639 pp | +0.000 pp | +33.227 pp | +6.880 pp |
| **Micro** | **1,581** | **4** | **+0.696 pp** | **+1.012 pp** | **+20.620 pp** | **+3.779 pp** |

The strict upper bound decomposes into observed drop-only B3 change **+0.696 pp** plus perfect rescue opportunity **+19.924 pp**. The conservative projection contributes **+3.084 pp** beyond drop-only. Neither rescue quantity is an observed complementary-window model result.

## All-K diagnostics

The initial Stage-0 output omitted the preregistered all-K L1/L3 table. `STAGE0_REPORTING_RECOVERY.md` was committed before reconstructing it from the retained all-K geometry and frozen candidate bank. The recovery changes no nested gate, K selection, geometry, aggregator, or authorization.

| K | Newly covered | Partial to higher | Lost all coverage | B3 drop-only | M1 drop-only |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 93 | 16 | 0 | +0.063 pp | +0.253 pp |
| 3 | 128 | 37 | 0 | +0.380 pp | +0.696 pp |
| 4 | 147 | 53 | 0 | +0.696 pp | +1.012 pp |

At selected K=4, 147 of the 225 originally crop-uncovered target centers become covered. No row loses all target-center crop coverage. Exact transition matrices for K=2, 3, and 4 are retained in `STAGE0_ALL_K.json`.

Drop-only improves both canonical aggregators at K=4. This is an observed candidate-pool effect, but it does not show that model outputs from the new windows will preserve or improve those gains.

## Geometry and retention

All 1,581 rows passed exact dimension checks. Existing and complementary crops are 1288 by 728 pixels. Exact all-integer top-left integral-image search, sequential zero-coverage updates, and recomputed greedy pairwise-IoU drops were used. The 1,581 selected window records and 1,581 exact uint8 coverage-count PNGs are retained with SHA-256 metadata.

## Decision boundary

W-G1 permits writing a separate Stage-1 execution amendment. It does not authorize GPU inference. W-K5 records that the frozen grid cannot identify an interior optimum; the grid must not be expanded after seeing this result. Any amendment must use the committed K=4 windows, GTA1 V-only N12 primary pool, frozen controls, trace policy, and explicit call budget. Stage 1 remains prohibited until that amendment is committed and separately authorized.