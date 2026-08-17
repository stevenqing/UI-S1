# COVER Proposer Coverage and Cross-Benchmark Dependence Report

Date: 2026-08-17

Outcome: `COVER_COMPLEMENTARY_SPEC_AUTHORIZED_COMMON_ORDERING_STRENGTH_SPLIT`

COVER is a zero-GPU post-selection diagnostic. It changes no prior result and makes no method claim.

## Arm A: crop-only coverage headroom

Arm A analyzes the 11 GTA1 proposer crop ranks. View 0 is the full-image baseline and is excluded from crop-only intersection/union. All three model lineages share these regions, so this is proposer-rank geometry, not lineage spatial diversity.

Across rows, the common 11-crop intersection occupies median **8.47%** of image area and the crop union occupies median **32.19%**. Median uncovered area is **67.81%**. All 1,581 exact uint8 coverage maps are retained.

| Target-center crop coverage | Rows | Fraction | B3 accuracy |
| --- | ---: | ---: | ---: |
| common_11 | 931 | 58.89% | 81.95% |
| partial_1_10 | 425 | 26.88% | 57.41% |
| uncovered_0 | 225 | 14.23% | 0.00% |

Low coverage (`partial_1_10 + uncovered_0`) contains **41.11%** of rows. Common-coverage B3 accuracy exceeds low-coverage accuracy by **+44.42 pp**, 99% CI **[+34.45,+53.96]**. A-G1 and A-G2 fail; A-G3 passes.

| Spatial stratum | Selected correct | Recoverable | Zero candidate-success coverage |
| --- | ---: | ---: | ---: |
| common_11 | 763 | 89 | 79 |
| partial_1_10 | 244 | 102 | 79 |
| uncovered_0 | 0 | 54 | 171 |

The 225 completely uncovered target centers have zero B3 successes; 54 are recoverable by another existing C-uni candidate and 171 have zero candidate-success coverage. Spatial coverage and candidate-success coverage are distinct.

The recorded human decision authorizes writing a complementary-window pilot specification only. GPU remains unauthorized. The design-only protocol is `runs/complementary-window/2026-08-17/SPEC.md` and requires a public gate plus a result-free net-benefit ledger before any inference.

## Arm B: cross-benchmark dependence

| Benchmark | Within-model phi | Cross-model phi | Phi N_eff |
| --- | ---: | ---: | ---: |
| ScreenSpot-Pro | 0.672 | 0.577 | 1.573 |
| Mind2Web | 0.541 | 0.360 | 2.181 |
| AndroidControl reference | 0.895 | 0.398 | NA |

M2W within-model phi is 0.541, fold range [0.526,0.553]. Cross-model phi is 0.360, range [0.351,0.370]. Its empirical phi $N_{\mathrm{eff}}$ is 2.181, above ScreenSpot-Pro's 1.573.

| Benchmark | Within-model cross-slot | Cross-model matched-role | Cross-model unmatched-role | Ordering |
| --- | ---: | ---: | ---: | --- |
| ScreenSpot-Pro | 0.672 | 0.632 | 0.558 | within_model_cross_slot > cross_model_matched_role > cross_model_unmatched_role |
| Mind2Web | 0.541 | 0.392 | 0.350 | within_model_cross_slot > cross_model_matched_role > cross_model_unmatched_role |

Both benchmarks share the descriptive ordering `within-model > cross-model matched-role > cross-model unmatched-role`. Dependence strength is benchmark-specific: M2W cross-model phi 0.360 is materially below ScreenSpot-Pro 0.577. COVER therefore supports a common source/stage distance ordering, not a universal high-correlation level.

The M2W trend is not a model-scale law. TongUI-7B, CogAgent-18B, and UI-TARS-7B differ in family, architecture, training, and size; slot roles also mix full, view1, and stage2 crops.

## Boundaries

The complementary-window direction is not X2 or SPLIT revival: it minimizes overlap, adds proposals directly, forbids flips/verifiers/two-mode restrictions, and has no GPU authorization. All label-dependent COVER quantities are evaluation-side. Existing statuses remain unchanged.
