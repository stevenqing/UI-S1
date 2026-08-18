# OWIN Oracle Coverage Measurement Report

Date: 2026-08-17

Outcome: `OWIN_O_I2_GT_ORACLE_NON_DEPLOYABLE`

OWIN is a post-selection, single-benchmark measurement, not a method. Every Arm A value below is `GT_ORACLE_NON_DEPLOYABLE`: GT bbox geometry constructs the windows and no runtime placement rule is implied.

## Arm A: GT_ORACLE_NON_DEPLOYABLE pool measurement

| Stratum | Existing B3 | Raw oracle-pool B3 | Corrected oracle-pool B3 |
| --- | ---: | ---: | ---: |
| uncovered_0 | 0.00% | 50.34% | 46.40% |
| partial_1_10 | 57.41% | 64.58% | 60.64% |
| common_11 | 81.95% | 85.90% | 81.95% |

Corrected B3 perfect-coverage opportunity is **+7.470 pp**, 99% CI [+2.855, +11.518] pp. It maps to **O_I2** under the frozen 5/10 pp thresholds. Constant-shift is heterogeneous and residual pool-dependence comparability is unavailable or indeterminate in affected units.

| Stratum | Existing M1 | Raw oracle-pool M1 | Corrected oracle-pool M1 | Existing single | Raw zero-jitter single | Corrected single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uncovered_0 | 0.00% | 49.61% | 45.19% | 0.00% | 43.89% | 38.43% |
| partial_1_10 | 48.47% | 65.80% | 61.38% | 30.14% | 62.57% | 57.11% |
| common_11 | 80.45% | 84.87% | 80.45% | 76.90% | 82.36% | 76.90% |

Corrected M1_ccm gain is +9.903 pp, 99% CI [+5.615, +13.943] pp. Corrected single-forward gain is +12.719 pp, CI [+7.953, +17.787] pp. Neither drives O-I. Both remain GT-oracle measurements subject to constant-shift and residual dependence limitations.

Small-target calibration sensitivity changes the B3 opportunity to -0.074 pp, CI [-8.334, +6.010] pp. It does not drive O-I and is shown beside the primary value because the constant-shift assumption is violated by target-size heterogeneity.

### Named limitations

Constant-shift is not validated. Common small-minus-large calibration heterogeneity is +14.924 pp, CI [+4.500, +27.351] pp, labeled `CONSTANT_SHIFT_SIZE_HETEROGENEITY_DETECTED`. Raw and corrected values must remain adjacent; small-target sensitivity is reported beside the primary estimate.

Residual pool-dependence comparability could not be quantified for the affected stratum(s). Affected units: uncovered_0/fold0:UNDEFINED_DEGENERATE_COMPARATOR, uncovered_0/fold1:UNDEFINED_DEGENERATE_COMPARATOR, uncovered_0/fold2:UNDEFINED_DEGENERATE_COMPARATOR, uncovered_0/fold3:UNDEFINED_DEGENERATE_COMPARATOR, uncovered_0/fold4:UNDEFINED_DEGENERATE_COMPARATOR, common_11/fold0:DEPENDENCE_MATCH_INDETERMINATE, common_11/fold1:DEPENDENCE_MATCH_INDETERMINATE, common_11/fold2:DEPENDENCE_MATCH_INDETERMINATE, common_11/fold3:DEPENDENCE_MATCH_INDETERMINATE, common_11/fold4:DEPENDENCE_MATCH_INDETERMINATE.

No historical N_eff value substitutes for unavailable matched diagnostics. These limitations apply beside every pool-level oracle opportunity.

## Arm B: fixed equal-budget geometry

| N | Median union | Center covered | Full-bbox covered | Factorized G_N | 99% CI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 78.34% | 1056/1581 | 993/1581 | +4.132 pp | [+1.131, +6.761] pp |
| 5 | 89.27% | 1276/1581 | 1236/1581 | +5.504 pp | [+1.506, +9.296] pp |
| 6 | 100.00% | 1332/1581 | 1265/1581 | +6.038 pp | [+1.631, +10.242] pp |
| 7 | 100.00% | 1418/1581 | 1381/1581 | +6.094 pp | [+1.820, +10.326] pp |
| 8 | 100.00% | 1482/1581 | 1418/1581 | +6.529 pp | [+1.893, +11.290] pp |
| 9 | 100.00% | 1531/1581 | 1497/1581 | +7.097 pp | [+1.988, +12.808] pp |
| 10 | 100.00% | 1536/1581 | 1474/1581 | +7.099 pp | [+1.985, +12.813] pp |
| 11 | 100.00% | 1556/1581 | 1519/1581 | +7.340 pp | [+2.122, +13.040] pp |

Frozen saturation status is `N_star=NONE`. Existing 11-crop median union is 32.19%; fixed tiling reaches the values above without model signals. Factorized G_N is descriptive, not observed deployable gain.

## Execution and boundaries

Formal execution retained exactly 6,000 traces with zero final-shard failures. Passing smoke retained 36 traces. Two earlier smoke failures and the first inference-input isolation failure remain retained. Token logprobs, entropy, margins, coordinate spans, decoded output, and hashes follow the extended trace policy.

OWIN changes no prior result or status. X2 and SPLIT remain closed, M2W is excluded, and any follow-up requires a new GT-free specification plus a net-benefit ledger on original-correct and crop-covered rows.
