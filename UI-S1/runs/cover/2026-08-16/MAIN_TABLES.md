# COVER Main Tables

## Arm A target coverage

| Target-center crop coverage | Rows | Fraction | B3 accuracy |
| --- | ---: | ---: | ---: |
| common_11 | 931 | 58.89% | 81.95% |
| partial_1_10 | 425 | 26.88% | 57.41% |
| uncovered_0 | 225 | 14.23% | 0.00% |

## Arm A row-class cross-table

| Spatial stratum | Selected correct | Recoverable | Zero candidate-success coverage |
| --- | ---: | ---: | ---: |
| common_11 | 763 | 89 | 79 |
| partial_1_10 | 244 | 102 | 79 |
| uncovered_0 | 0 | 54 | 171 |

## Arm B direct dependence

| Benchmark | Within-model phi | Cross-model phi | Phi N_eff |
| --- | ---: | ---: | ---: |
| ScreenSpot-Pro | 0.672 | 0.577 | 1.573 |
| Mind2Web | 0.541 | 0.360 | 2.181 |
| AndroidControl reference | 0.895 | 0.398 | NA |

## Arm B source/stage trend

| Benchmark | Within-model cross-slot | Cross-model matched-role | Cross-model unmatched-role | Ordering |
| --- | ---: | ---: | ---: | --- |
| ScreenSpot-Pro | 0.672 | 0.632 | 0.558 | within_model_cross_slot > cross_model_matched_role > cross_model_unmatched_role |
| Mind2Web | 0.541 | 0.392 | 0.350 | within_model_cross_slot > cross_model_matched_role > cross_model_unmatched_role |
