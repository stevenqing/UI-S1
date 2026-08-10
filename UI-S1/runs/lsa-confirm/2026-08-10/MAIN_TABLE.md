# LSA No-Action Cross-Arm Confirmation

Outcome: `PARTIAL_TRANSFER`

模型、特征与阈值均由 C-uni discovery 冻结；本轮只在 C-cond/C-rand/C-self 上评估。

## Safe LSA − CEV-A

| Benchmark | Arm | CEV-A | LSA safe | Delta | 99% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| Mind2Web | C-cond | 32.45% | 33.41% | +0.96 pp | [−0.19,+2.12] |
| Mind2Web | C-rand | 31.83% | 32.16% | +0.34 pp | [−0.54,+1.23] |
| Mind2Web | C-self | 31.39% | 32.07% | +0.67 pp | [−0.63,+1.98] |
| ScreenSpot-Pro | C-cond | 66.48% | 66.60% | +0.13 pp | [0.00,+0.37] |
| ScreenSpot-Pro | C-rand | 61.10% | 61.10% | 0.00 pp | [0.00,0.00] |
| ScreenSpot-Pro | C-self | 65.15% | 65.09% | −0.06 pp | [−0.33,+0.20] |

## Equal-arm means

| Benchmark | Mean delta vs CEV-A | 99% CI | Gate |
| --- | ---: | ---: | --- |
| Mind2Web | +0.66 pp | [−0.02,+1.32] | T2 FAIL，CI 刚跨零 |
| ScreenSpot-Pro | +0.02 pp | [−0.09,+0.14] | T3 PASS |

相对 nested dev-selection 的 equal-benchmark/equal-arm standardized mean 为 1.56，99% CI [+0.57,+2.58]，T4 PASS；但 CEV-A 是更强的必要对照。

## Gates

| Gate | State |
| --- | --- |
| T1 all six cells safe | PASS |
| T2 Mind2Web equal-arm significant | **FAIL** |
| T3 ScreenSpot neutral | PASS |
| T4 better than dev-selection balanced | PASS |
| LT-K1 / LT-K2 / LT-K4 | false |
| LT-K3 | **true** |
