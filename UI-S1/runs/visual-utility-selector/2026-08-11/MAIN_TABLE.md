# Visual Utility Set Ranker Main Table

状态：`VUS_SET_RANKER_METHOD_CANDIDATE`

定位：当前最强经验统一聚合器；仍需独立 benchmark 确认。CEV-A 保持为最强 training-free 统一规则。

## Safe VUS-SR − CEV-A

| Benchmark | Arm | CEV-A | VUS-SR safe | Delta | 99% paired CI |
| --- | --- | ---: | ---: | ---: | ---: |
| Mind2Web | C-uni | 32.02% | **34.81%** | **+2.79 pp** | **[+0.94,+4.59]** |
| Mind2Web | C-cond | 32.45% | **34.62%** | **+2.16 pp** | **[+0.57,+3.80]** |
| Mind2Web | C-rand | 31.83% | **35.48%** | **+3.65 pp** | **[+1.75,+5.69]** |
| Mind2Web | C-self | 31.39% | **34.76%** | **+3.37 pp** | **[+1.59,+5.22]** |
| Mind2Web | Equal-arm mean | 31.92% | **34.92%** | **+2.99 pp** | **[+2.10,+3.91]** |
| ScreenSpot-Pro | C-uni | 63.88% | 64.14% | +0.25 pp | [−0.14,+0.64] |
| ScreenSpot-Pro | C-cond | 66.48% | 66.48% | 0.00 pp | [−0.82,+0.69] |
| ScreenSpot-Pro | C-rand | 61.10% | 61.29% | +0.19 pp | [−0.14,+0.58] |
| ScreenSpot-Pro | C-self | 65.15% | 65.15% | 0.00 pp | [−0.52,+0.54] |
| ScreenSpot-Pro | Equal-arm mean | 64.15% | 64.26% | +0.11 pp | [−0.17,+0.37] |

## Learned controls

| Comparison | Mind2Web | 99% CI | ScreenSpot-Pro | 99% CI |
| --- | ---: | ---: | ---: | ---: |
| VUS-SR − Utility-LSA | **+3.02 pp** | **[+2.09,+3.92]** | −0.14 pp | [−0.48,+0.20] |
| VUS-SR − correctness-LSA | **+2.07 pp** | **[+1.12,+3.02]** | +0.11 pp | [−0.17,+0.39] |
| VUS-SR − blind visual anchor | **+1.35 pp** | **[+0.50,+2.21]** | +0.14 pp | [−0.13,+0.40] |

相对 Utility-LSA 的 equal-benchmark/equal-arm standardized effect 为 `+2.37 MDE`，99% CI `[+1.57,+3.17]`。

## Fold selections

| Outer fold | Configuration | Inner selected epochs | Final epochs |
| ---: | --- | --- | ---: |
| 0 | S1 listwise | 30 / 30 / 30 / 30 | 30 |
| 1 | S2 listwise + downside BCE | 30 / 30 / 30 / 30 | 30 |
| 2 | S2 listwise + downside BCE | 30 / 30 / 30 / 30 | 30 |
| 3 | S2 listwise + downside BCE | 30 / 30 / 30 / 30 | 30 |
| 4 | S2 listwise + downside BCE | 30 / 30 / 30 / 30 | 30 |

## Promotion gates

| Gate | Result |
| --- | --- |
| SR1 all eight cells safe | PASS |
| SR2 at least one benchmark gains 1.0 pp | PASS; Mind2Web +2.99 pp |
| SR3 other benchmark noninferior | PASS; ScreenSpot-Pro +0.11 pp |
| SR4 balanced 99% CI positive vs Utility-LSA | PASS; [+1.57,+3.17] MDE |
