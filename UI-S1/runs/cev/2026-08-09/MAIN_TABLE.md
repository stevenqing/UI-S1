# CEV / CEV-A Main Table

状态：`COMPLETE_EXPLANATORY_CONTRIBUTION_WITH_C_K5`

本轮是 `post-leakage reconstructed preregistration`。五个已知 ScreenSpot-Pro 格子只用于 V1 实现锚与污染披露。

## V1–V4

| Gate | Benchmark / comparison | Delta | 99% CI | Result |
| --- | --- | ---: | ---: | --- |
| V1 | ScreenSpot G4 complete-link candidate votes vs frozen A2 aggregate | 0.0000 pp | — | PASS，均为 63.8836%；逐行一致率 97.72% |
| V2 | ScreenSpot CEV-A − A2 | 0.0000 pp | [−0.57,+0.62] | PASS |
| V2 | Mind2Web CEV-A − majority | 0.0000 pp | [0.00,0.00] | PASS |
| V3 | ScreenSpot CEV-A − majority | **+4.05 pp** | **[+2.14,+5.99]** | PASS |
| V3 | Mind2Web CEV-A − sequential | **+5.34 pp** | **[+2.49,+8.21]** | PASS |
| V4 | ScreenSpot CEV-A − nested dev-selection | +0.06 pp | [−0.50,+0.71] | 打平 |
| V4 | Mind2Web CEV-A − nested dev-selection | +0.19 pp | [−0.58,+0.96] | 打平 |

V4：`EXPLANATORY_CONTRIBUTION`。CEV-A 不显著优于按 benchmark 做 nested aggregator selection，因此不是方法优势；价值是用一个 complete-link voting 过程恢复两个已知极点。

## C-uni accuracy 与外折选择

| Benchmark | CEV-A | 当地极点 | Nested dev-selection | CEV-A 选择 |
| --- | ---: | ---: | ---: | --- |
| Mind2Web | 32.0192% | majority 32.0192% | 31.8269% | G0/G0/G2/G0/G0；五折均选 global |
| ScreenSpot-Pro | 63.8836% | A2 63.8836% | 63.8204% | 五折固定 G4 |

强制 dev-selection 每折选择：

- Mind2Web：A0 / majority / A0 / majority / majority。
- ScreenSpot-Pro：ours / ours / A4 / A2 / ours。

## 四臂 robustness：CEV-A arm − C-uni

| Benchmark | Arm | Delta | 99% CI |
| --- | --- | ---: | ---: |
| Mind2Web | C-cond | +0.43 pp | [−1.65,+2.54] |
| Mind2Web | C-rand | −0.19 pp | [−0.60,+0.19] |
| Mind2Web | C-self | −0.63 pp | [−2.38,+1.09] |
| ScreenSpot-Pro | C-cond | **+2.59 pp** | **[+1.10,+4.26]** |
| ScreenSpot-Pro | C-rand | **−2.78 pp** | **[−4.62,−0.93]** |
| ScreenSpot-Pro | C-self | +1.27 pp | [−0.57,+2.96] |

## 消融阶梯

| Variant | Mind2Web | ScreenSpot-Pro | Interpretation |
| --- | ---: | ---: | --- |
| G0 action endpoint | 32.0192% | — | majority endpoint |
| G4 coordinate endpoint | — | 63.8836% | A2 endpoint |
| Global fixed-threshold granularity | 32.0192% | — | 无增益 |
| Action-conditional fixed threshold | 32.0192% | — | 无增益 |
| Parameter threshold selection | 32.0192% | — | 无增益 |
| Full CEV-A | 32.0192% | 63.8836% | 恢复两端极点 |
| Lineage cap 1 | 31.9712% | 63.0614% | ScreenSpot 明显受损；污染分析 |
| Lineage cap 2 | 31.9712% | 64.0101% | ScreenSpot 点估计略高；分析性，不升级主方法 |
| Single-link | 32.0192% | 63.2511% | ScreenSpot 更差；污染分析 |

## Kill conditions

| ID | State | Consequence |
| --- | --- | --- |
| C-K1 | false | 实现锚通过 |
| C-K2 | false | 连续端非劣通过 |
| C-K3 | false | 积动作端非劣通过 |
| C-K4 | false | 未显著劣于 dev-selection |
| C-K5 | **true** | 中央容差排序跨折翻转；禁止普适、无容差依赖的表述 |
