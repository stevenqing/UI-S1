# Utility-LSA Main Table

状态：`SAFE_EXPLORATORY_OVERRIDE`

训练信号：候选相对 exact cross-fitted CEV-A fallback 的 `−1/0/+1` 净效用。五折 OOF 选择：U-GRPO / U-HYBRID / U-HYBRID / U-GRPO / U-GRPO，均为 H3，均为有限阈值。

## Safe Utility-LSA − CEV-A

| Benchmark | Arm | CEV-A | Utility safe | Delta |
| --- | --- | ---: | ---: | ---: |
| Mind2Web | C-uni | 32.02% | 32.12% | +0.10 pp |
| Mind2Web | C-cond | 32.45% | 32.21% | −0.24 pp |
| Mind2Web | C-rand | 31.83% | 31.92% | +0.10 pp |
| Mind2Web | C-self | 31.39% | 31.35% | −0.05 pp |
| ScreenSpot-Pro | C-uni | 63.88% | 64.01% | +0.13 pp |
| ScreenSpot-Pro | C-cond | 66.48% | 66.29% | −0.19 pp |
| ScreenSpot-Pro | C-rand | 61.10% | 62.05% | +0.95 pp |
| ScreenSpot-Pro | C-self | 65.15% | 65.28% | +0.13 pp |

## Equal-arm paired effects

| Comparison | Mind2Web | 99% CI | ScreenSpot-Pro | 99% CI |
| --- | ---: | ---: | ---: | ---: |
| Utility safe − CEV-A | −0.02 pp | [−0.28,+0.23] | **+0.25 pp** | **[+0.04,+0.47]** |
| Utility safe − nested dev-selection | +0.14 pp | [−0.59,+0.88] | **+1.19 pp** | **[+0.58,+1.80]** |
| Utility safe − correctness-LSA | **−0.95 pp** | **[−1.56,−0.33]** | **+0.25 pp** | **[+0.02,+0.50]** |

## Gates

| Gate | State |
| --- | --- |
| UR1 eight-cell safety | PASS |
| UR2 robust Mind2Web gain | FAIL |
| UR3 ScreenSpot preservation/gain | PASS |
| UR4 vs dev-selection balanced | PASS；standardized CI [+0.22,+1.70] |
| UR5 better-aligned than correctness-LSA | FAIL；standardized CI [−1.12,−0.07] |
| UR-K1/K2/K3/K4 | false |
| UR-K5 | false；main−no-MVP ScreenSpot +0.25 pp，[+0.05,+0.48] |

## Fixed ablations

| Main Utility-safe − ablation | Mind2Web | 99% CI | ScreenSpot-Pro | 99% CI |
| --- | ---: | ---: | ---: | ---: |
| no-MVP-structure | −0.11 pp | [−0.38,+0.17] | **+0.25 pp** | **[+0.05,+0.48]** |
| absolute-only | −0.61 pp | [−1.33,+0.10] | −0.16 pp | [−0.41,+0.08] |

no-MVP 在 ScreenSpot 显著更差，UR-K5 的第一项为 false，因此无需 permutation importance。absolute-only 的 equal-benchmark standardized `main−ablation` 99% CI 为 `[−1.25,−0.003]` MDE，说明 fallback-pair 扩展未优于较简单的 absolute representation。
