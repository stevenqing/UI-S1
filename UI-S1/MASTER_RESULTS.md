# GUI Test-Time Scaling — Master Results

更新日期：2026-08-11

本文只汇总冻结口径。完整方法、逐折选择、污染边界和限制以各 run 的 `REPORT.md` 为准。

## 1. 论文结果层级

| Level | Result | Frozen conclusion |
| --- | --- | --- |
| Primary | F1 aggregator matching | Mind2Web majority−sequential +5.34 pp [2.50,8.04]；ScreenSpot majority−B3 −3.86 pp [−5.84,−1.92] |
| Learned method candidate | VUS-SR | Mind2Web equal-arm +2.99 pp [2.10,3.91]；ScreenSpot +0.11 pp [−0.17,0.37]；八 cells safe |
| Explanatory follow-up | CEV-A | 恢复 Mind2Web G0 与 ScreenSpot G4；与 nested dev-selection 打平，V4=`EXPLANATORY_CONTRIBUTION` |
| Learned precursors | LSA / Utility-LSA | correctness-LSA partial transfer；Utility-LSA safe exploratory；均被 VUS-SR 显著超过 |
| Secondary | Q1 consensus RoI | 密度聚合器下 ScreenSpot +2.21 pp、Mind2Web +4.90 pp；CEV-A 下 Mind2Web pool effect 被吸收 |
| Mechanism | E3 high-start condition | rank decay 转为性能下降需要高起点提议器；两点定性 |
| Selective prediction | R4 / SafeGround port | AUROC 0.744→0.830；80% coverage 下 +7.12 pp；无原论文 FDR 继承 |

## 2. F1 与 CEV-A

| Benchmark | Wrong endpoint | Local endpoint | CEV-A | CEV-A vs dev-selection |
| --- | ---: | ---: | ---: | ---: |
| Mind2Web C-uni | Sequential 26.68% | Majority 32.02% | 32.02% | +0.19 pp [−0.58,+0.96] |
| ScreenSpot-Pro C-uni | Majority 59.84% | A2 63.88% | 63.88% | +0.06 pp [−0.50,+0.71] |

CEV-A 选择：Mind2Web G0/G0/G2/G0/G0，ScreenSpot-Pro 五折 G4。C-K5 触发：中央容差排名跨折翻转。

## 3. VUS-SR learned aggregator

| Benchmark | CEV-A equal-arm | VUS-SR safe | Delta | 99% CI |
| --- | ---: | ---: | ---: | ---: |
| Mind2Web | 31.92% | **34.92%** | **+2.99 pp** | **[+2.10,+3.91]** |
| ScreenSpot-Pro | 64.15% | 64.26% | +0.11 pp | [−0.17,+0.37] |

Mind2Web 四臂 CI 均为正；ScreenSpot 四臂均安全。相对 Utility-LSA 的 equal-benchmark/equal-arm standardized 99% CI 为 `[+1.57,+3.17]` MDE。相对同一 blind Qwen3-VL anchor，Mind2Web 仍增加 +1.35 pp `[+0.50,+2.21]`，证明 listwise utility/downside training 有独立增量。

VUS-SR 为当前最强经验统一聚合器，但仍是同两 benchmark 上的 nested discovery；需第三个独立 benchmark 才能最终确认。CEV-A 保持最强 training-free 规则。

## 4. Q1 与聚合器限定

| Benchmark | Sequential/density C-cond−C-uni | CEV-A C-cond−C-uni |
| --- | ---: | ---: |
| Mind2Web | +4.90 pp [2.94,6.86] | +0.43 pp [−1.57,+2.57] |
| ScreenSpot-Pro | +2.21 pp [0.50,4.16] | +2.59 pp [1.10,4.26] |

Mind2Web difference-in-differences：−4.47 pp，99% CI [−7.34,−1.68]。该结果直接支持 pool × aggregator 交互。

## 5. E3 containment

| Benchmark | Rank-0 | Rank-11 | Drop | V-only N16−N4 |
| --- | ---: | ---: | ---: | ---: |
| ScreenSpot-Pro | 99.94% | 61.04% | 38.90 pp | −2.91 pp [−5.58,−0.36] |
| Mind2Web | 40.38% | 31.15% | 9.23 pp | +0.34 pp [−1.22,+1.98] |

## 6. 关键历史锚

| Item | Frozen value |
| --- | --- |
| C-uni ScreenSpot B3 | 63.69% canonical |
| Drop-in pool gain | +3.60 pp [1.31,6.22] |
| M1 pool gain | +3.42 pp [1.41,5.67] |
| Kappa | 视角 0.895；同族跨规模 0.618；跨族 0.398 |
| Source bias | 7B p=4.12e-152, V=0.779；72B p=1.21e-273, V=0.822 |
| 72B lineage normalization | 70.59% vs B3 41.24%，但近 best-single 71.41% |
| AndroidControl partial F3 | Low/High majority−sequential +0.73/+0.95 pp；F-K3，附录 |

## 7. Run status

| Run | Status | Key decision |
| --- | --- | --- |
| `runs/dominance/2026-08-06/` | COMPLETE | R7 修复；不改变 nested selections |
| `runs/consolidate/2026-08-06/` | COMPLETE | Q1 pass；Q2a/Q2b fail |
| `runs/xfer/2026-08-07/` | COMPLETE | Mind2Web transfer pass；新 trace retained |
| `runs/close/2026-08-08/` | COMPLETE | E-K1；E2/AC cancelled；E3 pass |
| `runs/aggmatch/2026-08-09/` | COMPLETE | F1 becomes primary；F-K3 only |
| `runs/eqv/2026-08-09/` | STOPPED | U-K4 implementation self-check |
| `runs/cev/2026-08-09/` | COMPLETE | V4 explanatory contribution；C-K5 |
| `runs/lsa/2026-08-10/` | COMPLETE | L1/L4 pass，L2/L3 fail；主模型安全但不显著 |
| `runs/lsa-confirm/2026-08-10/` | COMPLETE | T1/T3/T4 pass，T2 fail；LT-K3 partial transfer |
| `runs/lsa-utility/2026-08-11/` | COMPLETE | safe exploratory；UR2/UR5 fail；fixed ablations complete，UR-K5 false |
| `runs/visual-utility-selector/2026-08-11/` | COMPLETE | VUS-SR method candidate；SR1–SR4 pass |

## 8. Reproducibility boundaries

- Historical AndroidControl/Mind2Web row traces are permanently lost；见 `runs/xfer/2026-08-07/LOST_TRACES.md`。
- New Mind2Web unified-prompt results cannot be mixed with historical native-prompt aggregates.
- CEV is a post-leakage reconstructed preregistration；五个 ScreenSpot cells 仅为污染锚。
- 不主张 absolute SOTA、普适 rank-decay law 或 aggregator-independent C-cond superiority。
- 08-06 后至 Utility-LSA 的聚合分析均为 retained candidate banks 上的零 GPU 重算；VUS 显式例外，使用 Qwen3-VL-8B blind visual logits 与 GPU set-ranker。
- VUS-SR 是已知两 benchmark 上的 nested discovery，不是第三 benchmark 独立确认。
- VUS 首次 formal process eager-loaded all-fold labels；Correction 006 物理 fold-seal 后 bit-identical rerun，只有 hardened outputs 用于结论。
