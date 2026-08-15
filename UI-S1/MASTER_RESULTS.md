# GUI Test-Time Scaling — Master Results

更新日期：2026-08-14

本文只汇总冻结口径。完整方法、逐折选择、污染边界和限制以各 run 的 `REPORT.md` 为准。

## 1. 论文结果层级

| Level | Result | Frozen conclusion |
| --- | --- | --- |
| Primary | F1 aggregator matching | Mind2Web majority−sequential +5.34 pp [2.50,8.04]；ScreenSpot majority−B3 −3.86 pp [−5.84,−1.92] |
| Learned method candidate | VUS-SR | Mind2Web equal-arm +2.99 pp [2.10,3.91]；ScreenSpot +0.11 pp [−0.17,0.37]；八 cells safe |
| Explanatory follow-up | CEV-A | 恢复 Mind2Web G0 与 ScreenSpot G4；与 nested dev-selection 打平，V4=`EXPLANATORY_CONTRIBUTION` |
| Learned precursors | LSA / Utility-LSA | correctness-LSA partial transfer；Utility-LSA safe exploratory；均被 VUS-SR 显著超过 |
| Post-VUS negative | CARE A1 router | corrected structural routing fails：M2W pass@12 −1.01 pp；SSPro safe −1.27 pp；`CLOSE_ROUTING` |
| Post-VUS negative | DELTA late fusion | M2W −0.41 pp `[−1.20,+0.40]`；SSPro +0.11 pp `[−0.20,+0.41]`；same-capacity/placebo gates fail；`DELTA_NOT_SUPPORTED` |
| Post-VUS partial negative | CIVA-A0 admission | raw-direct M2W +1.57 pp `[+0.79,+2.39]`、SSPro +5.41 pp `[+4.12,+6.81]`；matched-random/placebo pass，text attribution/cell safety fail；`CIVA_ADMISSION_NOT_SUPPORTED` |
| Mechanism follow-up | GRAN | M2W CLICK high-$\hat p$ margin +11.74 pp `[+7.29,+16.51]`；跨 benchmark 曲线与两端点统一失败 |
| Pre-GPU falsification | SPLIT | $\Delta_2$ 6.45 pp，但 geometry failure 26.79% 触发 Z-K6；有效正例 76 触发 Z-K7；未执行模型 forward |
| Pre-GPU proposer | MASK | 理想三票 $N_{\mathrm{eff}}$ calibration 最大 +0.538 pp，低于 0.70 pp MDE；M-K1；未执行模型 forward |
| Closure diagnostic | CEIL | recoverable subset：M2W cheap AUROC 0.688 `[0.665,0.709]`，C-D2；SSPro 0.540 `[0.501,0.583]`，C-D1；只授权另立 M2W spec |
| Scoping only | ORTH | 两CPU OCR在SSPro recoverable/zero-coverage有覆盖且 error $\kappa$ 约0.10–0.20；信号限于text targets；全标签方向选择使后续同数据研究只能是 post-selection validation |
| Post-selection negative | OTEXT | EasyOCR nested Stage-0 双基线最小增益 +0.064 pp，低于 0.70 pp O-G1；RapidOCR 0；O-K1，未授权 Stage 1 |
| Structural feasibility | XSCR | 同屏结构稀缺：singleton screens M2W 97.5%、AC 99.5%；M2W 最佳 repair−damage proxy +0.479 pp < 0.70 pp MDE，AC 0；仅授权探索性 spec |
| Descriptive decomposition | DECOMP | SSPro B2–B8 可识别预算中谱系边际全为正、视角边际多数近零/负；Arm 2 singleton screens 98.52%、碰撞≤0.253%；生成模型 logprob 未留存 |
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

### 3.1 Post-VUS diagnostic and CARE A1

VUS-SR 剩余瓶颈主要是 candidate identification，而非 safe gate：Mind2Web pass@12 59.21%、candidate-ranking gap 18.52 pp、gate gap 5.77 pp；ScreenSpot-Pro 对应 79.57%、14.60 pp、0.71 pp。最小目标 quartile 的 conditional ranking failure 为 53.57%/32.99%，显著高于最大 quartile 的 34.75%/13.32%。

四臂前六候选逐行相同，虽存在 oracle stage-2 routing coverage gain +6.06/+3.67 pp，但 corrected CARE A1 structural router 未能泛化：相对 nested static C-cond，Mind2Web pass@12 −1.01 pp `[−2.10,0.00]`，ScreenSpot +0.06 pp `[−0.81,+1.05]`，且 ScreenSpot safe −1.27 pp。三项 A1 gates 全失败，routing 分支关闭。初版遗漏 cross-fitted reliability 的实现已作废；Correction 002 重跑后结论更强。

RAVEL E0 随后验证 local 相对 random-center 有信号，但 pixel-level early fusion 使 Mind2Web 最终下降 2.19 pp，触发 RAVEL-K4。DELTA 再测试 independently locked channels 的 decision-level late fusion：相对 VUS-SR，Mind2Web −0.41 pp `[−1.20,+0.40]`，ScreenSpot +0.11 pp `[−0.20,+0.41]`。FULL 无法超过 VUS_ONLY 或 RANDOM_PLACEBO，且显著差于 VUS_GLOBAL；证据互补性不成立，distillation/third-benchmark confirmation 取消。

CIVA-A0 最后把问题改成 pre-admission incremental value。对 raw VUS-binding direct，REAL_FULL 在 Mind2Web/ScreenSpot-Pro 提升 +1.57/+5.41 pp，并显著超过 matched-random 与 random-channel placebo；但 FULL 不优于 NO_TEXT，且 Mind2Web C-uni 未满足严格 noninferiority。CIVA-5/6 失败，不能支持 instruction-conditioned admission，也不授权接入 VUS-SR safe policy。

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
| `runs/care/2026-08-11/` | A1 COMPLETE | structural acquisition router fails；`CLOSE_ROUTING` |
| `runs/ravel/2026-08-11/` | STOPPED | local evidence early fusion fails；Mind2Web −2.19 pp [−2.98,−1.41]；RAVEL-K4 |
| `runs/delta/2026-08-11/` | COMPLETE | only DELTA-2/6 pass；local channels dilute M2W global/binding evidence；`DELTA_NOT_SUPPORTED` |
| `runs/civa/2026-08-11/` | COMPLETE | raw-direct admission signal positive；instruction attribution and all-cell safety fail；`CIVA_ADMISSION_NOT_SUPPORTED` |
| `runs/gran/2026-08-14/` | COMPLETE | within-M2W $\hat p$ explanation supported；cross-benchmark/endpoint unification failed |
| `runs/split/2026-08-14/` | STOPPED PRE-GPU | Z-G1 pass；Z-K6 geometry 与 Z-K7 low-$n$；all GPU endpoints not run |
| `runs/mask/2026-08-14/` | STOPPED PRE-GPU | M-G1 fail；理想 density/F1 gains +0.538/+0.219 pp；M-K1；all GPU endpoints not run |
| `runs/ceil/2026-08-14/` | COMPLETE | M2W C-D2、SSPro C-D1；overall `OPEN_NEW_SPEC_C_D2`；Arm A post-hoc且远端渐近弱识别 |
| `runs/orth/2026-08-14/` | COMPLETE SCOPING | OCR text-target follow-up scoped；DOM historical data currently missing；all results non-paper exploratory |
| `runs/otext/2026-08-14/` | STOPPED STAGE 0 | `POST_SELECTION_VALIDATION`；EasyOCR O-G1 fail；`OTEXT_STOPPED_O_K1_STAGE0`；Stage 1 not run |
| `runs/xscr/2026-08-14/` | COMPLETE FEASIBILITY | `POST_SELECTION_FEASIBILITY`；best M2W proxy +0.479 pp below MDE；AC 0；nominal holdout contaminated by all-fold label loading |
| `runs/decomp/2026-08-14/` | COMPLETE DESCRIPTIVE | SSPro 4,083 subsets at B2–B12；lineage marginal positive through identifiable B8；Arm 2 low structure；Arm 3 `LOGPROB_CHANNEL_NOT_RETAINED` |

## 8. Reproducibility boundaries

- Historical AndroidControl/Mind2Web row traces are permanently lost；见 `runs/xfer/2026-08-07/LOST_TRACES.md`。
- New Mind2Web unified-prompt results cannot be mixed with historical native-prompt aggregates.
- CEV is a post-leakage reconstructed preregistration；五个 ScreenSpot cells 仅为污染锚。
- 不主张 absolute SOTA、普适 rank-decay law 或 aggregator-independent C-cond superiority。
- 08-06 后至 Utility-LSA 的聚合分析均为 retained candidate banks 上的零 GPU 重算；VUS 显式例外，使用 Qwen3-VL-8B blind visual logits 与 GPU set-ranker。
- VUS-SR 是已知两 benchmark 上的 nested discovery，不是第三 benchmark 独立确认。
- VUS 首次 formal process eager-loaded all-fold labels；Correction 006 物理 fold-seal 后 bit-identical rerun，只有 hardened outputs 用于结论。
- CARE/RAVEL 均为 VUS 结果已知后的 post-hoc research sequence；RAVEL 必须在第三个 untouched benchmark 才能升级为确认结果。
- RAVEL local evidence beats random centers in AUROC but loses global/unique-candidate information; no relational/LoRA stage was run. Any late-fusion follow-up is a new study, not a RAVEL rescue.
- DELTA is a multi-call research oracle protocol over already locked channels; even a positive result requires one-call distillation and untouched GUI-Odyssey confirmation.
- DELTA was negative: FULL did not beat VUS_ONLY or RANDOM_PLACEBO, so no distillation or GUI-Odyssey confirmation was run. VUS_GLOBAL remains a diagnostic control, not a post-hoc selected method.
- CIVA-A0 improved only the raw direct evidence policy, not VUS-SR safe Step-SR. NO_TEXT is a post-result diagnostic control and cannot be promoted; no policy-level follow-up was run.
- GRAN supports $\hat p$ only as a label-dependent within-Mind2Web CLICK explanatory variable. It does not provide a runtime selector or a common Mind2Web/ScreenSpot-Pro coordinate, and the CEV two-endpoint unification is invalid.
- SPLIT found 6.45 pp two-mode candidate headroom, but the frozen matched-window geometry failed on 26.79% of gated rows and left 76 positives. No probe forward was run, so there is no evidence that falsification-crop confidence supplies an orthogonal channel.
- MASK closed the proposer variant before GPU: all 4,095 C-uni source subsets predict at most +0.538 pp under ideal new-vote independence, below the 0.70 pp MDE. This benchmark-local monotone calibration does not restore the rejected universal $N_{\mathrm{eff}}$ law or support consensus occlusion as an orthogonal proposer.
- CEIL finds benchmark-specific conditional ranking signal rather than a shared rule: Mind2Web passes C-D2 while ScreenSpot-Pro passes C-D1. This authorizes only a separately preregistered Mind2Web full-candidate reweighting study. Arm A's large Mind2Web parametric $\Delta_\infty$ values extrapolate far beyond support and are weakly identified sensitivity outputs; finite three-vote isotonic gains are much smaller.
- ORTH is not a paper result. Its wide-grid CPU OCR scoping used all 1,581 ScreenSpot-Pro labels to choose the text-target direction, so OTEXT on the same rows is post-selection validation rather than confirmation. Full Mind2Web DOM/AX evaluation remains blocked until the historically audited official dataset is restored and hashed.
- OTEXT independently regenerated EasyOCR and RapidOCR, then stopped at the preregistered Stage-0 gate: EasyOCR's nested minimum gain over both majority and dev-selection was +0.064 pp versus the 0.70 pp MDE, and RapidOCR's was 0. Stage 1 was not authorized; a confirmatory method claim requires new untouched data.
- XSCR finds very little same-screen multiplicity: 97.5% of exploratory Mind2Web screens and 99.5% of AndroidControl screens are singletons. The best optimistic Mind2Web repair-minus-damage proxy is +0.479 pp, below the 0.70 pp MDE; AndroidControl is non-positive. The nominal 30% holdout was excluded from all reported aggregates but all private-label files were parsed during input locking, so it is not an unread prospective holdout. Any current-data method follow-up is nested post-selection exploration; independent validation requires new data.
- DECOMP is a post-hoc decomposition of the existing ScreenSpot-Pro +3.605 pp mixed-pool result, not a new method. Across identifiable B2–B8 budgets, adding a lineage has a positive marginal effect for both density B3 and F1 majority, while view marginals are generally smaller and often negative; every selected budget cell touches a support boundary. Mind2Web has no compliant aligned 3-lineage × 4-view pool and is not evaluated. Label-free ScreenSpot-Pro same-screen collision is at most 0.253%, and no generating-model token logprob/sequence score was retained in either ScreenSpot-Pro or Mind2Web traces. Downstream selector logits are not generation logprobs.
