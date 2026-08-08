# 聚合器与动作空间匹配：结果报告

日期：2026-08-09

上游：`runs/close/2026-08-08/`

计算约束：全程零新推理、零 GPU；AndroidControl 四臂保持取消。

## 1. 结论

新主结果通过。固定 C-uni 候选池时，Mind2Web 上 majority 相对原 sequential aggregator 提升 **5.34 pp**，99% CI **[+2.50,+8.04]**，超过 MDE 0.61 pp；ScreenSpot-Pro 上方向相反，majority 相对预冻结的 A2 density medoid 低 **4.05 pp**，CI **[−6.11,−2.08]**。`F-K1=false`，`F-K2=false`。

这支持一个经验适用域：聚合器不能脱离输出动作空间和候选错误结构单独选择。纯坐标 ScreenSpot-Pro 更适合离散密度族；积动作 Mind2Web 更适合 action plurality 驱动的 majority。该结果直接解释 E1：sequential 下 C-cond 相对 C-uni 的 +4.90 pp，在 majority 下缩为 +0.29 pp，因为更合适的聚合器吸收了大部分候选池差异。

解释必须收紧。Mind2Web 的动作分层显示优势主要来自 CLICK（+6.26 pp），TYPE 与 SELECT 均不可区分。因此结果证明的是两个 benchmark 上的“动作空间/错误结构—聚合器匹配”现象，不是“参数动作导致密度失效”的因果证明。仅有两个主证据点，不应写成普遍定律。

## 2. F0：AndroidControl trace 归档

F0 在所有分析前完成。现有 8 个 JSONL 分片逐行解析，检查唯一 ID、model/setting、prediction 字段和跨 shard 重复；随后原子复制、文件 fsync、目录 fsync，并对每个 shard、lane 和排序后的 row-ID 集计算 SHA-256。

| Lane | Rows |
| --- | ---: |
| UI-AGILE Low / High | 2,000 / 2,000 |
| GUI-R1 Low / High | 1,096 / 1,056 |
| UI-R1-E Low / High | 1,824 / 1,792 |

独立备份根为 `/scratch/workspaceblobstore/aggmatch-traces/2026-08-09/`，manifest 为 `BACKUP_MANIFEST.json`。源 writer 的逐行 `write → flush → fsync` 合约也记录了代码 SHA。`raw/`、`predictions*.jsonl` 与现有 JSONL 被标记为禁止递归清理。完整 SHA 见 `f0_ac_archive.json`。

## 3. F1：主结果

### 3.1 设计

- 主 arm：C-uni；C-cond/C-rand/C-self 只作稳健性附表。
- Mind2Web：在每个 website fold 内按 episode 配对重采样。
- ScreenSpot-Pro：在每个 fold 内按 application group 配对重采样。
- 10,000 次 bootstrap，99% percentile CI。
- 所有效应均定义为 `majority - density`。
- ScreenSpot 的 official B3 在 E1 中映射为 `ours`，不声明与 A2 实现等价。

完整四臂结果位于 `f1_aggregator_matching.json`；主表见 `MAIN_TABLE.md`，可视化见 `fig_aggregator_matching.pdf`。

### 3.2 C-uni 主判据

Mind2Web 上 majority 对 sequential、A1、A2、A3、A4 的点差全部为正；sequential 主对照 CI 下界为正且超过 MDE。ScreenSpot-Pro 上 A2/A3/A4/B3 全部显著优于 majority，预冻结 A2 主对照方向反转。A1 在 ScreenSpot 上不可区分，但不影响预冻结 gate。

### 3.3 动作类型

| Action | Rows | Majority − Sequential | 99% CI |
| --- | ---: | ---: | ---: |
| CLICK | 1,774 | +6.26 pp | [+2.86,+9.55] |
| TYPE | 227 | +0.44 pp | [−2.48,+3.64] |
| SELECT | 79 | −1.27 pp | [−5.88,0.00] |

该分层否定了“主要由非 CLICK 参数动作驱动”的精化假设。更可能的解释包括跨类型 plurality 先过滤错误、CLICK 候选的多模态/多簇错误形态，以及 sequential complete-link 的次序敏感性；本轮没有设计能在三者之间作因果区分。

## 4. F2：arm 排序一致性

F2 明确是事后探索。每次分层 bootstrap 内先计算七个聚合器各自的 C-cond−C-uni，再跨聚合器平均，因此保留共享候选造成的格子依赖，不使用简单符号检验。

- Mind2Web：合并效应 +2.05 pp，99% CI [+1.23,+2.95]；C-cond 在 4/7 个聚合器点估计最优；聚合器间效应 SD 1.78 pp。
- ScreenSpot-Pro：合并效应 +1.92 pp，CI [+0.84,+3.23]；C-cond 在 7/7 个聚合器点估计最优；SD 0.60 pp。
- `F-K4=false`。

允许的写法是：“arm 排序在跨聚合器的事后合并分析中一致，但单个预注册 majority 判据未通过。”不得用 F2 恢复四臂主张。

## 5. F3：AndroidControl 部分行

F3 只比较同池聚合器，不生成任何 C-cond 结果。逐行完整性交集为 Low 1,096、High 1,056，均高于 800。当前 checkpoint 只有 stage1 单视角，因此每行池严格定义为 `3 models × 1 view = 3 forwards`，不是 6，也不是 Mind2Web 的 12。

Low/High 的 majority−sequential 分别为 +0.73 pp、+0.95 pp，方向与 Mind2Web 一致，但 99% CI 都跨零。更重要的是，子集单模分数相对历史 7,708 行全量分数出现超过 2 pp 的偏差：Low 最大 2.22 pp（UI-R1-E），High 最大 2.91 pp（UI-AGILE）。因此 `F-K3=true`，F3 降为附录，不能把主结论写成三个数据点。

这批行来自已取消协议且由完成顺序决定，可能同时包含抽样偏差和重跑偏差。方向一致只能视为有限的外部支持。

## 6. 论文形状

### 主线

主线改为聚合器与输出空间/错误结构的匹配。主证据是 ScreenSpot-Pro 与 Mind2Web 的反向配对 CI；AndroidControl 仅作有偏附录。措辞使用“两个 benchmark 的经验适用域”，不使用“普遍预算定律”或动作维度的单因素因果表述。

### 次级结果

跨谱系共识 RoI 在原密度聚合器下显著：ScreenSpot-Pro +2.21 pp，CI [+0.50,+4.16]；Mind2Web +4.90 pp，CI [+2.94,+6.86]。同时必须并列写出 majority 下不可区分：Mind2Web +0.29 pp，ScreenSpot-Pro +1.27 pp，两个 CI 下界均为负。它说明池改进与聚合器存在交互，不说明 C-cond 聚合器无关地优越。

### 机制与负结果

E3 的 high-start 条件保留为两 benchmark 定性机制：ScreenSpot rank0 99.94% 且到 rank11 下降 38.90 pp；Mind2Web rank0 40.38%，只下降 9.23 pp。CALA 的覆盖率、NOA 的有效样本量、C-cond 的候选池改进都说明同一边界：候选池、错误结构与聚合器必须联合优化。

### 不主张

- 不主张绝对分数或 SOTA；native anchors 未重跑。
- 不主张 C-cond 的聚合器无关优越性。
- 不主张参数动作是 F1 反转的原因。
- 不主张 rank decay 是普遍预算定律。
- 不主张任意跨谱系池优于单谱系池。
- 不主张 AndroidControl 四臂或 C-cond 结果。

## 7. Kill conditions

| ID | 状态 | 结果 |
| --- | --- | --- |
| F-K1 | false | 新论文主线成立 |
| F-K2 | false | 两 benchmark 方向反转成立 |
| F-K3 | **true** | AndroidControl 仅附录 |
| F-K4 | false | F2 仅允许事后一致性表述 |

## 8. 复现

```bash
.venv-scaleup/bin/python runs/aggmatch/2026-08-09/f0_ac_archive.py
.venv-scaleup/bin/python runs/aggmatch/2026-08-09/f1_aggregator_matching.py
.venv-scaleup/bin/python runs/aggmatch/2026-08-09/f2_arm_consistency.py
.venv-scaleup/bin/python runs/aggmatch/2026-08-09/f3_androidcontrol_aggregator.py
```

四个命令均不加载模型、不启动 vLLM、不恢复 AndroidControl worker。
