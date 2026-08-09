# Unified Aggregator (EQV) Fail-Closed Report

日期：2026-08-09

上游：`runs/aggmatch/2026-08-09/`

计算约束：零 GPU、零新推理；仅复用 E1 四臂候选 bank。

## 1. 结论

EQV 统一规则的主判定没有启动。第一顺序的 ABL-4 实现自检触发 `U-K4`：ScreenSpot-Pro C-uni 上，coordinate-only EQV（complete-link + 谱系去重）为 63.0614%，A2 为 63.8836%，差 −0.8223 pp，绝对值超过冻结 MDE 0.70 pp。

按协议，U1–U3、强制 dev-fold aggregator selection、其余消融和两个附带诊断全部停止。F1 的“聚合器与动作空间/错误结构匹配”继续作为现有主结果；本轮不能声称统一规则成立或失败于某个 benchmark，因为主 EQV 从未进入双 benchmark 判定。

## 2. 两个定义修正

### 2.1 “评测器等价类”不能无 GT 严格实现

AndroidControl 有固定 0.14 半径，但 Mind2Web 与 ScreenSpot-Pro 的 scorer 都是 point-in-bbox。给定两个候选点，在不知道目标 bbox 时，无法判定评测器是否一定给它们相同标签。因此冻结实现使用 `GT-free evaluator-aligned proxy equivalence`：

- ScreenSpot-Pro 使用 E1/A2 家族的 14 px 几何尺度；
- Mind2Web 原计划仅用 outer-dev bbox 尺度估计阈值，禁止 test GT；
- 报告禁止使用“当且仅当 evaluator 同判”的字面主张。

由于 U-K4 在 ScreenSpot 自检阶段触发，Mind2Web 阈值没有进入结果计算。

### 2.2 E1 majority 的实际定义

- ScreenSpot-Pro 只有一个隐含动作，E1 majority 确实退化为 dev-best slot；连续坐标没有被计票。
- Mind2Web E1 majority 不是完整候选 exact match，而是动作类型 plurality 后输出 dev-priority 真实候选。

因此 F1 的反向现象仍有“规则覆盖的等价关系不完整”这一机制解释，但不能把两端都描述成 exact-candidate majority。

## 3. ABL-4 结果与定位

冻结主 EQV 使用 deterministic greedy complete-link，每个谱系在类内最多一票，并输出胜出类中 dev reliability 最高的真实候选。ABL-4 只忽略动作与参数，保留谱系去重。

| 变体 | Accuracy | 相对 A2 |
| --- | ---: | ---: |
| A2 | 63.8836% | 0 |
| Complete-link + lineage dedup | 63.0614% | −0.8223 pp |
| Complete-link + candidate votes | 63.8836% | 0.0000 pp |
| Single-link + lineage dedup | 62.5553% | −1.3283 pp |
| Single-link + candidate votes | 63.2511% | −0.6325 pp |

这组 debug 足以排除主要实现错误：complete-link candidate voting 与 A2 的 point accuracy 完全相同。U-K4 来自谱系去重伤害了 ScreenSpot 的密度信号，而不是 complete-link 写错。

但这也揭示了自检设计的混杂：A2 按候选计密度，主 EQV 按谱系计票。比较两者时不仅替换了等价关系，也替换了计票单位。若把 ABL-4 事后改成 candidate votes，自检会通过，但这违反“配置先于结果”和主方法谱系去重的定义，因此没有这样做。

## 4. 与 PKA 的区别

EQV 的设计动机仍与 PKA 不同：它不比较跨流形核密度，不使用自核票，并只输出真实候选。PKA 过去的失败涉及自票、跨流形尺度不可比，以及离散动作核值与坐标自核不平衡。

本轮没有得到 EQV 性能结果，因此只能保留算法定义上的区别，不能声称 EQV 已修复 PKA 的经验失败。

## 5. 未运行项目

- U1：EQV 对 ScreenSpot A2/A3 的非劣判据；
- U2：EQV 对 Mind2Web majority 的非劣判据；
- U3：两端相对当地最差族的显著改进；
- 强制 nested dev-selection；
- ABL-3 类型-only；
- ABL-5 容差扫描；
- B1 来源偏置复查；
- Mind2Web CLICK/TYPE/SELECT 分层复查。

对应 JSON 均明确标注 `CANCELLED_NOT_RUN_BY_U_K4`，不得把缺失结果解释成零效应。

## 6. 论文定位

F1 继续维持主结果。论文必须加入两个机制观察：

1. ScreenSpot majority 在单一隐含动作下退化为 dev-best slot，没有利用连续坐标邻近性；
2. A2 与 coordinate-only complete-link candidate voting 在总体准确率上完全一致，而谱系去重会损失 0.82 pp，说明“去来源偏置”与“保留密度信号”存在真实张力。

EQV 本轮定位为一次被实现自检提前终止的统一尝试，而不是成功方法，也不是完整负结果。

## 7. 预注册与完整性

初始配置 commit `35cb9b6` 含一处 YAML 缺失冒号；在任何结果计算前以 syntax-only commit `4660661` 修复。两份配置随后解析和断言通过。容差、阈值、成类方式与谱系去重均未在看到结果后调整。

独立备份根为 `/scratch/workspaceblobstore/eqv-traces/2026-08-09/`，由锁定 manifest 记录每个配置、脚本、结果和报告的 SHA-256。

外部 PID 2274 未触碰，无 vLLM 或模型 worker 启动。
