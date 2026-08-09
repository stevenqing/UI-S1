# Unified Aggregator (EQV) Main Table

状态：`PAUSED_U_K4_IMPLEMENTATION_SELF_CHECK`

按冻结顺序，本轮只允许先运行 ScreenSpot-Pro C-uni 的 ABL-4 自检。`U-K4` 触发后，U1–U3、嵌套 dev-selection、ABL-3/ABL-5 和附带诊断均未启动。

## ABL-4 自检

| 变体 | Accuracy | 相对 A2 | 99% CI | 结论 |
| --- | ---: | ---: | ---: | --- |
| A2 reference | 63.8836% | 0 | — | E1 anchor |
| Complete-link + lineage dedup（主 EQV 的 coordinate-only 版本） | 63.0614% | **−0.8223 pp** | [−1.9894,+0.2571] | 绝对差超过 MDE 0.70 pp，`U-K4=true` |
| Complete-link + candidate votes（ABL-1 debug） | 63.8836% | **0.0000 pp** | [−0.5597,+0.6300] | 总体准确率与 A2 完全一致 |
| Single-link + lineage dedup | 62.5553% | −1.3283 pp | [−2.6215,−0.1255] | 更差 |
| Single-link + candidate votes | 63.2511% | −0.6325 pp | [−1.3191,+0.0707] | MDE 内但不等价 |

## Gate 状态

| ID | 状态 | 后果 |
| --- | --- | --- |
| U-K1 | 未判定 | U1 未运行 |
| U-K2 | 未判定 | U2 未运行 |
| U-K3 | 未判定 | dev-selection 未运行 |
| U-K4 | **true** | 暂停统一规则主判定，优先排查实现/定义 |

## 诊断

坐标 complete-link 本身没有实现漂移：去掉谱系去重后，其 point accuracy 与 A2 都是 63.8836%。超过 MDE 的差异完全出现在加入谱系去重以后。因此 U-K4 不是代码错误，而是冻结自检同时改变了“坐标成类”和“谱系计票单位”两个因素。

预注册主 EQV 明确要求谱系去重，不能在看到结果后把主方法改成 candidate votes。故本轮保持 fail-closed 暂停，不报告 U1–U3。
