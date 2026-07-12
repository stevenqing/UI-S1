# Pass@8 + 更强 Corrector 实验总结

## 核心问题

此前已有两个不同层面的 Pass@8 正信号：

1. Android Control 全任务采样中，UI-TARS-7B 从 Pass@1 13.48% 提升到 Pass@8 20.61%，Qwen2.5-VL-7B 从 6.67% 提升到 16.05%。
2. GUI-360 的 962 个历史 critical steps 上，多模型候选并集的 high-quality coverage 为 30.77%，相对 matched-budget 单 SFT 的 4.89% 高 25.88pp。

这些结果只说明“候选里经常存在正确动作”，并没有证明无 GT 的 selector 能把它选出来。本实验直接检验：更强多模态模型能否把 Pass@8 diversity 转化为 student-relative rescue utility。

## 冻结协议

- 候选源：SFT anchor、Qwen3-VL-8B、Qwen3.5-9B、LLaVA-1.5-7B，每个源固定取 K=8。
- 候选包只含截图、goal、teacher-forced 历史、匿名 action 与匿名 support；不含 GT、reward、correctness、模型身份和 GT-derived diagnostics。
- exact action 全量保留；坐标 bucket 只用于邻域 support，不会删掉 oracle action。
- episode-disjoint split：
  - smoke：12 episodes / 23 steps；
  - dev：133 episodes / 231 steps；
  - locked test：398 episodes / 708 steps。
- dev 与 locked prediction 都在任何 dev/locked label 解封之前完成。
- 模型：
  - current：Qwen3.5-9B；
  - strong：Qwen3.5-35B-A3B；
  - control：exact plurality、cross-source consensus。
- 所有新 GPU 任务严格只使用物理 GPU 4–7；PID 1911 全程存活且未被发送信号。

## Locked Test 结果

| selector | baseline acc | selected acc | oracle ceiling | oracle capture | net utility | rescue / regress | 95% episode-cluster CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B current | 0.42% | **6.78%** | 30.79% | **20.93%** | **+6.36pp** | **46 / 1** | **[+4.53pp, +8.31pp]** |
| Qwen3.5-35B-A3B strong | 0.42% | 4.94% | 30.79% | 14.88% | +4.52pp | 33 / 1 | [+2.93pp, +6.20pp] |
| exact plurality | 0.42% | 3.67% | 30.79% | 10.70% | +3.25pp | 25 / 2 | [+1.87pp, +4.73pp] |
| cross-source consensus | 0.42% | 5.79% | 30.79% | 17.67% | +5.37pp | 39 / 1 | [+3.61pp, +7.25pp] |

Dev→locked 一致：Qwen3.5-9B 的 utility 从 +6.49pp 到 +6.36pp；Qwen3.5-35B-A3B 从 +2.60pp 到 +4.52pp；两个 split 上所有 selector 的 cluster-bootstrap 下界均大于 0。

## Paired 结论

- **strong − current：-1.84pp**，95% CI **[-3.69pp, +0.00pp]**。
- strong-only correct 13 步，current-only correct 26 步；动作一致率 53.53%。
- 35B changed coverage 为 34.18%，9B 为 60.03%；35B 更保守，但没有更准确。
- cross-source consensus − current 为 -0.99pp，95% CI [-2.98pp, +1.12pp]：简单匿名跨模型共识已经解释了很大一部分增益，与 9B 的差异不显著。

## 科研结论

1. **Pass@8 方向成立。** 候选 diversity 不只是 oracle 假象；无 GT fixed-choice selector 能稳定获得显著正 student-relative utility。
2. **“更大 corrector 更好”不成立。** Qwen3.5-35B-A3B 没有超过 Qwen3.5-9B，反而少捕获 6.05 个百分点的 oracle headroom。
3. **关键机制更像 proposal consensus，而不是单纯 scale。** 零 GPU 的 cross-source consensus 达到 +5.37pp，接近 9B 的 +6.36pp。
4. **仍有巨大未利用空间。** 最佳 selector 只捕获 20.93% oracle headroom；packet oracle 为 30.79%，实际 selected accuracy 仅 6.78%。
5. **作用域有限。** 这是 selector-fresh、不是 benchmark-fresh；962 步来自历史 GT-conditioned critical set，不能外推为 arbitrary-state online router。

## Selector → Training 纯度缺口

Positive utility 不等于 SFT labels 足够干净。在 student-wrong 行上，selector 再次选错的 utility 为0，但错误 action 作为训练 target 会主动伤害模型。

| GT-free 构造 | Changed rows | Correct labels | SFT purity | Wilson 95% |
|---|---:|---:|---:|---:|
| 全部9B changes | 425 | 46 | 10.82% | [8.21%,14.14%] |
| 全部consensus changes | 334 | 39 | 11.68% | [8.66%,15.56%] |
| 9B/consensus同动作交集 | 114 | 13 | 11.40% | [6.79%,18.54%] |

因此三种构造都不能直接进入SFT；交集在locked上只降低coverage，没有提高purity。

9B存在1.36× self-source enrichment：425个changed selections中197个含Qwen3.5 exact source。Self-only purity仅6.99%，Qwen3.5与其他来源共同支持时purity为18.52%，进一步支持independent agreement而非self-selection。

## Gate 与下一步

预注册 locked selector gate 已通过，但它只授权桥接实验，不授权直接训练。允许的下一步是：

1. 运行P100/P80/P60/P40、固定25% revision + 75% clean replay的受控LoRA纯度曲线，得到最低可容忍纯度 $p_{min}^{train}$；
2. 在独立 **train split** 上生成同构Pass@8 packets，冻结9B、consensus及同动作交集三个GT-free构造；
3. Matcher只做aggregate diagnosis，不做逐行选择；只有 $LB_{95}(p_v)\ge p_{min}^{train}$ 的variant可放行；
4. 单独建立uniform general-state、student-correct事后分层的回归安全对照；
5. 通过桥接gate后才构建正式25/75 LoRA arm，并做完整held-out policy evaluation。

本实验的dev/locked rows严禁用于训练。当前不应继续扩大corrector，也不应直接比较9B与consensus的训练增益；应先验证它们是否达到训练可容忍纯度。

完整桥接预登记见 [Pass@8 Selector → Training Bridge](../../docs/pass8_selector_to_training_bridge_zh.md)。
