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

## Gate 与下一步

预注册 locked selector gate 已通过。允许的下一步是：

1. 在完全独立的 **train split** 上生成同构 Pass@8 candidate packets；
2. 使用冻结的 Qwen3.5-9B selector 或零 GPU cross-source consensus 选择训练动作；
3. 仅构建 **25% selected revision + 75% clean replay** 的预授权训练臂；
4. 本实验的 dev/locked rows 严禁用于训练；
5. 训练后仍需完整 held-out policy evaluation，不能把 critical-step selector utility 当作最终 TSR。

当前不应因为 35B 更大而继续盲目扩大 corrector；优先比较 9B 与 cross-source consensus 在 train-split 25/75 arm 上的真实政策增益。
