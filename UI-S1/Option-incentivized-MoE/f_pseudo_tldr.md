# f_pseudo 分析 TL;DR

## 结论：当前形式的 f_pseudo 对 long-horizon 很可能无效

## 三个致命问题

| # | 问题 | 数据 |
|---|------|------|
| 1 | **无方向性** — f(s_start) ≈ f(s_end)，累积和≈0 | 所有长度分桶均值 < 0.003 |
| 2 | **信号稀疏** — 超半数步骤为零（自环） | 54.6% 零值，Long 达 72.6% |
| 3 | **长轨迹退化为噪声** — 正负振荡 | Long 轨迹 68.9% 符号一致性 < 0.3 |

## 根因

- Eigenfunction 学的是 **Fiedler vector**（图切割），不含起→终方向
- Screenshot hash 太粗，click/type 不改变 hash → 54.6% 自环

## 已修复

Off-by-one bug：f_pseudo 1-indexed，RL trainer 0-indexed → 查找 `step_id+1`，跳过最后一步

## 正面

- 覆盖率高：Medium/Long 100%，总体 96.3%
- Eigenfunction 本身可区分：per-app std≈1.0

## 下一步（按优先级）

1. **Ablation**：λ=0 vs λ=0.1 对比 TSR，确认是否有效
2. **替换为 Progress Estimator**：训 p(s)∈[0,1] 预测完成进度，保证有方向性
3. **改进状态表示**：用视觉 embedding 替代 screenshot hash，减少自环
