# Option-Incentivized MoE — 实现任务清单

> 基于深度覆盖选项的长视野 Web Agent 训练
> 参考文档：`Option-incentivized-MoE/Instruction`

---

## 概述

本任务清单实现基于谱图分析的选项发现方法，将 Deep Covering Options（Jinnai et al., ICLR 2020）迁移到 MoE GUI 自动化 agent 训练中，目标基准为 GUI-360° 和 AndroidWorld。

**已有基础设施：** 标准 MoE（路由器 + 4个专家 LoRA + 负载均衡）已在 `verl/models/moe/` 中完整实现（约4,400行代码）。RL/SFT 训练流水线、评估脚本、数据集加载器均已就绪。

**需要新建的模块：** 谱图分析、瓶颈引导的快捷操作发现、连通性感知路由、层次化策略。

---

## 任务依赖图

```
任务1: 状态表示 ✅
    └── 任务2: 转移数据收集 ✅
        └── 任务3: 拉普拉斯特征函数 (f_net) ✅
            ├── 任务3.1: VLM 特征函数 (LoRA + eigenfunction loss) ✅
            ├── 任务4: 瓶颈验证 ✅ GO (5/5)
            │   └── 任务5: Tool-Use 快捷操作 SFT（数据准备 ✅ / 训练待开始）
            │       └── 任务7: 快捷操作执行器 + 层次化推理
            │           └── 任务10: MVP 实验（GUI-360°）
            │               └── 任务11: AndroidWorld 扩展
            ├── 任务6: RL 伪奖励快捷操作训练（方法B）🔄 训练中 [任务5的替代方案]
            └── 任务8: 连通性感知路由损失（MoE）
                └── 任务9: 渐进式专家发现
                    └── 任务12: MoE 完整集成 ← 同时依赖 [8, 10]
                        └── 任务13: 多智能体扩展 (MA-GUI)

注：Tool-Use 快捷操作（层1，时间抽象）与 MoE Expert（层2，功能特化）正交组合，
    不互相替代。见任务5"架构分析"小节。
```

**MVP 关键路径：** 1 → 2 → 3 → 4 → 5 → 7 → 10
**VLM 特征函数路径：** 3 → 3.1（独立于 MVP 路径，用于对比验证）
**MoE 集成路径：** 3 → 8 → 9 → 12

---

## 第一阶段：基础 — 状态表示与图构建s

### 任务1：实现状态表示工具（用于图构建）
- **对应章节：** 5.2（Step 0）
- **状态：** `已完成` ✅
- **前置依赖：** 无
- **阻塞：** 任务2
- **范围：** 仅 GUI-360（AndroidControl 暂不实现）

**目标：** 从无障碍树（accessibility tree）和截图中提取结构化特征，构建转移图。

**方案A — 轻量级（优先）：** 基于无障碍元数据的哈希状态ID：
```python
# GUI-360
state_id = hash(app_domain, active_tab_signature, dialog_state, control_fingerprint)
```

**方案B — 表示学习：** 使用 SigLIP 编码截图，在嵌入空间构建 k-NN 图。表达能力更强，但引入近似噪声。

**已实现的模块和文件：**

| 文件 | 说明 |
|------|------|
| `verl/models/moe/state_representation.py` | 核心模块：状态哈希 + 稠密嵌入 + 轨迹处理器 |
| `tests/moe/test_state_representation.py` | 单元测试 + 真实数据集成测试（34个测试全通过） |
| `verl/models/moe/__init__.py` | 已更新，导出所有 state_representation 公共 API |

**核心 API：**
- `GUI360StepData.from_raw(raw_jsonl_dict)` → 结构化步骤数据
- `extract_state_id(step_data)` → `StateID`（MD5 哈希，用于图节点）
- `extract_state_embedding(step_data)` → `np.ndarray` shape=(43,)（用于 f_net 输入）
- `GUI360TrajectoryProcessor` → 批量处理轨迹文件，构建邻接表、缓存嵌入、保存/加载
- `save_transitions()` / `load_transitions()` → 转移记录 JSONL I/O

**状态哈希组成（方案A）：**
- `app_domain`：excel / word / ppt
- `active_tab_signature`：可见的标准 Ribbon 标签（排序、去插件标签）
- `dialog_state`：ui_tree 中 level-1 Window 子节点名（如 "Format Cells"、"Paragraph"）
- `control_fingerprint`：控件类型分布的分桶指纹（DataItem/Button/TabItem/... 计数分桶）

**稠密嵌入组成（43维）：**
- App domain one-hot（3维）
- 控件类型归一化分布（18维）
- Ribbon 标签存在向量（20维）
- 对话框指示符（1维）
- 控件总数 log-scaled（1维）

**可复用的已有代码：**
- `verl/utils/dataset/rl_dataset.py` — 视觉+文本的轨迹处理
- `x/data/agent/base.py`、`x/io/image_io.py` — 数据/图像 I/O
- GUI-360 数据集包含无障碍元数据；AndroidControl 包含 a11y 树

---

### 任务2：构建转移数据收集流水线
- **对应章节：** 5.3（Step 1）
- **状态：** `已完成` ✅
- **前置依赖：** 任务1
- **阻塞：** 任务3
- **范围：** 仅 GUI-360

**目标：** 从已有轨迹数据集中提取状态转移，构建状态转移图。

**已实现的模块和文件：**

| 文件 | 说明 |
|------|------|
| `scripts/collect_transitions.py` | 转移数据收集脚本（CLI，支持 --granularity / --max-files） |
| `outputs/transitions/gui360_full/` | fine 模式全量运行输出 |
| `outputs/transitions/gui360_coarse/` | coarse 模式全量运行输出 |

**输出文件（每个输出目录结构相同）：**

| 文件 | 大小(fine) | 说明 |
|------|------|------|
| `transitions.jsonl` | 18M | 所有 (state_hash, next_state_hash, action) 转移记录 |
| `transition_pairs.npz` | 31M | 稠密嵌入对 (src_embeddings, dst_embeddings)，直接用于 f_net 训练 |
| `all_state_embeddings.npz` | 1.9M | 所有唯一状态的稠密嵌入 (N×43) |
| `state_registry.json` | 2.3M | 每个状态哈希的元数据（app_domain, tabs, dialog, fingerprint）|
| `adjacency.json` | 859K | 邻接表 {src_hash: {dst_hash: count}} |
| `statistics.json` | 1.1K | 图统计摘要 |
| `per_app_statistics.json` | 302B | 按应用分类的统计 |

**全量运行结果（13,750 条成功轨迹，~233秒）— Fine vs Coarse 对比：**

| 指标 | Fine 模式 | Coarse 模式 |
|------|-----------|-------------|
| 唯一状态数 | 6,511 | **989** (↓6.6x) |
| 总转移数 | 91,618 | 91,618 |
| 唯一边数 | 16,606 | **2,515** |
| 图密度 | 0.000392 | **0.002574** (↑6.6x) |
| 连通分量数 | 123 | **16** |
| 单次访问状态占比 | 59.1% | **52.8%** |
| 平均出度 | 2.75 | 2.75 |
| 最大出度 | 89 | 92 |

**Per-App 统计（Fine / Coarse）：**

| 应用 | States (F/C) | Transitions | Edges (F/C) |
|------|-------------|-------------|-------------|
| Excel | 1,414 / 243 | 25,015 | 3,484 / 613 |
| Word | 3,518 / 481 | 36,109 | 8,717 / 1,215 |
| PPT | 1,579 / 265 | 30,494 | 4,405 / 687 |

**图结构分析与对方法的影响：**

1. **图天然按应用分离** — Excel/Word/PPT 各自构成独立子图（无跨应用边）。这是正确的：不同 Office 应用的 UI 状态不互通。Coarse 模式下每个 app 的子图各自构成**单一连通分量**（Word=455, PPT=265, Excel=242），Laplacian 在每个 app 内部可直接使用。
2. **Fine 模式碎片化严重** — 6,511 状态中 59% 仅被 1 条轨迹访问，根因是 `control_fingerprint` 有 3,219 个唯一值（控件数量微小变化即产生新状态）。图被打散为 123 个连通分量，影响 Laplacian 分析质量。
3. **稀疏 = 链式拓扑 = 瓶颈清晰** — 图密度 0.0004~0.003 说明状态空间呈"长链"结构，正是 Deep Covering Options 最擅长的场景：链上连接稀疏处是 λ₂ 特征向量能精确定位的瓶颈。
4. **对话框是主要的状态区分因子** — 41.6% 的状态处于对话框打开状态（358 种唯一对话框），高频对话框如 "Word Options"(190)、"Excel Options"(156)、"Format Cells"(104) 对应已知的复杂操作路径。
5. **高复用 hub 状态存在** — 54 个状态被 >100 条轨迹访问（主界面/Home tab），是连接不同任务的"交通枢纽"，枢纽之间的薄弱连接就是瓶颈。

**训练数据与粒度解耦（已修复的关键设计）：**

`transition_pairs.npz` 存储的是每一步的**实际嵌入**（直接从原始 JSONL 数据计算），而非通过 hash 间接查找。因此 **f_net 的训练数据与 fine/coarse 粒度完全无关**（已验证两种模式生成的 `transition_pairs.npz` 逐元素相同）。

| 用途 | 数据源 | 与粒度的关系 |
|------|--------|-------------|
| f_net 训练 | `transition_pairs.npz`（任意模式均可） | **无关** — 91,618 对 43 维实际嵌入 |
| 图结构/Laplacian 分析 | `adjacency.json` (coarse) | coarse 更好 — per-app 各自连通 |
| 瓶颈验证/可视化 | `state_registry.json` (coarse) | coarse 更好 — 状态可读性高 |

Wu et al. 的平滑性目标自动让相邻步骤获得相似的 f 值 — 控件分布上的噪声被 smoothness 项自然平滑。

**对后续任务的建议：**
- **Task 3 f_net 训练：直接用 `transition_pairs.npz`**（任一模式均可，数据完全相同）
- **Task 4 瓶颈可视化/验证：用 coarse 模式的图**，per-app 子图各自连通，便于解读
- **Task 3 应 per-app 分别训练 f_net**（或依赖嵌入中 app one-hot 的隐式分离）

**用法：**
```bash
# Fine 模式全量运行（~4分钟）
conda run -n qwen3-eval python scripts/collect_transitions.py

# Coarse 模式
conda run -n qwen3-eval python scripts/collect_transitions.py --granularity coarse --output-dir outputs/transitions/gui360_coarse

# 快速测试
conda run -n qwen3-eval python scripts/collect_transitions.py --max-files 200
```

**已有数据位置：**
- `datasets/GUI-360/train/data/` — GUI-360 原始轨迹（13,750 个 JSONL 文件）

---

## 第二阶段：谱分析 — 特征函数与瓶颈发现

### 任务3：实现拉普拉斯特征函数近似（f_net）
- **对应章节：** 5.4（Step 2）
- **状态：** `已完成` ✅
- **前置依赖：** 任务2
- **阻塞：** 任务4、6、8

**目标：** 训练神经网络近似图拉普拉斯的第二特征函数，使用 Wu et al. (2019) 的无约束目标函数。

**核心目标函数（修正版）：**
$$\tilde{G}(f) = \frac{1}{2}\mathbb{E}_{(s,s')\sim H}\left[(f(s)-f(s'))^2\right] + \eta \cdot \left[(\mathbb{E}[f^2]-1)^2 + (\mathbb{E}[f])^2\right]$$

- 第一项（平滑性）：相连状态的 $f$ 值应接近 → 收敛值 = λ₂（第二特征值）
- 第二项（排斥性，权重 $\eta$）：归一化约束 $\mathbb{E}[f^2]=1$ + 正交约束 $\mathbb{E}[f]=0$
- 仅需**采样的状态转移对**，不需要完整邻接矩阵

**⚠️ 关键 bug 修复 — 排斥项公式：**

原始 TASK_LIST 中的公式 $\mathbb{E}[(f^2-1)(f'^2-1) + f^2f'^2]$ 展开为 $2(\mathbb{E}[f^2])^2 - 2\mathbb{E}[f^2] + 1$，其最小值在 $\mathbb{E}[f^2] = 0.5$ 处取到（值=0.5），导致 f_net 收敛到常数解 $f(s) = \pm 1/\sqrt{2}$（smoothness=0, repulsive=0.5, total=0.5），而非真正的特征向量。修正后的排斥项 $(\mathbb{E}[f^2]-1)^2 + (\mathbb{E}[f])^2$ 最小值在 $\mathbb{E}[f^2]=1, \mathbb{E}[f]=0$ 处取到（值=0），正确地强制 f 为归一化且正交于常数的特征向量。

| 指标 | 修复前（常数解） | 修复后（特征向量） |
|------|---------|---------|
| f_mean | ±0.707 | ≈ 0.001 |
| f_std | 0.012-0.020 | ≈ 0.997 |
| smoothness | ~6e-6 | ~0.010 (= λ₂) |
| repulsive | 0.5000 | ~0.001 |
| total loss | 0.5000 | ~0.012 |

**已实现的模块和文件：**

| 文件 | 说明 |
|------|------|
| `verl/models/moe/graph_analysis.py` | 核心模块：EigenfunctionNet + 训练 + 瓶颈识别 + 描述映射 |
| `scripts/train_eigenfunction.py` | CLI 训练脚本（per-app 模式，支持 --app / --epochs / --device） |
| `scripts/slurm_train_fnet.slurm` | SLURM 提交脚本（1 GPU, 200 epochs, all 3 apps） |
| `verl/models/moe/__init__.py`（已更新） | 导出 graph_analysis API |

**核心 API：**
- `EigenfunctionNet(input_dim, hidden_dims, output_dim=1)` — MLP 网络，Xavier 初始化
- `EigenfunctionConfig` — 超参数 dataclass（hidden_dims, epochs, batch_size, lr, eta, percentile_k, per_app, app_filter）
- `eigenfunction_loss(f_net, s, s_next, s_rand, eta)` → (total_loss, metrics_dict)
- `train_eigenfunction(transition_pairs_path, state_embeddings_path, config, output_dir, device, state_registry_path)` → (f_net, results_dict)
- `identify_bottlenecks(f_values, state_hashes, percentile_k)` → bottleneck_info dict
- `map_bottlenecks_to_descriptions(bottleneck_info, registry_path)` → described list（含 dialog_state, tabs 等）
- `load_f_net(checkpoint_path, config, device)` → 加载已训练模型

**训练配置：**
- 输入：`transition_pairs.npz`（91,618 对 43 维嵌入）、`all_state_embeddings.npz`
- Per-app 训练：通过 app one-hot（dim 0-2）过滤转移和状态
- 网络：MLP [43 → 256 → 256 → 1]，ReLU，Xavier init
- 优化：Adam lr=1e-3，Cosine LR schedule，weight_decay=1e-5
- 超参：eta=1.0，percentile_k=30.0，batch_size=2048，epochs=200
- 输出（per-app）：f_net_final.pt、f_values.npz、bottlenecks.json、bottlenecks_described.json、results.json、training_history.json、config.json

**GPU 训练结果（GH200 120GB，200 epochs，总计 ~103 秒）：**

| App | 训练时间 | Loss | Smoothness (λ₂) | f_mean | f_std | f_range | 瓶颈数 |
|-----|---------|------|---------|--------|-------|---------|--------|
| Excel | 29s | 0.0105 | 0.0100 | 0.001 | 0.995 | [-2.42, 1.74] | 424 |
| Word | 41s | 0.0128 | 0.0099 | 0.001 | 0.998 | [-3.39, 1.84] | 1056 |
| PPT | 33s | 0.0127 | 0.0121 | 0.001 | 0.997 | [-1.72, 3.36] | 474 |

**Top 瓶颈状态（f-value 最低）：**
- **Excel**: Afrikaans 本地化界面的 Data 标签视图（"Ordeningswaarskuwing"=排序警告, "Formateer selle"=格式化单元格, "Verwyder Duplikate"=删除重复项）— 非英语 UI 路径是罕见且难到达的状态
- **Word**: Table Layout 模式（双 Layout 标签，表示表格编辑上下文）、Developer 标签视图 — 需要特定操作序列才能进入
- **PPT**: 无 Home 标签的 Draw/Design 模式、"Insert Slide Zoom" 对话框 — 非标准视图模式

**训练曲线（200 epochs, GPU GH200）：**

*Excel (1,414 states, 25,015 transitions):*

| Epoch | Loss | Smoothness | Repulsive | NormPen | OrthoPen | f_mean | f_std |
|------:|-----:|-----------:|----------:|--------:|---------:|-------:|------:|
| 1 | 0.9409 | 0.0017 | 0.9393 | 0.4352 | 0.5041 | -0.092 | 0.095 |
| 10 | 0.0212 | 0.0170 | 0.0042 | 0.0018 | 0.0024 | 0.374 | 0.738 |
| 50 | 0.0145 | 0.0117 | 0.0028 | 0.0008 | 0.0020 | 0.322 | 0.752 |
| 100 | 0.0132 | 0.0107 | 0.0025 | 0.0008 | 0.0017 | 0.368 | 0.755 |
| 200 | 0.0105 | 0.0100 | 0.0005 | 0.0002 | 0.0003 | 0.362 | 0.756 |

*Word (3,518 states, 36,109 transitions):*

| Epoch | Loss | Smoothness | Repulsive | NormPen | OrthoPen | f_mean | f_std |
|------:|-----:|-----------:|----------:|--------:|---------:|-------:|------:|
| 1 | 0.7682 | 0.0038 | 0.7644 | 0.3243 | 0.4401 | 0.685 | 0.151 |
| 10 | 0.0291 | 0.0197 | 0.0094 | 0.0025 | 0.0068 | 0.355 | 0.860 |
| 50 | 0.0205 | 0.0150 | 0.0054 | 0.0026 | 0.0029 | 0.321 | 0.810 |
| 100 | 0.0151 | 0.0113 | 0.0037 | 0.0019 | 0.0018 | 0.259 | 0.775 |
| 200 | 0.0128 | 0.0099 | 0.0030 | 0.0025 | 0.0005 | 0.226 | 0.752 |

*PPT (1,579 states, 30,494 transitions):*

| Epoch | Loss | Smoothness | Repulsive | NormPen | OrthoPen | f_mean | f_std |
|------:|-----:|-----------:|----------:|--------:|---------:|-------:|------:|
| 1 | 0.7723 | 0.0033 | 0.7690 | 0.3650 | 0.4041 | -0.646 | 0.154 |
| 10 | 0.0397 | 0.0345 | 0.0052 | 0.0043 | 0.0009 | -0.148 | 0.771 |
| 50 | 0.0185 | 0.0140 | 0.0044 | 0.0019 | 0.0025 | -0.235 | 0.778 |
| 100 | 0.0155 | 0.0129 | 0.0026 | 0.0016 | 0.0011 | -0.217 | 0.759 |
| 200 | 0.0127 | 0.0121 | 0.0006 | 0.0004 | 0.0002 | -0.210 | 0.743 |

*训练曲线解读：*
- **Epoch 1→10：快速收敛阶段** — Loss 从 ~0.8 降到 ~0.03，repulsive 从 ~0.8 降到 ~0.005（约束迅速满足）
- **Epoch 10→100：平滑性优化** — Smoothness 缓慢下降（逼近 λ₂），repulsive 保持在 ~0.003
- **Epoch 100→200：精细调整** — Loss 变化 <0.003，repulsive 降到 <0.001（约束高度满足）
- 注：表中 f_mean/f_std 为训练 batch 级统计（偏向高频状态），全状态统计见上方 results 表（f_mean≈0.001, f_std≈0.997）

**输出目录：**

```
outputs/fnet/gui360/
├── excel/
│   ├── f_net_final.pt              # 训练好的模型 checkpoint
│   ├── f_values.npz                # 所有状态的 f 值（hashes + f_values）
│   ├── bottlenecks.json            # 瓶颈状态列表（hash + f_value）
│   ├── bottlenecks_described.json  # 瓶颈状态 + 人类可读描述
│   ├── results.json                # 训练结果摘要
│   ├── training_history.json       # 逐 epoch 训练指标
│   ├── config.json                 # 训练配置
│   └── checkpoint_epoch{50,100,150,200}.pt  # 中间 checkpoints
├── word/  (同上结构)
└── ppt/   (同上结构)
```

**瓶颈解读 — Fiedler 向量揭示的 UI 模式切换瓶颈**

Fiedler 向量（第二特征向量）揭示的是图的**基本二分结构** — 将状态空间沿连接最薄弱处切成两半。f-value 最低的状态和最高的状态分别位于两半的深处，而 f ≈ 0 的状态位于窄通道（切割边界）上。

*关键发现 1：f-value 与节点度/访问频率几乎无相关性*

| 区域 | 平均出度 | 平均入度 | 平均访问次数 |
|------|---------|---------|------------|
| 底部 30%（瓶颈端） | 2.1-2.4 | 2.1-2.4 | 6-10 |
| 中间 40% | 2.8-3.4 | 2.8-3.4 | 15-34 |
| 顶部 30%（hub 端） | 1.9-2.4 | 1.9-2.4 | 6-12 |

f-value vs 出度的相关系数 r ≈ 0.04（几乎为零）。瓶颈状态不是"连接少"的节点，而是**位于图二分结构一侧深处**的节点 — 从另一侧到达它们必须穿过窄通道。

*关键发现 2：每个 App 的二分结构对应明确的 UI 模式切换*

**Excel — Data 分析视图 vs 完整 Ribbon 编辑视图：**

| f 值区间 | UI 模式 | 典型状态特征 |
|---------|--------|------------|
| f ≪ 0（瓶颈端） | Data-only tab 视图 | 只有 Data 标签可见，Afrikaans 本地化，Power Query / 数据排序对话框 |
| f ≫ 0（hub 端） | 完整 Ribbon | Home/Insert/Draw/Page Layout 等全部可见，Format Cells / Style 等常用对话框 |

**Word — 表格编辑模式 vs 普通文字编辑模式：**

| f 值区间 | UI 模式 | 典型状态特征 |
|---------|--------|------------|
| f ≪ 0（瓶颈端） | Table Layout 模式 | 双 Layout 标签（Page Layout + Table Layout），Home/Insert 消失，Developer 可见 |
| f ≫ 0（hub 端） | 普通编辑模式 | Home/Draw/Design/Layout/References/Mailings 全部可见，无对话框 |

**PPT — 非标准视图模式 vs 标准/嵌入编辑：**

| f 值区间 | UI 模式 | 典型状态特征 |
|---------|--------|------------|
| f ≪ 0（瓶颈端） | Draw/Design 无 Home 的视图 | Transitions/Animations 可见但缺 Home，Slide Zoom 等高级功能 |
| f ≫ 0（hub 端） | 嵌入 Excel 图表编辑 | PPT 内嵌 Excel 时出现 Page Layout/Data/Formulas 标签，Developer 可见 |

*关键发现 3：轨迹层面验证 — 仅 3.1% 的轨迹跨越瓶颈边界*

```
轨迹 f-value range 分布（12,177 条轨迹）：
  p50 = 0.045    ← 一半的轨迹几乎不移动
  p90 = 0.384
  p95 = 0.725
  p99 = 1.544

跨越瓶颈的轨迹比例：
  range > 1.0:  383 / 12,177 (3.1%)
  range > 2.0:   57 / 12,177 (0.5%)
  range > 3.0:    6 / 12,177 (0.05%)
```

*关键发现 4：跨越瓶颈的轨迹具有锐利的单步 f-value 跳变*

以下是实际跨越轨迹的完整 f-value 序列和对应任务：

**Word 示例 — 表格编辑 ↔ 普通编辑 跨越：**

```
word_4_3108 — "Delete the blank page in the Word document" (14 steps)
  step 0: f=-3.06  [Layout,Ref,Review,View,Help,Dev]  select_text  ← 表格编辑模式
  step 1: f=-3.06  [Layout,Ref,Review,View,Help,Dev]  type
  step 2: f=-3.04  [Layout,Ref,Review,View,Help,Dev]  click
  step 3: f=+0.14  [空]                               click         ← CROSSING (Δf=+3.18)
  step 4: f=-3.04  [Layout,Ref,Review,View,Help,Dev]  select_text   ← 返回表格模式
  ...
  step11: f=+0.14  [空]                               click         ← 再次跨出
  step12: f=+0.14  [空]                               click
  step13: f=+0.21  [空]                               (end)
```
→ Agent 需要**点击表格外部**才能退出 Table Layout 模式。

```
word_4_3092 — "Disable the AutoRecover feature in Word" (15 steps)
  step 0: f=-3.04  [Layout,Ref,Review,View,Help,Dev]  click         ← 表格编辑模式
  step 1: f=+0.14  [空]                               click         ← CROSSING: 退出表格
  step 2: f=+0.14  [空]                               click
  step 3: f=-2.98  [Layout,...,Dev] dialog=Word Options click        ← 返回+打开设置
  step 4: f=-3.04  [Layout,...,Dev] dialog=Word Options click
  step 5: f=-3.04  [Layout,...,Dev] dialog=Word Options click
  ...
  step 9: f=+0.14  [空]                               click         ← 再次跨出
  step10: f=+0.14  [空]                               click
  step11: f=-3.08  [Layout,...,Dev] dialog=Word Options click        ← 返回设置
  ...
```
→ 修改 AutoRecover 需要反复在表格模式和文件菜单之间切换。

**Excel 示例 — Data 视图 ↔ 标准视图 跨越：**

```
excel_4_7609 — "Protect the Excel workbook with a password" (18 steps)
  step 0: f=-2.08  [Data]                              click         ← Data-only 视图
  step 1: f=-2.08  [Data]                              click
  ...                                                                ← 反复操作 Data 视图
  step 9: f=-2.08  [Data]                              click
  step10: f=+0.05  [空]                                click         ← CROSSING: 退出 Data (Δf=+2.13)
  step11: f=+0.18  [空]                                click
  ...
  step13: f=+0.22  [空] dialog=Enkripteer Dokument     type          ← 加密文档对话框
  ...
```
→ 从 Data 分析视图跳转到 File > Info 加密工作簿，需要先退出 Data 模式。

```
excel_4_3559 — "Apply Accent 6 color to cells" (7 steps)
  step 0: f=-1.77  [Home,Insert,Page Layout,...,Data]  click
  step 1: f=-1.67  [Home,Insert,Page Layout,...,Data]  click
  ...
  step 4: f=+0.60  [空]                                type          ← CROSSING (Δf=+2.36)
  step 5: f=+0.60  [空]                                click
  step 6: f=+0.60  [空]                                (end)
```
→ 从受限 Ribbon 进入颜色设置需要切换视图模式。

**PPT 示例 — 嵌入编辑 ↔ 幻灯片编辑 跨越：**

```
ppt_4s_1703 — "Embed an Excel chart into Slide 4" (13 steps)
  step 0: f=+3.30  [Insert,Draw,Page Layout,...,Data]  click         ← 嵌入 Excel 编辑模式
  step 1: f=+3.36  [...] dialog=Insert Chart           click
  step 2: f=+3.17  [...] dialog=Automatically saving   (空)
  ...
  step 5: f=+1.29  [Home,Insert,Draw,Design,...,Dev]   click         ← CROSSING: 退回 PPT (Δf=-1.88)
  step 6: f=+1.30  [Home,Insert,Draw,Design,...,Dev]   click         ← PPT 正常编辑
  ...
```
→ 在 PPT 中嵌入 Excel 图表后，UI 进入 Excel 编辑模式（Data/Formulas 标签出现），关闭后跳回 PPT 编辑模式。

```
ppt_4s_2245 — "Find and apply gothic background to Slide 1" (24 steps)
  step 0: f=+0.34  [空]                                type
  step 1: f=+1.35  [空]                                click
  ...                                                                ← 搜索+选择背景
  step12: f=+1.41  [空]                                (空)
  step13: f=-0.54  [Home,Insert,Draw,Design,...]       click         ← CROSSING: 进入编辑 (Δf=-1.95)
  step14: f=-0.55  [Home,Insert,Draw,Design,...]       click
  ...
  step19: f=-0.67  [...] dialog=Insert Picture          click         ← 插入图片对话框
  ...
```
→ 从搜索/浏览视图回到幻灯片编辑模式，然后插入背景图片。

*瓶颈的本质：UI 模式切换是 Office 应用的基本拓扑结构*

| 瓶颈类型 | App | 窄通道 | 跨越动作 | 频率 |
|----------|-----|--------|---------|------|
| 表格编辑 ↔ 普通编辑 | Word | 点击表格外部 / 点击回表格 | 单步 click | 45 条轨迹 |
| Data 视图 ↔ 完整 Ribbon | Excel | 切换标签 / File 菜单 | 单步 click | 51 条轨迹 |
| 嵌入对象编辑 ↔ 幻灯片编辑 | PPT | 点击图表外部 / 双击图表 | 单步 click | 34 条轨迹 |

*对后续任务的影响与建议*

1. **瓶颈是真实的** — f_net 成功识别了 Office 应用内 UI 模式之间的结构性分隔，跨越轨迹的 f-value 跳变模式与物理含义一致
2. **跨越本身通常是单步** — 跳变发生在 1 步内（如"点击表格外部"），因此快捷操作的价值不在于自动化跨越动作本身
3. **快捷操作的价值在于"接近+跨越+着陆"序列** — 例如 word_4_3092 的"退出表格 → 打开 Word Options → 修改设置 → 返回表格"是一个 5-8 步的完整子策略，整体可作为一个快捷操作
4. **Task 5 应提取包含跨越的完整子序列** — 不仅仅是跨越点，而是跨越前 2-3 步 + 跨越 + 跨越后 2-3 步的完整 segment
5. **考虑更高阶特征向量** — 当前 Fiedler vector 只给出一个二分；第 3、4 特征向量可能揭示更细粒度的分区（如 Excel 内的 "公式编辑 vs 格式设置"）
6. **percentile_k=30 可能需要调整** — 30% 的状态被标记为瓶颈过于宽泛，对于 Task 5 提取跨越片段，可考虑 k=10 或基于 |f| 的绝对阈值

**参考文献：** Wu et al. 2019; Jinnai et al. ICLR 2020

---

### 任务3.1：VLM 特征函数训练（Qwen2.5-VL + LoRA + eigenfunction loss）
- **对应章节：** 5.4（Step 2 的 VLM 扩展）
- **状态：** `已完成` ✅
- **前置依赖：** 任务3
- **阻塞：** 无（独立于 MVP 路径，用于对比验证）

**目标：** 用 Qwen2.5-VL-7B + LoRA + 回归头替代 43 维 MLP，直接在截图上训练 Laplacian 第二特征函数。

**动机：** Task 3 的 43 维手工嵌入（Ribbon 标签计数 + 控件类型分布 + 对话框指示符）丢失了截图中的视觉信息。VLM 可以直接从像素中学习状态表示，捕获手工特征无法表达的视觉差异。

**两种输入模式：**
1. `screenshot` — 仅图片 + 固定 prompt
2. `screenshot_a11y` — 图片 + 压缩 a11y tree 文本（推荐）

**已实现的模块和文件：**

| 文件 | 说明 |
|------|------|
| `verl/models/moe/vlm_eigenfunction.py` | 核心模块：VLMEigenfunctionModel + 数据集 + 损失函数 |
| `scripts/prepare_vlm_eigenfunction_data.py` | 数据准备：state_hash → screenshot + a11y 映射 |
| `scripts/train_vlm_eigenfunction.py` | 训练脚本（per-app，支持 --app/--input-mode/--lora-rank） |
| `scripts/slurm_train_vlm_fnet.slurm` | SLURM 提交脚本（1 GPU, 24h） |
| `verl/models/moe/__init__.py`（已更新） | 导出 vlm_eigenfunction API |

**核心 API：**
- `VLMEigenfunctionConfig` — 配置 dataclass（model_path, lora_rank, input_mode, batch_size, ...）
- `VLMEigenfunctionModel(config)` — Qwen2.5-VL + LoRA + RegressionHead
  - `forward(input_ids, attention_mask, pixel_values, image_grid_thw)` → (B, 1) f-values
  - `save_adapter(output_dir)` / `load_adapter(adapter_dir, config)` — 保存/加载 LoRA + head
- `VLMTransitionDataset(transitions_path)` → 转移对 (src_hash, dst_hash)
- `VLMStateDataset(manifest_path)` → 所有唯一状态 hash（用于随机采样）
- `VLMCollator(manifest, processor, input_mode)` → 从 hash 构建 VLM batch
- `vlm_eigenfunction_loss(f_src, f_dst, f_rand, eta)` → 特征函数损失

**模型架构（GPU 实测参数）：**
```
Qwen2.5-VL-7B-Instruct (frozen, bf16)
  ├── LoRA (r=16, α=32) on q/k/v/o_proj → 10,092,544 params (0.12%)
  ├── RegressionHead: LayerNorm(3584) → Linear(3584, 1) → 10,753 params (bf16)
  └── Mean-pool hidden states → scalar f-value
  Total base parameters: 8,302,259,200
```

**训练流程：**
```
Step 1: prepare_vlm_eigenfunction_data.py
    transitions.jsonl → state_hash → (execution_id, step_id)
                      → screenshot_path + ui_tree → compressed a11y text
    输出: state_manifest.json + transition_pairs.json (per-app)

Step 2: train_vlm_eigenfunction.py
    For each batch:
      1. Sample (src_hash, dst_hash) transition pairs
      2. Sample random state hashes (for repulsive term)
      3. Stack all 3 groups → single VLM forward pass (3B images)
      4. Split outputs → f_src, f_dst, f_rand
      5. Eigenfunction loss → backward → update LoRA + head
    保存: LoRA adapter + regression_head.pt + f_values.npz + bottlenecks.json
```

**关键设计决策：**
- 三组图片（src, dst, rand）堆叠成单个 batch 做一次 VLM forward，然后拆分输出
- 固定 448×448 分辨率 → ~256 visual tokens/image → B=2 时 6 images ≈ 1536 visual tokens
- gradient_checkpointing + bf16 以适应 GH200 120GB
- LoRA 参数 10,092,544 (0.12%)，回归头 10,753 参数

**GPU 实测张量形状（batch_size=2, 即 6 images per step）：**
```
pixel_values:  torch.Size([6144, 1176]), dtype=torch.float32
input_ids:     torch.Size([6, 349]),     dtype=torch.int64
attention_mask: torch.Size([6, 349]),    dtype=torch.int64
image_grid_thw: torch.Size([6, 3]),     dtype=torch.int64
f_all (output): torch.Size([6, 1]),     dtype=torch.bfloat16
```

**a11y 压缩格式：**
```
Window: Excel | Tab: Data,Insert,... | Dialog: Format Cells | Controls: Button:3,Edit:1,...
```

**训练配置（最终版）：**
- 模型：Qwen2.5-VL-7B-Instruct + LoRA (r=16, α=32, q/k/v/o_proj)
- 输入：448×448 截图 + 压缩 a11y text（~200 chars）
- 优化：AdamW lr=1e-4, CosineAnnealing (T_max=2), weight_decay=1e-5
- Batch: 16 transitions → 48 images per step (src + dst + rand)
- Epochs: 2
- 损失：eigenfunction loss（smoothness + η × repulsive），非 SFT
- Checkpoint 频率：每 epoch 保存一次（save_every=1）
- 3 个 app 各自独立 SLURM 作业并行训练

**GPU 训练结果（GH200 120GB, batch_size=16, 2 epochs, 3 个并行 SLURM 作业）：**

| App | SLURM Job | Node | 训练时间 | Loss | Smoothness | Repulsive | f_mean | f_std | f_range | 瓶颈数 |
|-----|-----------|------|---------|------|-----------|-----------|--------|-------|---------|--------|
| Excel | 2510568 | nid011221 | 5.1h | 0.136 | 0.034 | 0.102 | 0.058 | 0.969 | [-1.45, 2.06] | 423 |
| Word | 2510569 | nid011247 | 7.3h | 0.145 | 0.049 | 0.096 | 0.049 | 0.980 | [-1.42, 1.67] | 1051 |
| PPT | 2510570 | nid011247 | 6.5h | 0.144 | 0.031 | 0.113 | -0.085 | 1.050 | [-1.57, 2.09] | 470 |

**训练曲线（2 epochs）：**

| App | Epoch | Loss | Smooth | Repulsive | f_src_mean | f_rand_std |
|-----|------:|-----:|-------:|----------:|-----------:|-----------:|
| Excel | 1 | 0.219 | 0.052 | 0.167 | -0.376 | 0.978 |
| Excel | 2 | 0.136 | 0.034 | 0.102 | -0.356 | 0.989 |
| Word | 1 | 0.235 | 0.079 | 0.157 | -0.136 | 0.972 |
| Word | 2 | 0.145 | 0.049 | 0.096 | -0.175 | 0.986 |
| PPT | 1 | 0.222 | 0.037 | 0.184 | -0.200 | 0.979 |
| PPT | 2 | 0.144 | 0.031 | 0.113 | -0.078 | 0.986 |

*训练观察：*
- f_rand_std ≈ 0.99（接近目标 1.0）：归一化约束基本满足
- Smoothness (0.031-0.049) 高于 MLP (0.010-0.012)：VLM 更难优化，但仍在下降
- Repulsive (0.096-0.113) 仍显著：正交+归一化约束未完全满足，可能需要更多 epochs
- Epoch 2 lr=0（CosineAnnealing T_max=2 到达最低点），考虑未来用 T_max > num_epochs

**VLM vs MLP 瓶颈对比：**

| App | 公共状态 | VLM 瓶颈 | MLP 瓶颈 | 重叠 | VLM独有 | MLP独有 | **Jaccard** |
|-----|---------|---------|---------|------|--------|--------|:----------:|
| Excel | 1,414 | 423 | 424 | 125 | 298 | 299 | **0.173** |
| Word | 3,518 | 1,051 | 1,056 | 345 | 706 | 711 | **0.196** |
| PPT | 1,573 | 470 | 471 | 221 | 249 | 250 | **0.307** |

**VLM Top-10 瓶颈状态（f-value 最低）：**

*Excel:*
1. f=-1.45  "Type your text here" (SmartArt 文本输入) — 完整 Ribbon + Data/Formulas
2. f=-1.34  "Using multiple panes?" — Developer + Data/Formulas 标签
3. f=-1.33  无对话框 — Developer + Data/Formulas 标签
4. f=-1.32  "Insert Chart" — Developer + Data/Formulas
5. f=-1.30  "How do I turn on AutoSave?" — Developer + Data/Formulas

*Word:*
1. f=-1.42  "Word Options" — 完整 Ribbon + Mailings/References
2. f=-1.42  "Word Options" — 同上（不同控件状态）
3. f=-1.39  "Caption" — 空 Ribbon（全部标签不可见）
4. f=-1.38  无对话框 — 空 Ribbon
5. f=-1.38  "Word Options" — 双 Layout 标签（表格编辑模式）

*PPT:*
1. f=-1.57  "Insert Stock Image" — Design/Transitions/Animations 可见
2. f=-1.56  "Insert Stock Image" — 同上
3. f=-1.51  无对话框 — Developer + Transitions/Animations
4. f=-1.49  无对话框 — Transitions/Animations 无 Design
5. f=-1.49  "Type your text here" — Developer + Transitions/Animations

**VLM vs MLP 瓶颈解读 — 两种模型捕获不同的 UI 复杂性维度：**

| 维度 | MLP (43-dim embedding) | VLM (screenshot + a11y) |
|------|----------------------|------------------------|
| **Excel 瓶颈** | Afrikaans 本地化界面 + Data-only tab | 对话框交互状态 (SmartArt, Insert Chart, AutoSave) |
| **Word 瓶颈** | 双 Layout 标签 (表格编辑模式) | Word Options 设置 + Caption 对话框 + 空 Ribbon |
| **PPT 瓶颈** | Draw/Design 无 Home 的非标准视图 | Insert Stock Image/Picture 对话框 + 动画标签 |
| **共同特征** | 非常规 Ribbon 标签组合 | 对话框 + 非常规 Ribbon 标签组合 |

*关键差异：*
- **MLP** 对**标签签名**高度敏感（Data-only、双 Layout）→ 瓶颈主要是**标签组合罕见**的状态
- **VLM** 能看到**截图视觉内容** → 瓶颈更多是**对话框交互复杂**的状态（SmartArt 编辑、图表插入、图片选择）
- Jaccard 0.17-0.31（部分重叠）说明两种方法捕获**互补**的信息
- PPT 重叠最高 (0.31)：该 app 的瓶颈更依赖标签结构，两种方法一致性较好

**数据准备结果（已完成）：**

| App | States | Transitions | Missing Screenshots | A11y 覆盖率 | 平均 A11y 长度 |
|-----|--------|-------------|--------------------:|:-----------:|:-------------:|
| Excel | 1,414 | 25,015 | 0 | 100% | 181 chars |
| Word | 3,518 | 36,109 | 0 | 100% | 170 chars |
| PPT | 1,573 | 30,484 | 6 (0.38%) | 100% | 187 chars |

**已修复的 Bug（SLURM 调试历程）：**

1. **Job 2507371 — dtype 不匹配 (`RuntimeError: expected BFloat16 but found Float`)**
   - 原因：VLM 以 bf16 加载，但 `RegressionHead` 默认初始化为 float32
   - 修复：`.to(dtype)` + `from_pretrained()` 中 `torch_dtype=` → `dtype=`

2. **Job 2510018 — 训练挂起 13+ 分钟无进度**
   - 原因：`gradient_checkpointing_enable()` 在 LoRA apply 之前调用
   - 修复：移到 `get_peft_model()` 之后 + `enable_input_require_grads()`

3. **52 个 missing screenshots（初始数据准备）**
   - 原因：GUI-360 截图文件编号有跳跃
   - 修复：索引匹配回退机制，Excel/Word → 0, PPT → 6

4. **f-values 全为 0（推理阶段 `Got unsupported ScalarType BFloat16`）**
   - 原因：`f_batch.cpu().numpy()` 失败，numpy 不支持 bf16
   - 修复：添加 `.float()` → `f_batch.float().cpu().numpy()`
   - 重新计算：`scripts/recompute_vlm_fvalues.py` 从已保存 checkpoint 重新推理

**输出目录：**
```
outputs/vlm_fnet/
├── data/              # ✅ 已生成
│   ├── excel/state_manifest.json + transition_pairs.json
│   ├── word/state_manifest.json + transition_pairs.json
│   └── ppt/state_manifest.json + transition_pairs.json
├── excel/             # ✅ 训练完成
│   ├── lora_adapter/             # LoRA adapter weights
│   ├── regression_head.pt        # Regression head weights
│   ├── checkpoint_epoch{1,2}/    # Per-epoch checkpoints
│   ├── f_values.npz              # VLM f-values (recomputed)
│   ├── bottlenecks.json          # 423 瓶颈状态
│   ├── bottlenecks_described.json
│   ├── mlp_comparison.json       # Jaccard=0.173
│   ├── results.json
│   ├── training_history.json
│   └── config.json
├── word/              # ✅ 训练完成 (1051 bottlenecks, Jaccard=0.196)
├── ppt/               # ✅ 训练完成 (470 bottlenecks, Jaccard=0.307)
├── vlm_fnet_2510568.{log,err}    # Excel SLURM 日志
├── vlm_fnet_2510569.{log,err}    # Word SLURM 日志
└── vlm_fnet_2510570.{log,err}    # PPT SLURM 日志
```

**用法：**
```bash
# 数据准备
python scripts/prepare_vlm_eigenfunction_data.py --app all

# 训练（单 app SLURM）
sbatch --job-name=vlm_excel --export=APP=excel scripts/slurm_train_vlm_fnet.slurm

# 重新计算 f-values（从 checkpoint）
python scripts/recompute_vlm_fvalues.py --app all --device auto
```

---

### 任务4：验证瓶颈识别结果（对照已知困难转移）
- **对应章节：** 8.2 Step 3
- **状态：** `已完成` ✅
- **前置依赖：** 任务3
- **阻塞：** 任务5
- **Go/No-Go 判定：** **GO**（5/5 标准全部通过）

**目标：** 验证 f 值识别的瓶颈是否与人类直觉一致（已知的困难 UI 转移），通过五大分析模块产出结构化报告，并给出 Go/No-Go 判定。

**已实现的模块和文件：**

| 文件 | 说明 |
|------|------|
| `scripts/validate_bottlenecks.py` | 验证脚本（~660 行，CPU 运行，<5 秒完成） |
| `outputs/bottleneck_validation/report.json` | 结构化 JSON 报告 |
| `outputs/bottleneck_validation/report.md` | 人类可读 Markdown 报告 |
| `outputs/bottleneck_validation/{excel,word,ppt}/*.png` | 12 张 per-app 可视化（直方图、箱线图、聚类柱状图） |
| `outputs/bottleneck_validation/aggregate/*.png` | 4 张跨应用可视化（KDE 叠加、轨迹散点图、覆盖率热力图） |

**用法：**
```bash
conda run -n qwen3-eval python scripts/validate_bottlenecks.py           # 全部 3 个 app
conda run -n qwen3-eval python scripts/validate_bottlenecks.py --app excel  # 单 app
conda run -n qwen3-eval python scripts/validate_bottlenecks.py --percentile-k 10  # 更严格阈值
```

**五大分析模块结果：**

*模块 1：类别分类 -- 预期瓶颈类别 vs 实际识别*

每个 app 定义了 3 个预期瓶颈类别，计算 precision（多少瓶颈属于已知类别）和 recall（每类别中多少被标为瓶颈）。

**Excel（precision: 66.98%，3/3 类别命中）：**

| 类别 | 描述 | 瓶颈数 | 全部匹配 | Recall |
|------|------|--------|---------|--------|
| ribbon_nav_chain | Data-only 视图（缺 Home 标签） | 72 | 76 | 94.74% |
| dialog_chain | 对话框密集状态（Format Cells / Excel Options 等） | 195 | 716 | 27.23% |
| localized_ui | 非英语 / Afrikaans UI 路径 | 17 | 17 | 100.00% |

**Word（precision: 85.61%，3/3 类别命中）：**

| 类别 | 描述 | 瓶颈数 | 全部匹配 | Recall |
|------|------|--------|---------|--------|
| table_layout_mode | 双 Layout 标签（表格编辑上下文） | 79 | 179 | 44.13% |
| dialog_chain | 对话框密集状态（Word Options 等） | 293 | 1,213 | 24.15% |
| developer_mode | Developer 标签 + 非标准视图 | 532 | 716 | 74.30% |

**PPT（precision: 90.51%，3/3 类别命中）：**

| 类别 | 描述 | 瓶颈数 | 全部匹配 | Recall |
|------|------|--------|---------|--------|
| embedded_object_edit | 嵌入对象编辑（含 Data/Formulas 标签） | 1 | 6 | 16.67% |
| non_standard_view | 非标准视图（缺 Home 标签） | 150 | 175 | 85.71% |
| dialog_chain | 对话框密集状态（PPT Options / Insert Picture 等） | 278 | 763 | 36.44% |

*关键发现：* 67-91% 的瓶颈可归类到预期类别中，证明 f_net 识别的瓶颈与人类直觉高度一致。ribbon_nav_chain（94.7% recall）和 localized_ui（100% recall）两类瓶颈几乎被完全捕获。dialog_chain 的 recall 较低（24-36%）是因为该类别覆盖面广，大量对话框状态位于 f-value 中间区间而非极端区间。

*模块 2：瓶颈聚类*

按 (tab_category, dialog_state) 分组，合并小簇（<3 成员）后生成人类可读聚类。

| App | 聚类数 | 最大聚类 | 说明 |
|-----|--------|---------|------|
| Excel | 27 | full_ribbon_developer\|none (133) | 开发者模式 + 无对话框 = 常规编辑的"深处" |
| Word | 40 | reduced_ribbon\|none (532) | 缺 Home 的精简视图 = 表格/开发者模式主体 |
| PPT | 33 | reduced_ribbon\|none (150) | 无 Home 的 Draw/Design 视图 |

*模块 3：图结构验证*

在 fine-grained 邻接图上计算瓶颈分区的切割质量。

| 指标 | Excel | Word | PPT |
|------|-------|------|-----|
| Cut ratio | 0.0064 | 0.0283 | 0.0322 |
| **Normalized cut** | **0.0586** | **0.1857** | **0.2549** |
| Conductance | 0.0512 | 0.1509 | 0.2171 |
| 瓶颈/非瓶颈状态 | 424/990 | 1056/2462 | 474/1105 |

*关键发现：* Normalized cut 均远低于 0.5 阈值（Excel 0.059 尤为突出），证明 f_net 学到的瓶颈分区对应图中**真实的稀疏连接处**。Excel 的 cut ratio 仅 0.64%，意味着跨越瓶颈边界的边极少。

*模块 4：轨迹级分析*

按 execution_id 分组，分析跨越行为。

| 指标 | Excel | Word | PPT |
|------|-------|------|-----|
| 总轨迹数 | 4,169 | 5,410 | 3,658 |
| 跨越率 (range>1.0) | 2.28% | 1.37% | 5.90% |
| 跨越率 (range>2.0) | 0.46% | 0.59% | 0.16% |
| **跨越轨迹 avg max jump** | **1.4931** | **1.7603** | **1.1578** |
| 全轨迹 avg max jump | 0.1068 | 0.0990 | 0.1701 |
| 跨越轨迹平均长度 | 9.0 步 | 11.3 步 | 9.0 步 |
| 非跨越轨迹平均长度 | 5.9 步 | 6.6 步 | 8.3 步 |

*关键发现：*
1. **跨越轨迹极少**（1.4-5.9%），与 Task 3 的 3.1% 发现一致
2. **跨越轨迹的 f-value 跳变非常锐利**（avg max jump 1.16-1.76），说明瓶颈穿越通常发生在 1-2 步内
3. **跨越轨迹更长**（9-11 步 vs 6-8 步），确认跨越任务本身更复杂

*模块 5：相关性与分布分析*

| 指标 | Excel | Word | PPT |
|------|-------|------|-----|
| **Spearman(f, out_degree) \|r\|** | **0.026** | **0.004** | **0.061** |
| Spearman(f, in_degree) \|r\| | 0.032 | 0.013 | 0.056 |
| Spearman(f, visit_freq) \|r\| | 0.026 | 0.004 | 0.061 |
| Bimodality coefficient (Sarle's BC) | 0.613 | 0.703 | 0.753 |
| Gap ratio | 1.000 | 0.994 | 1.000 |

*关键发现：*
1. **f-value 与度数完全不相关**（\|r\| < 0.07），进一步确认瓶颈不等于低度数节点
2. **BC > 0.555** 对所有 app 成立（双峰性判据），f-value 分布确实是双峰的
3. **Gap ratio 约等于 1.0**，直方图中存在明显的谷底（瓶颈区与主区之间）

**Go/No-Go 判定 -- 5 条标准全部通过：**

| # | 标准 | 阈值 | Excel | Word | PPT | 通过 |
|---|------|------|-------|------|-----|------|
| 1 | 每个 app >= 2 个预期类别被识别 | >= 2 | 3 | 3 | 3 | YES |
| 2 | >= 40% 瓶颈属于预期类别 | >= 0.40 | 66.98% | 85.61% | 90.51% | YES |
| 3 | \|Spearman r(f, degree)\| < 0.3 | < 0.3 | 0.026 | 0.004 | 0.061 | YES |
| 4 | 跨越轨迹 avg max jump > 0.5 | > 0.5 | 1.493 | 1.760 | 1.158 | YES |
| 5 | Normalized cut < 0.5 | < 0.5 | 0.059 | 0.186 | 0.255 | YES |

**对后续任务的影响与建议：**

1. **Task 5 可以安全启动** -- 瓶颈验证全部通过，f_net 产出的瓶颈状态集是可靠的
2. **Task 5 提取跨越片段时，建议使用 f-value range > 1.0 作为跨越判定** -- 此阈值下每 app 有 50-216 条跨越轨迹可用于 SFT
3. **dialog_chain 类别的低 recall 不影响 Task 5** -- 因为 SFT 基于跨越轨迹而非瓶颈分类，低 recall 仅说明部分对话框状态的 f-value 不够极端
4. **PPT 的 embedded_object_edit 类别仅 1 个瓶颈**（recall 16.67%），建议 Task 5 对 PPT 重点关注 non_standard_view 和 dialog_chain

**输出目录：**

```
outputs/bottleneck_validation/
├── report.json                    # 结构化 JSON 报告（含完整数值）
├── report.md                      # 人类可读 Markdown 报告
├── excel/
│   ├── f_value_histogram.png      # f-value 分布直方图（瓶颈区域标红）
│   ├── f_value_by_dialog.png      # 按对话框状态分组的箱线图
│   ├── f_value_by_tab_category.png # 按标签类别分组的箱线图
│   └── bottleneck_cluster_summary.png # 瓶颈聚类柱状图
├── word/  (同上结构)
├── ppt/   (同上结构)
└── aggregate/
    ├── cross_app_f_distribution.png    # 三应用 KDE 叠加（清晰显示双峰）
    ├── crossing_trajectory_analysis.png # 轨迹 f-value range vs max jump 散点图
    ├── expected_vs_identified.png       # 预期类别 vs 实际识别对比柱状图
    └── category_coverage_heatmap.png   # Recall 热力图（覆盖率一目了然）
```

---

## 第三阶段：快捷操作构建

### 任务5：Tool-Use 快捷操作 SFT 训练
- **对应章节：** 5.6（方法A — 基于SFT，无需RL）
- **状态：** `数据准备已完成` ✅ / `训练待开始`
- **前置依赖：** 任务4 ✅
- **阻塞：** 任务7

**目标：** 从成功轨迹中提取跨越瓶颈边界的片段，以 tool-use 范式训练模型在合适时机调用快捷操作 tool。

**为什么需要这一步 -- 从瓶颈发现到快捷操作的逻辑链：**

GUI agent 当前面对的核心难题是**长视野决策**：一个 15-20 步的任务（如"在 Word 中禁
用 AutoRecover"），agent 每一步都要从截图出发决定点哪里。15 步 x 每步约 15 个候选动作 = 巨大的搜索空间，RL 的 credit assignment 困难（第 1 步的好坏要等第 15 步才知道）。

Task 3-4 揭示了关键事实：
1. **绝大多数轨迹（95-99%）都待在"主城区"（f > 0）内**，只有 1-6% 的轨迹穿越瓶颈边界
2. **恰恰是这些穿越轨迹对应最难的任务**（平均 9-11 步 vs 非穿越的 6-8 步）
3. **穿越的模式非常固定**——不管具体任务是什么，只要需要从 Table Layout 模式跳到 Word Options，都是类似的 5-8 步操作序列

因此，核心思路是：**把这些反复出现的"过窄路"子序列打包成快捷操作，让 agent 一键调用**。

```
之前（15 步的原始决策）：
  step 1:  click(表格外部)     ← agent 需要从零学习怎么退出表格模式
  step 2:  click(空白处)
  step 3:  click(File 菜单)
  step 4:  click(Options)
  step 5:  click(Save 选项卡)
  step 6:  取消勾选 AutoRecover
  step 7:  click(OK)
  ...共 15 步

有了快捷操作后（4 个高层决策，tool-use 格式）：
  decision 1: navigate_to_dialog(target_dialog="Word Options")  ← 一步搞定 4 步
  decision 2: click(Save tab)                                    ← 普通单步
  decision 3: click(AutoRecover checkbox)                        ← 普通单步
  decision 4: navigate_and_return(operation="confirm and close") ← 一步搞定 2 步

  有效决策视野从 15 缩短到 4 → RL/规划变得可行
```

这就是 Deep Covering Options (Jinnai et al., ICLR 2020) 的核心贡献——通过瓶颈识别发现快捷操作，缩短有效决策视野。

**为什么选 SFT 而非 RL：**
- 我们已经有成功穿越瓶颈的真实轨迹（Task 4 确认每 app 有 50-216 条 crossing 轨迹可用）
- 直接从这些轨迹中截取穿越片段，用 SFT 教会子策略"看到这种截图时，执行这个操作序列"
- 简单、可靠、数据充足。RL 方案（Task 6）作为备选，当 SFT 数据不够时使用

**Task 4 对本任务的数据支撑（已验证）：**

| App | 跨越轨迹数 (range>1.0) | 平均跨越长度 | 可用 SFT 样本 |
|-----|----------------------|------------|-------------|
| Excel | 95 条 | 9.0 步 | ~190 个片段（双向跨越） |
| Word | 74 条 | 11.3 步 | ~148 个片段 |
| PPT | 216 条 | 9.0 步 | ~432 个片段 |

**跨越片段提取：**
```python
crossing_segments = []
for traj in successful_trajectories:
    for i, state in enumerate(traj.states):
        if state not in bottleneck_states:
            # 找到轨迹进入瓶颈区域的位置
            for j in range(i + 1, len(traj.states)):
                if traj.states[j] in bottleneck_states:
                    segment = traj[i:j+1]  # (状态, 动作) 子序列
                    crossing_segments.append(segment)
                    break

macro_dataset = format_as_sft_data(crossing_segments)
macro_policy = finetune_vlm(base_model, macro_dataset)
```

**产出：**
1. 从成功的 GUI-360 轨迹中提取跨越片段的脚本
2. 格式化为 SFT 训练数据（兼容已有 SFT 流水线）
3. 训练模型识别快捷操作触发时机并输出对应 tool call
4. 快捷操作执行手册（playbook）供推理时执行器使用
5. 每个快捷操作替代约3-8步子序列

**可复用的已有基础设施：**
- SFT 训练器：`verl/trainer/fsdp_sft_trainer.py`
- GUI-360 SFT：`train_GUI_360/sft_scripts/`、`train_GUI_360/config/`
- 多轮 SFT 数据集：`verl/utils/dataset/gui_multiturn_sft_dataset.py`

#### 实现记录（2026-03-01）

**方案选择：Tool-Use 范式 vs 独立 LoRA 子策略**

最终实现采用 **tool-use 快捷操作**范式。原计划的"每个瓶颈训练一个独立 LoRA 子策略"面临以下问题：
- 推理时需要额外的控制器决定何时切换到子策略、何时切回
- 多个 LoRA adapter 的加载/切换增加系统复杂度
- 359 条跨越轨迹不足以训练 3 个独立子策略

Tool-use 方案的优势：
1. 现有 SFT 数据已经是 `<tool_call>` 格式，快捷操作只是新增的 tool
2. 不需要额外的高层控制器 — 模型自己决定何时调用
3. Qwen2.5-VL 天然支持 tool-use，training config 已有 `tools_key` 字段
4. 可组合、可扩展 — 新发现瓶颈只需添加 tool 定义 + 补充 SFT 数据

**3 个快捷操作 Tool：**

| Tool | 参数 | 用途 | 触发数 |
|------|------|------|--------|
| `navigate_to_dialog` | `target_dialog: str` | 多步导航到对话框（如 File > Options） | 338 |
| `switch_ui_mode` | `from_mode, to_mode: str` | UI 编辑模式切换 | 0* |
| `navigate_and_return` | `operation: str` | 多步往返操作 | 21 |

*`switch_ui_mode` 未触发，因大多数 UI 模式切换伴随对话框状态，被 `navigate_to_dialog` 优先捕获。

**运行结果：**

| App | 跨越轨迹数 | 快捷操作样本数 | navigate_to_dialog | navigate_and_return |
|-----|-----------|---------|-------------------|-------------------|
| Excel | 89 | 89 | 87 | 2 |
| Word | 69 | 69 | 59 | 10 |
| PPT | 201 | 201 | 192 | 9 |
| **Total** | **359** | **359** | **338** | **21** |

**混合数据集：** 原始 SFT 105,340 + 快捷操作增强 10x 上采样 3,590 = **108,930 samples**（快捷操作占 3.3%）

**输出 `outputs/macro_sft/`：**
```
├── crossing_trajectories/{excel,word,ppt}_crossings.json
├── macro_playbook.json          (359 entries)
├── macro_augmented_train.parquet  (359 samples, 5.2M)
├── macro_mixed_train.parquet      (108,930 samples, 354M)
├── macro_mixed_eval.parquet       (26,308 samples, 53M)
├── macro_tool_definitions.json    (3 tool JSON Schema)
└── statistics.json
```

**新建文件：**
| 文件 | 说明 |
|------|------|
| `scripts/prepare_macro_sft_data.py` | 数据准备 |
| `train_GUI_360/config/gui360_sft_macro.yaml` | 训练配置（LR=5e-6，继承 v4 config） |

**CLI：**
```bash
python scripts/prepare_macro_sft_data.py                    # 数据准备（CPU，~15秒）
python scripts/prepare_macro_sft_data.py --upsample-factor 20  # 增大快捷操作比例到 ~6.5%

python -m verl.trainer.fsdp_sft_trainer \
  --config-path train_GUI_360/config --config-name gui360_sft_macro  # 训练
```

#### Tool-Use 快捷操作 vs MoE 结构：架构分析

Task 5 的实现引出一个关键架构问题：**快捷操作学习应该用 tool-use SFT 还是 MoE expert routing？**

**核心差异：两者作用在不同的抽象层级**

| 维度 | Tool-Use 快捷操作（Task 5） | MoE Expert（`verl/models/moe/`） |
|------|----------------------|--------------------------------|
| **抽象层级** | 时间抽象（temporal）：一个快捷操作 = 5-8 步动作序列 | 功能特化（functional）：一个 expert = 单步动作的参数化变体 |
| **路由信号** | 视觉状态 + 任务进度（"UI 处于瓶颈前"） | 指令语义类型（"这是 click/type 指令"） |
| **执行方式** | 外部执行器逐步执行 playbook 动作序列 | 模型单次前向传播，expert LoRA 调整输出 |
| **训练数据** | 359 条跨越轨迹（极度稀疏） | 105K 全量 SFT 数据 |
| **粒度** | 决策级别：何时启动多步子策略 | Token 级别：每个 token 由哪个 expert 处理 |

**为什么不应该用 MoE 替代 tool-use 快捷操作：**
1. MoE expert 是并行路径（每个 token 选 expert），无法表达"接下来 5 步都用同一子策略"
2. "是否需要跨越瓶颈"是视觉状态判断，instruction 文本不含此信息 → TextOnlyRouter 无法路由
3. 359 条数据不足以训练有意义的 expert 特化

**正确的组合方式：正交层叠**

```
┌──────────────────────────────────────────────────────┐
│ 层1: Tool-Use 快捷操作决策（时间抽象，Task 5）                │
│   模型看到截图 → 判断是否调用快捷操作 tool                      │
│   如果是 → 外部执行器处理 5-8 步                         │
│   如果否 → 进入层2                                     │
├──────────────────────────────────────────────────────┤
│ 层2: MoE Expert 路由（功能特化，现有 MoE）                │
│   TextOnlyRouter → 选 expert LoRA                     │
│   click expert / type expert / scroll expert           │
│   各自生成精确动作参数                                   │
└──────────────────────────────────────────────────────┘
```

两层独立训练、独立生效：
- **快捷操作 SFT** 教模型在正确时机输出 `navigate_to_dialog(...)` tool call
- **MoE RL** 教模型在每一步用最合适的 expert 生成动作参数

**后续可探索（Task 8-12 范畴）：** 将 `macro_trigger` 作为 MoE 新 expert——不替代 tool-use，而是让 expert LoRA 更准确地生成快捷操作 tool call token：

```
Expert 1-4: click/type/navigate/scroll specialist (现有)
Expert 5:   macro_trigger expert (新增 — 专门优化生成快捷操作 tool call)
```

**结论：** Tool-Use 快捷操作和 MoE Expert 是正交的两个维度，应该**层叠组合**而非互相替代。

---

### 任务6：基于 f_net 伪奖励的 RL 训练
- **对应章节：** 5.6（方法B — 基于RL的伪奖励）
- **状态：** `训练中` 🔄 — 3 组对比实验已成功启动（Job 2523626/2523627/2523628）
- **前置依赖：** 任务3（f_net 训练完成 ✅）
- **阻塞：** 无（任务5的替代方案，适用于标注数据不足时）
- **优先级：** 高 — Task 5 SFT 仅产出 359 条跨越样本，数据太少且多样性不足（50% 只有 2 步跨越，55% 目标是泛化的 "settings"），转向 RL 方法

**目标：** 利用 f_net 的谱分析信号作为奖励塑形（reward shaping），不需要大量标注数据。用 `f_pseudo(t) = f(s_t) - f(s_{t+1})` 作为每步额外奖励信号。当 agent 朝瓶颈方向行动（从高 f 区域向低 f 区域移动）时获得正奖励，自然引导模型学会穿越瓶颈。

**奖励公式：**
```
r_total(t) = r_action_match(t) + λ * f_pseudo(t)

其中:
  r_action_match = format_score * 0.1 + action_score * 0.9   (现有，0-1)
  f_pseudo = f(s_t) - f(s_{t+1})                              (新增，约 -3 到 +3)
  λ = 0.1                                                      (可调超参数)

效果：
  - 多数步骤：f_pseudo ≈ 0（同区域内移动），不干扰正常训练
  - 瓶颈跨越步：f_pseudo >> 0（跨区域），强化跨越动作的学习
  - 错误方向步：f_pseudo < 0，抑制远离目标的动作
```

**Reward 计算详细流程：**

```
┌─────────────────────────────────────────────────────────────────┐
│ FPseudoDAPORewardManager.__call__(data)                        │
│   verl/workers/reward_manager/f_pseudo_dapo.py:111-227         │
│                                                                 │
│ 1. 解码 prompt + response → 文本                                │
│ 2. 获取 ground_truth (check_options + num_steps)               │
│ 3. compute_score(data_source, response, ground_truth)          │
│         ↓                                                       │
│ ┌─────────────────────────────────────────────────────┐        │
│ │ gui_traj.gui_action_match_compute_score()           │        │
│ │   verl/utils/reward_score/gui_traj.py:16-61         │        │
│ │                                                     │        │
│ │ Step A: parse_response(solution_str)                │        │
│ │   → 提取 <think>...</think> 和 <action>...</action> │        │
│ │   → JSON 解析 action_content                        │        │
│ │                                                     │        │
│ │ Step B: format_score                                │        │
│ │   → 1.0 if think 非空                               │        │
│ │   → 0.0 if think 为空或缺失                         │        │
│ │                                                     │        │
│ │ Step C: check_response_match(pred, gt, w, h, ...)   │        │
│ │   verl/utils/reward_score/gui_utils/utils.py:101    │        │
│ │   → (type_match, extract_match)                     │        │
│ │                                                     │        │
│ │ Step D: action_score                                │        │
│ │   → 1.0 if extract_match (类型+参数全匹配)          │        │
│ │   → 0.5 if type_match only (类型匹配参数不匹配)     │        │
│ │   → 0.1 if 可解析但类型不匹配                       │        │
│ │   → 0.0 if 无法解析                                 │        │
│ │                                                     │        │
│ │ return: score = format_score*0.1 + action_score*0.9 │        │
│ └─────────────────────────────────────────────────────┘        │
│                                                                 │
│ 4. f_pseudo lookup (execution_id, step_id)                     │
│    → f_pseudo_map[execution_id][step_id] (预计算 JSON)         │
│    → 未命中返回 0.0                                            │
│                                                                 │
│ 5. total_reward = score + λ * f_pseudo                         │
│    → 放置在 response 最后一个 token 的位置                      │
└─────────────────────────────────────────────────────────────────┘
```

**常见分数值对照表：**

| score | format_score | action_score | 场景 |
|-------|-------------|-------------|------|
| **0.0** | 0.0 | 0.0 | 解析失败，无 think 无 action |
| **0.1** | 1.0 | 0.0 | 有 think，但无 action |
| **0.19** | 1.0 | 0.1 | 有 think，动作类型不匹配 |
| **0.55** | 1.0 | 0.5 | 有 think，类型匹配但参数不匹配 |
| **1.0** | 1.0 | 1.0 | 完美匹配（类型+参数全对） |

**参数匹配规则（`check_response_match`）：**

| 动作类型 | 匹配条件 |
|---------|---------|
| click/long_press | 坐标在 candidate_bbox 的 1.2x 放大框内，或与 GT 点 L2 距离 ≤ 0.04 |
| type/answer/key | 预测文本与 GT 文本互为子串（大小写不敏感） |
| swipe | 预测方向与 GT 方向一致（上下左右） |
| system_button | 按钮名称完全匹配（Back/Home/Menu/Enter，大小写不敏感） |
| wait/terminate | 动作类型匹配即可（无参数） |

**相关代码文件：**

| 文件 | 关键函数 | 说明 |
|------|---------|------|
| `verl/workers/reward_manager/f_pseudo_dapo.py` | `__call__()` L111-227 | 总 reward 计算入口 |
| `verl/utils/reward_score/gui_traj.py` | `gui_action_match_compute_score()` L16-61 | base score 计算 |
| `verl/utils/reward_score/gui_utils/utils.py` | `check_response_match()` L101-165 | 参数级匹配 |
| `verl/utils/reward_score/gui_utils/utils.py` | `check_click()` L57-65 | 坐标匹配（bbox + L2） |
| `verl/utils/reward_score/gui_utils/utils.py` | `check_text()` L30-35 | 文本匹配（子串） |
| `x/data/agent/json.py` | `parse_response()` L112-117 | 模型输出解析 |

**已实现的模块和文件：**

| 文件 | 类型 | 说明 |
|------|------|------|
| `scripts/precompute_f_pseudo.py` | **新建** | f_pseudo 离线预计算脚本 |
| `verl/workers/reward_manager/f_pseudo_dapo.py` | **新建** | FPseudoDAPORewardManager（继承 DAPORewardManager） |
| `verl/workers/reward_manager/__init__.py` | 修改 | 注册新 reward manager（+2行） |
| `verl/utils/dataset/universal_multiround.py` | 修改 | `fetch_batch()` 传递 `execution_id` 到 reward manager（+1行） |
| `train_GUI_360/config/gui360_rl_f_pseudo.yaml` | **新建** | 训练配置（UIS1 advantage + f_pseudo shaping） |

**未修改的现有文件（直接复用）：**
- `uis1/core_uis1.py` — UIS1 advantage estimator 直接复用
- `verl/utils/reward_score/gui_traj.py` — base reward 函数不修改
- `verl/trainer/ppo/dapo_ray_trainer.py` — 训练循环不修改
- `verl/trainer/ppo/reward.py` — `load_reward_manager()` 通过 `reward_kwargs` 传参，不修改

**集成架构：**
```
┌─────────────────────────────────────────────────┐
│ 离线预计算 (一次性，CPU，~3秒)                      │
│                                                   │
│ 输入:                                              │
│   outputs/fnet/gui360/{app}/f_values.npz          │
│     → 6,511 个状态的 hash→f_value 映射             │
│   outputs/transitions/gui360_full/transitions.jsonl│
│     → 91,618 条 (state_hash, next_state_hash) 转移 │
│                                                   │
│ 处理:                                              │
│   1. 合并 3 个 app 的 f_value_map (6,511 条)       │
│   2. 遍历 transitions.jsonl 每行:                  │
│      src_f = f_value_map[state_hash]               │
│      dst_f = f_value_map[next_state_hash]          │
│      f_pseudo = src_f - dst_f                      │
│   3. 按 execution_id 分组输出                      │
│                                                   │
│ 输出: outputs/f_pseudo/f_pseudo_map.json (1.4MB)  │
│   {execution_id: {step_id: f_pseudo_value, ...}}  │
│ 输出: outputs/f_pseudo/statistics.json             │
│                                                   │
│ 备选路径（transitions.jsonl 不可用时）：             │
│   直接处理 datasets/GUI-360/train/data/ 原始轨迹    │
│   → extract_state_id() 提取哈希                    │
│   → 对未知状态用 f_net 推理 43 维嵌入              │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│ RL 训练 (GPU)                                     │
│                                                   │
│ 数据流:                                            │
│   TrajDataset.__getitem__                         │
│     → line dict（含 execution_id）                 │
│   MultiRoundGenerator.fetch_batch()               │
│     → row_dict['execution_id'] = line.execution_id│
│     → row_dict['step_id'] = state['step_id']      │
│     → collate_fn → DataProto.non_tensor_batch     │
│                                                   │
│ FPseudoDAPORewardManager.__call__(data):          │
│   for i in range(len(data)):                      │
│     1. base_reward = compute_score(response, gt)  │
│        → format_score * 0.1 + action_score * 0.9  │
│     2. exec_id = non_tensor_batch['execution_id'] │
│        step_id = non_tensor_batch['step_id']      │
│        f_pseudo = f_pseudo_map[exec_id][step_id]  │
│     3. reward = base_reward + λ * f_pseudo        │
│     4. reward_extra_info 记录 f_pseudo/base_reward│
│                                                   │
│ UIS1 Advantage Estimator (已有，不需修改)           │
│   step_discounted_returns(rewards + f_bonus)      │
│   episode_advantage + step_advantage              │
└─────────────────────────────────────────────────┘
```

**核心 API：**

`scripts/precompute_f_pseudo.py`:
- `load_f_value_maps(fnet_dir)` → 合并 3 个 app 的 `{hash: f_value}` 映射
- `load_f_nets(fnet_dir, device)` → 加载训练好的 `EigenfunctionNet`（用于未知状态推理）
- `process_transitions(path, f_value_map)` → 从 `transitions.jsonl` 计算 f_pseudo
- `process_raw_trajectories(data_dir, f_value_map, f_nets)` → 从原始轨迹计算（备选）
- `compute_statistics(f_pseudo_map)` → 分布统计

`verl/workers/reward_manager/f_pseudo_dapo.py`:
- `FPseudoDAPORewardManager(tokenizer, num_examine, ..., f_pseudo_path, f_pseudo_lambda)`
  - 继承 `DAPORewardManager`，通过 `reward_model.reward_kwargs` 接收额外参数
  - `_load_f_pseudo_map(path)` → 加载预计算 JSON
  - `_lookup_f_pseudo(execution_id, step_id)` → 查找，未命中返回 0.0
  - `__call__(data, return_dict)` → 完整 reward 计算（base + f_pseudo shaping）
  - `reward_extra_info` 记录 `f_pseudo`, `f_pseudo_bonus`, `base_reward`（供 wandb 监控）
  - 定期打印命中率统计（每 1000 次 lookup）

`verl/utils/dataset/universal_multiround.py` 的修改（1行）:
```python
# MultiRoundGenerator.fetch_batch() 中新增:
row_dict['execution_id'] = self.task_queue[ptr].line.get('execution_id', '')
```

**可复用的已有代码：**

| 来源 | 复用内容 |
|------|---------|
| `verl/models/moe/graph_analysis.py` | `load_f_net()`, `EigenfunctionNet`, `EigenfunctionConfig` |
| `verl/models/moe/state_representation.py` | `GUI360StepData.from_raw()`, `extract_state_id()`, `extract_state_embedding()` |
| `outputs/fnet/gui360/{app}/f_values.npz` | 已训练好的 f-values（Excel 1,414 / Word 3,518 / PPT 1,579 状态） |
| `outputs/fnet/gui360/{app}/f_net_final.pt` | 已训练好的 f_net MLP（用于未知状态推理） |
| `outputs/transitions/gui360_full/transitions.jsonl` | 91,618 条状态转移记录 |
| `verl/workers/reward_manager/dapo.py` | `DAPORewardManager` 基类 |
| `verl/workers/reward_manager/registry.py` | `@register` 装饰器、`get_reward_manager_cls()` |
| `verl/trainer/ppo/reward.py` | `load_reward_manager()` 自动传递 `reward_kwargs` |
| `uis1/core_uis1.py` | UIS1 advantage estimator |

**预计算运行结果（已验证）：**

| 指标 | 值 |
|------|------|
| 覆盖率 | **100%**（91,618/91,618 transitions，所有状态哈希均在 f_values 中找到） |
| 轨迹数 | 13,237 |
| 总步数 | 91,618 |
| f_pseudo 均值 | 0.0002（接近零，符合预期） |
| f_pseudo 标准差 | 0.1504 |
| f_pseudo 范围 | [-3.348, +3.221] |
| 正向（瓶颈跨越）占比 | 16.2%（14,866 步，f_pseudo > 0.01） |
| 负向（远离瓶颈）占比 | 16.5%（15,153 步，f_pseudo < -0.01） |
| 近零（同区域）占比 | 67.2%（61,599 步，|f_pseudo| ≤ 0.01） |
| P5 / P25 / P50 / P75 / P95 | -0.071 / 0.0 / 0.0 / 0.0 / +0.069 |
| 输出文件大小 | f_pseudo_map.json: 1.4MB, statistics.json: 577B |

**分布分析：** 67% 步骤 f_pseudo≈0（agent 在同一 UI 区域内操作，不干扰正常训练），约 1/3 步骤有显著 f_pseudo 信号。极端值（±3）对应跨越主要 UI 模式边界（如从主界面进入深层对话框或反向），正是需要奖励塑形引导的关键步骤。

**训练配置关键设置（`gui360_rl_f_pseudo.yaml`）：**
```yaml
algorithm:
  adv_estimator: uis1           # 使用 UIS1 轨迹级 advantage
  gamma: 0.99                   # 折扣因子
  uis1:
    episode_advantage_w: 1.0    # 轨迹级 advantage 权重
    step_advantage_w: 1.0       # 步级 advantage 权重

reward_model:
  reward_manager: f_pseudo_dapo # 使用 f_pseudo reward manager
  reward_kwargs:
    f_pseudo_path: outputs/f_pseudo/f_pseudo_map.json
    f_pseudo_lambda: 0.1        # 关键超参：f_pseudo 在总奖励中的权重
```

**超参数：**

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|---------|
| `f_pseudo_lambda` | 0.1 | f_pseudo 在总奖励中的权重 | 关键超参。0=基线（无shaping），0.1=保守，0.3=激进 |
| `gamma` | 0.99 | 折扣因子 | UIS1 step reward 的折扣 |
| `step_advantage_w` | 1.0 | UIS1 step advantage 权重 | 与 episode_advantage_w 平衡 |
| `episode_advantage_w` | 1.0 | UIS1 episode advantage 权重 | 与 step_advantage_w 平衡 |
| `lr` | 1e-5 | 学习率 | 保守设置，防止 f_pseudo shaping 导致不稳定 |
| `clip_ratio` | 0.1 | PPO clip ratio | 保守设置 |
| `kl_loss_coef` | 0.1 | KL 散度损失系数 | 防止策略偏离过远 |

**CLI 命令：**
```bash
# Step 1: 预计算 f_pseudo（一次性，~3秒，CPU）
conda run -n qwen3-eval python scripts/precompute_f_pseudo.py

# 可选参数：
#   --fnet_dir outputs/fnet/gui360          # f_net 输出目录
#   --transitions_path outputs/transitions/gui360_full/transitions.jsonl
#   --data_dir datasets/GUI-360/train/data  # 原始轨迹（备选路径）
#   --output_dir outputs/f_pseudo           # 输出目录
#   --device cpu                            # 推理设备

# Step 2: RL 训练
python -m verl.trainer.main_dapo \
  --config-path train_GUI_360/config \
  --config-name gui360_rl_f_pseudo
```

**验证步骤：**

1. **预计算验证（✅ 已通过）：**
   - 覆盖率 100%（> 80% 阈值）
   - f_pseudo 分布合理：均值≈0，67% 近零，16% 正向/负向
   - 输出文件可正常加载，lookup 命中/未命中正确

2. **Reward manager 回归测试：**
   - f_pseudo=0（execution_id 不匹配）时行为与 DAPORewardManager 一致
   - `get_reward_manager_cls('dapo')` 仍正常工作

3. **训练监控（wandb）：**
   - `reward_extra_info` 包含 `f_pseudo`, `f_pseudo_bonus`, `base_reward`
   - 打印日志包含 f_pseudo lookup 命中率

4. **对比实验计划：**
   - lambda=0（纯 action_match 基线）
   - lambda=0.1（保守 shaping）
   - lambda=0.3（激进 shaping）
   - 观察指标：action_match accuracy、瓶颈跨越步的 reward 分布、训练稳定性

**RL 训练执行记录（2025-03-02）：**

**SLURM 脚本：** `train/train_rl_f_pseudo.slurm`（新建）
- 基于 `train/train_ui_s1_single.slurm` 模板，单节点 4-GPU Ray 集群
- 支持 `--export=LAMBDA=x` 参数化，3 组对比实验同时提交
- 关键训练参数：batch_size=8, n_rollouts=8, lr=1e-6, gamma=0.5, 3 epochs = 375 training steps
- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`

**提交命令：**
```bash
sbatch --export=LAMBDA=0.0 train/train_rl_f_pseudo.slurm  # Job 2523397 (基线)
sbatch --export=LAMBDA=0.1 train/train_rl_f_pseudo.slurm  # Job 2523398 (保守)
sbatch --export=LAMBDA=0.3 train/train_rl_f_pseudo.slurm  # Job 2523399 (激进)
```

**环境兼容性问题与修复（共 6 轮迭代）：**

| 轮次 | 环境 | 错误 | 原因 | 修复 |
|------|------|------|------|------|
| 1 | qwen3-eval | `PermissionError: '/local/user/...'` | Ray 无法在计算节点创建临时目录 | SLURM 脚本添加 `RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}` |
| 2 | qwen3-eval | `ModuleNotFoundError: 'vllm.lora.models'` | vLLM 0.15.1 重命名模块 | `verl/utils/vllm_utils.py` 添加 try/except fallback imports |
| 3 | qwen3-eval | `RuntimeError: split_group not supported` | PyTorch 2.9.1 DeviceMesh API 变更 | `verl/workers/fsdp_workers.py` `_build_rollout` 添加 fallback 逻辑 |
| 4 | qwen3-eval | `ModuleNotFoundError: 'vllm.worker'` | vLLM 0.15.1 完全重构 worker 模块 | **放弃 qwen3-eval**，切换回 ui-s1 环境 |
| 5 | ui-s1 | `KeyError: 'mrope'` | transformers 缺少 Qwen2.5-VL 的 mrope RoPE 类型 | 手动 patch `modeling_rope_utils.py` 添加 mrope 条目 |
| 6 | ui-s1 | `ImportError: Numba needs NumPy 2.2 or less` | force-reinstall transformers 升级了 numpy 到 2.4 | `pip install "numpy<2.3"` 降级到 2.2.6 |

**代码修改清单：**

| 文件 | 修改内容 |
|------|---------|
| `train/train_rl_f_pseudo.slurm` | **新建** — SLURM 脚本，支持 `--export=LAMBDA=x` 参数化 |
| `verl/utils/vllm_utils.py` | 添加 vLLM 0.15.1 兼容 try/except imports（lora.models → lora.lora_model 等） |
| `verl/workers/fsdp_workers.py` | `_build_rollout` 添加 PyTorch 2.9 DeviceMesh split_group fallback（移除 bound_device_id 再 init_device_mesh） |
| `site-packages/transformers/modeling_rope_utils.py` | 手动添加 `"mrope": _compute_default_rope_parameters` 到 `ROPE_INIT_FUNCTIONS` |

**最终运行环境（ui-s1）：**

| 包 | 版本 | 备注 |
|----|------|------|
| PyTorch | 2.6.0 | CUDA 12.6 |
| vLLM | 0.8.5 | 与 verl 兼容 |
| transformers | 4.57.6 | 需手动 patch mrope |
| numpy | 2.2.6 | numba 兼容 |
| ray | 2.44.1 | |
| verl | 0.2.0.dev | 本地开发版 |

**训练结果：3 个 job 均成功初始化但在首个 rollout 步崩溃**

成功阶段：
- ✅ Ray 集群初始化（1 head + 4 GPU workers）
- ✅ 模型加载 + FSDP 封装
- ✅ vLLM rollout engine 创建
- ✅ wandb 连接
- ✅ 训练循环启动（显示 `Training Progress: 0%| | 0/375`）

失败点：
```
AttributeError: 'NoneType' object has no attribute 'keys'
  File "verl/protocol.py", line 537, in pop
  File "verl/trainer/ppo/dapo_ray_trainer.py", line 976, in fit
```

**根因分析：** rollout 生成返回了 None batch。`trainer.fit()` → `batch.pop()` 对 None 调用 `.keys()`。可能原因：
1. `VLLM_USE_V1=1` 环境变量（在 `train/env_config.sh` 中设置）激活了 vLLM v1 engine，与 vLLM 0.8.5 的 SPMD rollout 不完全兼容
2. 数据格式问题 — rollout 生成的 prompt 格式不匹配模型输入预期
3. vLLM rollout worker 内部异常被静默吞掉，返回 None 而非抛出错误

**后续修复（第 7-15 轮迭代）：**

| 轮次 | 错误 | 原因 | 修复 |
|------|------|------|------|
| 7 | `AttributeError: 'NoneType' has no attribute 'keys'` (protocol.py:537) | `main_dapo.py` 通过 project_name 选择 trainer，`gui360_rl_f_pseudo` 不匹配 `gui_traj_grpo*`，使用了 `RayPPOTrainer`（期望预分词数据）而非 `RayTrajDAPOTrainer`（处理原始轨迹） | `main_dapo.py` 添加 `trainer_cls` config 选项 + TrajDataset 自动检测 |
| 8 | `ConfigAttributeError: Key 'trainer_cls' is not in struct` | Hydra 要求新增 config key 使用 `+` 前缀 | SLURM 脚本改为 `+trainer.trainer_cls=traj_dapo` |
| 9 | `RuntimeError: nccl does not support allgather_into_tensor_coalesced` | DTensor.full_tensor() 在 FSDP→vLLM 权重同步时使用了 GH200 NCCL 不支持的 coalesced allgather | `fsdp_vllm.py` 添加 `_dtensor_to_full` 回退函数 |
| 10 | `AttributeError: 'NoneType' has no attribute 'enable'` | `FPseudoDAPORewardManager.__call__` 未检查 `overlong_buffer_cfg is None` | `f_pseudo_dapo.py` 添加 None guard |
| 11 | `OutOfMemoryError: node running low on memory` | 单节点 batch_size=8, n_rollouts=8, gpu_mem=0.9 → 4 workers × 60GB RAM | 减小 batch_size=4, n_rollouts=4, gpu_mem=0.5, free_cache=True |
| 12 | OOM（~28min，9 步后） | prompt_length/max 飙升至 14,759（远超通常 ~10K）触发内存峰值 | 转为多节点训练（2 nodes × 4 GPU = 8 GPU） |
| 13 | `execve(): hostname/bash: No such file or directory` | srun 不继承 PATH，找不到 `hostname`/`bash` | 使用绝对路径 `/usr/bin/hostname`、`/bin/bash` |
| 14 | `ValueError: Malformed host:` (RAY_ADDRESS 为空) | 内层 bash -c 中 `$head_node_ip` 被转义，内层 shell 找不到变量 | 改为外层 shell 展开（不转义） |
| 15 | `RuntimeError: Detected mismatch between collectives on ranks` (shape [428,1280] vs [424,1280]) | **根因深层分析：** try/except `full_tensor()` 失败后 NCCL sequence counter 被破坏，后续 `dist.all_gather` 在不同 rank 上实际 gather 了不同参数（428 vs 424 是两个不同层的权重，差 4 而非 uneven sharding 的差 1）→ 静默数据错误 | **彻底重构：** 启动时检测 GH200 GPU，一旦检测到则**完全跳过** `full_tensor()`，始终走 manual `dist.all_gather` 路径（pad-gather-strip 处理 uneven sharding） |

**额外代码修改清单：**

| 文件 | 修改内容 |
|------|---------|
| `verl/trainer/main_dapo.py` | 添加 `trainer.trainer_cls` config 选项 + TrajDataset 自动检测逻辑 |
| `verl/workers/sharding_manager/fsdp_vllm.py` | GH200 检测 + manual `dist.all_gather` 替代 `full_tensor()`（pad-gather-strip 处理 uneven sharding） |
| `verl/workers/reward_manager/f_pseudo_dapo.py` | `overlong_buffer_cfg` 添加 None guard |
| `train/train_rl_f_pseudo_multinode.slurm` | **新建** — 2 nodes × 4 GPU Ray 集群多节点训练脚本 |

**关键指标说明：**

| 指标名 | 含义 | 计算方式 |
|--------|------|---------|
| `critic/score/mean` | Episode 级综合 reward 均值 | `format_score * 0.1 + action_score * 0.9`（+ lambda * f_pseudo） |
| `critic/step_success_rate` | Step 级 action 正确率 | `1 - error_step_num / total_step_num`。衡量模型生成的 action 与 ground truth 是否匹配 |
| `actor/entropy` | 策略熵 | 衡量探索性。越高表示策略越分散 |
| `actor/pg_loss` | 策略梯度损失 | PPO 的 clipped surrogate objective |
| `actor/grad_norm` | 梯度范数 | 训练稳定性指标 |
| `actor/kl_loss` | KL 散度损失 | 策略偏离参考模型的程度 |
| `actor/pg_clipfrac` | PPO clip 比例 | 被 clip 的样本占比 |
| `timing_s/step` | 每步耗时（秒） | 含 rollout + log_prob + ref + advantage + actor_update |
| `timing_s/reshard` | 权重同步耗时（秒） | FSDP→vLLM weight sync（allgather 路径） |
| `perf/max_memory_allocated_gb` | GPU 峰值显存 | |
| `perf/cpu_memory_used_gb` | CPU 内存使用 | |
| `prompt_length/max` | 最大 prompt 长度 | 过长会导致 OOM |

`step_success_rate` 的 patch 机制：multi-turn rollout 中，如果某步模型预测错误（`extract_match=False`），系统会用 ground truth action 替换模型的错误回复（patch），让 trajectory 能继续执行。当某条 trajectory 的 patch 次数超过 `patch_threshold` 时，直接标记为结束。因此 `step_success_rate=30%` 表示约 70% 的步骤需要被 patch。

**数据处理流水线（GUI-360 → TrajDataset 格式）：**

GUI-360 原始数据为 per-step JSONL 格式，RL 训练需要 per-episode 格式。转换脚本：`scripts/GUI_360/prepare_gui360_rl_data.py`

```
原始格式（per-step，每个 jsonl 文件一个 episode 的所有步骤）：
  {split}/data/{app}/{category}/{success|fail}/{exec_id}.jsonl
  每行: {"execution_id", "request", "step_id", "step": {"screenshot_clean", "action", "thought"}}

图像路径（与 data 目录分离）：
  {split}/image/{app}/{category}/{success|fail}/{exec_id}/action_step{N}.png

TrajDataset 格式（per-episode，一行一个完整 trajectory）：
  {"goal": "...", "is_successful": true, "execution_id": "...",
   "steps": [{"action_content": {...}, "screenshot": "/abs/path/to/img.png", "thought": "..."}]}
```

**数据统计（`datasets/GUI-360/rl_data/`）：**

| 文件 | 数据量 | 来源 |
|------|--------|------|
| `gui360_train.jsonl` | 13,750 episodes | train/excel(4,348) + word(5,633) + ppt(3,769) |
| `gui360_test.jsonl` | 3,439 episodes | test/excel(1,087) + word(1,409) + ppt(943) |
| `gui360_val_small.jsonl` | 50 episodes | test 随机抽样（seed=42） |

**数据处理相关文件：**

| 文件 | 说明 |
|------|------|
| `scripts/GUI_360/prepare_gui360_rl_data.py` | 转换脚本（per-app 处理避免 OOM） |
| `scripts/GUI_360/prepare_gui360_rl_data.slurm` | SLURM 提交脚本（64G 内存，30min） |
| `scripts/GUI_360/prepare_gui360_rl_data.log` | 运行日志 |

**数据处理修复历史：**

| 问题 | 原因 | 修复 |
|------|------|------|
| 登录节点 OOM | `glob.glob` 加载 13K+ 文件内存不足 | 改用 SLURM 提交（64G 内存） |
| `PIL.UnidentifiedImageError` | 截图路径错误：使用 `{split}/data/` 而非 `{split}/image/`；硬编码 `in_app` 而非动态获取 category | 修复 `image_base` 为 `{split}/image/`；从 jsonl 目录路径提取 category |
| 缺少 `online` 类别 | `CATEGORIES` 列表为 `[in_app, search, cross_app]`，实际为 `[in_app, search, online]` | 更正 CATEGORIES |

**多节点训练配置：**

| 配置项 | 值 |
|--------|------|
| 节点数 | 2 nodes × 4 GPU = 8 GPU |
| SLURM 脚本 | `train/train_rl_f_pseudo_multinode.slurm` |
| 集群架构 | Ray head (node 0) + Ray worker (node 1) |
| 训练数据 | `datasets/GUI-360/rl_data/gui360_train.jsonl` (13,247 episodes) |
| 验证数据 | `datasets/GUI-360/rl_data/gui360_val_small.jsonl` (50 episodes) |
| batch_size | 8 |
| n_rollouts | 4 |
| lr | 1e-6 |
| gamma | 0.5 |
| gpu_memory_utilization | 0.5 |
| max_model_len | 16384 |
| max_prompt_length | 8192 |
| max_response_length | 512 |
| free_cache_engine | True |
| enforce_eager | True |
| kl_loss_coef | 0.0001 |
| kl_loss_type | low_var_kl |
| FSDP device_mesh | `[0,1,2,3,4,5,6,7]` (全 8 GPU) |
| trainer_cls | traj_dapo (RayTrajDAPOTrainer) |
| adv_estimator | uis1 |
| test_freq | 10 |
| save_freq | 5 |

**测试 Job 2524048（lambda=0.1，多节点验证，AndroidControl 数据）训练指标：**

| Step | score/mean | step_success_rate | entropy | pg_loss | grad_norm | kl_loss | prompt_len/max | mem_alloc_GB | mem_reserve_GB | cpu_mem_GB | time/step (s) |
|------|-----------|-------------------|---------|---------|-----------|---------|---------------|-------------|---------------|-----------|--------------|
| 1 | 0.436 | 30.7% | 0.598 | -0.003 | 27.1 | 0.009 | 9,090 | 64.0 | 84.2 | 527 | 355 |
| 2 | 0.427 | 28.9% | 0.679 | 0.101 | 32.8 | 0.003 | 8,976 | 64.1 | 84.2 | 599 | 336 |
| 3 | 0.527 | 33.2% | 0.715 | -0.396 | 23.3 | 0.002 | 10,600 | 64.1 | 87.6 | 594 | 432 |
| ... | 9 步成功后在 step 10（validation）OOM |||||||||

**关键观察：**
- **训练 9 步正常** — GPU 显存稳定在 64/87 GB，CPU 内存 527-599 GB
- **权重同步正常** — `timing_s/reshard ≈ 14s`（manual allgather 路径工作正常）
- **Step 10 Validation OOM** — `test_freq=10` 触发验证，1,543 sample 的 `android_control_evaluation_std.jsonl` 导致内存溢出
- **score/mean 0.43→0.53** — 第 3 步出现上升趋势
- **step_success_rate 29-33%** — 约 70% 步骤需要 patch
- **~350s/step** — 合理，3 epochs ≈ 375 steps ≈ 36h

**各阶段耗时分布（Step 2 为例）：**

| 阶段 | 耗时 (s) | 占比 |
|------|---------|------|
| gen (vLLM rollout) | 103.9 | 30.9% |
| update_actor (FSDP 梯度) | 148.6 | 44.2% |
| old_log_prob | 42.1 | 12.5% |
| ref (reference model) | 41.3 | 12.3% |
| reshard (FSDP→vLLM) | 14.1 | 4.2% |
| **总计** | **336.2** | **100%** |

**非致命错误（已 graceful 处理）：**
- `TypeError: object of type 'NoneType' has no len()` — `check_click()` 中 `candidate_bbox` 为 None，已被 try/except 捕获
- `JSONDecodeError` — 模型偶尔生成格式错误的 JSON action（如多余 `}`），被标记为 error response

**问题修复总结：**
1. Job 2524048 (AndroidControl 数据) — 训练 9 步后 validation OOM → 改用 50-sample 小验证集
2. 训练/验证数据应使用 GUI-360 而非 AndroidControl → 创建 GUI-360 数据转换流水线
3. GUI-360 截图路径错误 → 修复 image_base + category 提取逻辑

**正式训练 Job 2524588（lambda=0.1，GUI-360 数据）训练指标：**

| Step | score/mean | step_success_rate | entropy | pg_loss | grad_norm | prompt_len/max | mem_alloc_GB | cpu_mem_GB | time/step (s) |
|------|-----------|-------------------|---------|---------|-----------|---------------|-------------|-----------|--------------|
| 1 | 0.329 | 1.0% | 0.805 | 0.037 | 32.4 | 3,280 | 62.1 | 441 | 216 |
| 2 | 0.432 | 10.0% | 0.814 | -0.030 | 33.0 | 3,810 | 62.1 | 476 | 247 |
| 3 | 0.376 | 9.7% | 0.866 | 0.043 | 35.5 | 3,916 | 62.1 | 465 | 232 |

**与 AndroidControl 数据对比：**

| 指标 | AndroidControl (Job 2524048) | GUI-360 (Job 2524588) | 说明 |
|------|---------------------------|---------------------|------|
| step_success_rate | 29-33% | 1-10% | GUI-360 任务更复杂（更多步骤、更多动作类型） |
| score/mean | 0.43-0.53 | 0.33-0.43 | 与 step_success_rate 一致 |
| prompt_length/max | 9,090-10,600 | 3,280-3,916 | GUI-360 prompt 更短（无 a11y tree？） |
| time/step | 336-432s | 216-247s | 更短的 prompt 导致更快的 rollout |
| score/max | 1.0 | 1.069 | GUI-360 有 f_pseudo bonus（AndroidControl 无 f_pseudo map） |
| score/min | 0.0 | -0.072 | GUI-360 有 f_pseudo penalty |

**关键观察：**
- **f_pseudo 生效** — score/max=1.069 > 1.0, score/min=-0.072 < 0，说明 f_pseudo bonus/penalty 正在工作
- **GUI-360 step_success_rate 很低（1-10%）** — 与 AndroidControl 的 30% 差距大，可能因为：
  1. GUI-360 包含更多特殊动作类型（如 `insert_excel_table`）模型不认识
  2. 坐标精度要求更高（Office UI 控件更小）
  3. 早期 step 的 success_rate 可能需要更多训练步数才能提升
- **无 OOM** — prompt_length/max 仅 3,916（远低于 AndroidControl 的 10K+），内存稳定
- **~230s/step** — 比 AndroidControl 快 40%，3 epochs ≈ 5,154 steps ≈ 13.8 天

**额外修复 — coordinate [None, None] 崩溃（Job 2524588 step 14）：**

| 问题 | 原因 | 修复 |
|------|------|------|
| `TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'` | GUI-360 中 171/105,368 步 (0.2%) 的 ground truth action 含 `coordinate: [None, None]`。`deal_with_coordinate()` 检查了列表是否为 None，但未检查列表内元素 | `x/data/agent/base.py:12-17` 添加 `all(v is not None for v in coordinate)` 检查 |

**多节点扩展实验 — 性能对比（Step 1 指标）：**

| 指标 | 2N/8GPU/bs=8 | 8N/32GPU/bs=32 | 8N/32GPU/bs=64 |
|------|-------------|----------------|----------------|
| **time/step (s)** | **216** | 524 | 853 |
| gen (rollout) | 64 | 201 | 303 |
| update_actor (FSDP) | 107 | 213 | 344 |
| old_log_prob | 23 | 58 | 108 |
| ref | 22 | 51 | 96 |
| reshard (allgather) | 11 | 31 | 31 |
| tokens/step | 232K | 1,020K | 1,945K |
| **throughput (tok/s)** | 1,075 | 1,948 | **2,279** |
| mem_alloc_GB (per GPU) | 62.1 | 50.1 | 50.1 |
| cpu_mem_GB (per node) | 441 | 309 | 317 |
| score/mean | 0.329 | 0.367 | 0.353 |
| step_success_rate | 1.0% | 9.0% | 8.0% |
| total steps (3 epochs) | 5,154 | 1,287 | 642 |
| **est. total time** | 344h | 187h | **152h** |
| steps in 24h | ~400 | ~165 | ~101 |

**分析：**
- 8 节点使用 socket transport (`NCCL_NET=Socket` over Slingshot `hsn0`)，无 InfiniBand
- **FSDP 通信是主要瓶颈** — `update_actor` 从 2 节点的 107s 增至 8 节点的 213-344s，因 32 rank 的 all-reduce 经过网络而非 NVLink
- **reshard (manual allgather)** 从 11s → 31s，增幅可控
- **batch=64 throughput 最高** (2,279 tok/s)，比 2 节点提升 2.1x，但距线性加速 (4x) 差距大
- **无 OOM** — 32-way FSDP 分片使每 GPU 显存降至 50GB（vs 8-way 的 62GB）
- 所有配置均无法在 24h 内完成 3 epochs；需要 checkpoint resume 或更长 SLURM 时限

**Job 记录：**

| Job ID | 配置 | 结果 |
|--------|------|------|
| 2524588 | 2N/8GPU, bs=8, GUI-360 | 13 步后崩溃（coordinate [None,None]） |
| 2526439 | 8N/32GPU, bs=32 | Step 1 完成后取消（性能测试） |
| 2526910 | 8N/32GPU, bs=64 | 运行中 🔄 |

**当前状态：** `训练中` 🔄 — Job 2526910（8 nodes, batch=64, lambda=0.1）运行中。
```bash
# 对比实验（待 batch size 确定后提交）：
sbatch --export=LAMBDA=0.0,BATCH_SIZE=64 train/train_rl_f_pseudo_multinode.slurm  # 基线
sbatch --export=LAMBDA=0.1,BATCH_SIZE=64 train/train_rl_f_pseudo_multinode.slurm  # 保守
sbatch --export=LAMBDA=0.3,BATCH_SIZE=64 train/train_rl_f_pseudo_multinode.slurm  # 激进
```

---

## 第四阶段：层次化策略与 MVP 实验

### 任务7：构建 Tool-Use 快捷操作执行器 + 层次化推理流水线
- **对应章节：** 5.7（Step 4）
- **状态：** `待开始`
- **前置依赖：** 任务5（数据准备 ✅ + 训练）
- **阻塞：** 任务10

**目标：** 实现快捷操作 tool call 的推理时执行器，将 tool-use 快捷操作集成到端到端推理流水线。

**Task 5 已为本任务提供的基础设施：**
- `outputs/macro_sft/macro_playbook.json`：359 条快捷操作的典型动作序列
- `outputs/macro_sft/macro_tool_definitions.json`：3 个快捷操作 tool 的 JSON Schema
- 模型已通过 SFT 学会在正确时机输出 `<tool_call>` 快捷操作调用

**增强的动作空间（tool-use 统一格式）：**
```
原始动作 tool call:
<tool_call>
{"function": "click", "args": {"element_id": 18}, "status": "CONTINUE"}
</tool_call>

快捷操作 tool call:
<tool_call>
{"function": "navigate_to_dialog", "args": {"target_dialog": "Excel Options"}, "status": "CONTINUE"}
</tool_call>
```

**快捷操作执行器（`scripts/macro_executor.py`，本任务实现）：**
```
1. 模型输出: navigate_to_dialog(target_dialog="Excel Options")
2. 执行器在 macro_playbook.json 中匹配 (app, macro_type, target)
   → 获取典型动作序列: [click(File), click(Options)]
3. 逐步执行动作（或由同一模型在 constrained mode 下执行）
4. 截取当前截图作为快捷操作执行结果
5. 反馈给主模型: "Macro executed. Here is the current screen:"
6. 主模型继续正常决策
```

**回退机制：** playbook 无匹配或某步失败 → 回退到原始模型逐步决策。

**预期有效视野缩短效果：**

| 基准 | 原始视野 | 加入快捷操作后 | 步数预算 |
|---|---|---|---|
| GUI-360°（复杂，>15步） | 15-20步 | ~6个高层决策 | 可变 |
| GUI-360°（中等，8-15步） | 8-15步 | ~5个高层决策 | 可变 |
| AndroidWorld（困难） | >20步 | ~8个高层决策 | 10-30步 |

**产出：**
1. `scripts/macro_executor.py` — 快捷操作 playbook 执行器
2. 推理流水线集成（检测 tool call → 执行快捷操作 → 返回结果截图）
3. 有效视野缩短指标追踪
4. 评估脚本扩展（支持快捷操作 tool call 的 action matching）

**可复用的已有基础设施：**
- Actor：`verl/workers/actor/dp_actor.py`
- 评估脚本：`evaluation/eval_*.py`
- 快捷操作数据：`outputs/macro_sft/macro_playbook.json`、`outputs/macro_sft/macro_tool_definitions.json`

---

### 任务10：在 GUI-360° 复杂任务子集上运行 MVP 实验
- **对应章节：** 8（最小可行实验）
- **状态：** `待开始`
- **前置依赖：** 任务7
- **阻塞：** 任务11、12

**目标：** 在 GUI-360° 上进行端到端 MVP 实验。纯数据驱动，无需 RL。

**实验设置：**
- 数据集：GUI-360° 训练集（120万步动作）
- 任务子集：GUI-360°-Bench 中 >15步 的复杂任务
- 基线：在 GUI-360° 动作预测上 SFT 微调的 Qwen2.5-VL-7B

**实验流水线：**
```
Step 1: 提取状态转移 → 构建图（任务1-2）
Step 2: 训练 f_net（单 GPU 约30分钟）→ 可视化 f 值（任务3）
Step 3: 识别 top-3 瓶颈 → 与已知困难转移对照验证（任务4）
Step 4: 提取跨越片段 → SFT 训练3个快捷操作（任务5）
Step 5: 在 GUI-360°-Bench 复杂子集上评估
        → 对比：基线 VLM  vs  基线 VLM + 3个快捷操作
        → 指标：动作预测成功率、任务完成率
```

**成功标准：**
1. **定性：** f 值识别的瓶颈与人类直觉关于困难 UI 转移的认知一致
2. **定量：** 快捷操作增强 agent 在复杂任务（>15步）上成功率提升 >10%
3. **效率：** 有效决策视野可测量地缩短（统计每个任务的高层决策次数）

**产出：** SLURM 脚本 + GUI-360° 完整 MVP 流水线结果

---

### 任务11：扩展实验到 AndroidWorld 困难子集
- **对应章节：** 8.4
- **状态：** `待开始`
- **前置依赖：** 任务10
- **阻塞：** 无（输入到任务12）

**目标：** 在 GUI-360° 验证成功后，将方法扩展到 AndroidWorld。

**步骤：**
1. 在 AndroidWorld 的116个任务上收集基线 agent 的 rollout 数据
2. 从 rollout 构建转移图
3. 在 AndroidWorld 转移数据上训练 f_net
4. 识别瓶颈（深层设置、跨应用、表单、列表选择）
5. 提取跨越片段 → SFT 快捷操作策略
6. 在困难子集上评估（>20步、跨应用工作流）

**产出：**
- AndroidWorld 的转移图
- 瓶颈分析报告
- 3-6个 Android 专用快捷操作
- 评估结果：基线 agent vs 快捷操作增强 agent（困难子集）
- 与已有 UI-S1 结果的对比

---

## 第五阶段：MoE 集成与扩展

### 任务8：实现连通性感知路由损失（MoE）
- **对应章节：** 3.3
- **状态：** `待开始`
- **前置依赖：** 任务3
- **阻塞：** 任务9

**目标：** 在 MoE 路由损失中增加连通性感知项，最大化 λ₂。

**增强损失：**
$$\mathcal{L}_{\text{aux}} = \alpha \cdot \text{LoadBalance} + \beta \cdot (-\lambda_2(\text{专家增强任务图}))$$

最小化 $-\lambda_2$ = 最大化连通性 = 最小化任何任务类型到最近胜任专家的最大距离。

**产出：**
1. `verl/models/moe/moe_loss.py` 中新增 `ConnectivityLoss`
   - 在任务嵌入空间构建 k-NN 图（从路由器特征）
   - 可微的 λ₂ 近似（幂迭代法或类似方法）
   - 返回 $-\lambda_2$ 作为损失
2. MoEConfig 中新增 `connectivity_weight`（β）
3. 集成到 `MoELoss`（与 LoadBalance 和 RouterZ 并列）
4. 将 λ₂ 作为训练指标记录
5. 更新配置 `traj_grpo_moe.yaml`

**已有代码：**
- `verl/models/moe/moe_loss.py` — LoadBalanceLoss, RouterZLoss（510行）
- `verl/models/moe/router.py` — TextOnlyRouter（626行）
- 现有配置：`balance_weight=0.2`、`z_loss_weight=0.01`

---

### 任务9：实现渐进式专家发现机制
- **对应章节：** 3.2
- **状态：** `待开始`
- **前置依赖：** 任务8
- **阻塞：** 任务12

**目标：** 迭代式专家特化，类比迭代选项发现。

**流水线：**
```
阶段1: 训练基础模型（无专家特化）
阶段2: 在任务嵌入图上计算 f → 识别覆盖瓶颈
阶段3: 添加新专家，目标 = 极端 f 值的任务聚类
阶段4: 使用增强 MoE 重新采样路由统计
阶段5: 重新计算 f → 发现新瓶颈 → 重复（2-3次迭代）
```

快捷操作库在2-3次迭代后稳定，通常每个应用领域产出3-6个快捷操作。

**产出：**
1. 迭代专家发现循环的脚本/工具
2. 动态专家添加：初始化新 LoRA 针对极端 f 值聚类
3. 路由器更新以容纳新专家
4. 迭代训练间的检查点管理
5. 收敛检测：当 λ₂ 边际增益 < 阈值时停止

**已有代码：**
- `verl/models/moe/expert_lora.py` — ExpertLoRACollection
- `verl/models/moe/moe_wrapper.py` — MoEVLMWrapper

---

### 任务12：将 Option-Incentivized 损失集成到 MoE 训练流水线
- **对应章节：** 3.2-3.4（完整集成）
- **状态：** `待开始`
- **前置依赖：** 任务8、9、10
- **阻塞：** 任务13

**目标：** 端到端 Option-Incentivized MoE 训练，整合所有组件。

**完整损失：**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rl}} + \alpha \cdot \text{LoadBalance} + \beta \cdot (-\lambda_2) + \gamma \cdot \text{ExpertSpecialization}$$

**产出：**
1. 更新 `MoERayTrajDAPOTrainer`，加入 option-incentivized 阶段
2. 新配置 `traj_grpo_moe_options.yaml`（包含全部超参数）
3. 基于 f 值变动的专家特化信号
4. 带迭代瓶颈消除的端到端训练脚本
5. 日志记录：λ₂ 趋势、按瓶颈区域统计的专家利用率、覆盖度指标

---

### 任务13：实现多智能体扩展（MA-GUI）
- **对应章节：** 7
- **状态：** `待开始`
- **前置依赖：** 任务12
- **阻塞：** 无
- **优先级：** 最低 — 研究性扩展

**目标：** 扩展到多智能体 GUI 自动化，支持角色特化的 agent。

**核心概念：**

**单 agent 快捷操作发现（第7.1节）：**
```
Agent A（数据录入专家）：→ macro_fill_form, macro_navigate_spreadsheet
Agent B（导航专家）：    → macro_cross_app_switch, macro_deep_settings
```

**跨 agent 协调瓶颈（第7.2节）：**
- 某些瓶颈仅在**联合**状态空间中显现
- 示例：Agent A 必须完成登录 → Agent B 才能导航到设置
- 联合快捷操作：`(Agent A: macro_login) → (Agent B: macro_navigate_settings)`

**有效层次（第7.3节）：**
```
原始：每个 agent 10-15个原始动作 → 联合空间不可处理
加快捷操作：每个 agent 3-4个快捷操作级决策 → 联合空间可管理
```

**产出：**
1. 单 agent 转移图构建
2. 联合状态空间拉普拉斯分析
3. 联合快捷操作定义与执行
4. 多智能体评估框架

---

## 总结

| 阶段 | 任务 | 描述 | 优先级 | 状态 |
|---|---|---|---|---|
| **第一阶段** | 1, 2 | 状态表示 + 图构建 | 高 | ✅ 全部完成 |
| **第二阶段** | 3, 3.1, 4 | 特征函数训练（MLP+VLM） + 瓶颈验证 | 高 | ✅ 全部完成 |
| **第三阶段** | 5, 6 | 快捷操作构建（Tool-Use SFT + RL 两种方法） | 高(5) / 中(6) | 5 数据准备✅ 训练待开始 / 6 待开始 |
| **第四阶段** | 7, 10, 11 | 层次化策略 + MVP 实验 | 高 | 待开始 |
| **第五阶段** | 8, 9, 12, 13 | MoE 集成 + 扩展 | 中(8-12) / 低(13) | 待开始 |

**MVP 关键路径：** 任务 1 ✅ → 2 ✅ → 3 ✅ → 4 ✅ → 5 数据✅/训练 → 7 → 10（7个任务，纯数据驱动，无需 RL）

**实现范围：** 仅针对 GUI-360 数据集

### 已创建文件索引

| 任务 | 文件路径 | 说明 |
|------|---------|------|
| 任务1 | `verl/models/moe/state_representation.py` | 状态哈希 + 稠密嵌入 + 轨迹处理器 |
| 任务1 | `tests/moe/test_state_representation.py` | 34个测试（全部通过） |
| 任务1 | `verl/models/moe/__init__.py`（已更新） | 导出 state_representation API |
| 任务2 | `scripts/collect_transitions.py` | 转移数据收集 CLI 脚本（支持 --granularity fine/coarse） |
| 任务2 | `outputs/transitions/gui360_full/` | fine 模式输出（6,511 states, 91,618 transitions） |
| 任务2 | `outputs/transitions/gui360_coarse/` | coarse 模式输出（989 states, per-app 连通） |
| 任务3 | `verl/models/moe/graph_analysis.py` | EigenfunctionNet + 训练 + 瓶颈识别 + 描述映射 |
| 任务3 | `scripts/train_eigenfunction.py` | CLI 训练脚本（per-app，--app/--epochs/--device） |
| 任务3 | `scripts/slurm_train_fnet.slurm` | SLURM 提交脚本（1 GPU, 200 epochs） |
| 任务3 | `outputs/fnet/gui360/` | GPU 训练输出（per-app 子目录：excel/word/ppt） |
| 任务3.1 | `verl/models/moe/vlm_eigenfunction.py` | VLM 特征函数核心模块（模型+数据集+损失） |
| 任务3.1 | `scripts/prepare_vlm_eigenfunction_data.py` | 数据准备：state_hash → screenshot + a11y 映射 |
| 任务3.1 | `scripts/train_vlm_eigenfunction.py` | VLM 特征函数训练脚本（per-app） |
| 任务3.1 | `scripts/slurm_train_vlm_fnet.slurm` | SLURM 提交脚本（per-app, 12h, --export=APP=xxx） |
| 任务3.1 | `scripts/recompute_vlm_fvalues.py` | 从 checkpoint 重新计算 f-values（修复 bf16 bug） |
| 任务3.1 | `outputs/vlm_fnet/data/{app}/` | 数据准备输出（state_manifest.json + transition_pairs.json） |
| 任务3.1 | `outputs/vlm_fnet/{app}/` | 训练输出（LoRA adapter + f_values + bottlenecks + mlp_comparison） |
| 任务3.1 | `checkpoints/Qwen2.5-VL-7B-Instruct/` | 基座模型（8.3B params, bf16） |
| 任务4 | `scripts/validate_bottlenecks.py` | 瓶颈验证脚本（5 大分析模块 + Go/No-Go 判定） |
| 任务4 | `outputs/bottleneck_validation/report.json` | 结构化 JSON 报告 |
| 任务4 | `outputs/bottleneck_validation/report.md` | 人类可读 Markdown 报告 |
| 任务4 | `outputs/bottleneck_validation/{excel,word,ppt}/*.png` | 12 张 per-app 可视化 |
| 任务4 | `outputs/bottleneck_validation/aggregate/*.png` | 4 张跨应用聚合可视化 |
| 任务5 | `scripts/prepare_macro_sft_data.py` | 跨越轨迹识别 + 快捷操作分类 + SFT 样本构造 |
| 任务5 | `train_GUI_360/config/gui360_sft_macro.yaml` | 快捷操作增强 SFT 训练配置 |
| 任务5 | `outputs/macro_sft/macro_augmented_train.parquet` | 359 条快捷操作增强 SFT 样本 |
| 任务5 | `outputs/macro_sft/macro_mixed_train.parquet` | 108,930 条混合训练数据 |
| 任务5 | `outputs/macro_sft/macro_mixed_eval.parquet` | 26,308 条混合评估数据 |
| 任务5 | `outputs/macro_sft/macro_playbook.json` | 359 条快捷操作执行手册（推理时执行器用） |
| 任务5 | `outputs/macro_sft/macro_tool_definitions.json` | 3 个快捷操作 tool JSON Schema |
| 任务5 | `outputs/macro_sft/crossing_trajectories/` | Per-app 跨越轨迹列表 |
| 任务5 | `outputs/macro_sft/statistics.json` | 数据统计 |

**核心参考文献：**
- Jinnai et al. (2020). *Exploration in RL with Deep Covering Options.* ICLR 2020.
- Wu et al. (2019). *The Laplacian in RL: Learning Representations with Efficient Approximations.* ICLR 2019.
- Mu et al. (2025). *GUI-360°.* arXiv 2511.04307.
- Rawles et al. (2024). *AndroidWorld.* arXiv 2405.14573.
