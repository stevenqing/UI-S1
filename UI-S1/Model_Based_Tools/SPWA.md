## 完整Summary：从问题定位到解法框架

---

### 一、问题重新定位：不是Grounding，是Planning

**起点**：领域普遍认为GUI agent失败是grounding能力不足（坐标预测不准）。

**我们的发现**：三种独立方法（LLM分类、thought-based、hit-test）在两个数据集上一致否定这个假设。独立错误中71-98%是planning error（选错了UI元素），grounding error只占2-6%。

**进一步分解**（Exp2b/c，Cognitive Interference）：
- 无截图时function selection已经74-87%正确——模型知道该做什么
- 失败完全集中在coordinate prediction（有截图49%，无截图仅6%）
- 结论：模型知道该做什么，但在截图里找不到正确的目标元素——是visual element identification的失败

---

### 二、Visual Element Identification失败的原因：Attention竞争

**Probing实验**（v2）：单模型处理截图+历史+任务时，三类信息的表示都很弱（UI state +17pp，target element +8pp，correctness +6pp）。信号弱的根本原因是attention被多维输入同时竞争。

**Multi-agent验证**（Exp2d）：

| 条件 | Step Acc | TSR | Δ |
|------|:---:|:---:|:---:|
| C0（单模型） | 36.5% | 2.97% | — |
| F4（V+H分工） | **41.5%** | **6.93%** | **+5pp*** |
| F6（Ensemble×3） | 34.9% | 4.46% | -1.6pp |

*p<0.0001，提升完全来自args match（坐标），直接对应visual element identification。专门分工远胜于增加compute。

**Agent质量的关键性**：Agent V提到GT目标控件时F4 accuracy=58.3%，未提到时25.9%，差距32.4pp——Agent V的coverage（当前48%）是最大的单一瓶颈。

---

### 三、Long-horizon Failure的机制

**核心问题**：即使F4把per-step accuracy提升到41.5%，10步轨迹TSR仍然接近0（0.415^10≈0.01%）。

**Exp2e揭示的机制**：

```
Per-step accuracy随preceding wrong steps线性下降：
  pw=0：50.0%
  pw=3：22.7%
  gradient：-6.6pp/step，R²=0.91

30%的步骤陷入failure zone（pw=3+，pos=3+）
accuracy只有20.5%，所有intervention失效
```

**Mismatch Tax量化**（Oracle History实验）：
- GT history vs 预测history：+4.9pp step accuracy，TSR 2.5倍
- 但即使消除mismatch，long-horizon TSR仍接近0
- **结论：mismatch是contributing factor（5pp），不是主因；per-step accuracy不足才是根本**

**Offline evaluation的structural问题**：截图永远是GT的，历史是模型自己生成的——当模型出错后两者描述不同的世界，随错误累积不可逆地拉低后续步骤accuracy。

---

### 四、Inference-time方法的天花板

系统测试所有可能的inference-time干预：

| 方法 | Step Acc | TSR | 结论 |
|------|:---:|:---:|------|
| C0 baseline | 36.5% | 2.97% | — |
| F4 multi-agent | **41.5%** | **6.93%** | +5pp上限 |
| Oracle history | 41.4% | 7.43% | 需要GT，不可部署 |
| Verifier+resample | 29.3% | 4.46% | 重采样加噪声 |
| Reset机制 | 36.4% | 2.97% | 完全无效 |

**Reset失败原因**：Probe distribution shift（43.5% false alarm）、wrong history仍有价值（清空历史比有错误历史更差）、failure zone步骤无法恢复。

**天花板确认**：所有inference-time方法上限是F4（+5pp）。Failure zone（30%步骤）无法被任何inference-time方法触及。

---

### 五、Sequential Progress作为Long-horizon的正确度量

**三种指标的本质**：
```
TSR：所有步骤都对才算1，乘法诅咒使其接近0
Scattered Progress：每步独立判断，≈step accuracy的per-trajectory版本
Sequential Progress：first_error_step / total_steps
  → 保留了顺序性
  → 直接度量"出第一个错之前能走多远"
  → 是TSR的合理软化版
```

**为什么Sequential Progress是正确的训练目标**：
- 比TSR更dense（非0/1，连续值）
- 比scattered progress更有long-horizon信息（惩罚早期错误）
- 直接对应SPWA的设计动机：step t对sequential progress的边际贡献 = SPWA(t)

---

### 六、Training-time框架：Semi-online RL + Sequential Progress

**领域共识**（UI-S1确认）：

| 范式 | 机制 | 问题 |
|------|------|------|
| Offline RL | GT history，step-wise supervision | 训练和部署distribution mismatch，multi-turn灾难性失败 |
| Online RL | 真实环境交互 | 计算代价极高，reward稀疏 |
| Semi-online RL | 模型自己的history + GT截图 | 中间路线，我们的设计空间 |

**我们在semi-online rollout基础上的三个贡献**：

**贡献1：Sequential Progress作为reward**
```
不是binary step accuracy（UI-S1的做法）
而是：SP = first_error_step / total_steps

K条rollout：
  rollout A：step 3出错 → SP=0.30
  rollout B：step 7出错 → SP=0.70
  rollout C：step 9出错 → SP=0.90

→ 有明确区分度，捕捉"错误越晚越好"的long-horizon信息
→ UI-S1的binary reward无法区分A和B
```

**贡献2：Cross-trajectory Step-level比较（GiGPO思想）**
```
GT截图固定的特性：每个step position天然是anchor group
K条rollout在step t的截图完全相同

Step t的advantage：
  A(step t, rollout i) = SP(rollout i) - mean(SP over K rollouts)

→ 精确告诉模型"step t的哪个action让trajectory走得更远"
→ GiGPO在online setting需要找重访的state，我们的offline setting免费获得
```

**贡献3：OPD提取trajectory-level Hindsight**
```
对比高SP vs 低SP的rollout，在出错步骤提取textual hint：

rollout A（SP=0.30）在step 3点了Format Cells
rollout B（SP=0.70）在step 3点了Merge & Center

OPD hint：
  "You should click Merge & Center (located in Home ribbon, 
   Alignment group, position ~234,89) to unmerge cells,
   not Format Cells which opens a dialog requiring more steps"

→ 从scalar advantage(-0.25)升级为token-level directional signal
→ 告诉模型"具体哪些token应该不同，怎么改，为什么"
→ 直接对应failure zone步骤的recovery训练
```

---

### 七、Multi-agent在训练框架中的位置

**串行Multi-agent的Rollout**：
```
每步t三个串行forward pass：
  π_V(screenshot_t) → visual_desc_t
  π_S(task, history_t) → progress_t + status_t  
  π_A(screenshot_t + task + history_t + visual_desc_t + progress_t) → action_t

history_{t+1} = history_t + [action_t]
```

**三个Agent各自的reward信号**：
```
π_V：r_V = thought_hit_t × step_correct_t × SPWA(t)
     告诉π_V"你的描述有没有帮助π_A做对"

π_S：r_S = status_correct_t × SPWA(t)
          + reset_effectiveness（reset后下个segment的SP提升）
     告诉π_S"你的boundary detection准不准"

π_A：r_A = SP_advantage × SPWA(t)
     告诉π_A"你的action让trajectory走了多远"
```

**SPWA在Serial Multi-agent下的统一作用**：
```
SPWA(t) = ∂E[SP] / ∂P(step t correct)

→ 在failure zone（pw=3+）：SPWA低，但不为0
  OPD的token-level signal弥补了scalar reward的稀疏
  
→ 在subtask boundary：SPWA额外高
  π_S的boundary detection被重点强化

→ 在clean context（pw=0）：SPWA高
  π_V的visual description被重点强化
  
三个问题被同一个权重函数统一处理，不需要人工设计
```

---

### 八、和已有工作的精确定位

```
UI-S1：解决了training distribution问题
         semi-online rollout（模型history + GT截图）
         但reward是binary step accuracy（局部的）
         没有cross-trajectory比较
         没有OPD
         没有multi-agent分工

我们：在semi-online rollout基础上：
  + Sequential Progress（long-horizon reward）
  + Cross-trajectory step-level比较（GiGPO思想的offline版本）
  + OPD trajectory-level hindsight
  + Multi-agent串行分工（π_V/π_S/π_A各有专门的训练信号）
  + 量化了mismatch tax（4.9pp，Exp2e）

GiGPO/SALT：解决了cross-trajectory step-level比较（online setting）
我们：把同样的思想用到offline/semi-online setting
     + Sequential Progress替代binary TSR
     + OPD的hindsight signal
```

---

### 九、整体Story Arc

```
诊断（Exp1→Exp2f）：
  Planning error是主因（71-98%）
  Visual element identification失败是根本机制
  Multi-agent input decomposition有效（+5pp）但有天花板
  Failure zone（30%步骤）所有inference-time方法失效
  Offline mismatch tax可量化（4.9pp）但不是long-horizon的主因
  Per-step accuracy不足是数学根本（0.415^10≈0.01%）

度量（Sequential Progress）：
  TSR过于严格（乘法诅咒）
  Scattered Progress信息量不足（≈step accuracy）
  Sequential Progress是正确的long-horizon度量
  SPWA是sequential progress的step-level credit assignment

训练框架：
  Semi-online rollout（UI-S1确认的正确机制）
  + Sequential Progress作为reward
  + K条rollout cross-trajectory比较（offline下免费的GiGPO）
  + OPD trajectory-level hindsight
  + Multi-agent串行分工
  → 让训练信号本身包含long-horizon信息
  → 这是UI-S1之后的自然下一步
```