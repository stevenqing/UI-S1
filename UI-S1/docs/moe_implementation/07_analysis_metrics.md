# 分析与评估指标

## 1. 核心评估问题

### 1.1 需要回答的问题

| 问题 | 对应指标 | 期望结果 |
|------|---------|---------|
| Experts 是否分化？ | Specialization Score | > 0.6 (random=0.25) |
| MoE 是否比 Single LoRA 好？ | Task Accuracy | MoE > Single |
| 分化是否统计显著？ | P-value | < 0.05 |
| 是否有 expert 坍塌？ | Expert Utilization | 每个 > 15% |

---

## 2. Expert 分化分析

### 2.1 Routing Matrix 计算

```python
# verl/utils/moe_analysis/specialization.py

import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import defaultdict


def compute_routing_matrix(
    routing_weights: torch.Tensor,
    instruction_types: List[str],
    num_experts: int = 4,
) -> np.ndarray:
    """
    计算 Routing Matrix: P(expert | instruction_type)

    Args:
        routing_weights: [N, num_experts] 所有样本的 routing weights
        instruction_types: [N] 每个样本的 instruction 类型

    Returns:
        routing_matrix: [num_types, num_experts]
            routing_matrix[i, j] = P(expert j | type i)
    """
    types = ['click', 'type', 'navigate', 'scroll']
    num_types = len(types)

    routing_matrix = np.zeros((num_types, num_experts))

    # 使用 hard routing (argmax) 来计算分布
    routing_np = routing_weights.detach().cpu().numpy()
    dominant_experts = routing_np.argmax(axis=1)

    for i, t in enumerate(types):
        mask = np.array([it == t for it in instruction_types])
        if mask.sum() > 0:
            for j in range(num_experts):
                routing_matrix[i, j] = (dominant_experts[mask] == j).mean()

    return routing_matrix


def compute_specialization_score(routing_matrix: np.ndarray) -> float:
    """
    计算 Specialization Score

    如果完美分化：每行应该有一个接近 1 的值
    Score = mean(max(row))

    Range:
    - 0.25: 完全随机 (4 experts)
    - 1.0: 完美分化

    Args:
        routing_matrix: [num_types, num_experts]

    Returns:
        specialization_score: float in [0.25, 1.0]
    """
    row_max = routing_matrix.max(axis=1)
    return row_max.mean()


def compute_mutual_information(routing_matrix: np.ndarray) -> float:
    """
    计算 Instruction Type 和 Expert 之间的互信息

    高互信息 = 高相关性 = 好的分化

    Args:
        routing_matrix: [num_types, num_experts]

    Returns:
        mutual_information: float >= 0
    """
    # Normalize to joint probability
    p_joint = routing_matrix / (routing_matrix.sum() + 1e-10)

    # Marginals
    p_type = p_joint.sum(axis=1, keepdims=True)
    p_expert = p_joint.sum(axis=0, keepdims=True)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    with np.errstate(divide='ignore', invalid='ignore'):
        mi_matrix = p_joint * np.log(p_joint / (p_type * p_expert + 1e-10) + 1e-10)
        mi_matrix = np.nan_to_num(mi_matrix)

    return mi_matrix.sum()


def compute_expert_utilization(routing_weights: torch.Tensor) -> np.ndarray:
    """
    计算每个 expert 的使用率

    Args:
        routing_weights: [N, num_experts]

    Returns:
        utilization: [num_experts] 每个 expert 被选中的比例
    """
    dominant_experts = routing_weights.argmax(dim=-1)
    num_experts = routing_weights.size(1)

    utilization = torch.bincount(
        dominant_experts, minlength=num_experts
    ).float() / routing_weights.size(0)

    return utilization.cpu().numpy()


class SpecializationAnalyzer:
    """
    综合分析 Expert 分化情况
    """

    def __init__(self, num_experts: int = 4):
        self.num_experts = num_experts
        self.types = ['click', 'type', 'navigate', 'scroll']

    def analyze(
        self,
        routing_weights: torch.Tensor,
        instruction_types: List[str],
    ) -> Dict:
        """
        执行完整的分化分析

        Returns:
            Dict with all metrics
        """
        # 1. Routing Matrix
        routing_matrix = compute_routing_matrix(
            routing_weights, instruction_types, self.num_experts
        )

        # 2. Specialization Score
        spec_score = compute_specialization_score(routing_matrix)

        # 3. Mutual Information
        mi = compute_mutual_information(routing_matrix)

        # 4. Expert Utilization
        utilization = compute_expert_utilization(routing_weights)

        # 5. 检测坍塌
        is_collapsed = utilization.max() > 0.7

        # 6. 统计显著性
        significance = self.test_significance(
            routing_weights, instruction_types
        )

        return {
            'routing_matrix': routing_matrix,
            'specialization_score': spec_score,
            'mutual_information': mi,
            'expert_utilization': utilization,
            'is_collapsed': is_collapsed,
            'significance': significance,
        }

    def test_significance(
        self,
        routing_weights: torch.Tensor,
        instruction_types: List[str],
        num_permutations: int = 1000,
    ) -> Dict:
        """
        使用 Permutation Test 检验分化的统计显著性

        H0: routing 与 instruction type 无关
        H1: routing 与 instruction type 相关
        """
        # 真实的 specialization score
        real_matrix = compute_routing_matrix(
            routing_weights, instruction_types, self.num_experts
        )
        real_score = compute_specialization_score(real_matrix)

        # Permutation: 打乱 instruction_types
        permuted_scores = []
        for _ in range(num_permutations):
            shuffled_types = np.random.permutation(instruction_types).tolist()
            perm_matrix = compute_routing_matrix(
                routing_weights, shuffled_types, self.num_experts
            )
            perm_score = compute_specialization_score(perm_matrix)
            permuted_scores.append(perm_score)

        permuted_scores = np.array(permuted_scores)

        # P-value
        p_value = (permuted_scores >= real_score).mean()

        return {
            'real_score': real_score,
            'permuted_mean': permuted_scores.mean(),
            'permuted_std': permuted_scores.std(),
            'p_value': p_value,
            'is_significant': p_value < 0.05,
        }
```

---

## 3. 可视化工具

### 3.1 Routing Matrix 热力图

```python
# verl/utils/moe_analysis/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
from pathlib import Path


def plot_routing_matrix(
    routing_matrix: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Routing Pattern: P(Expert | Instruction Type)",
):
    """
    可视化 Routing Matrix 为热力图
    """
    types = ['click', 'type', 'navigate', 'scroll']
    experts = [f'Expert {i}' for i in range(routing_matrix.shape[1])]

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        routing_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=experts,
        yticklabels=types,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Routing Probability'},
    )

    plt.title(title)
    plt.xlabel('Expert')
    plt.ylabel('Instruction Type')

    # 标记主导 expert
    for i in range(routing_matrix.shape[0]):
        max_j = routing_matrix[i].argmax()
        ax.add_patch(plt.Rectangle(
            (max_j, i), 1, 1,
            fill=False, edgecolor='red', linewidth=2
        ))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_specialization_over_training(
    specialization_history: List[float],
    save_path: Optional[str] = None,
):
    """
    可视化训练过程中 Specialization Score 的变化
    """
    plt.figure(figsize=(12, 5))

    epochs = range(1, len(specialization_history) + 1)

    plt.plot(epochs, specialization_history, 'b-o', linewidth=2, markersize=8)

    # 参考线
    plt.axhline(y=0.25, color='r', linestyle='--', label='Random (0.25)')
    plt.axhline(y=0.6, color='g', linestyle='--', label='Target (0.6)')
    plt.axhline(y=1.0, color='orange', linestyle='--', label='Perfect (1.0)')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Specialization Score', fontsize=12)
    plt.title('Expert Specialization Over Training', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_expert_utilization(
    utilization: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    可视化 Expert 使用率
    """
    num_experts = len(utilization)
    experts = [f'Expert {i}' for i in range(num_experts)]

    plt.figure(figsize=(8, 5))

    colors = ['#3498db' if u > 0.15 else '#e74c3c' for u in utilization]
    bars = plt.bar(experts, utilization * 100, color=colors)

    # 参考线
    plt.axhline(y=100/num_experts, color='g', linestyle='--',
                label=f'Uniform ({100/num_experts:.1f}%)')
    plt.axhline(y=15, color='r', linestyle='--', label='Min threshold (15%)')

    plt.xlabel('Expert')
    plt.ylabel('Utilization (%)')
    plt.title('Expert Utilization')
    plt.legend()

    # 数值标签
    for bar, u in zip(bars, utilization):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{u*100:.1f}%',
            ha='center',
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_moe_vs_single_comparison(
    moe_accuracy: Dict[str, float],
    single_accuracy: Dict[str, float],
    save_path: Optional[str] = None,
):
    """
    对比 MoE 和 Single LoRA 的性能
    """
    types = list(moe_accuracy.keys())
    x = np.arange(len(types))
    width = 0.35

    plt.figure(figsize=(10, 6))

    moe_values = [moe_accuracy[t] * 100 for t in types]
    single_values = [single_accuracy[t] * 100 for t in types]

    bars1 = plt.bar(x - width/2, moe_values, width, label='MoE', color='#3498db')
    bars2 = plt.bar(x + width/2, single_values, width, label='Single LoRA', color='#e74c3c')

    plt.xlabel('Instruction Type')
    plt.ylabel('Accuracy (%)')
    plt.title('MoE vs Single LoRA Performance')
    plt.xticks(x, types)
    plt.legend()

    # 添加数值
    for bar in bars1 + bars2:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.1f}',
            ha='center',
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
```

---

## 4. 完整评估流程

### 4.1 评估脚本

```python
# scripts/evaluate_moe.py

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

from verl.utils.moe_analysis.specialization import SpecializationAnalyzer
from verl.utils.moe_analysis.visualization import (
    plot_routing_matrix,
    plot_expert_utilization,
    plot_moe_vs_single_comparison,
)


def load_routing_data(checkpoint_dir: str) -> dict:
    """加载训练过程中保存的 routing 数据"""
    routing_path = Path(checkpoint_dir) / 'routing_history.pt'
    return torch.load(routing_path)


def evaluate_moe_model(
    moe_checkpoint: str,
    test_data_path: str,
    output_dir: str,
):
    """
    评估 MoE 模型

    Args:
        moe_checkpoint: MoE checkpoint 目录
        test_data_path: 测试数据路径
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型和数据
    print("Loading model and data...")
    # ... 加载逻辑 ...

    # 2. 运行推理收集 routing weights
    print("Running inference...")
    all_routing_weights = []
    all_instruction_types = []
    all_predictions = []
    all_labels = []

    # ... 推理循环 ...

    routing_weights = torch.cat(all_routing_weights, dim=0)

    # 3. 分化分析
    print("Analyzing specialization...")
    analyzer = SpecializationAnalyzer(num_experts=4)
    analysis = analyzer.analyze(routing_weights, all_instruction_types)

    # 4. 打印结果
    print("\n" + "="*60)
    print("MoE Evaluation Results")
    print("="*60)

    print(f"\nSpecialization Score: {analysis['specialization_score']:.3f}")
    print(f"  - Random baseline: 0.25")
    print(f"  - Target: > 0.6")

    print(f"\nMutual Information: {analysis['mutual_information']:.4f}")

    print(f"\nExpert Utilization:")
    for i, u in enumerate(analysis['expert_utilization']):
        status = "OK" if u > 0.15 else "LOW"
        print(f"  - Expert {i}: {u*100:.1f}% [{status}]")

    print(f"\nCollapse Detection: {'YES' if analysis['is_collapsed'] else 'NO'}")

    print(f"\nStatistical Significance:")
    sig = analysis['significance']
    print(f"  - P-value: {sig['p_value']:.4f}")
    print(f"  - Significant: {'YES' if sig['is_significant'] else 'NO'}")

    # 5. 生成可视化
    print("\nGenerating visualizations...")

    plot_routing_matrix(
        analysis['routing_matrix'],
        save_path=str(output_dir / 'routing_matrix.png'),
    )

    plot_expert_utilization(
        analysis['expert_utilization'],
        save_path=str(output_dir / 'expert_utilization.png'),
    )

    # 6. 保存完整报告
    report = {
        'specialization_score': analysis['specialization_score'],
        'mutual_information': analysis['mutual_information'],
        'expert_utilization': analysis['expert_utilization'].tolist(),
        'is_collapsed': analysis['is_collapsed'],
        'significance': {
            'p_value': sig['p_value'],
            'is_significant': sig['is_significant'],
        },
        'routing_matrix': analysis['routing_matrix'].tolist(),
    }

    with open(output_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def compare_moe_single(
    moe_results: str,
    single_results: str,
    output_dir: str,
):
    """
    对比 MoE 和 Single LoRA 结果
    """
    with open(moe_results) as f:
        moe = json.load(f)

    with open(single_results) as f:
        single = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 对比准确率
    print("\n" + "="*60)
    print("MoE vs Single LoRA Comparison")
    print("="*60)

    print("\nOverall Accuracy:")
    print(f"  - MoE: {moe['accuracy']*100:.2f}%")
    print(f"  - Single: {single['accuracy']*100:.2f}%")
    print(f"  - Improvement: {(moe['accuracy']-single['accuracy'])*100:+.2f}%")

    print("\nPer-Type Accuracy:")
    for t in ['click', 'type', 'navigate', 'scroll']:
        moe_acc = moe['type_accuracy'].get(t, 0)
        single_acc = single['type_accuracy'].get(t, 0)
        diff = moe_acc - single_acc
        print(f"  - {t}: MoE={moe_acc*100:.1f}%, Single={single_acc*100:.1f}% ({diff*100:+.1f}%)")

    # 可视化对比
    plot_moe_vs_single_comparison(
        moe['type_accuracy'],
        single['type_accuracy'],
        save_path=str(output_dir / 'moe_vs_single.png'),
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # evaluate 命令
    eval_parser = subparsers.add_parser('evaluate')
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    eval_parser.add_argument('--test_data', type=str, required=True)
    eval_parser.add_argument('--output', type=str, required=True)

    # compare 命令
    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('--moe_results', type=str, required=True)
    compare_parser.add_argument('--single_results', type=str, required=True)
    compare_parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    if args.command == 'evaluate':
        evaluate_moe_model(args.checkpoint, args.test_data, args.output)
    elif args.command == 'compare':
        compare_moe_single(args.moe_results, args.single_results, args.output)


if __name__ == "__main__":
    main()
```

---

## 5. 结果解读指南

### 5.1 Routing Matrix 解读

```
理想结果 (高分化):
           Expert 0  Expert 1  Expert 2  Expert 3
click        0.82      0.06      0.08      0.04
type         0.05      0.78      0.10      0.07
navigate     0.08      0.12      0.73      0.07
scroll       0.04      0.08      0.09      0.79

→ 每行有一个明显的主导 expert (>0.7)
→ Specialization Score ≈ 0.78
→ 每个 expert 专注于不同的 instruction type


不理想结果 (坍塌):
           Expert 0  Expert 1  Expert 2  Expert 3
click        0.92      0.03      0.03      0.02
type         0.88      0.05      0.04      0.03
navigate     0.85      0.06      0.05      0.04
scroll       0.90      0.04      0.03      0.03

→ Expert 0 主导所有类型
→ 需要增加 balance_weight


不理想结果 (均匀但无分化):
           Expert 0  Expert 1  Expert 2  Expert 3
click        0.26      0.25      0.24      0.25
type         0.25      0.26      0.24      0.25
navigate     0.24      0.25      0.26      0.25
scroll       0.25      0.24      0.25      0.26

→ 虽然均匀，但没有分化
→ 可能 balance_weight 太大，或模型没有学到有意义的 routing
```

### 5.2 决策树

```
                    ┌─────────────────────┐
                    │ Specialization Score │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         < 0.4            0.4-0.6            > 0.6
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │ 检查 Expert     │ │ 中等分化    │ │ 好的分化        │
    │ Utilization    │ │             │ │                 │
    └────────┬────────┘ │ 继续训练或  │ │ 检查 MoE vs     │
             │          │ 调整超参数   │ │ Single 准确率   │
             ▼          └─────────────┘ └────────┬────────┘
    ┌─────────────────┐                          │
    │ 某 Expert > 70% │                          ▼
    │ = 坍塌          │              ┌─────────────────────┐
    │                 │              │ MoE > Single?       │
    │ 增加            │              ├──────────┬──────────┤
    │ balance_weight  │              │   YES    │    NO    │
    └─────────────────┘              │          │          │
                                     ▼          ▼
                              ┌──────────┐ ┌──────────────┐
                              │ 成功！   │ │ 分化了但没   │
                              │ 继续完整 │ │ 帮助，分析   │
                              │ 实验     │ │ 为什么       │
                              └──────────┘ └──────────────┘
```

---

## 6. 实验报告模板

### 6.1 Markdown 模板

```markdown
# MoE Pilot Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-VL-7B |
| Num Experts | 4 |
| Expert LoRA Rank | 16 |
| Balance Weight | 0.1 |
| Top-K | 1 |
| Training Epochs | 10 |
| Training Samples | 4000 |

## Results Summary

### Specialization Analysis

| Metric | Value | Target |
|--------|-------|--------|
| Specialization Score | X.XX | > 0.6 |
| Mutual Information | X.XXXX | Higher is better |
| P-value | X.XXXX | < 0.05 |
| Collapsed | YES/NO | NO |

### Expert Utilization

| Expert | Utilization | Status |
|--------|------------|--------|
| Expert 0 | XX.X% | OK/LOW |
| Expert 1 | XX.X% | OK/LOW |
| Expert 2 | XX.X% | OK/LOW |
| Expert 3 | XX.X% | OK/LOW |

### Routing Matrix

![Routing Matrix](./routing_matrix.png)

### Performance Comparison

| Metric | MoE | Single LoRA | Diff |
|--------|-----|-------------|------|
| Overall Accuracy | XX.X% | XX.X% | +X.X% |
| Click Accuracy | XX.X% | XX.X% | +X.X% |
| Type Accuracy | XX.X% | XX.X% | +X.X% |
| Navigate Accuracy | XX.X% | XX.X% | +X.X% |
| Scroll Accuracy | XX.X% | XX.X% | +X.X% |

## Conclusion

[Your analysis and conclusions here]

## Next Steps

[Recommendations for next steps]
```

---

## 7. 进一步方案: 基于KL散度的Expert多样性正则化

### 7.1 背景: 当前Balance Loss的局限性

当前的实现使用三种balance loss来防止expert collapse:
- `mse`: 与均匀分布的MSE距离
- `switch`: Switch Transformer风格的auxiliary loss
- `entropy`: 最大化路由熵

**局限性**: 这些方法都只在**routing层面**进行约束，没有直接确保**expert输出**之间的差异。即使routing是平衡的，不同expert的LoRA层可能仍然学到相似的表示，导致:
1. Expert行为趋同，失去分化的意义
2. MoE退化为多个重复的LoRA ensemble
3. 无法真正发挥MoE的specialization优势

### 7.2 核心思想: Expert输出KL散度

通过在**输出层面**添加不同expert之间的KL散度约束，直接鼓励expert产生不同的行为。

```
Loss = LM_loss + balance_weight * routing_balance_loss +
       diversity_weight * expert_diversity_loss
```

其中 `expert_diversity_loss` = -KL(p_i || p_j) (负号因为我们要最大化KL)

### 7.3 实现方案

#### 方案A: 输出分布KL散度 (推荐)

```python
# verl/models/moe/moe_loss.py

class ExpertDiversityLoss(nn.Module):
    """
    通过expert输出分布的KL散度鼓励diversity

    核心思想:
    - 对每个batch样本，获取不同expert的LoRA输出
    - 计算输出分布之间的KL散度
    - KL越大 = expert行为越diverse

    优点:
    - 直接约束expert行为差异
    - 不需要额外forward pass
    - 与routing balance loss互补

    缺点:
    - 需要在forward时保存所有expert输出
    - 增加内存消耗
    """

    def __init__(self, num_experts: int, diversity_weight: float = 0.05):
        super().__init__()
        self.num_experts = num_experts
        self.diversity_weight = diversity_weight

    def forward(
        self,
        expert_outputs: torch.Tensor,  # [B, num_experts, seq_len, hidden]
        routing_weights: torch.Tensor,  # [B, num_experts]
    ) -> torch.Tensor:
        """
        计算基于KL散度的diversity loss

        Loss = -weighted_mean(KL(p_i || p_j)) for i != j
        负号因为我们想要MAXIMIZE diversity
        """
        B, E, S, H = expert_outputs.shape

        # 归一化到概率分布 (在hidden维度上softmax)
        probs = F.softmax(expert_outputs, dim=-1)  # [B, E, S, H]

        # 计算pairwise KL (只对活跃的expert pairs)
        kl_losses = []
        for i in range(E):
            for j in range(i + 1, E):
                # 根据两个expert的使用频率加权
                weight = (routing_weights[:, i] * routing_weights[:, j]).mean()

                if weight > 1e-6:  # 只计算同时使用的expert pairs
                    # KL(p_i || p_j)
                    kl = F.kl_div(
                        probs[:, i].log_softmax(dim=-1),
                        probs[:, j],
                        reduction='batchmean'
                    )
                    kl_losses.append(weight * kl)

        # 负号因为我们想要MAXIMIZE diversity
        diversity_loss = -sum(kl_losses) / max(len(kl_losses), 1)

        return diversity_loss
```

#### 方案B: LoRA参数分布KL散度

```python
class LoRAParameterDiversityLoss(nn.Module):
    """
    通过LoRA参数分布的KL散度鼓励diversity

    核心思想:
    - 将每个expert的LoRA参数视为分布
    - 计算不同expert参数分布之间的KL散度
    - 参数level的约束，比输出level更稳定

    优点:
    - 直接约束参数空间
    - 计算效率高
    - 不依赖batch数据

    缺点:
    - 参数diversity ≠ 输出diversity
    - 可能过于约束，限制模型capacity
    """

    def __init__(
        self,
        expert_collection: ExpertLoRACollection,
        param_weight: float = 0.01,
    ):
        super().__init__()
        self.expert_collection = expert_collection
        self.param_weight = param_weight

    def forward(self) -> torch.Tensor:
        """计算LoRA参数的KL散度"""
        total_loss = 0.0
        count = 0

        num_layers = self.expert_collection.num_layers
        target_modules = self.expert_collection.target_modules
        num_experts = self.expert_collection.num_experts

        for layer_idx in range(num_layers):
            for module_name in target_modules:
                # 获取每个expert的LoRA参数
                params = []
                for expert_idx in range(num_experts):
                    expert = self.expert_collection.experts[expert_idx]
                    key = f"layer_{layer_idx}_{module_name}"
                    if key in expert.lora_layers:
                        lora = expert.lora_layers[key]
                        # 将A和B矩阵flatten为向量
                        param_vec = torch.cat([
                            lora.lora_A.flatten(),
                            lora.lora_B.flatten()
                        ])
                        params.append(param_vec)

                # 计算pairwise KL (假设参数为高斯分布)
                for i in range(len(params)):
                    for j in range(i + 1, len(params)):
                        # 估计分布参数
                        mu_i, sigma_i = params[i].mean(), params[i].std() + 1e-8
                        mu_j, sigma_j = params[j].mean(), params[j].std() + 1e-8

                        # 两个高斯分布的KL散度
                        kl = (torch.log(sigma_j / sigma_i) +
                              (sigma_i**2 + (mu_i - mu_j)**2) / (2 * sigma_j**2) - 0.5)
                        total_loss += kl
                        count += 1

        # 负号最大化diversity
        return -self.param_weight * (total_loss / count if count > 0 else 0)
```

#### 方案C: 层级化KL散度 (推荐组合)

```python
class HierarchicalMoELoss(nn.Module):
    """
    结合routing和output diversity的层级化loss

    三个层次:
    1. Routing balance: 确保expert使用均匀
    2. Expert output diversity: 确保expert行为不同
    3. Layer-wise consistency: 确保跨层的一致性

    配置:
        balance_weight: routing balance loss权重 (推荐: 0.1)
        diversity_weight: output diversity loss权重 (推荐: 0.05)
        consistency_weight: cross-layer consistency权重 (推荐: 0.02)
    """

    def __init__(
        self,
        num_experts: int,
        balance_weight: float = 0.1,
        diversity_weight: float = 0.05,
        consistency_weight: float = 0.02,
        balance_type: str = 'mse',
    ):
        super().__init__()
        self.num_experts = num_experts

        self.balance_loss_fn = LoadBalanceLoss(num_experts=num_experts, balance_type=balance_type)
        self.diversity_loss_fn = ExpertDiversityLoss(num_experts=num_experts)

        self.balance_weight = balance_weight
        self.diversity_weight = diversity_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        lm_loss: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_outputs: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
    ) -> MoELossOutput:
        """计算combined MoE loss"""
        # 1. Standard routing balance loss
        balance_loss = self.balance_loss_fn(routing_weights, router_logits)

        # 2. Expert diversity loss (如果提供了outputs)
        diversity_loss = torch.tensor(0.0, device=lm_loss.device)
        if expert_outputs is not None:
            diversity_loss = self.diversity_loss_fn(expert_outputs, routing_weights)

        # 3. Total loss
        total_loss = (
            lm_loss +
            self.balance_weight * balance_loss +
            self.diversity_weight * diversity_loss
        )

        return MoELossOutput(
            total_loss=total_loss,
            lm_loss=lm_loss,
            balance_loss=balance_loss,
            z_loss=diversity_loss,  # 复用z_loss字段
        )
```

### 7.4 集成修改

#### 修改ExpertLoRACollection

```python
# verl/models/moe/expert_lora.py

class ExpertLoRACollection(nn.Module):
    # ... 现有代码 ...

    def get_all_expert_outputs(
        self,
        layer_idx: int,
        module_name: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取所有expert的输出用于diversity计算

        Args:
            layer_idx: Transformer层索引
            module_name: 目标模块名
            x: 输入 [B, seq_len, hidden_size]

        Returns:
            outputs: [B, num_experts, seq_len, hidden_size]
        """
        outputs = []
        for expert_idx in range(self.num_experts):
            delta = self.experts[expert_idx].get_lora_delta(layer_idx, module_name, x)
            outputs.append(delta.unsqueeze(1))  # [B, 1, seq_len, hidden]

        return torch.cat(outputs, dim=1)  # [B, E, S, H]
```

#### 修改MoE Trainer

```python
# verl/trainer/ppo/moe_dapo_trainer.py

class MoERayTrajDAPOTrainer(RayTrajDAPOTrainer):
    # ... 现有代码 ...

    def compute_moe_loss(
        self,
        lm_loss: torch.Tensor,
        routing_weights: torch.Tensor,
        router_logits: torch.Tensor,
        expert_outputs: Optional[torch.Tensor] = None,  # 新增
    ) -> Dict[str, torch.Tensor]:
        """计算包含diversity的MoE loss"""
        if not self.moe_enabled or not self._moe_initialized:
            return {'total_loss': lm_loss}

        # 如果启用了diversity loss
        if hasattr(self, '_use_diversity_loss') and self._use_diversity_loss:
            loss_output = self._moe_loss_fn(
                lm_loss=lm_loss,
                routing_weights=routing_weights,
                router_logits=router_logits,
                expert_outputs=expert_outputs,  # 传递expert outputs
            )
        else:
            # 原有的loss计算
            loss_output = self._moe_loss_fn(
                lm_loss=lm_loss,
                routing_weights=routing_weights,
                router_logits=router_logits,
            )

        return {
            'total_loss': loss_output.total_loss,
            'lm_loss': loss_output.lm_loss,
            'balance_loss': loss_output.balance_loss,
            'diversity_loss': loss_output.z_loss if loss_output.z_loss is not None else torch.tensor(0.0),
        }
```

### 7.5 配置示例

```yaml
# examples/qwen_gui_moe/config/traj_grpo_moe_diversity.yaml

model:
  moe:
    enabled: true
    num_experts: 4
    top_k: 1
    expert_lora_r: 16
    expert_lora_alpha: 32
    target_modules: ['q_proj', 'v_proj']

    # Router配置
    router_hidden: 256
    router_temperature: 1.0

    # Loss配置
    balance_weight: 0.1          # routing balance (标准)
    balance_type: 'mse'
    diversity_weight: 0.05       # output diversity KL (新增)
    diversity_type: 'output_kl'  # 或 'param_kl'
    use_diversity_loss: true     # 启用diversity loss

    # Z-loss (可选)
    z_loss_weight: 0.001

trainer:
  trainer_class: verl.trainer.ppo.moe_dapo_trainer.MoERayTrajDAPOTrainer

  # MoE logging
  log_diversity_metrics: true
  log_routing_matrix_freq: 100
```

### 7.6 评估指标

添加diversity loss后，需要监控额外的指标:

```python
def compute_diversity_metrics(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
) -> Dict[str, float]:
    """
    计算diversity相关指标

    Returns:
        - avg_pairwise_kl: 平均pairwise KL散度
        - min_pairwise_kl: 最小KL (检查是否有过于相似的experts)
        - max_pairwise_kl: 最大KL
        - diversity_score: 标准化diversity分数 [0, 1]
    """
    B, E, S, H = expert_outputs.shape
    probs = F.softmax(expert_outputs, dim=-1)

    kls = []
    for i in range(E):
        for j in range(i + 1, E):
            weight = (routing_weights[:, i] * routing_weights[:, j]).mean()
            if weight > 1e-6:
                kl = F.kl_div(
                    probs[:, i].log_softmax(dim=-1),
                    probs[:, j],
                    reduction='batchmean'
                ).item()
                kls.append(kl)

    if not kls:
        return {
            'avg_pairwise_kl': 0.0,
            'min_pairwise_kl': 0.0,
            'max_pairwise_kl': 0.0,
            'diversity_score': 0.0,
        }

    return {
        'avg_pairwise_kl': np.mean(kls),
        'min_pairwise_kl': np.min(kls),
        'max_pairwise_kl': np.max(kls),
        # 标准化: KL > 1.0 认为是好的diversity
        'diversity_score': min(1.0, np.mean(kls)),
    }
```

### 7.7 超参数调优建议

| 超参数 | 推荐范围 | 说明 |
|--------|---------|------|
| `balance_weight` | 0.05 - 0.2 | 太大可能导致过度均匀routing |
| `diversity_weight` | 0.01 - 0.1 | 从小开始，逐步增加 |
| `diversity_type` | 'output_kl', 'param_kl', 'both' | output_kl通常更有效 |
| `z_loss_weight` | 0.001 - 0.01 | 训练稳定性 |

调优策略:
1. 先只用balance_weight训练baseline
2. 观察expert outputs的KL，如果太低(< 0.1)添加diversity loss
3. 从小的diversity_weight开始(0.01)，观察效果
4. 如果expert过于diverse导致性能下降，减小weight

### 7.8 预期效果

使用KL散度diversity loss后，预期:

1. **更高的Pairwise KL**: Expert outputs之间的KL散度应该从~0.01提升到>0.5
2. **更好的Specialization**: Specialization score应该从0.4-0.5提升到>0.7
3. **保持Task Performance**: 在增加diversity的同时，task accuracy不应该下降太多
4. **更稳定的Routing**: 路由模式应该更加清晰，每个expert专注于特定类型

### 7.9 实验检查清单

- [ ] 实现ExpertDiversityLoss
- [ ] 修改ExpertLoRACollection添加get_all_expert_outputs
- [ ] 修改MoE loss计算集成diversity
- [ ] 添加diversity metrics logging
- [ ] 运行ablation study (with/without diversity loss)
- [ ] 对比balance_only vs balance+diversity
- [ ] 分析pairwise KL随训练的变化趋势

---

## 8. 完成 MoE 实施

恭喜！你已经阅读完所有 MoE 实施文档。

### 实施检查清单

- [ ] 阅读 00_overview.md - 理解项目目标
- [ ] 阅读 01_architecture.md - 理解系统架构
- [ ] 实现 Router (02_router_implementation.md)
- [ ] 实现 Expert LoRAs (03_expert_lora.md)
- [ ] 集成到 verl 训练框架 (04_training_integration.md)
- [ ] 集成 vLLM 推理 (05_vllm_inference.md)
- [ ] 准备数据管道 (06_data_pipeline.md)
- [ ] 运行评估分析 (07_analysis_metrics.md)

### 快速开始命令

```bash
# 1. 准备数据
python scripts/prepare_moe_data.py \
    --input datasets/ui_s1_train.jsonl \
    --output datasets/ui_s1_train_moe.jsonl

# 2. 运行训练
sbatch train/train_moe.slurm

# 3. 评估结果
python scripts/evaluate_moe.py evaluate \
    --checkpoint checkpoints/moe_experiment \
    --test_data datasets/ui_s1_val_moe.jsonl \
    --output outputs/moe_evaluation

# 4. 对比 Single LoRA
python scripts/evaluate_moe.py compare \
    --moe_results outputs/moe_evaluation/evaluation_report.json \
    --single_results outputs/single_evaluation/evaluation_report.json \
    --output outputs/comparison
```
