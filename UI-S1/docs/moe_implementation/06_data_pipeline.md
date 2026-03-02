# 数据管道设计

## 1. 现有数据管道概览

```
verl/utils/dataset/
├── rl_dataset.py              # TrajDataset - RL 数据集
├── multiturn_sft_dataset.py   # 多轮 SFT 数据集
└── vision_utils.py            # 视觉处理工具

x/data/agent/
├── base.py                    # BaseFormatAbs - 基础数据处理
├── mobile_use.py              # 移动端数据
└── space/
    └── std_space.py           # 标准动作空间
```

### 1.1 现有数据格式

```json
// datasets/ui_s1_train.jsonl 示例
{
  "id": "sample_001",
  "trajectory": [
    {
      "observation": {
        "screenshot": "path/to/image.png",
        "app_info": "..."
      },
      "action": {
        "type": "click",
        "target": "search_button",
        "coordinates": [120, 450]
      },
      "instruction": "Click on the search button"
    },
    // ... more steps
  ]
}
```

---

## 2. MoE 数据增强

### 2.1 添加 Instruction Type 标签

MoE 训练需要知道每个样本的 instruction 类型用于分析（不用于训练）。

```python
# x/data/agent/moe_utils.py

from typing import List, Tuple, Optional
import re


# Instruction 类型定义
INSTRUCTION_TYPES = ['click', 'type', 'navigate', 'scroll', 'other']


def classify_instruction(instruction: str) -> str:
    """
    根据关键词分类 instruction

    Args:
        instruction: 自然语言指令

    Returns:
        instruction_type: 'click', 'type', 'navigate', 'scroll', or 'other'
    """
    instruction = instruction.lower().strip()

    # 优先级顺序很重要
    classification_rules = [
        # Type/Input 类
        ('type', [
            r'\btype\b', r'\benter\b', r'\binput\b', r'\bwrite\b',
            r'\bfill\b', r'\bsearch for\b', r'\bsend\b.*message',
            r'\bcompose\b', r'\bedit\b.*text',
        ]),

        # Scroll/Swipe 类
        ('scroll', [
            r'\bscroll\b', r'\bswipe\b', r'\bslide\b', r'\bflick\b',
            r'\bpull\b.*down', r'\bpull\b.*up', r'\brefresh\b',
        ]),

        # Navigate 类
        ('navigate', [
            r'\bnavigate\b', r'\bgo to\b', r'\bopen\b', r'\bvisit\b',
            r'\baccess\b', r'\blaunch\b', r'\bswitch to\b',
            r'\breturn\b', r'\bback\b', r'\bhome\b',
        ]),

        # Click/Tap 类 (最后，因为最通用)
        ('click', [
            r'\bclick\b', r'\btap\b', r'\bpress\b', r'\bselect\b',
            r'\bchoose\b', r'\btoggle\b', r'\benable\b', r'\bdisable\b',
            r'\bcheck\b', r'\buncheck\b',
        ]),
    ]

    for type_name, patterns in classification_rules:
        for pattern in patterns:
            if re.search(pattern, instruction):
                return type_name

    return 'other'


def classify_action_type(action: dict) -> str:
    """
    根据 action 结构分类

    这是一个 fallback，当 instruction 分类不确定时使用
    """
    action_type = action.get('type', '').lower()

    type_mapping = {
        'click': 'click',
        'tap': 'click',
        'press': 'click',
        'type': 'type',
        'input': 'type',
        'scroll': 'scroll',
        'swipe': 'scroll',
        'navigate': 'navigate',
        'open': 'navigate',
        'back': 'navigate',
    }

    return type_mapping.get(action_type, 'other')


def add_instruction_type_to_sample(sample: dict) -> dict:
    """
    为单个样本添加 instruction_type 标签

    Args:
        sample: 原始数据样本

    Returns:
        添加了 instruction_type 的样本
    """
    instruction = sample.get('instruction', '')

    # 尝试从 instruction 文本分类
    instruction_type = classify_instruction(instruction)

    # 如果分类为 'other'，尝试从 action 分类
    if instruction_type == 'other' and 'action' in sample:
        instruction_type = classify_action_type(sample['action'])

    sample['instruction_type'] = instruction_type
    return sample


def get_type_statistics(samples: List[dict]) -> dict:
    """
    统计数据集中各类型的分布

    Returns:
        {type: count, ...}
    """
    from collections import Counter

    types = [s.get('instruction_type', 'unknown') for s in samples]
    return dict(Counter(types))
```

### 2.2 数据预处理脚本

```python
# scripts/prepare_moe_data.py

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from x.data.agent.moe_utils import (
    add_instruction_type_to_sample,
    get_type_statistics,
    INSTRUCTION_TYPES,
)


def process_jsonl(input_path: str, output_path: str, balance: bool = False):
    """
    处理 JSONL 数据文件，添加 instruction_type

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        balance: 是否平衡各类型的样本数量
    """
    print(f"Processing {input_path}...")

    samples = []
    with open(input_path, 'r') as f:
        for line in tqdm(f, desc="Loading"):
            sample = json.loads(line)

            # 处理 trajectory 格式
            if 'trajectory' in sample:
                for step in sample['trajectory']:
                    step_sample = {
                        'id': f"{sample['id']}_{len(samples)}",
                        'instruction': step.get('instruction', ''),
                        'action': step.get('action', {}),
                        'observation': step.get('observation', {}),
                    }
                    step_sample = add_instruction_type_to_sample(step_sample)
                    samples.append(step_sample)
            else:
                sample = add_instruction_type_to_sample(sample)
                samples.append(sample)

    # 统计
    stats = get_type_statistics(samples)
    print("\nType distribution:")
    for t, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = count / len(samples) * 100
        print(f"  {t}: {count} ({pct:.1f}%)")

    # 可选：平衡数据
    if balance:
        samples = balance_samples(samples, stats)
        new_stats = get_type_statistics(samples)
        print("\nBalanced distribution:")
        for t, count in sorted(new_stats.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")

    # 保存
    print(f"\nSaving {len(samples)} samples to {output_path}...")
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print("Done!")


def balance_samples(samples: list, stats: dict, target_per_type: int = None) -> list:
    """
    平衡各类型的样本数量

    策略：
    1. 下采样多数类
    2. 上采样少数类（可选）
    """
    import random

    # 按类型分组
    type_samples = defaultdict(list)
    for s in samples:
        type_samples[s['instruction_type']].append(s)

    # 确定目标数量
    if target_per_type is None:
        # 使用中位数
        counts = [len(v) for v in type_samples.values() if len(v) > 0]
        target_per_type = sorted(counts)[len(counts) // 2]

    print(f"\nTarget samples per type: {target_per_type}")

    # 平衡
    balanced = []
    for type_name, type_samples_list in type_samples.items():
        if len(type_samples_list) > target_per_type:
            # 下采样
            selected = random.sample(type_samples_list, target_per_type)
        else:
            # 保持原样 (或可以上采样)
            selected = type_samples_list

        balanced.extend(selected)

    random.shuffle(balanced)
    return balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--balance", action="store_true", help="Balance sample types")
    args = parser.parse_args()

    process_jsonl(args.input, args.output, args.balance)


if __name__ == "__main__":
    main()
```

---

## 3. MoE Dataset 类

### 3.1 扩展 TrajDataset

```python
# verl/utils/dataset/moe_dataset.py

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Callable
import json
from pathlib import Path

from verl.utils.dataset.rl_dataset import TrajDataset
from x.data.agent.moe_utils import classify_instruction, create_instruction_mask


class MoETrajDataset(TrajDataset):
    """
    支持 MoE 的轨迹数据集

    扩展:
    1. 添加 instruction_type 标签
    2. 添加 instruction_mask (用于 feature extraction)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor=None,
        include_instruction_type: bool = True,
        **kwargs,
    ):
        """
        Args:
            data_path: 数据文件路径
            tokenizer: Tokenizer
            processor: VLM processor (for image processing)
            include_instruction_type: 是否包含 instruction type
        """
        super().__init__(data_path, tokenizer, processor, **kwargs)

        self.include_instruction_type = include_instruction_type
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本

        Returns:
            Dict with:
                - input_ids
                - attention_mask
                - pixel_values (if available)
                - labels
                - instruction_type (if include_instruction_type)
                - instruction_mask (for router)
        """
        # 获取基类的输出
        sample = super().__getitem__(idx)

        # 添加 instruction_type
        if self.include_instruction_type:
            raw_sample = self._get_raw_sample(idx)
            instruction = raw_sample.get('instruction', '')

            # 分类
            if 'instruction_type' in raw_sample:
                sample['instruction_type'] = raw_sample['instruction_type']
            else:
                sample['instruction_type'] = classify_instruction(instruction)

        # 添加 instruction_mask
        sample['instruction_mask'] = self._create_instruction_mask(
            sample['input_ids'],
            sample.get('raw_instruction', ''),
        )

        return sample

    def _get_raw_sample(self, idx: int) -> dict:
        """获取原始样本（未处理）"""
        # 实现取决于数据存储方式
        if hasattr(self, 'raw_data'):
            return self.raw_data[idx]
        return {}

    def _create_instruction_mask(
        self,
        input_ids: torch.Tensor,
        instruction: str,
    ) -> torch.Tensor:
        """
        创建 instruction 部分的 mask

        标记哪些 tokens 属于 instruction 文本
        """
        # 使用 tokenizer 找到 instruction 的位置
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # 简化实现：找到 instruction tokens
        instruction_ids = self.tokenizer.encode(
            instruction,
            add_special_tokens=False,
        )

        input_list = input_ids.tolist()
        instr_len = len(instruction_ids)

        # 滑动窗口查找
        for i in range(len(input_list) - instr_len + 1):
            if input_list[i:i+instr_len] == instruction_ids:
                mask[i:i+instr_len] = True
                break

        return mask


def collate_fn_moe(batch: List[Dict]) -> Dict:
    """
    MoE 数据的 collate 函数

    处理:
    1. Tensor padding
    2. instruction_type 聚合
    3. instruction_mask padding
    """
    from torch.nn.utils.rnn import pad_sequence

    # 分离 tensor 和 non-tensor 数据
    tensor_keys = ['input_ids', 'attention_mask', 'labels', 'instruction_mask']
    non_tensor_keys = ['instruction_type']

    result = {}

    # 处理 tensor 数据
    for key in tensor_keys:
        if key in batch[0]:
            tensors = [item[key] for item in batch]
            if key == 'labels':
                # Labels 通常使用 -100 padding
                result[key] = pad_sequence(tensors, batch_first=True, padding_value=-100)
            else:
                result[key] = pad_sequence(tensors, batch_first=True, padding_value=0)

    # 处理 non-tensor 数据
    for key in non_tensor_keys:
        if key in batch[0]:
            result[key] = [item[key] for item in batch]

    # 处理 pixel_values (如果存在)
    if 'pixel_values' in batch[0]:
        result['pixel_values'] = torch.stack([item['pixel_values'] for item in batch])

    return result
```

---

## 4. 数据验证工具

### 4.1 数据质量检查

```python
# scripts/validate_moe_data.py

import json
import argparse
from collections import Counter
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np


def validate_data(data_path: str) -> Dict:
    """
    验证 MoE 数据质量

    检查:
    1. instruction_type 分布
    2. 空 instruction 比例
    3. 样本完整性
    """
    issues = []
    type_counts = Counter()
    total = 0

    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total += 1
            try:
                sample = json.loads(line)

                # 检查必要字段
                if 'instruction' not in sample:
                    issues.append(f"Line {line_num}: Missing 'instruction'")

                if 'instruction_type' not in sample:
                    issues.append(f"Line {line_num}: Missing 'instruction_type'")
                else:
                    type_counts[sample['instruction_type']] += 1

                # 检查空 instruction
                if not sample.get('instruction', '').strip():
                    issues.append(f"Line {line_num}: Empty instruction")

            except json.JSONDecodeError:
                issues.append(f"Line {line_num}: Invalid JSON")

    # 生成报告
    report = {
        'total_samples': total,
        'type_distribution': dict(type_counts),
        'issues': issues[:100],  # 只显示前 100 个问题
        'issue_count': len(issues),
    }

    return report


def visualize_distribution(type_counts: Dict[str, int], output_path: str = None):
    """可视化 instruction type 分布"""
    types = list(type_counts.keys())
    counts = list(type_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(types, counts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6'])

    plt.xlabel('Instruction Type')
    plt.ylabel('Count')
    plt.title('Instruction Type Distribution')

    # 添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            str(count),
            ha='center',
            va='bottom',
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved distribution plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Data file path")
    parser.add_argument("--plot", type=str, help="Output path for distribution plot")
    args = parser.parse_args()

    report = validate_data(args.data)

    print("\n=== Data Validation Report ===")
    print(f"Total samples: {report['total_samples']}")
    print(f"Issues found: {report['issue_count']}")

    print("\nType distribution:")
    for t, count in sorted(report['type_distribution'].items(), key=lambda x: -x[1]):
        pct = count / report['total_samples'] * 100
        print(f"  {t}: {count} ({pct:.1f}%)")

    if report['issues']:
        print(f"\nFirst {min(10, len(report['issues']))} issues:")
        for issue in report['issues'][:10]:
            print(f"  - {issue}")

    if args.plot:
        visualize_distribution(report['type_distribution'], args.plot)


if __name__ == "__main__":
    main()
```

---

## 5. 数据加载配置

### 5.1 YAML 配置

```yaml
# examples/qwen_gui_moe/config/data_config.yaml

data:
  # 训练数据
  train:
    path: datasets/ui_s1_train_moe.jsonl
    dataset_class: verl.utils.dataset.moe_dataset.MoETrajDataset
    include_instruction_type: true

  # 验证数据
  validation:
    path: datasets/ui_s1_val_moe.jsonl
    dataset_class: verl.utils.dataset.moe_dataset.MoETrajDataset
    include_instruction_type: true

  # 数据加载参数
  loader:
    batch_size: 32
    num_workers: 4
    shuffle: true
    collate_fn: verl.utils.dataset.moe_dataset.collate_fn_moe

  # 预处理
  preprocessing:
    max_length: 2048
    truncation: true
    padding: max_length
```

---

## 6. 使用流程

### 6.1 完整数据准备流程

```bash
# 1. 添加 instruction_type 标签
python scripts/prepare_moe_data.py \
    --input datasets/ui_s1_train.jsonl \
    --output datasets/ui_s1_train_moe.jsonl

# 2. 验证数据
python scripts/validate_moe_data.py \
    --data datasets/ui_s1_train_moe.jsonl \
    --plot outputs/type_distribution.png

# 3. (可选) 平衡数据
python scripts/prepare_moe_data.py \
    --input datasets/ui_s1_train.jsonl \
    --output datasets/ui_s1_train_balanced.jsonl \
    --balance

# 4. 划分训练/验证集
python scripts/split_data.py \
    --input datasets/ui_s1_train_moe.jsonl \
    --train datasets/ui_s1_train_split.jsonl \
    --val datasets/ui_s1_val_split.jsonl \
    --val_ratio 0.1
```

---

## 7. 下一步

完成数据管道后：
- [07_analysis_metrics.md](./07_analysis_metrics.md) - 分析与评估指标
