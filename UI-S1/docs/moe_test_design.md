# MoE Tool Agent Pilot Experiment: Technical Architecture

## 1. Research Question

**核心问题**：MoE 结构能否让 expert LoRAs 自动分化，各自专注于不同类型的 GUI instructions？

**为什么重要**：
- 如果分化有效 → MoE 比单一 LoRA 更有效率（相同参数量，更好性能）
- 如果分化无效 → 不需要 MoE，简化架构

---

## 2. Technical Design Decisions

### 2.1 MoE 核心机制

```
┌─────────────────────────────────────────────────────────────────┐
│                     MoE Forward Pass                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input: (screenshot, instruction)                              │
│                    │                                            │
│                    ▼                                            │
│   ┌────────────────────────────────────┐                       │
│   │         Base Model Encoding         │                       │
│   │   features = base_model(input)      │                       │
│   └────────────────┬───────────────────┘                       │
│                    │                                            │
│        ┌───────────┴───────────┐                               │
│        │                       │                               │
│        ▼                       ▼                               │
│   ┌─────────┐           ┌───────────────────────────────┐     │
│   │ Router  │           │      Expert LoRAs              │     │
│   │         │           │  ┌───┐ ┌───┐ ┌───┐ ┌───┐      │     │
│   │ Linear  │──────────→│  │E0 │ │E1 │ │E2 │ │E3 │      │     │
│   │ → Softmax│  weights  │  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘      │     │
│   └─────────┘           │    │     │     │     │        │     │
│                         │    └──┬──┴──┬──┴──┬──┘        │     │
│                         │       │  weighted sum          │     │
│                         │       ▼                        │     │
│                         │   combined_features            │     │
│                         └───────────────────────────────┘     │
│                                    │                           │
│                                    ▼                           │
│                            Output Heads                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键设计选择

| 设计点 | 选项 | 我们的选择 | 原因 |
|--------|------|-----------|------|
| **Router 输入** | Token-level vs Sequence-level | Sequence-level (pooled) | 简单；instruction 级别的路由 |
| **Routing 类型** | Soft (weighted sum) vs Hard (top-k) | Top-k (k=1 or 2) | 更清晰的分化；易于分析 |
| **Expert 结构** | 独立 LoRA vs 共享部分 | 独立 LoRA | 最大分化潜力 |
| **Load Balance** | 无 vs Auxiliary loss | Auxiliary loss | 防止坍塌 |
| **Expert 数量** | 2/4/8 | 4 (与 instruction 类型对应) | 便于验证 |

### 2.3 Router 设计

**问题**：Router 应该基于什么信息来决定路由？

```python
# Option A: 只基于 instruction text (不看 screenshot)
# 优点: 简单，routing 可解释
# 缺点: 可能忽略 visual context

class TextOnlyRouter(nn.Module):
    def __init__(self, text_dim, num_experts):
        self.proj = nn.Linear(text_dim, num_experts)
    
    def forward(self, text_features):
        return F.softmax(self.proj(text_features.mean(dim=1)), dim=-1)


# Option B: 基于 instruction + screenshot
# 优点: 更多信息
# 缺点: 可能过拟合；routing 不可解释

class MultimodalRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        self.proj = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, fused_features):
        return F.softmax(self.proj(fused_features.mean(dim=1)), dim=-1)


# 我们的选择: Option A (Text-Only Router)
# 原因:
# 1. Pilot 实验目标是验证 expert 按 instruction 类型分化
# 2. Text-only 使得 routing 完全可解释
# 3. 如果 text-only 有效，说明 instruction 本身足以决定需要什么 expert
```

### 2.4 Load Balancing

**问题**：如何防止所有 samples 都路由到一个 expert？

```python
# Auxiliary Loss: 让 expert 使用率尽量均匀

def compute_load_balance_loss(routing_weights, num_experts):
    """
    routing_weights: [batch_size, num_experts]
    
    目标: 每个 expert 处理约 1/num_experts 的 samples
    """
    # 方法 1: 简单 MSE
    avg_routing = routing_weights.mean(dim=0)  # [num_experts]
    target = torch.ones_like(avg_routing) / num_experts
    loss = F.mse_loss(avg_routing, target)
    
    # 方法 2: Switch Transformer 风格
    # f_i = fraction of tokens routed to expert i
    # P_i = average routing probability for expert i
    # loss = num_experts * sum(f_i * P_i)
    
    return loss
```

**关键超参数**：`balance_weight`

```
balance_weight = 0.01  → 可能坍塌 (一个 expert 主导)
balance_weight = 0.1   → 适中 (推荐起点)
balance_weight = 1.0   → 可能强制均匀 (失去分化意义)
```

---

## 3. Model Architecture

### 3.1 完整模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

class LoRAExpert(nn.Module):
    """
    单个 Expert: 一个 LoRA adapter
    """
    def __init__(self, base_model, r=16, alpha=32):
        super().__init__()
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=['q_proj', 'v_proj'],  # 简化：只适配 attention
            lora_dropout=0.05,
        )
        
        # 注意：这里不直接 wrap base_model
        # 而是创建 LoRA 参数，手动应用
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        
        for name, module in base_model.named_modules():
            if any(t in name for t in ['q_proj', 'v_proj']):
                in_features = module.in_features
                out_features = module.out_features
                
                self.lora_A[name.replace('.', '_')] = nn.Parameter(
                    torch.randn(r, in_features) * 0.01
                )
                self.lora_B[name.replace('.', '_')] = nn.Parameter(
                    torch.zeros(out_features, r)
                )
        
        self.scaling = alpha / r
    
    def forward(self, base_output, module_name):
        """
        应用 LoRA delta 到 base output
        """
        key = module_name.replace('.', '_')
        if key in self.lora_A:
            # LoRA: output = base_output + (x @ A^T @ B^T) * scaling
            # 简化：直接在 base_output 上加 delta
            delta = self.lora_B[key] @ self.lora_A[key]
            return base_output + delta * self.scaling
        return base_output


class MoEToolAgent(nn.Module):
    """
    Mixture of Experts Tool Agent
    
    Architecture:
    - Base VLM (frozen)
    - Router (learned): instruction → expert weights
    - K Expert LoRAs (learned): specialized adapters
    - Output heads (learned): action_type + exec_score
    """
    
    def __init__(
        self, 
        base_model,
        num_experts: int = 4,
        expert_r: int = 16,
        expert_alpha: int = 32,
        top_k: int = 1,
        router_hidden: int = 256,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        hidden_size = base_model.config.hidden_size
        
        # Router: text features → expert weights
        # 使用 instruction 的 text embedding 来 route
        self.router = nn.Sequential(
            nn.Linear(hidden_size, router_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(router_hidden, num_experts),
        )
        
        # Expert LoRAs
        self.experts = nn.ModuleList([
            self._create_expert_lora(hidden_size, expert_r, expert_alpha)
            for _ in range(num_experts)
        ])
        
        # Output heads
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),  # click, type, navigate, scroll
        )
        
        self.exec_score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def _create_expert_lora(self, hidden_size, r, alpha):
        """创建一个 expert 的 LoRA 参数"""
        return nn.ModuleDict({
            'down': nn.Linear(hidden_size, r, bias=False),
            'up': nn.Linear(r, hidden_size, bias=False),
            'scaling': nn.Parameter(torch.tensor(alpha / r)),
        })
    
    def _apply_expert(self, features, expert_idx):
        """应用指定 expert 的 LoRA"""
        expert = self.experts[expert_idx]
        delta = expert['up'](expert['down'](features)) * expert['scaling']
        return features + delta
    
    def forward(self, screenshot, instruction_text):
        """
        Forward pass with MoE routing.
        
        Args:
            screenshot: [B, C, H, W] 图像
            instruction_text: List[str] 指令文本
        
        Returns:
            action_type_logits: [B, 4]
            exec_score: [B, 1]
            routing_weights: [B, num_experts]
            expert_indices: [B, top_k]
        """
        # 1. Base model encoding
        # 假设 base_model 返回 (text_features, image_features, fused_features)
        outputs = self.base_model(
            pixel_values=screenshot,
            input_ids=self._tokenize(instruction_text),
        )
        
        # 使用 last hidden state 的 pooled 表示
        features = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]
        
        # 2. Router: 计算 expert weights
        router_logits = self.router(features)  # [B, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 3. Top-k selection
        top_k_weights, top_k_indices = routing_weights.topk(self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # renormalize
        
        # 4. Apply selected experts
        batch_size = features.size(0)
        expert_features = torch.zeros_like(features)
        
        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = top_k_indices[b, k].item()
                weight = top_k_weights[b, k]
                expert_out = self._apply_expert(features[b:b+1], expert_idx)
                expert_features[b:b+1] += weight * expert_out
        
        # 5. Output heads
        action_type_logits = self.action_type_head(expert_features)
        exec_score = self.exec_score_head(expert_features)
        
        return {
            'action_type_logits': action_type_logits,
            'exec_score': exec_score,
            'routing_weights': routing_weights,
            'top_k_indices': top_k_indices,
            'top_k_weights': top_k_weights,
        }
    
    def _tokenize(self, texts):
        """Tokenize instruction texts"""
        # 使用 base_model 的 tokenizer
        return self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).input_ids
```

### 3.2 Baseline: Single Expert

```python
class SingleExpertToolAgent(nn.Module):
    """
    Baseline: 单一 LoRA，参数量与 MoE 相同
    
    MoE: 4 experts × r=16 = 64 total rank
    Single: 1 expert × r=64 = 64 total rank
    """
    
    def __init__(self, base_model, r=64, alpha=128):
        super().__init__()
        
        self.base_model = base_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        hidden_size = base_model.config.hidden_size
        
        # Single LoRA expert
        self.expert = nn.ModuleDict({
            'down': nn.Linear(hidden_size, r, bias=False),
            'up': nn.Linear(r, hidden_size, bias=False),
            'scaling': nn.Parameter(torch.tensor(alpha / r)),
        })
        
        # Output heads (same as MoE)
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )
        
        self.exec_score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, screenshot, instruction_text):
        # 1. Base model encoding
        outputs = self.base_model(
            pixel_values=screenshot,
            input_ids=self._tokenize(instruction_text),
        )
        features = outputs.last_hidden_state.mean(dim=1)
        
        # 2. Apply single expert
        delta = self.expert['up'](self.expert['down'](features)) * self.expert['scaling']
        expert_features = features + delta
        
        # 3. Output heads
        action_type_logits = self.action_type_head(expert_features)
        exec_score = self.exec_score_head(expert_features)
        
        return {
            'action_type_logits': action_type_logits,
            'exec_score': exec_score,
        }
```

---

## 4. Training Design

### 4.1 Loss Functions

```python
class MoETrainingLoss(nn.Module):
    """
    MoE 训练的完整 loss
    """
    
    def __init__(self, num_experts, balance_weight=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.balance_weight = balance_weight
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from MoEToolAgent.forward()
            targets: dict with 'action_type', 'exec_score'
        
        Returns:
            total_loss, loss_dict
        """
        # Loss 1: Action type classification
        loss_action = F.cross_entropy(
            outputs['action_type_logits'],
            targets['action_type']
        )
        
        # Loss 2: exec_score regression
        loss_exec = F.mse_loss(
            outputs['exec_score'].squeeze(),
            targets['exec_score']
        )
        
        # Loss 3: Load balancing
        routing_weights = outputs['routing_weights']
        avg_routing = routing_weights.mean(dim=0)
        target_balance = torch.ones(self.num_experts, device=avg_routing.device) / self.num_experts
        loss_balance = F.mse_loss(avg_routing, target_balance)
        
        # Total loss
        total_loss = loss_action + loss_exec + self.balance_weight * loss_balance
        
        return total_loss, {
            'loss_action': loss_action.item(),
            'loss_exec': loss_exec.item(),
            'loss_balance': loss_balance.item(),
            'total_loss': total_loss.item(),
        }
```

### 4.2 Training Loop

```python
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    
    epoch_stats = defaultdict(list)
    routing_records = []  # 记录 routing 用于分析
    
    for batch in tqdm(dataloader):
        # Move to device
        screenshots = batch['screenshot'].to(device)
        instructions = batch['instruction']
        targets = {
            'action_type': batch['action_type'].to(device),
            'exec_score': batch['exec_score'].to(device),
        }
        
        # Forward
        outputs = model(screenshots, instructions)
        
        # Loss
        loss, loss_dict = loss_fn(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record stats
        for k, v in loss_dict.items():
            epoch_stats[k].append(v)
        
        # Record routing for analysis
        routing_records.append({
            'instruction_types': batch['instruction_type'],  # ground truth type
            'routing_weights': outputs['routing_weights'].detach().cpu(),
            'top_k_indices': outputs['top_k_indices'].detach().cpu(),
        })
    
    # Aggregate stats
    avg_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
    
    return avg_stats, routing_records
```

### 4.3 Evaluation

```python
def evaluate(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_targets = []
    all_routing = []
    all_instruction_types = []
    
    with torch.no_grad():
        for batch in dataloader:
            screenshots = batch['screenshot'].to(device)
            instructions = batch['instruction']
            
            outputs = model(screenshots, instructions)
            
            # Action type predictions
            preds = outputs['action_type_logits'].argmax(dim=-1).cpu()
            targets = batch['action_type']
            
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            all_routing.append(outputs['routing_weights'].cpu())
            all_instruction_types.extend(batch['instruction_type'])
    
    # Compute metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    
    # Per-type accuracy
    type_accuracy = {}
    for t in ['click', 'type', 'navigate', 'scroll']:
        mask = np.array(all_instruction_types) == t
        if mask.sum() > 0:
            type_accuracy[t] = np.mean(
                np.array(all_preds)[mask] == np.array(all_targets)[mask]
            )
    
    # Routing analysis
    all_routing = torch.cat(all_routing, dim=0)
    
    return {
        'accuracy': accuracy,
        'type_accuracy': type_accuracy,
        'routing_weights': all_routing,
        'instruction_types': all_instruction_types,
    }
```

---

## 5. Analysis Methods

### 5.1 Expert Specialization Analysis

```python
def analyze_expert_specialization(routing_weights, instruction_types, num_experts=4):
    """
    分析每个 expert 是否专注于特定类型的 instructions
    
    Args:
        routing_weights: [N, num_experts] 所有样本的 routing weights
        instruction_types: [N] 每个样本的 instruction 类型
    
    Returns:
        specialization_score: 分化程度 (0=随机, 1=完美分化)
        routing_matrix: [num_types, num_experts] 每种类型到每个 expert 的路由比例
    """
    types = ['click', 'type', 'navigate', 'scroll']
    num_types = len(types)
    
    # 构建 routing matrix
    # routing_matrix[i, j] = P(expert j | instruction type i)
    routing_matrix = np.zeros((num_types, num_experts))
    
    for i, t in enumerate(types):
        mask = np.array(instruction_types) == t
        if mask.sum() > 0:
            type_routing = routing_weights[mask].numpy()
            # 使用 hard routing (argmax) 来计算分布
            dominant_experts = type_routing.argmax(axis=1)
            for j in range(num_experts):
                routing_matrix[i, j] = (dominant_experts == j).mean()
    
    # Specialization score
    # 如果完美分化：每行应该有一个接近 1 的值
    # 计算每行的 max，然后取平均
    row_max = routing_matrix.max(axis=1)
    specialization_score = row_max.mean()
    
    # 额外：计算互信息 (Mutual Information)
    # 高互信息 = instruction type 和 expert 高度相关
    mi = compute_mutual_information(routing_matrix)
    
    return {
        'specialization_score': specialization_score,
        'routing_matrix': routing_matrix,
        'mutual_information': mi,
    }


def compute_mutual_information(routing_matrix):
    """
    计算 instruction type 和 expert 之间的互信息
    """
    # Normalize to joint probability
    p_joint = routing_matrix / routing_matrix.sum()
    
    # Marginals
    p_type = p_joint.sum(axis=1, keepdims=True)
    p_expert = p_joint.sum(axis=0, keepdims=True)
    
    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    with np.errstate(divide='ignore', invalid='ignore'):
        mi_matrix = p_joint * np.log(p_joint / (p_type * p_expert + 1e-10) + 1e-10)
        mi_matrix = np.nan_to_num(mi_matrix)
    
    return mi_matrix.sum()
```

### 5.2 Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_routing_matrix(routing_matrix, save_path=None):
    """
    可视化 routing matrix 为热力图
    """
    types = ['click', 'type', 'navigate', 'scroll']
    experts = [f'Expert {i}' for i in range(routing_matrix.shape[1])]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        routing_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=experts,
        yticklabels=types,
    )
    plt.title('Routing Pattern: P(Expert | Instruction Type)')
    plt.xlabel('Expert')
    plt.ylabel('Instruction Type')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_specialization_over_training(specialization_history, save_path=None):
    """
    可视化训练过程中 specialization score 的变化
    """
    plt.figure(figsize=(10, 4))
    plt.plot(specialization_history, marker='o')
    plt.axhline(y=0.25, color='r', linestyle='--', label='Random (0.25)')
    plt.axhline(y=1.0, color='g', linestyle='--', label='Perfect (1.0)')
    plt.xlabel('Epoch')
    plt.ylabel('Specialization Score')
    plt.title('Expert Specialization Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### 5.3 Statistical Tests

```python
from scipy import stats

def test_specialization_significance(routing_weights, instruction_types, num_permutations=1000):
    """
    检验 expert 分化是否显著（vs 随机 baseline）
    
    方法: Permutation test
    - H0: routing 与 instruction type 无关
    - H1: routing 与 instruction type 相关
    """
    # 计算真实的 specialization score
    real_result = analyze_expert_specialization(routing_weights, instruction_types)
    real_score = real_result['specialization_score']
    
    # Permutation: 打乱 instruction_types，重新计算
    permuted_scores = []
    for _ in range(num_permutations):
        shuffled_types = np.random.permutation(instruction_types)
        perm_result = analyze_expert_specialization(routing_weights, shuffled_types)
        permuted_scores.append(perm_result['specialization_score'])
    
    # P-value: 有多少 permutation 的 score >= real score
    p_value = (np.array(permuted_scores) >= real_score).mean()
    
    return {
        'real_score': real_score,
        'permuted_mean': np.mean(permuted_scores),
        'permuted_std': np.std(permuted_scores),
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
```

---

## 6. Data Pipeline

### 6.1 Dataset Design

```python
from torch.utils.data import Dataset, DataLoader

class PilotGUIDataset(Dataset):
    """
    Pilot 实验数据集
    
    每个样本:
    - screenshot: 图像
    - instruction: 自然语言指令
    - instruction_type: 类型标签 (用于分析，不用于训练 MoE)
    - action_type: 动作类型 (训练目标)
    - exec_score: 可执行性分数 (训练目标)
    """
    
    def __init__(self, data_path, transform=None):
        self.data = self._load_data(data_path)
        self.transform = transform
        
        # Instruction type to action type mapping
        self.type_to_idx = {'click': 0, 'type': 1, 'navigate': 2, 'scroll': 3}
    
    def _load_data(self, data_path):
        # Load from preprocessed file
        return torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        screenshot = item['screenshot']
        if self.transform:
            screenshot = self.transform(screenshot)
        
        return {
            'screenshot': screenshot,
            'instruction': item['instruction'],
            'instruction_type': item['instruction_type'],
            'action_type': self.type_to_idx[item['instruction_type']],
            'exec_score': item['exec_score'],
        }


def create_dataloaders(train_path, val_path, batch_size=32):
    """创建 train/val dataloaders"""
    
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = PilotGUIDataset(train_path, transform=transform)
    val_dataset = PilotGUIDataset(val_path, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    return train_loader, val_loader
```

### 6.2 Data Preparation Script

```python
def prepare_pilot_data(source_datasets, output_dir, samples_per_type=1000):
    """
    从原始数据集中准备 pilot 数据
    
    目标: 4 类 × 1000 samples = 4000 samples
    """
    
    # 收集每类 instructions
    type_samples = {t: [] for t in ['click', 'type', 'navigate', 'scroll']}
    
    for dataset_name, dataset_path in source_datasets.items():
        print(f"Processing {dataset_name}...")
        
        raw_data = load_raw_dataset(dataset_path)
        
        for sample in tqdm(raw_data):
            instruction = sample['instruction']
            instruction_type = classify_instruction(instruction)
            
            if instruction_type in type_samples:
                if len(type_samples[instruction_type]) < samples_per_type:
                    type_samples[instruction_type].append({
                        'screenshot': sample['screenshot'],
                        'instruction': instruction,
                        'instruction_type': instruction_type,
                        'exec_score': 1.0 if sample.get('success', True) else 0.0,
                        'source': dataset_name,
                    })
    
    # 统计
    print("\nData statistics:")
    for t, samples in type_samples.items():
        print(f"  {t}: {len(samples)} samples")
    
    # 合并并打乱
    all_samples = []
    for samples in type_samples.values():
        all_samples.extend(samples)
    
    np.random.shuffle(all_samples)
    
    # 划分 train/val (90/10)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # 保存
    torch.save(train_samples, os.path.join(output_dir, 'train.pt'))
    torch.save(val_samples, os.path.join(output_dir, 'val.pt'))
    
    print(f"\nSaved {len(train_samples)} train, {len(val_samples)} val samples")


def classify_instruction(instruction):
    """
    基于关键词分类 instruction
    """
    instruction = instruction.lower()
    
    # Priority order matters
    if any(w in instruction for w in ['type', 'enter', 'input', 'write', 'fill']):
        return 'type'
    elif any(w in instruction for w in ['scroll', 'swipe', 'slide']):
        return 'scroll'
    elif any(w in instruction for w in ['navigate', 'go to', 'open', 'visit', 'access']):
        return 'navigate'
    elif any(w in instruction for w in ['click', 'tap', 'press', 'select', 'choose']):
        return 'click'
    
    return None  # 不属于四类之一
```

---

## 7. Experiment Configurations

### 7.1 Hyperparameters

```python
# Model configs
MODEL_CONFIGS = {
    'moe_4experts': {
        'num_experts': 4,
        'expert_r': 16,
        'expert_alpha': 32,
        'top_k': 1,
        'router_hidden': 256,
    },
    'moe_4experts_topk2': {
        'num_experts': 4,
        'expert_r': 16,
        'expert_alpha': 32,
        'top_k': 2,
        'router_hidden': 256,
    },
    'single_expert': {
        'r': 64,  # 4 * 16 = 64, same total params
        'alpha': 128,
    },
}

# Training configs
TRAIN_CONFIGS = {
    'default': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 10,
        'balance_weight': 0.1,
        'warmup_steps': 100,
    },
    'high_balance': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 10,
        'balance_weight': 0.5,  # 更强的 balance
        'warmup_steps': 100,
    },
    'low_balance': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 10,
        'balance_weight': 0.01,  # 更弱的 balance
        'warmup_steps': 100,
    },
}
```

### 7.2 Experiment Matrix

```python
EXPERIMENTS = [
    # Main comparison
    {'name': 'moe_default', 'model': 'moe_4experts', 'train': 'default'},
    {'name': 'single_default', 'model': 'single_expert', 'train': 'default'},
    
    # Ablation: balance weight
    {'name': 'moe_high_balance', 'model': 'moe_4experts', 'train': 'high_balance'},
    {'name': 'moe_low_balance', 'model': 'moe_4experts', 'train': 'low_balance'},
    
    # Ablation: top-k
    {'name': 'moe_topk2', 'model': 'moe_4experts_topk2', 'train': 'default'},
]
```

---

## 8. Expected Results & Success Criteria

### 8.1 Success Criteria

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Specialization Score | > 0.6 | 明显分化 (random = 0.25) |
| Accuracy (MoE) | > Accuracy (Single) | MoE 更有效 |
| P-value (specialization) | < 0.05 | 分化是显著的 |
| Expert Utilization | 每个 > 15% | 没有坍塌 |

### 8.2 Expected Outcomes

**理想结果**:
```
Routing Matrix:
           Expert 0  Expert 1  Expert 2  Expert 3
click        0.82      0.06      0.08      0.04
type         0.05      0.78      0.10      0.07
navigate     0.08      0.12      0.73      0.07
scroll       0.04      0.08      0.09      0.79

Specialization Score: 0.78
P-value: < 0.001

Accuracy:
  MoE: 85.2%
  Single: 81.7%
```

**不理想结果 (坍塌)**:
```
Routing Matrix:
           Expert 0  Expert 1  Expert 2  Expert 3
click        0.92      0.03      0.03      0.02
type         0.88      0.05      0.04      0.03
navigate     0.85      0.06      0.05      0.04
scroll       0.90      0.04      0.03      0.03

→ Expert 0 主导，需要增加 balance_weight
```

### 8.3 Interpretation Guide

| 观察 | 解释 | 行动 |
|------|------|------|
| 高分化 + MoE > Single | MoE 有效 ✓ | 继续完整实验 |
| 高分化 + MoE ≈ Single | 分化了但没帮助 | 分析为什么；可能任务太简单 |
| 低分化 + MoE > Single | 意外的好 | 分析 routing pattern |
| 低分化 + MoE ≈ Single | MoE 无效 | 检查 balance loss；可能需要更强的监督 |
| Expert 坍塌 | balance_weight 太小 | 增加 balance_weight |
| 完全均匀 routing | balance_weight 太大 | 减少 balance_weight |

---

## 9. Implementation Timeline

| Day | Task | Output |
|-----|------|--------|
| 1 | 数据准备 | `train.pt`, `val.pt` |
| 2 | 模型实现 | `moe_model.py`, `single_model.py` |
| 3 | Training loop | `train.py`, `evaluate.py` |
| 4 | 运行实验 | 5 个实验的 checkpoints |
| 5 | 分析 + 报告 | `analysis.ipynb`, `pilot_report.md` |

---

## 10. Code Repository Structure

```
pilot_moe_experiment/
├── configs/
│   ├── model_configs.yaml
│   └── train_configs.yaml
├── data/
│   ├── prepare_data.py
│   └── dataset.py
├── models/
│   ├── moe_tool_agent.py
│   ├── single_tool_agent.py
│   └── components.py
├── training/
│   ├── train.py
│   ├── loss.py
│   └── utils.py
├── analysis/
│   ├── specialization.py
│   ├── visualization.py
│   └── statistics.py
├── scripts/
│   ├── run_all_experiments.sh
│   └── generate_report.py
├── notebooks/
│   └── analysis.ipynb
└── README.md
```

---

## 11. Next Steps After Pilot

**如果 pilot 成功**:
1. 扩展到完整 Tool Agent (action sequence generation)
2. 加入 Planner 形成完整系统
3. 在 OSWorld 上评估

**如果 pilot 不成功**:
1. 分析失败原因
2. 尝试替代方案:
   - 更强的 routing supervision
   - 不同的 expert 架构
   - 简化为 2 experts
3. 如果仍不行，放弃 MoE，使用 single expert