# GUI-360 Full-Parameter SFT with LlamaFactory

## 概述

使用 LlamaFactory 对 Qwen2.5-VL-7B-Instruct 进行 GUI-360 数据集的全参数微调 (Full-Parameter Fine-Tuning)。

- **模型**: Qwen2.5-VL-7B-Instruct (本地 checkpoint)
- **数据**: GUI-360 (101,800 samples, ShareGPT format)
- **方法**: Full-parameter SFT + DeepSpeed ZeRO-3
- **环境**: conda env `qwen3-eval`

## 文件结构

```
train_GUI_360/llamafactory/
├── data/
│   ├── dataset_info.json              # LlamaFactory 数据集配置
│   ├── gui360_train.json              # 训练集 (prepare_data.py 生成)
│   └── gui360_val.json                # 验证集 (prepare_data.py 生成)
├── ds_z3_config.json                  # DeepSpeed ZeRO-3 配置
├── prepare_data.py                    # 数据转换脚本
├── qwen25vl_gui360_full_sft.yaml      # 训练配置
├── train_gui360_llamafactory.slurm    # SLURM 训练脚本
└── README.md                          # 本文档
```

## Step 1: 安装 LlamaFactory

```bash
conda activate qwen3-eval

cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

验证安装:
```bash
llamafactory-cli version
```

## Step 2: 准备数据

源数据 `training_data.json` 已经是 ShareGPT 格式 (`from`/`value`, `human`/`gpt`)，只需转换图片路径为绝对路径。

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1
python train_GUI_360/llamafactory/prepare_data.py
```

默认参数:
- 输入: `datasets/GUI-360/processed_data/action_prediction_train_resize/training_data.json`
- 输出: `train_GUI_360/llamafactory/data/gui360_train.json` + `gui360_val.json`
- 验证集: 100 samples

可选参数:
```bash
python train_GUI_360/llamafactory/prepare_data.py \
    --validate-images \       # 检查图片文件是否存在
    --val-size 200 \          # 调整验证集大小
    --max-samples 1000        # 限制样本数 (用于测试)
```

## Step 3: 提交训练

```bash
sbatch train_GUI_360/llamafactory/train_gui360_llamafactory.slurm
```

SLURM 脚本会自动:
1. 检查并准备数据 (如果 `gui360_train.json` 不存在)
2. 启动 2 节点 × 4 GPU 的分布式训练

## 训练配置说明

| 参数 | 值 | 说明 |
|---|---|---|
| `model_name_or_path` | `Qwen2.5-VL-7B-Instruct` | 本地 checkpoint |
| `template` | `qwen2_vl` | LlamaFactory 中 Qwen2.5-VL 的模板名 |
| `finetuning_type` | `full` | 全参数微调 |
| `freeze_vision_tower` | `true` | 冻结视觉编码器 |
| `freeze_multi_modal_projector` | `true` | 冻结多模态投影层 |
| `freeze_language_model` | `false` | 训练语言模型部分 |
| `deepspeed` | ZeRO-3 | 7B 全参数必需 |
| `cutoff_len` | 8192 | 最大序列长度 |
| `learning_rate` | 1e-5 | |
| `num_train_epochs` | 2 | |
| `per_device_train_batch_size` | 1 | |
| `gradient_accumulation_steps` | 32 | |
| **effective global batch size** | **1 × 32 × 8 GPUs = 256** | |
| `bf16` | `true` | |
| `save_steps` | 500 | |
| `eval_steps` | 500 | |

## dataset_info.json 格式说明

参考 [LlamaFactory data README](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md):

```json
{
  "gui360_train": {
    "file_name": "gui360_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
```

数据样本格式:
```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nYou are a helpful assistant..."},
    {"from": "gpt", "value": "<tool_call>\n{\"function\": \"click\", ...}\n</tool_call>"}
  ],
  "images": ["/absolute/path/to/screenshot.png"]
}
```

## 注意事项

1. 当前使用的是**无 thinking/reasoning** 的数据版本。如需带 thinking，需要从 raw JSONL 另行转换。
2. `<image>` token 数量必须与 `images` 数组长度一致 (每个 sample 1 张图)。
3. Vision tower 和 projector 冻结，只训练 language model 部分 (与 LlamaFactory 官方 VL 示例一致)。
4. 训练输出保存到 `train_GUI_360/llamafactory/output/gui360_full_sft/`。
