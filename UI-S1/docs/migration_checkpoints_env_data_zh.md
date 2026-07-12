# UI-S1 当前研究线迁移指南：Checkpoint、uv环境与数据集

> 目标：把当前 **multi-agent revision → Pass@8 selector → selector-to-training purity bridge** 研究线迁移到另一台机器或另一个代码仓库，并保持模型、数据、split、prompt和评估结果可复现。

## 0. 先读：不要直接复制整个工作目录

当前工作目录包含约45 GiB历史实验输出、多个重复模型、LoRA checkpoint、日志和临时环境。推荐拆成三类迁移：

1. **Git代码与小型报告**：从GitHub clone；
2. **公开模型与公开数据集**：按固定Hub revision重新下载；
3. **Git忽略的冻结实验产物**：从旧机或对象存储复制，不能靠Git恢复。

不要复制 `.venv-*` 到新机器。虚拟环境包含绝对路径、平台相关二进制和CUDA wheel，必须用uv重建。

---

## 1. 两档迁移范围

### 1.1 Bridge-minimal：继续当前下一步，推荐

用途：

- 重现Pass@8 locked结果；
- 运行P100/P80/P60/P40纯度响应曲线；
- 在train split生成候选并运行冻结9B/consensus；
- 构建selector-to-training bridge。

需要：

- 两个uv环境；
- 4个核心checkpoint；
- GUI-360 balanced parquet与导出的train/test episode+PNG；
- Pass@8候选缓存、blind/sealed split和selector输出；
- A1 GT replay、A5 revision negatives、starting-student eval、A13 oracle rescue、A15 25/75 anchor。

模型权重约67.6 GB（约63 GiB），数据约4 GB，两个环境建议预留40–60 GB。推荐至少150 GB空闲空间；若Hub cache与local directory并存，预留250 GB。

### 1.2 Full-lineage：重现完整研究链

在Bridge-minimal基础上增加：

- InternVL3 actor；
- Qwen3.5-35B-A3B scale control；
- Qwen2.5-VL base；
- actor trajectories与global corrections；
- 可选的历史LoRA/fullparam模型。

七个核心模型总权重约172 GB。加环境、数据、cache和训练输出后，推荐至少350 GB空闲空间，500 GB更安全。

### 1.3 不属于当前必需项

以下本地目录不是继续purity bridge的必要条件：

- `Molmo-7B-D-0924`
- `Pixtral-12B-2409`
- `UI-S1-7B`（仅原UI-S1/AndroidControl复现需要）
- `Llama-3.2-11B-Vision-Instruct`：当前旧机目录只有约76 KiB元数据，没有权重，**不要迁移为有效checkpoint**。

---

## 2. 源机器的已验证基线

| 项目 | 当前已验证值 |
|---|---|
| OS | Linux x86_64, kernel 6.8 |
| GPU | NVIDIA A100-SXM4-80GB |
| Driver | 580.126.16 |
| uv | 0.11.28 |
| Python | 3.11.15 |
| Git remote | `https://github.com/stevenqing/UI-S1.git` |
| 最低代码版本 | commit `5876e7f` 或更新的 `main` |

Qwen3环境使用CUDA 12.8 PyTorch wheel；Qwen3.5环境使用CUDA 13.0 wheel。目标机器的NVIDIA driver必须兼容相应CUDA runtime。若目标driver较旧，应优先升级driver，不要随意降级Torch/vLLM版本。

---

## 3. Clone代码

Git仓库是monorepo，研究代码位于clone目录下的 `UI-S1/` 子目录：

```bash
git clone https://github.com/stevenqing/UI-S1.git ui-s1-monorepo
cd ui-s1-monorepo/UI-S1
git checkout main
git pull --ff-only
git rev-parse HEAD
```

若需要固定本次迁移前的最低状态：

```bash
git checkout 5876e7f
```

推荐使用包含本指南和验证器的更新版 `main`。

---

## 4. 安装uv与Python 3.11.15

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
uv python install 3.11.15
```

目标版本：

```text
uv 0.11.28
Python 3.11.15
```

uv小版本更新通常可接受，但环境内包版本必须与requirements一致。

---

## 5. 重建两个独立环境

### 5.1 Qwen3/训练环境

用途：

- Qwen3-VL、InternVL、LLaVA推理；
- GUI-360 parquet导出；
- LLaMA-Factory/DeepSpeed/LoRA训练；
- reward matcher与大部分分析脚本。

```bash
uv venv --python 3.11.15 .venv-qwen3-vllm
uv pip install \
  --python .venv-qwen3-vllm/bin/python \
  -r requirements-qwen3-vllm.txt \
  -r requirements-qwen3-training-extra.txt
```

两个requirements合并后共191个固定版本，已与当前实机 `uv pip freeze` 完全一致。关键版本：

| Package | Version |
|---|---:|
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0 |
| transformers | 4.57.1 |
| vLLM | 0.11.0 |
| DeepSpeed | 0.16.9 |
| LLaMA-Factory | 0.9.5 |
| pyarrow | 24.0.0 |
| datasets | 4.0.0 |
| pandas | 2.3.3 |
| PEFT | 0.18.1 |
| TRL | 0.24.0 |
| qwen-vl-utils | 0.0.14 |

不要只安装 `requirements-qwen3-vllm.txt`：它能serve模型，但缺少后装的LLaMA-Factory、PyArrow、datasets、PEFT和TRL，不能完整训练或导出GUI-360。

### 5.2 Qwen3.5 serving环境

用途：

- Qwen3.5-9B fixed-choice selector；
- Qwen3.5-35B-A3B TP=4 serving；
- Qwen3.5多模态OpenAI-compatible endpoint。

```bash
uv venv --python 3.11.15 .venv-qwen35-vllm
uv pip install \
  --python .venv-qwen35-vllm/bin/python \
  -r requirements-qwen35-vllm.txt
```

关键版本：

| Package | Version |
|---|---:|
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0 |
| transformers | 5.12.1 |
| vLLM | 0.23.0 |
| huggingface-hub | 1.19.0 |
| flashinfer-python | 0.6.12 |

这个环境有意不安装DeepSpeed、LLaMA-Factory、PyArrow和qwen-vl-utils。不要用它执行GUI-360 parquet导出或正式训练。

### 5.3 作为另一个代码仓库的依赖

若新代码仍直接import本仓库模块，推荐保留原仓库为 `third_party/UI-S1`，并设置：

```bash
export UIS1_ROOT=/absolute/path/to/ui-s1-monorepo/UI-S1
export PYTHONPATH="$UIS1_ROOT:$PYTHONPATH"
```

也可以在Qwen3环境中安装本地包，但必须禁用依赖解析，避免 `setup.py` 将Transformers降到4.51.1：

```bash
uv pip install \
  --python .venv-qwen3-vllm/bin/python \
  --no-deps -e .
```

若始终从UI-S1项目根目录执行脚本，则无需editable install。

### 5.4 mRoPE兼容

当前代码在 `v15_gui_360/train_trajectory_gspo.py` 中通过 `_patch_legacy_mrope_config()` 对加载后的legacy Qwen2.5-VL config做进程内归一化。迁移时：

- 必须保留当前代码版本；
- 不需要手改site-packages；
- 不要用旧installation guide中的全局Transformers mRoPE patch覆盖当前方案。

---

## 6. 下载checkpoint：固定revision

所有下列Hub模型在当前检查时均为public、非gated。

### 6.1 Bridge-minimal核心模型

| 本地目录 | Architecture | Hub repo | Revision | 权重字节 |
|---|---|---|---|---:|
| `gui360-fullparam-sft-step250` | `Qwen2_5_VLForConditionalGeneration` | `Stevenshuqing/gui360-fullparam-sft-step250` | `89a3556d0e3b38702deae86d1fa090b3eb4748d1` | 16,584,414,544 |
| `Qwen3-VL-8B-Instruct` | `Qwen3VLForConditionalGeneration` | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 17,534,339,512 |
| `Qwen3.5-9B` | `Qwen3_5ForConditionalGeneration` | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 19,306,310,880 |
| `llava-1.5-7b-hf` | `LlavaForConditionalGeneration` | `llava-hf/llava-1.5-7b-hf` | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` | 14,126,946,048 |

### 6.2 Full-lineage附加模型

| 本地目录 | Architecture | Hub repo | Revision | 权重字节 |
|---|---|---|---|---:|
| `InternVL3-8B` | `InternVLChatModel` | `OpenGVLab/InternVL3-8B` | `853e3a797a661694b1b8ece0cb72dc2b23e3dac9` | 15,888,831,920 |
| `Qwen3.5-35B-A3B` | `Qwen3_5MoeForConditionalGeneration` | `Qwen/Qwen3.5-35B-A3B` | `59d61f3ce65a6d9863b86d2e96597125219dc754` | 71,903,878,016 |
| `Qwen2.5-VL-7B-Instruct` | `Qwen2_5_VLForConditionalGeneration` | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | 16,584,414,560 |

### 6.3 一次性下载脚本

先设置可选token以提高rate limit；模型本身不要求token：

```bash
# 只有确实有token时才取消下一行注释：
# export HF_TOKEN=hf_xxx
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p checkpoints
```

Bridge-minimal下载：

```bash
.venv-qwen35-vllm/bin/python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    ("Stevenshuqing/gui360-fullparam-sft-step250", "89a3556d0e3b38702deae86d1fa090b3eb4748d1", "gui360-fullparam-sft-step250"),
    ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "Qwen3-VL-8B-Instruct"),
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "Qwen3.5-9B"),
    ("llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9", "llava-1.5-7b-hf"),
]
for repo, revision, local_name in models:
    print("downloading", repo, revision)
    snapshot_download(repo_id=repo, revision=revision, local_dir=f"checkpoints/{local_name}", max_workers=8)
PY
```

Full-lineage附加下载：

```bash
.venv-qwen35-vllm/bin/python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    ("OpenGVLab/InternVL3-8B", "853e3a797a661694b1b8ece0cb72dc2b23e3dac9", "InternVL3-8B"),
    ("Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754", "Qwen3.5-35B-A3B"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5", "Qwen2.5-VL-7B-Instruct"),
]
for repo, revision, local_name in models:
    print("downloading", repo, revision)
    snapshot_download(repo_id=repo, revision=revision, local_dir=f"checkpoints/{local_name}", max_workers=8)
PY
```

若目标机器不能访问Hub，可从旧机复制相同目录：

```bash
rsync -aH --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/checkpoints/ \
  ./checkpoints/
```

复制前先删除无效的 `Llama-3.2-11B-Vision-Instruct` 占位目录，或明确忽略它。

---

## 7. 下载与导出GUI-360数据

### 7.1 下载balanced parquet

| Dataset | Revision | 旧机大小 |
|---|---|---:|
| `Stevenshuqing/gui360-balanced` | `e682c0bd79ca5e1520bd22cd6d15b6ae2ff913e1` | 约1.9 GiB |

```bash
mkdir -p datasets/gui360-balanced
.venv-qwen35-vllm/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Stevenshuqing/gui360-balanced",
    repo_type="dataset",
    revision="e682c0bd79ca5e1520bd22cd6d15b6ae2ff913e1",
    local_dir="datasets/gui360-balanced",
    max_workers=8,
)
PY
```

预期parquet：3个train shards、2个test shards。

### 7.2 推荐：复制冻结导出

当前研究所有JSONL内的截图路径均指向：

```text
outputs/validation_2k/data/images/{train|test}/{episode_id}/step_NNN.png
```

最稳妥方法是复制完整冻结导出，保留相对路径与PNG内容：

```bash
mkdir -p outputs/validation_2k
rsync -aH --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/outputs/validation_2k/data/ \
  outputs/validation_2k/data/
```

预期：

| Split | Manifest文件 | Episodes | PNG steps | Episode manifest SHA256 |
|---|---|---:|---:|---|
| train | `outputs/validation_2k/data/train_episodes.jsonl` | 1,573 | 12,574 | `7af451fb32cd3df60c19a3f281c4b59cb574300519d0bb7d5e961d0bf9d6958e` |
| test | `outputs/validation_2k/data/test_episodes.jsonl` | 1,000 | 7,498 | `0f6fb7154e259eff9edd5e0cd59c7780293f71750b9a3b501af72c8860c258b5` |

冻结导出总大小约1.8 GiB。

### 7.3 备选：从parquet重新导出

仅当无法从旧机复制时使用：

```bash
.venv-qwen3-vllm/bin/python scripts/minimal_validation.py export \
  --data-dir datasets/gui360-balanced/data \
  --split train \
  --output-dir outputs/validation_2k/data

.venv-qwen3-vllm/bin/python scripts/minimal_validation.py export \
  --data-dir datasets/gui360-balanced/data \
  --split test \
  --output-dir outputs/validation_2k/data
```

导出后必须用第13节验证hash与PNG数量。若hash不同，不要继续使用旧的frozen candidate packet，因为它的screenshot namespace绑定旧导出。

---

## 8. 复制Git无法恢复的Pass@8冻结产物

这些文件被 `.gitignore` 排除，clone后只有小型报告，没有raw packets和selector outputs。为精确续跑，必须从旧机复制。

### 8.0 最快路径：使用已验证资产包

源工作区已经把第8节、第9节和正确train manifest打成一个确定性压缩包：

```text
outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.tar.gz
```

- 包内62个文件；
- 原始130,025,816 bytes；
- 压缩后9,201,720 bytes；
- SHA256：`04f97f020d264f65121d97c42088af6181e83f670a337f2930c234559f41ab8f`。

将该文件复制到目标代码根目录后执行：

```bash
.venv-qwen35-vllm/bin/python scripts/restore_migration_asset_bundle.py \
  --bundle outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.tar.gz \
  --root .
```

恢复器会保留hash正确的文件、补齐缺失文件，并拒绝静默覆盖冲突。详细分类、逐文件hash与覆盖规则见 [缺失资产清单](../outputs/migration_verification/missing_assets_zh.md)。资产包本身不提交Git，必须从原机器或对象存储复制。

### 8.1 必需文件

```text
outputs/rl_feasibility/per_step.jsonl
outputs/multiagent_complementarity/target_ids.json
outputs/multiagent_complementarity/qwen3_vl_candidates.jsonl
outputs/multiagent_complementarity/qwen35_candidates.jsonl
outputs/multiagent_complementarity/extra_tiers/llava15_candidates.jsonl
outputs/pass8_selector_study/
```

复制后的 `outputs/pass8_selector_study/` 至少必须保留：

```text
frozen_v1/blind/{dev,locked_test}.jsonl
frozen_v1/sealed_labels/{dev,locked_test}.jsonl
runtime/selectors/current/locked_test.jsonl
runtime/selectors/strong/locked_test.jsonl
runtime/selectors/cross_source_consensus/locked_test.jsonl
eval/locked_test/current_per_step.jsonl
eval/locked_test/cross_source_consensus_per_step.jsonl
```

推荐复制命令：

```bash
mkdir -p \
  outputs/rl_feasibility \
  outputs/multiagent_complementarity/extra_tiers

rsync -aH --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/outputs/rl_feasibility/per_step.jsonl \
  outputs/rl_feasibility/

rsync -aH --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/outputs/multiagent_complementarity/target_ids.json \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/outputs/multiagent_complementarity/qwen3_vl_candidates.jsonl \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/outputs/multiagent_complementarity/qwen35_candidates.jsonl \
  outputs/multiagent_complementarity/

rsync -aH --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/outputs/multiagent_complementarity/extra_tiers/llava15_candidates.jsonl \
  outputs/multiagent_complementarity/extra_tiers/

rsync -aH --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/outputs/pass8_selector_study/ \
  outputs/pass8_selector_study/
```

当前旧机规模：

- `per_step.jsonl`：约27 MiB；
- `multiagent_complementarity`完整目录约232 MiB，但上述最小文件只有约26 MiB；
- `pass8_selector_study`：约14 MiB。

### 8.2 核心SHA256

| File | SHA256 |
|---|---|
| `outputs/rl_feasibility/per_step.jsonl` | `71ab0df74d5f25a8aba5a77cba15959ca3ae390b6dcc979d4178dc4d67e22cfc` |
| `target_ids.json` | `509327c7de49565e423afd9d8078631fe1df02a5c2af1440c2682e59870f71ad` |
| `qwen3_vl_candidates.jsonl` | `c3e4e0f6984ac5a106b5dc585ce584e1046e033a08ed4d37086df5fd9e63a6ca` |
| `qwen35_candidates.jsonl` | `aaf05b9cc36c081b69f48abcc4b6b7c26a1648fa2ca12fe7f84a40070d4fe08f` |
| `llava15_candidates.jsonl` | `fa168f868559467eca220d49b11eebe9559bac6e64a76e4a39b3299d0a7f005e` |
| `frozen_v1/manifest.json` | `a993d8d18b3ad997622dbd0503257aa624cb7715320729d73a76f888a28df89c` |

不要只复制 `eval/summary.json`：后续self-source、purity和train-bridge分析需要blind packets、sealed provenance、raw selector IDs与per-step结果。

---

## 9. 复制purity bridge与oracle replay核心数据

不要复制整个45 GiB `outputs/multiagent_trajectory_revision`。继续下一阶段只需以下文件：

| File | Bytes | SHA256 | 用途 |
|---|---:|---|---|
| `causal_arms/a1_gt_target_gt_history.jsonl` | 26,946,055 | `03865e32e0bac2b59177d7cf656bacba3837b5666655ec8078e1401fac2db046` | GT clean replay pool |
| `causal_arms/a5_revision_target_gt_history.jsonl` | 27,726,919 | `f10bc370acd8997d830cd944497780d48f939d72e699a6f1b412c5a1a7fd8df3` | revision/hard-negative pool |
| `causal_eval/a5_gt_history_grid/merged.jsonl` | 1,828,270 | `9e286d50434f68bb7c4a4520f898608a8ee62f9ea5d03a31f7b858f9f48421fd` | starting-student correctness |
| `utility_gate/a13_oracle_student_rescue_gt_history.jsonl` | 1,668,375 | `fb0e892e6df3d7edc894741dffac7ca3c85007ca9d314046d1b9bfa405352001` | 800-row pure rescue bank |
| `utility_gate/a15_student_rescue25_replay75.jsonl` | 1,698,151 | `7f47ec0eb2a9ae37d01a8af324242cf60aa391a367be1debdadfc427c220242b` | 已验证25/75 anchor |

复制命令：

```bash
ROOT=outputs/multiagent_trajectory_revision/full_v1
mkdir -p \
  "$ROOT/causal_arms" \
  "$ROOT/causal_eval/a5_gt_history_grid" \
  "$ROOT/utility_gate"

rsync -aH --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/$ROOT/causal_arms/a1_gt_target_gt_history.jsonl \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/$ROOT/causal_arms/a5_revision_target_gt_history.jsonl \
  "$ROOT/causal_arms/"

rsync -aH --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/$ROOT/causal_eval/a5_gt_history_grid/merged.jsonl \
  "$ROOT/causal_eval/a5_gt_history_grid/"

rsync -aH --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/$ROOT/utility_gate/a13_oracle_student_rescue_gt_history.jsonl \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/$ROOT/utility_gate/a15_student_rescue25_replay75.jsonl \
  "$ROOT/utility_gate/"
```

这些JSONL引用第7节的train PNG；必须保持相同相对目录。

### 9.1 可选：完整研究档案

若要重做CRU/source/prefix分析，再复制：

| File | Size | SHA256 |
|---|---:|---|
| `full_v1/actor_trajectories.jsonl` | 32,030,875 bytes | `c0db1735e1b708bc0ef0ea5ede9946e3ce34ff8e7668de4dd278ce712be7fa8a` |
| `full_v1/global_corrections_recovered.jsonl` | 15,713,108 bytes | `9a96c222ebd9cbb8d523e2b44130ba0f67729d32754821f72a8446436d648191` |

历史模型目录不建议默认复制：

- `full_v1/fullparam_model`：约16 GiB；
- `full_v1/lora_screen`：约14 GiB；
- `full_v1/utility_gate`完整目录：约7.1 GiB。

只有需要复查旧checkpoint推理时才单独迁移。

---

## 10. 原UI-S1 / AndroidControl是可选分支

原UI-S1数据集：

| Dataset | Revision |
|---|---|
| `mPLUG/UI_S1_dataset` | `580738153ac7621f369f6bb948807587dd74b474` |

原模型：

| Model | Revision |
|---|---|
| `mPLUG/UI-S1-7B` | `78efe009d34080c92679d3b0663bf3cbd86dd2e4` |

当前仓库中的 `datasets/android_control_evaluation_std.jsonl` 使用绝对截图路径 `/datasets/AndroidControl/images/...`，但当前旧机已不存在该图片目录。因此：

- 继续当前GUI-360 purity bridge不需要AndroidControl；
- 若要重跑AndroidControl Pass@8，必须从原数据源重新下载图片；
- 推荐将JSON中的绝对路径改写为新资产根目录，或在目标机挂载兼容的 `/datasets/AndroidControl/images`；
- 不能仅迁移两个JSONL就声称AndroidControl数据完整。

---

## 11. 迁移到另一个代码仓库的路径策略

### 11.1 推荐：代码与资产分离

```text
/data/ui-s1-assets/
  checkpoints/
  datasets/
  outputs/

/work/new-agent-code/
  checkpoints -> /data/ui-s1-assets/checkpoints
  datasets    -> /data/ui-s1-assets/datasets
  outputs     -> /data/ui-s1-assets/outputs
```

在新代码根目录创建链接：

```bash
ASSET_ROOT=/data/ui-s1-assets
ln -sfn "$ASSET_ROOT/checkpoints" checkpoints
ln -sfn "$ASSET_ROOT/datasets" datasets
ln -sfn "$ASSET_ROOT/outputs" outputs
```

同时保留原UI-S1代码用于reward/parser/data builder：

```bash
export UIS1_ROOT=/absolute/path/to/ui-s1-monorepo/UI-S1
export PYTHONPATH="$UIS1_ROOT:$PYTHONPATH"
```

### 11.2 相对路径约束

GUI-360和Pass@8 JSONL使用仓库根相对路径。最安全做法：

- 新代码根目录提供同名 `outputs/`、`datasets/`、`checkpoints/` 链接；
- 从新代码根目录运行；
- 不移动 `outputs/validation_2k/data/images` 内部层级。

若必须改变路径，应写一次性JSONL path-rewriter，并同时更新manifest/hash；不要静默改路径后继续声称使用原frozen split。

---

## 12. GPU serving验收

先确认目标GPU没有其他作业：

```bash
nvidia-smi
```

Qwen3.5-9B需要1张A100-80GB。Qwen3.5-35B-A3B推荐4张A100-80GB、TP=4。当前验证参数：

```text
max_model_len=16384
gpu_memory_utilization=0.65
kv_cache_memory_bytes=8G
enforce_eager=true
limit_mm_per_prompt={"image":1}
```

不要照搬源机器GPU编号。显式设置目标机允许的 `CUDA_VISIBLE_DEVICES`，并先检查共享GPU上的外部进程。

当前源机器另有外部训练PID 2190159占用GPU 2–7；迁移打包过程中不得停止或干扰它。

---

## 13. 一键CPU验证

迁移完成后，在新UI-S1项目根目录执行Bridge-minimal验证：

```bash
.venv-qwen35-vllm/bin/python scripts/verify_migration_setup.py \
  --profile bridge \
  --output outputs/migration_verification/bridge.json
```

Full-lineage验证：

```bash
.venv-qwen35-vllm/bin/python scripts/verify_migration_setup.py \
  --profile full \
  --output outputs/migration_verification/full.json
```

验证器不会初始化CUDA或启动模型。它检查：

- 两个Python环境与关键package版本；
- 模型architecture、weight bytes、Hub revision和缺失分片；
- 13个核心数据文件SHA256；
- Pass@8 blind/sealed/raw-selector/per-step路径；
- train 12,574张与test 7,498张PNG。

当前源机器上两个profile均已通过：

```text
bridge: ok=true, failures=[]
full:   ok=true, failures=[]
```

若验证失败，不要启动训练。先修复所有 `failures`，尤其是episode hash、截图数量、模型revision和missing index shards。

---

## 14. 最终验收清单

### 代码

- [ ] clone后进入monorepo的 `UI-S1/` 子目录；
- [ ] `main` 至少包含commit `5876e7f`；
- [ ] `PYTHONPATH` 能找到原UI-S1模块；
- [ ] 未使用旧版 `requirements.txt` 覆盖两个专用环境。

### 环境

- [ ] `.venv-qwen3-vllm` Python 3.11.15；
- [ ] base + training-extra共191个固定package；
- [ ] `.venv-qwen35-vllm` 共190个固定package；
- [ ] Qwen3环境有LLaMA-Factory 0.9.5和PyArrow 24.0.0；
- [ ] Qwen3.5环境有vLLM 0.23.0和Transformers 5.12.1。

### Checkpoint

- [ ] Bridge-minimal四个模型architecture正确；
- [ ] 所有 `model.safetensors.index.json` 引用分片都存在；
- [ ] 本地Hub revision与表格一致；
- [ ] 若复现scale/full lineage，附加三个模型也完整；
- [ ] 未把76 KiB Llama占位目录当作模型。

### 数据

- [ ] GUI-360 balanced revision为 `e682c0...`；
- [ ] train/test episode分别1,573/1,000；
- [ ] train/test PNG分别12,574/7,498；
- [ ] episode manifests SHA256一致；
- [ ] Pass@8 raw候选、blind、sealed、selector output和per-step eval已复制；
- [ ] A1/A5/student-eval/A13/A15五个bridge文件hash一致；
- [ ] 没有把dev/locked rows用于训练。

### 运行前

- [ ] `verify_migration_setup.py --profile bridge` 返回 `ok=true`；
- [ ] GPU进程与显存已重新审计；
- [ ] 目标GPU allowlist明确；
- [ ] 先做单请求smoke，再做数据生成或训练；
- [ ] purity-response与train-purity bridge通过前，不启动正式25/75训练。

---

## 15. 最短迁移路径

若只想尽快在新代码上继续当前下一步，按以下顺序：

1. clone最新Git代码；
2. 用两个requirements文件组重建两个uv环境；
3. 从Hub下载Bridge-minimal四个模型；
4. 下载GUI-360 balanced并复制冻结的 `outputs/validation_2k/data`；
5. 复制第8节全部Pass@8冻结产物；
6. 复制第9节五个bridge核心JSONL；
7. 在新代码中链接 `checkpoints/`、`datasets/`、`outputs/` 并设置 `PYTHONPATH`；
8. 运行bridge verifier，必须 `ok=true`；
9. 冻结purity-curve与train-candidate manifests；
10. GPU审计通过后，再运行P100/P80/P60/P40和train-split诊断。

这条路径不需要迁移45 GiB历史outputs，也不需要重新生成已经冻结的locked Pass@8结果。
