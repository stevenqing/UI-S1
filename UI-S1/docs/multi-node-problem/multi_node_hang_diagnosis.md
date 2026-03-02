# 多节点训练卡住问题诊断日志

## 问题概述

- **现象**: 2-node 训练时在 `_ALLGATHER_BASE` 操作处卡住
- **触发点**: `rollout_sharding_manager.__enter__()` 中的 `state_dict()` 调用
- **关键文件**:
  - `verl/workers/fsdp_workers.py:642` - generate_sequences 入口
  - `verl/workers/sharding_manager/fsdp_vllm.py:169` - state_dict() 调用
  - `verl/workers/sharding_manager/fsdp_sglang.py:103` - state_dict() 调用

---

## SLURM 验证脚本

按顺序提交以下作业进行诊断：

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

# Step 1: 基础诊断 - 测试网络和 FSDP
sbatch train/diagnose_multinode.slurm

# Step 2: 测试关闭 offload 是否解决问题
sbatch train/train_ui_s1_test_offload.slurm

# Step 3: 如果 Step 2 成功但需要 offload，启用详细日志定位问题
sbatch train/train_ui_s1_with_diag.slurm
```

---

## 验证步骤

### Step 1: 确认基本环境信息

```bash
# 执行以下命令并记录结果

# PyTorch 版本 (决定 FSDP1 vs FSDP2)
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# NCCL 版本
python -c "import torch; print(f'NCCL: {torch.cuda.nccl.version()}')"

# GPU 信息
nvidia-smi --query-gpu=name,memory.total --format=csv

# 网络接口
ip addr | grep -E "ib|eth|eno"
```

**记录结果**:
```
PyTorch:
NCCL:
GPU:
Network interfaces:
```

---

### Step 2: 确认当前配置

检查你的训练配置文件，记录以下关键参数：

```yaml
# 找到这些配置项并记录当前值

actor:
  fsdp_config:
    param_offload: ???      # true/false
    optimizer_offload: ???  # true/false

rollout:
  name: ???                 # vllm/sglang
  load_format: ???          # dummy_hf/safetensors/dtensor
  layered_summon: ???       # true/false (如果存在)

model:
  # 如果使用 LoRA
  lora:
    enable: ???
    rank: ???
```

**记录当前配置**:
```
param_offload:
optimizer_offload:
rollout.name:
rollout.load_format:
layered_summon:
lora.enable:
lora.rank:
```

---

### Step 3: 验证是否是 offload 导致的问题

#### 3.1 测试：关闭 offload

修改配置：
```yaml
actor:
  fsdp_config:
    param_offload: false
    optimizer_offload: false
```

运行训练，观察是否还卡住。

**测试结果**:
```
[ ] 仍然卡住 -> 继续 Step 4
[ ] 不再卡住 -> 确认是 offload 问题，跳到 Step 6 看解决方案
[ ] OOM -> 需要其他优化方案，跳到 Step 7
```

---

### Step 4: 启用 NCCL 详细日志

在启动脚本中添加：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 可选：设置 NCCL 超时（秒）
export NCCL_TIMEOUT=1800
```

运行训练，观察日志输出。

**关键日志分析**:

```
# 如果看到类似这样的日志，记录下来：

# 正常初始化完成的标志：
NCCL INFO Init done

# 卡住时可能看到的：
NCCL WARN Timeout waiting for ...
NCCL INFO NET/...

# 记录卡住时的最后几行日志：

```

**诊断结果**:
```
[ ] 卡在 INIT 阶段 -> 网络/IB 配置问题，继续 Step 5
[ ] INIT 完成但卡在 COLL -> 慢 rank 问题，跳到 Step 6
[ ] 其他错误 -> 记录错误信息
```

---

### Step 5: 排除 IB/RDMA 问题

#### 5.1 测试：禁用 IB

```bash
export NCCL_IB_DISABLE=1
```

运行训练。

**测试结果**:
```
[ ] 禁用 IB 后稳定 -> IB 配置问题，需要检查 IB 设置
[ ] 仍然卡住 -> 不是 IB 问题，继续排查
```

#### 5.2 如果是 IB 问题，检查：

```bash
# 检查 IB 状态
ibstat

# 检查 IB 网卡
ibv_devinfo

# 检查可达性
ibping -S  # 在一个节点
ibping -c <other_node_ip>  # 在另一个节点
```

**IB 检查结果**:
```

```

---

### Step 6: 代码级诊断

#### 6.1 添加详细时间戳日志

临时修改 `verl/workers/sharding_manager/fsdp_vllm.py`，在 `__enter__` 方法中添加：

```python
# 在 line 156 附近添加
import time
import torch.distributed as dist

rank = dist.get_rank() if dist.is_initialized() else 0
print(f"[DIAG][Rank {rank}] __enter__ started at {time.time()}")

# 在 line 160 (load_fsdp_model_to_gpu 之前)
print(f"[DIAG][Rank {rank}] Before load_fsdp_model_to_gpu at {time.time()}")

# 在 line 162 (load_fsdp_model_to_gpu 之后)
print(f"[DIAG][Rank {rank}] After load_fsdp_model_to_gpu at {time.time()}")

# 在 line 169 (state_dict 之前)
print(f"[DIAG][Rank {rank}] Before state_dict at {time.time()}")
dist.barrier()  # 添加显式同步
print(f"[DIAG][Rank {rank}] After barrier, calling state_dict at {time.time()}")

# 在 line 170 (state_dict 之后)
print(f"[DIAG][Rank {rank}] After state_dict at {time.time()}")
```

运行训练，收集各 rank 的时间戳。

**时间戳对比**:
```
# Rank 0:
__enter__ started:
Before load_fsdp_model_to_gpu:
After load_fsdp_model_to_gpu:
Before state_dict:
After barrier:
After state_dict:

# Rank 1 (另一个节点):
__enter__ started:
Before load_fsdp_model_to_gpu:
After load_fsdp_model_to_gpu:
Before state_dict:
After barrier:
After state_dict:

# 时间差分析：
load_fsdp_model_to_gpu 耗时差异:
barrier 等待时间:
state_dict 耗时差异:
```

---

### Step 7: 解决方案选择

根据上述诊断，选择合适的解决方案：

#### 方案 A: offload 是瓶颈 -> 优化 offload 逻辑

如果 `load_fsdp_model_to_gpu` 耗时差异大：

```python
# 修改 fsdp_vllm.py __enter__ 方法
# 在 load_fsdp_model_to_gpu 后添加同步

if self.offload_param:
    load_fsdp_model_to_gpu(self.module)
    # 添加同步，确保所有 rank 加载完成后再继续
    if dist.is_initialized():
        dist.barrier()
        get_torch_device().synchronize()  # 确保 GPU 操作完成
```

#### 方案 B: 使用 LoRA + layered_summon

如果你使用 LoRA，启用分层参数收集：

```yaml
rollout:
  load_format: safetensors  # 不能是 dummy_hf
  layered_summon: true
```

这会让代码只同步 LoRA 参数而不是整个模型。

#### 方案 C: FSDP2 优化

如果使用 PyTorch 2.4+ (FSDP2)，修改 state_dict 调用：

```python
# fsdp_vllm.py:169 替换为
if fsdp_version(self.module) == 2:
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    options = StateDictOptions(full_state_dict=False, cpu_offload=self.offload_param)
    params = get_model_state_dict(self.module, options=options)
else:
    params = self.module.state_dict()
```

#### 方案 D: 增加超时容忍度

临时方案，增加 NCCL 超时：

```bash
export NCCL_TIMEOUT=3600  # 1小时
export TORCH_NCCL_BLOCKING_WAIT=1
```

---

## 验证记录表

| 步骤 | 日期 | 结果 | 备注 |
|-----|------|------|------|
| Step 1: 环境信息 | | | |
| Step 2: 配置确认 | | | |
| Step 3: 关闭 offload | | | |
| Step 4: NCCL 日志 | | | |
| Step 5: 禁用 IB | | | |
| Step 6: 时间戳诊断 | | | |
| Step 7: 应用方案 | | | |

---

## 最终解决方案

**选用方案**:

**修改内容**:
```

```

**验证结果**:
```
[ ] 2-node 训练正常运行
[ ] 性能可接受
[ ] 无其他副作用
```

---

## 附录：快速诊断脚本

将以下脚本保存为 `diagnose_multi_node.sh`：

```bash
#!/bin/bash

echo "=== Multi-Node Hang Diagnosis ==="
echo "Date: $(date)"
echo ""

echo "=== Environment ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import torch; print(f'NCCL: {torch.cuda.nccl.version()}')" 2>/dev/null || echo "NCCL: N/A"
echo ""

echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "=== Network Interfaces ==="
ip addr | grep -E "^[0-9]+:|inet " | head -20
echo ""

echo "=== IB Status ==="
which ibstat > /dev/null 2>&1 && ibstat || echo "ibstat not available"
echo ""

echo "=== NCCL Environment Variables ==="
env | grep -i nccl
echo ""

echo "=== Current Process Group Status ==="
# 这部分需要在 Python 脚本中运行
python -c "
import torch.distributed as dist
if dist.is_initialized():
    print(f'World size: {dist.get_world_size()}')
    print(f'Rank: {dist.get_rank()}')
    print(f'Backend: {dist.get_backend()}')
else:
    print('Distributed not initialized')
" 2>/dev/null || echo "Cannot check distributed status"

echo ""
echo "=== Diagnosis Complete ==="
```

---

## 联系和参考

- verl 仓库: https://github.com/volcengine/verl
- PyTorch FSDP 文档: https://pytorch.org/docs/stable/fsdp.html
- NCCL 调试指南: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html
