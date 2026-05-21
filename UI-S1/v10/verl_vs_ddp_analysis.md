# verl (FSDP2 + vLLM) vs DDP (torchrun + HF generate) 性能分析

## 实验设置

两种实现都训练相同的 cooperative dual-LoRA GRPO：
- **模型**: Qwen2.5-VL-7B-Instruct + 2×LoRA (grounder + actor, r=128)
- **数据**: AndroidControl, 6482 train / 54 val samples
- **GRPO**: K=8 rollouts, clip=0.2, kl_coef=0.001
- **学习率**: grounder 1e-5, actor 5e-6
- **Effective batch**: 32 prompts per gradient update (256 sequences)
- **硬件**: NVIDIA GH200 (95GB), Slingshot interconnect (无 InfiniBand)

## 配置对比

| 参数 | Old DDP (4 nodes) | verl (8 nodes) |
|------|-------------------|----------------|
| Framework | torchrun + DDP + PEFT | Ray + FSDP2 + vLLM |
| GPUs | 16 (4×4) | 32 (8×4) |
| Batch/GPU | 1 prompt | 1 prompt |
| Grad accumulation | 2 | 1 (不需要) |
| Effective batch | 16×1×2 = 32 prompts | 32×1 = 32 prompts |
| 生成引擎 | HF `model.generate()` | vLLM (hybrid engine) |
| 并行策略 | DDP (全参数复制) | FSDP2 (参数分片) |
| GPU 显存 | ~22 GB/GPU | ~53 GB/GPU |

## 性能对比

### Per-step timing breakdown

| 阶段 | Old DDP (16 GPU) | verl (32 GPU) | 说明 |
|------|------------------|---------------|------|
| **Generation** | 2×38s = **76s** | 19+63 = **82s** | 基本相同 |
| **Log prob** | 包含在 update 里 | **44s** | verl 需要额外 forward pass |
| **Update** | **44s** | **62s** | FSDP2 > DDP 通信开销 |
| **其他** | 12s | ~0s | |
| **总计** | **132s** | **188s** | verl 慢 1.42x |

> Old DDP 的 `t=84s`（日志里的值）是 GRPO update 时间，不含 generation。
> 真实 wall-clock = generation + update = ~132s/step。

### 完整训练时间 (4 epochs, 202 steps/epoch)

| | Old DDP | verl 8-node | 差距 |
|--|---------|-------------|------|
| GPUs | 16 | 32 | 2x |
| s/step | 132s | 188s | 1.42x slower |
| 4 epochs | **29.6h** | **42.2h** | 1.42x slower |
| GPU-hours | **474** | **1,350** | **2.85x worse** |

## 根本原因分析

### 1. 额外的 log_prob forward pass (+44s/step)

这是最大的开销来源。

- **Old DDP**: 在 GRPO update 的 forward pass 中直接计算 current log_prob 和 ref log_prob（关闭 LoRA 就是 ref model），只需 1 次 forward。
- **verl**: rollout (vLLM mode) 和 training (FSDP mode) 完全分离。rollout 后必须单独做一次 actor forward + ref forward 来计算 log_prob。= 2 次额外 forward pass。

### 2. FSDP2 通信开销 (+18s/step)

- **DDP**: 每个 GPU 持有完整模型副本，backward 时只需 all-reduce 梯度（一次通信）。
- **FSDP2**: 参数分片存储，每次 forward/backward 都需要 all-gather（拉取参数）+ reduce-scatter（分片梯度）。通信量和次数都远大于 DDP。
- 32 GPUs 跨 8 节点 vs 16 GPUs 跨 4 节点，Slingshot Socket 网络延迟更高。

### 3. 生成效率无优势 (~相同)

vLLM 在低 batch (8 seqs/GPU) 下对比 HF generate 没有速度优势：
- Hybrid engine 需要 FSDP → vLLM 权重同步
- `gpu_memory_utilization=0.4` 限制了 KV cache
- vLLM 的 paged attention 在短序列 + 低并发时开销大于收益

### 4. 显存效率差

| | Old DDP | verl |
|--|---------|------|
| GPU 显存 | ~22 GB | ~53 GB |
| 利用率 | 23% of 95GB | 56% of 95GB |

7B 模型 + LoRA 在单卡上只需 ~22GB。FSDP2 虽然分片了参数，但 vLLM 的 KV cache、activation 等额外占用反而让总显存更高。

## 结论

**对于 7B 模型 + LoRA 训练，DDP 是最优方案。** 原因：

1. 模型完全放得进单卡（22GB << 95GB），FSDP 分片是纯开销
2. DDP 的 all-reduce 通信比 FSDP2 的 all-gather + reduce-scatter 简单高效
3. HF generate 在低并发场景下和 vLLM 一样快，但没有模式切换开销
4. verl 的 rollout/logprob/update 三步分离架构引入了不必要的额外 forward pass

**verl/FSDP2 的适用场景**：70B+ 大模型，单卡放不下，必须跨 GPU 分片时才有优势。

## 8-node DDP 扩展方案

将 Old DDP 从 4 nodes 扩展到 8 nodes (32 GPUs)：
- `grad_accumulation_steps: 2 → 1`（保持 effective batch = 32）
- 预期 s/step: ~70s（generation 减半，update 略增）
- 预期 4 epochs: ~15h（vs 原来 29.6h）
- GPU-hours: 32 × 15 = 480（vs 原来 474，基本相同）
