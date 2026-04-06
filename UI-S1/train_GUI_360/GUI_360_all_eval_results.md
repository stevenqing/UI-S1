# GUI-360 全模型评测汇总

## 1. 核心模型对比 (Step-Level)

| Model | Grounding | Action(Visual) | Action(A11y) | 备注 |
|-------|:---------:|:--------------:|:------------:|------|
| **GUI-360 Paper SFT** | **82.30** | **50.08** | **25.78** | 原论文 |
| Qwen2.5-VL-7B Base | 42.47 | 18.05 | 14.53 | Zero-shot |
| Qwen3-VL-8B Base | 25.76 | 10.08 | 22.01 | Zero-shot (fixed eval) |
| Qwen3.5-9B | 0.00 | 0.00 | — | 格式不兼容,全部失败 |
| SFT v1 (full, 低分辨率) | 12.87 | 11.14 | 13.40 | 灾难性遗忘 |
| **SFT v2 (full, 1ep)** | **70.56** | **46.90** | 17.51 | 综合最优 |
| SFT v2 (full, 2ep) | 70.77 | 49.37 | 未评测 | Visual略优 |
| MoE v1 LoRA (6expert, r=16) | 60.32 | 33.20 | 5.47 | Router collapsed |
| LoRA v3 (r=32, frozen proj) | 56.34 | 24.67 | **20.54** | A11y最佳, RL init |

## 2. Grounding SFT v3 (Full-param, 多任务含Grounding标注)

| Checkpoint | Grounding | Action(Visual) | Action(A11y) |
|------------|:---------:|:--------------:|:------------:|
| ckpt150 | 77.66 | 3.89 | 未评测 |
| ckpt200 | 78.55 | 3.63 | 10.93 |
| ckpt250 | 79.39 | 3.08 | 11.08 |
| ckpt300 | 79.61 | 3.06 | 11.02 |
| **Final** | **79.48** | 3.07 | 10.88 |

> Grounding接近Paper(79.48 vs 82.30), 但Action严重过拟合(3%~11%)

## 3. LoRA v3 Continued Training (接续LoRA v3继续训练)

| Checkpoint | Grounding | Action(Visual) | Action(A11y) |
|------------|:---------:|:--------------:|:------------:|
| LoRA v3 原始 | 56.34 | 24.67 | 20.54 |
| cont ckpt200 | 58.65 | 30.84 | 19.55 |
| cont ckpt400 | 60.52 | 34.67 | 19.35 |
| cont ckpt600 | 60.61 | 35.76 | 20.56 |
| **cont ckpt796** | **61.01** | **35.87** | 20.48 |

> 继续训练提升Grounding +4.67, Visual +11.20, A11y基本不变

## 4. LoRA v4 (新一轮LoRA训练)

| Checkpoint | Grounding | Action(Visual) | Action(A11y) |
|------------|:---------:|:--------------:|:------------:|
| ckpt200 | 62.83 | 25.06 | 20.58 |
| ckpt250 | 63.93 | 27.05 | 20.43 |
| ckpt300 | 64.25 | 27.28 | 20.43 |
| **ckpt354** | **64.37** | **27.53** | 20.31 |

> Grounding优于LoRA v3 cont, 但Visual反而更低

## 5. SVD LoRA (从Full SFT v2分解出LoRA)

| Rank | Grounding | Action(Visual) | Action(A11y) |
|:----:|:---------:|:--------------:|:------------:|
| r=32 | 61.60 | 37.35 | 未评测 |
| r=64 | 62.81 | 42.08 | 未评测 |
| r=128 | 65.85 | 44.75 | 未评测 |
| **r=256** | **68.12** | **47.00** | 未评测 |

> 随rank增大稳步提升, r=256接近SFT v2 (68.12 vs 70.56 grounding, 47.00 vs 46.90 visual)

## 6. MoE v3 SFT (仅Grounding评测)

| Checkpoint | Grounding |
|------------|:---------:|
| ckpt200 | 51.17 |
| ckpt400 | 55.32 |
| ckpt600 | 55.58 |
| ckpt800 | 56.27 |
| ckpt1000 | 56.17 |
| **Final** | **56.24** |

> 收敛于56%, 低于MoE v1(60.32%), Action未评测

---

## 7. Trajectory-Level 评测 (Semi-Online AR, stop_on_error)

### 7.1 Overall

| 指标 | Base | SFT v2 |
|------|:----:|:------:|
| Total Trajectories | 3233 | 3233 |
| Avg Steps/Traj | 1.3 | 1.9 |
| Step-Level SR | 22.10% | 55.28% |
| **Trajectory SR (TSR)** | **1.64%** | **16.21%** |
| Avg Progress Rate | 12.30% | 36.70% |
| Func Match Rate | 62.54% | 77.45% |
| Args Match Rate | 1.86% | 17.07% |

### 7.2 By Trajectory Length

| Length | Base TSR | SFT v2 TSR | Base Progress | SFT v2 Progress |
|--------|:--------:|:----------:|:-------------:|:----------------:|
| Short (1-5) | 1.64% | 15.89% | 12.30% | 35.68% |
| Medium (6-15) | 0.00% | 32.79% | 0.00% | 89.64% |
| Long (16+) | 0.00% | 0.00% | 0.00% | 0.00% |

### 7.3 By Domain (SFT v2)

| Domain | # Traj | TSR | Progress |
|--------|:------:|:---:|:--------:|
| PPT | 865 | 18.27% | 46.44% |
| Word | 1369 | 15.49% | 35.37% |
| Excel | 999 | 15.42% | 30.09% |

### 7.4 Non-Stop 评测 (评估所有步骤, 不中断)

| 指标 | Base | SFT v2 |
|------|:----:|:------:|
| Step-Level SR | 18.05% | 46.90% |
| **TSR** | **2.85%** | **16.95%** |
| Avg Progress | 20.32% | 50.60% |

---

## 8. 未评测项汇总

| Model | 缺失 |
|-------|------|
| SFT v2 2ep | Action(A11y) |
| SVD LoRA r=32/64/128/256 | Action(A11y) |
| MoE v3 所有checkpoint | Action(Visual), Action(A11y) |
| Grounding SFT v3 ckpt150 | Action(A11y) |
| Qwen3.5-9B | Action(A11y), 且格式不兼容导致全0 |
