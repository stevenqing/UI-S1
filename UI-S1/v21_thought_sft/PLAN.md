# V21: Thought-Augmented SFT

## Motivation

V19 ablation proved that the fundamental bottleneck in long-horizon GUI navigation is **planning**, not execution:

| Experiment | Oracle Thought | TSR | Delta |
|---|---|---|---|
| Standard SFT-272 | No | 21.9% | baseline |
| Subtask (oracle) | Yes (GT thought) | 35.2% | +13.3pp |

The oracle thought provides per-step planning instructions (what to do next and why). 97.7% of the improvement comes from this oracle information, not from other factors like history format.

**Root cause**: SFT-272 was trained with `response = <tool_call>action</tool_call>` — the GT thought was available in the training data but was discarded. The model never learned to plan; it only learned to execute.

**Fix**: Re-train with `response = thought + <tool_call>action</tool_call>` so the model learns to plan before acting.

## Approach

### Single-model thought-augmented SFT

The simplest approach from first principles: include GT thought in the training response.

**Current training format** (SFT-272):
```
Human: [screenshot] + [goal + history + actions prompt]
GPT: <tool_call>{"function": "click", "args": {"coordinate": [139, 71]}}</tool_call>
```

**New format** (V21):
```
Human: [screenshot] + [goal + history + actions prompt]
GPT: To add a rectangle shape, I need to go to the 'Insert' tab, where the
     'Shapes' option is located. The first step is to click the 'Insert' tab.

     <tool_call>{"function": "click", "args": {"coordinate": [139, 71]}}</tool_call>
```

The model learns to:
1. First reason about current state + what to do next (planning)
2. Then output the action (execution)

At test time, the model generates its own thought before acting. The self-generated thought helps guide action prediction through attention.

### Why this is better than hierarchical (separate Planner → Executor)

| Factor | Single Model | Hierarchical |
|---|---|---|
| Forward passes/step | 1 | 2 |
| GUI-360 domain knowledge | Full (same model) | Planner has none (fresh LoRA on base) |
| Inference cost | 1x | 2x |
| Implementation complexity | Minimal (just change training data) | High (two vLLM instances, pipeline) |
| Thought-action alignment | Naturally aligned (same model) | May diverge (different models) |

The hierarchical approach may be useful later (Phase B/C) for adding Observer or RL, but for the initial baseline, single model is strictly better.

## Data

### Training data
- Source: `v15_gui_360/data/gui360_balanced_train.jsonl` (17,264 steps from 2,000 balanced episodes)
- GT thought: mapped from `datasets/GUI-360/rl_data/gui360_train.jsonl` via screenshot path matching
- Match rate: **17,264/17,264 (100%)**
- Average thought length: 236 chars
- Output: `v21_thought_sft/data/gui360_thought_train.jsonl`

### Validation data
- Source: `v15_gui_360/data/gui360_balanced_val.jsonl` (259 steps)
- Match rate: **259/259 (100%)**
- Output: `v21_thought_sft/data/gui360_thought_val.jsonl`

### Test data
- `v12_gui_360/data/gui360_test_1000_balanced_with_thought.jsonl` (1,000 episodes, 7,498 steps)
- Already created in V19 for subtask oracle eval

## Training Configuration

Matches SFT-272 as closely as possible — the only change is the training data (thought in response).

| Parameter | SFT-272 | V21 Thought SFT |
|---|---|---|
| Base model | Qwen2.5-VL-7B-Instruct | Qwen2.5-VL-7B-Instruct |
| Finetuning type | Full parameter | Full parameter |
| Dataset | gui360_balanced_train (17K) | gui360_thought_train (17K, +thought) |
| LR | 1e-5 | 1e-5 |
| Epochs | 4 | 4 |
| Batch size (effective) | 256 | 256 |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.05 | 0.05 |
| freeze_vision_tower | true | true |
| DeepSpeed | ZeRO-3 | ZeRO-3 |
| Nodes × GPUs | 4 × 4 | 4 × 4 |

### Expected differences from SFT-272
- Response is longer (~370 chars vs ~165 chars) → training loss starts higher
- More tokens per sample → slightly longer training time
- Model learns to generate text before `<tool_call>` → may need cutoff_len increase

## Evaluation Plan

### Eval 1: Standard prompt (pred history)
Run V13 evaluator with standard prompt. Compare directly with SFT-272 baseline (21.9%).
- The model should generate thought text before `<tool_call>` even with standard prompt, because it learned this pattern during training.

### Eval 2: Type-focused prompt (pred history)
Compare with best prompt result (type_focused 23.6%).

### Eval 3: Subtask prompt with model-generated thought
Instead of feeding GT thought, let the model generate its own thought, then feed that to the standard action prompt. This tests whether self-generated thoughts approach oracle quality.

### Eval 4: BoN-5 + LogProb
Compare with current best overall (26.6%).

### Key metrics
- TSR, StepSR, by task length (1, 2-3, 4-5, 6+)
- Whether model actually generates reasoning text (SFT-272 generates 0)
- Reasoning quality: does the generated thought match GT thought semantically?

## Target

| Outcome | TSR | Interpretation |
|---|---|---|
| Strong success | ≥ 28% | Recovered ≥ 50% of oracle headroom |
| Moderate success | 24-28% | Thought helps but model's self-generated thoughts are imperfect |
| No improvement | 22-24% | Model generates thoughts but they don't help action prediction |
| Regression | < 22% | Thought training hurts — investigate |

## File Structure

```
v21_thought_sft/
├── PLAN.md                           # This file
├── prepare_thought_sft_data.py       # Data preparation script
├── data/
│   ├── gui360_thought_train.jsonl    # Thought-augmented training data (17,264 steps)
│   └── gui360_thought_val.jsonl      # Thought-augmented val data (259 steps)
└── scripts/
    ├── train_thought_sft.slurm       # Training SLURM script
    ├── eval_thought_sft.slurm        # Evaluation SLURM script
    └── logs/
```

## Timeline

| Step | Time |
|---|---|
| Data preparation | Done |
| Register dataset + create training config | 30 min |
| Training (4 epochs, 4 nodes) | ~6-8 hours |
| Evaluation (4 experiments) | ~2-3 hours |
| Analysis | 1 hour |
| **Total** | **~1 day** |
